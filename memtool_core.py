#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memtool_core: SQLite 记忆管理核心逻辑（无 CLI I/O）
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
import uuid
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from memtool_lifecycle import DEFAULT_STALE_THRESHOLD, cleanup_candidates, lifecycle_meta
from memtool_rank import score_item
from memtool_recommend import recommend_items

DEFAULT_DB = os.environ.get("MEMTOOL_DB", "./memtool.db")
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY: set[str] = set()
_MIN_TOKEN_LEN = 2


class MemtoolError(Exception):
    def __init__(self, payload: Dict[str, Any], exit_code: int) -> None:
        self.payload = payload
        self.exit_code = exit_code
        super().__init__(payload.get("message") or payload.get("error") or "MemtoolError")


def utcnow_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def gen_id() -> str:
    # UUID4 足够用于 MVP；如果你更想要可排序 id，可换 ULID 实现
    return uuid.uuid4().hex


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # 更适合并发读写的 WAL
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")  # 5 秒超时
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_items (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK (type IN ('project','feature','run')),
  task_id TEXT,
  step_id TEXT,
  key TEXT NOT NULL,
  content TEXT NOT NULL,
  content_search TEXT NOT NULL DEFAULT '',
  tags_json TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
  source TEXT,
  weight REAL NOT NULL DEFAULT 1.0,
  version INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_type_key ON memory_items(type, key);
CREATE INDEX IF NOT EXISTS idx_memory_task_step ON memory_items(task_id, step_id);
CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_items(updated_at);

-- 可选：全文检索（FTS5）。如果宿主 SQLite 不支持，会在 init 时提示并自动降级为 LIKE。
"""


def _fts_status(conn: sqlite3.Connection) -> Tuple[bool, bool]:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id, content_search, tokenize='unicode61');")
        conn.execute("SELECT 1 FROM memory_fts LIMIT 1;").fetchone()
    except sqlite3.OperationalError:
        return False, False
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(memory_fts)").fetchall()]
    return True, "content_search" not in cols


def _create_fts_triggers(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP TRIGGER IF EXISTS memory_ai;
    DROP TRIGGER IF EXISTS memory_ad;
    DROP TRIGGER IF EXISTS memory_au;
    CREATE TRIGGER memory_ai AFTER INSERT ON memory_items BEGIN
      INSERT INTO memory_fts(id, content_search)
      VALUES (new.id, COALESCE(NULLIF(new.content_search, ''), new.content));
    END;
    CREATE TRIGGER memory_ad AFTER DELETE ON memory_items BEGIN
      DELETE FROM memory_fts WHERE id = old.id;
    END;
    CREATE TRIGGER memory_au AFTER UPDATE OF content, content_search ON memory_items BEGIN
      UPDATE memory_fts
      SET content_search = COALESCE(NULLIF(new.content_search, ''), new.content)
      WHERE id = new.id;
    END;
    """)


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP TRIGGER IF EXISTS memory_ai;
    DROP TRIGGER IF EXISTS memory_ad;
    DROP TRIGGER IF EXISTS memory_au;
    DROP TABLE IF EXISTS memory_fts;
    """)
    conn.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(id, content_search, tokenize='unicode61');")
    _create_fts_triggers(conn)
    conn.execute("""
    INSERT INTO memory_fts(id, content_search)
    SELECT id, COALESCE(NULLIF(content_search, ''), content) FROM memory_items
    """)


def _ensure_schema(conn: sqlite3.Connection) -> bool:
    conn.executescript(SCHEMA_SQL)
    added_column = _ensure_content_search_column(conn)
    _ensure_confidence_level_column(conn)  # 添加 confidence_level 列
    _ensure_verified_by_column(conn)  # 添加 verified_by 列

    needs_backfill = added_column
    if not needs_backfill:
        row = conn.execute("""
        SELECT 1 FROM memory_items
        WHERE content_search IS NULL OR content_search = ''
        LIMIT 1
        """).fetchone()
        needs_backfill = row is not None

    fts_ok, fts_needs_rebuild = _fts_status(conn)
    if fts_ok and (fts_needs_rebuild or needs_backfill):
        if needs_backfill:
            _backfill_content_search(conn)
        _rebuild_fts(conn)
    else:
        if fts_ok:
            _create_fts_triggers(conn)
        if needs_backfill:
            _backfill_content_search(conn)
        if fts_ok:
            conn.execute("""
            INSERT INTO memory_fts(id, content_search)
            SELECT id, COALESCE(NULLIF(content_search, ''), content)
            FROM memory_items
            WHERE id NOT IN (SELECT id FROM memory_fts)
            """)
    return fts_ok


def init_db(db_path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    conn = connect(db_path)
    fts_ok = _ensure_schema(conn)
    conn.commit()
    conn.close()

    result: Dict[str, Any] = {
        "ok": True,
        "db": os.path.abspath(db_path),
        "fts5": fts_ok
    }

    if not fts_ok:
        result["warning"] = "FTS5_UNAVAILABLE"
        result["message"] = "FTS5 not available, search will use LIKE fallback"

    return result


def _ensure_schema_once(db_path: str) -> None:
    if db_path in _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if db_path in _SCHEMA_READY:
            return
        init_db(db_path)
        _SCHEMA_READY.add(db_path)


def parse_tags(tags: Optional[List[str]]) -> List[str]:
    if not tags:
        return []
    out: List[str] = []
    for t in tags:
        for part in t.split(","):
            part = part.strip()
            if part:
                out.append(part)
    # 去重但保序
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def _tokenize_for_search(content: str) -> List[str]:
    if not content:
        return []
    try:
        import jieba
    except Exception:
        return []
    tokens: List[str] = []
    seen = set()
    for tok in jieba.cut_for_search(content):
        t = tok.strip()
        if len(t) < _MIN_TOKEN_LEN:
            continue
        if t in seen:
            continue
        tokens.append(t)
        seen.add(t)
    return tokens


def _jaccard_similarity(set1: set, set2: set) -> float:
    """计算两个集合的 Jaccard 相似度 (0.0 - 1.0)"""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _extract_keywords(content: str) -> set:
    """从内容中提取关键词集合（不依赖 jieba）"""
    if not content:
        return set()

    # 简单的分词：按空格、标点符号分割
    import re
    # 保留中文、英文、数字，其他作为分隔符
    words = re.findall(r'[\u4e00-\u9fff]+|\w+', content.lower())
    # 过滤太短的词（< 2个字符）
    keywords = {w for w in words if len(w) >= _MIN_TOKEN_LEN}
    return keywords


def _build_content_search(content: str) -> str:
    if not content:
        return ""
    tokens = _tokenize_for_search(content)
    if not tokens:
        return content
    return f"{content}\n{' '.join(tokens)}"


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _ensure_content_search_column(conn: sqlite3.Connection) -> bool:
    if _column_exists(conn, "memory_items", "content_search"):
        return False
    conn.execute("ALTER TABLE memory_items ADD COLUMN content_search TEXT NOT NULL DEFAULT ''")
    return True


def _ensure_confidence_level_column(conn: sqlite3.Connection) -> bool:
    """添加 confidence_level 列（默认为 medium）"""
    if _column_exists(conn, "memory_items", "confidence_level"):
        return False
    conn.execute("ALTER TABLE memory_items ADD COLUMN confidence_level TEXT NOT NULL DEFAULT 'medium'")
    return True


def _ensure_verified_by_column(conn: sqlite3.Connection) -> bool:
    """添加 verified_by 列（可空，用于标记来源）"""
    if _column_exists(conn, "memory_items", "verified_by"):
        return False
    conn.execute("ALTER TABLE memory_items ADD COLUMN verified_by TEXT")
    return True


def _backfill_content_search(conn: sqlite3.Connection) -> int:
    cur = conn.execute("""
    SELECT id, content FROM memory_items
    WHERE content_search IS NULL OR content_search = ''
    """)
    updates: List[Tuple[str, str]] = []
    count = 0
    for row in cur:
        content = row["content"] or ""
        content_search = _build_content_search(content)
        updates.append((content_search, row["id"]))
        if len(updates) >= 200:
            conn.executemany("UPDATE memory_items SET content_search = ? WHERE id = ?", updates)
            count += len(updates)
            updates.clear()
    if updates:
        conn.executemany("UPDATE memory_items SET content_search = ? WHERE id = ?", updates)
        count += len(updates)
    return count


def _coerce_weight(weight: Any) -> float:
    try:
        return float(weight)
    except (TypeError, ValueError):
        raise MemtoolError({
            "ok": False,
            "error": "PARAM_ERROR",
            "message": "weight must be a number"
        }, 2)


def find_by_logical_key(conn: sqlite3.Connection, type_: str, key: str,
                        task_id: Optional[str], step_id: Optional[str]) -> Optional[sqlite3.Row]:
    """按逻辑键 (type, key, task_id?, step_id?) 查找记录"""
    where = ["type = ?", "key = ?"]
    params: List[Any] = [type_, key]

    if task_id is not None:
        where.append("task_id = ?")
        params.append(task_id)
    else:
        where.append("task_id IS NULL")

    if step_id is not None:
        where.append("step_id = ?")
        params.append(step_id)
    else:
        where.append("step_id IS NULL")

    sql = f"SELECT * FROM memory_items WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT 1"
    return conn.execute(sql, tuple(params)).fetchone()


def _row_to_obj(r: sqlite3.Row) -> Dict[str, Any]:
    obj = {
        "id": r["id"],
        "type": r["type"],
        "task_id": r["task_id"],
        "step_id": r["step_id"],
        "key": r["key"],
        "content": r["content"],
        "tags": json.loads(r["tags_json"] or "[]"),
        "source": r["source"],
        "weight": r["weight"],
        "version": r["version"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }

    # 安全地添加可选字段
    try:
        obj["confidence_level"] = r["confidence_level"]
    except (KeyError, IndexError):
        obj["confidence_level"] = "medium"

    try:
        obj["verified_by"] = r["verified_by"]
    except (KeyError, IndexError):
        obj["verified_by"] = None

    return obj


def _db_error_payload(e: sqlite3.Error) -> Dict[str, Any]:
    error_msg = str(e)
    hint = ""

    if isinstance(e, sqlite3.OperationalError):
        if "locked" in error_msg.lower():
            hint = "Another process may be writing. Retry after a short delay."
        elif "readonly" in error_msg.lower():
            hint = "Database file is read-only. Check file permissions."
    elif isinstance(e, sqlite3.DatabaseError):
        hint = "Database may be corrupted or disk is full."

    return {
        "ok": False,
        "error": "DB_ERROR",
        "message": f"Database error: {error_msg}",
        "hint": hint
    }


def _raise_db_error(e: sqlite3.Error) -> None:
    raise MemtoolError(_db_error_payload(e), 4)


class MemoryStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        _ensure_schema_once(db_path)

    def init_db(self) -> Dict[str, Any]:
        return init_db(self._db_path)

    def find_similar_items(
        self,
        content: str,
        type: Optional[str] = None,
        threshold: float = 0.8,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        查找与给定 content 相似的已有记录（基于 Jaccard 相似度）

        Args:
            content: 要比对的内容
            type: 可选，仅在该 type 的记录中搜索
            threshold: 相似度阈值 (0.0 - 1.0)，默认 80%
            limit: 最多返回多少条相似结果

        Returns:
            相似度 >= threshold 的记录列表，按相似度降序排列
        """
        if not content:
            return []

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)

            # 提取查询内容的关键词
            query_keywords = _extract_keywords(content)
            if not query_keywords:
                return []

            # 获取候选记录
            where = []
            params: List[Any] = []
            if type:
                where.append("type = ?")
                params.append(type)

            where_clause = f" WHERE {' AND '.join(where)}" if where else ""
            sql = f"SELECT id, content FROM memory_items{where_clause} ORDER BY updated_at DESC LIMIT 1000"

            rows = conn.execute(sql, tuple(params)).fetchall()

            # 计算相似度
            results = []
            for row in rows:
                candidate_keywords = _extract_keywords(row["content"])
                similarity = _jaccard_similarity(query_keywords, candidate_keywords)

                if similarity >= threshold:
                    obj = _row_to_obj(conn.execute("SELECT * FROM memory_items WHERE id = ?", (row["id"],)).fetchone())
                    obj["similarity"] = round(similarity, 3)
                    results.append(obj)

            # 按相似度降序排列
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]

        except sqlite3.Error as e:
            return []  # 相似度检查不应该因为 DB 错误而失败，直接返回空
        finally:
            if conn is not None:
                conn.close()

    def put(
        self,
        *,
        item_id: Optional[str],
        type: str,
        key: str,
        content: str,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        weight: float = 1.0,
        confidence_level: str = "medium",
        verified_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            now = utcnow_iso()
            tags_json = json.dumps(tags or [], ensure_ascii=False)
            weight_value = _coerce_weight(weight)
            content_search = _build_content_search(content)

            if item_id:
                cur = conn.execute("SELECT id, version FROM memory_items WHERE id = ?", (item_id,))
                row = cur.fetchone()
                if row is None:
                    raise MemtoolError({
                        "ok": False,
                        "error": "NOT_FOUND",
                        "message": f"No memory item found with id: {item_id}",
                        "id": item_id
                    }, 3)

                version = int(row["version"]) + 1
                conn.execute("""
                UPDATE memory_items
                SET type=?, task_id=?, step_id=?, key=?, content=?, content_search=?, tags_json=?, source=?, weight=?, version=?, updated_at=?, confidence_level=?, verified_by=?
                WHERE id=?
                """, (
                    type, task_id, step_id, key, content, content_search,
                    tags_json, source, weight_value, version, now, confidence_level, verified_by, item_id
                ))
                action = "updated"
                final_id = item_id
            else:
                existing = find_by_logical_key(conn, type, key, task_id, step_id)
                if existing:
                    final_id = existing["id"]
                    version = int(existing["version"]) + 1
                    conn.execute("""
                    UPDATE memory_items
                    SET type=?, task_id=?, step_id=?, key=?, content=?, content_search=?, tags_json=?, source=?, weight=?, version=?, updated_at=?, confidence_level=?, verified_by=?
                    WHERE id=?
                    """, (
                        type, task_id, step_id, key, content, content_search,
                        tags_json, source, weight_value, version, now, confidence_level, verified_by, final_id
                    ))
                    action = "updated"
                else:
                    final_id = gen_id()
                    version = 1
                    conn.execute("""
                    INSERT INTO memory_items(id, type, task_id, step_id, key, content, content_search, tags_json, source, weight, version, created_at, updated_at, confidence_level, verified_by)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        final_id, type, task_id, step_id, key, content, content_search,
                        tags_json, source, weight_value, version, now, now, confidence_level, verified_by
                    ))
                    action = "inserted"

            conn.commit()
            result = {"ok": True, "action": action, "id": final_id, "version": version}

            # 相似度检查：在 insert 或 update 后，检查是否有类似的记录
            similar_items = self.find_similar_items(content, type=type, threshold=0.8, limit=3)
            if similar_items:
                result["warning"] = "duplicate_detection"
                result["similar_items"] = [
                    {
                        "id": item["id"],
                        "key": item["key"],
                        "similarity": item["similarity"],
                        "updated_at": item["updated_at"],
                    }
                    for item in similar_items
                ]

            return result

        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return {"ok": False, "error": "UNKNOWN"}

    def get(
        self,
        *,
        item_id: Optional[str] = None,
        type: Optional[str] = None,
        key: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            if item_id:
                row = conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
                if row is None:
                    raise MemtoolError({"ok": False, "error": "NOT_FOUND", "id": item_id}, 3)
                return _row_to_obj(row)

            where = ["type = ?", "key = ?"]
            params: List[Any] = [type, key]
            if task_id is not None:
                where.append("task_id = ?")
                params.append(task_id)
            else:
                where.append("task_id IS NULL")
            if step_id is not None:
                where.append("step_id = ?")
                params.append(step_id)
            else:
                where.append("step_id IS NULL")

            sql = f"SELECT * FROM memory_items WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT 1"
            row = conn.execute(sql, tuple(params)).fetchone()
            if row is None:
                raise MemtoolError({
                    "ok": False,
                    "error": "NOT_FOUND",
                    "query": {"type": type, "key": key, "task_id": task_id, "step_id": step_id}
                }, 3)
            return _row_to_obj(row)

        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return {"ok": False, "error": "UNKNOWN"}

    def list(
        self,
        *,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        key: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        sort_by: str = "updated",
        include_stale: bool = True,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            where = []
            params: List[Any] = []
            if type:
                where.append("type = ?")
                params.append(type)
            if task_id is not None:
                where.append("task_id = ?")
                params.append(task_id)
            if step_id is not None:
                where.append("step_id = ?")
                params.append(step_id)
            if key:
                where.append("key = ?")
                params.append(key)
            if tags:
                for t in tags:
                    where.append("tags_json LIKE ?")
                    params.append(f'%"{t}"%')

            sql = "SELECT * FROM memory_items"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            fetch_limit = int(limit)
            if sort_by in {"confidence", "recency", "mixed"}:
                fetch_limit = min(max(fetch_limit * 5, 200), 1000)
            params.append(fetch_limit)

            rows = conn.execute(sql, tuple(params)).fetchall()
            items = [_row_to_obj(r) for r in rows]

            now = _dt.datetime.now(tz=_dt.timezone.utc)
            for item in items:
                item.update(lifecycle_meta(item, now=now, stale_threshold=stale_threshold))
                item.update(score_item(item, now=now))

            if not include_stale:
                items = [i for i in items if not i.get("is_stale")]

            if sort_by == "confidence":
                items.sort(key=lambda x: (x.get("confidence_score", 0.0), x.get("updated_at") or ""), reverse=True)
            elif sort_by == "recency":
                items.sort(key=lambda x: (x.get("recency_score", 0.0), x.get("updated_at") or ""), reverse=True)
            elif sort_by == "mixed":
                items.sort(key=lambda x: (x.get("mixed_score", 0.0), x.get("updated_at") or ""), reverse=True)

            return items[: int(limit)]

        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return []

    def search(
        self,
        *,
        query: str,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 20,
        sort_by: str = "updated",
        include_stale: bool = True,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    ) -> Dict[str, Any]:
        q = query.strip()
        if not q:
            raise MemtoolError({"ok": False, "error": "PARAM_ERROR", "message": "Query string cannot be empty"}, 2)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            fts_ok = True
            try:
                conn.execute("SELECT 1 FROM memory_fts LIMIT 1;").fetchone()
            except sqlite3.OperationalError:
                fts_ok = False

            where = []
            params: List[Any] = []
            if type:
                where.append("m.type = ?")
                params.append(type)
            if task_id is not None:
                where.append("m.task_id = ?")
                params.append(task_id)
            if step_id is not None:
                where.append("m.step_id = ?")
                params.append(step_id)
            if key:
                where.append("m.key = ?")
                params.append(key)

            rows: List[sqlite3.Row]
            if fts_ok:
                try:
                    sql = """
                    SELECT m.*
                    FROM memory_fts f
                    JOIN memory_items m ON m.id = f.id
                    WHERE f.content_search MATCH ?
                    """
                    f_params: List[Any] = [q]
                    if where:
                        sql += " AND " + " AND ".join(where)
                        f_params += params
                    sql += " ORDER BY m.updated_at DESC LIMIT ?"
                    fetch_limit = int(limit)
                    if sort_by in {"confidence", "recency", "mixed"}:
                        fetch_limit = min(max(fetch_limit * 5, 200), 1000)
                    f_params.append(fetch_limit)
                    rows = conn.execute(sql, tuple(f_params)).fetchall()
                except sqlite3.Error:
                    fts_ok = False

            if not fts_ok:
                sql = "SELECT m.* FROM memory_items m WHERE COALESCE(NULLIF(m.content_search, ''), m.content) LIKE ?"
                like_params: List[Any] = [f"%{q}%"]
                if where:
                    sql += " AND " + " AND ".join(where)
                    like_params += params
                sql += " ORDER BY m.updated_at DESC LIMIT ?"
                fetch_limit = int(limit)
                if sort_by in {"confidence", "recency", "mixed"}:
                    fetch_limit = min(max(fetch_limit * 5, 200), 1000)
                like_params.append(fetch_limit)
                rows = conn.execute(sql, tuple(like_params)).fetchall()

            items = [_row_to_obj(r) for r in rows]
            now = _dt.datetime.now(tz=_dt.timezone.utc)
            for item in items:
                item.update(lifecycle_meta(item, now=now, stale_threshold=stale_threshold))
                item.update(score_item(item, now=now))

            if not include_stale:
                items = [i for i in items if not i.get("is_stale")]

            if sort_by == "confidence":
                items.sort(key=lambda x: (x.get("confidence_score", 0.0), x.get("updated_at") or ""), reverse=True)
            elif sort_by == "recency":
                items.sort(key=lambda x: (x.get("recency_score", 0.0), x.get("updated_at") or ""), reverse=True)
            elif sort_by == "mixed":
                items.sort(key=lambda x: (x.get("mixed_score", 0.0), x.get("updated_at") or ""), reverse=True)

            return {"ok": True, "fts5": fts_ok, "items": items[: int(limit)]}

        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return {"ok": False, "error": "UNKNOWN", "items": []}

    def delete(self, *, item_id: str) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            cur = conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
            conn.commit()
            return {"ok": True, "deleted": cur.rowcount}

        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return {"ok": False, "error": "UNKNOWN"}

    def recommend(
        self,
        *,
        context: Optional[str] = None,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        key_prefix: Optional[str] = None,
        limit: int = 10,
        include_stale: bool = False,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    ) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            where = []
            params: List[Any] = []
            if type:
                where.append("type = ?")
                params.append(type)
            if task_id is not None:
                where.append("task_id = ?")
                params.append(task_id)
            if key_prefix:
                where.append("key LIKE ?")
                params.append(f"{key_prefix}%")
            if tags:
                for t in tags:
                    where.append("tags_json LIKE ?")
                    params.append(f'%"{t}"%')

            sql = "SELECT * FROM memory_items"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            fetch_limit = min(max(int(limit) * 20, 200), 1000)
            params.append(fetch_limit)
            rows = conn.execute(sql, tuple(params)).fetchall()
            items = [_row_to_obj(r) for r in rows]

            recs = recommend_items(
                items,
                context=context,
                limit=int(limit),
                include_stale=include_stale,
                stale_threshold=stale_threshold,
            )
            return {"ok": True, "items": recs, "candidates": len(items), "limit": int(limit)}
        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return {"ok": False, "error": "UNKNOWN"}

    def cleanup(
        self,
        *,
        type: Optional[str] = None,
        older_than_days: Optional[float] = None,
        stale_threshold: Optional[float] = None,
        limit: int = 1000,
        apply: bool = False,
    ) -> Dict[str, Any]:
        if older_than_days is None and stale_threshold is None:
            raise MemtoolError({
                "ok": False,
                "error": "PARAM_ERROR",
                "message": "cleanup requires older_than_days or stale_threshold"
            }, 2)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            where = []
            params: List[Any] = []
            if type:
                where.append("type = ?")
                params.append(type)
            sql = "SELECT * FROM memory_items"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY updated_at ASC LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(sql, tuple(params)).fetchall()
            items = [_row_to_obj(r) for r in rows]
            candidates = cleanup_candidates(
                items,
                older_than_days=older_than_days,
                stale_threshold=stale_threshold,
            )

            deleted = 0
            if apply and candidates:
                ids = [(c["id"],) for c in candidates]
                conn.executemany("DELETE FROM memory_items WHERE id = ?", ids)
                conn.commit()
                deleted = len(ids)

            return {
                "ok": True,
                "apply": apply,
                "deleted": deleted,
                "candidates": candidates,
            }
        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

    def export_items(self) -> List[Dict[str, Any]]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            rows = conn.execute("SELECT * FROM memory_items ORDER BY updated_at ASC").fetchall()
            return [_row_to_obj(r) for r in rows]

        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return []

    def import_items(self, items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect(self._db_path)
            now = utcnow_iso()
            inserted = 0
            updated = 0
            for obj in items:
                item_id = obj.get("id") or gen_id()
                cur = conn.execute("SELECT id FROM memory_items WHERE id = ?", (item_id,))
                row = cur.fetchone()
                tags = obj.get("tags", [])
                weight_value = _coerce_weight(obj.get("weight", 1.0))
                content = obj.get("content", "")
                content_search = obj.get("content_search") or _build_content_search(content)
                if row is None:
                    conn.execute("""
                    INSERT INTO memory_items(id, type, task_id, step_id, key, content, content_search, tags_json, source, weight, version, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        item_id,
                        obj.get("type", "feature"),
                        obj.get("task_id"),
                        obj.get("step_id"),
                        obj.get("key", "unknown"),
                        content,
                        content_search,
                        json.dumps(tags, ensure_ascii=False),
                        obj.get("source"),
                        weight_value,
                        int(obj.get("version", 1)),
                        obj.get("created_at", now),
                        obj.get("updated_at", now),
                    ))
                    inserted += 1
                else:
                    conn.execute("""
                    UPDATE memory_items
                    SET type=?, task_id=?, step_id=?, key=?, content=?, content_search=?, tags_json=?, source=?, weight=?, version=?, updated_at=?
                    WHERE id=?
                    """, (
                        obj.get("type", "feature"),
                        obj.get("task_id"),
                        obj.get("step_id"),
                        obj.get("key", "unknown"),
                        content,
                        content_search,
                        json.dumps(tags, ensure_ascii=False),
                        obj.get("source"),
                        weight_value,
                        int(obj.get("version", 1)),
                        obj.get("updated_at", now),
                        item_id
                    ))
                    updated += 1
            conn.commit()
            return {"ok": True, "inserted": inserted, "updated": updated}

        except sqlite3.Error as e:
            _raise_db_error(e)
        finally:
            if conn is not None:
                conn.close()

        return {"ok": False, "error": "UNKNOWN"}

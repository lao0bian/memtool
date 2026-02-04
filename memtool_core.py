#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memtool_core: SQLite 记忆管理核心逻辑（无 CLI I/O）
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sqlite3
import uuid
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from memtool_lifecycle import DEFAULT_STALE_THRESHOLD, cleanup_candidates, lifecycle_meta
from memtool_rank import score_item
from memtool_recommend import recommend_items
from memtool.storage.connection import SQLiteConnectionPool
from memtool.utils import _extract_keywords

# Try to import semantic search mixin
try:
    from memtool.embedding.semantic import SemanticSearchMixin
    _HAS_VECTOR = True
except ImportError:
    SemanticSearchMixin = object  # type: ignore
    _HAS_VECTOR = False

DEFAULT_DB = os.environ.get("MEMTOOL_DB", "./memtool.db")
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY: set[str] = set()
_MIN_TOKEN_LEN = 2
logger = logging.getLogger(__name__)


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
  -- Phase 2-2: Contextual Memory Fields
  context_tags_json TEXT NOT NULL DEFAULT '[]',
  emotional_valence REAL NOT NULL DEFAULT 0.0,
  urgency_level INTEGER NOT NULL DEFAULT 0,
  related_json TEXT NOT NULL DEFAULT '[]',
  session_id TEXT,
  source TEXT,
  weight REAL NOT NULL DEFAULT 1.0,
  version INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  -- Phase 2-1: Memory Consolidation Fields
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TEXT,
  consolidation_score REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_memory_type_key ON memory_items(type, key);
CREATE INDEX IF NOT EXISTS idx_memory_task_step ON memory_items(task_id, step_id);
CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_items(updated_at);
CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(type);
CREATE INDEX IF NOT EXISTS idx_memory_weight ON memory_items(weight);

-- 可选：全文检索（FTS5）。如果宿主 SQLite 不支持，会在 init 时提示并自动降级为 LIKE。
"""


def _ensure_phase2_indexes(conn: sqlite3.Connection) -> None:
    """Phase 2-1: 创建记忆巩固字段的索引（仅当字段存在时）"""
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_access_count ON memory_items(access_count)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_consolidation ON memory_items(consolidation_score)")
    except sqlite3.OperationalError:
        # 字段还不存在，稍后会添加
        pass


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
    _ensure_phase2_columns(conn)  # Phase 2-1: 添加记忆巩固字段
    _ensure_phase2_2a_columns(conn)  # Phase 2-2a: 添加情境字段
    _ensure_history_table(conn)  # Phase 2.6: 添加历史表
    
    # Create indexes after ensuring columns exist
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_confidence ON memory_items(confidence_level)")
    _ensure_phase2_indexes(conn)  # Phase 2-1: 创建巩固字段的索引
    _ensure_phase2_2a_indexes(conn)  # Phase 2-2a: 创建情境字段索引

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


_jieba = None


def _get_jieba():
    """获取 jieba 实例（带缓存）"""
    global _jieba
    if _jieba is None:
        try:
            import jieba
            _jieba = jieba
        except ImportError:
            _jieba = False
    return _jieba if _jieba else None


def _tokenize_for_search(content: str) -> List[str]:
    if not content:
        return []
    jieba = _get_jieba()
    if jieba is None:
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


# _extract_keywords moved to memtool.utils


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


def _ensure_phase2_columns(conn: sqlite3.Connection) -> bool:
    """Phase 2-1: 添加记忆巩固字段"""
    added = False
    
    if not _column_exists(conn, "memory_items", "access_count"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0")
        added = True
    
    if not _column_exists(conn, "memory_items", "last_accessed_at"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN last_accessed_at TEXT")
        added = True
    
    if not _column_exists(conn, "memory_items", "consolidation_score"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN consolidation_score REAL NOT NULL DEFAULT 0.0")
        added = True
    
    return added


def _ensure_phase2_2a_columns(conn: sqlite3.Connection) -> bool:
    """Phase 2-2a: 添加情境字段"""
    added = False

    if not _column_exists(conn, "memory_items", "context_tags_json"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN context_tags_json TEXT NOT NULL DEFAULT '[]'")
        added = True

    if not _column_exists(conn, "memory_items", "emotional_valence"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN emotional_valence REAL NOT NULL DEFAULT 0.0")
        added = True

    if not _column_exists(conn, "memory_items", "urgency_level"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN urgency_level INTEGER NOT NULL DEFAULT 0")
        added = True

    if not _column_exists(conn, "memory_items", "related_json"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN related_json TEXT NOT NULL DEFAULT '[]'")
        added = True

    if not _column_exists(conn, "memory_items", "session_id"):
        conn.execute("ALTER TABLE memory_items ADD COLUMN session_id TEXT")
        added = True

    return added


def _ensure_phase2_2a_indexes(conn: sqlite3.Connection) -> None:
    """Phase 2-2a: 创建情境字段索引（仅当字段存在时）"""
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_emotional ON memory_items(emotional_valence)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_urgency ON memory_items(urgency_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_items(session_id)")
    except sqlite3.OperationalError:
        pass


def _ensure_history_table(conn: sqlite3.Connection) -> bool:
    """Phase 2.6: 确保 memory_history 表存在"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            weight REAL NOT NULL DEFAULT 1.0,
            confidence_level TEXT NOT NULL DEFAULT 'medium',
            changed_at TEXT NOT NULL,
            change_type TEXT NOT NULL CHECK (change_type IN ('update', 'delete'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_item_id ON memory_history(item_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_version ON memory_history(item_id, version)")
    return True


def _save_history(
    conn: sqlite3.Connection,
    item_id: str,
    old_row: sqlite3.Row,
    change_type: str = "update"
) -> None:
    """Phase 2.6: 将旧版本保存到历史表（在同一事务内）"""
    # Safely extract confidence_level (sqlite3.Row doesn't have .get() method)
    try:
        confidence_level = old_row["confidence_level"]
    except (KeyError, IndexError):
        confidence_level = "medium"
    
    conn.execute("""
        INSERT INTO memory_history(
            item_id, version, content, tags_json, weight, 
            confidence_level, changed_at, change_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item_id,
        old_row["version"],
        old_row["content"],
        old_row["tags_json"],
        old_row["weight"],
        confidence_level,
        utcnow_iso(),
        change_type
    ))


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

    # Phase 2-1: 添加记忆巩固字段
    try:
        obj["access_count"] = r["access_count"]
    except (KeyError, IndexError):
        obj["access_count"] = 0

    try:
        obj["last_accessed_at"] = r["last_accessed_at"]
    except (KeyError, IndexError):
        obj["last_accessed_at"] = None

    try:
        obj["consolidation_score"] = r["consolidation_score"]
    except (KeyError, IndexError):
        obj["consolidation_score"] = 0.0

    try:
        obj["context_tags"] = json.loads(r["context_tags_json"] or "[]")
    except (KeyError, IndexError, TypeError):
        obj["context_tags"] = []

    try:
        obj["emotional_valence"] = r["emotional_valence"]
    except (KeyError, IndexError):
        obj["emotional_valence"] = 0.0

    try:
        obj["urgency_level"] = r["urgency_level"]
    except (KeyError, IndexError):
        obj["urgency_level"] = 0

    try:
        obj["related"] = json.loads(r["related_json"] or "[]")
    except (KeyError, IndexError, TypeError):
        obj["related"] = []

    try:
        obj["session_id"] = r["session_id"]
    except (KeyError, IndexError):
        obj["session_id"] = None

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


class MemoryStore(SemanticSearchMixin):
    """Memory storage with SQLite backend and optional vector search
    
    Vector search is enabled when:
    - MEMTOOL_VECTOR_ENABLED=on (force enable)
    - MEMTOOL_VECTOR_ENABLED=auto (enable if dependencies available)
    - MEMTOOL_VECTOR_ENABLED=off (disable)
    
    Environment variables:
    - MEMTOOL_VECTOR_DIR: Vector storage directory (default: alongside db)
    - MEMTOOL_EMBEDDING_PROVIDER: local, openai, or ollama
    - MEMTOOL_EMBEDDING_MODEL: Override default model
    """
    
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        _ensure_schema_once(db_path)
        # Initialize connection pool for better performance
        self._pool = SQLiteConnectionPool(db_path)
        if hasattr(self, "_init_vector_attrs"):
            self._init_vector_attrs()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection from pool (thread-safe)"""
        return self._pool.get_connection()

    def _compute_consolidation(
        self,
        access_count: int,
        created_at: str,
        updated_at: str,
        now: Optional[_dt.datetime] = None
    ) -> float:
        """
        Phase 2-1: 计算记忆巩固分数
        
        巩固分数 = 访问频率分 × 时间权重
        
        - 访问频率分: log(1 + access_count) 归一化到 0-1
        - 时间权重: 考虑创建时间和最近更新时间
        
        Returns:
            float: 巩固分数 (0.0 - 1.0)
        """
        if now is None:
            now = _dt.datetime.now(tz=_dt.timezone.utc)
        
        # 访问频率分 (对数增长，避免线性爆炸)
        # access_count=0 → 0.0, access_count=100 → 1.0
        import math
        frequency_score = min(math.log(1 + access_count) / math.log(100), 1.0)
        
        # 时间跨度分 (存在越久 = 越重要)
        created_dt = _dt.datetime.fromisoformat(created_at)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=_dt.timezone.utc)
        age_days = (now - created_dt).total_seconds() / 86400.0
        longevity_score = min(age_days / 365.0, 1.0)
        
        # 活跃度分 (最近有更新 = 仍在使用)
        from memtool_lifecycle import decay_score
        recency = decay_score(updated_at, "feature", now=now)
        
        # 综合巩固分数
        consolidation = (
            frequency_score * 0.5 +
            longevity_score * 0.2 +
            recency * 0.3
        )
        
        return consolidation

    def _track_access(self, item_id: str) -> None:
        """
        Phase 2-1: 记录访问，更新巩固分数
        
        在以下场景触发：
        - get() 方法调用时
        - search() 返回的每个结果
        - recommend() 返回的每个结果
        """
        # 使用新连接以避免事务冲突
        conn = None
        try:
            conn = connect(self._db_path)
            now = utcnow_iso()
            
            # 更新访问记录
            conn.execute("""
                UPDATE memory_items
                SET access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE id = ?
            """, (now, item_id))
            
            # 重新计算巩固分数
            row = conn.execute(
                "SELECT access_count, created_at, updated_at FROM memory_items WHERE id = ?",
                (item_id,)
            ).fetchone()
            
            if row:
                consolidation = self._compute_consolidation(
                    access_count=row["access_count"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                conn.execute(
                    "UPDATE memory_items SET consolidation_score = ? WHERE id = ?",
                    (consolidation, item_id)
                )
            
            conn.commit()
        except sqlite3.Error:
            # 访问追踪失败不应影响主流程
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _track_access_batch(self, item_ids: List[str]) -> bool:
        """批量更新访问记录（不影响主流程）"""
        if not item_ids:
            return True

        try:
            conn = self._get_conn()
            now = utcnow_iso()
            placeholders = ",".join("?" * len(item_ids))
            conn.execute(
                f"""
                UPDATE memory_items
                SET access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE id IN ({placeholders})
                """,
                [now] + item_ids,
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.warning("Failed to track access for %s items: %s", len(item_ids), e)
            return False

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
            conn = self._get_conn()

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
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._get_conn()
            now = utcnow_iso()
            tags_json = json.dumps(tags or [], ensure_ascii=False)
            weight_value = _coerce_weight(weight)
            content_search = _build_content_search(content)
            from memtool.context.extractor import ContextExtractor
            context_tags, emotional_valence, urgency_level = ContextExtractor.extract(
                content=content,
                metadata={"type": type, "task_id": task_id, "step_id": step_id},
            )
            context_tags_json = json.dumps(context_tags, ensure_ascii=False)

            if item_id:
                cur = conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,))
                row = cur.fetchone()
                if row is None:
                    raise MemtoolError({
                        "ok": False,
                        "error": "NOT_FOUND",
                        "message": f"No memory item found with id: {item_id}",
                        "id": item_id
                    }, 3)

                # Phase 2.6: 保存旧版本到历史
                _save_history(conn, item_id, row, "update")

                version = int(row["version"]) + 1
                try:
                    related_json = row["related_json"] or "[]"
                except (KeyError, IndexError):
                    related_json = "[]"
                conn.execute("""
                UPDATE memory_items
                SET type=?, task_id=?, step_id=?, key=?, content=?, content_search=?, tags_json=?,
                    context_tags_json=?, emotional_valence=?, urgency_level=?, related_json=?, session_id=?,
                    source=?, weight=?, version=?, updated_at=?, confidence_level=?, verified_by=?
                WHERE id=?
                """, (
                    type, task_id, step_id, key, content, content_search,
                    tags_json, context_tags_json, emotional_valence, urgency_level, related_json, session_id,
                    source, weight_value, version, now, confidence_level, verified_by, item_id
                ))
                action = "updated"
                final_id = item_id
            else:
                existing = find_by_logical_key(conn, type, key, task_id, step_id)
                if existing:
                    final_id = existing["id"]
                    
                    # Phase 2.6: 保存旧版本到历史
                    old_row = conn.execute("SELECT * FROM memory_items WHERE id = ?", (final_id,)).fetchone()
                    if old_row:
                        _save_history(conn, final_id, old_row, "update")

                    version = int(existing["version"]) + 1
                    try:
                        related_json = old_row["related_json"] or "[]"
                    except (KeyError, IndexError, TypeError):
                        related_json = "[]"
                    conn.execute("""
                    UPDATE memory_items
                    SET type=?, task_id=?, step_id=?, key=?, content=?, content_search=?, tags_json=?,
                        context_tags_json=?, emotional_valence=?, urgency_level=?, related_json=?, session_id=?,
                        source=?, weight=?, version=?, updated_at=?, confidence_level=?, verified_by=?
                    WHERE id=?
                    """, (
                        type, task_id, step_id, key, content, content_search,
                        tags_json, context_tags_json, emotional_valence, urgency_level, related_json, session_id,
                        source, weight_value, version, now, confidence_level, verified_by, final_id
                    ))
                    action = "updated"
                else:
                    final_id = gen_id()
                    version = 1
                    conn.execute("""
                    INSERT INTO memory_items(
                        id, type, task_id, step_id, key, content, content_search, tags_json,
                        context_tags_json, emotional_valence, urgency_level, related_json, session_id,
                        source, weight, version, created_at, updated_at, confidence_level, verified_by
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        final_id, type, task_id, step_id, key, content, content_search,
                        tags_json, context_tags_json, emotional_valence, urgency_level, "[]", session_id,
                        source, weight_value, version, now, now, confidence_level, verified_by
                    ))
                    action = "inserted"

            conn.commit()
            result = {"ok": True, "action": action, "id": final_id, "version": version}

            # Auto-index to vector store if available
            if _HAS_VECTOR:
                try:
                    self.vector_index({
                        "id": final_id,
                        "content": content,
                        "type": type,
                        "key": key,
                        "task_id": task_id,
                        "step_id": step_id,
                        "tags": tags or [],
                        "weight": weight_value,
                        "confidence_level": confidence_level,
                        "updated_at": now,
                    })
                except Exception:
                    pass  # Don't fail put if vector indexing fails

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
            if conn is not None:
                conn.rollback()
            _raise_db_error(e)

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
            conn = self._get_conn()
            if item_id:
                row = conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
                if row is None:
                    raise MemtoolError({"ok": False, "error": "NOT_FOUND", "id": item_id}, 3)
                # Phase 2-1: 追踪访问
                self._track_access(item_id)
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
            # Phase 2-1: 追踪访问
            self._track_access(row["id"])
            return _row_to_obj(row)

        except sqlite3.Error as e:
            _raise_db_error(e)

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
        offset: int = 0,
        sort_by: str = "updated",
        include_stale: bool = True,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._get_conn()
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
            sql += " ORDER BY updated_at DESC"
            
            # Use SQL LIMIT and OFFSET for better performance
            fetch_limit = int(limit)
            fetch_offset = int(offset)
            if sort_by in {"confidence", "recency", "mixed"}:
                # For complex sorting, fetch more rows and sort in memory
                fetch_limit = min(max(fetch_limit * 5, 200), 1000)
                fetch_offset = 0  # Can't use offset with complex sorting
                sql += " LIMIT ?"
                params.append(fetch_limit)
            else:
                # For simple sorting, use SQL LIMIT OFFSET
                sql += " LIMIT ? OFFSET ?"
                params.append(fetch_limit)
                params.append(fetch_offset)

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

            # For complex sorting, apply offset after sorting
            if sort_by in {"confidence", "recency", "mixed"}:
                items = items[fetch_offset:fetch_offset + int(limit)]
            else:
                items = items[:int(limit)]

            return items

        except sqlite3.Error as e:
            _raise_db_error(e)

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
            conn = self._get_conn()
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

            # Phase 2-1: 追踪访问（对搜索结果）
            access_ids = [item["id"] for item in items[: int(limit)]]
            self._track_access_batch(access_ids)

            return {"ok": True, "fts5": fts_ok, "items": items[: int(limit)]}

        except sqlite3.Error as e:
            _raise_db_error(e)

        return {"ok": False, "error": "UNKNOWN", "items": []}

    def contextual_search(
        self,
        *,
        query: str,
        context_tags: Optional[List[str]] = None,
        emotional_filter: Optional[str] = None,
        urgency_min: Optional[int] = None,
        urgency_level: Optional[int] = None,
        limit: int = 10,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        include_stale: bool = True,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    ) -> Dict[str, Any]:
        """情境检索：基于上下文标签/情绪/紧急度过滤。"""
        try:
            fetch_limit = max(int(limit) * 3, 30)
        except (TypeError, ValueError):
            fetch_limit = 30

        try:
            base_results = self.hybrid_search(
                query=query,
                type=type,
                task_id=task_id,
                limit=fetch_limit,
            )
        except Exception:
            base_results = self.search(
                query=query,
                type=type,
                task_id=task_id,
                limit=fetch_limit,
                include_stale=include_stale,
                stale_threshold=stale_threshold,
            )

        items = base_results.get("items", [])
        if not include_stale:
            items = [i for i in items if not i.get("is_stale")]

        filtered: List[Dict[str, Any]] = []
        for item in items:
            item_tags = item.get("context_tags", []) or []
            item_valence = item.get("emotional_valence", 0.0) or 0.0
            item_urgency = item.get("urgency_level", 0) or 0

            if context_tags:
                overlap = len(set(context_tags) & set(item_tags))
                if overlap == 0:
                    continue
                item["context_match_score"] = overlap / len(context_tags)

            if emotional_filter:
                if emotional_filter == "positive" and item_valence <= 0:
                    continue
                if emotional_filter == "negative" and item_valence >= 0:
                    continue
                if emotional_filter == "neutral" and item_valence != 0:
                    continue

            if urgency_level is not None and item_urgency != int(urgency_level):
                continue
            if urgency_min is not None and item_urgency < int(urgency_min):
                continue

            filtered.append(item)

        if context_tags:
            filtered.sort(key=lambda x: x.get("context_match_score", 0), reverse=True)

        return {
            "ok": True,
            "items": filtered[: int(limit)],
            "total_found": len(filtered),
            "filters_applied": {
                "context_tags": context_tags,
                "emotional_filter": emotional_filter,
                "urgency_min": urgency_min,
                "urgency_level": urgency_level,
            },
        }

    def delete(self, *, item_id: str) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
            conn.commit()
            return {"ok": True, "deleted": cur.rowcount}

        except sqlite3.Error as e:
            if conn is not None:
                conn.rollback()
            _raise_db_error(e)

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
            conn = self._get_conn()
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
            
            # Phase 2-1: 追踪访问（对推荐结果）
            access_ids = [item["id"] for item in recs]
            self._track_access_batch(access_ids)
            
            return {"ok": True, "items": recs, "candidates": len(items), "limit": int(limit)}
        except sqlite3.Error as e:
            _raise_db_error(e)

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
            conn = self._get_conn()
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

    def export_items(self) -> List[Dict[str, Any]]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._get_conn()
            rows = conn.execute("SELECT * FROM memory_items ORDER BY updated_at ASC").fetchall()
            return [_row_to_obj(r) for r in rows]

        except sqlite3.Error as e:
            _raise_db_error(e)

        return []

    def import_items(self, items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._get_conn()
            now = utcnow_iso()
            inserted = 0
            updated = 0
            for obj in items:
                item_id = obj.get("id") or gen_id()
                cur = conn.execute("SELECT id FROM memory_items WHERE id = ?", (item_id,))
                row = cur.fetchone()
                tags = obj.get("tags", [])
                context_tags = obj.get("context_tags")
                if context_tags is None:
                    context_tags = obj.get("context_tags_json")
                if isinstance(context_tags, str):
                    context_tags_json = context_tags
                else:
                    context_tags_json = json.dumps(context_tags or [], ensure_ascii=False)
                related = obj.get("related")
                if related is None:
                    related = obj.get("related_json")
                if isinstance(related, str):
                    related_json = related
                else:
                    related_json = json.dumps(related or [], ensure_ascii=False)
                emotional_valence = obj.get("emotional_valence", 0.0)
                urgency_level = int(obj.get("urgency_level", 0))
                session_id = obj.get("session_id")
                weight_value = _coerce_weight(obj.get("weight", 1.0))
                content = obj.get("content", "")
                content_search = obj.get("content_search") or _build_content_search(content)
                if row is None:
                    conn.execute("""
                    INSERT INTO memory_items(
                        id, type, task_id, step_id, key, content, content_search, tags_json,
                        context_tags_json, emotional_valence, urgency_level, related_json, session_id,
                        source, weight, version, created_at, updated_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        item_id,
                        obj.get("type", "feature"),
                        obj.get("task_id"),
                        obj.get("step_id"),
                        obj.get("key", "unknown"),
                        content,
                        content_search,
                        json.dumps(tags, ensure_ascii=False),
                        context_tags_json,
                        emotional_valence,
                        urgency_level,
                        related_json,
                        session_id,
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
                    SET type=?, task_id=?, step_id=?, key=?, content=?, content_search=?, tags_json=?,
                        context_tags_json=?, emotional_valence=?, urgency_level=?, related_json=?, session_id=?,
                        source=?, weight=?, version=?, updated_at=?
                    WHERE id=?
                    """, (
                        obj.get("type", "feature"),
                        obj.get("task_id"),
                        obj.get("step_id"),
                        obj.get("key", "unknown"),
                        content,
                        content_search,
                        json.dumps(tags, ensure_ascii=False),
                        context_tags_json,
                        emotional_valence,
                        urgency_level,
                        related_json,
                        session_id,
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

        return {"ok": False, "error": "UNKNOWN"}

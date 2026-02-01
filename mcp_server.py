#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP server for memtool (stdio transport).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from memtool_core import DEFAULT_DB, MemtoolError, MemoryStore, init_db

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - depends on runtime env
    FastMCP = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

LOG = logging.getLogger("memtool.mcp")

_WRITE_LOCK = threading.Lock()
_INIT_LOCK = threading.Lock()
_VALID_TYPES = {"project", "feature", "run"}


def _setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, handlers=[logging.StreamHandler(sys.stderr)], force=True)
    LOG.setLevel(level)


def _resolve_db_path(db_path: Optional[str]) -> str:
    if db_path:
        return db_path
    env_db = os.environ.get("MEMTOOL_DB")
    if env_db:
        return env_db
    return DEFAULT_DB


def _ensure_db(db_path: str) -> None:
    if os.path.exists(db_path):
        return
    with _INIT_LOCK:
        if os.path.exists(db_path):
            return
        init_db(db_path)


def _store_for(db_path: Optional[str]) -> MemoryStore:
    path = _resolve_db_path(db_path)
    _ensure_db(path)
    return MemoryStore(path)


def assess_knowledge(
    topic: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase 2-1: 评估 Agent 对某个主题的记忆可信度（元认知）
    
    独立函数版本，可供测试和其他模块使用
    """
    if not topic or not str(topic).strip():
        return _param_error("topic cannot be empty")
    
    try:
        store = _store_for(db_path)
        
        # 搜索相关记忆（使用 hybrid 搜索更准确）
        try:
            results = store.hybrid_search(query=str(topic), limit=50)
        except Exception:
            # Fallback to regular search if hybrid not available
            results = store.search(query=str(topic), limit=50)
        
        items = results.get("items", [])
        
        if not items:
            return {
                "ok": True,
                "confidence": "none",
                "score": 0.0,
                "message": f"我对「{topic}」没有记忆",
                "suggestions": [
                    "建议通过 web_search 或查阅文档获取信息",
                    "获取信息后用 memory_store 记录"
                ],
                "evidence": [],
                "stats": {
                    "total_memories": 0,
                    "avg_confidence": 0.0,
                    "avg_recency": 0.0,
                    "avg_consolidation": 0.0,
                    "coverage": 0.0
                }
            }
        
        # 计算元认知分数
        from memtool_rank import score_item
        import datetime as dt
        now = dt.datetime.now(tz=dt.timezone.utc)
        
        # 为每个 item 计算分数（如果还没有）
        for item in items:
            if "confidence_score" not in item:
                item.update(score_item(item, now=now))
        
        avg_confidence = sum(item.get("confidence_score", 0.6) for item in items) / len(items)
        coverage = min(len(items) / 10.0, 1.0)  # 10 条记忆 = 完整覆盖
        avg_recency = sum(item.get("recency_score", 0.5) for item in items) / len(items)
        avg_consolidation = sum(item.get("consolidation_score", 0.0) for item in items) / len(items)
        
        # 综合元认知分数
        meta_score = (
            avg_confidence * 0.35 +
            coverage * 0.25 +
            avg_recency * 0.20 +
            avg_consolidation * 0.20
        )
        
        # 分级评估
        if meta_score >= 0.8:
            level = "high"
            message = f"我对「{topic}」很有把握，有 {len(items)} 条相关记忆"
            suggestions = ["可以直接使用这些记忆回答问题"]
        elif meta_score >= 0.5:
            level = "medium"
            message = f"我对「{topic}」有一些了解，但不完全确定"
            suggestions = [
                "建议交叉验证这些记忆",
                "如果是关键决策，最好查询最新资料"
            ]
        elif meta_score >= 0.2:
            level = "low"
            message = f"我对「{topic}」的记忆比较模糊"
            suggestions = [
                "记忆可能过时或不完整",
                "建议重新学习或查询外部资源",
                f"找到 {len(items)} 条记忆但质量较低"
            ]
        else:
            level = "very_low"
            message = f"我对「{topic}」几乎没有可靠记忆"
            suggestions = [
                "强烈建议查询外部资源",
                "获取新信息后更新记忆库"
            ]
        
        # 提取高质量证据
        evidence = sorted(items, key=lambda x: x.get("confidence_score", 0), reverse=True)[:5]
        
        return {
            "ok": True,
            "confidence": level,
            "score": round(meta_score, 4),
            "message": message,
            "suggestions": suggestions,
            "evidence": [
                {
                    "id": e["id"],
                    "key": e["key"],
                    "confidence_level": e.get("confidence_level"),
                    "access_count": e.get("access_count", 0),
                    "consolidation_score": round(e.get("consolidation_score", 0.0), 3),
                    "updated_at": e.get("updated_at"),
                    "snippet": e.get("content", "")[:100] + "..."
                }
                for e in evidence
            ],
            "stats": {
                "total_memories": len(items),
                "avg_confidence": round(avg_confidence, 3),
                "avg_recency": round(avg_recency, 3),
                "avg_consolidation": round(avg_consolidation, 3),
                "coverage": round(coverage, 3)
            }
        }
    except MemtoolError as e:
        return e.payload
    except Exception as e:
        return _unexpected_error("assess_knowledge", e)


def _param_error(message: str, hint: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "error": "PARAM_ERROR", "message": message}
    if hint:
        payload["hint"] = hint
    return payload


def _unexpected_error(context: str, exc: Exception) -> Dict[str, Any]:
    LOG.exception("Unhandled error in %s", context)
    return {"ok": False, "error": "GENERAL_ERROR", "message": f"{context} failed"}


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _normalize_tags(tags: Optional[Any]) -> Optional[List[str]]:
    if tags is None:
        return None
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    if isinstance(tags, list):
        out: List[str] = []
        for t in tags:
            if t is None:
                continue
            s = str(t).strip()
            if s:
                out.append(s)
        return out
    return None


if FastMCP is not None:
    mcp = FastMCP("memtool")

    @mcp.tool()
    def memory_store(
        type: str,
        key: str,
        content: Any,
        item_id: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        tags: Optional[Any] = None,
        source: Optional[str] = None,
        weight: float = 1.0,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store or update memory (upsert by id or logical key)."""
        if not type or not key:
            return _param_error("type and key are required")
        if type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        if content is None:
            return _param_error("content is required")
        if tags is not None and not isinstance(tags, (str, list)):
            return _param_error("tags must be a string or list")
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            return _param_error("weight must be a number")
        try:
            content_value = _coerce_content(content)
        except (TypeError, ValueError):
            return _param_error("content must be JSON-serializable or string")

        try:
            store = _store_for(db_path)
            with _WRITE_LOCK:
                return store.put(
                    item_id=item_id,
                    type=type,
                    key=key,
                    content=content_value,
                    task_id=task_id,
                    step_id=step_id,
                    tags=_normalize_tags(tags),
                    source=source,
                    weight=weight_value,
                )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_store", e)

    @mcp.tool()
    def memory_recall(
        item_id: Optional[str] = None,
        type: Optional[str] = None,
        key: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recall a memory by id or logical key (type+key, optional task/step)."""
        if not item_id and (not type or not key):
            return _param_error("item_id or (type + key) is required")
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")

        try:
            store = _store_for(db_path)
            return store.get(
                item_id=item_id,
                type=type,
                key=key,
                task_id=task_id,
                step_id=step_id,
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_recall", e)

    @mcp.tool()
    def memory_search(
        query: str,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 20,
        sort_by: str = "updated",
        include_stale: bool = True,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full-text search (FTS5 if available, else LIKE)."""
        if not query or not str(query).strip():
            return _param_error("query cannot be empty")
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return _param_error("limit must be an integer")
        if limit_value < 1:
            return _param_error("limit must be >= 1")

        try:
            store = _store_for(db_path)
            return store.search(
                query=str(query),
                type=type,
                task_id=task_id,
                step_id=step_id,
                key=key,
                limit=limit_value,
                sort_by=sort_by,
                include_stale=include_stale,
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_search", e)

    @mcp.tool()
    def memory_list(
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        key: Optional[str] = None,
        tags: Optional[Any] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated",
        include_stale: bool = True,
        db_path: Optional[str] = None,
    ) -> Any:
        """List memories with optional filters."""
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        if tags is not None and not isinstance(tags, (str, list)):
            return _param_error("tags must be a string or list")
        try:
            limit_value = int(limit)
            offset_value = int(offset)
        except (TypeError, ValueError):
            return _param_error("limit/offset must be integers")
        if limit_value < 1:
            return _param_error("limit must be >= 1")
        if offset_value < 0:
            return _param_error("offset must be >= 0")

        try:
            store = _store_for(db_path)
            rows = store.list(
                type=type,
                task_id=task_id,
                step_id=step_id,
                key=key,
                tags=_normalize_tags(tags),
                limit=limit_value,
                offset=offset_value,
                sort_by=sort_by,
                include_stale=include_stale,
            )
            return {"ok": True, "items": rows, "limit": limit_value, "offset": offset_value}
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_list", e)

    @mcp.tool()
    def memory_recommend(
        context: Optional[str] = None,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        tags: Optional[Any] = None,
        key_prefix: Optional[str] = None,
        limit: int = 10,
        include_stale: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recommend related memories based on context/task."""
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return _param_error("limit must be an integer")
        if limit_value < 1:
            return _param_error("limit must be >= 1")

        try:
            store = _store_for(db_path)
            return store.recommend(
                context=context,
                type=type,
                task_id=task_id,
                tags=_normalize_tags(tags),
                key_prefix=key_prefix,
                limit=limit_value,
                include_stale=include_stale,
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_recommend", e)

    @mcp.tool()
    def memory_cleanup(
        type: Optional[str] = None,
        older_than_days: Optional[float] = None,
        stale_threshold: Optional[float] = None,
        limit: int = 1000,
        apply: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cleanup stale memories (dry-run by default)."""
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return _param_error("limit must be an integer")
        if limit_value < 1:
            return _param_error("limit must be >= 1")

        try:
            store = _store_for(db_path)
            return store.cleanup(
                type=type,
                older_than_days=older_than_days,
                stale_threshold=stale_threshold,
                limit=limit_value,
                apply=apply,
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_cleanup", e)

    @mcp.tool()
    def memory_delete(
        item_id: str,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a memory by id."""
        if not item_id:
            return _param_error("item_id is required")

        try:
            store = _store_for(db_path)
            with _WRITE_LOCK:
                return store.delete(item_id=item_id)
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_delete", e)

    @mcp.tool()
    def memory_export(
        output_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Any:
        """Export memories. If output_path is provided, write JSONL and return meta; else return items list."""
        if output_path is not None and not isinstance(output_path, str):
            return _param_error("output_path must be a string")
        if isinstance(output_path, str) and not output_path.strip():
            return _param_error("output_path cannot be empty")

        try:
            store = _store_for(db_path)
            rows = store.export_items()
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                return {"ok": True, "items": rows, "output": os.path.abspath(output_path), "count": len(rows)}
            return {"ok": True, "items": rows}
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_export", e)

    @mcp.tool()
    def memory_semantic_search(
        query: str,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.3,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Semantic vector search using embeddings.
        
        Uses local embedding model by default (BAAI/bge-small-zh-v1.5).
        Configure with environment variables:
        - MEMTOOL_EMBEDDING_PROVIDER: local, openai, or ollama
        - MEMTOOL_EMBEDDING_MODEL: Override default model
        - MEMTOOL_VECTOR_ENABLED: on, off, or auto
        """
        if not query or not str(query).strip():
            return _param_error("query cannot be empty")
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return _param_error("limit must be an integer")
        if limit_value < 1:
            return _param_error("limit must be >= 1")

        try:
            store = _store_for(db_path)
            return store.semantic_search(
                query=str(query),
                type=type,
                task_id=task_id,
                limit=limit_value,
                min_score=min_score,
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_semantic_search", e)

    @mcp.tool()
    def memory_hybrid_search(
        query: str,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 10,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Hybrid search combining FTS5 keyword search and vector semantic search.
        
        Best of both worlds: keyword matching + semantic understanding.
        """
        if not query or not str(query).strip():
            return _param_error("query cannot be empty")
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return _param_error("limit must be an integer")
        if limit_value < 1:
            return _param_error("limit must be >= 1")

        try:
            store = _store_for(db_path)
            return store.hybrid_search(
                query=str(query),
                type=type,
                task_id=task_id,
                limit=limit_value,
                fts_weight=fts_weight,
                vector_weight=vector_weight,
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_hybrid_search", e)

    @mcp.tool()
    def memory_vector_sync(
        force: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sync all memories to vector index.
        
        Args:
            force: If True, clear and rebuild entire index
        """
        try:
            store = _store_for(db_path)
            return store.vector_sync(force=force)
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_vector_sync", e)

    @mcp.tool()
    def memory_vector_status(
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get vector search status and statistics."""
        try:
            store = _store_for(db_path)
            return store.vector_status()
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_vector_status", e)

    @mcp.tool()
    def memory_assess_knowledge(
        topic: str,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Phase 2-1: 评估 Agent 对某个主题的记忆可信度（元认知）
        
        通过综合分析相关记忆的质量、数量、新鲜度和巩固程度，
        评估 Agent 对某个主题的掌握程度。
        
        Args:
            topic: 要评估的主题
            db_path: 数据库路径（可选）
        
        Returns:
            - confidence: "high" | "medium" | "low" | "none"
            - score: 元认知分数 (0.0 - 1.0)
            - message: 人类可读的评估
            - suggestions: 改进建议列表
            - evidence: 支撑证据列表（最多5条高质量记忆）
            - stats: 统计数据
        """
        return assess_knowledge(topic, db_path)

else:
    mcp = None


def main() -> None:
    if mcp is None:
        msg = "mcp package not available. Install with: pip install 'mcp[cli]'"
        if _IMPORT_ERROR is not None:
            msg += f"\nImport error: {_IMPORT_ERROR}"
        raise SystemExit(msg)

    _setup_logging()
    mcp.run()


if __name__ == "__main__":
    main()

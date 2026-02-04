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

from memtool.observability import compute_stats, health_check
from memtool.history import get_history
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
    """Phase 2.7: 评估 Agent 对某个主题的记忆可信度（元认知）

    独立函数版本，可供测试和其他模块使用

    增强版本：4 维度 breakdown + bottleneck 识别 + 针对性建议
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
                "breakdown": {
                    "quantity_score": 0.0,
                    "quality_score": 0.0,
                    "recency_score": 0.0,
                    "access_score": 0.0,
                },
                "issues": ["no_data"],
                "suggestions": [
                    "建议通过 web_search 或查阅文档获取信息",
                    "获取信息后用 memory_store 记录"
                ],
                "evidence": [],
                "stats": {
                    "total_memories": 0,
                    "avg_confidence": 0.0,
                    "avg_recency": 0.0,
                    "coverage": 0.0
                }
            }

        # Phase 2.7: 使用 metacognition 模块计算细粒度评估
        from memtool.metacognition import (
            compute_breakdown,
            identify_issues,
            generate_suggestions,
            find_bottleneck,
        )
        from memtool_rank import score_item
        import datetime as dt
        now = dt.datetime.now(tz=dt.timezone.utc)

        # 为每个 item 计算分数（如果还没有）
        for item in items:
            if "confidence_score" not in item:
                item.update(score_item(item, now=now))

        # 计算 4 维度 breakdown
        breakdown = compute_breakdown(items, now=now)

        # 识别问题
        issues = identify_issues(breakdown)

        # 生成针对性建议
        suggestions = generate_suggestions(topic, breakdown, issues, items)

        # 找出最薄弱维度
        bottleneck = find_bottleneck(breakdown)

        # 计算综合元认知分数（使用 breakdown 加权）
        meta_score = (
            breakdown["quality_score"] * 0.35 +
            breakdown["quantity_score"] * 0.25 +
            breakdown["recency_score"] * 0.25 +
            breakdown["access_score"] * 0.15
        )

        # 分级评估
        if meta_score >= 0.8:
            level = "high"
            message = f"我对「{topic}」很有把握，有 {len(items)} 条相关记忆"
        elif meta_score >= 0.5:
            level = "medium"
            # 根据 bottleneck 生成更具体的消息
            if bottleneck == "recency_score":
                message = f"我对「{topic}」有一些了解，但相关记忆较久远"
            elif bottleneck == "quantity_score":
                message = f"我对「{topic}」有一些了解，但记录较少"
            elif bottleneck == "quality_score":
                message = f"我对「{topic}」有一些了解，但置信度不高"
            else:
                message = f"我对「{topic}」有一些了解，但不完全确定"
        elif meta_score >= 0.2:
            level = "low"
            message = f"我对「{topic}」的记忆比较模糊"
        else:
            level = "very_low"
            message = f"我对「{topic}」几乎没有可靠记忆"

        # 提取高质量证据（只返回核心字段）
        evidence = sorted(items, key=lambda x: x.get("confidence_score", 0), reverse=True)[:5]

        # 计算统计数据
        avg_confidence = sum(item.get("confidence_score", 0.6) for item in items) / len(items)
        avg_recency = breakdown["recency_score"]
        total_access = sum(item.get("access_count", 0) for item in items)
        coverage = breakdown["quantity_score"]

        return {
            "ok": True,
            "confidence": level,
            "score": round(meta_score, 4),
            "message": message,
            "breakdown": breakdown,
            "issues": issues,
            "bottleneck": bottleneck,
            "suggestions": suggestions,
            "evidence": [
                {
                    "id": e.get("id"),
                    "key": e.get("key"),
                    "confidence_level": e.get("confidence_level"),
                    "access_count": e.get("access_count", 0),
                    "updated_at": e.get("updated_at"),
                    "snippet": (e.get("content") or "")[:100] + "..."
                }
                for e in evidence
            ],
            "stats": {
                "total_memories": len(items),
                "avg_confidence": round(avg_confidence, 3),
                "avg_recency": round(avg_recency, 3),
                "total_access": total_access,
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


def memory_contextual_search_impl(
    *,
    query: str,
    context_tags: Optional[Any] = None,
    emotional_filter: Optional[str] = None,
    urgency_min: Optional[int] = None,
    urgency_level: Optional[int] = None,
    limit: int = 10,
    type: Optional[str] = None,
    task_id: Optional[str] = None,
    include_stale: bool = True,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
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
    if emotional_filter is not None and emotional_filter not in {"positive", "negative", "neutral"}:
        return _param_error("emotional_filter must be one of: positive, negative, neutral")
    try:
        urgency_min_value = int(urgency_min) if urgency_min is not None else None
    except (TypeError, ValueError):
        return _param_error("urgency_min must be an integer")
    try:
        urgency_level_value = int(urgency_level) if urgency_level is not None else None
    except (TypeError, ValueError):
        return _param_error("urgency_level must be an integer")

    try:
        store = _store_for(db_path)
        return store.contextual_search(
            query=str(query),
            context_tags=_normalize_tags(context_tags),
            emotional_filter=emotional_filter,
            urgency_min=urgency_min_value,
            urgency_level=urgency_level_value,
            limit=limit_value,
            type=type,
            task_id=task_id,
            include_stale=include_stale,
        )
    except MemtoolError as e:
        return e.payload
    except Exception as e:
        return _unexpected_error("memory_contextual_search", e)


def memory_parse_context_impl(
    *,
    natural_query: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not natural_query or not str(natural_query).strip():
        return _param_error("natural_query cannot be empty")

    import re
    from memtool.context.extractor import ContextTags

    query_text = str(natural_query)
    context_tags: List[str] = []
    emotional_filter: Optional[str] = None
    urgency_level: Optional[int] = None

    time_patterns = {
        r"昨晚|昨天晚上|last\\s+night": ContextTags.TIME_LATE_NIGHT,
        r"今早|今天早上|this\\s+morning": ContextTags.TIME_EARLY_MORNING,
        r"周末|weekend": ContextTags.TIME_WEEKEND,
        r"上班时间|工作时间|work\\s+hours?": ContextTags.TIME_WORK_HOURS,
        r"今晚|今天晚上|this\\s+evening": ContextTags.TIME_EVENING,
    }

    emotion_patterns = [
        (r"失败|没搞定|问题|failed|broken", "negative"),
        (r"成功|解决了|搞定|succeeded?|fixed", "positive"),
    ]

    urgency_patterns = {
        r"P0|紧急|阻塞|马上|critical|blocking": 3,
        r"P1|urgent|asap|重要|优先": 2,
        r"P2|soon|尽快": 1,
    }

    task_patterns = {
        r"调试|debug": ContextTags.TASK_DEBUGGING,
        r"测试|test": ContextTags.TASK_TESTING,
        r"部署|deploy": ContextTags.TASK_DEPLOYMENT,
        r"重构|refactor": ContextTags.TASK_REFACTOR,
        r"接口|api|endpoint": ContextTags.TASK_API_DESIGN,
        r"数据库|表结构|schema|migration": ContextTags.TASK_DATA_MODEL,
    }

    for pattern, tag in time_patterns.items():
        if re.search(pattern, query_text, re.IGNORECASE):
            context_tags.append(tag)
            query_text = re.sub(pattern, "", query_text, flags=re.IGNORECASE)

    for pattern, emotion in emotion_patterns:
        if re.search(pattern, query_text, re.IGNORECASE):
            emotional_filter = emotion
            query_text = re.sub(pattern, "", query_text, flags=re.IGNORECASE)
            break

    for pattern, level in urgency_patterns.items():
        if re.search(pattern, query_text, re.IGNORECASE):
            urgency_level = level
            query_text = re.sub(pattern, "", query_text, flags=re.IGNORECASE)
            break

    for pattern, tag in task_patterns.items():
        if re.search(pattern, query_text, re.IGNORECASE):
            context_tags.append(tag)

    query_text = re.sub(r"(那个|的|这个|上次|之前|that|the|this)", "", query_text, flags=re.IGNORECASE)
    query_text = query_text.strip()

    return {
        "ok": True,
        "original_query": natural_query,
        "parsed": {
            "query": query_text or natural_query,
            "context_tags": context_tags,
            "emotional_filter": emotional_filter,
            "urgency_level": urgency_level,
        },
        "suggested_call": {
            "tool": "memory_contextual_search",
            "args": {
                "query": query_text or natural_query,
                "context_tags": context_tags if context_tags else None,
                "emotional_filter": emotional_filter,
                "urgency_level": urgency_level,
            },
        },
    }


if FastMCP is not None:
    mcp = FastMCP("memtool")

    @mcp.tool()
    def memory_store(
        type: str,
        key: str,
        content: Any,
        tags: Optional[Any] = None,
        confidence_level: str = "medium",
        verified_by: Optional[str] = None,
        item_id: Optional[str] = None,
        db_path: Optional[str] = None,
        # 废弃参数（保留兼容性，但忽略）
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        source: Optional[str] = None,
        weight: float = 1.0,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store or update memory (upsert by id or logical key).
        
        Args:
            type: Memory type (project/feature/run)
            key: Unique identifier
            content: Memory content
            tags: Optional tags
            confidence_level: Confidence level (low/medium/high)
            verified_by: Optional verification source
            item_id: Optional id for update
        
        Deprecated (ignored):
            task_id, step_id, source, weight, session_id
        """
        if not type or not key:
            return _param_error("type and key are required")
        if type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        if content is None:
            return _param_error("content is required")
        if tags is not None and not isinstance(tags, (str, list)):
            return _param_error("tags must be a string or list")
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
                    tags=_normalize_tags(tags),
                    confidence_level=confidence_level,
                    verified_by=verified_by,
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
        db_path: Optional[str] = None,
        # 废弃参数（保留兼容性，但忽略）
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recall a memory by id or logical key (type+key).
        
        Args:
            item_id: Get by id
            type: Memory type
            key: Unique identifier
        
        Deprecated (ignored):
            task_id, step_id
        """
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
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_recall", e)

    @mcp.tool()
    def memory_search(
        query: str,
        type: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 20,
        sort_by: str = "updated",
        include_stale: bool = True,
        db_path: Optional[str] = None,
        # 废弃参数（保留兼容性，但忽略）
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full-text search (FTS5 if available, else LIKE).
        
        Args:
            query: Search query
            type: Memory type filter
            key: Key filter
            limit: Max results
        
        Deprecated (ignored):
            task_id, step_id
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
            return store.search(
                query=str(query),
                type=type,
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
    def memory_contextual_search(
        query: str,
        context_tags: Optional[Any] = None,
        emotional_filter: Optional[str] = None,
        urgency_min: Optional[int] = None,
        urgency_level: Optional[int] = None,
        limit: int = 10,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        include_stale: bool = True,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """情境检索：基于上下文标签/情绪/紧急度过滤"""
        return memory_contextual_search_impl(
            query=query,
            context_tags=context_tags,
            emotional_filter=emotional_filter,
            urgency_min=urgency_min,
            urgency_level=urgency_level,
            limit=limit,
            type=type,
            task_id=task_id,
            include_stale=include_stale,
            db_path=db_path,
        )

    @mcp.tool()
    def memory_parse_context(
        natural_query: str,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """解析自然语言查询，提取情境条件"""
        return memory_parse_context_impl(natural_query=natural_query, db_path=db_path)

    @mcp.tool()
    def memory_list(
        type: Optional[str] = None,
        key: Optional[str] = None,
        tags: Optional[Any] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated",
        include_stale: bool = True,
        db_path: Optional[str] = None,
        # 废弃参数（保留兼容性，但忽略）
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Any:
        """List memories with optional filters.
        
        Args:
            type: Memory type filter
            key: Key filter
            tags: Tags filter
            limit, offset: Pagination
        
        Deprecated (ignored):
            task_id, step_id
        """
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
        tags: Optional[Any] = None,
        key_prefix: Optional[str] = None,
        limit: int = 10,
        include_stale: bool = False,
        db_path: Optional[str] = None,
        # 废弃参数（保留兼容性，但忽略）
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recommend related memories based on context.
        
        Args:
            context: Current context or question
            type: Memory type filter
            tags: Tags filter
            key_prefix: Key prefix filter
            limit: Max results
        
        Deprecated (ignored):
            task_id
        """
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
    def memory_stats(
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get basic memory statistics."""
        try:
            store = _store_for(db_path)
            stats = compute_stats(store)
            return {"ok": True, **stats}
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_stats", e)

    @mcp.tool()
    def memory_health_check(
        thresholds: Optional[Dict[str, float]] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check memory health with optional thresholds."""
        if thresholds is not None and not isinstance(thresholds, dict):
            return _param_error("thresholds must be a dict")
        try:
            store = _store_for(db_path)
            return health_check(store, thresholds=thresholds)
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_health_check", e)

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

    @mcp.tool()
    def memory_history(
        item_id: str,
        limit: int = 10,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Phase 2.6: 查看记忆的版本历史
        
        Args:
            item_id: 记忆的 ID
            limit: 最多返回多少条历史记录（默认 10）
            db_path: 可选的数据库路径
            
        Returns:
            包含版本历史的字典
            
        Example:
            memory_history(item_id="abc123")
            → {"ok": True, "history": [{"version": 2, "content": "..."}]}
        """
        if not item_id or not str(item_id).strip():
            return _param_error("item_id cannot be empty")
        
        try:
            store = _store_for(db_path)
            return get_history(store, str(item_id).strip(), limit=limit)
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_history", e)

    @mcp.tool()
    def memory_suggest_merge(
        type: Optional[str] = None,
        threshold: float = 0.85,
        limit: int = 10,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Phase 2.6: 找出可能需要合并的相似记忆
        
        扫描记忆库，找出高相似度的记忆对，建议合并以减少冗余。
        
        Args:
            type: 可选，仅在该 type 中搜索
            threshold: 相似度阈值（默认 0.85，即 85%）
            limit: 最多返回多少组建议（默认 10）
            db_path: 可选的数据库路径
            
        Returns:
            合并建议列表
            
        Example:
            memory_suggest_merge(type="project", threshold=0.9)
        """
        from memtool.merge import suggest_merges
        
        if type is not None and type not in _VALID_TYPES:
            return _param_error("type must be one of: project, feature, run")
        
        try:
            store = _store_for(db_path)
            return suggest_merges(
                store,
                type=type,
                threshold=threshold,
                limit=limit
            )
        except MemtoolError as e:
            return e.payload
        except Exception as e:
            return _unexpected_error("memory_suggest_merge", e)

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

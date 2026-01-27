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
                limit=limit_value + offset_value,
                sort_by=sort_by,
                include_stale=include_stale,
            )
            items = rows[offset_value:] if offset_value else rows
            return {"ok": True, "items": items, "limit": limit_value, "offset": offset_value}
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

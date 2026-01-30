#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP end-to-end tests for memtool.
"""
from __future__ import annotations

import anyio
import json
import os
import sqlite3
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@dataclass
class CaseResult:
    name: str
    ok: bool
    seconds: float
    details: str = ""


def _now_ts() -> float:
    return time.perf_counter()


def _extract_payload(result: Any) -> Dict[str, Any]:
    if getattr(result, "isError", False):
        # best effort extraction
        if getattr(result, "structuredContent", None):
            payload = result.structuredContent  # type: ignore[assignment]
            if isinstance(payload, dict) and "result" in payload and "ok" not in payload:
                return payload.get("result", {})
            return payload  # type: ignore[return-value]
        if getattr(result, "content", None):
            c0 = result.content[0]
            if hasattr(c0, "text"):
                text = c0.text
                if isinstance(text, str) and text.startswith("Error executing tool"):
                    return {"ok": False, "error": "TOOL_VALIDATION", "message": text}
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
        return {"error": "tool_error"}

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and "result" in structured and "ok" not in structured:
            return structured.get("result", {})
        return structured

    content = getattr(result, "content", None) or []
    if content:
        c0 = content[0]
        if hasattr(c0, "text"):
            text = c0.text
            if isinstance(text, str) and text.startswith("Error executing tool"):
                return {"ok": False, "error": "TOOL_VALIDATION", "message": text}
            try:
                return json.loads(text)
            except Exception:
                return {"raw": text}
    return {}


def _check(cond: bool, msg: str, errors: List[str]) -> None:
    if not cond:
        errors.append(msg)


def _server_params(db_path: str, root: Path) -> StdioServerParameters:
    venv_bin = root / ".venv" / "bin"
    cmd = venv_bin / "memtool-mcp"
    if not cmd.exists():
        # fallback to module execution
        cmd = venv_bin / "python"
        return StdioServerParameters(
            command=str(cmd),
            args=["-m", "mcp_server"],
            env={"MEMTOOL_DB": db_path, "LOG_LEVEL": "INFO"},
            cwd=str(root),
        )
    return StdioServerParameters(
        command=str(cmd),
        args=[],
        env={"MEMTOOL_DB": db_path, "LOG_LEVEL": "INFO"},
        cwd=str(root),
    )


@asynccontextmanager
async def _with_session(db_path: str, root: Path):
    params = _server_params(db_path, root)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def _call(session: ClientSession, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        result = await session.call_tool(name, args or {})
        return _extract_payload(result)
    except Exception as exc:  # avoid raising inside session context
        return {"ok": False, "error": "EXCEPTION", "message": str(exc)}


async def run_tests() -> List[CaseResult]:
    results: List[CaseResult] = []
    root = Path(__file__).resolve().parents[1]

    def record(name: str, ok: bool, start: float, details: str = "") -> None:
        results.append(CaseResult(name=name, ok=ok, seconds=round(_now_ts() - start, 4), details=details))

    async def run_case(name: str, fn):
        start = _now_ts()
        try:
            await fn()
            record(name, True, start)
        except Exception as exc:
            record(name, False, start, details=str(exc))

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mcp_test.db")

        async def case_list_tools():
            errors: List[str] = []
            async with _with_session(db_path, root) as session:
                try:
                    tools = await session.list_tools()
                except Exception as exc:
                    tools = None
                    errors.append(f"list_tools exception: {exc}")
                if tools is not None:
                    names = {t.name for t in tools.tools}
                    expected = {
                        "memory_store",
                        "memory_recall",
                        "memory_search",
                        "memory_list",
                        "memory_recommend",
                        "memory_cleanup",
                        "memory_delete",
                        "memory_export",
                    }
                    _check(expected.issubset(names), f"missing tools: {expected - names}", errors)
            if errors:
                raise AssertionError("; ".join(errors))

        async def case_store_recall_update():
            errors: List[str] = []
            async with _with_session(db_path, root) as session:
                res1 = await _call(session, "memory_store", {
                    "type": "project",
                    "key": "spec:alpha",
                    "content": "alpha content",
                    "tags": "a,b",
                })
                _check(res1.get("ok") is True, f"store failed: {res1}", errors)
                item_id = res1.get("id")
                _check(bool(item_id), "missing id", errors)

                res2 = await _call(session, "memory_recall", {"item_id": item_id})
                _check(res2.get("id") == item_id, "recall by id mismatch", errors)

                # logical key recall
                res3 = await _call(session, "memory_recall", {"type": "project", "key": "spec:alpha"})
                _check(res3.get("id") == item_id, "recall by key mismatch", errors)

                # update same logical key -> version increment
                res4 = await _call(session, "memory_store", {
                    "type": "project",
                    "key": "spec:alpha",
                    "content": {"v": 2},
                })
                _check(res4.get("ok") is True, f"update failed: {res4}", errors)
                _check(res4.get("id") == item_id, "upsert should keep id", errors)
                _check(res4.get("version", 0) >= 2, "version should increment", errors)
            if errors:
                raise AssertionError("; ".join(errors))

        async def case_list_search_tags():
            errors: List[str] = []
            async with _with_session(db_path, root) as session:
                await _call(session, "memory_store", {
                    "type": "feature",
                    "key": "search:beta",
                    "content": "beta feature with redis timeout",
                    "tags": ["search", "redis"],
                })
                await _call(session, "memory_store", {
                    "type": "feature",
                    "key": "search:gamma",
                    "content": "gamma feature with sqlite lock",
                    "tags": ["search"],
                })

                listed = await _call(session, "memory_list", {"type": "feature", "limit": 10})
                _check(listed.get("ok") is True, f"list failed: {listed}", errors)
                _check(len(listed.get("items", [])) >= 2, "list should include >=2 items", errors)

                tagged = await _call(session, "memory_list", {"tags": "redis"})
                _check(tagged.get("ok") is True, f"list tags failed: {tagged}", errors)
                _check(any(i.get("key") == "search:beta" for i in tagged.get("items", [])), "tag filter missing", errors)

                searched = await _call(session, "memory_search", {"query": "redis timeout", "type": "feature"})
                _check(searched.get("ok") is True, f"search failed: {searched}", errors)
                _check(any(i.get("key") == "search:beta" for i in searched.get("items", [])), "search miss", errors)
            if errors:
                raise AssertionError("; ".join(errors))

        async def case_recommend_export_delete():
            errors: List[str] = []
            async with _with_session(db_path, root) as session:
                await _call(session, "memory_store", {
                    "type": "run",
                    "key": "debug:redis_timeout",
                    "content": "Redis connection timeout, add retry",
                })
                await _call(session, "memory_store", {
                    "type": "run",
                    "key": "debug:db_lock",
                    "content": "SQLite database is locked, increase retry",
                })

                rec = await _call(session, "memory_recommend", {"context": "redis timeout"})
                _check(rec.get("ok") is True, f"recommend failed: {rec}", errors)
                _check(bool(rec.get("items")), "recommend empty", errors)

                export = await _call(session, "memory_export", {})
                _check(export.get("ok") is True, f"export failed: {export}", errors)
                items = export.get("items", [])
                _check(len(items) >= 1, "export items empty", errors)

                out_path = os.path.join(os.path.dirname(db_path), "export.jsonl")
                export2 = await _call(session, "memory_export", {"output_path": out_path})
                _check(export2.get("ok") is True, f"export output failed: {export2}", errors)
                _check(Path(out_path).exists(), "export file missing", errors)

                # delete one item
                if items:
                    victim_id = items[0]["id"]
                    deleted = await _call(session, "memory_delete", {"item_id": victim_id})
                    _check(deleted.get("ok") is True, f"delete failed: {deleted}", errors)
                    _check(deleted.get("deleted") == 1, f"delete count unexpected: {deleted}", errors)
                    missing = await _call(session, "memory_recall", {"item_id": victim_id})
                    _check(missing.get("ok") is False and missing.get("error") == "NOT_FOUND", "deleted item should be missing", errors)
            if errors:
                raise AssertionError("; ".join(errors))

        async def case_cleanup_and_stale_filter():
            errors: List[str] = []
            async with _with_session(db_path, root) as session:
                old_item = await _call(session, "memory_store", {
                    "type": "run",
                    "key": "old:item",
                    "content": "old",
                })
                new_item = await _call(session, "memory_store", {
                    "type": "run",
                    "key": "new:item",
                    "content": "new",
                })
                old_id = old_item.get("id")
                new_id = new_item.get("id")

                # backdate old item to make stale
                if old_id:
                    conn = sqlite3.connect(db_path)
                    conn.execute(
                        "UPDATE memory_items SET updated_at = datetime('now', '-60 days'), created_at = datetime('now', '-60 days') WHERE id = ?",
                        (old_id,),
                    )
                    conn.commit()
                    conn.close()

                listed = await _call(session, "memory_list", {"type": "run", "include_stale": False, "limit": 50})
                _check(listed.get("ok") is True, f"list stale failed: {listed}", errors)
                ids = {i.get("id") for i in listed.get("items", [])}
                _check(old_id not in ids and new_id in ids, "include_stale filter incorrect", errors)

                dry = await _call(session, "memory_cleanup", {"type": "run", "older_than_days": 30, "apply": False})
                _check(dry.get("ok") is True, f"cleanup dry failed: {dry}", errors)
                _check(bool(dry.get("candidates")), "cleanup dry should find candidates", errors)

                applied = await _call(session, "memory_cleanup", {"type": "run", "older_than_days": 30, "apply": True})
                _check(applied.get("ok") is True, f"cleanup apply failed: {applied}", errors)
                _check(applied.get("deleted", 0) >= 1, "cleanup delete count unexpected", errors)
            if errors:
                raise AssertionError("; ".join(errors))

        async def case_param_errors():
            errors: List[str] = []
            async with _with_session(db_path, root) as session:
                bad1 = await _call(session, "memory_store", {"type": "project", "key": ""})
                _check(bad1.get("ok") is False and bad1.get("error") in {"PARAM_ERROR", "TOOL_VALIDATION"}, "missing content/type/key should error", errors)

                bad2 = await _call(session, "memory_store", {"type": "bad", "key": "x", "content": "y"})
                _check(bad2.get("ok") is False and bad2.get("error") == "PARAM_ERROR", "invalid type should error", errors)

                bad3 = await _call(session, "memory_store", {"type": "run", "key": "x", "content": "y", "tags": 123})
                _check(bad3.get("ok") is False and bad3.get("error") == "PARAM_ERROR", "invalid tags should error", errors)

                bad4 = await _call(session, "memory_store", {"type": "run", "key": "x", "content": "y", "weight": "oops"})
                _check(bad4.get("ok") is False and bad4.get("error") in {"PARAM_ERROR", "TOOL_VALIDATION"}, "invalid weight should error", errors)

                bad5 = await _call(session, "memory_recall", {})
                _check(bad5.get("ok") is False and bad5.get("error") == "PARAM_ERROR", "recall missing params should error", errors)

                bad6 = await _call(session, "memory_search", {"query": ""})
                _check(bad6.get("ok") is False and bad6.get("error") == "PARAM_ERROR", "empty query should error", errors)

                bad7 = await _call(session, "memory_list", {"limit": 0})
                _check(bad7.get("ok") is False and bad7.get("error") == "PARAM_ERROR", "limit 0 should error", errors)

                bad8 = await _call(session, "memory_list", {"offset": -1})
                _check(bad8.get("ok") is False and bad8.get("error") == "PARAM_ERROR", "offset -1 should error", errors)
            if errors:
                raise AssertionError("; ".join(errors))

        async def case_concurrency():
            # concurrent writes via multiple processes on same db
            params = _server_params(db_path, root)
            errors: List[str] = []
            lock = anyio.Lock()

            async def worker(i: int) -> None:
                try:
                    async with stdio_client(params) as (r, w):
                        async with ClientSession(r, w) as session:
                            await session.initialize()
                            res = await _call(session, "memory_store", {
                                "type": "run",
                                "key": f"concurrency:item:{i}",
                                "content": f"payload {i}",
                            })
                            if res.get("ok") is not True:
                                async with lock:
                                    errors.append(f"concurrent store failed: {res}")
                except Exception as exc:
                    async with lock:
                        errors.append(f"concurrent worker exception: {exc}")

            async with anyio.create_task_group() as tg:
                for i in range(5):
                    tg.start_soon(worker, i)
            if errors:
                raise AssertionError("; ".join(errors))

        await run_case("list_tools", case_list_tools)
        await run_case("store_recall_update", case_store_recall_update)
        await run_case("list_search_tags", case_list_search_tags)
        await run_case("recommend_export_delete", case_recommend_export_delete)
        await run_case("cleanup_and_stale_filter", case_cleanup_and_stale_filter)
        await run_case("param_errors", case_param_errors)
        await run_case("concurrency", case_concurrency)

    return results


def write_report(results: List[CaseResult], path: Path) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.ok)
    failed = total - passed

    lines = []
    lines.append("# MCP Test Report")
    lines.append("")
    lines.append(f"- Total: {total}")
    lines.append(f"- Passed: {passed}")
    lines.append(f"- Failed: {failed}")
    lines.append("")
    lines.append("## Case Results")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        lines.append(f"- {r.name}: {status} ({r.seconds}s)")
        if r.details:
            lines.append(f"  - details: {r.details}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = anyio.run(run_tests)
    report_path = Path(__file__).resolve().parents[1] / "MCP_TEST_REPORT.md"
    write_report(results, report_path)

    failed = [r for r in results if not r.ok]
    if failed:
        print("FAILED")
        for r in failed:
            print(f"- {r.name}: {r.details}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

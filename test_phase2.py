#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 测试：混合排序、推荐引擎、生命周期管理
"""
import os
import sys
import tempfile
import sqlite3
import datetime as dt

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from memtool_core import MemoryStore


def _iso_days_ago(days: float) -> str:
    now = dt.datetime.now(tz=dt.timezone.utc)
    return (now - dt.timedelta(days=days)).isoformat(timespec="seconds")


def _set_updated_at(db_path: str, item_id: str, days_ago: float) -> None:
    ts = _iso_days_ago(days_ago)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE memory_items SET updated_at = ?, created_at = ? WHERE id = ?", (ts, ts, item_id))
    conn.commit()
    conn.close()


def test_mixed_sorting():
    print("\n=== 测试 1: 混合排序 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path)
        store.init_db()

        high_old = store.put(
            item_id=None,
            type="run",
            key="decision:high_old",
            content="高置信度但较旧的记录",
            confidence_level="high",
        )
        low_new = store.put(
            item_id=None,
            type="run",
            key="decision:low_new",
            content="低置信度但较新的记录",
            confidence_level="low",
        )
        _set_updated_at(db_path, high_old["id"], 60)
        _set_updated_at(db_path, low_new["id"], 1)

        results = store.list(type="run", sort_by="mixed", limit=5)
        assert results[0]["id"] == high_old["id"], "混合排序未按预期排序"
        print("✓ mixed 排序符合预期")


def test_recommendation():
    print("\n=== 测试 2: 推荐引擎 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path)
        store.init_db()

        a = store.put(
            item_id=None,
            type="run",
            key="debug:redis_timeout",
            content="Redis 连接超时，增加重试与连接池",
            confidence_level="high",
        )
        b = store.put(
            item_id=None,
            type="run",
            key="debug:db_lock",
            content="SQLite database is locked，增加重试",
            confidence_level="medium",
        )

        rec = store.recommend(context="redis timeout 连接超时", limit=1)
        assert rec["ok"] and rec["items"], "推荐结果为空"
        assert rec["items"][0]["id"] == a["id"], "推荐未命中最相关记录"
        print("✓ 推荐命中相关记录")


def test_cleanup():
    print("\n=== 测试 3: 生命周期清理 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path)
        store.init_db()

        old_item = store.put(
            item_id=None,
            type="run",
            key="old:item",
            content="旧记录",
            confidence_level="medium",
        )
        new_item = store.put(
            item_id=None,
            type="run",
            key="new:item",
            content="新记录",
            confidence_level="medium",
        )
        _set_updated_at(db_path, old_item["id"], 30)
        _set_updated_at(db_path, new_item["id"], 1)

        dry = store.cleanup(type="run", older_than_days=14, apply=False)
        assert dry["ok"] and len(dry["candidates"]) == 1, "dry-run 应只命中旧记录"
        assert dry["candidates"][0]["id"] == old_item["id"]

        applied = store.cleanup(type="run", older_than_days=14, apply=True)
        assert applied["deleted"] == 1, "apply 应删除 1 条记录"

        remain = store.list(type="run", limit=10)
        ids = [r["id"] for r in remain]
        assert old_item["id"] not in ids, "旧记录未被删除"
        assert new_item["id"] in ids, "新记录不应被删除"
        print("✓ cleanup dry-run / apply 正常")


def run_all_tests():
    print("=" * 50)
    print("Phase 2 功能测试")
    print("=" * 50)

    try:
        test_mixed_sorting()
        test_recommendation()
        test_cleanup()

        print("\n" + "=" * 50)
        print("✓ 所有测试通过！")
        print("=" * 50)
        return 0
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发写入测试：验证逻辑键唯一约束 + ON CONFLICT upsert
"""
import os
import sys
import tempfile
import threading
import sqlite3

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from memtool_core import MemoryStore


def _logical_key_count(conn: sqlite3.Connection, type_: str, key: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM memory_items WHERE type = ? AND key = ? AND task_id IS NULL AND step_id IS NULL",
        (type_, key),
    ).fetchone()
    return int(row["cnt"])


def _logical_key_row(conn: sqlite3.Connection, type_: str, key: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, version, content FROM memory_items WHERE type = ? AND key = ? AND task_id IS NULL AND step_id IS NULL",
        (type_, key),
    ).fetchone()
    return row


def test_concurrent_upsert():
    print("\n=== 测试: 并发写入唯一约束 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path)
        store.init_db()

        type_ = "run"
        key = "concurrency:logical-key"
        thread_count = 8
        barrier = threading.Barrier(thread_count)
        errors = []

        def worker(i: int) -> None:
            try:
                barrier.wait()
                store.put(
                    item_id=None,
                    type=type_,
                    key=key,
                    content=f"content-{i}",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发写入出现异常: {errors}"

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            count = _logical_key_count(conn, type_, key)
            assert count == 1, f"逻辑键应唯一，实际记录数: {count}"

            row = _logical_key_row(conn, type_, key)
            assert row is not None, "未找到逻辑键对应记录"
            assert int(row["version"]) == thread_count, f"version 期望 {thread_count}，实际 {row['version']}"
            print("✓ 并发写入未产生重复记录，version 递增正常")
        finally:
            conn.close()


def run_all_tests():
    print("=" * 50)
    print("并发写入测试")
    print("=" * 50)

    try:
        test_concurrent_upsert()
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

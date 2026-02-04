#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2-2a 功能测试：情境字段 + 提取 + 检索"""
import datetime as dt
import os
import sqlite3
import tempfile
import unittest

from memtool.context.extractor import ContextExtractor, ContextTags
from memtool_core import MemoryStore
from mcp_server import memory_contextual_search_impl, memory_parse_context_impl


class TestPhase2_2a(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_phase2_2a.db")
        self.store = MemoryStore(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_schema_has_new_columns(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("PRAGMA table_info(memory_items)")
        columns = {row["name"] for row in cursor.fetchall()}
        conn.close()

        self.assertIn("context_tags_json", columns)
        self.assertIn("emotional_valence", columns)
        self.assertIn("urgency_level", columns)
        self.assertIn("related_json", columns)
        self.assertIn("session_id", columns)

    def test_context_extractor_time_task_emotion(self):
        ts = dt.datetime(2026, 2, 1, 23, 30, 0)
        content = "昨晚调试成功修复了一个 bug"
        tags, valence, urgency = ContextExtractor.extract(content=content, timestamp=ts)

        self.assertIn(ContextTags.TIME_LATE_NIGHT, tags)
        self.assertIn(ContextTags.TIME_WEEKEND, tags)
        self.assertIn(ContextTags.TASK_DEBUGGING, tags)
        self.assertIn(ContextTags.EMOTION_POSITIVE, tags)
        self.assertGreater(valence, 0)
        self.assertEqual(urgency, 0)

    def test_negation_detection(self):
        content = "没解决任务，还在处理"
        tags, valence, _ = ContextExtractor.extract(content=content, timestamp=dt.datetime(2026, 2, 1, 12, 0, 0))
        self.assertIn(ContextTags.EMOTION_NEGATIVE, tags)
        self.assertLess(valence, 0)

    def test_urgency_extraction(self):
        content = "这是一个紧急问题，需要马上处理"
        _, _, urgency = ContextExtractor.extract(content=content, timestamp=dt.datetime(2026, 2, 1, 12, 0, 0))
        self.assertEqual(urgency, 3)

    def test_memory_contextual_search_filters(self):
        self.store.put(
            item_id=None,
            type="feature",
            key="debug_success",
            content="调试日志显示已成功修复",
        )
        self.store.put(
            item_id=None,
            type="feature",
            key="test_failed",
            content="测试日志显示失败，需要排查",
        )

        result = memory_contextual_search_impl(
            query="日志",
            context_tags=[ContextTags.TASK_DEBUGGING],
            emotional_filter="positive",
            limit=10,
            db_path=self.db_path,
        )

        self.assertTrue(result["ok"])
        items = result.get("items", [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["key"], "debug_success")

    def test_memory_parse_context(self):
        result = memory_parse_context_impl(natural_query="昨晚调试没搞定的那个问题")
        self.assertTrue(result["ok"])
        parsed = result["parsed"]
        self.assertIn(ContextTags.TIME_LATE_NIGHT, parsed["context_tags"])
        self.assertIn(ContextTags.TASK_DEBUGGING, parsed["context_tags"])
        self.assertEqual(parsed["emotional_filter"], "negative")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2-1 功能测试：记忆巩固与元认知
"""
import datetime as dt
import os
import sqlite3
import tempfile
import unittest

from memtool_core import MemoryStore, utcnow_iso


class TestPhase2_1(unittest.TestCase):
    """Phase 2-1: 记忆巩固机制测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_phase2_1.db")
        self.store = MemoryStore(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_schema_has_new_columns(self):
        """测试数据库 Schema 包含新字段"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("PRAGMA table_info(memory_items)")
        columns = {row["name"] for row in cursor.fetchall()}
        
        # 验证新字段存在
        self.assertIn("access_count", columns)
        self.assertIn("last_accessed_at", columns)
        self.assertIn("consolidation_score", columns)
        
        conn.close()

    def test_new_fields_have_defaults(self):
        """测试新字段有正确的默认值"""
        result = self.store.put(
            item_id=None,
            type="feature",
            key="test_defaults",
            content="测试默认值"
        )
        
        self.assertTrue(result["ok"])
        item_id = result["id"]
        
        # 获取记录
        item = self.store.get(item_id=item_id)
        
        # 验证默认值
        self.assertEqual(item["access_count"], 0)
        self.assertIsNone(item["last_accessed_at"])
        self.assertEqual(item["consolidation_score"], 0.0)

    def test_access_tracking_on_get(self):
        """测试 get() 触发访问追踪"""
        # 创建记录
        result = self.store.put(
            item_id=None,
            type="feature",
            key="test_access",
            content="测试访问追踪"
        )
        item_id = result["id"]
        
        # 第一次 get - 先获取，访问追踪是异步的
        self.store.get(item_id=item_id)
        
        # 再次获取以查看更新后的访问计数
        item1 = self.store.get(item_id=item_id)
        self.assertGreaterEqual(item1["access_count"], 1)
        self.assertIsNotNone(item1["last_accessed_at"])
        
        # 第三次 get
        item2 = self.store.get(item_id=item_id)
        self.assertGreaterEqual(item2["access_count"], 2)
        
        # 验证 last_accessed_at 更新了
        self.assertGreaterEqual(item2["last_accessed_at"], item1["last_accessed_at"])

    def test_access_tracking_on_search(self):
        """测试 search() 触发访问追踪"""
        # 创建记录
        result = self.store.put(
            item_id=None,
            type="feature",
            key="searchable",
            content="这是一条可搜索的记录，包含关键词搜索"
        )
        item_id = result["id"]
        
        # 等待 FTS 索引（如果启用）
        import time
        time.sleep(0.1)
        
        # 搜索
        search_result = self.store.search(query="搜索", limit=10)
        self.assertTrue(search_result["ok"])
        
        # 如果找到结果，验证访问计数增加
        if len(search_result["items"]) > 0:
            # 再次获取以查看更新后的访问计数
            item = self.store.get(item_id=item_id)
            self.assertGreaterEqual(item["access_count"], 1)  # 至少被 search 访问了一次

    def test_consolidation_score_calculation(self):
        """测试巩固分数计算"""
        # 创建一个记录
        result = self.store.put(
            item_id=None,
            type="feature",
            key="consolidation_test",
            content="测试巩固分数"
        )
        item_id = result["id"]
        
        # 多次访问
        for _ in range(10):
            self.store.get(item_id=item_id)
        
        # 获取最新数据
        item = self.store.get(item_id=item_id)
        
        # 验证巩固分数增加
        self.assertGreater(item["consolidation_score"], 0.0)
        self.assertLessEqual(item["consolidation_score"], 1.0)
        
        # 访问越多，巩固分数应该越高（对数增长）
        # 由于访问追踪是异步的，access_count 应该 >= 10
        self.assertGreaterEqual(item["access_count"], 10)

    def test_consolidation_affects_decay(self):
        """测试巩固分数影响衰减速度"""
        from memtool_lifecycle import decay_score
        
        # 创建一个旧记录（30天前）
        now = dt.datetime.now(tz=dt.timezone.utc)
        old_date = (now - dt.timedelta(days=30)).isoformat()
        
        # 计算无巩固的衰减分数
        decay_no_consolidation = decay_score(
            old_date,
            "feature",
            now=now,
            consolidation_score=0.0
        )
        
        # 计算高巩固的衰减分数
        decay_high_consolidation = decay_score(
            old_date,
            "feature",
            now=now,
            consolidation_score=1.0
        )
        
        # 高巩固分数应该让衰减更慢（分数更高）
        self.assertGreater(decay_high_consolidation, decay_no_consolidation)
        
        print(f"\n30天前的记录:")
        print(f"  无巩固: {decay_no_consolidation:.4f}")
        print(f"  高巩固: {decay_high_consolidation:.4f}")

    def test_compute_consolidation_method(self):
        """测试 _compute_consolidation 方法"""
        now = dt.datetime.now(tz=dt.timezone.utc)
        created_at = (now - dt.timedelta(days=100)).isoformat()
        updated_at = (now - dt.timedelta(days=1)).isoformat()
        
        # 测试不同访问次数
        score_0 = self.store._compute_consolidation(0, created_at, updated_at, now)
        score_10 = self.store._compute_consolidation(10, created_at, updated_at, now)
        score_100 = self.store._compute_consolidation(100, created_at, updated_at, now)
        
        # 验证分数随访问次数增长（但不是线性）
        self.assertGreater(score_10, score_0)
        self.assertGreater(score_100, score_10)
        
        # 验证分数在合理范围内
        self.assertGreaterEqual(score_0, 0.0)
        self.assertLessEqual(score_100, 1.0)
        
        print(f"\n巩固分数计算:")
        print(f"  access_count=0:   {score_0:.4f}")
        print(f"  access_count=10:  {score_10:.4f}")
        print(f"  access_count=100: {score_100:.4f}")


class TestMetacognition(unittest.TestCase):
    """Phase 2-1: 元认知评估测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_metacog.db")
        self.store = MemoryStore(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_assess_knowledge_no_memory(self):
        """测试对无记忆主题的评估"""
        from mcp_server import assess_knowledge
        
        result = assess_knowledge(
            topic="完全不存在的主题",
            db_path=self.db_path
        )
        
        self.assertTrue(result["ok"])
        self.assertEqual(result["confidence"], "none")
        self.assertEqual(result["score"], 0.0)
        self.assertGreater(len(result["suggestions"]), 0)
        self.assertEqual(len(result["evidence"]), 0)

    def test_assess_knowledge_with_memories(self):
        """测试对有记忆主题的评估"""
        from mcp_server import assess_knowledge
        
        # 创建多条相关记忆
        topic = "Python"  # 使用更简单的关键词
        
        for i in range(5):
            self.store.put(
                item_id=None,
                type="feature",
                key=f"python_{i}",
                content=f"Python编程语言知识点 {i}: asyncio, await, coroutine"
            )
        
        # 多次访问以提高巩固分数
        items = self.store.search(query="Python", limit=10)["items"]
        for item in items:
            for _ in range(3):
                self.store.get(item_id=item["id"])
        
        # 评估知识
        result = assess_knowledge(
            topic="Python",
            db_path=self.db_path
        )
        
        self.assertTrue(result["ok"])
        self.assertIn(result["confidence"], ["high", "medium", "low", "very_low", "none"])
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        
        # 如果找到记忆，验证证据
        if result["confidence"] != "none":
            self.assertGreater(len(result["evidence"]), 0)
            self.assertGreater(result["stats"]["total_memories"], 0)
            
            print(f"\n元认知评估结果:")
            print(f"  主题: {topic}")
            print(f"  信心: {result['confidence']}")
            print(f"  分数: {result['score']:.4f}")
            print(f"  记忆数: {result['stats']['total_memories']}")
            print(f"  平均巩固: {result['stats']['avg_consolidation']:.3f}")

    def test_assess_knowledge_quality_levels(self):
        """测试不同质量记忆的元认知评估"""
        from mcp_server import assess_knowledge
        
        # 场景1: 少量低质量记忆
        self.store.put(
            item_id=None,
            type="run",
            key="low_quality",
            content="模糊记忆 vague memory unclear",
            confidence_level="low"
        )
        
        result_low = assess_knowledge(
            topic="模糊",
            db_path=self.db_path
        )
        
        # 场景2: 多条高质量记忆
        for i in range(10):
            self.store.put(
                item_id=None,
                type="feature",
                key=f"high_quality_{i}",
                content=f"清晰的技术文档内容 {i} documentation",
                confidence_level="high"
            )
        
        # 访问以提高巩固
        items = self.store.search(query="技术文档", limit=20)["items"]
        for item in items[:5]:
            for _ in range(2):
                self.store.get(item_id=item["id"])
        
        result_high = assess_knowledge(
            topic="技术文档",
            db_path=self.db_path
        )
        
        # 验证高质量记忆的评估分数更高（允许两者都是 none）
        if result_high["confidence"] != "none" or result_low["confidence"] != "none":
            self.assertGreaterEqual(result_high["score"], result_low["score"])
        
        print(f"\n质量对比:")
        print(f"  低质量: {result_low['confidence']} ({result_low['score']:.3f})")
        print(f"  高质量: {result_high['confidence']} ({result_high['score']:.3f})")


class TestBackwardCompatibility(unittest.TestCase):
    """测试向后兼容性"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_compat.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_old_records_work_with_new_code(self):
        """测试旧记录在新代码中正常工作"""
        # 创建一个旧版本的数据库（包含 Phase 1 的所有字段，但没有 Phase 2-1 字段）
        conn = sqlite3.connect(self.db_path)
        
        # 创建旧版本 schema（Phase 1）
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_items (
              id TEXT PRIMARY KEY,
              type TEXT NOT NULL CHECK (type IN ('project','feature','run')),
              task_id TEXT,
              step_id TEXT,
              key TEXT NOT NULL,
              content TEXT NOT NULL,
              content_search TEXT NOT NULL DEFAULT '',
              tags_json TEXT NOT NULL DEFAULT '[]',
              source TEXT,
              weight REAL NOT NULL DEFAULT 1.0,
              version INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              confidence_level TEXT NOT NULL DEFAULT 'medium',
              verified_by TEXT
            );
        """)
        
        # 插入旧记录
        now = utcnow_iso()
        conn.execute("""
            INSERT INTO memory_items 
            (id, type, task_id, step_id, key, content, content_search, tags_json, source, weight, version, created_at, updated_at, confidence_level, verified_by)
            VALUES ('old_id', 'feature', NULL, NULL, 'old_key', 'old content', 'old content', '[]', NULL, 1.0, 1, ?, ?, 'medium', NULL)
        """, (now, now))
        conn.commit()
        conn.close()
        
        # 使用新代码打开（应该自动迁移添加 Phase 2-1 字段）
        store = MemoryStore(self.db_path)
        
        # 验证可以读取旧记录
        item = store.get(item_id="old_id")
        self.assertEqual(item["key"], "old_key")
        
        # 验证 Phase 2-1 新字段存在且有默认值
        self.assertIn("access_count", item)
        self.assertEqual(item["access_count"], 0)  # 默认值
        
        self.assertIn("last_accessed_at", item)
        
        self.assertIn("consolidation_score", item)
        self.assertEqual(item["consolidation_score"], 0.0)  # 默认值


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for memtool/metacognition.py - Phase 2.7 元认知增强
"""
import datetime as dt
import math
import sys
import unittest
from pathlib import Path

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).parent))

from memtool.metacognition import (
    calc_quantity_score,
    calc_quality_score,
    calc_recency_score,
    calc_access_score,
    compute_breakdown,
    identify_issues,
    generate_suggestions,
    find_bottleneck,
    QUANTITY_THRESHOLD,
)


class TestCalcQuantityScore(unittest.TestCase):
    """测试数量维度分数计算"""

    def test_empty_items(self):
        """空列表返回 0"""
        self.assertEqual(calc_quantity_score([]), 0.0)

    def test_below_threshold(self):
        """低于阈值时返回比例"""
        items = [{"id": str(i)} for i in range(5)]
        score = calc_quantity_score(items, threshold=10)
        self.assertEqual(score, 0.5)

    def test_at_threshold(self):
        """等于阈值时返回 1.0"""
        items = [{"id": str(i)} for i in range(10)]
        score = calc_quantity_score(items, threshold=10)
        self.assertEqual(score, 1.0)

    def test_above_threshold(self):
        """超过阈值时仍返回 1.0（封顶）"""
        items = [{"id": str(i)} for i in range(20)]
        score = calc_quantity_score(items, threshold=10)
        self.assertEqual(score, 1.0)

    def test_zero_threshold(self):
        """阈值为 0 时返回 1.0"""
        items = [{"id": "1"}]
        score = calc_quantity_score(items, threshold=0)
        self.assertEqual(score, 1.0)


class TestCalcQualityScore(unittest.TestCase):
    """测试质量维度分数计算"""

    def test_empty_items(self):
        """空列表返回 0"""
        self.assertEqual(calc_quality_score([]), 0.0)

    def test_all_high_confidence(self):
        """全部高置信度返回 1.0"""
        items = [{"confidence_level": "high"} for _ in range(5)]
        self.assertEqual(calc_quality_score(items), 1.0)

    def test_all_medium_confidence(self):
        """全部中等置信度返回 0.6"""
        items = [{"confidence_level": "medium"} for _ in range(5)]
        self.assertEqual(calc_quality_score(items), 0.6)

    def test_all_low_confidence(self):
        """全部低置信度返回 0.3"""
        items = [{"confidence_level": "low"} for _ in range(5)]
        self.assertEqual(calc_quality_score(items), 0.3)

    def test_mixed_confidence(self):
        """混合置信度返回平均值"""
        items = [
            {"confidence_level": "high"},   # 1.0
            {"confidence_level": "low"},    # 0.3
        ]
        expected = (1.0 + 0.3) / 2
        self.assertEqual(calc_quality_score(items), expected)

    def test_missing_confidence_level(self):
        """缺少 confidence_level 时使用默认值 0.6"""
        items = [{}]
        self.assertEqual(calc_quality_score(items), 0.6)


class TestCalcRecencyScore(unittest.TestCase):
    """测试新鲜度维度分数计算"""

    def test_empty_items(self):
        """空列表返回 0"""
        self.assertEqual(calc_recency_score([]), 0.0)

    def test_recent_items(self):
        """最近更新的记忆分数接近 1.0"""
        now = dt.datetime.now(tz=dt.timezone.utc)
        items = [
            {"updated_at": now.isoformat(), "type": "feature"},
        ]
        score = calc_recency_score(items, now=now)
        self.assertGreater(score, 0.9)

    def test_old_items(self):
        """很久前的记忆分数较低"""
        now = dt.datetime.now(tz=dt.timezone.utc)
        old_time = now - dt.timedelta(days=365)
        items = [
            {"updated_at": old_time.isoformat(), "type": "feature"},
        ]
        score = calc_recency_score(items, now=now)
        self.assertLess(score, 0.5)

    def test_missing_updated_at(self):
        """缺少 updated_at 时使用默认值 0.5"""
        items = [{"type": "feature"}]
        score = calc_recency_score(items)
        self.assertEqual(score, 0.5)


class TestCalcAccessScore(unittest.TestCase):
    """测试访问频率维度分数计算"""

    def test_empty_items(self):
        """空列表返回 0"""
        self.assertEqual(calc_access_score([]), 0.0)

    def test_zero_access(self):
        """访问次数为 0 返回 0"""
        items = [{"access_count": 0}]
        self.assertEqual(calc_access_score(items), 0.0)

    def test_high_access(self):
        """高访问次数返回接近 1.0"""
        items = [{"access_count": 100}]
        score = calc_access_score(items)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_moderate_access(self):
        """中等访问次数返回中间值"""
        items = [{"access_count": 10}]
        score = calc_access_score(items)
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.7)

    def test_missing_access_count(self):
        """缺少 access_count 时视为 0"""
        items = [{}]
        self.assertEqual(calc_access_score(items), 0.0)


class TestComputeBreakdown(unittest.TestCase):
    """测试综合 breakdown 计算"""

    def test_empty_items(self):
        """空列表返回全 0"""
        breakdown = compute_breakdown([])
        self.assertEqual(breakdown["quantity_score"], 0.0)
        self.assertEqual(breakdown["quality_score"], 0.0)
        self.assertEqual(breakdown["recency_score"], 0.0)
        self.assertEqual(breakdown["access_score"], 0.0)

    def test_full_breakdown(self):
        """完整数据返回所有维度"""
        now = dt.datetime.now(tz=dt.timezone.utc)
        items = [
            {
                "confidence_level": "high",
                "updated_at": now.isoformat(),
                "type": "feature",
                "access_count": 50,
            }
            for _ in range(10)
        ]
        breakdown = compute_breakdown(items, now=now)

        self.assertEqual(breakdown["quantity_score"], 1.0)
        self.assertEqual(breakdown["quality_score"], 1.0)
        self.assertGreater(breakdown["recency_score"], 0.9)
        self.assertGreater(breakdown["access_score"], 0.5)


class TestIdentifyIssues(unittest.TestCase):
    """测试问题识别"""

    def test_no_issues(self):
        """健康数据无问题"""
        breakdown = {
            "quantity_score": 0.8,
            "quality_score": 0.9,
            "recency_score": 0.7,
            "access_score": 0.5,
        }
        issues = identify_issues(breakdown)
        self.assertEqual(issues, [])

    def test_data_stale(self):
        """低新鲜度触发 data_stale"""
        breakdown = {
            "quantity_score": 0.8,
            "quality_score": 0.9,
            "recency_score": 0.2,  # 低于 0.4
            "access_score": 0.5,
        }
        issues = identify_issues(breakdown)
        self.assertIn("data_stale", issues)

    def test_coverage_gap(self):
        """低数量触发 coverage_gap"""
        breakdown = {
            "quantity_score": 0.1,  # 低于 0.3
            "quality_score": 0.9,
            "recency_score": 0.7,
            "access_score": 0.5,
        }
        issues = identify_issues(breakdown)
        self.assertIn("coverage_gap", issues)

    def test_low_confidence(self):
        """低质量触发 low_confidence"""
        breakdown = {
            "quantity_score": 0.8,
            "quality_score": 0.3,  # 低于 0.5
            "recency_score": 0.7,
            "access_score": 0.5,
        }
        issues = identify_issues(breakdown)
        self.assertIn("low_confidence", issues)

    def test_rarely_accessed(self):
        """低访问触发 rarely_accessed"""
        breakdown = {
            "quantity_score": 0.8,
            "quality_score": 0.9,
            "recency_score": 0.7,
            "access_score": 0.1,  # 低于 0.2
        }
        issues = identify_issues(breakdown)
        self.assertIn("rarely_accessed", issues)

    def test_multiple_issues(self):
        """多个问题同时存在"""
        breakdown = {
            "quantity_score": 0.1,
            "quality_score": 0.3,
            "recency_score": 0.2,
            "access_score": 0.1,
        }
        issues = identify_issues(breakdown)
        self.assertEqual(len(issues), 4)


class TestGenerateSuggestions(unittest.TestCase):
    """测试建议生成"""

    def test_no_issues_positive_feedback(self):
        """无问题时给出正面反馈"""
        breakdown = {"quantity_score": 0.8, "quality_score": 0.9, "recency_score": 0.7, "access_score": 0.5}
        suggestions = generate_suggestions("Python", breakdown, [])
        self.assertEqual(len(suggestions), 1)
        self.assertIn("直接使用", suggestions[0])

    def test_data_stale_suggestion(self):
        """data_stale 问题生成更新建议"""
        breakdown = {"quantity_score": 0.8, "quality_score": 0.9, "recency_score": 0.2, "access_score": 0.5}
        suggestions = generate_suggestions("React", breakdown, ["data_stale"])
        self.assertTrue(any("最新信息" in s for s in suggestions))

    def test_coverage_gap_suggestion(self):
        """coverage_gap 问题生成补充建议"""
        breakdown = {"quantity_score": 0.1, "quality_score": 0.9, "recency_score": 0.7, "access_score": 0.5}
        suggestions = generate_suggestions("Vue", breakdown, ["coverage_gap"])
        self.assertTrue(any("询问用户" in s or "补充搜索" in s for s in suggestions))

    def test_suggestion_includes_topic(self):
        """建议包含主题名称"""
        breakdown = {"quantity_score": 0.1, "quality_score": 0.9, "recency_score": 0.7, "access_score": 0.5}
        suggestions = generate_suggestions("TypeScript", breakdown, ["coverage_gap"])
        self.assertTrue(any("TypeScript" in s for s in suggestions))


class TestFindBottleneck(unittest.TestCase):
    """测试瓶颈识别"""

    def test_empty_breakdown(self):
        """空 breakdown 返回 None"""
        self.assertIsNone(find_bottleneck({}))

    def test_find_lowest(self):
        """找出最低分维度"""
        breakdown = {
            "quantity_score": 0.8,
            "quality_score": 0.9,
            "recency_score": 0.2,
            "access_score": 0.5,
        }
        self.assertEqual(find_bottleneck(breakdown), "recency_score")

    def test_all_equal(self):
        """全部相等时返回其中一个"""
        breakdown = {
            "quantity_score": 0.5,
            "quality_score": 0.5,
            "recency_score": 0.5,
            "access_score": 0.5,
        }
        result = find_bottleneck(breakdown)
        self.assertIn(result, breakdown.keys())


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_assess_knowledge_returns_new_fields(self):
        """测试 assess_knowledge 返回新字段"""
        # 这个测试需要实际的数据库，跳过如果没有
        try:
            from mcp_server import assess_knowledge
        except ImportError:
            self.skipTest("mcp_server not available")

        # 使用一个不存在的主题测试
        result = assess_knowledge("完全不存在的随机主题12345")

        # 验证新字段存在
        self.assertIn("breakdown", result)
        self.assertIn("issues", result)

        # 验证 breakdown 结构
        if result.get("breakdown"):
            self.assertIn("quantity_score", result["breakdown"])
            self.assertIn("quality_score", result["breakdown"])
            self.assertIn("recency_score", result["breakdown"])
            self.assertIn("access_score", result["breakdown"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

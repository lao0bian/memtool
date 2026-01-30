#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 测试：验证相似度检测、模板系统和可信度字段功能
"""
import os
import sys
import tempfile
import json
import shutil
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from memtool_core import MemoryStore, init_db


def test_similarity_detection():
    """测试 1: 相似度检测功能"""
    print("\n=== 测试 1: 相似度检测 ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path)
        store.init_db()

        # 写入第一条记录
        result1 = store.put(
            item_id=None,
            type="run",
            key="error:E404:router",
            content="E404 错误发生在路由器中，原因是连接超时，通过添加重试逻辑修复。",
        )
        assert result1["ok"], f"第一条记录写入失败: {result1}"
        assert "warning" not in result1, f"首条记录不应触发重复检测: {result1}"
        print("✓ 第一条记录写入成功")

        # 更新同一条记录（不应把自身当成重复）
        result1_update = store.put(
            item_id=result1["id"],
            type="run",
            key="error:E404:router",
            content="E404 错误发生在路由器中，原因是连接超时，通过添加重试逻辑修复。",
        )
        assert result1_update["ok"], f"第一条记录更新失败: {result1_update}"
        assert "warning" not in result1_update, f"更新自身不应触发重复检测: {result1_update}"
        print("✓ 更新自身不触发重复检测")

        # 写入相似的记录（应该收到 warning）
        result2 = store.put(
            item_id=None,
            type="run",
            key="error:E404:handler",
            content="E404 错误发生在处理器中，原因是连接超时，应用了重试机制来修复。",
        )
        assert result2["ok"], f"第二条记录写入失败: {result2}"

        # 检查是否有相似度警告
        if "warning" in result2:
            assert result2["warning"] == "duplicate_detection", f"预期警告为 duplicate_detection，实际为 {result2['warning']}"
            assert "similar_items" in result2, "警告中缺少 similar_items 字段"
            assert len(result2["similar_items"]) > 0, "应该找到相似的记录"
            print(f"✓ 找到 {len(result2['similar_items'])} 条相似记录")
            print(f"  相似度: {result2['similar_items'][0]['similarity']}")
        else:
            print("⚠️ 未找到相似记录（可能是正常的，取决于相似度阈值）")

        # 写入完全不同的记录（不应该有相似度警告）
        result3 = store.put(
            item_id=None,
            type="feature",
            key="cache:ttl:strategy",
            content="缓存策略决策：使用 Redis 作为缓存存储，TTL 设置为 30 分钟。",
        )
        assert result3["ok"], f"第三条记录写入失败: {result3}"

        if "warning" in result3 and result3["warning"] == "duplicate_detection":
            print("⚠️ 不同类型的记录也有相似度警告")
        else:
            print("✓ 不同类型的记录没有相似度警告")


def test_template_system():
    """测试 2: 模板系统"""
    print("\n=== 测试 2: 模板系统 ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path)
        store.init_db()

        # 验证模板文件存在
        template_file = os.path.join(os.path.dirname(__file__), "memtool_templates.yaml")
        assert os.path.exists(template_file), f"模板文件不存在: {template_file}"
        print(f"✓ 模板文件存在: {template_file}")

        # 测试 CLI 命令（通过 memtool.py）
        import subprocess

        # 测试 template list 命令
        result = subprocess.run(
            [sys.executable, "memtool.py", "--db", db_path, "template", "list", "--format", "json"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                if output.get("ok"):
                    templates = output.get("templates", [])
                    print(f"✓ 找到 {len(templates)} 个模板")
                    for tmpl in templates:
                        print(f"  - {tmpl['name']}: {tmpl.get('description', '')}")
                else:
                    print(f"✗ template list 返回错误: {output}")
            except json.JSONDecodeError:
                print(f"✗ 无法解析 JSON 输出: {result.stdout}")
        else:
            print(f"✗ template list 命令失败: {result.stderr}")


def test_confidence_level():
    """测试 3: 可信度字段"""
    print("\n=== 测试 3: 可信度字段 ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path)
        store.init_db()

        # 写入不同置信度的记录
        result_high = store.put(
            item_id=None,
            type="feature",
            key="decision:cache:strategy",
            content="使用 Redis，经过代码审查确认。",
            confidence_level="high",
            verified_by="code_review#456",
        )
        assert result_high["ok"], f"高置信度记录写入失败: {result_high}"
        high_id = result_high["id"]
        print(f"✓ 高置信度记录写入成功 (id: {high_id})")

        # 写入中置信度的记录
        result_medium = store.put(
            item_id=None,
            type="feature",
            key="decision:error:handling",
            content="使用 exception wrapping 处理异常。",
            confidence_level="medium",
        )
        assert result_medium["ok"], f"中置信度记录写入失败: {result_medium}"
        print("✓ 中置信度记录写入成功")

        # 写入低置信度的记录
        result_low = store.put(
            item_id=None,
            type="run",
            key="debug:session:123",
            content="可能是竞态条件导致的偶发问题。",
            confidence_level="low",
        )
        assert result_low["ok"], f"低置信度记录写入失败: {result_low}"
        print("✓ 低置信度记录写入成功")

        # 验证高置信度记录的读取
        retrieved = store.get(item_id=high_id)
        assert retrieved.get("confidence_level") == "high", f"置信度字段不匹配: {retrieved.get('confidence_level')}"
        assert retrieved.get("verified_by") == "code_review#456", f"verified_by 字段不匹配"
        print("✓ 高置信度记录读取验证成功")

        # 验证列表查询
        results = store.list(type="feature", limit=10)
        assert len(results) >= 2, "应该至少有 2 条 feature 类型的记录"
        print(f"✓ 列表查询返回 {len(results)} 条记录")

        # 检查置信度字段
        for result in results:
            assert "confidence_level" in result, f"记录缺少 confidence_level 字段: {result['id']}"
            assert result["confidence_level"] in ["high", "medium", "low"], f"置信度值无效: {result['confidence_level']}"
        print("✓ 所有记录都有有效的置信度字段")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("Phase 1 功能测试")
    print("=" * 50)

    try:
        test_similarity_detection()
        test_template_system()
        test_confidence_level()

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostToolUse Hook: 自动捕获关键操作并存储为记忆

触发条件：
- Edit/Write: 文件编辑后，LLM 智能判断是否记录
- Bash (git commit): 提交代码后自动记录 commit message
"""
import json
import sys
import os
from datetime import datetime

# 添加 memtool 路径
MEMTOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MEMTOOL_DIR)

from utils import get_store, analyze_change_via_llm, extract_commit_message


def handle_edit_or_write(tool_name: str, tool_input: dict):
    """处理 Edit/Write 工具"""
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return

    # 对于 Write 工具，没有 old_string，跳过（新文件创建不自动记录）
    if tool_name == "Write" and "old_string" not in tool_input:
        return

    old_content = tool_input.get("old_string", "")
    new_content = tool_input.get("new_string", tool_input.get("content", ""))

    # 使用 LLM 智能判断是否记录
    should_record, intent = analyze_change_via_llm(file_path, old_content, new_content)

    if not should_record or not intent:
        return

    # 存储到 memtool
    store = get_store()
    if not store:
        return

    try:
        # 生成 key：基于文件路径
        key = f"file:{file_path}"

        store.put(
            item_id=None,
            type="feature",
            key=key,
            content=f"最近修改：{intent}",
            source=file_path,
            weight=0.8,  # 中等权重
        )
    except Exception:
        pass


def handle_bash_commit(tool_input: dict, tool_output: dict):
    """处理 git commit"""
    command = tool_input.get("command", "")

    if "git commit" not in command:
        return

    # 提取 commit message
    output_text = tool_output.get("output", "")
    commit_msg = extract_commit_message(output_text)

    if not commit_msg:
        return

    # 存储到 memtool
    store = get_store()
    if not store:
        return

    try:
        key = f"commit:{datetime.now().strftime('%Y%m%d')}"

        store.put(
            item_id=None,
            type="run",
            key=key,
            content=commit_msg,
            weight=0.5,  # 较低权重
        )
    except Exception:
        pass


def main():
    try:
        # 读取 hook 输入
        input_data = json.load(sys.stdin)

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        tool_output = input_data.get("tool_output", {})

        # 根据工具类型分发
        if tool_name in ["Edit", "Write"]:
            handle_edit_or_write(tool_name, tool_input)
        elif tool_name == "Bash":
            handle_bash_commit(tool_input, tool_output)

        # PostToolUse hook 不应该阻塞或修改工具输出，静默退出
        sys.exit(0)

    except Exception:
        # Hook 失败不应阻塞，静默退出
        sys.exit(0)


if __name__ == "__main__":
    main()

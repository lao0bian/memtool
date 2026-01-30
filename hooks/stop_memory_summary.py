#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stop Hook: 会话结束时自动总结重要发现

触发条件：
- 会话对话轮数 >= 5 时才触发（避免短会话产生成本）
- 使用 haiku 模型降低成本
"""
import json
import sys
import os
from datetime import datetime

# 添加 memtool 路径
MEMTOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MEMTOOL_DIR)

from utils import get_store, call_llm, count_turns, read_jsonl, filter_sensitive_content


_MIN_TURNS = int(os.environ.get("MEMTOOL_HOOK_MIN_TURNS", "5"))


def _entry_role(entry: dict) -> str:
    """兼容不同 transcript 字段命名"""
    return entry.get("role") or entry.get("type") or ""


def _entry_content(entry: dict) -> str:
    """兼容不同 transcript 字段命名"""
    return entry.get("content") or entry.get("text") or ""


def summarize_session(transcript: list) -> str:
    """
    总结会话的关键发现

    Args:
        transcript: 会话记录（JSONL 解析后的列表）

    Returns:
        总结文本（如果无重要内容则返回空字符串）
    """
    if not transcript:
        return ""

    # 提取用户消息和助手响应（先过滤再截断，避免 tool 日志挤占窗口）
    conversation_entries = []
    for entry in transcript:
        role = _entry_role(entry)
        if role in {"user", "human", "assistant", "assistant_response", "assistant_message"}:
            content = _entry_content(entry)
            if content:
                conversation_entries.append((role, content))

    if not conversation_entries:
        return ""

    conversation = []
    for role, content in conversation_entries[-20:]:  # 只取最近 20 条，避免 token 过多
        if role in {"user", "human"}:
            conversation.append(f"用户: {content[:200]}")  # 限制长度
        else:
            conversation.append(f"助手: {content[:200]}")

    if not conversation:
        return ""

    conversation_text = "\n".join(conversation)

    # 过滤敏感信息
    conversation_text = filter_sensitive_content(conversation_text)

    prompt = f"""分析以下会话记录，提取值得长期记住的关键信息。

会话记录：
{conversation_text}

请提取：
1. 新发现的架构约束或技术决策
2. 重要的 bug 修复经验
3. 值得记住的代码模式或设计模式

要求：
- 只返回真正有价值的信息，无关紧要的日常操作不需要记录
- 每条信息用一句话概括，不超过 100 字
- 如果没有值得记录的内容，直接返回 "无"

输出格式（示例）：
1. 认证模块改用 JWT，过期时间设为 7 天
2. 修复了分页查询的 offset 计算错误
3. 使用策略模式重构了支付流程

或者直接返回：
无
"""

    summary = call_llm(prompt, model="haiku")

    if not summary or summary.strip() == "无":
        return ""

    return summary.strip()


def main():
    # 已弃用：请改用 SessionEnd Hook（hooks/session_end_memory_summary.py）
    sys.exit(0)


if __name__ == "__main__":
    main()

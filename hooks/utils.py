#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hooks 共享工具函数
"""
import json
import os
import sys
import re
from typing import Optional, Tuple, Dict, Any

# 添加 memtool 路径
MEMTOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MEMTOOL_DIR)

from memtool_core import MemoryStore


def get_db_path() -> str:
    """获取数据库路径"""
    return os.environ.get("MEMTOOL_DB", os.path.expanduser("~/.memtool/shared.db"))


def get_store() -> Optional[MemoryStore]:
    """获取 MemoryStore 实例"""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return None
    try:
        return MemoryStore(db_path)
    except Exception:
        return None


def call_llm(prompt: str, model: str = "haiku") -> str:
    """
    调用 LLM（优先使用 codex CLI，降级到 anthropic API）

    Args:
        prompt: 提示词
        model: 模型名称（haiku, sonnet, opus）

    Returns:
        LLM 生成的文本
    """
    prefer_codex_only = str(model).lower() in {"gpt-5.2", "gpt5.2", "codex"}

    # 策略 1：尝试使用 codex CLI（更稳定，使用 GPT-5.2）
    try:
        import subprocess

        result = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    if prefer_codex_only:
        return ""

    # 策略 2：降级到 anthropic API
    try:
        import anthropic

        # 从环境变量获取 API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")

        if not api_key:
            # 尝试从 ~/.claude/settings.json 读取
            settings_path = os.path.expanduser("~/.claude/settings.json")
            if os.path.exists(settings_path):
                import json
                try:
                    with open(settings_path, 'r') as f:
                        settings = json.load(f)
                        env_config = settings.get("env", {})
                        api_key = env_config.get("ANTHROPIC_API_KEY")
                        if not base_url:
                            base_url = env_config.get("ANTHROPIC_BASE_URL")
                except Exception:
                    pass

        if api_key is not None and not api_key.strip():
            api_key = None
        if base_url is not None and not base_url.strip():
            base_url = None
        if base_url:
            base_url = base_url.strip()
            # 规范化 base_url，避免重复拼接 /v1 或 /v1/messages
            for suffix in ("/v1/messages", "/v1"):
                if base_url.endswith(suffix):
                    base_url = base_url[:-len(suffix)]
                    break
            base_url = base_url.rstrip("/") or None

        if not api_key:
            return ""

        # 映射模型名称
        model_map = {
            "haiku": "claude-3-5-haiku-20241022",
            "sonnet": "claude-sonnet-4-5-20250929",
            "opus": "claude-opus-4-5-20251101",
        }
        model_id = model_map.get(model, model)

        # 创建客户端
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = anthropic.Anthropic(**client_kwargs)

        # 调用 API
        message = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # 兼容 thinking/text 多 block 输出，只提取 text 内容
        parts = []
        for block in getattr(message, "content", []) or []:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
                continue
            text = getattr(block, "text", None)
            if text:
                parts.append(text)

        return "\n".join(parts).strip()

    except Exception as e:
        # LLM 调用失败，返回空字符串
        return ""


def analyze_change_via_llm(file_path: str, old_content: str, new_content: str) -> Tuple[bool, str]:
    """
    使用 LLM 分析代码变更是否值得记录（有降级策略）

    Args:
        file_path: 文件路径
        old_content: 旧内容（Edit 工具的 old_string）
        new_content: 新内容（Edit 工具的 new_string）

    Returns:
        (是否值得记录, 修改意图描述)
    """
    # 限制 diff 长度，避免 token 成本过高
    max_len = 500
    if len(old_content) > max_len:
        old_content = old_content[:max_len] + "..."
    if len(new_content) > max_len:
        new_content = new_content[:max_len] + "..."

    prompt = f"""分析以下代码变更，判断是否值得记录到记忆系统。

文件路径：{file_path}

旧代码：
```
{old_content}
```

新代码：
```
{new_content}
```

判断标准：
- 值得记录：架构变更、API 修改、重要逻辑优化、Bug 修复
- 不值得记录：格式调整、注释修改、小的重构、日常维护

请按以下格式回复（必须严格遵守）：
RECORD: yes/no
INTENT: 一句话描述修改意图（如果 RECORD=no 则留空）

示例1：
RECORD: yes
INTENT: 将 JWT 过期时间从 24 小时改为 7 天

示例2：
RECORD: no
INTENT:
"""

    response = call_llm(prompt, model="haiku")

    if not response:
        # LLM 调用失败，使用简单规则降级
        return _fallback_analysis(file_path, old_content, new_content)

    # 解析 LLM 响应
    should_record = False
    intent = ""

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("RECORD:"):
            value = line.split(":", 1)[1].strip().lower()
            should_record = value in ["yes", "true", "是"]
        elif line.startswith("INTENT:"):
            intent = line.split(":", 1)[1].strip()

    return should_record, intent


def _fallback_analysis(file_path: str, old_content: str, new_content: str) -> Tuple[bool, str]:
    """
    LLM 不可用时的降级分析策略

    使用简单规则判断：
    1. 配置文件 → 总是记录
    2. 架构关键词 → 总是记录
    3. 其他 → 不记录
    """
    # 配置文件
    config_patterns = [
        "config", "settings", ".env", ".yaml", ".yml", ".json",
        "package.json", "tsconfig", "webpack", "vite"
    ]
    if any(pattern in file_path.lower() for pattern in config_patterns):
        return True, f"配置文件修改：{file_path}"

    # 架构关键词
    arch_keywords = [
        "api", "auth", "database", "cache", "redis", "jwt",
        "interface", "schema", "model", "migration"
    ]
    combined = f"{old_content} {new_content}".lower()
    if any(keyword in combined for keyword in arch_keywords):
        return True, f"架构相关修改：{file_path}"

    # 其他不记录
    return False, ""


def extract_commit_message(bash_output: str) -> str:
    """
    从 git commit 输出中提取 commit message

    Args:
        bash_output: Bash 工具的输出

    Returns:
        commit message
    """
    # 匹配 git commit 输出格式
    # 例如：[main 1234567] Add dark mode
    match = re.search(r'\[.+?\]\s+(.+)', bash_output)
    if match:
        return match.group(1).strip()
    return ""


def _entry_role(entry: dict) -> str:
    """兼容不同 transcript 字段命名"""
    return entry.get("role") or entry.get("type") or ""


def count_turns(transcript: list) -> int:
    """
    统计对话轮数

    Args:
        transcript: 会话记录（JSONL 格式解析后的列表）

    Returns:
        对话轮数
    """
    if not transcript:
        return 0

    # 统计用户消息的数量作为轮数
    user_turns = sum(1 for entry in transcript if _entry_role(entry) in {"user", "human"})
    return user_turns


def read_jsonl(file_path: str) -> list:
    """
    读取 JSONL 文件

    Args:
        file_path: 文件路径

    Returns:
        解析后的 JSON 对象列表
    """
    if not os.path.exists(file_path):
        return []

    items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    except Exception:
        pass

    return items


def filter_sensitive_content(text: str) -> str:
    """
    过滤敏感信息

    Args:
        text: 原始文本

    Returns:
        过滤后的文本
    """
    # 简单的敏感词过滤（可扩展）
    sensitive_patterns = [
        (r'(api[_-]?key|apikey)\s*[=:]\s*["\']?[\w-]+["\']?', r'\1=***'),
        (r'(password|passwd|pwd)\s*[=:]\s*["\']?[\w-]+["\']?', r'\1=***'),
        (r'(token|access[_-]?token)\s*[=:]\s*["\']?[\w.-]+["\']?', r'\1=***'),
        (r'(secret|secret[_-]?key)\s*[=:]\s*["\']?[\w-]+["\']?', r'\1=***'),
    ]

    filtered = text
    for pattern, replacement in sensitive_patterns:
        filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

    return filtered

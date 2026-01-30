#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UserPromptSubmit Hook: 自动注入相关记忆上下文

工作流程：
1. 接收用户输入
2. 先用 search 精确匹配，再用 recommend 补充
3. 只返回真正相关的记忆
"""
import json
import sys
import os
import re

# 添加 memtool 路径
MEMTOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MEMTOOL_DIR)

from memtool_core import MemoryStore

try:
    import jieba as _jieba
except Exception:
    _jieba = None


_STOPWORDS = {
    "的", "是", "在", "了", "和", "与", "或", "这", "那", "有", "我", "你", "他",
    "什么", "怎么", "如何", "为什么", "哪个", "哪些", "帮我", "请", "看看",
    "介绍", "介绍下", "一下", "简要", "描述",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "what", "how", "why", "when", "where", "which", "who", "whom",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


_MIN_TOKEN_LEN = max(_env_int("MEMTOOL_HOOK_MIN_TOKEN_LEN", 2), 1)
_MAX_TOKENS = max(_env_int("MEMTOOL_HOOK_MAX_TOKENS", 6), 1)
_MAX_ITEMS = max(_env_int("MEMTOOL_HOOK_MAX_ITEMS", 3), 1)
_MIN_RELEVANCE = _env_float("MEMTOOL_HOOK_MIN_RELEVANCE", 0.25)


def get_db_path() -> str:
    """获取数据库路径"""
    return os.environ.get("MEMTOOL_DB", os.path.expanduser("~/.memtool/shared.db"))


def extract_keywords(text: str) -> list:
    """提取关键词用于检索（分词 + 规则兜底，保序去重）"""
    if not text:
        return []
    lower = text.lower()
    tokens = []
    if _jieba is not None:
        try:
            for tok in _jieba.cut_for_search(lower):
                tokens.append(tok)
        except Exception:
            pass
    tokens.extend(re.findall(r"[\u4e00-\u9fff]+|\w+", lower))
    if _jieba is None:
        extra = []
        for tok in tokens:
            if not re.fullmatch(r"[\u4e00-\u9fff]+", tok or ""):
                continue
            parts = [p for p in re.split(r"[的和与或在是了上下]", tok) if len(p) >= _MIN_TOKEN_LEN]
            for part in parts:
                extra.append(part)
                if len(part) >= 3:
                    for i in range(0, len(part) - 1, 2):
                        extra.append(part[i:i + 2])
                    if len(part) % 2 == 1:
                        extra.append(part[-2:])
        tokens.extend(extra)

    seen = set()
    out = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if len(t) < _MIN_TOKEN_LEN:
            continue
        if t in _STOPWORDS:
            continue
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
    return out


def should_skip(user_prompt: str) -> bool:
    """判断是否跳过记忆检索"""
    prompt_lower = user_prompt.lower().strip()

    # 空输入才跳过（确保每次用户输入都会尝试检索）
    if not prompt_lower:
        return True

    return False


def calculate_relevance(query_tokens: list, item: dict) -> float:
    """计算查询与记忆的相关度"""
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)

    content = item.get("content", "") or ""
    key = item.get("key", "") or ""
    content_keywords = set(extract_keywords(content))
    key_keywords = set(extract_keywords(key))

    if not content_keywords and not key_keywords:
        return 0.0

    content_hit = len(query_set & content_keywords)
    key_hit = len(query_set & key_keywords)
    if content_hit == 0 and key_hit == 0:
        return 0.0

    coverage_content = content_hit / len(query_set)
    coverage_key = key_hit / len(query_set)
    relevance = (0.7 * coverage_content) + (0.3 * coverage_key)
    return relevance


def _scan_candidates(store: MemoryStore, query_tokens: list, limit: int = 200) -> list:
    """FTS 不可靠时的轻量候选扫描（子串匹配）"""
    try:
        items = store.list(limit=limit, include_stale=False, sort_by="recency")
    except Exception:
        return []
    tokens = [t.lower() for t in query_tokens if t]
    out = []
    for item in items:
        text = f"{item.get('content', '')} {item.get('key', '')}".lower()
        if any(t in text for t in tokens):
            out.append(item)
    return out


def format_memories(items: list) -> str:
    """格式化记忆条目为可读文本"""
    if not items:
        return ""

    lines = []
    for i, item in enumerate(items, 1):
        type_label = {"project": "项目", "feature": "功能", "run": "任务"}.get(item.get("type"), item.get("type"))
        key = item.get("key", "")
        content = item.get("content", "")

        # 截断过长的内容
        if len(content) > 400:
            content = content[:400] + "..."

        lines.append(f"{i}. [{type_label}] {key}\n   {content}")

    return "\n\n".join(lines)


def main():
    try:
        # 读取 hook 输入
        input_data = json.load(sys.stdin)
        # Claude Code 的 UserPromptSubmit 事件字段为 prompt，这里兼容旧字段
        user_prompt = (
            input_data.get("prompt")
            or input_data.get("user_prompt")
            or ""
        )

        # 判断是否需要跳过
        if should_skip(user_prompt):
            sys.exit(0)

        # 初始化 MemoryStore
        db_path = get_db_path()
        if not os.path.exists(db_path):
            sys.exit(0)

        store = MemoryStore(db_path)
        query_tokens = extract_keywords(user_prompt)
        if not query_tokens:
            sys.exit(0)

        query_tokens = query_tokens[:_MAX_TOKENS]

        # === 分层检索策略 ===

        # 1. 项目级约束（project）：精确搜索
        project_items = []
        seen_ids = set()
        try:
            result = store.search(
                query=" OR ".join(query_tokens[:3]) if len(query_tokens) > 1 else query_tokens[0],
                type="project",
                limit=5,
                include_stale=False
            )
            for item in result.get("items", []):
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    project_items.append(item)
        except Exception:
            pass

        # 2. 功能模块（feature）：推荐算法
        feature_items = []
        try:
            result = store.recommend(
                context=user_prompt,
                type="feature",
                limit=8,
                include_stale=False
            )
            for item in result.get("items", []):
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    feature_items.append(item)
        except Exception:
            pass

        # 3. 最近任务（run）：按时间排序
        run_items = []
        try:
            result = store.list(
                type="run",
                sort_by="updated",
                limit=3,
                include_stale=False
            )
            for item in result.get("items", []):
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    run_items.append(item)
        except Exception:
            pass

        # === 过滤与排序 ===

        # 对 project 和 feature 计算相关度并过滤
        def filter_by_relevance(items, min_rel=_MIN_RELEVANCE):
            scored = []
            for item in items:
                relevance = calculate_relevance(query_tokens, item)
                if relevance >= min_rel:
                    item["_relevance"] = relevance
                    scored.append(item)
            scored.sort(key=lambda x: x.get("_relevance", 0), reverse=True)
            return scored

        project_items = filter_by_relevance(project_items)[:3]
        feature_items = filter_by_relevance(feature_items, min_rel=0.2)[:5]
        # run 不需要相关度过滤，直接取最近 2 条
        run_items = run_items[:2]

        # 如果三层都为空，退出
        if not project_items and not feature_items and not run_items:
            sys.exit(0)

        # === 分层格式化输出 ===

        sections = []

        if project_items:
            sections.append(f"### 项目级约束\n{format_memories(project_items)}")

        if feature_items:
            sections.append(f"### 相关功能模块\n{format_memories(feature_items)}")

        if run_items:
            sections.append(f"### 最近任务上下文\n{format_memories(run_items)}")

        formatted = "\n\n".join(sections)

        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": f"""## 相关记忆（自动注入）

{formatted}
""",
            }
        }

        print(json.dumps(output, ensure_ascii=False))
        sys.exit(0)

    except Exception as e:
        # Hook 失败不应阻塞用户操作，静默退出
        sys.exit(0)


if __name__ == "__main__":
    main()

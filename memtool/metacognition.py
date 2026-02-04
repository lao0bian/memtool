#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metacognition enhancement: granular knowledge assessment.

Phase 2.7: 细粒度元认知评估
- 4 维度 breakdown（quantity/quality/recency/access）
- bottleneck 识别
- 针对性建议生成
"""
from __future__ import annotations

import datetime as _dt
import math
import os
from typing import Any, Dict, List, Optional

from memtool_lifecycle import decay_score

# 安全的环境变量解析函数
def _safe_env_int(key: str, default: int) -> int:
    """安全解析整数环境变量，解析失败时返回默认值"""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _safe_env_float(key: str, default: float) -> float:
    """安全解析浮点数环境变量，解析失败时返回默认值"""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


# 可配置阈值（支持环境变量覆盖，解析失败时使用默认值）
QUANTITY_THRESHOLD = _safe_env_int("MEMTOOL_QUANTITY_THRESHOLD", 10)
RECENCY_STALE_THRESHOLD = _safe_env_float("MEMTOOL_RECENCY_STALE_THRESHOLD", 0.4)
QUALITY_LOW_THRESHOLD = _safe_env_float("MEMTOOL_QUALITY_LOW_THRESHOLD", 0.5)
ACCESS_LOW_THRESHOLD = _safe_env_float("MEMTOOL_ACCESS_LOW_THRESHOLD", 0.2)
QUANTITY_LOW_THRESHOLD = _safe_env_float("MEMTOOL_QUANTITY_LOW_THRESHOLD", 0.3)

# confidence_level 映射
CONFIDENCE_LEVEL_MAP: Dict[str, float] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}


def calc_quantity_score(items: List[Dict[str, Any]], threshold: int = QUANTITY_THRESHOLD) -> float:
    """计算数量维度分数

    Args:
        items: 搜索结果列表
        threshold: 满分阈值（默认 10 条）

    Returns:
        0.0 - 1.0 的分数
    """
    if threshold <= 0:
        return 1.0
    return min(len(items) / float(threshold), 1.0)


def calc_quality_score(items: List[Dict[str, Any]]) -> float:
    """计算质量维度分数（基于 confidence_level）

    Args:
        items: 记忆列表

    Returns:
        0.0 - 1.0 的分数
    """
    if not items:
        return 0.0

    total = 0.0
    for item in items:
        level = item.get("confidence_level", "medium")
        total += CONFIDENCE_LEVEL_MAP.get(level, 0.6)

    return total / len(items)


def calc_recency_score(
    items: List[Dict[str, Any]],
    now: Optional[_dt.datetime] = None,
) -> float:
    """计算新鲜度维度分数（基于 decay_score）

    Args:
        items: 记忆列表
        now: 当前时间（可选）

    Returns:
        0.0 - 1.0 的分数
    """
    if not items:
        return 0.0

    if now is None:
        now = _dt.datetime.now(tz=_dt.timezone.utc)

    total = 0.0
    valid_count = 0

    for item in items:
        updated_at = item.get("updated_at")
        mem_type = item.get("type", "feature")
        consolidation = item.get("consolidation_score")

        if updated_at:
            score = decay_score(
                updated_at,
                mem_type,
                now=now,
                consolidation_score=consolidation,
            )
            total += score
            valid_count += 1
        else:
            # updated_at 缺失时使用默认值
            total += 0.5
            valid_count += 1

    if valid_count == 0:
        return 0.5  # 默认中等分数

    return total / valid_count


def calc_access_score(items: List[Dict[str, Any]]) -> float:
    """计算访问频率维度分数（基于 access_count）

    使用 log1p 归一化避免极端值影响

    Args:
        items: 记忆列表

    Returns:
        0.0 - 1.0 的分数
    """
    if not items:
        return 0.0

    total_access = sum(max(0, item.get("access_count", 0)) for item in items)

    # log1p(access) / log(100) 归一化
    # access=0 -> 0, access=100 -> 1.0
    log_base = math.log(100)
    if log_base <= 0:
        return 0.0

    return min(math.log1p(total_access) / log_base, 1.0)


def compute_breakdown(
    items: List[Dict[str, Any]],
    now: Optional[_dt.datetime] = None,
) -> Dict[str, float]:
    """计算 4 维度 breakdown

    Args:
        items: 记忆列表
        now: 当前时间

    Returns:
        包含 4 个维度分数的字典
    """
    return {
        "quantity_score": round(calc_quantity_score(items), 3),
        "quality_score": round(calc_quality_score(items), 3),
        "recency_score": round(calc_recency_score(items, now), 3),
        "access_score": round(calc_access_score(items), 3),
    }


def identify_issues(
    breakdown: Dict[str, float],
    *,
    recency_threshold: float = RECENCY_STALE_THRESHOLD,
    quantity_threshold: float = QUANTITY_LOW_THRESHOLD,
    quality_threshold: float = QUALITY_LOW_THRESHOLD,
    access_threshold: float = ACCESS_LOW_THRESHOLD,
) -> List[str]:
    """识别知识瓶颈（issues）

    Args:
        breakdown: 4 维度分数
        *_threshold: 各维度阈值

    Returns:
        issue 代码列表（机器可读）
    """
    issues: List[str] = []

    if breakdown.get("recency_score", 1.0) < recency_threshold:
        issues.append("data_stale")

    if breakdown.get("quantity_score", 1.0) < quantity_threshold:
        issues.append("coverage_gap")

    if breakdown.get("quality_score", 1.0) < quality_threshold:
        issues.append("low_confidence")

    if breakdown.get("access_score", 1.0) < access_threshold:
        issues.append("rarely_accessed")

    return issues


def generate_suggestions(
    topic: str,
    breakdown: Dict[str, float],
    issues: List[str],
    items: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """根据 breakdown 和 issues 生成针对性建议

    Args:
        topic: 评估主题
        breakdown: 4 维度分数
        issues: issue 代码列表
        items: 记忆列表（用于计算平均天数等）

    Returns:
        建议文案列表
    """
    suggestions: List[str] = []

    # 计算平均年龄（天数）
    avg_age_days = None
    if items:
        ages = []
        now = _dt.datetime.now(tz=_dt.timezone.utc)
        for item in items:
            updated_at = item.get("updated_at")
            if updated_at:
                try:
                    dt = _dt.datetime.fromisoformat(updated_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_dt.timezone.utc)
                    age = (now - dt).days
                    ages.append(age)
                except (ValueError, TypeError):
                    pass
        if ages:
            avg_age_days = sum(ages) // len(ages)

    # 根据 issues 生成建议
    if "data_stale" in issues:
        if avg_age_days is not None:
            suggestions.append(f"检索最新信息以更新「{topic}」相关的上下文（数据平均 {avg_age_days} 天前）。")
        else:
            suggestions.append(f"检索最新信息以更新「{topic}」相关的上下文。")

    if "coverage_gap" in issues:
        suggestions.append(f"询问用户更多关于「{topic}」的细节或进行补充搜索。")

    if "low_confidence" in issues:
        suggestions.append(f"向用户确认「{topic}」相关关键事实的准确性。")

    if "rarely_accessed" in issues:
        suggestions.append(f"建议回顾「{topic}」相关记忆以重新激活上下文。")

    # 如果没有问题，给出正面反馈
    if not issues:
        suggestions.append("可以直接使用这些记忆回答问题。")

    return suggestions


def find_bottleneck(breakdown: Dict[str, float]) -> Optional[str]:
    """找出最薄弱的维度

    Args:
        breakdown: 4 维度分数

    Returns:
        最低分维度名称，或 None
    """
    if not breakdown:
        return None

    min_score = 1.0
    min_dim = None

    for dim, score in breakdown.items():
        if score < min_score:
            min_score = score
            min_dim = dim

    return min_dim

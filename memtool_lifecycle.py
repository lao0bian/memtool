#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lifecycle helpers: decay score (forgetting curve) and stale 판단
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, Optional


DEFAULT_HALF_LIFE_DAYS: Dict[str, float] = {
    "run": 14.0,
    "feature": 180.0,
    "project": 365.0,
}

DEFAULT_STALE_THRESHOLD = 0.2


def _parse_dt(value: Optional[str]) -> Optional[_dt.datetime]:
    if not value:
        return None
    try:
        dt = _dt.datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def age_days(updated_at: Optional[str], now: Optional[_dt.datetime] = None) -> Optional[float]:
    dt = _parse_dt(updated_at)
    if dt is None:
        return None
    if now is None:
        now = _dt.datetime.now(tz=_dt.timezone.utc)
    delta = now - dt
    return delta.total_seconds() / 86400.0


def half_life_days(mem_type: Optional[str]) -> float:
    if mem_type in DEFAULT_HALF_LIFE_DAYS:
        return DEFAULT_HALF_LIFE_DAYS[mem_type]  # type: ignore[index]
    return 180.0


def decay_score(
    updated_at: Optional[str],
    mem_type: Optional[str],
    *,
    now: Optional[_dt.datetime] = None,
    half_life: Optional[float] = None,
    consolidation_score: Optional[float] = None,
) -> float:
    """
    计算记忆衰减分数（遗忘曲线）
    
    Phase 2-1: 支持动态半衰期，基于记忆巩固分数调整
    - consolidation_score=1.0 时，半衰期 × 3
    - consolidation_score=0.0 时，半衰期不变
    
    Args:
        updated_at: 更新时间（ISO格式）
        mem_type: 记忆类型（project/feature/run）
        now: 当前时间（可选）
        half_life: 显式指定半衰期（可选）
        consolidation_score: 巩固分数 0.0-1.0（可选，Phase 2-1）
    
    Returns:
        float: 衰减分数 (0.0 - 1.0)
    """
    age = age_days(updated_at, now=now)
    if age is None or age <= 0:
        return 1.0
    
    # 确定基础半衰期
    if half_life is not None:
        hl = half_life
    else:
        hl = half_life_days(mem_type)
    
    if hl <= 0:
        return 0.0
    
    # Phase 2-1: 根据巩固分数动态调整半衰期
    if consolidation_score is not None and consolidation_score > 0:
        # 巩固分数越高，半衰期越长（记忆更持久）
        # consolidation=0.0 → multiplier=1.0（不变）
        # consolidation=1.0 → multiplier=3.0（延长3倍）
        multiplier = 1.0 + (2.0 * consolidation_score)
        hl = hl * multiplier
    
    return math.exp(-math.log(2.0) * age / hl)


def is_stale(score: float, threshold: float = DEFAULT_STALE_THRESHOLD) -> bool:
    return score <= threshold


def lifecycle_meta(
    item: dict,
    *,
    now: Optional[_dt.datetime] = None,
    stale_threshold: float = DEFAULT_STALE_THRESHOLD,
) -> dict:
    """
    计算记忆生命周期元数据
    
    Phase 2-1: 支持巩固分数调整衰减速度
    """
    updated_at = item.get("updated_at")
    mem_type = item.get("type")
    consolidation = item.get("consolidation_score", 0.0)  # Phase 2-1
    
    age = age_days(updated_at, now=now)
    decay = decay_score(
        updated_at,
        mem_type,
        now=now,
        consolidation_score=consolidation  # Phase 2-1: 传递巩固分数
    )
    
    return {
        "age_days": age,
        "decay_score": round(decay, 6),
        "is_stale": is_stale(decay, threshold=stale_threshold),
    }


def cleanup_candidates(
    items: list,
    *,
    older_than_days: Optional[float] = None,
    stale_threshold: Optional[float] = None,
    now: Optional[_dt.datetime] = None,
) -> list:
    if older_than_days is None and stale_threshold is None:
        return []
    if stale_threshold is None:
        stale_threshold = DEFAULT_STALE_THRESHOLD
    out = []
    for item in items:
        meta = lifecycle_meta(item, now=now, stale_threshold=stale_threshold)
        age = meta["age_days"]
        decay = meta["decay_score"]
        hit_age = older_than_days is not None and age is not None and age >= older_than_days
        hit_decay = decay is not None and decay <= stale_threshold
        if hit_age or hit_decay:
            item.update(meta)
            out.append(item)
    return out

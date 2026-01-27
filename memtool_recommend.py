#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation helpers
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import List, Optional, Set

from memtool_lifecycle import DEFAULT_STALE_THRESHOLD, decay_score, lifecycle_meta
from memtool_rank import confidence_score


_MIN_TOKEN_LEN = 2


def _extract_keywords(content: Optional[str]) -> Set[str]:
    if not content:
        return set()
    words = re.findall(r"[\u4e00-\u9fff]+|\w+", content.lower())
    return {w for w in words if len(w) >= _MIN_TOKEN_LEN}


def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _recommend_score(
    *,
    relevance: float,
    confidence: float,
    recency: float,
    weight: float,
) -> float:
    score = (0.5 * relevance) + (0.3 * confidence) + (0.2 * recency)
    return score * weight


def recommend_items(
    items: List[dict],
    *,
    context: Optional[str],
    limit: int = 10,
    include_stale: bool = False,
    stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    now: Optional[_dt.datetime] = None,
) -> List[dict]:
    if limit < 1:
        return []
    if now is None:
        now = _dt.datetime.now(tz=_dt.timezone.utc)

    query_keywords = _extract_keywords(context)

    ranked: List[dict] = []
    for item in items:
        meta = lifecycle_meta(item, now=now, stale_threshold=stale_threshold)
        if (not include_stale) and meta["is_stale"]:
            continue

        content = item.get("content") or ""
        relevance = _jaccard_similarity(query_keywords, _extract_keywords(content)) if query_keywords else 0.0
        conf = confidence_score(item.get("confidence_level"))
        rec = decay_score(item.get("updated_at"), item.get("type"), now=now)
        weight = float(item.get("weight", 1.0) or 1.0)
        score = _recommend_score(relevance=relevance, confidence=conf, recency=rec, weight=weight)

        reasons: List[str] = []
        if relevance >= 0.3:
            reasons.append("context_match")
        if conf >= 0.9:
            reasons.append("high_confidence")
        if rec >= 0.7:
            reasons.append("recent")
        if weight > 1.0:
            reasons.append("high_weight")

        item = dict(item)
        item.update(meta)
        item["recommend_score"] = round(score, 6)
        item["recommend_reasons"] = reasons
        if query_keywords:
            overlap = sorted(list(query_keywords & _extract_keywords(content)))[:5]
            item["recommend_keywords"] = overlap
        ranked.append(item)

    ranked.sort(key=lambda x: (x.get("recommend_score", 0.0), x.get("updated_at") or ""), reverse=True)
    return ranked[:limit]

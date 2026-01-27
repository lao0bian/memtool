#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ranking helpers: confidence / recency / mixed
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

from memtool_lifecycle import decay_score


CONFIDENCE_SCORES = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}


def confidence_score(level: Optional[str]) -> float:
    if not level:
        return CONFIDENCE_SCORES["medium"]
    return CONFIDENCE_SCORES.get(str(level).lower(), CONFIDENCE_SCORES["medium"])


def recency_score(
    *,
    updated_at: Optional[str],
    mem_type: Optional[str],
    now: Optional[_dt.datetime] = None,
) -> float:
    return decay_score(updated_at, mem_type, now=now)


def mixed_score(conf: float, recency: float, w_conf: float = 0.6, w_recency: float = 0.4) -> float:
    return (w_conf * conf) + (w_recency * recency)


def score_item(item: dict, *, now: Optional[_dt.datetime] = None) -> dict:
    conf = confidence_score(item.get("confidence_level"))
    rec = recency_score(updated_at=item.get("updated_at"), mem_type=item.get("type"), now=now)
    mixed = mixed_score(conf, rec)
    return {
        "confidence_score": round(conf, 6),
        "recency_score": round(rec, 6),
        "mixed_score": round(mixed, 6),
    }

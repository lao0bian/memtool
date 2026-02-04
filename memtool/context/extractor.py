#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情境提取器 v2：增强版，含否定识别"""
from __future__ import annotations

import datetime as dt
import re
from typing import Dict, List, Optional, Tuple


class ContextTags:
    # 时间
    TIME_WORK_HOURS = "time:work_hours"
    TIME_LATE_NIGHT = "time:late_night"
    TIME_EARLY_MORNING = "time:early_morning"
    TIME_EVENING = "time:evening"
    TIME_WEEKEND = "time:weekend"

    # 任务
    TASK_DEBUGGING = "task:debugging"
    TASK_API_DESIGN = "task:api_design"
    TASK_DATA_MODEL = "task:data_model"
    TASK_REFACTOR = "task:refactor"
    TASK_TESTING = "task:testing"
    TASK_DEPLOYMENT = "task:deployment"

    # 情绪
    EMOTION_POSITIVE = "emotion:positive"
    EMOTION_NEGATIVE = "emotion:negative"

    # 语言
    LANG_ZH = "lang:zh"
    LANG_EN = "lang:en"


class ContextExtractor:
    """自动提取记忆的上下文标签和情感效价"""

    NEGATION_WORDS = [
        "not", "no", "never", "didn't", "don't", "won't", "can't", "failed to",
        "没", "未", "不", "无法", "没有", "不能", "未能", "没搞定"
    ]

    EMOTIONAL_KEYWORDS = {
        "positive": [
            "success", "solved", "fixed", "completed", "optimized", "improved",
            "成功", "解决", "修复", "完成", "优化", "改进", "搞定", "通过"
        ],
        "negative": [
            "error", "failed", "bug", "issue", "timeout", "crash", "exception",
            "错误", "失败", "问题", "超时", "崩溃", "异常", "报错", "卡住"
        ],
    }

    URGENCY_KEYWORDS = {
        3: ["P0", "critical", "blocking", "紧急", "阻塞", "马上"],
        2: ["P1", "urgent", "asap", "重要", "优先"],
        1: ["P2", "soon", "尽快"],
    }

    TASK_KEYWORDS = {
        ContextTags.TASK_DEBUGGING: ["debug", "trace", "stack", "调试", "排查", "定位"],
        ContextTags.TASK_API_DESIGN: ["api", "endpoint", "rest", "graphql", "接口"],
        ContextTags.TASK_DATA_MODEL: ["schema", "database", "table", "migration", "数据库", "表结构"],
        ContextTags.TASK_REFACTOR: ["refactor", "cleanup", "重构", "整理", "优化结构"],
        ContextTags.TASK_TESTING: ["test", "unittest", "pytest", "测试", "用例"],
        ContextTags.TASK_DEPLOYMENT: ["deploy", "release", "docker", "k8s", "部署", "发布"],
    }

    WORK_HOURS = (9, 18)
    MIN_CONTENT_LENGTH = 10

    @classmethod
    def extract(
        cls,
        content: str,
        metadata: Optional[Dict] = None,
        timestamp: Optional[dt.datetime] = None,
    ) -> Tuple[List[str], float, int]:
        """提取上下文标签、情感效价、紧急度"""
        metadata = metadata or {}
        now = timestamp or dt.datetime.now()

        if len(content.strip()) < cls.MIN_CONTENT_LENGTH:
            return ([], 0.0, 0)

        tags: List[str] = []
        valence = 0.0
        urgency = 0

        content_lower = content.lower()

        tags.extend(cls._extract_time_context(now))

        emotion_tags, valence = cls._extract_emotion(content_lower)
        tags.extend(emotion_tags)

        urgency = cls._extract_urgency(content_lower)

        tags.extend(cls._extract_task_type(content_lower, metadata))

        if cls._is_chinese_dominant(content):
            tags.append(ContextTags.LANG_ZH)
        else:
            tags.append(ContextTags.LANG_EN)

        return list(set(tags)), max(-1.0, min(1.0, valence)), urgency

    @classmethod
    def _extract_emotion(cls, content_lower: str) -> Tuple[List[str], float]:
        tags: List[str] = []
        valence = 0.0

        has_negation = any(neg in content_lower for neg in cls.NEGATION_WORDS)

        positive_count = sum(1 for kw in cls.EMOTIONAL_KEYWORDS["positive"] if kw in content_lower)
        negative_count = sum(1 for kw in cls.EMOTIONAL_KEYWORDS["negative"] if kw in content_lower)

        if has_negation:
            positive_count, negative_count = negative_count, positive_count

        if positive_count > negative_count:
            valence = min(0.3 + 0.1 * positive_count, 1.0)
            tags.append(ContextTags.EMOTION_POSITIVE)
        elif negative_count > positive_count:
            valence = max(-0.3 - 0.1 * negative_count, -1.0)
            tags.append(ContextTags.EMOTION_NEGATIVE)

        return tags, valence

    @classmethod
    def _extract_urgency(cls, content_lower: str) -> int:
        for level, keywords in cls.URGENCY_KEYWORDS.items():
            if any(kw.lower() in content_lower for kw in keywords):
                return level
        return 0

    @classmethod
    def _extract_time_context(cls, now: dt.datetime) -> List[str]:
        tags: List[str] = []
        hour = now.hour

        if cls.WORK_HOURS[0] <= hour < cls.WORK_HOURS[1]:
            tags.append(ContextTags.TIME_WORK_HOURS)
        elif 22 <= hour or hour < 6:
            tags.append(ContextTags.TIME_LATE_NIGHT)
        elif 6 <= hour < 9:
            tags.append(ContextTags.TIME_EARLY_MORNING)
        else:
            tags.append(ContextTags.TIME_EVENING)

        if now.weekday() >= 5:
            tags.append(ContextTags.TIME_WEEKEND)

        return tags

    @classmethod
    def _extract_task_type(cls, content_lower: str, metadata: Dict) -> List[str]:
        tags: List[str] = []

        for tag, keywords in cls.TASK_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)

        return tags

    @staticmethod
    def _is_chinese_dominant(text: str) -> bool:
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        total_chars = len(re.findall(r"\S", text))
        return chinese_chars > total_chars * 0.3 if total_chars > 0 else False

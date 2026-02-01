#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utility functions for memtool
"""
from __future__ import annotations

import re
from typing import Set

_MIN_TOKEN_LEN = 2


def _extract_keywords(content: str) -> Set[str]:
    """从内容中提取关键词集合（不依赖 jieba）
    
    Args:
        content: 要提取关键词的文本内容
        
    Returns:
        关键词集合，每个关键词长度 >= 2
    """
    if not content:
        return set()

    # 简单的分词：按空格、标点符号分割
    # 保留中文、英文、数字，其他作为分隔符
    words = re.findall(r'[\u4e00-\u9fff]+|\w+', content.lower())
    # 过滤太短的词（< 2个字符）
    keywords = {w for w in words if len(w) >= _MIN_TOKEN_LEN}
    return keywords

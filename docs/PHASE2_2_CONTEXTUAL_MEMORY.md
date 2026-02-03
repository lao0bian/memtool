# Phase 2-2: 情境记忆技术方案

## 📋 设计目标

让记忆系统从"我记住了什么"升级到"我在什么情境下记住了什么"。

| 能力 | 当前 | 目标 |
|------|------|------|
| 时间感知 | ❌ 只有 `created_at` | ✅ 工作时间/深夜/周末标签 |
| 情绪标记 | ❌ 无 | ✅ 正向经验/负向教训/紧急 |
| 关联记忆 | ❌ 无 | ✅ 自动建立相关记忆链接 |
| 情境检索 | ❌ 无 | ✅ "昨晚那个 bug" 直接命中 |

---

## 🗄️ 数据库扩展

### 新增字段

```sql
-- Phase 2-2: 情境记忆字段
ALTER TABLE memory_items ADD COLUMN context_tags_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memory_items ADD COLUMN emotional_valence REAL NOT NULL DEFAULT 0.0;
ALTER TABLE memory_items ADD COLUMN related_ids_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memory_items ADD COLUMN session_id TEXT;

-- 索引
CREATE INDEX IF NOT EXISTS idx_memory_emotional ON memory_items(emotional_valence);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_items(session_id);
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `context_tags_json` | TEXT | JSON 数组，如 `["work_hours", "debugging", "urgent"]` |
| `emotional_valence` | REAL | 情感效价 -1.0 ~ +1.0（负=教训，正=成功经验） |
| `related_ids_json` | TEXT | JSON 数组，相关记忆 ID 列表 |
| `session_id` | TEXT | 来源会话标识（可追溯对话上下文） |

---

## 🧠 上下文提取器

### 文件：`memtool/context/extractor.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情境提取器：自动从记忆内容中提取上下文标签和情感效价"""

from __future__ import annotations
import datetime as dt
import re
from typing import Dict, List, Tuple, Optional

class ContextExtractor:
    """自动提取记忆的上下文标签和情感效价"""
    
    # 情绪关键词库（中英双语）
    EMOTIONAL_KEYWORDS = {
        "positive": [
            "success", "solved", "fixed", "completed", "optimized", "improved",
            "成功", "解决", "修复", "完成", "优化", "改进", "搞定", "通过"
        ],
        "negative": [
            "error", "failed", "bug", "issue", "timeout", "crash", "exception",
            "错误", "失败", "问题", "超时", "崩溃", "异常", "报错", "卡住"
        ],
        "urgent": [
            "urgent", "critical", "blocking", "asap", "immediately",
            "紧急", "关键", "阻塞", "立即", "马上", "P0"
        ],
    }
    
    # 任务类型关键词
    TASK_KEYWORDS = {
        "debugging": ["debug", "trace", "stack", "调试", "排查", "定位"],
        "api_design": ["api", "endpoint", "rest", "graphql", "接口"],
        "data_model": ["schema", "database", "table", "migration", "数据库", "表结构"],
        "refactor": ["refactor", "cleanup", "重构", "整理", "优化结构"],
        "testing": ["test", "unittest", "pytest", "测试", "用例"],
        "deployment": ["deploy", "release", "docker", "k8s", "部署", "发布"],
    }
    
    WORK_HOURS = (9, 18)  # 工作时间 9:00-18:00
    
    @classmethod
    def extract(
        cls,
        content: str,
        metadata: Optional[Dict] = None,
        timestamp: Optional[dt.datetime] = None
    ) -> Tuple[List[str], float]:
        """
        提取上下文标签和情感效价
        
        Args:
            content: 记忆内容
            metadata: 元数据（type, task_id 等）
            timestamp: 时间戳，默认当前时间
        
        Returns:
            (context_tags, emotional_valence)
        """
        metadata = metadata or {}
        now = timestamp or dt.datetime.now()
        
        tags = []
        valence = 0.0
        
        content_lower = content.lower()
        
        # 1. 时间上下文
        tags.extend(cls._extract_time_context(now))
        
        # 2. 情绪检测
        emotion_tags, valence = cls._extract_emotion(content_lower)
        tags.extend(emotion_tags)
        
        # 3. 任务类型推断
        tags.extend(cls._extract_task_type(content_lower, metadata))
        
        # 4. 语言检测
        if cls._is_chinese_dominant(content):
            tags.append("lang:zh")
        else:
            tags.append("lang:en")
        
        # 去重并返回
        return list(set(tags)), max(-1.0, min(1.0, valence))
    
    @classmethod
    def _extract_time_context(cls, now: dt.datetime) -> List[str]:
        """提取时间上下文标签"""
        tags = []
        hour = now.hour
        
        if cls.WORK_HOURS[0] <= hour < cls.WORK_HOURS[1]:
            tags.append("time:work_hours")
        elif 22 <= hour or hour < 6:
            tags.append("time:late_night")
        elif 6 <= hour < 9:
            tags.append("time:early_morning")
        else:
            tags.append("time:evening")
        
        if now.weekday() >= 5:
            tags.append("time:weekend")
        
        return tags
    
    @classmethod
    def _extract_emotion(cls, content_lower: str) -> Tuple[List[str], float]:
        """提取情绪标签和效价"""
        tags = []
        valence = 0.0
        
        positive_count = sum(1 for kw in cls.EMOTIONAL_KEYWORDS["positive"] if kw in content_lower)
        negative_count = sum(1 for kw in cls.EMOTIONAL_KEYWORDS["negative"] if kw in content_lower)
        urgent_count = sum(1 for kw in cls.EMOTIONAL_KEYWORDS["urgent"] if kw in content_lower)
        
        if positive_count > negative_count:
            valence = min(0.3 + 0.1 * positive_count, 1.0)
            tags.append("emotion:positive")
        elif negative_count > positive_count:
            valence = max(-0.3 - 0.1 * negative_count, -1.0)
            tags.append("emotion:negative")
        
        if urgent_count > 0:
            valence += 0.2  # 紧急事项更重要
            tags.append("emotion:urgent")
        
        return tags, valence
    
    @classmethod
    def _extract_task_type(cls, content_lower: str, metadata: Dict) -> List[str]:
        """提取任务类型标签"""
        tags = []
        
        for task_type, keywords in cls.TASK_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(f"task:{task_type}")
        
        # 根据 metadata.type 补充
        mem_type = metadata.get("type")
        if mem_type == "run":
            tags.append("scope:execution")
        elif mem_type == "feature":
            tags.append("scope:development")
        elif mem_type == "project":
            tags.append("scope:project")
        
        return tags
    
    @staticmethod
    def _is_chinese_dominant(text: str) -> bool:
        """判断是否以中文为主"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'\S', text))
        return chinese_chars > total_chars * 0.3 if total_chars > 0 else False
```

---

## 🔗 关联记忆建立

### 文件：`memtool/context/linker.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆关联器：自动建立相关记忆之间的链接"""

from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memtool_core import MemoryStore


class MemoryLinker:
    """自动建立记忆关联"""
    
    # 相似度阈值
    LINK_THRESHOLD = 0.4      # 低于此值才建立链接（太相似=重复）
    DUPLICATE_THRESHOLD = 0.8 # 高于此值=重复，不链接
    MAX_LINKS = 5             # 每条记忆最多关联数
    
    def __init__(self, store: "MemoryStore"):
        self.store = store
    
    def find_related(
        self,
        content: str,
        mem_type: str,
        exclude_id: Optional[str] = None,
    ) -> List[str]:
        """
        查找与给定内容相关的记忆 ID
        
        Returns:
            相关记忆 ID 列表（不包含重复项）
        """
        try:
            # 优先使用向量搜索
            results = self.store.hybrid_search(
                query=content[:500],  # 截断避免过长
                limit=self.MAX_LINKS * 2
            )
        except Exception:
            # 降级到普通搜索
            results = self.store.search(
                query=content[:200],
                limit=self.MAX_LINKS * 2
            )
        
        related_ids = []
        for item in results.get("items", []):
            item_id = item.get("id")
            similarity = item.get("similarity", item.get("score", 0.5))
            
            # 排除自己
            if item_id == exclude_id:
                continue
            
            # 太相似=重复，跳过
            if similarity > self.DUPLICATE_THRESHOLD:
                continue
            
            # 相似度适中=相关
            if similarity >= self.LINK_THRESHOLD:
                related_ids.append(item_id)
            
            if len(related_ids) >= self.MAX_LINKS:
                break
        
        return related_ids
    
    def update_bidirectional_links(
        self,
        from_id: str,
        to_ids: List[str]
    ) -> int:
        """
        建立双向链接（A→B 时也更新 B→A）
        
        Returns:
            更新的链接数
        """
        updated = 0
        
        for to_id in to_ids:
            try:
                # 获取目标记忆的现有链接
                target = self.store.get(item_id=to_id)
                if not target or not target.get("ok"):
                    continue
                
                existing_links = target.get("related_ids", [])
                
                # 如果还没有反向链接，添加它
                if from_id not in existing_links:
                    existing_links.append(from_id)
                    # 限制最大链接数
                    existing_links = existing_links[-self.MAX_LINKS:]
                    
                    self.store._update_related_ids(to_id, existing_links)
                    updated += 1
            except Exception:
                continue
        
        return updated
```

---

## 🔍 情境检索

### 新增 MCP Tool：`memory_contextual_search`

```python
@mcp.tool()
def memory_contextual_search(
    query: str,
    context_tags: Optional[List[str]] = None,
    emotional_filter: Optional[str] = None,
    time_filter: Optional[str] = None,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """情境检索：基于上下文和情绪过滤
    
    Args:
        query: 搜索关键词
        context_tags: 上下文标签过滤，如 ["debugging", "late_night"]
        emotional_filter: 情绪过滤 (positive/negative/urgent)
        time_filter: 时间过滤 (work_hours/late_night/weekend)
        limit: 返回数量限制
        db_path: 数据库路径
    
    Examples:
        - "昨晚调试的那个问题"
          → context_tags=["time:late_night", "task:debugging"]
        - "上次成功解决的类似问题"
          → emotional_filter="positive"
    """
    store = _store_for(db_path)
    
    # 1. 基础检索（多取一些用于后续过滤）
    try:
        base_results = store.hybrid_search(query=query, limit=limit * 3)
    except Exception:
        base_results = store.search(query=query, limit=limit * 3)
    
    items = base_results.get("items", [])
    
    # 2. 上下文过滤
    filtered = []
    for item in items:
        item_tags = item.get("context_tags", [])
        item_valence = item.get("emotional_valence", 0.0)
        
        # 标签匹配
        if context_tags:
            # 计算标签重叠度
            overlap = len(set(context_tags) & set(item_tags))
            if overlap == 0:
                continue
            item["context_match_score"] = overlap / len(context_tags)
        
        # 时间过滤
        if time_filter:
            time_tag = f"time:{time_filter}"
            if time_tag not in item_tags:
                continue
        
        # 情绪过滤
        if emotional_filter:
            if emotional_filter == "positive" and item_valence <= 0:
                continue
            if emotional_filter == "negative" and item_valence >= 0:
                continue
            if emotional_filter == "urgent" and "emotion:urgent" not in item_tags:
                continue
        
        filtered.append(item)
    
    # 3. 按上下文匹配度重排序
    if context_tags:
        filtered.sort(
            key=lambda x: x.get("context_match_score", 0),
            reverse=True
        )
    
    return {
        "ok": True,
        "items": filtered[:limit],
        "total_found": len(filtered),
        "filters_applied": {
            "context_tags": context_tags,
            "emotional_filter": emotional_filter,
            "time_filter": time_filter
        }
    }
```

---

## 📊 自然语言情境解析

### 新增 MCP Tool：`memory_parse_context`

```python
@mcp.tool()
def memory_parse_context(
    natural_query: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """解析自然语言查询，提取情境条件
    
    将 "昨晚调的那个 OOM" 转换为结构化查询条件
    
    Returns:
        {
            "query": "OOM",
            "context_tags": ["time:late_night"],
            "emotional_filter": null,
            "suggested_search": {...}
        }
    """
    import re
    
    query_text = natural_query
    context_tags = []
    emotional_filter = None
    
    # 时间词典
    TIME_PATTERNS = {
        r"昨晚|昨天晚上|last\s+night": "time:late_night",
        r"今早|今天早上|this\s+morning": "time:early_morning",
        r"周末|weekend": "time:weekend",
        r"上班时间|工作时间|work\s+hours?": "time:work_hours",
    }
    
    # 情绪词典
    EMOTION_PATTERNS = {
        r"成功|解决了|搞定|succeeded?|fixed": "positive",
        r"失败|没搞定|问题|failed|broken": "negative",
        r"紧急|马上|urgent|asap": "urgent",
    }
    
    # 任务词典
    TASK_PATTERNS = {
        r"调试|debug": "task:debugging",
        r"测试|test": "task:testing",
        r"部署|deploy": "task:deployment",
        r"重构|refactor": "task:refactor",
    }
    
    # 提取时间上下文
    for pattern, tag in TIME_PATTERNS.items():
        if re.search(pattern, natural_query, re.IGNORECASE):
            context_tags.append(tag)
            query_text = re.sub(pattern, "", query_text, flags=re.IGNORECASE)
    
    # 提取情绪过滤
    for pattern, emotion in EMOTION_PATTERNS.items():
        if re.search(pattern, natural_query, re.IGNORECASE):
            emotional_filter = emotion
            query_text = re.sub(pattern, "", query_text, flags=re.IGNORECASE)
            break
    
    # 提取任务类型
    for pattern, tag in TASK_PATTERNS.items():
        if re.search(pattern, natural_query, re.IGNORECASE):
            context_tags.append(tag)
    
    # 清理查询文本
    query_text = re.sub(r"[那个|的|这个|上次|之前|that|the|this]", "", query_text)
    query_text = query_text.strip()
    
    return {
        "ok": True,
        "original_query": natural_query,
        "parsed": {
            "query": query_text or natural_query,
            "context_tags": context_tags,
            "emotional_filter": emotional_filter,
        },
        "suggested_call": {
            "tool": "memory_contextual_search",
            "args": {
                "query": query_text or natural_query,
                "context_tags": context_tags if context_tags else None,
                "emotional_filter": emotional_filter,
            }
        }
    }
```

---

## 🔧 核心模块修改

### `memtool_core.py` 修改

```python
# 在 put() 方法中集成情境提取

def put(
    self,
    type: str,
    key: str,
    content: str,
    ...,
    session_id: Optional[str] = None,  # 新增参数
    auto_link: bool = True,            # 新增参数
) -> Dict[str, Any]:
    """写入记忆，自动提取情境和建立关联"""
    
    # ... 原有逻辑 ...
    
    # Phase 2-2: 情境提取
    from memtool.context.extractor import ContextExtractor
    context_tags, emotional_valence = ContextExtractor.extract(
        content=content,
        metadata={"type": type, "task_id": task_id},
    )
    
    # Phase 2-2: 自动关联
    related_ids = []
    if auto_link:
        from memtool.context.linker import MemoryLinker
        linker = MemoryLinker(self)
        related_ids = linker.find_related(
            content=content,
            mem_type=type,
            exclude_id=final_id,
        )
    
    # 更新 SQL
    conn.execute("""
        UPDATE memory_items
        SET context_tags_json = ?,
            emotional_valence = ?,
            related_ids_json = ?,
            session_id = ?
        WHERE id = ?
    """, (
        json.dumps(context_tags, ensure_ascii=False),
        emotional_valence,
        json.dumps(related_ids, ensure_ascii=False),
        session_id,
        final_id
    ))
    
    # 建立双向链接
    if auto_link and related_ids:
        linker.update_bidirectional_links(final_id, related_ids)
    
    # ... 返回结果 ...
```

---

## 📁 目录结构

```
memtool_mvp/
├── memtool/
│   ├── context/              # 新增目录
│   │   ├── __init__.py
│   │   ├── extractor.py      # 上下文提取器
│   │   └── linker.py         # 记忆关联器
│   ├── embedding/            # 已有
│   └── ...
├── mcp_server.py             # 新增 MCP tools
├── memtool_core.py           # 修改 put() 方法
└── test_phase2_2.py          # 新增测试
```

---

## ✅ 实施清单

| # | 任务 | 预估 | 优先级 |
|---|------|------|--------|
| 1 | 数据库迁移（新增 4 个字段 + 2 个索引） | 15min | P0 |
| 2 | 实现 `context/extractor.py` | 45min | P0 |
| 3 | 实现 `context/linker.py` | 30min | P0 |
| 4 | 修改 `memtool_core.py` 的 `put()` | 30min | P0 |
| 5 | 新增 MCP Tool `memory_contextual_search` | 30min | P0 |
| 6 | 新增 MCP Tool `memory_parse_context` | 20min | P1 |
| 7 | 编写 `test_phase2_2.py` 测试 | 40min | P0 |
| 8 | 更新 README 文档 | 15min | P1 |

**总计：约 3.5 小时**

---

## 🎯 验收标准

1. ✅ 新记忆自动带上时间/情绪/任务类型标签
2. ✅ 相关记忆自动建立双向链接
3. ✅ `memory_contextual_search` 能按标签和情绪过滤
4. ✅ "昨晚那个 OOM" 能正确解析并检索
5. ✅ 所有现有测试仍通过
6. ✅ 性能无明显退化（< 20ms 额外延迟）

---

## 🚨 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 情绪检测不准确 | 保守策略（只在明确时标记），支持手动覆盖 |
| 关联建立耗时 | 异步处理，使用内容截断（500字符） |
| 向后兼容问题 | 所有新字段有默认值，旧数据不受影响 |
| 向量搜索不可用 | 自动降级到 FTS5 搜索 |

---

## 📈 后续迭代

### Phase 2-2.1: 上下文增强
- 支持自定义标签（用户手动添加）
- 基于 LLM 的智能情绪分析
- 时区感知

### Phase 2-2.2: 关联增强
- 图数据库可视化
- 关联强度权重
- 跨项目关联

---

_设计者：OpusCoder_
_创建时间：2026-02-03 14:05 GMT+8_
_待评审：Codex_

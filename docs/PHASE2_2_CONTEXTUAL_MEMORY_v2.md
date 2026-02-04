# Phase 2-2: 情境记忆技术方案（v2 修订版）

> **修订说明**：基于 Codex 评审反馈，调整为分阶段实施，关联改异步，增加校准机制。

## 📋 设计目标

让记忆系统从"我记住了什么"升级到"我在什么情境下记住了什么"。

| 能力 | 当前 | Phase 2.2a | Phase 2.2b |
|------|------|------------|------------|
| 时间感知 | ❌ | ✅ 时间标签 | ✅ |
| 情绪标记 | ❌ | ✅ 效价+紧急度 | ✅ |
| 关联记忆 | ❌ | ❌ | ✅ 异步建立 |
| 情境检索 | ❌ | ✅ | ✅ |

---

## 🔄 分阶段实施

### Phase 2.2a：情境字段 + 检索（同步，~2小时）

**范围**：
- 数据库扩展（4 字段）
- ContextExtractor（增强版）
- `memory_contextual_search` MCP Tool
- `memory_parse_context` MCP Tool

**不包含**：关联记忆（移至 2.2b）

### Phase 2.2b：异步关联 + 校准（~2小时）

**范围**：
- MemoryLinker（异步队列）
- 阈值校准脚本
- 双向链接机制
- 观测指标

---

## 🗄️ 数据库扩展

### 新增字段

```sql
-- Phase 2-2: 情境记忆字段
ALTER TABLE memory_items ADD COLUMN context_tags_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memory_items ADD COLUMN emotional_valence REAL NOT NULL DEFAULT 0.0;
ALTER TABLE memory_items ADD COLUMN urgency_level INTEGER NOT NULL DEFAULT 0;  -- 🆕 独立紧急度
ALTER TABLE memory_items ADD COLUMN related_json TEXT NOT NULL DEFAULT '[]';   -- 🆕 改为带权重
ALTER TABLE memory_items ADD COLUMN session_id TEXT;

-- 索引
CREATE INDEX IF NOT EXISTS idx_memory_emotional ON memory_items(emotional_valence);
CREATE INDEX IF NOT EXISTS idx_memory_urgency ON memory_items(urgency_level);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_items(session_id);
```

### 字段说明（修订）

| 字段 | 类型 | 说明 | 修订内容 |
|------|------|------|----------|
| `context_tags_json` | TEXT | **统一格式**：`["time:xxx", "task:xxx", "lang:xxx"]` | 🆕 统一命名空间 |
| `emotional_valence` | REAL | 情感效价 -1.0 ~ +1.0 | 不变 |
| `urgency_level` | INT | 紧急度 0-3（0=普通，3=P0） | 🆕 从 valence 分离 |
| `related_json` | TEXT | `[{"id": "xxx", "score": 0.65}]` | 🆕 带权重 |
| `session_id` | TEXT | 来源会话（需文档化格式） | 🆕 增加规范 |

### session_id 规范

```
格式: <channel>:<session_key>
示例: 
  - "openclaw:main"
  - "codex:019c2298-262d-7561-b4bb-eb8db0912467"
  - "manual:cli"
```

---

## 🧠 上下文提取器（增强版）

### 文件：`memtool/context/extractor.py`

**关键改进**：
1. 统一标签命名空间
2. 否定识别
3. 紧急度独立维度
4. 短文本兜底

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情境提取器 v2：增强版，含否定识别"""

from __future__ import annotations
import datetime as dt
import re
from typing import Dict, List, Tuple, Optional

# 🆕 统一标签常量（提取器和解析器共享）
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
    
    # 🆕 否定词（中英双语）
    NEGATION_WORDS = [
        "not", "no", "never", "didn't", "don't", "won't", "can't", "failed to",
        "没", "未", "不", "无法", "没有", "不能", "未能", "没搞定"
    ]
    
    # 情绪关键词库
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
    
    # 🆕 紧急度关键词（独立维度）
    URGENCY_KEYWORDS = {
        3: ["P0", "critical", "blocking", "紧急", "阻塞", "马上"],
        2: ["P1", "urgent", "asap", "重要", "优先"],
        1: ["P2", "soon", "尽快"],
    }
    
    # 任务类型关键词
    TASK_KEYWORDS = {
        ContextTags.TASK_DEBUGGING: ["debug", "trace", "stack", "调试", "排查", "定位"],
        ContextTags.TASK_API_DESIGN: ["api", "endpoint", "rest", "graphql", "接口"],
        ContextTags.TASK_DATA_MODEL: ["schema", "database", "table", "migration", "数据库", "表结构"],
        ContextTags.TASK_REFACTOR: ["refactor", "cleanup", "重构", "整理", "优化结构"],
        ContextTags.TASK_TESTING: ["test", "unittest", "pytest", "测试", "用例"],
        ContextTags.TASK_DEPLOYMENT: ["deploy", "release", "docker", "k8s", "部署", "发布"],
    }
    
    WORK_HOURS = (9, 18)
    MIN_CONTENT_LENGTH = 10  # 🆕 短文本阈值
    
    @classmethod
    def extract(
        cls,
        content: str,
        metadata: Optional[Dict] = None,
        timestamp: Optional[dt.datetime] = None
    ) -> Tuple[List[str], float, int]:
        """
        提取上下文标签、情感效价、紧急度
        
        Returns:
            (context_tags, emotional_valence, urgency_level)
        """
        metadata = metadata or {}
        now = timestamp or dt.datetime.now()
        
        # 🆕 短文本兜底
        if len(content.strip()) < cls.MIN_CONTENT_LENGTH:
            return ([], 0.0, 0)
        
        tags = []
        valence = 0.0
        urgency = 0
        
        content_lower = content.lower()
        
        # 1. 时间上下文
        tags.extend(cls._extract_time_context(now))
        
        # 2. 情绪检测（含否定识别）
        emotion_tags, valence = cls._extract_emotion(content_lower, content)
        tags.extend(emotion_tags)
        
        # 3. 🆕 紧急度（独立维度）
        urgency = cls._extract_urgency(content_lower)
        
        # 4. 任务类型
        tags.extend(cls._extract_task_type(content_lower, metadata))
        
        # 5. 语言检测
        if cls._is_chinese_dominant(content):
            tags.append(ContextTags.LANG_ZH)
        else:
            tags.append(ContextTags.LANG_EN)
        
        return list(set(tags)), max(-1.0, min(1.0, valence)), urgency
    
    @classmethod
    def _extract_emotion(cls, content_lower: str, content_orig: str) -> Tuple[List[str], float]:
        """提取情绪标签和效价（含否定识别）"""
        tags = []
        valence = 0.0
        
        # 🆕 检测否定上下文
        has_negation = any(neg in content_lower for neg in cls.NEGATION_WORDS)
        
        positive_count = sum(1 for kw in cls.EMOTIONAL_KEYWORDS["positive"] if kw in content_lower)
        negative_count = sum(1 for kw in cls.EMOTIONAL_KEYWORDS["negative"] if kw in content_lower)
        
        # 🆕 否定翻转逻辑
        if has_negation:
            # "没解决" = negative，"没问题" = positive
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
        """🆕 提取紧急度（0-3）"""
        for level, keywords in cls.URGENCY_KEYWORDS.items():
            if any(kw.lower() in content_lower for kw in keywords):
                return level
        return 0
    
    @classmethod
    def _extract_time_context(cls, now: dt.datetime) -> List[str]:
        """提取时间上下文标签"""
        tags = []
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
        """提取任务类型标签"""
        tags = []
        
        for tag, keywords in cls.TASK_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)
        
        return tags
    
    @staticmethod
    def _is_chinese_dominant(text: str) -> bool:
        """判断是否以中文为主"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'\S', text))
        return chinese_chars > total_chars * 0.3 if total_chars > 0 else False
```

---

## 🔗 关联记忆（Phase 2.2b - 异步）

### 设计变更

| 原方案 | 修订方案 |
|--------|----------|
| `put()` 中同步建立关联 | 异步队列，后台任务 |
| `related_ids_json` = `["id1", "id2"]` | `related_json` = `[{"id": "id1", "score": 0.65}]` |
| 硬编码阈值 0.4/0.8 | 基于分布校准 |

### 文件：`memtool/context/linker.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆关联器 v2：异步建立，带权重"""

from __future__ import annotations
import json
import logging
import threading
from typing import Dict, List, Optional, TYPE_CHECKING
from queue import Queue

if TYPE_CHECKING:
    from memtool_core import MemoryStore

LOG = logging.getLogger("memtool.linker")


class MemoryLinker:
    """异步记忆关联器"""
    
    # 🆕 动态阈值（可通过校准脚本更新）
    LINK_THRESHOLD = 0.4       # 相似度下限
    DUPLICATE_THRESHOLD = 0.8  # 重复上限
    MAX_LINKS = 5
    
    def __init__(self, store: "MemoryStore"):
        self.store = store
        self._queue: Queue = Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
    
    def start_worker(self):
        """启动后台关联线程"""
        if self._worker and self._worker.is_alive():
            return
        self._running = True
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        LOG.info("Linker worker started")
    
    def stop_worker(self):
        """停止后台线程"""
        self._running = False
        self._queue.put(None)  # 唤醒线程退出
        if self._worker:
            self._worker.join(timeout=5)
    
    def enqueue(self, item_id: str, content: str, mem_type: str):
        """🆕 加入关联队列（非阻塞）"""
        self._queue.put({
            "id": item_id,
            "content": content,
            "type": mem_type
        })
    
    def _process_queue(self):
        """后台处理关联任务"""
        while self._running:
            try:
                task = self._queue.get(timeout=1)
                if task is None:
                    break
                self._build_links(task["id"], task["content"], task["type"])
            except Exception as e:
                LOG.warning(f"Linker error: {e}")
    
    def _build_links(self, item_id: str, content: str, mem_type: str):
        """建立关联（后台执行）"""
        try:
            # 搜索相关记忆
            results = self.store.hybrid_search(
                query=content[:500],
                limit=self.MAX_LINKS * 2
            )
        except Exception:
            results = self.store.search(query=content[:200], limit=self.MAX_LINKS * 2)
        
        related = []
        for item in results.get("items", []):
            other_id = item.get("id")
            score = item.get("similarity", item.get("score", 0.5))
            
            if other_id == item_id:
                continue
            if score > self.DUPLICATE_THRESHOLD:
                continue
            if score >= self.LINK_THRESHOLD:
                related.append({"id": other_id, "score": round(score, 3)})
            
            if len(related) >= self.MAX_LINKS:
                break
        
        if related:
            # 更新当前记忆的关联
            self.store._update_related(item_id, related)
            # 双向更新
            self._update_reverse_links(item_id, related)
    
    def _update_reverse_links(self, from_id: str, related: List[Dict]):
        """更新反向链接"""
        for rel in related:
            try:
                target = self.store.get(item_id=rel["id"])
                if not target or not target.get("ok"):
                    continue
                
                existing = target.get("related", [])
                # 避免重复
                if not any(r["id"] == from_id for r in existing):
                    existing.append({"id": from_id, "score": rel["score"]})
                    existing = existing[-self.MAX_LINKS:]  # 保留最新
                    self.store._update_related(rel["id"], existing)
            except Exception as e:
                LOG.debug(f"Reverse link failed: {e}")


# 🆕 阈值校准工具
def calibrate_thresholds(store: "MemoryStore", sample_size: int = 100) -> Dict:
    """
    基于现有数据校准阈值
    
    Returns:
        {"link_threshold": 0.xx, "duplicate_threshold": 0.xx}
    """
    from random import sample
    
    # 获取样本
    all_items = store.list(limit=sample_size * 2)
    items = all_items.get("items", [])[:sample_size]
    
    if len(items) < 10:
        return {"error": "Not enough data for calibration"}
    
    # 计算相似度分布
    scores = []
    for i, item in enumerate(items[:20]):  # 限制计算量
        try:
            results = store.hybrid_search(query=item["content"][:200], limit=10)
            for r in results.get("items", []):
                if r["id"] != item["id"]:
                    scores.append(r.get("similarity", r.get("score", 0.5)))
        except Exception:
            continue
    
    if not scores:
        return {"error": "No similarity scores collected"}
    
    scores.sort()
    
    # 基于分位数设置阈值
    p10 = scores[int(len(scores) * 0.1)]
    p90 = scores[int(len(scores) * 0.9)]
    
    return {
        "link_threshold": round(p10, 2),       # top 10% 作为相关
        "duplicate_threshold": round(p90, 2),  # top 10% 作为重复
        "sample_count": len(scores),
        "score_range": [round(min(scores), 2), round(max(scores), 2)]
    }
```

---

## 📊 观测指标（Phase 2.2b）

### 新增统计项

```python
# 在 memtool/observability.py 中添加

def compute_context_stats(db_path: str) -> Dict:
    """情境记忆统计"""
    conn = sqlite3.connect(db_path)
    
    stats = {}
    
    # 标签分布
    rows = conn.execute("""
        SELECT context_tags_json FROM memory_items 
        WHERE context_tags_json != '[]'
    """).fetchall()
    
    tag_counts = {}
    for row in rows:
        tags = json.loads(row[0])
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    stats["tag_distribution"] = tag_counts
    
    # 情绪分布
    stats["valence_distribution"] = {
        "positive": conn.execute("SELECT COUNT(*) FROM memory_items WHERE emotional_valence > 0").fetchone()[0],
        "neutral": conn.execute("SELECT COUNT(*) FROM memory_items WHERE emotional_valence = 0").fetchone()[0],
        "negative": conn.execute("SELECT COUNT(*) FROM memory_items WHERE emotional_valence < 0").fetchone()[0],
    }
    
    # 关联统计
    stats["linking"] = {
        "with_links": conn.execute("SELECT COUNT(*) FROM memory_items WHERE related_json != '[]'").fetchone()[0],
        "avg_links": conn.execute("SELECT AVG(json_array_length(related_json)) FROM memory_items WHERE related_json != '[]'").fetchone()[0] or 0,
    }
    
    conn.close()
    return stats
```

---

## ✅ 实施清单（修订版）

### Phase 2.2a（同步，优先）

| # | 任务 | 预估 | 优先级 |
|---|------|------|--------|
| 1 | 数据库迁移（5 个字段 + 3 个索引） | 15min | P0 |
| 2 | ContextTags 常量定义 | 10min | P0 |
| 3 | ContextExtractor v2（含否定识别） | 45min | P0 |
| 4 | 修改 `put()` 提取情境（不含关联） | 20min | P0 |
| 5 | `memory_contextual_search` MCP Tool | 30min | P0 |
| 6 | `memory_parse_context` MCP Tool | 20min | P1 |
| 7 | 测试 `test_phase2_2a.py` | 30min | P0 |

**小计：~2.5 小时**

### Phase 2.2b（异步，后续）

| # | 任务 | 预估 | 优先级 |
|---|------|------|--------|
| 1 | MemoryLinker v2（异步队列） | 45min | P0 |
| 2 | 阈值校准脚本 | 30min | P1 |
| 3 | 观测指标 `compute_context_stats` | 20min | P1 |
| 4 | 集成到 `put()` 的 enqueue 调用 | 15min | P0 |
| 5 | 测试 `test_phase2_2b.py` | 30min | P0 |

**小计：~2.5 小时**

---

## 🎯 验收标准（修订版）

### Phase 2.2a

1. ✅ 新记忆自动带上 `time:xxx`/`task:xxx`/`emotion:xxx` 标签
2. ✅ "没解决" 正确识别为 negative（否定识别）
3. ✅ `memory_contextual_search` 能按标签和情绪过滤
4. ✅ "昨晚那个 OOM" 能解析并检索
5. ✅ `put()` 写入延迟增加 < 5ms
6. ✅ 所有现有测试通过

### Phase 2.2b

1. ✅ 关联在后台异步建立
2. ✅ `related_json` 带权重 `[{id, score}]`
3. ✅ 阈值可通过 `calibrate_thresholds()` 校准
4. ✅ 观测指标可统计标签分布、情绪分布、关联率

---

## 🔄 与原方案对比

| 维度 | 原方案 | 修订方案 |
|------|--------|----------|
| 实施方式 | 一次性 | 分 2.2a + 2.2b |
| 关联建立 | 同步 `put()` | 异步队列 |
| 紧急度 | 混入 valence | 独立维度 |
| 关联格式 | `["id"]` | `[{id, score}]` |
| 标签命名 | 不统一 | `prefix:name` 统一 |
| 否定识别 | 无 | 有 |
| 阈值 | 硬编码 | 可校准 |
| 时间估算 | 3.5h（乐观） | 5h（现实） |

---

_修订者：OpusCoder + Codex_
_修订时间：2026-02-03 16:30 GMT+8_

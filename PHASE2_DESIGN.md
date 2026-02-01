# memtool Phase 2 技术设计文档

## 项目概述
**目标**: 让 memtool 的记忆管理更接近人类认知模式，提升 AI Agent 的记忆质量和检索效果

**版本**: v0.3.0  
**开发周期**: 1-2 周  
**负责人**: Codex  
**审核人**: OpusCoder  

---

## 核心设计理念

人类记忆系统的三大特征：
1. **记忆会被巩固** - 重要的、经常回忆的记忆会变得更牢固
2. **记忆有情境** - 记忆与场景、情绪、时间紧密关联
3. **有元认知** - 人类知道自己"知道"还是"不知道"

---

## Phase 2-1: 记忆巩固机制 (Memory Consolidation)

### 1.1 数据库 Schema 扩展

**新增字段**:
```sql
ALTER TABLE memory_items ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memory_items ADD COLUMN last_accessed_at TEXT;
ALTER TABLE memory_items ADD COLUMN consolidation_score REAL NOT NULL DEFAULT 0.0;
```

**索引优化**:
```sql
CREATE INDEX IF NOT EXISTS idx_memory_access_count ON memory_items(access_count);
CREATE INDEX IF NOT EXISTS idx_memory_consolidation ON memory_items(consolidation_score);
```

### 1.2 访问追踪

**实现位置**: `memtool_core.py`

```python
def _track_access(self, item_id: str) -> None:
    """记录访问，更新巩固分数"""
    conn = self._get_conn()
    now = utcnow_iso()
    
    # 更新访问记录
    conn.execute("""
        UPDATE memory_items
        SET access_count = access_count + 1,
            last_accessed_at = ?
        WHERE id = ?
    """, (now, item_id))
    
    # 重新计算巩固分数
    row = conn.execute("SELECT access_count, created_at, updated_at FROM memory_items WHERE id = ?", (item_id,)).fetchone()
    if row:
        consolidation = self._compute_consolidation(
            access_count=row["access_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
        conn.execute("UPDATE memory_items SET consolidation_score = ? WHERE id = ?", (consolidation, item_id))
    
    conn.commit()
```

**触发点**:
- `get()` 方法调用时
- `search()` 返回的每个结果
- `recommend()` 返回的每个结果

### 1.3 巩固分数算法

**公式设计**:
```python
def _compute_consolidation(
    access_count: int,
    created_at: str,
    updated_at: str,
    now: Optional[datetime] = None
) -> float:
    """
    巩固分数 = 访问频率分 × 时间权重
    
    访问频率分: log(1 + access_count) 归一化到 0-1
    时间权重: 考虑创建时间和最近更新时间
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)
    
    # 访问频率分 (对数增长，避免线性爆炸)
    frequency_score = min(math.log(1 + access_count) / math.log(100), 1.0)
    
    # 时间跨度分 (存在越久 = 越重要)
    age_days = (now - parse_dt(created_at)).total_seconds() / 86400.0
    longevity_score = min(age_days / 365.0, 1.0)
    
    # 活跃度分 (最近有更新 = 仍在使用)
    recency = decay_score(updated_at, "feature", now=now)
    
    # 综合巩固分数
    consolidation = (
        frequency_score * 0.5 +
        longevity_score * 0.2 +
        recency * 0.3
    )
    
    return consolidation
```

### 1.4 衰减曲线调整

**原逻辑**: 固定半衰期  
**新逻辑**: 根据巩固分数动态调整

```python
def adaptive_half_life(item: dict) -> float:
    """基于巩固分数动态调整半衰期"""
    base_half_life = half_life_days(item.get("type"))
    consolidation = item.get("consolidation_score", 0.0)
    
    # 巩固分数越高，半衰期越长
    # consolidation=1.0 时，半衰期 × 3
    # consolidation=0.0 时，半衰期不变
    multiplier = 1.0 + (2.0 * consolidation)
    
    return base_half_life * multiplier
```

**修改位置**:
- `memtool_lifecycle.py`: `decay_score()` 函数接受 `consolidation_score` 参数
- `memtool_core.py`: 所有调用 `decay_score` 的地方传入 `consolidation_score`

---

## Phase 2-2: 情境记忆 (Contextual Memory)

### 2.1 数据库 Schema 扩展

```sql
ALTER TABLE memory_items ADD COLUMN context_tags_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memory_items ADD COLUMN emotional_weight REAL NOT NULL DEFAULT 0.0;
ALTER TABLE memory_items ADD COLUMN related_ids_json TEXT NOT NULL DEFAULT '[]';
```

### 2.2 上下文自动提取

**实现位置**: `memtool/context_extractor.py` (新建)

```python
import datetime as dt
import re
from typing import Dict, List, Tuple

class ContextExtractor:
    """自动提取记忆的上下文标签"""
    
    # 情绪关键词库
    EMOTIONAL_KEYWORDS = {
        "positive": ["success", "解决", "完成", "优化", "改进"],
        "negative": ["error", "failed", "bug", "问题", "超时", "崩溃"],
        "urgent": ["urgent", "critical", "紧急", "立即", "blocking"],
        "calm": ["refactor", "review", "重构", "优化", "整理"],
    }
    
    # 时间上下文规则
    WORK_HOURS = (9, 18)
    
    @staticmethod
    def extract(content: str, metadata: Dict) -> Tuple[List[str], float]:
        """
        提取上下文标签和情感权重
        
        Returns:
            (context_tags, emotional_weight)
        """
        tags = []
        emotional_weight = 0.0
        
        # 1. 时间上下文
        now = dt.datetime.now()
        hour = now.hour
        
        if WORK_HOURS[0] <= hour < WORK_HOURS[1]:
            tags.append("work_hours")
        elif 22 <= hour or hour < 6:
            tags.append("late_night")
        
        if now.weekday() >= 5:
            tags.append("weekend")
        
        # 2. 情绪检测
        content_lower = content.lower()
        
        positive_count = sum(1 for kw in EMOTIONAL_KEYWORDS["positive"] if kw in content_lower)
        negative_count = sum(1 for kw in EMOTIONAL_KEYWORDS["negative"] if kw in content_lower)
        
        if positive_count > negative_count:
            emotional_weight = 0.5
            tags.append("positive")
        elif negative_count > positive_count:
            emotional_weight = -0.3
            tags.append("negative")
        
        # 3. 紧急程度
        if any(kw in content_lower for kw in EMOTIONAL_KEYWORDS["urgent"]):
            emotional_weight += 0.3
            tags.append("urgent")
        
        # 4. 任务类型推断
        if metadata.get("type") == "run":
            if "error" in content_lower or "exception" in content_lower:
                tags.append("debugging")
            if "test" in content_lower:
                tags.append("testing")
        elif metadata.get("type") == "feature":
            if "api" in content_lower or "endpoint" in content_lower:
                tags.append("api_design")
            if "db" in content_lower or "database" in content_lower:
                tags.append("data_model")
        
        # 规范化情感权重到 [-1.0, 1.0]
        emotional_weight = max(-1.0, min(1.0, emotional_weight))
        
        return tags, emotional_weight
```

### 2.3 关联记忆自动建立

**实现位置**: `memtool_core.py` 的 `put()` 方法

```python
def put(self, ..., auto_link: bool = True) -> Dict[str, Any]:
    """写入记忆，自动建立关联"""
    
    # ... 原有逻辑 ...
    
    # 自动提取上下文
    from memtool.context_extractor import ContextExtractor
    context_tags, emotional_weight = ContextExtractor.extract(
        content=content,
        metadata={"type": type, "task_id": task_id}
    )
    
    # 查找相关记忆（不是重复，而是相关）
    related_ids = []
    if auto_link:
        similar = self.find_similar_items(
            content=content,
            type=type,
            threshold=0.5,  # 更宽松的阈值
            limit=5
        )
        related_ids = [item["id"] for item in similar if item["similarity"] < 0.8]
    
    # 更新 SQL，添加新字段
    conn.execute("""
        UPDATE memory_items
        SET ...,
            context_tags_json = ?,
            emotional_weight = ?,
            related_ids_json = ?
        WHERE id = ?
    """, (
        ...,
        json.dumps(context_tags, ensure_ascii=False),
        emotional_weight,
        json.dumps(related_ids, ensure_ascii=False),
        final_id
    ))
```

### 2.4 情境检索增强

**新增 MCP Tool**: `memory_contextual_search`

```python
@mcp.tool()
def memory_contextual_search(
    query: str,
    context_tags: Optional[List[str]] = None,
    emotional_filter: Optional[str] = None,  # positive/negative/urgent
    limit: int = 10,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """情境检索：基于上下文和情绪过滤
    
    示例:
    - "找出昨晚加班时调试的那个问题"
      → context_tags=["late_night", "debugging"]
    - "上次成功解决的类似问题"
      → emotional_filter="positive"
    """
    store = _store_for(db_path)
    
    # 基础检索
    base_results = store.hybrid_search(query=query, limit=limit * 3)
    
    # 上下文过滤
    filtered = []
    for item in base_results["items"]:
        item_tags = item.get("context_tags", [])
        
        # 标签匹配
        if context_tags:
            tag_overlap = len(set(context_tags) & set(item_tags))
            if tag_overlap == 0:
                continue
            item["context_match"] = tag_overlap / len(context_tags)
        
        # 情绪过滤
        if emotional_filter:
            if emotional_filter == "positive" and item.get("emotional_weight", 0) <= 0:
                continue
            if emotional_filter == "negative" and item.get("emotional_weight", 0) >= 0:
                continue
            if emotional_filter == "urgent" and "urgent" not in item_tags:
                continue
        
        filtered.append(item)
    
    return {
        "ok": True,
        "items": filtered[:limit],
        "context_tags": context_tags,
        "emotional_filter": emotional_filter
    }
```

---

## Phase 2-3: 元认知监控 (Metacognition)

### 3.1 记忆可信度评估

**新增 MCP Tool**: `memory_assess_knowledge`

```python
@mcp.tool()
def memory_assess_knowledge(
    topic: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """评估 Agent 对某个主题的记忆可信度
    
    返回:
    - confidence: "high" | "medium" | "low" | "none"
    - message: 人类可读的评估
    - suggestions: 改进建议
    - evidence: 支撑证据列表
    """
    store = _store_for(db_path)
    
    # 搜索相关记忆
    results = store.hybrid_search(query=topic, limit=50)
    items = results.get("items", [])
    
    if not items:
        return {
            "ok": True,
            "confidence": "none",
            "score": 0.0,
            "message": f"我对「{topic}」没有记忆",
            "suggestions": [
                "建议通过 web_search 或查阅文档获取信息",
                "获取信息后用 memory_store 记录"
            ],
            "evidence": []
        }
    
    # 计算元认知分数
    avg_confidence = sum(item.get("confidence_score", 0.6) for item in items) / len(items)
    coverage = min(len(items) / 10.0, 1.0)  # 10 条记忆 = 完整覆盖
    avg_recency = sum(item.get("recency_score", 0.5) for item in items) / len(items)
    avg_consolidation = sum(item.get("consolidation_score", 0.0) for item in items) / len(items)
    
    # 综合元认知分数
    meta_score = (
        avg_confidence * 0.35 +
        coverage * 0.25 +
        avg_recency * 0.20 +
        avg_consolidation * 0.20
    )
    
    # 分级评估
    if meta_score >= 0.8:
        level = "high"
        message = f"我对「{topic}」很有把握，有 {len(items)} 条相关记忆"
        suggestions = ["可以直接使用这些记忆回答问题"]
    elif meta_score >= 0.5:
        level = "medium"
        message = f"我对「{topic}」有一些了解，但不完全确定"
        suggestions = [
            "建议交叉验证这些记忆",
            "如果是关键决策，最好查询最新资料"
        ]
    elif meta_score >= 0.2:
        level = "low"
        message = f"我对「{topic}」的记忆比较模糊"
        suggestions = [
            "记忆可能过时或不完整",
            "建议重新学习或查询外部资源",
            f"找到 {len(items)} 条记忆但质量较低"
        ]
    else:
        level = "very_low"
        message = f"我对「{topic}」几乎没有可靠记忆"
        suggestions = [
            "强烈建议查询外部资源",
            "获取新信息后更新记忆库"
        ]
    
    # 提取高质量证据
    evidence = sorted(items, key=lambda x: x.get("confidence_score", 0), reverse=True)[:5]
    
    return {
        "ok": True,
        "confidence": level,
        "score": round(meta_score, 4),
        "message": message,
        "suggestions": suggestions,
        "evidence": [
            {
                "id": e["id"],
                "key": e["key"],
                "confidence_level": e.get("confidence_level"),
                "access_count": e.get("access_count", 0),
                "updated_at": e.get("updated_at"),
                "snippet": e.get("content", "")[:100] + "..."
            }
            for e in evidence
        ],
        "stats": {
            "total_memories": len(items),
            "avg_confidence": round(avg_confidence, 3),
            "avg_recency": round(avg_recency, 3),
            "avg_consolidation": round(avg_consolidation, 3),
            "coverage": round(coverage, 3)
        }
    }
```

### 3.2 知识图谱可视化

**新增 MCP Tool**: `memory_knowledge_map`

```python
@mcp.tool()
def memory_knowledge_map(
    center_topic: str,
    depth: int = 2,
    min_score: float = 0.3,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """生成知识图谱（以某个主题为中心）
    
    返回:
    - nodes: 记忆节点列表
    - edges: 关联边列表
    - clusters: 主题聚类
    """
    store = _store_for(db_path)
    
    # 中心节点
    center_results = store.hybrid_search(query=center_topic, limit=5)
    center_items = center_results.get("items", [])
    
    if not center_items:
        return {"ok": True, "nodes": [], "edges": [], "message": "No memories found"}
    
    nodes = {}
    edges = []
    visited = set()
    
    # BFS 扩展关联记忆
    queue = [(item["id"], 0) for item in center_items]
    
    while queue:
        item_id, current_depth = queue.pop(0)
        
        if item_id in visited or current_depth >= depth:
            continue
        
        visited.add(item_id)
        
        # 获取记忆
        item = store.get(item_id=item_id)
        nodes[item_id] = {
            "id": item_id,
            "key": item["key"],
            "type": item["type"],
            "confidence": item.get("confidence_score", 0.6),
            "consolidation": item.get("consolidation_score", 0.0),
            "depth": current_depth,
            "is_center": current_depth == 0
        }
        
        # 获取关联记忆
        related_ids = item.get("related_ids", [])
        for related_id in related_ids:
            if related_id not in visited:
                queue.append((related_id, current_depth + 1))
                edges.append({
                    "from": item_id,
                    "to": related_id,
                    "type": "related"
                })
    
    return {
        "ok": True,
        "nodes": list(nodes.values()),
        "edges": edges,
        "center_topic": center_topic,
        "depth": depth,
        "total_nodes": len(nodes)
    }
```

---

## Phase 2-4: 记忆强化建议

### 4.1 主动记忆复习提醒

**新增 MCP Tool**: `memory_review_due`

```python
@mcp.tool()
def memory_review_due(
    type: Optional[str] = None,
    min_importance: float = 0.5,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """获取需要复习的记忆（间隔重复算法）
    
    类似 Anki 的复习机制：
    - 新记忆: 1天后复习
    - 复习成功: 间隔×2
    - 访问频繁: 延长间隔
    """
    store = _store_for(db_path)
    
    all_items = store.list(type=type, limit=1000, include_stale=False)
    
    now = datetime.now(tz=timezone.utc)
    review_due = []
    
    for item in all_items:
        # 计算理想复习时间
        last_accessed = parse_dt(item.get("last_accessed_at") or item["updated_at"])
        access_count = item.get("access_count", 0)
        
        # 间隔重复公式
        if access_count == 0:
            interval_days = 1
        else:
            interval_days = min(2 ** access_count, 90)  # 最多 90 天
        
        next_review = last_accessed + timedelta(days=interval_days)
        
        # 计算重要性
        importance = (
            item.get("confidence_score", 0.6) * 0.4 +
            item.get("consolidation_score", 0.0) * 0.3 +
            abs(item.get("emotional_weight", 0.0)) * 0.3
        )
        
        if importance < min_importance:
            continue
        
        if now >= next_review:
            days_overdue = (now - next_review).total_seconds() / 86400.0
            review_due.append({
                **item,
                "next_review": next_review.isoformat(),
                "days_overdue": round(days_overdue, 1),
                "importance": round(importance, 3),
                "interval_days": interval_days
            })
    
    # 按重要性和逾期天数排序
    review_due.sort(
        key=lambda x: (x["importance"] * (1 + x["days_overdue"])),
        reverse=True
    )
    
    return {
        "ok": True,
        "items": review_due[:limit],
        "total_due": len(review_due),
        "limit": limit
    }
```

---

## 实施步骤

### Step 1: 数据库迁移 (30 分钟)
- [ ] 在 `memtool_core.py` 中添加新字段的迁移函数
- [ ] 更新 `SCHEMA_SQL` 添加索引
- [ ] 运行测试确保向后兼容

### Step 2: 访问追踪 (1 小时)
- [ ] 实现 `_track_access()` 方法
- [ ] 在 `get()`, `search()`, `recommend()` 中集成
- [ ] 实现巩固分数计算
- [ ] 编写单元测试

### Step 3: 上下文提取 (1.5 小时)
- [ ] 创建 `memtool/context_extractor.py`
- [ ] 实现时间、情绪、任务类型检测
- [ ] 在 `put()` 方法中集成
- [ ] 编写单元测试

### Step 4: 新增 MCP Tools (2 小时)
- [ ] 实现 `memory_contextual_search`
- [ ] 实现 `memory_assess_knowledge`
- [ ] 实现 `memory_knowledge_map`
- [ ] 实现 `memory_review_due`
- [ ] 更新 `mcp_server.py`

### Step 5: 测试与验证 (1 小时)
- [ ] 创建 `test_phase2_advanced.py`
- [ ] 测试所有新功能
- [ ] 性能基准测试
- [ ] 更新 README 文档

---

## 预期收益

### 技术指标
- **访问追踪**: 热门记忆保留期延长 2-3 倍
- **情境检索**: 召回准确率提升 30%+
- **元认知**: 减少 AI 幻觉，明确知识边界

### 用户体验
1. **更智能的遗忘**
   - 重要记忆不会过期
   - 一次性记录自然淘汰

2. **更精准的检索**
   - "昨晚那个 bug" → 自动匹配 late_night + debugging
   - "上次解决 timeout 的方法" → 匹配 positive + timeout

3. **知道边界**
   - Agent 能说"我对此不确定，建议查询"
   - 避免基于模糊记忆做决策

---

## 风险与挑战

| 风险 | 缓解措施 |
|------|---------|
| 性能开销（访问追踪） | 异步更新，批量提交 |
| 上下文提取不准确 | 保守策略，支持手动覆盖 |
| 数据库迁移失败 | 完整备份，事务保护 |
| 向后兼容问题 | 新字段全部 DEFAULT，旧代码不受影响 |

---

## 成功标准

- [ ] 所有现有测试通过（6/6）
- [ ] 新增测试全部通过
- [ ] 访问追踪正确更新 `access_count`
- [ ] 上下文标签准确率 > 80%
- [ ] 元认知评估符合人类直觉
- [ ] 性能无明显退化（< 10% 延迟增加）

---

## 后续规划 (Phase 3)

1. **记忆整合与抽象**
   - 自动检测重复模式
   - 生成抽象记忆（"这类错误通常因为..."）

2. **主动遗忘优化**
   - 根据 consolidation_score 动态调整清理策略
   - 保留高巩固分的记忆，即使很久没更新

3. **记忆重播** (Memory Replay)
   - 定期"重播"重要记忆以增强巩固
   - 类似人类的"睡眠巩固记忆"

---

## 附录

### A. 关键算法伪代码

#### 巩固分数计算
```
consolidation_score = 
    log(1 + access_count) / log(100) * 0.5 +     # 访问频率
    min(age_days / 365, 1.0) * 0.2 +              # 长期存在
    decay_score(updated_at) * 0.3                 # 仍在活跃
```

#### 情境匹配度
```
context_match = 
    len(query_tags ∩ item_tags) / len(query_tags)
```

#### 元认知评估
```
meta_score = 
    avg(confidence_score) * 0.35 +
    min(count / 10, 1.0) * 0.25 +
    avg(recency_score) * 0.20 +
    avg(consolidation_score) * 0.20
```

### B. 依赖变更

**新增依赖**: 无（全部使用标准库和已有依赖）

**可选依赖**: 
- `matplotlib` (用于知识图谱可视化，非必需)

---

## 交付清单

1. **代码**
   - 修改: `memtool_core.py`, `mcp_server.py`, `memtool_lifecycle.py`
   - 新增: `memtool/context_extractor.py`
   - 测试: `test_phase2_advanced.py`

2. **文档**
   - 更新 `README.md` 添加新功能说明
   - 创建 `PHASE2_SUMMARY.md` 总结实施结果

3. **迁移脚本**
   - `scripts/migrate_to_phase2.py` - 数据库迁移工具

---

**审核要点**:
- 所有新字段有默认值
- 性能测试通过
- 向后兼容性验证
- 文档完整性检查

---

_此文档由 OpusCoder 设计，交付 Codex 实施_  
_创建时间: 2026-02-01 16:52 GMT+8_

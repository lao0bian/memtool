# memtool Phase 2.6 技术设计方案

> 基于 Phase 2.5 完成情况设计，修复遗留问题并新增高价值功能

## 📋 版本信息
- **版本**: 0.3.1
- **目标**: 版本历史 + 衰减统计 + Bug 修复
- **预估工作量**: 2-3 天
- **设计者**: OpusCoder
- **日期**: 2026-02-03

---

## 🎯 目标

| 优先级 | 目标 | 度量标准 | 备注 |
|--------|------|----------|------|
| P0 | 修复 vector_coverage Bug | `memory_stats` 返回正确的覆盖率 | Phase 2.5 遗留问题 |
| P1 | 记忆版本历史 | `memory_history` 可用 | 原 Phase 2.5 推迟项 |
| P2 | 衰减统计（采样） | `memory_stats` 含 stale_count | 原 Phase 2.5 推迟项 |
| P3 | 记忆合并建议 | `memory_suggest_merge` 可用 | 新增功能 |

---

## 🔧 模块设计

### 1. 修复 vector_coverage Bug (P0)

**问题分析**：
- `compute_stats()` 中检查 `store._vector_store` 时，向量库可能尚未初始化
- 需要先调用 `_init_vector_store()` 确保向量库已加载

**修改文件**: `memtool/observability.py`

```python
def compute_stats(store: "MemoryStore") -> Dict[str, Any]:
    """计算记忆库统计信息"""
    conn = store._get_conn()
    
    total = conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
    
    if total == 0:
        return {
            "total_items": 0,
            "by_type": {},
            "by_confidence": {},
            "access": {"avg_count": 0, "max_count": 0, "never_accessed": 0},
            "storage_size_mb": 0,
            "vector_coverage": 0,
            "stale_count": 0,  # P2 新增
        }
    
    # ... 其他统计 ...
    
    # 修复: 先初始化向量库再检查
    vector_coverage = 0.0
    vector_count = 0
    if hasattr(store, "_init_vector_store"):
        try:
            if store._init_vector_store():  # 确保已初始化
                vector_count = store._vector_store.count()
                vector_coverage = vector_count / total if total > 0 else 0.0
        except Exception as exc:
            logger.warning("Failed to get vector count: %s", exc)
    
    return {
        # ...
        "vector_coverage": round(vector_coverage, 3),
        "vector_count": vector_count,  # 新增：绝对数量
        # ...
    }
```

**验收标准**:
```bash
mcporter call memtool.memory_vector_sync force:true
mcporter call memtool.memory_stats
# 返回 vector_coverage: 1.0 (或接近 1.0)
```

---

### 2. 记忆版本历史 (P1)

**设计思路**：
- 新建 `memory_history` 表存储历史版本
- 在 `put()` 更新时，将旧版本写入历史表（事务内）
- 提供 `memory_history` MCP 工具查询版本历史

#### 2.1 数据库 Schema

```sql
CREATE TABLE IF NOT EXISTS memory_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    weight REAL NOT NULL DEFAULT 1.0,
    confidence_level TEXT NOT NULL DEFAULT 'medium',
    changed_at TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK (change_type IN ('update', 'delete')),
    FOREIGN KEY (item_id) REFERENCES memory_items(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_history_item_id ON memory_history(item_id);
CREATE INDEX IF NOT EXISTS idx_history_version ON memory_history(item_id, version);
```

#### 2.2 写入历史记录

**修改文件**: `memtool_core.py`

```python
def _save_history(
    conn: sqlite3.Connection,
    item_id: str,
    old_row: sqlite3.Row,
    change_type: str = "update"
) -> None:
    """将旧版本保存到历史表（在同一事务内）"""
    conn.execute("""
        INSERT INTO memory_history(
            item_id, version, content, tags_json, weight, 
            confidence_level, changed_at, change_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item_id,
        old_row["version"],
        old_row["content"],
        old_row["tags_json"],
        old_row["weight"],
        old_row.get("confidence_level", "medium"),
        utcnow_iso(),
        change_type
    ))
```

**在 `put()` 中集成**:
```python
def put(self, *, item_id, type, key, content, ...):
    conn = self._get_conn()
    # ...
    
    if item_id:
        cur = conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,))
        row = cur.fetchone()
        if row:
            # 保存旧版本到历史
            _save_history(conn, item_id, row, "update")
            # ... 执行更新 ...
    else:
        existing = find_by_logical_key(conn, type, key, task_id, step_id)
        if existing:
            # 保存旧版本到历史
            _save_history(conn, existing["id"], existing, "update")
            # ... 执行更新 ...
```

#### 2.3 查询历史 MCP 工具

**新增文件**: `memtool/history.py`

```python
"""记忆版本历史管理"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memtool_core import MemoryStore


def get_history(
    store: "MemoryStore",
    item_id: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """获取记忆的版本历史
    
    Args:
        store: MemoryStore 实例
        item_id: 记忆 ID
        limit: 最多返回多少条历史
        
    Returns:
        包含历史版本列表的字典
    """
    conn = store._get_conn()
    
    # 检查记忆是否存在
    current = conn.execute(
        "SELECT * FROM memory_items WHERE id = ?", (item_id,)
    ).fetchone()
    
    if not current:
        return {
            "ok": False,
            "error": "NOT_FOUND",
            "message": f"Memory item not found: {item_id}"
        }
    
    # 查询历史
    rows = conn.execute("""
        SELECT version, content, tags_json, weight, 
               confidence_level, changed_at, change_type
        FROM memory_history
        WHERE item_id = ?
        ORDER BY version DESC
        LIMIT ?
    """, (item_id, limit)).fetchall()
    
    history = [
        {
            "version": row[0],
            "content": row[1],
            "tags": json.loads(row[2] or "[]"),
            "weight": row[3],
            "confidence_level": row[4],
            "changed_at": row[5],
            "change_type": row[6],
        }
        for row in rows
    ]
    
    return {
        "ok": True,
        "item_id": item_id,
        "current_version": current["version"],
        "history": history,
        "history_count": len(history),
    }


def rollback_to_version(
    store: "MemoryStore",
    item_id: str,
    target_version: int,
) -> Dict[str, Any]:
    """回滚到指定版本（可选功能，Phase 2.7）"""
    # TODO: 实现版本回滚
    return {"ok": False, "error": "NOT_IMPLEMENTED"}
```

#### 2.4 MCP 工具注册

**修改文件**: `mcp_server.py`

```python
from memtool.history import get_history

@mcp.tool()
def memory_history(
    item_id: str,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """查看记忆的版本历史
    
    Args:
        item_id: 记忆的 ID
        limit: 最多返回多少条历史记录（默认 10）
        db_path: 可选的数据库路径
        
    Returns:
        包含版本历史的字典
        
    Example:
        memory_history(item_id="abc123")
        → {"ok": True, "history": [{"version": 2, "content": "..."}]}
    """
    if not item_id or not str(item_id).strip():
        return _param_error("item_id cannot be empty")
    
    try:
        store = _store_for(db_path)
        return get_history(store, str(item_id).strip(), limit=limit)
    except MemtoolError as e:
        return e.payload
    except Exception as e:
        return _unexpected_error("memory_history", e)
```

**验收标准**:
```bash
# 先存储一条记忆
mcporter call memtool.memory_store type:project key:test content:"v1"
# 更新它
mcporter call memtool.memory_store type:project key:test content:"v2"
# 查看历史
mcporter call memtool.memory_history item_id:"<id>"
# 返回 history: [{version: 1, content: "v1", ...}]
```

---

### 3. 衰减统计（采样策略）(P2)

**设计思路**：
- 避免 O(n) 全表扫描
- 使用采样策略：随机采样 100-200 条计算衰减统计
- 提供估算值而非精确值

**修改文件**: `memtool/observability.py`

```python
import random
from memtool_lifecycle import decay_score

SAMPLE_SIZE = 200  # 采样数量


def _compute_decay_stats_sampled(
    store: "MemoryStore",
    sample_size: int = SAMPLE_SIZE,
) -> Dict[str, Any]:
    """采样计算衰减统计（避免 O(n) 遍历）
    
    Returns:
        stale_ratio: 估算的过期比例
        avg_decay_score: 平均衰减分数
        sampled: 采样数量
    """
    conn = store._get_conn()
    
    # 获取总数
    total = conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
    if total == 0:
        return {"stale_ratio": 0.0, "avg_decay_score": 1.0, "sampled": 0}
    
    # 采样策略：使用 RANDOM() 随机采样
    actual_sample = min(sample_size, total)
    rows = conn.execute(f"""
        SELECT id, type, updated_at, consolidation_score
        FROM memory_items
        ORDER BY RANDOM()
        LIMIT {actual_sample}
    """).fetchall()
    
    if not rows:
        return {"stale_ratio": 0.0, "avg_decay_score": 1.0, "sampled": 0}
    
    import datetime as dt
    now = dt.datetime.now(tz=dt.timezone.utc)
    
    stale_count = 0
    total_decay = 0.0
    
    for row in rows:
        item_type = row["type"] or "feature"
        updated_at = row["updated_at"]
        consolidation = row["consolidation_score"] or 0.0
        
        # 计算衰减分数
        d_score = decay_score(updated_at, item_type, now=now)
        
        # 根据巩固分数调整（巩固分高的不易过期）
        # 巩固分 1.0 → 阈值降为 0.1
        # 巩固分 0.0 → 阈值保持 0.3
        adjusted_threshold = 0.3 - (0.2 * consolidation)
        
        if d_score < adjusted_threshold:
            stale_count += 1
        
        total_decay += d_score
    
    stale_ratio = stale_count / len(rows)
    avg_decay = total_decay / len(rows)
    
    # 根据采样率推算总数
    estimated_stale_count = int(stale_ratio * total)
    
    return {
        "stale_ratio": round(stale_ratio, 3),
        "stale_count_estimated": estimated_stale_count,
        "avg_decay_score": round(avg_decay, 3),
        "sampled": len(rows),
        "total": total,
    }


def compute_stats(store: "MemoryStore") -> Dict[str, Any]:
    """计算记忆库统计信息（含采样衰减统计）"""
    # ... 原有代码 ...
    
    # P2: 添加衰减统计（采样）
    decay_stats = _compute_decay_stats_sampled(store)
    
    return {
        "total_items": total,
        "by_type": type_dist,
        "by_confidence": confidence_dist,
        "access": {
            "avg_count": round(access_row[0], 2),
            "max_count": access_row[1],
            "never_accessed": access_row[2] or 0,
        },
        "storage_size_mb": round(storage_size_bytes / 1024 / 1024, 2),
        "vector_coverage": round(vector_coverage, 3),
        "vector_count": vector_count,
        # P2 新增
        "decay": decay_stats,
    }
```

**验收标准**:
```bash
mcporter call memtool.memory_stats
# 返回 decay: {stale_ratio: 0.x, stale_count_estimated: N, ...}
```

---

### 4. 记忆合并建议 (P3)

**设计思路**：
- 利用已有的 `find_similar_items()` 方法
- 扫描记忆库，找出高相似度的记忆对
- 提供合并建议（不自动合并，需用户确认）

**新增文件**: `memtool/merge.py`

```python
"""记忆合并建议"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memtool_core import MemoryStore


def suggest_merges(
    store: "MemoryStore",
    type: Optional[str] = None,
    threshold: float = 0.85,
    limit: int = 10,
) -> Dict[str, Any]:
    """找出可能需要合并的相似记忆
    
    Args:
        store: MemoryStore 实例
        type: 可选，仅在该 type 中搜索
        threshold: 相似度阈值（默认 85%）
        limit: 最多返回多少组建议
        
    Returns:
        合并建议列表
    """
    conn = store._get_conn()
    
    # 获取候选记忆
    where = []
    params = []
    if type:
        where.append("type = ?")
        params.append(type)
    
    where_clause = f" WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT id, type, key, content, updated_at FROM memory_items{where_clause} ORDER BY updated_at DESC LIMIT 500"
    
    rows = conn.execute(sql, tuple(params)).fetchall()
    
    if len(rows) < 2:
        return {
            "ok": True,
            "suggestions": [],
            "message": "记忆数量不足，无需合并"
        }
    
    # 找出相似对
    suggestions = []
    checked = set()
    
    for row in rows:
        if row["id"] in checked:
            continue
        
        similar = store.find_similar_items(
            content=row["content"],
            type=row["type"],
            threshold=threshold,
            limit=5
        )
        
        # 过滤掉自己
        similar = [s for s in similar if s["id"] != row["id"]]
        
        if similar:
            for s in similar:
                checked.add(s["id"])
            
            suggestions.append({
                "primary": {
                    "id": row["id"],
                    "key": row["key"],
                    "type": row["type"],
                    "updated_at": row["updated_at"],
                    "content_preview": row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"]
                },
                "similar": [
                    {
                        "id": s["id"],
                        "key": s["key"],
                        "similarity": s["similarity"],
                        "updated_at": s["updated_at"],
                        "content_preview": s["content"][:100] + "..." if len(s["content"]) > 100 else s["content"]
                    }
                    for s in similar
                ],
                "action_hint": f"可用 memory_delete 删除重复项，或用 memory_store 合并内容"
            })
            
            checked.add(row["id"])
        
        if len(suggestions) >= limit:
            break
    
    return {
        "ok": True,
        "suggestions": suggestions,
        "total_suggestions": len(suggestions),
        "threshold": threshold,
    }
```

**MCP 工具注册**:
```python
@mcp.tool()
def memory_suggest_merge(
    type: Optional[str] = None,
    threshold: float = 0.85,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """找出可能需要合并的相似记忆
    
    扫描记忆库，找出高相似度的记忆对，建议合并以减少冗余。
    
    Args:
        type: 可选，仅在该 type 中搜索
        threshold: 相似度阈值（默认 0.85，即 85%）
        limit: 最多返回多少组建议（默认 10）
        db_path: 可选的数据库路径
        
    Returns:
        合并建议列表
        
    Example:
        memory_suggest_merge(type="project", threshold=0.9)
    """
    from memtool.merge import suggest_merges
    
    try:
        store = _store_for(db_path)
        return suggest_merges(
            store,
            type=type,
            threshold=threshold,
            limit=limit
        )
    except MemtoolError as e:
        return e.payload
    except Exception as e:
        return _unexpected_error("memory_suggest_merge", e)
```

**验收标准**:
```bash
mcporter call memtool.memory_suggest_merge threshold:0.8
# 返回 suggestions: [{primary: {...}, similar: [...]}]
```

---

## 📁 文件变更清单

| 文件 | 变更类型 | 描述 |
|------|----------|------|
| `memtool/observability.py` | 修改 | 修复 vector_coverage + 添加衰减统计 |
| `memtool/history.py` | **新增** | 版本历史查询逻辑 |
| `memtool/merge.py` | **新增** | 合并建议逻辑 |
| `memtool_core.py` | 修改 | 历史表 Schema + _save_history + put 集成 |
| `mcp_server.py` | 修改 | 新增 2 个 MCP 工具 |

---

## 📊 新增 MCP 工具

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `memory_history` | 查看记忆版本历史 | item_id, limit?, db_path? |
| `memory_suggest_merge` | 找出相似记忆并建议合并 | type?, threshold?, limit?, db_path? |

---

## 🗂️ 数据库迁移

**迁移函数** (添加到 `memtool_core.py`):

```python
def _ensure_history_table(conn: sqlite3.Connection) -> bool:
    """确保 memory_history 表存在"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            weight REAL NOT NULL DEFAULT 1.0,
            confidence_level TEXT NOT NULL DEFAULT 'medium',
            changed_at TEXT NOT NULL,
            change_type TEXT NOT NULL CHECK (change_type IN ('update', 'delete'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_item_id ON memory_history(item_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_version ON memory_history(item_id, version)")
    return True
```

在 `_ensure_schema()` 中调用:
```python
def _ensure_schema(conn: sqlite3.Connection) -> bool:
    conn.executescript(SCHEMA_SQL)
    # ... 其他 ensure ...
    _ensure_history_table(conn)  # Phase 2.6: 历史表
    # ...
```

---

## ✅ 验收标准

### P0: vector_coverage 修复
```bash
mcporter call memtool.memory_vector_sync force:true
mcporter call memtool.memory_stats
# vector_coverage 应为 1.0（或接近）
```

### P1: 版本历史
```bash
mcporter call memtool.memory_store type:project key:test content:"v1"
mcporter call memtool.memory_store type:project key:test content:"v2"
mcporter call memtool.memory_history item_id:"<返回的id>"
# 返回 history: [{version: 1, content: "v1", ...}]
```

### P2: 衰减统计
```bash
mcporter call memtool.memory_stats
# 返回 decay: {stale_ratio: X, stale_count_estimated: N, ...}
```

### P3: 合并建议
```bash
mcporter call memtool.memory_suggest_merge threshold:0.8
# 返回 suggestions 列表
```

---

## 🚀 实施步骤

### Day 1: P0 + P1
1. 修复 `observability.py` 的 vector_coverage bug
2. 添加 `memory_history` 表和迁移
3. 实现 `_save_history()` 和 `put()` 集成
4. 实现 `memory_history` MCP 工具
5. 单元测试

### Day 2: P2 + P3
1. 实现采样衰减统计
2. 更新 `compute_stats()` 返回结构
3. 实现 `suggest_merges()` 逻辑
4. 实现 `memory_suggest_merge` MCP 工具
5. 集成测试

### Day 3: 收尾
1. 端到端测试
2. 更新 README 文档
3. 发布 0.3.1

---

## 📝 Phase 2.7 预览 (后续)

- **记忆回滚**: `memory_rollback(item_id, version)` - 回滚到指定版本
- **智能清理**: 根据衰减+巩固分数自动建议清理
- **记忆导出/导入**: JSON 格式备份恢复
- **向量索引增量同步**: 仅同步新增/修改的记忆

---

## ⚠️ 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 历史表增长过快 | 可配置历史保留策略（默认保留最近 N 个版本） |
| 采样统计不准确 | 采样数 200 条，误差可控在 5% 以内 |
| 合并建议误报 | 使用高阈值（85%），仅建议不自动执行 |
| 向量库初始化开销 | Lazy init，首次访问才加载 |

---

_此文档由 OpusCoder 设计_
_创建时间: 2026-02-03 11:20 GMT+8_

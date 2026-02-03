# memtool Phase 2.5 技术设计方案 (v2)

> 根据 Codex CR 反馈修订

## 📋 版本信息
- **版本**: 0.3.0
- **目标**: 稳定化 + 可观测性 + 向量搜索修复
- **预估工作量**: 2-3 天 (采用保守策略)

---

## 🎯 目标 (修订后)

| 优先级 | 目标 | 度量标准 | 备注 |
|--------|------|----------|------|
| P0 | 修复向量搜索 | `memory_semantic_search` 可用 | 本次必须完成 |
| P1 | 基础可观测性 | `memory_stats` 可用 | 简化版,不含衰减统计 |
| P2 | 健康检查 | `memory_health_check` 可用 | 阈值可配置 |
| **P3** | 记忆版本历史 | `memory_history` 可用 | **推迟到 Phase 2.6** |

**策略调整**: 采用 Codex 建议的保守方案,先稳定向量搜索,再逐步引入复杂功能。

---

## 🔧 模块设计

### 1. ChromaDB 迁移 (P0) ✅ 改进

**修改文件**: `memtool/embedding/vector_store.py`

```python
from packaging import version as pkg_version

def _ensure_client(self):
    """Lazy initialization with version-safe API selection"""
    if self._client is not None:
        return
    
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "chromadb is required for vector search. "
            "Install with: pip install chromadb"
        )
    
    self._persist_dir.mkdir(parents=True, exist_ok=True)
    
    # 健壮的版本解析 (处理 rc/alpha/beta 版本)
    try:
        chroma_ver = pkg_version.parse(chromadb.__version__)
        use_new_api = chroma_ver >= pkg_version.parse("0.4.0")
    except Exception:
        # 解析失败时默认使用新 API
        use_new_api = True
        logger.warning(f"Failed to parse ChromaDB version: {chromadb.__version__}, assuming >= 0.4")
    
    if use_new_api:
        self._init_persistent_client()
    else:
        self._init_legacy_client()


def _init_persistent_client(self):
    """ChromaDB 0.4+ API"""
    import chromadb
    from chromadb.config import Settings
    
    self._client = chromadb.PersistentClient(
        path=str(self._persist_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    self._collection = self._client.get_or_create_collection(
        name=self._collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Initialized ChromaDB (new API): {self._persist_dir}")


def _init_legacy_client(self):
    """ChromaDB < 0.4 API (deprecated)"""
    import chromadb
    from chromadb.config import Settings
    
    self._client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(self._persist_dir),
        anonymized_telemetry=False
    ))
    self._collection = self._client.get_or_create_collection(
        name=self._collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Initialized ChromaDB (legacy API): {self._persist_dir}")
```

**新增依赖**: `packaging` (用于安全的版本解析)

---

### 2. memory_stats 工具 (P1) ✅ 简化版

**新增文件**: `memtool/observability.py`

```python
"""
Observability module for memtool
Phase 2.5: 基础统计 (不含衰减统计,避免 O(n) 遍历)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from memtool_core import MemoryStore

logger = logging.getLogger(__name__)


def compute_stats(store: "MemoryStore") -> Dict[str, Any]:
    """计算记忆库统计信息 (轻量版)
    
    Phase 2.5: 仅基础 COUNT/分布统计,不含衰减计算
    Phase 2.6: 将增加采样衰减统计
    """
    conn = store._get_conn()
    
    # 基础统计 (单次查询)
    total = conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
    
    if total == 0:
        return {
            "total_items": 0,
            "by_type": {},
            "by_confidence": {},
            "access": {"avg_count": 0, "max_count": 0, "never_accessed": 0},
            "storage_size_mb": 0,
            "vector_coverage": 0,
        }
    
    # 按类型分布
    type_rows = conn.execute(
        "SELECT type, COUNT(*) FROM memory_items GROUP BY type"
    ).fetchall()
    type_dist = {row[0] or "unknown": row[1] for row in type_rows}
    
    # 按置信度分布
    conf_rows = conn.execute(
        "SELECT confidence_level, COUNT(*) FROM memory_items GROUP BY confidence_level"
    ).fetchall()
    confidence_dist = {row[0] or "unknown": row[1] for row in conf_rows}
    
    # 访问统计 (单次聚合查询)
    access_row = conn.execute("""
        SELECT 
            COALESCE(AVG(access_count), 0) as avg_access,
            COALESCE(MAX(access_count), 0) as max_access,
            SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as never_accessed
        FROM memory_items
    """).fetchone()
    
    # 向量覆盖率
    vector_coverage = 0.0
    if hasattr(store, '_vector_store') and store._vector_store:
        try:
            vector_count = store._vector_store.count()
            vector_coverage = vector_count / total if total > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to get vector count: {e}")
    
    # 存储大小
    storage_size_bytes = 0
    try:
        db_path = store._db_path
        if os.path.exists(db_path):
            storage_size_bytes = os.path.getsize(db_path)
    except Exception:
        pass
    
    return {
        "total_items": total,
        "by_type": type_dist,
        "by_confidence": confidence_dist,
        "access": {
            "avg_count": round(access_row[0], 2),
            "max_count": access_row[1],
            "never_accessed": access_row[2] or 0
        },
        "storage_size_mb": round(storage_size_bytes / 1024 / 1024, 2),
        "vector_coverage": round(vector_coverage, 3),
    }
```

---

### 3. memory_health_check 工具 (P2) ✅ 阈值可配置

```python
# 默认阈值 (可通过配置覆盖)
DEFAULT_HEALTH_THRESHOLDS = {
    "stale_ratio_warning": 0.3,      # 过期比例警告阈值
    "never_accessed_warning": 0.5,   # 从未访问比例警告阈值
    "vector_coverage_warning": 0.9,  # 向量覆盖率警告阈值
    "min_items_for_vector_check": 10 # 最小记录数才检查向量覆盖
}


def health_check(
    store: "MemoryStore",
    thresholds: Dict[str, float] | None = None
) -> Dict[str, Any]:
    """检查记忆库健康状态
    
    Args:
        store: MemoryStore 实例
        thresholds: 可选的阈值覆盖
    """
    # 合并阈值
    th = {**DEFAULT_HEALTH_THRESHOLDS, **(thresholds or {})}
    
    issues = []
    recommendations = []
    
    stats = compute_stats(store)
    total = stats["total_items"]
    
    if total == 0:
        return {
            "ok": True,
            "status": "empty",
            "message": "记忆库为空",
            "issues": [],
            "recommendations": ["使用 memory_store 添加第一条记忆"],
            "stats": stats
        }
    
    # 检查从未访问的记忆
    never_accessed = stats["access"]["never_accessed"]
    never_accessed_ratio = never_accessed / total
    if never_accessed_ratio > th["never_accessed_warning"]:
        issues.append({
            "type": "low_usage",
            "severity": "info",
            "message": f"{never_accessed} 条记忆从未被访问 ({never_accessed_ratio*100:.1f}%)"
        })
    
    # 检查向量覆盖率
    if total >= th["min_items_for_vector_check"]:
        if stats["vector_coverage"] < th["vector_coverage_warning"]:
            issues.append({
                "type": "incomplete_vector_index",
                "severity": "warning",
                "message": f"向量索引覆盖率 {stats['vector_coverage']*100:.1f}%"
            })
            recommendations.append("运行 memory_vector_sync(force=True) 重建向量索引")
    
    # 确定整体状态
    severity_scores = {"critical": 3, "warning": 2, "info": 1}
    max_severity = max(
        [severity_scores.get(i["severity"], 0) for i in issues],
        default=0
    )
    
    if max_severity >= 3:
        status = "unhealthy"
        ok = False
    elif max_severity >= 2:
        status = "degraded"
        ok = True  # degraded 仍然 ok,只是有警告
    else:
        status = "healthy"
        ok = True
    
    return {
        "ok": ok,
        "status": status,
        "issues": issues,
        "recommendations": recommendations,
        "stats": stats,
        "thresholds_used": th  # 返回使用的阈值,便于调试
    }
```

---

### 4. 性能优化 (P2) ✅ 改进错误处理

```python
def _track_access_batch(self, item_ids: List[str]) -> bool:
    """批量更新访问记录
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not item_ids:
        return True
    
    try:
        conn = self._get_conn()
        now = utcnow_iso()
        
        placeholders = ",".join("?" * len(item_ids))
        conn.execute(f"""
            UPDATE memory_items
            SET access_count = access_count + 1,
                last_accessed_at = ?
            WHERE id IN ({placeholders})
        """, [now] + item_ids)
        
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        # 不吞错误,记录日志
        logger.warning(f"Failed to track access for {len(item_ids)} items: {e}")
        return False
```

---

### 5. 类变量修复 (P2) ✅

**修改文件**: `memtool/embedding/semantic.py`

```python
class SemanticSearchMixin:
    """Mixin for semantic search capabilities
    
    注意: 所有状态变量都是实例变量,不是类变量
    """
    
    def _init_vector_attrs(self) -> None:
        """Initialize vector store attributes
        
        必须在 __init__ 中调用此方法
        """
        self._vector_store: Optional[VectorStore] = None
        self._vector_lock: threading.Lock = threading.Lock()
        self._vector_initialized: bool = False
```

在 `MemoryStore.__init__` 中调用:
```python
def __init__(self, db_path: str) -> None:
    self._db_path = db_path
    self._pool = SQLiteConnectionPool(db_path)
    self._init_vector_attrs()  # 初始化向量相关实例变量
```

---

## 📁 文件变更清单 (修订后)

| 文件 | 变更类型 | 描述 |
|------|----------|------|
| `memtool/embedding/vector_store.py` | 修改 | ChromaDB API 迁移 + 版本安全解析 |
| `memtool/embedding/semantic.py` | 修改 | 类变量→实例变量 |
| `memtool/observability.py` | **新增** | stats + health_check (简化版) |
| `memtool_core.py` | 修改 | 批量追踪 + 错误日志 |
| `mcp_server.py` | 修改 | 新增 2 个 MCP 工具 |
| `pyproject.toml` | 修改 | 添加 packaging 依赖 |

---

## 📊 新增 MCP 工具 (修订后)

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `memory_stats` | 获取统计信息 | db_path? |
| `memory_health_check` | 健康检查 | db_path?, thresholds? |

**推迟到 Phase 2.6**:
- `memory_history` (需要完整的事务写入策略)
- 衰减统计 (需要采样策略)

---

## ✅ 验收标准 (修订后)

### P0: 向量搜索
```bash
mcporter call memtool.memory_semantic_search query:"测试"
# 返回 ok: true, items: [...]
```

### P1: 统计信息
```bash
mcporter call memtool.memory_stats
# 返回 total_items, by_type, access, vector_coverage
```

### P2: 健康检查
```bash
mcporter call memtool.memory_health_check
# 返回 status: healthy/degraded/unhealthy, issues, recommendations
```

---

## 🚀 实施步骤

1. **Day 1**: ChromaDB 迁移 + 测试
   - 实现版本安全解析
   - 测试新旧 API 兼容
   - 验证 semantic_search 可用

2. **Day 2**: 可观测性
   - 实现 `observability.py`
   - 注册 MCP 工具
   - 修复类变量问题
   - 改进错误日志

3. **Day 3**: 测试 + 文档
   - 集成测试
   - 更新 README
   - 发布 0.3.0

---

## 📝 Phase 2.6 预览 (后续)

- `memory_history`: 版本历史 (需设计事务写入)
- 衰减统计: 采样策略 (避免 O(n) 全表扫描)
- 记忆合并: 相似记忆自动合并建议

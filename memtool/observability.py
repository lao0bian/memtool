"""
Observability module for memtool
Phase 2.5: 基础统计 (不含衰减统计,避免 O(n) 遍历)
Phase 2.6: 添加采样衰减统计
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from memtool_core import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_THRESHOLDS = {
    "stale_ratio_warning": 0.3,
    "never_accessed_warning": 0.5,
    "vector_coverage_warning": 0.9,
    "min_items_for_vector_check": 10,
}

SAMPLE_SIZE = 200  # Phase 2.6: 采样数量


def _compute_decay_stats_sampled(
    store: "MemoryStore",
    sample_size: int = SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Phase 2.6: 采样计算衰减统计（避免 O(n) 遍历）
    
    Returns:
        stale_ratio: 估算的过期比例
        stale_count_estimated: 估算的过期记忆数量
        avg_decay_score: 平均衰减分数
        sampled: 采样数量
        total: 总记忆数量
    """
    conn = store._get_conn()
    
    # 获取总数
    total = conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
    if total == 0:
        return {
            "stale_ratio": 0.0,
            "stale_count_estimated": 0,
            "avg_decay_score": 1.0,
            "sampled": 0,
            "total": 0,
        }
    
    # 采样策略：使用 RANDOM() 随机采样
    actual_sample = min(sample_size, total)
    rows = conn.execute(f"""
        SELECT id, type, updated_at, consolidation_score
        FROM memory_items
        ORDER BY RANDOM()
        LIMIT {actual_sample}
    """).fetchall()
    
    if not rows:
        return {
            "stale_ratio": 0.0,
            "stale_count_estimated": 0,
            "avg_decay_score": 1.0,
            "sampled": 0,
            "total": total,
        }
    
    import datetime as dt
    from memtool_lifecycle import decay_score
    
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
    """计算记忆库统计信息 (含采样衰减统计)"""
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
            "vector_count": 0,
            "decay": {
                "stale_ratio": 0.0,
                "stale_count_estimated": 0,
                "avg_decay_score": 1.0,
                "sampled": 0,
                "total": 0,
            },
        }

    type_rows = conn.execute(
        "SELECT type, COUNT(*) FROM memory_items GROUP BY type"
    ).fetchall()
    type_dist = {row[0] or "unknown": row[1] for row in type_rows}

    conf_rows = conn.execute(
        "SELECT confidence_level, COUNT(*) FROM memory_items GROUP BY confidence_level"
    ).fetchall()
    confidence_dist = {row[0] or "unknown": row[1] for row in conf_rows}

    access_row = conn.execute(
        """
        SELECT
            COALESCE(AVG(access_count), 0) as avg_access,
            COALESCE(MAX(access_count), 0) as max_access,
            SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as never_accessed
        FROM memory_items
        """
    ).fetchone()

    # P0: Fix vector_coverage bug - ensure vector store is initialized first
    vector_coverage = 0.0
    vector_count = 0
    if hasattr(store, "_init_vector_store"):
        try:
            if store._init_vector_store():  # Ensure vector store is initialized
                vector_count = store._vector_store.count()
                vector_coverage = vector_count / total if total > 0 else 0.0
        except Exception as exc:
            logger.warning("Failed to get vector count: %s", exc)

    storage_size_bytes = 0
    try:
        db_path = store._db_path
        if os.path.exists(db_path):
            storage_size_bytes = os.path.getsize(db_path)
    except Exception:
        pass

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


def health_check(
    store: "MemoryStore",
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """检查记忆库健康状态"""
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
            "stats": stats,
        }

    never_accessed = stats["access"]["never_accessed"]
    never_accessed_ratio = never_accessed / total
    if never_accessed_ratio > th["never_accessed_warning"]:
        issues.append(
            {
                "type": "low_usage",
                "severity": "info",
                "message": f"{never_accessed} 条记忆从未被访问 ({never_accessed_ratio*100:.1f}%)",
            }
        )

    if total >= th["min_items_for_vector_check"]:
        if stats["vector_coverage"] < th["vector_coverage_warning"]:
            issues.append(
                {
                    "type": "incomplete_vector_index",
                    "severity": "warning",
                    "message": f"向量索引覆盖率 {stats['vector_coverage']*100:.1f}%",
                }
            )
            recommendations.append("运行 memory_vector_sync(force=True) 重建向量索引")

    severity_scores = {"critical": 3, "warning": 2, "info": 1}
    max_severity = max(
        [severity_scores.get(i["severity"], 0) for i in issues],
        default=0,
    )

    if max_severity >= 3:
        status = "unhealthy"
        ok = False
    elif max_severity >= 2:
        status = "degraded"
        ok = True
    else:
        status = "healthy"
        ok = True

    return {
        "ok": ok,
        "status": status,
        "issues": issues,
        "recommendations": recommendations,
        "stats": stats,
        "thresholds_used": th,
    }

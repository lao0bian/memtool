"""记忆版本历史管理 - Phase 2.6"""
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

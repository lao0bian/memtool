"""记忆合并建议 - Phase 2.6"""
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

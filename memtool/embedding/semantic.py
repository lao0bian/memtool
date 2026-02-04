#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic search integration for memtool

Provides vector-based semantic search as an enhancement to the existing
FTS5 keyword search. Supports automatic sync with SQLite database.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 核心字段（与 memtool_core.py 保持一致）
CORE_FIELDS = [
    'id', 'type', 'key', 'content', 'tags',
    'created_at', 'updated_at', 'version',
    'access_count', 'last_accessed_at',
    'confidence_level', 'verified_by'
]


def _filter_core_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """过滤只返回核心字段（瘦身后使用）"""
    return {k: v for k, v in item.items() if k in CORE_FIELDS}


# Environment variables
MEMTOOL_VECTOR_ENABLED = os.environ.get("MEMTOOL_VECTOR_ENABLED", "auto").lower()
MEMTOOL_VECTOR_DIR = os.environ.get("MEMTOOL_VECTOR_DIR", "")
MEMTOOL_EMBEDDING_PROVIDER = os.environ.get("MEMTOOL_EMBEDDING_PROVIDER", "local")
MEMTOOL_EMBEDDING_MODEL = os.environ.get("MEMTOOL_EMBEDDING_MODEL", "")


class SemanticSearchMixin:
    """Mixin class to add semantic search to MemoryStore
    
    Add to MemoryStore inheritance to enable vector search:
    
        class MemoryStore(SemanticSearchMixin):
            ...
    
    Or use standalone:
    
        semantic = SemanticSearch(db_path="./memtool.db")
        results = semantic.semantic_search("find errors", limit=10)
    """

    def _init_vector_attrs(self) -> None:
        """Initialize vector store attributes (instance-scoped)."""
        self._vector_store: Optional["VectorStore"] = None
        self._vector_lock = threading.Lock()
        self._vector_initialized = False
    
    def _get_vector_dir(self) -> Path:
        """Get vector storage directory"""
        if MEMTOOL_VECTOR_DIR:
            return Path(MEMTOOL_VECTOR_DIR)
        # Default: alongside the SQLite database
        db_path = getattr(self, "_db_path", "./memtool.db")
        return Path(db_path).parent / "vectors"
    
    def _init_vector_store(self) -> bool:
        """Initialize vector store if enabled
        
        Returns:
            True if vector store is available, False otherwise
        """
        if self._vector_initialized:
            return self._vector_store is not None
        
        with self._vector_lock:
            if self._vector_initialized:
                return self._vector_store is not None
            
            self._vector_initialized = True
            
            # Check if vector search is enabled
            if MEMTOOL_VECTOR_ENABLED == "off":
                logger.info("Vector search disabled via MEMTOOL_VECTOR_ENABLED=off")
                return False
            
            try:
                from memtool.embedding import VectorStore, get_embedder
                
                vector_dir = self._get_vector_dir()
                embedder = get_embedder(
                    provider=MEMTOOL_EMBEDDING_PROVIDER,
                    model=MEMTOOL_EMBEDDING_MODEL or None
                )
                
                self._vector_store = VectorStore(
                    persist_dir=str(vector_dir),
                    collection_name="memtool",
                    embedder=embedder
                )
                
                logger.info(f"Vector store initialized: {vector_dir}")
                return True
                
            except ImportError as e:
                if MEMTOOL_VECTOR_ENABLED == "on":
                    logger.error(f"Vector search required but dependencies missing: {e}")
                    raise
                logger.info(f"Vector search not available (missing dependencies): {e}")
                return False
            except Exception as e:
                if MEMTOOL_VECTOR_ENABLED == "on":
                    logger.error(f"Vector search failed to initialize: {e}")
                    raise
                logger.warning(f"Vector search disabled due to error: {e}")
                return False
    
    def vector_index(self, item: Dict[str, Any]) -> bool:
        """Index a single item for vector search
        
        Args:
            item: Memory item dict with id, content, type, etc.
            
        Returns:
            True if indexed successfully
        """
        if not self._init_vector_store():
            return False
        
        try:
            content = item.get("content", "")
            if not content:
                return False
            
            metadata = {
                "type": item.get("type"),
                "key": item.get("key"),
                "task_id": item.get("task_id"),
                "step_id": item.get("step_id"),
                "tags": item.get("tags", []),
                "weight": item.get("weight", 1.0),
                "confidence_level": item.get("confidence_level", "medium"),
                "updated_at": item.get("updated_at"),
            }
            
            self._vector_store.add(
                item_id=item["id"],
                content=content,
                metadata=metadata
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to index item {item.get('id')}: {e}")
            return False
    
    def vector_index_batch(self, items: List[Dict[str, Any]]) -> int:
        """Batch index items for vector search
        
        Args:
            items: List of memory item dicts
            
        Returns:
            Number of items indexed
        """
        if not self._init_vector_store():
            return 0
        
        try:
            batch = []
            for item in items:
                content = item.get("content", "")
                if not content:
                    continue
                
                metadata = {
                    "type": item.get("type"),
                    "key": item.get("key"),
                    "task_id": item.get("task_id"),
                    "step_id": item.get("step_id"),
                    "weight": item.get("weight", 1.0),
                    "confidence_level": item.get("confidence_level", "medium"),
                }
                
                batch.append((item["id"], content, metadata))
            
            return self._vector_store.add_batch(batch)
            
        except Exception as e:
            logger.warning(f"Failed to batch index: {e}")
            return 0
    
    def vector_delete(self, item_id: str) -> bool:
        """Remove item from vector index
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if deleted
        """
        if not self._init_vector_store():
            return False
        
        try:
            return self._vector_store.delete(item_id)
        except Exception as e:
            logger.warning(f"Failed to delete from vector index: {e}")
            return False
    
    def semantic_search(
        self,
        query: str,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> Dict[str, Any]:
        """Semantic vector search
        
        Args:
            query: Natural language query
            type: Optional type filter
            task_id: Optional task_id filter
            limit: Maximum results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            Dict with ok, items, and metadata
        """
        if not self._init_vector_store():
            return {
                "ok": False,
                "error": "VECTOR_UNAVAILABLE",
                "message": "Vector search not available",
                "items": []
            }
        
        try:
            # Build filter
            where = {}
            if type:
                where["type"] = type
            if task_id:
                where["task_id"] = task_id
            
            results = self._vector_store.search(
                query=query,
                limit=limit,
                where=where if where else None,
                min_score=min_score
            )
            
            # 瘦身后：过滤只返回核心字段
            filtered_results = [_filter_core_fields(item) for item in results]
            return {
                "ok": True,
                "items": filtered_results,
                "count": len(filtered_results),
                "limit": limit,
                "min_score": min_score
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                "ok": False,
                "error": "SEARCH_ERROR",
                "message": str(e),
                "items": []
            }
    
    def hybrid_search(
        self,
        query: str,
        type: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 10,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> Dict[str, Any]:
        """Hybrid search combining FTS and vector search
        
        Args:
            query: Search query
            type: Optional type filter
            task_id: Optional task_id filter
            limit: Maximum results
            fts_weight: Weight for FTS results (0-1)
            vector_weight: Weight for vector results (0-1)
            
        Returns:
            Dict with ok, items, and metadata
        """
        # First, get FTS results
        fts_result = self.search(
            query=query,
            type=type,
            task_id=task_id,
            limit=limit * 2
        )
        
        fts_items = fts_result.get("items", []) if fts_result.get("ok") else []
        
        # If vector search not available, return FTS only
        if not self._init_vector_store():
            return fts_result
        
        try:
            from memtool.embedding.vector_store import HybridSearcher
            
            # Build filter
            where = {}
            if type:
                where["type"] = type
            if task_id:
                where["task_id"] = task_id
            
            searcher = HybridSearcher(
                vector_store=self._vector_store,
                fts_weight=fts_weight,
                vector_weight=vector_weight
            )
            
            results = searcher.search(
                query=query,
                fts_results=fts_items,
                limit=limit,
                where=where if where else None
            )
            
            # 瘦身后：过滤只返回核心字段
            filtered_results = [_filter_core_fields(item) for item in results]
            return {
                "ok": True,
                "items": filtered_results,
                "count": len(filtered_results),
                "limit": limit,
                "mode": "hybrid",
                "fts_weight": fts_weight,
                "vector_weight": vector_weight
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}, falling back to FTS")
            return fts_result
    
    def vector_sync(self, force: bool = False) -> Dict[str, Any]:
        """Sync all items to vector index
        
        Args:
            force: If True, clear and rebuild index
            
        Returns:
            Dict with sync statistics
        """
        if not self._init_vector_store():
            return {
                "ok": False,
                "error": "VECTOR_UNAVAILABLE",
                "message": "Vector search not available"
            }
        
        try:
            if force:
                self._vector_store.clear()
            
            # Get all items from database
            items = self.export_items()
            
            # Filter out already indexed items (unless force)
            if not force:
                existing_count = self._vector_store.count()
                if existing_count >= len(items):
                    return {
                        "ok": True,
                        "message": "Index up to date",
                        "indexed": existing_count,
                        "total": len(items)
                    }
            
            # Batch index
            indexed = self.vector_index_batch(items)
            
            return {
                "ok": True,
                "indexed": indexed,
                "total": len(items),
                "force": force
            }
            
        except Exception as e:
            logger.error(f"Vector sync failed: {e}")
            return {
                "ok": False,
                "error": "SYNC_ERROR",
                "message": str(e)
            }
    
    def vector_status(self) -> Dict[str, Any]:
        """Get vector search status
        
        Returns:
            Dict with status information
        """
        if not self._init_vector_store():
            return {
                "enabled": False,
                "reason": "not initialized or dependencies missing"
            }
        
        try:
            return {
                "enabled": True,
                "count": self._vector_store.count(),
                "directory": str(self._get_vector_dir()),
                "provider": MEMTOOL_EMBEDDING_PROVIDER,
                "model": MEMTOOL_EMBEDDING_MODEL or "default"
            }
        except Exception as e:
            return {
                "enabled": False,
                "error": str(e)
            }

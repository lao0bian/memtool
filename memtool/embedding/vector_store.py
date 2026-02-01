#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector store for semantic search using ChromaDB

Provides:
- Automatic embedding generation
- Similarity search with filters
- Incremental updates (add/update/delete)
- Persistence to disk
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for semantic search
    
    Args:
        persist_dir: Directory for ChromaDB persistence
        collection_name: Name of the collection (default: "memtool")
        embedder: Embedder instance (default: local embedder)
        
    Example:
        >>> store = VectorStore("./vectors")
        >>> store.add("id1", "Hello world", {"type": "greeting"})
        >>> results = store.search("hi there", limit=5)
    """
    
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "memtool",
        embedder: Optional[Embedder] = None,
    ):
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._embedder = embedder or get_embedder("local")
        self._client = None
        self._collection = None
        
    def _ensure_client(self):
        """Lazy initialization of ChromaDB client"""
        if self._client is not None:
            return
            
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for vector search. "
                "Install with: pip install chromadb"
            )
        
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self._persist_dir),
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(
            f"Initialized VectorStore: {self._persist_dir}, "
            f"collection={self._collection_name}, "
            f"count={self._collection.count()}"
        )
    
    def add(
        self,
        item_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a single item
        
        Args:
            item_id: Unique identifier
            content: Text content to embed
            metadata: Optional metadata dict (will be JSON-serialized)
        """
        self._ensure_client()
        
        # Generate embedding
        embedding = self._embedder.embed_query(content)
        
        # Prepare metadata (ChromaDB only accepts str/int/float/bool)
        safe_metadata = self._prepare_metadata(metadata)
        
        # Upsert (add or update)
        self._collection.upsert(
            ids=[item_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[safe_metadata] if safe_metadata else None
        )
    
    def add_batch(
        self,
        items: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> int:
        """Add or update multiple items efficiently
        
        Args:
            items: List of (item_id, content, metadata) tuples
            
        Returns:
            Number of items added
        """
        if not items:
            return 0
            
        self._ensure_client()
        
        ids = [item[0] for item in items]
        contents = [item[1] for item in items]
        metadatas = [self._prepare_metadata(item[2]) for item in items]
        
        # Batch embed
        embeddings = self._embedder.embed(contents)
        
        # Batch upsert
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=metadatas if any(metadatas) else None
        )
        
        return len(items)
    
    def delete(self, item_id: str) -> bool:
        """Delete an item by ID
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if deleted, False if not found
        """
        self._ensure_client()
        
        try:
            self._collection.delete(ids=[item_id])
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {item_id}: {e}")
            return False
    
    def delete_batch(self, item_ids: List[str]) -> int:
        """Delete multiple items
        
        Args:
            item_ids: List of item identifiers
            
        Returns:
            Number of items deleted
        """
        if not item_ids:
            return 0
            
        self._ensure_client()
        
        try:
            self._collection.delete(ids=item_ids)
            return len(item_ids)
        except Exception as e:
            logger.warning(f"Failed to delete batch: {e}")
            return 0
    
    def search(
        self,
        query: str,
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search
        
        Args:
            query: Search query text
            limit: Maximum results to return
            where: Optional filter dict (ChromaDB where clause)
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of results with id, content, metadata, score
        """
        self._ensure_client()
        
        if self._collection.count() == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._embedder.embed_query(query)
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to list of dicts
        output = []
        if results["ids"] and results["ids"][0]:
            for i, item_id in enumerate(results["ids"][0]):
                # ChromaDB returns cosine distance, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # cosine similarity = 1 - cosine distance
                
                if score < min_score:
                    continue
                
                output.append({
                    "id": item_id,
                    "content": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": round(score, 6)
                })
        
        return output
    
    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item by ID
        
        Args:
            item_id: Item identifier
            
        Returns:
            Item dict or None if not found
        """
        self._ensure_client()
        
        results = self._collection.get(
            ids=[item_id],
            include=["documents", "metadatas"]
        )
        
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }
        return None
    
    def count(self) -> int:
        """Get total number of items"""
        self._ensure_client()
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all items from the collection"""
        self._ensure_client()
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _prepare_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (only primitive types allowed)"""
        if not metadata:
            return {}
        
        safe = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                safe[key] = value
            elif isinstance(value, list):
                # Convert list to JSON string
                safe[f"{key}_json"] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                continue
            else:
                # Convert other types to string
                safe[key] = str(value)
        
        return safe


class HybridSearcher:
    """Hybrid search combining FTS and vector search
    
    Combines traditional FTS5 keyword search with semantic vector search
    for better recall and precision.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.vector_store = vector_store
        self.fts_weight = fts_weight
        self.vector_weight = vector_weight
    
    def search(
        self,
        query: str,
        fts_results: List[Dict[str, Any]],
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search
        
        Args:
            query: Search query
            fts_results: Results from FTS5 search (must include 'id' field)
            limit: Maximum results
            where: Optional vector search filter
            
        Returns:
            Combined and re-ranked results
        """
        # Get vector search results
        vector_results = self.vector_store.search(
            query=query,
            limit=limit * 2,  # Get more for merging
            where=where
        )
        
        # Build score maps
        fts_scores = {}
        for i, item in enumerate(fts_results):
            # Normalize FTS rank to 0-1 (position-based)
            fts_scores[item["id"]] = 1.0 - (i / max(len(fts_results), 1))
        
        vector_scores = {item["id"]: item["score"] for item in vector_results}
        
        # Combine scores
        all_ids = set(fts_scores.keys()) | set(vector_scores.keys())
        combined = []
        
        for item_id in all_ids:
            fts_score = fts_scores.get(item_id, 0.0)
            vec_score = vector_scores.get(item_id, 0.0)
            
            combined_score = (
                self.fts_weight * fts_score + 
                self.vector_weight * vec_score
            )
            
            # Find the original item data
            item_data = None
            for item in fts_results:
                if item["id"] == item_id:
                    item_data = item
                    break
            if item_data is None:
                for item in vector_results:
                    if item["id"] == item_id:
                        item_data = {"id": item_id, "content": item.get("content")}
                        break
            
            if item_data:
                combined.append({
                    **item_data,
                    "hybrid_score": round(combined_score, 6),
                    "fts_score": round(fts_score, 6),
                    "vector_score": round(vec_score, 6),
                })
        
        # Sort by combined score
        combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return combined[:limit]

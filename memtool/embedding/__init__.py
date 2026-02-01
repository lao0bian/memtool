"""Embedding module for vector-based semantic search

Optional: requires chromadb and sentence-transformers
Install with: pip install memtool[vector]
"""

from .embedder import Embedder, get_embedder
from .vector_store import VectorStore
from .semantic import SemanticSearchMixin

__all__ = ["Embedder", "get_embedder", "VectorStore", "SemanticSearchMixin"]

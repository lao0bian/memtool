"""Embedding module for vector-based semantic search

Optional: requires chromadb and sentence-transformers
Install with: pip install memtool[vector]

All imports are lazy to avoid loading heavy dependencies (~20s) at startup.
"""


def __getattr__(name):
    """Lazy module-level imports to avoid cold-start penalty."""
    if name == "Embedder" or name == "get_embedder":
        from .embedder import Embedder, get_embedder
        globals()["Embedder"] = Embedder
        globals()["get_embedder"] = get_embedder
        return globals()[name]
    elif name == "VectorStore":
        from .vector_store import VectorStore
        globals()["VectorStore"] = VectorStore
        return VectorStore
    elif name == "SemanticSearchMixin":
        from .semantic import SemanticSearchMixin
        globals()["SemanticSearchMixin"] = SemanticSearchMixin
        return SemanticSearchMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Embedder", "get_embedder", "VectorStore", "SemanticSearchMixin"]

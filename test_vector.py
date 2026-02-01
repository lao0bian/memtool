#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for vector search functionality
"""
import os
import tempfile
import pytest

# Skip all tests if vector dependencies not available
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")


class TestEmbedder:
    """Test embedding providers"""
    
    def test_local_embedder(self):
        from memtool.embedding import get_embedder
        
        embedder = get_embedder("local")
        
        # Single query
        embedding = embedder.embed_query("Hello world")
        assert embedding.shape == (embedder.dimension,)
        
        # Batch
        embeddings = embedder.embed(["Hello", "World", "Test"])
        assert embeddings.shape == (3, embedder.dimension)
    
    def test_embedder_cache(self):
        from memtool.embedding import get_embedder
        
        e1 = get_embedder("local")
        e2 = get_embedder("local")
        assert e1 is e2  # Should be cached


class TestVectorStore:
    """Test vector store operations"""
    
    @pytest.fixture
    def store(self):
        from memtool.embedding import VectorStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir)
            yield store
    
    def test_add_and_search(self, store):
        store.add("id1", "Python is a programming language", {"type": "feature"})
        store.add("id2", "JavaScript runs in the browser", {"type": "feature"})
        store.add("id3", "Machine learning uses neural networks", {"type": "project"})
        
        # Search for programming
        results = store.search("programming language", limit=2)
        assert len(results) >= 1
        assert results[0]["id"] == "id1"  # Python should be most relevant
    
    def test_add_batch(self, store):
        items = [
            ("id1", "First document", {"type": "run"}),
            ("id2", "Second document", {"type": "run"}),
            ("id3", "Third document", {"type": "feature"}),
        ]
        count = store.add_batch(items)
        assert count == 3
        assert store.count() == 3
    
    def test_delete(self, store):
        store.add("id1", "Test content", {})
        assert store.count() == 1
        
        store.delete("id1")
        assert store.count() == 0
    
    def test_filter_search(self, store):
        store.add("id1", "Error in production", {"type": "run"})
        store.add("id2", "Error in development", {"type": "feature"})
        
        # Filter by type
        results = store.search("error", where={"type": "run"})
        assert len(results) == 1
        assert results[0]["id"] == "id1"
    
    def test_chinese_content(self, store):
        store.add("id1", "这是一个关于机器学习的文档", {"type": "project"})
        store.add("id2", "Python编程语言教程", {"type": "feature"})
        store.add("id3", "深度学习神经网络入门", {"type": "project"})
        
        results = store.search("机器学习", limit=2)
        assert len(results) >= 1
        # Chinese embedding should find relevant content


class TestSemanticSearch:
    """Test semantic search integration"""
    
    @pytest.fixture
    def memory_store(self):
        from memtool_core import MemoryStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            os.environ["MEMTOOL_VECTOR_ENABLED"] = "on"
            os.environ["MEMTOOL_VECTOR_DIR"] = os.path.join(tmpdir, "vectors")
            
            store = MemoryStore(db_path)
            yield store
            
            del os.environ["MEMTOOL_VECTOR_ENABLED"]
            del os.environ["MEMTOOL_VECTOR_DIR"]
    
    def test_auto_index_on_put(self, memory_store):
        result = memory_store.put(
            item_id=None,
            type="feature",
            key="test",
            content="This is a test document about Python programming",
        )
        assert result["ok"]
        
        # Check vector status
        status = memory_store.vector_status()
        assert status["enabled"]
        assert status["count"] >= 1
    
    def test_semantic_search(self, memory_store):
        # Add some documents
        memory_store.put(
            item_id=None, type="feature", key="python",
            content="Python is a versatile programming language"
        )
        memory_store.put(
            item_id=None, type="feature", key="javascript",
            content="JavaScript enables interactive web pages"
        )
        memory_store.put(
            item_id=None, type="project", key="ml",
            content="Deep learning models for image recognition"
        )
        
        # Semantic search
        results = memory_store.semantic_search("coding language", limit=2)
        assert results["ok"]
        assert len(results["items"]) >= 1
    
    def test_hybrid_search(self, memory_store):
        memory_store.put(
            item_id=None, type="run", key="error1",
            content="Connection timeout error in database"
        )
        memory_store.put(
            item_id=None, type="run", key="error2",
            content="Memory allocation failure in worker"
        )
        
        # Hybrid combines FTS and vector
        results = memory_store.hybrid_search("database timeout", limit=5)
        assert results["ok"]
        assert results.get("mode") == "hybrid"
    
    def test_vector_sync(self, memory_store):
        # Add without triggering auto-index
        memory_store.put(
            item_id=None, type="feature", key="doc1",
            content="Document one"
        )
        
        # Force sync
        result = memory_store.vector_sync(force=True)
        assert result["ok"]
        assert result["indexed"] >= 1


class TestHybridSearcher:
    """Test hybrid search algorithm"""
    
    def test_score_combination(self):
        from memtool.embedding.vector_store import HybridSearcher, VectorStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir)
            store.add("id1", "exact keyword match here", {})
            store.add("id2", "semantic meaning similar", {})
            
            searcher = HybridSearcher(store, fts_weight=0.3, vector_weight=0.7)
            
            fts_results = [
                {"id": "id1", "content": "exact keyword match here"},
            ]
            
            results = searcher.search("keyword", fts_results, limit=5)
            assert len(results) >= 1
            assert "hybrid_score" in results[0]
            assert "fts_score" in results[0]
            assert "vector_score" in results[0]

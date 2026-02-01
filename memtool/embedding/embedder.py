#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding providers for memtool

Supports:
- Local: sentence-transformers (default, no API key needed)
- OpenAI: text-embedding-3-small (requires OPENAI_API_KEY)
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default local model - good balance of quality and speed for Chinese + English
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
# Fallback English model
FALLBACK_LOCAL_MODEL = "all-MiniLM-L6-v2"


class Embedder(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension"""
        pass


class LocalEmbedder(Embedder):
    """Local embedding using sentence-transformers
    
    Uses BAAI/bge-small-zh-v1.5 by default (good for Chinese + English)
    Falls back to all-MiniLM-L6-v2 if the Chinese model fails to load
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self._model = None
        self._model_name = model_name or DEFAULT_LOCAL_MODEL
        self._dimension: Optional[int] = None
        
    def _load_model(self):
        if self._model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model with dimension: {self._dimension}")
        except Exception as e:
            logger.warning(f"Failed to load {self._model_name}: {e}, trying fallback")
            self._model_name = FALLBACK_LOCAL_MODEL
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded fallback model with dimension: {self._dimension}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        if not texts:
            return np.array([])
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        embedding = self._model.encode([query], normalize_embeddings=True)
        return np.array(embedding[0])
    
    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension or 384


class OpenAIEmbedder(Embedder):
    """OpenAI embedding using text-embedding-3-small
    
    Requires OPENAI_API_KEY environment variable
    """
    
    DIMENSION = 1536  # text-embedding-3-small dimension
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        
        client = self._get_client()
        
        # OpenAI API has a limit of 8191 tokens per request
        # Process in batches if needed
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self._model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=[query]
        )
        return np.array(response.data[0].embedding)
    
    @property
    def dimension(self) -> int:
        return self.DIMENSION


class OllamaEmbedder(Embedder):
    """Ollama embedding for fully local operation
    
    Requires Ollama running locally with an embedding model
    Default model: nomic-embed-text
    """
    
    def __init__(
        self, 
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        self._model = model
        self._base_url = base_url
        self._dimension: Optional[int] = None
    
    def _call_ollama(self, texts: List[str]) -> List[List[float]]:
        import requests
        
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
            
            if self._dimension is None:
                self._dimension = len(data["embedding"])
        
        return embeddings
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self._call_ollama(texts)
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        embeddings = self._call_ollama([query])
        return np.array(embeddings[0])
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe dimension with a test embedding
            self.embed_query("test")
        return self._dimension or 768


# Global embedder cache
_embedder_cache: dict = {}


def get_embedder(
    provider: str = "local",
    model: Optional[str] = None,
    **kwargs
) -> Embedder:
    """Get or create an embedder instance
    
    Args:
        provider: "local", "openai", or "ollama"
        model: Optional model name override
        **kwargs: Additional arguments passed to embedder constructor
        
    Returns:
        Embedder instance
        
    Example:
        >>> embedder = get_embedder("local")
        >>> embedder = get_embedder("openai", api_key="sk-...")
        >>> embedder = get_embedder("ollama", model="nomic-embed-text")
    """
    cache_key = f"{provider}:{model or 'default'}"
    
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]
    
    if provider == "local":
        embedder = LocalEmbedder(model_name=model)
    elif provider == "openai":
        embedder = OpenAIEmbedder(model=model or "text-embedding-3-small", **kwargs)
    elif provider == "ollama":
        embedder = OllamaEmbedder(model=model or "nomic-embed-text", **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
    
    _embedder_cache[cache_key] = embedder
    return embedder

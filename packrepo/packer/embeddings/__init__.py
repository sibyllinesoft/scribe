"""
V2 Embeddings System for PackRepo

Provides semantic embeddings generation, caching, and management for enhanced 
code chunk selection using sentence transformers and mixed centroids clustering.
"""

from .base import EmbeddingProvider, EmbeddingCache, CodeEmbedding, CachedEmbeddingProvider
from .sentence_transformers import SentenceTransformersProvider, create_fast_provider
from .cache import SHA256EmbeddingCache

__all__ = [
    'EmbeddingProvider',
    'EmbeddingCache', 
    'CodeEmbedding',
    'CachedEmbeddingProvider',
    'SentenceTransformersProvider',
    'create_fast_provider',
    'SHA256EmbeddingCache',
]
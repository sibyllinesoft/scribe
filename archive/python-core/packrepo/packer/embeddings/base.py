"""
Base classes for V2 embeddings system.

Provides abstract interfaces for embedding providers and caches, along with
core data structures for semantic embeddings of code chunks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path

from ..chunker.base import Chunk


@dataclass
class CodeEmbedding:
    """
    Semantic embedding for a code chunk with metadata.
    
    Stores both the embedding vector and associated metadata for
    cache management and similarity computations.
    """
    
    chunk_id: str
    file_path: str
    content_sha: str  # SHA-256 of chunk content for cache invalidation
    embedding: np.ndarray  # Dense embedding vector
    model_name: str  # Model used to generate embedding
    model_version: str  # Model version for compatibility
    created_at: float  # Unix timestamp
    
    def __post_init__(self):
        """Validate embedding properties."""
        if not isinstance(self.embedding, np.ndarray):
            raise ValueError("Embedding must be numpy array")
        if self.embedding.ndim != 1:
            raise ValueError("Embedding must be 1-dimensional array")
        if len(self.content_sha) != 64:
            raise ValueError("content_sha must be 64-character SHA-256 hex")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding)
    
    def cosine_similarity(self, other: 'CodeEmbedding') -> float:
        """Calculate cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        dot_product = np.dot(self.embedding, other.embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other.embedding)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def euclidean_distance(self, other: 'CodeEmbedding') -> float:
        """Calculate Euclidean distance to another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        return float(np.linalg.norm(self.embedding - other.embedding))


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding generation providers.
    
    Defines interface for generating semantic embeddings from code chunks
    using various embedding models (sentence transformers, code-specific models, etc.).
    """
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name identifier."""
        pass
    
    @abstractmethod
    def get_model_version(self) -> str:
        """Get the model version identifier."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of generated embeddings."""
        pass
    
    @abstractmethod
    def encode_chunk(self, chunk: Chunk) -> CodeEmbedding:
        """
        Generate embedding for a single code chunk.
        
        Args:
            chunk: Code chunk to embed
            
        Returns:
            CodeEmbedding with generated embedding vector and metadata
        """
        pass
    
    @abstractmethod
    def encode_chunks(self, chunks: List[Chunk]) -> List[CodeEmbedding]:
        """
        Generate embeddings for multiple code chunks efficiently.
        
        Args:
            chunks: List of code chunks to embed
            
        Returns:
            List of CodeEmbeddings in same order as input chunks
        """
        pass
    
    def prepare_chunk_text(self, chunk: Chunk) -> str:
        """
        Prepare chunk text for embedding generation.
        
        Can be overridden by specific providers to customize text preprocessing.
        Default implementation combines signature, docstring, and content.
        
        Args:
            chunk: Code chunk to prepare
            
        Returns:
            Processed text ready for embedding
        """
        parts = []
        
        # Add kind and name as context
        if chunk.kind and chunk.name:
            parts.append(f"{chunk.kind.value}: {chunk.name}")
        
        # Add signature if available
        if chunk.signature and chunk.signature.strip():
            parts.append(f"Signature: {chunk.signature}")
        
        # Add docstring if available
        if chunk.docstring and chunk.docstring.strip():
            parts.append(f"Documentation: {chunk.docstring}")
        
        # Add content (limited to avoid excessive length)
        content = chunk.content or ""
        if len(content) > 2000:  # Limit to ~2000 chars for embedding efficiency
            content = content[:2000] + "..."
        parts.append(f"Code: {content}")
        
        return " ".join(parts)


class EmbeddingCache(ABC):
    """
    Abstract base class for embedding cache implementations.
    
    Provides interface for efficient storage and retrieval of embeddings
    with SHA-based invalidation for file content changes.
    """
    
    @abstractmethod
    def get(self, content_sha: str) -> Optional[CodeEmbedding]:
        """
        Retrieve embedding by content SHA.
        
        Args:
            content_sha: SHA-256 hash of chunk content
            
        Returns:
            CodeEmbedding if found in cache, None otherwise
        """
        pass
    
    @abstractmethod
    def put(self, embedding: CodeEmbedding) -> None:
        """
        Store embedding in cache.
        
        Args:
            embedding: CodeEmbedding to cache
        """
        pass
    
    @abstractmethod
    def get_batch(self, content_shas: List[str]) -> Dict[str, CodeEmbedding]:
        """
        Retrieve multiple embeddings efficiently.
        
        Args:
            content_shas: List of SHA-256 hashes
            
        Returns:
            Dictionary mapping SHA to CodeEmbedding for found entries
        """
        pass
    
    @abstractmethod
    def put_batch(self, embeddings: List[CodeEmbedding]) -> None:
        """
        Store multiple embeddings efficiently.
        
        Args:
            embeddings: List of CodeEmbeddings to cache
        """
        pass
    
    @abstractmethod
    def invalidate(self, content_sha: str) -> bool:
        """
        Remove embedding from cache.
        
        Args:
            content_sha: SHA-256 hash to remove
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """
        Clear all entries from cache.
        
        Returns:
            Number of entries removed
        """
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of entries in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics (hits, misses, size, etc.)
        """
        pass


class CachedEmbeddingProvider:
    """
    Wrapper that adds caching to any EmbeddingProvider.
    
    Provides automatic cache management with SHA-based invalidation
    for efficient embedding reuse across multiple pack generations.
    """
    
    def __init__(self, provider: EmbeddingProvider, cache: EmbeddingCache):
        """
        Initialize cached embedding provider.
        
        Args:
            provider: Underlying embedding provider
            cache: Cache implementation for storing embeddings
        """
        self.provider = provider
        self.cache = cache
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'embeddings_generated': 0,
            'batch_requests': 0,
        }
    
    def encode_chunk(self, chunk: Chunk) -> CodeEmbedding:
        """
        Generate or retrieve cached embedding for a chunk.
        
        Args:
            chunk: Code chunk to embed
            
        Returns:
            CodeEmbedding from cache or newly generated
        """
        content_sha = self._compute_content_sha(chunk)
        
        # Try cache first
        cached = self.cache.get(content_sha)
        if cached is not None:
            self._stats['cache_hits'] += 1
            return cached
        
        # Generate new embedding
        self._stats['cache_misses'] += 1
        embedding = self.provider.encode_chunk(chunk)
        self._stats['embeddings_generated'] += 1
        
        # Store in cache
        self.cache.put(embedding)
        
        return embedding
    
    def encode_chunks(self, chunks: List[Chunk]) -> List[CodeEmbedding]:
        """
        Generate or retrieve cached embeddings for multiple chunks.
        
        Args:
            chunks: List of code chunks to embed
            
        Returns:
            List of CodeEmbeddings in same order as input chunks
        """
        self._stats['batch_requests'] += 1
        
        # Compute content SHAs
        content_shas = [self._compute_content_sha(chunk) for chunk in chunks]
        
        # Check cache for batch
        cached_embeddings = self.cache.get_batch(content_shas)
        
        # Find chunks that need embedding generation
        missing_chunks = []
        missing_indices = []
        
        for i, (chunk, sha) in enumerate(zip(chunks, content_shas)):
            if sha not in cached_embeddings:
                missing_chunks.append(chunk)
                missing_indices.append(i)
        
        # Generate missing embeddings
        new_embeddings = []
        if missing_chunks:
            new_embeddings = self.provider.encode_chunks(missing_chunks)
            self._stats['embeddings_generated'] += len(new_embeddings)
            
            # Cache new embeddings
            self.cache.put_batch(new_embeddings)
        
        # Assemble results in original order
        results = []
        new_idx = 0
        
        for i, (chunk, sha) in enumerate(zip(chunks, content_shas)):
            if sha in cached_embeddings:
                results.append(cached_embeddings[sha])
                self._stats['cache_hits'] += 1
            else:
                results.append(new_embeddings[new_idx])
                new_idx += 1
                self._stats['cache_misses'] += 1
        
        return results
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the underlying provider."""
        return {
            'model_name': self.provider.get_model_name(),
            'model_version': self.provider.get_model_version(),
            'embedding_dimension': str(self.provider.get_embedding_dimension()),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get combined provider and cache statistics."""
        cache_stats = self.cache.get_stats()
        return {**self._stats, **cache_stats}
    
    def _compute_content_sha(self, chunk: Chunk) -> str:
        """Compute SHA-256 hash of chunk content for cache key."""
        import hashlib
        
        # Include relevant chunk properties that affect embedding
        content_parts = [
            chunk.rel_path,
            str(chunk.kind.value if chunk.kind else ''),
            chunk.name or '',
            chunk.content or '',
            chunk.signature or '',
            chunk.docstring or '',
            self.provider.get_model_name(),
            self.provider.get_model_version(),
        ]
        
        content_str = '|'.join(content_parts)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
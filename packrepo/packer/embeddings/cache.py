"""
SHA-based embedding cache implementation for V2 system.

Provides efficient caching of embeddings with SHA-256 content-based keys
for automatic invalidation when file contents change.
"""

import json
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np

from .base import EmbeddingCache, CodeEmbedding


class SHA256EmbeddingCache(EmbeddingCache):
    """
    SQLite-based embedding cache using SHA-256 content hashes.
    
    Provides persistent storage of embeddings with automatic invalidation
    when file contents change. Uses SQLite for efficient querying and
    atomic operations.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_entries: int = 100000,
        cleanup_threshold: float = 0.8,
        enable_wal: bool = True,
    ):
        """
        Initialize SHA-256 embedding cache.
        
        Args:
            cache_dir: Directory to store cache database
            max_entries: Maximum number of entries before cleanup
            cleanup_threshold: Fraction of max_entries to trigger cleanup
            enable_wal: Enable SQLite WAL mode for better concurrency
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_entries = max_entries
        self.cleanup_threshold = cleanup_threshold
        self.db_path = self.cache_dir / "embeddings.db"
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'batch_gets': 0,
            'batch_puts': 0,
            'invalidations': 0,
            'cleanups': 0,
        }
        
        # Initialize database
        self._init_database(enable_wal)
    
    def _init_database(self, enable_wal: bool):
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            if enable_wal:
                conn.execute("PRAGMA journal_mode = WAL")
            
            # Performance optimizations
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            conn.execute("PRAGMA temp_store = MEMORY")
            
            # Create table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_sha TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    embedding_data BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL
                )
            """)
            
            # Create indexes for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model 
                ON embeddings(model_name, model_version)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path 
                ON embeddings(file_path)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON embeddings(last_accessed DESC)
            """)
            
            conn.commit()
    
    def get(self, content_sha: str) -> Optional[CodeEmbedding]:
        """
        Retrieve embedding by content SHA.
        
        Args:
            content_sha: SHA-256 hash of chunk content
            
        Returns:
            CodeEmbedding if found in cache, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM embeddings WHERE content_sha = ?
            """, (content_sha,))
            
            row = cursor.fetchone()
            if row is None:
                self._stats['misses'] += 1
                return None
            
            # Update last accessed time
            cursor.execute("""
                UPDATE embeddings SET last_accessed = ? WHERE content_sha = ?
            """, (time.time(), content_sha))
            conn.commit()
            
            # Deserialize embedding
            try:
                embedding = self._row_to_embedding(row)
                self._stats['hits'] += 1
                return embedding
            except Exception:
                # Corrupted entry, remove it
                cursor.execute("DELETE FROM embeddings WHERE content_sha = ?", (content_sha,))
                conn.commit()
                self._stats['misses'] += 1
                return None
    
    def put(self, embedding: CodeEmbedding) -> None:
        """
        Store embedding in cache.
        
        Args:
            embedding: CodeEmbedding to cache
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize embedding data
            embedding_data = pickle.dumps(embedding.embedding)
            current_time = time.time()
            
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (
                    content_sha, chunk_id, file_path, model_name, model_version,
                    embedding_dimension, embedding_data, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                embedding.content_sha,
                embedding.chunk_id,
                embedding.file_path,
                embedding.model_name,
                embedding.model_version,
                embedding.dimension,
                embedding_data,
                embedding.created_at,
                current_time,
            ))
            
            conn.commit()
            self._stats['puts'] += 1
            
            # Check if cleanup is needed
            if self._should_cleanup(cursor):
                self._cleanup_old_entries(cursor, conn)
    
    def get_batch(self, content_shas: List[str]) -> Dict[str, CodeEmbedding]:
        """
        Retrieve multiple embeddings efficiently.
        
        Args:
            content_shas: List of SHA-256 hashes
            
        Returns:
            Dictionary mapping SHA to CodeEmbedding for found entries
        """
        if not content_shas:
            return {}
        
        self._stats['batch_gets'] += 1
        results = {}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Query in batches to handle large lists efficiently
            batch_size = 500
            current_time = time.time()
            found_shas = []
            
            for i in range(0, len(content_shas), batch_size):
                batch_shas = content_shas[i:i + batch_size]
                placeholders = ','.join(['?' for _ in batch_shas])
                
                cursor.execute(f"""
                    SELECT * FROM embeddings WHERE content_sha IN ({placeholders})
                """, batch_shas)
                
                for row in cursor.fetchall():
                    try:
                        embedding = self._row_to_embedding(row)
                        results[row['content_sha']] = embedding
                        found_shas.append(row['content_sha'])
                    except Exception:
                        # Skip corrupted entries
                        continue
            
            # Update last accessed times for found entries
            if found_shas:
                for i in range(0, len(found_shas), batch_size):
                    batch_shas = found_shas[i:i + batch_size]
                    placeholders = ','.join(['?' for _ in batch_shas])
                    cursor.execute(f"""
                        UPDATE embeddings SET last_accessed = ? 
                        WHERE content_sha IN ({placeholders})
                    """, [current_time] + batch_shas)
            
            conn.commit()
            
            # Update statistics
            self._stats['hits'] += len(results)
            self._stats['misses'] += len(content_shas) - len(results)
            
            return results
    
    def put_batch(self, embeddings: List[CodeEmbedding]) -> None:
        """
        Store multiple embeddings efficiently.
        
        Args:
            embeddings: List of CodeEmbeddings to cache
        """
        if not embeddings:
            return
        
        self._stats['batch_puts'] += 1
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            current_time = time.time()
            
            # Prepare batch data
            batch_data = []
            for embedding in embeddings:
                embedding_data = pickle.dumps(embedding.embedding)
                batch_data.append((
                    embedding.content_sha,
                    embedding.chunk_id,
                    embedding.file_path,
                    embedding.model_name,
                    embedding.model_version,
                    embedding.dimension,
                    embedding_data,
                    embedding.created_at,
                    current_time,
                ))
            
            # Execute batch insert
            cursor.executemany("""
                INSERT OR REPLACE INTO embeddings (
                    content_sha, chunk_id, file_path, model_name, model_version,
                    embedding_dimension, embedding_data, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_data)
            
            conn.commit()
            self._stats['puts'] += len(embeddings)
            
            # Check if cleanup is needed
            if self._should_cleanup(cursor):
                self._cleanup_old_entries(cursor, conn)
    
    def invalidate(self, content_sha: str) -> bool:
        """
        Remove embedding from cache.
        
        Args:
            content_sha: SHA-256 hash to remove
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embeddings WHERE content_sha = ?", (content_sha,))
            removed = cursor.rowcount > 0
            conn.commit()
            
            if removed:
                self._stats['invalidations'] += 1
            
            return removed
    
    def clear(self) -> int:
        """
        Clear all entries from cache.
        
        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            
            cursor.execute("DELETE FROM embeddings")
            cursor.execute("VACUUM")  # Reclaim space
            conn.commit()
            
            return count
    
    def size(self) -> int:
        """Get number of entries in cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics (hits, misses, size, etc.)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get size and storage info
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            entry_count = cursor.fetchone()[0]
            
            # Get file size
            file_size_mb = self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0
            
            # Calculate hit rate
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(1, total_requests)
            
            # Get model distribution
            cursor.execute("""
                SELECT model_name, model_version, COUNT(*) as count
                FROM embeddings
                GROUP BY model_name, model_version
                ORDER BY count DESC
            """)
            model_distribution = {f"{row[0]}:{row[1]}": row[2] for row in cursor.fetchall()}
            
            return {
                **self._stats,
                'entry_count': entry_count,
                'file_size_mb': round(file_size_mb, 2),
                'hit_rate': round(hit_rate, 3),
                'max_entries': self.max_entries,
                'model_distribution': model_distribution,
            }
    
    def cleanup_by_model(self, model_name: str, model_version: Optional[str] = None) -> int:
        """
        Remove entries for specific model version.
        
        Args:
            model_name: Model name to clean up
            model_version: Model version to clean up (None for all versions)
            
        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_version is None:
                cursor.execute("DELETE FROM embeddings WHERE model_name = ?", (model_name,))
            else:
                cursor.execute(
                    "DELETE FROM embeddings WHERE model_name = ? AND model_version = ?",
                    (model_name, model_version)
                )
            
            removed = cursor.rowcount
            conn.commit()
            
            return removed
    
    def _row_to_embedding(self, row) -> CodeEmbedding:
        """Convert database row to CodeEmbedding."""
        embedding_array = pickle.loads(row['embedding_data'])
        
        return CodeEmbedding(
            chunk_id=row['chunk_id'],
            file_path=row['file_path'],
            content_sha=row['content_sha'],
            embedding=embedding_array,
            model_name=row['model_name'],
            model_version=row['model_version'],
            created_at=row['created_at'],
        )
    
    def _should_cleanup(self, cursor) -> bool:
        """Check if cleanup is needed based on entry count."""
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        return count > self.max_entries * self.cleanup_threshold
    
    def _cleanup_old_entries(self, cursor, conn):
        """Remove oldest entries to stay within limits."""
        # Keep only the most recently accessed entries
        keep_count = int(self.max_entries * 0.7)  # Keep 70% after cleanup
        
        cursor.execute("""
            DELETE FROM embeddings WHERE rowid NOT IN (
                SELECT rowid FROM embeddings 
                ORDER BY last_accessed DESC 
                LIMIT ?
            )
        """, (keep_count,))
        
        removed = cursor.rowcount
        conn.commit()
        
        self._stats['cleanups'] += 1
        
        # Run VACUUM periodically to reclaim space
        if self._stats['cleanups'] % 10 == 0:
            cursor.execute("VACUUM")


class InMemoryEmbeddingCache(EmbeddingCache):
    """
    In-memory embedding cache for testing and development.
    
    Provides fast access but no persistence. Useful for development
    and testing scenarios where persistence is not needed.
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize in-memory cache.
        
        Args:
            max_entries: Maximum number of entries before eviction
        """
        self.max_entries = max_entries
        self._cache: Dict[str, CodeEmbedding] = {}
        self._access_order: List[str] = []  # LRU tracking
        self._stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0,
        }
    
    def get(self, content_sha: str) -> Optional[CodeEmbedding]:
        """Retrieve embedding by content SHA."""
        if content_sha in self._cache:
            # Update access order (move to end)
            self._access_order.remove(content_sha)
            self._access_order.append(content_sha)
            self._stats['hits'] += 1
            return self._cache[content_sha]
        else:
            self._stats['misses'] += 1
            return None
    
    def put(self, embedding: CodeEmbedding) -> None:
        """Store embedding in cache."""
        content_sha = embedding.content_sha
        
        # Remove if already exists
        if content_sha in self._cache:
            self._access_order.remove(content_sha)
        
        # Add new entry
        self._cache[content_sha] = embedding
        self._access_order.append(content_sha)
        self._stats['puts'] += 1
        
        # Evict oldest if over limit
        while len(self._cache) > self.max_entries:
            oldest_sha = self._access_order.pop(0)
            del self._cache[oldest_sha]
            self._stats['evictions'] += 1
    
    def get_batch(self, content_shas: List[str]) -> Dict[str, CodeEmbedding]:
        """Retrieve multiple embeddings."""
        results = {}
        for sha in content_shas:
            embedding = self.get(sha)
            if embedding is not None:
                results[sha] = embedding
        return results
    
    def put_batch(self, embeddings: List[CodeEmbedding]) -> None:
        """Store multiple embeddings."""
        for embedding in embeddings:
            self.put(embedding)
    
    def invalidate(self, content_sha: str) -> bool:
        """Remove embedding from cache."""
        if content_sha in self._cache:
            del self._cache[content_sha]
            self._access_order.remove(content_sha)
            return True
        return False
    
    def clear(self) -> int:
        """Clear all entries from cache."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        return count
    
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / max(1, total_requests)
        
        return {
            **self._stats,
            'entry_count': len(self._cache),
            'hit_rate': round(hit_rate, 3),
            'max_entries': self.max_entries,
        }
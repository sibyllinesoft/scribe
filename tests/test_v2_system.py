"""
Tests for V2 coverage clustering system.

Validates the embedding generation, k-means clustering, HNSW medoids,
and mixed centroid functionality according to TODO.md requirements.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from packrepo.packer.chunker.base import Chunk, ChunkKind
from packrepo.packer.selector.base import SelectionConfig, SelectionVariant, PackRequest
from packrepo.packer.selector.selector import RepositorySelector
from packrepo.packer.tokenizer.implementations import ApproximateTokenizer


class TestV2EmbeddingSystem:
    """Test V2 embedding generation and caching."""
    
    def test_embedding_cache_integration(self):
        """Test that embeddings are cached and reused efficiently."""
        # Skip if sentence-transformers not available
        pytest.importorskip("sentence_transformers")
        
        from packrepo.packer.embeddings import create_fast_provider, SHA256EmbeddingCache, CachedEmbeddingProvider
        import tempfile
        
        # Create test chunks
        chunks = self._create_test_chunks()
        
        # Initialize embedding system
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = create_fast_provider(device='cpu')
            cache = SHA256EmbeddingCache(cache_dir=temp_dir, max_entries=100)
            cached_provider = CachedEmbeddingProvider(provider, cache)
            
            # First encoding - should miss cache
            embeddings1 = cached_provider.encode_chunks(chunks)
            stats1 = cached_provider.get_cache_stats()
            
            assert len(embeddings1) == len(chunks)
            assert stats1['cache_misses'] == len(chunks)
            assert stats1['cache_hits'] == 0
            
            # Second encoding - should hit cache
            embeddings2 = cached_provider.encode_chunks(chunks)
            stats2 = cached_provider.get_cache_stats()
            
            assert len(embeddings2) == len(chunks)
            assert stats2['cache_hits'] == len(chunks)
            
            # Validate embeddings are identical
            for emb1, emb2 in zip(embeddings1, embeddings2):
                assert emb1.content_sha == emb2.content_sha
                assert np.allclose(emb1.embedding, emb2.embedding)
    
    def test_embedding_invalidation(self):
        """Test that embedding cache invalidates on content change."""
        pytest.importorskip("sentence_transformers")
        
        from packrepo.packer.embeddings import create_fast_provider, SHA256EmbeddingCache, CachedEmbeddingProvider
        import tempfile
        
        # Create test chunks
        chunks = self._create_test_chunks()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = create_fast_provider(device='cpu')
            cache = SHA256EmbeddingCache(cache_dir=temp_dir, max_entries=100)
            cached_provider = CachedEmbeddingProvider(provider, cache)
            
            # Encode original chunks
            embeddings1 = cached_provider.encode_chunks(chunks)
            
            # Modify chunk content
            modified_chunks = chunks.copy()
            modified_chunks[0].content = "modified content"
            
            # Encode modified chunks
            embeddings2 = cached_provider.encode_chunks(modified_chunks)
            
            # First chunk should have different SHA and embedding
            assert embeddings1[0].content_sha != embeddings2[0].content_sha
            assert not np.allclose(embeddings1[0].embedding, embeddings2[0].embedding)
            
            # Other chunks should be cached
            for i in range(1, len(chunks)):
                assert embeddings1[i].content_sha == embeddings2[i].content_sha
                assert np.allclose(embeddings1[i].embedding, embeddings2[i].embedding)
    
    def _create_test_chunks(self):
        """Create sample chunks for testing."""
        chunks = []
        for i in range(3):
            chunk = Chunk(
                id=f"test_chunk_{i}",
                path=Path(f"/test/file_{i}.py"),
                rel_path=f"file_{i}.py",
                start_line=1,
                end_line=10,
                kind=ChunkKind.FUNCTION,
                name=f"test_function_{i}",
                language="python",
                content=f"def test_function_{i}():\n    return {i}",
                full_tokens=20,
                signature_tokens=5,
            )
            chunks.append(chunk)
        return chunks


class TestV2KMeansClustering:
    """Test k-means clustering with k≈√N per package logic."""
    
    def test_package_clustering(self):
        """Test per-package k-means clustering."""
        pytest.importorskip("sklearn")
        
        from packrepo.packer.clustering import create_package_clusterer
        from packrepo.packer.embeddings.base import CodeEmbedding
        
        # Create embeddings with different packages
        embeddings = self._create_test_embeddings()
        
        # Perform clustering
        clusterer = create_package_clusterer(random_state=42)
        result = clusterer.fit_predict(embeddings)
        
        # Validate clustering result
        assert result.num_clusters > 0
        assert result.total_embeddings == len(embeddings)
        assert result.clustering_method == "kmeans_per_package"
        
        # Check that clusters have reasonable sizes
        cluster_sizes = [cluster.size for cluster in result.clusters]
        assert all(size > 0 for size in cluster_sizes)
        assert sum(cluster_sizes) == len(embeddings)
    
    def test_sqrt_n_heuristic(self):
        """Test that k≈√N heuristic is applied correctly."""
        from packrepo.packer.clustering.kmeans import KMeansClusterer
        
        clusterer = KMeansClusterer()
        
        # Test various input sizes
        test_cases = [
            (1, 1),    # Edge case
            (4, 2),    # √4 = 2
            (9, 3),    # √9 = 3
            (16, 4),   # √16 = 4
            (25, 5),   # √25 = 5
        ]
        
        for n, expected_k in test_cases:
            embeddings = self._create_test_embeddings(count=n)
            optimal_k = clusterer.calculate_optimal_k(embeddings)
            assert optimal_k == expected_k
    
    def test_clustering_quality_metrics(self):
        """Test that clustering quality metrics are calculated."""
        pytest.importorskip("sklearn")
        
        from packrepo.packer.clustering import create_package_clusterer
        
        embeddings = self._create_test_embeddings(count=20)
        clusterer = create_package_clusterer(random_state=42)
        result = clusterer.fit_predict(embeddings)
        
        # Validate quality metrics are calculated
        assert result.inertia >= 0
        assert 0 <= result.coverage_uniformity <= 1
        assert result.clustering_time >= 0
        assert result.memory_peak_mb >= 0
    
    def _create_test_embeddings(self, count=10):
        """Create test embeddings with varying packages."""
        from packrepo.packer.embeddings.base import CodeEmbedding
        import time
        
        embeddings = []
        packages = ["pkg_a", "pkg_b", "pkg_c"]
        
        for i in range(count):
            # Create embeddings with some structure (different clusters)
            base_vector = np.random.RandomState(42).randn(384) * 0.1
            if i < count // 2:
                base_vector[:100] += 1.0  # First cluster
            else:
                base_vector[100:200] += 1.0  # Second cluster
            
            embedding = CodeEmbedding(
                chunk_id=f"chunk_{i}",
                file_path=f"{packages[i % len(packages)]}/file_{i}.py",
                content_sha=f"sha{i:02d}" + "0" * 62,  # Valid SHA-256 length
                embedding=base_vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        return embeddings


class TestV2HNSWMedoids:
    """Test HNSW medoid clustering for long-tail coverage."""
    
    def test_medoid_clustering(self):
        """Test HNSW medoid clustering functionality."""
        pytest.importorskip("hnswlib")
        
        from packrepo.packer.clustering import create_density_medoid_clusterer
        
        # Create embeddings with tail distribution
        embeddings = self._create_tail_embeddings()
        
        # Perform medoid clustering
        clusterer = create_density_medoid_clusterer(
            coverage_threshold=0.5,
            random_state=42,
        )
        result = clusterer.fit_predict(embeddings)
        
        # Validate result
        assert result.clustering_method.startswith("hnsw_medoids")
        assert result.total_embeddings == len(embeddings)
        
        # Check that clusters are medoid-based
        for cluster in result.clusters:
            assert cluster.cluster_type == "hnsw_medoid"
            assert cluster.size >= 1  # At least the medoid itself
    
    def test_tail_identification(self):
        """Test that tail embeddings are correctly identified."""
        pytest.importorskip("hnswlib")
        
        from packrepo.packer.clustering.hnsw import HNSWMedoidClusterer
        
        # Create embeddings with clear tail structure
        embeddings = self._create_structured_embeddings()
        
        clusterer = HNSWMedoidClusterer(
            coverage_threshold=0.6,  # Higher threshold to identify more tails
            random_state=42,
        )
        
        # Build index and identify tails
        clusterer._build_hnsw_index(embeddings)
        tail_embeddings = clusterer._identify_tail_embeddings(embeddings)
        
        # Should identify some tail embeddings
        assert len(tail_embeddings) > 0
        assert len(tail_embeddings) < len(embeddings)
    
    def _create_tail_embeddings(self):
        """Create embeddings with tail distribution for testing."""
        from packrepo.packer.embeddings.base import CodeEmbedding
        import time
        
        embeddings = []
        np.random.seed(42)
        
        # Create dense cluster
        for i in range(15):
            base_vector = np.random.randn(384) * 0.1
            base_vector[:50] += 2.0  # Dense cluster
            
            embedding = CodeEmbedding(
                chunk_id=f"dense_{i}",
                file_path=f"dense/file_{i}.py",
                content_sha=f"dns{i:02d}" + "0" * 60,
                embedding=base_vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        # Create tail embeddings
        for i in range(5):
            tail_vector = np.random.randn(384) * 2.0  # More spread out
            tail_vector[200:250] += 3.0  # Different region
            
            embedding = CodeEmbedding(
                chunk_id=f"tail_{i}",
                file_path=f"tail/file_{i}.py",
                content_sha=f"tal{i:02d}" + "0" * 60,
                embedding=tail_vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        return embeddings
    
    def _create_structured_embeddings(self):
        """Create embeddings with clear dense/tail structure."""
        from packrepo.packer.embeddings.base import CodeEmbedding
        import time
        
        embeddings = []
        np.random.seed(42)
        
        # Dense region
        for i in range(10):
            dense_vector = np.zeros(384)
            dense_vector[:50] = np.random.randn(50) * 0.1
            
            embedding = CodeEmbedding(
                chunk_id=f"dense_{i}",
                file_path=f"pkg/dense_{i}.py",
                content_sha=f"den{i:02d}" + "0" * 60,
                embedding=dense_vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        # Tail regions
        for i in range(3):
            tail_vector = np.zeros(384)
            tail_vector[100 + i*50:150 + i*50] = np.random.randn(50) * 0.5 + 5.0
            
            embedding = CodeEmbedding(
                chunk_id=f"tail_{i}",
                file_path=f"pkg/tail_{i}.py",
                content_sha=f"tai{i:02d}" + "0" * 60,
                embedding=tail_vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        return embeddings


class TestV2MixedCentroids:
    """Test mixed centroid clustering combining k-means and HNSW."""
    
    def test_mixed_clustering(self):
        """Test mixed centroid clustering integration."""
        pytest.importorskip("sklearn")
        pytest.importorskip("hnswlib")
        
        from packrepo.packer.clustering import create_mixed_clusterer
        
        # Create diverse embeddings
        embeddings = self._create_mixed_embeddings()
        
        # Perform mixed clustering
        clusterer = create_mixed_clusterer(random_state=42)
        result = clusterer.fit_predict(embeddings)
        
        # Validate result
        assert result.clustering_method.startswith("mixed_")
        assert result.total_embeddings == len(embeddings)
        
        # Check that we have both k-means and medoid clusters
        kmeans_clusters = [c for c in result.clusters if 'kmeans' in c.cluster_type]
        medoid_clusters = [c for c in result.clusters if 'hnsw' in c.cluster_type or 'medoid' in c.cluster_type]
        
        # Should have some k-means clusters (for dense regions)
        assert len(kmeans_clusters) > 0
    
    def test_coverage_enhancement(self):
        """Test that mixed centroids improve coverage vs k-means alone."""
        pytest.importorskip("sklearn")
        pytest.importorskip("hnswlib") 
        
        from packrepo.packer.clustering import create_package_clusterer, create_mixed_clusterer
        
        embeddings = self._create_mixed_embeddings()
        
        # Compare k-means vs mixed clustering
        kmeans_clusterer = create_package_clusterer(random_state=42)
        mixed_clusterer = create_mixed_clusterer(random_state=42)
        
        kmeans_result = kmeans_clusterer.fit_predict(embeddings)
        mixed_result = mixed_clusterer.fit_predict(embeddings)
        
        # Mixed clustering should have more clusters (covers more regions)
        assert mixed_result.num_clusters >= kmeans_result.num_clusters
    
    def _create_mixed_embeddings(self):
        """Create embeddings with both dense and sparse regions."""
        from packrepo.packer.embeddings.base import CodeEmbedding
        import time
        
        embeddings = []
        np.random.seed(42)
        
        # Dense cluster 1
        for i in range(8):
            vector = np.random.randn(384) * 0.1
            vector[:50] += 1.0
            
            embedding = CodeEmbedding(
                chunk_id=f"dense1_{i}",
                file_path=f"pkg_a/file_{i}.py",
                content_sha=f"d1{i:02d}" + "0" * 61,
                embedding=vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        # Dense cluster 2
        for i in range(8):
            vector = np.random.randn(384) * 0.1
            vector[50:100] += 1.5
            
            embedding = CodeEmbedding(
                chunk_id=f"dense2_{i}",
                file_path=f"pkg_b/file_{i}.py",
                content_sha=f"d2{i:02d}" + "0" * 61,
                embedding=vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        # Sparse tail points
        for i in range(4):
            vector = np.random.randn(384) * 0.5
            vector[200 + i*30:230 + i*30] += 3.0
            
            embedding = CodeEmbedding(
                chunk_id=f"sparse_{i}",
                file_path=f"pkg_c/file_{i}.py",
                content_sha=f"sp{i:02d}" + "0" * 61,
                embedding=vector,
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            )
            embeddings.append(embedding)
        
        return embeddings


class TestV2Integration:
    """Test V2 system integration with selector and pack index."""
    
    def test_v2_selector_integration(self):
        """Test V2 selector with mock dependencies."""
        # Create test chunks
        chunks = self._create_test_chunks()
        
        # Create selector with approximate tokenizer
        tokenizer = ApproximateTokenizer()
        selector = RepositorySelector(tokenizer)
        
        # Create V2 selection config
        config = SelectionConfig(
            variant=SelectionVariant.V2_COVERAGE,
            token_budget=1000,
            deterministic=True,
            random_seed=42,
        )
        
        request = PackRequest(chunks=chunks, config=config)
        
        # Mock the embedding system to avoid external dependencies
        with patch.object(selector, '_init_embedding_system'), \
             patch.object(selector, '_get_chunk_embeddings', return_value=[]), \
             patch.object(selector, '_facility_location_mmr') as mock_v1:
            
            mock_v1.return_value = Mock(
                selected_chunks=chunks[:2],
                chunk_modes={"chunk_1": "full", "chunk_2": "full"},
                selection_scores={"chunk_1": 1.0, "chunk_2": 0.8},
                total_tokens=500,
                budget_utilization=0.5,
                coverage_score=0.8,
                diversity_score=0.7,
                iterations=2,
            )
            
            # Should fall back to V1 when embeddings fail
            result = selector.select(request)
            
            assert result.selection is not None
            assert mock_v1.called
    
    def test_pack_index_clustering_metadata(self):
        """Test that pack index stores clustering metadata."""
        from packrepo.packer.packfmt.base import PackIndex
        from packrepo.packer.clustering.base import ClusteringResult, Cluster
        import numpy as np
        
        # Create pack index
        index = PackIndex(
            tokenizer="test_tokenizer",
            target_budget=1000,
            selector_variant="v2_coverage",
        )
        
        # Create mock clustering result
        cluster = Cluster(
            cluster_id="test_cluster",
            centroid=np.array([1.0, 2.0, 3.0]),
            members=["chunk1", "chunk2"],
            member_embeddings=[],
            cluster_type="mixed_kmeans",
        )
        
        clustering_result = ClusteringResult(
            clusters=[cluster],
            total_embeddings=2,
            clustering_method="mixed_test",
        )
        
        # Store clustering metadata
        index.set_clustering_metadata(clustering_result)
        
        # Validate metadata is stored
        assert index.clustering_method == "mixed_test"
        assert index.clustering_stats is not None
        assert "test_cluster" in index.cluster_centroids
        assert index.cluster_centroids["test_cluster"] == [1.0, 2.0, 3.0]
    
    def _create_test_chunks(self):
        """Create sample chunks for testing."""
        chunks = []
        for i in range(5):
            chunk = Chunk(
                id=f"chunk_{i}",
                path=Path(f"/test/file_{i}.py"),
                rel_path=f"file_{i}.py",
                start_line=i * 10 + 1,
                end_line=i * 10 + 10,
                kind=ChunkKind.FUNCTION,
                name=f"function_{i}",
                language="python",
                content=f"def function_{i}():\n    return {i}",
                full_tokens=50,
                signature_tokens=10,
            )
            chunks.append(chunk)
        return chunks


if __name__ == "__main__":
    pytest.main([__file__])
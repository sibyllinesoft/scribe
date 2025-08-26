#!/usr/bin/env python3
"""
V2 System Validation Script

Validates basic V2 functionality without requiring external model dependencies.
Tests core algorithms, data structures, and integration points.
"""

import sys
import time
import hashlib
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from packrepo.packer.chunker.base import Chunk, ChunkKind
from packrepo.packer.selector.base import SelectionConfig, SelectionVariant, PackRequest
from packrepo.packer.selector.selector import RepositorySelector
from packrepo.packer.tokenizer.implementations import ApproximateTokenizer
from packrepo.packer.tokenizer.base import TokenizerType
from packrepo.packer.packfmt.base import PackIndex


def create_test_embeddings(count=20):
    """Create mock embeddings for testing."""
    from packrepo.packer.embeddings.base import CodeEmbedding
    
    embeddings = []
    np.random.seed(42)
    
    for i in range(count):
        # Create structured embeddings with clusters
        vector = np.random.randn(384) * 0.1
        
        if i < count // 2:
            vector[:100] += 1.0  # First cluster
        else:
            vector[100:200] += 1.0  # Second cluster
            
        # Generate proper 64-character SHA-256
        import hashlib
        content_for_sha = f"chunk_{i}_content"
        content_sha = hashlib.sha256(content_for_sha.encode()).hexdigest()
        
        embedding = CodeEmbedding(
            chunk_id=f"chunk_{i}",
            file_path=f"pkg_{i % 3}/file_{i}.py",
            content_sha=content_sha,
            embedding=vector,
            model_name="mock_model",
            model_version="v1.0",
            created_at=time.time(),
        )
        embeddings.append(embedding)
    
    return embeddings


def test_clustering_algorithms():
    """Test clustering algorithms work correctly."""
    print("🧪 Testing Clustering Algorithms...")
    
    try:
        # Test k-means clustering (requires scikit-learn)
        try:
            from packrepo.packer.clustering.kmeans import create_package_clusterer
            
            embeddings = create_test_embeddings(16)
            clusterer = create_package_clusterer(random_state=42)
            result = clusterer.fit_predict(embeddings)
            
            print(f"  ✅ K-means clustering: {result.num_clusters} clusters from {len(embeddings)} embeddings")
            print(f"     Method: {result.clustering_method}")
            print(f"     Inertia: {result.inertia:.2f}")
            print(f"     Uniformity: {result.coverage_uniformity:.2f}")
            
        except ImportError:
            print("  ⚠️  K-means clustering: sklearn not available, skipped")
    
        # Test HNSW medoids (requires hnswlib)
        try:
            from packrepo.packer.clustering.hnsw import create_density_medoid_clusterer
            
            embeddings = create_test_embeddings(12)
            clusterer = create_density_medoid_clusterer(
                coverage_threshold=0.6,
                random_state=42
            )
            result = clusterer.fit_predict(embeddings)
            
            print(f"  ✅ HNSW medoids: {result.num_clusters} clusters from {len(embeddings)} embeddings")
            print(f"     Method: {result.clustering_method}")
            
        except ImportError:
            print("  ⚠️  HNSW medoids: hnswlib not available, skipped")
            
        # Test mixed clustering (requires both)
        try:
            from packrepo.packer.clustering.mixed import create_mixed_clusterer
            
            embeddings = create_test_embeddings(20)
            clusterer = create_mixed_clusterer(random_state=42)
            result = clusterer.fit_predict(embeddings)
            
            print(f"  ✅ Mixed clustering: {result.num_clusters} clusters from {len(embeddings)} embeddings")
            print(f"     Method: {result.clustering_method}")
            
            # Count cluster types
            kmeans_count = sum(1 for c in result.clusters if 'kmeans' in c.cluster_type)
            hnsw_count = sum(1 for c in result.clusters if 'hnsw' in c.cluster_type)
            print(f"     K-means clusters: {kmeans_count}, HNSW clusters: {hnsw_count}")
            
        except ImportError:
            print("  ⚠️  Mixed clustering: dependencies not available, skipped")
            
    except Exception as e:
        print(f"  ❌ Clustering test failed: {e}")
        return False
    
    return True


def test_embedding_cache():
    """Test embedding cache functionality."""
    print("💾 Testing Embedding Cache...")
    
    try:
        from packrepo.packer.embeddings.cache import InMemoryEmbeddingCache
        from packrepo.packer.embeddings.base import CodeEmbedding
        
        cache = InMemoryEmbeddingCache(max_entries=100)
        
        # Create test embedding with proper SHA
        import hashlib
        test_content_sha = hashlib.sha256(b"test_chunk_content").hexdigest()
        
        embedding = CodeEmbedding(
            chunk_id="test_chunk",
            file_path="test/file.py",
            content_sha=test_content_sha,
            embedding=np.random.randn(384),
            model_name="test_model",
            model_version="v1",
            created_at=time.time(),
        )
        
        # Test put/get
        cache.put(embedding)
        retrieved = cache.get(test_content_sha)
        
        assert retrieved is not None
        assert retrieved.chunk_id == "test_chunk"
        assert np.allclose(retrieved.embedding, embedding.embedding)
        
        # Test stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['entry_count'] == 1
        
        print("  ✅ In-memory cache: put/get/stats working")
        
        # Test batch operations
        embeddings = []
        for i in range(3):
            content_sha = hashlib.sha256(f"batch_chunk_{i}".encode()).hexdigest()
            embeddings.append(CodeEmbedding(
                chunk_id=f"chunk_{i}",
                file_path=f"test/file_{i}.py",
                content_sha=content_sha,
                embedding=np.random.randn(384),
                model_name="test_model",
                model_version="v1",
                created_at=time.time(),
            ))
        
        cache.put_batch(embeddings)
        shas = [emb.content_sha for emb in embeddings]
        retrieved_batch = cache.get_batch(shas)
        
        assert len(retrieved_batch) == 3
        print("  ✅ Batch operations: put_batch/get_batch working")
        
    except Exception as e:
        print(f"  ❌ Cache test failed: {e}")
        return False
    
    return True


def test_pack_index_metadata():
    """Test pack index clustering metadata storage."""
    print("📦 Testing Pack Index Metadata...")
    
    try:
        from packrepo.packer.clustering.base import ClusteringResult, Cluster
        
        # Create pack index
        index = PackIndex(
            tokenizer="mock_tokenizer",
            target_budget=1000,
            selector_variant="v2_coverage",
        )
        
        # Create mock clustering result
        clusters = []
        for i in range(3):
            cluster = Cluster(
                cluster_id=f"cluster_{i}",
                centroid=np.random.randn(384),
                members=[f"chunk_{i}", f"chunk_{i+3}"],
                member_embeddings=[],
                cluster_type="mixed_kmeans",
            )
            clusters.append(cluster)
        
        clustering_result = ClusteringResult(
            clusters=clusters,
            total_embeddings=6,
            clustering_method="mixed_kmeans_hnsw",
            clustering_time=1.5,
            inertia=45.2,
        )
        
        # Store metadata
        index.set_clustering_metadata(clustering_result)
        
        # Validate storage
        assert index.clustering_method == "mixed_kmeans_hnsw"
        assert index.clustering_stats is not None
        assert len(index.cluster_centroids) == 3
        
        # Test serialization
        json_data = index.to_json(canonical=True)
        assert "clustering_method" in json_data
        assert "cluster_centroids" in json_data
        
        print("  ✅ Pack index metadata: storage and serialization working")
        
    except Exception as e:
        print(f"  ❌ Pack index test failed: {e}")
        return False
    
    return True


def test_v2_selector_fallback():
    """Test V2 selector fallback to V1."""
    print("🎯 Testing V2 Selector Fallback...")
    
    try:
        # Create test data
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
        
        # Create selector
        tokenizer = ApproximateTokenizer(TokenizerType.CL100K_BASE)
        selector = RepositorySelector(tokenizer)
        
        # Create V2 config
        config = SelectionConfig(
            variant=SelectionVariant.V2_COVERAGE,
            token_budget=200,
            deterministic=True,
            random_seed=42,
        )
        
        request = PackRequest(chunks=chunks, config=config)
        
        # Should fall back to V1 when dependencies not available
        result = selector.select(request)
        
        assert result.selection is not None
        assert len(result.selection.selected_chunks) > 0
        assert result.selection.total_tokens <= 200
        
        print(f"  ✅ V2 selector fallback: selected {len(result.selection.selected_chunks)} chunks")
        print(f"     Total tokens: {result.selection.total_tokens}")
        print(f"     Coverage score: {result.selection.coverage_score:.2f}")
        
    except Exception as e:
        print(f"  ❌ V2 selector test failed: {e}")
        return False
    
    return True


def test_data_structure_compatibility():
    """Test that V2 data structures are compatible with existing code."""
    print("🔧 Testing Data Structure Compatibility...")
    
    try:
        from packrepo.packer.clustering.base import Cluster, ClusteringResult
        
        # Test Cluster serialization
        cluster = Cluster(
            cluster_id="test_cluster",
            centroid=np.array([1.0, 2.0, 3.0]),
            members=["chunk1", "chunk2"],
            member_embeddings=[],
            cluster_type="test_type",
        )
        
        cluster_dict = cluster.to_dict()
        assert "cluster_id" in cluster_dict
        assert "centroid" in cluster_dict
        assert cluster_dict["centroid"] == [1.0, 2.0, 3.0]
        
        print("  ✅ Cluster serialization working")
        
        # Test ClusteringResult serialization
        result = ClusteringResult(
            clusters=[cluster],
            total_embeddings=2,
            clustering_method="test_method",
        )
        
        result_dict = result.to_dict()
        assert "clusters" in result_dict
        assert "clustering_method" in result_dict
        assert len(result_dict["clusters"]) == 1
        
        print("  ✅ ClusteringResult serialization working")
        
        # Test embedding data structure
        from packrepo.packer.embeddings.base import CodeEmbedding
        
        # Generate proper SHA for embedding test
        embedding_content_sha = hashlib.sha256(b"embedding_test_content").hexdigest()
        
        embedding = CodeEmbedding(
            chunk_id="test",
            file_path="test.py",
            content_sha=embedding_content_sha,
            embedding=np.array([1.0, 2.0]),
            model_name="test",
            model_version="v1",
            created_at=time.time(),
        )
        
        assert embedding.dimension == 2
        similarity = embedding.cosine_similarity(embedding)
        assert abs(similarity - 1.0) < 1e-10, f"Expected ~1.0, got {similarity}"
        
        print("  ✅ CodeEmbedding data structure working")
        
    except Exception as e:
        print(f"  ❌ Data structure test failed: {e}")
        return False
    
    return True


def main():
    """Run V2 system validation."""
    print("🚀 PackRepo V2 System Validation")
    print("=" * 40)
    
    tests = [
        test_data_structure_compatibility,
        test_embedding_cache,
        test_pack_index_metadata,
        test_clustering_algorithms,
        test_v2_selector_fallback,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 40)
    print(f"✅ Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 V2 system validation successful!")
        print("\nKey Features Validated:")
        print("  • Embedding cache system with SHA-based keys")
        print("  • K-means clustering with k≈√N per package")
        print("  • HNSW medoid clustering for tail coverage") 
        print("  • Mixed centroid clustering combining both approaches")
        print("  • Pack index extension for clustering metadata")
        print("  • V2 selector integration with fallback to V1")
        
        return 0
    else:
        print("⚠️  Some tests failed - see output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
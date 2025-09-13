"""
K-means clustering implementation for V2 system.

Provides k-means clustering with k≈√N per package logic for creating
coverage centroids with quality metrics and validation.
"""

import time
import tracemalloc
from typing import List, Dict, Optional, Any
import numpy as np
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .base import Cluster, ClusteringResult, ClusteringProvider
from ..embeddings.base import CodeEmbedding


class KMeansClusterer(ClusteringProvider):
    """
    K-means clustering provider with per-package optimization.
    
    Implements k≈√N clustering per package as specified in TODO.md,
    with comprehensive quality metrics and deterministic behavior.
    """
    
    def __init__(
        self,
        per_package: bool = True,
        min_k: int = 1,
        max_k: Optional[int] = None,
        random_state: int = 42,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
    ):
        """
        Initialize k-means clusterer.
        
        Args:
            per_package: Whether to cluster per package separately
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters (None for no limit)
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for k-means
            tol: Convergence tolerance
            n_init: Number of random initializations
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for KMeansClusterer. "
                "Install with: pip install scikit-learn"
            )
        
        self.per_package = per_package
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
    
    def get_method_name(self) -> str:
        """Get the clustering method name."""
        return "kmeans_per_package" if self.per_package else "kmeans_global"
    
    def fit_predict(self, embeddings: List[CodeEmbedding]) -> ClusteringResult:
        """
        Perform k-means clustering on embeddings.
        
        Args:
            embeddings: List of code embeddings to cluster
            
        Returns:
            ClusteringResult with k-means clusters and quality metrics
        """
        if not embeddings:
            return ClusteringResult(
                clusters=[],
                total_embeddings=0,
                clustering_method=self.get_method_name(),
            )
        
        # Start performance tracking
        start_time = time.time()
        tracemalloc.start()
        
        try:
            if self.per_package:
                result = self._cluster_per_package(embeddings)
            else:
                result = self._cluster_global(embeddings)
            
            # Calculate quality metrics
            self._calculate_quality_metrics(result, embeddings)
            
            # Add performance metadata
            result.clustering_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            result.memory_peak_mb = peak / 1024 / 1024
            
            return result
            
        finally:
            tracemalloc.stop()
    
    def _cluster_per_package(self, embeddings: List[CodeEmbedding]) -> ClusteringResult:
        """
        Perform k-means clustering separately for each package.
        
        Uses k≈√N heuristic per package as specified in TODO.md.
        """
        # Group embeddings by package
        package_groups = self.group_embeddings_by_package(embeddings)
        
        all_clusters = []
        total_inertia = 0.0
        
        for package_path, package_embeddings in package_groups.items():
            if len(package_embeddings) < self.min_k:
                # Create single cluster for small packages
                cluster = self._create_single_cluster(
                    package_embeddings, 
                    f"{package_path}_single",
                    package_path
                )
                all_clusters.append(cluster)
                continue
            
            # Calculate optimal k for this package using √N rule
            k = self.calculate_optimal_k(package_embeddings, self.max_k)
            k = max(self.min_k, k)
            k = min(k, len(package_embeddings))
            
            # Perform k-means for this package
            package_clusters, inertia = self._run_kmeans(
                package_embeddings, k, package_path
            )
            
            all_clusters.extend(package_clusters)
            total_inertia += inertia
        
        return ClusteringResult(
            clusters=all_clusters,
            total_embeddings=len(embeddings),
            clustering_method=self.get_method_name(),
            inertia=total_inertia,
        )
    
    def _cluster_global(self, embeddings: List[CodeEmbedding]) -> ClusteringResult:
        """
        Perform global k-means clustering across all embeddings.
        """
        # Calculate global k using √N rule
        k = self.calculate_optimal_k(embeddings, self.max_k)
        k = max(self.min_k, k)
        k = min(k, len(embeddings))
        
        # Perform k-means
        clusters, inertia = self._run_kmeans(embeddings, k, package_path=None)
        
        return ClusteringResult(
            clusters=clusters,
            total_embeddings=len(embeddings),
            clustering_method=self.get_method_name(),
            inertia=inertia,
        )
    
    def _run_kmeans(
        self,
        embeddings: List[CodeEmbedding],
        k: int,
        package_path: Optional[str]
    ) -> tuple[List[Cluster], float]:
        """
        Run k-means clustering on a group of embeddings.
        
        Args:
            embeddings: Embeddings to cluster
            k: Number of clusters
            package_path: Package path for cluster metadata
            
        Returns:
            Tuple of (clusters, total_inertia)
        """
        # Prepare embedding matrix
        X = self.prepare_embeddings_matrix(embeddings)
        
        # Run k-means
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
        )
        
        labels = kmeans.fit_predict(X)
        
        # Create clusters from results
        clusters = []
        for cluster_idx in range(k):
            cluster_mask = labels == cluster_idx
            cluster_embeddings = [emb for i, emb in enumerate(embeddings) if cluster_mask[i]]
            
            if not cluster_embeddings:
                continue  # Skip empty clusters
            
            # Create cluster
            cluster_id = f"{package_path or 'global'}_kmeans_{cluster_idx}"
            centroid = kmeans.cluster_centers_[cluster_idx]
            
            # Calculate cluster inertia (within-cluster sum of squares)
            cluster_inertia = 0.0
            for emb in cluster_embeddings:
                distance_squared = np.sum((emb.embedding - centroid) ** 2)
                cluster_inertia += distance_squared
            
            cluster = Cluster(
                cluster_id=cluster_id,
                centroid=centroid,
                members=[emb.content_sha for emb in cluster_embeddings],
                member_embeddings=cluster_embeddings,
                cluster_type="kmeans",
                inertia=cluster_inertia,
                package_path=package_path,
            )
            
            clusters.append(cluster)
        
        return clusters, kmeans.inertia_
    
    def _create_single_cluster(
        self,
        embeddings: List[CodeEmbedding],
        cluster_id: str,
        package_path: Optional[str]
    ) -> Cluster:
        """
        Create a single cluster from embeddings (for small groups).
        """
        if not embeddings:
            raise ValueError("Cannot create cluster from empty embeddings")
        
        # Calculate centroid as mean of embeddings
        embedding_matrix = self.prepare_embeddings_matrix(embeddings)
        centroid = np.mean(embedding_matrix, axis=0)
        
        # Calculate inertia
        inertia = 0.0
        for emb in embeddings:
            distance_squared = np.sum((emb.embedding - centroid) ** 2)
            inertia += distance_squared
        
        return Cluster(
            cluster_id=cluster_id,
            centroid=centroid,
            members=[emb.content_sha for emb in embeddings],
            member_embeddings=embeddings,
            cluster_type="kmeans_single",
            inertia=inertia,
            package_path=package_path,
        )
    
    def _calculate_quality_metrics(self, result: ClusteringResult, embeddings: List[CodeEmbedding]):
        """
        Calculate clustering quality metrics.
        
        Adds silhouette score, Calinski-Harabasz index, and Davies-Bouldin index
        to the clustering result for quality assessment.
        """
        if result.num_clusters <= 1 or len(embeddings) <= 1:
            # Cannot calculate metrics for single cluster or single point
            return
        
        try:
            # Prepare data
            X = self.prepare_embeddings_matrix(embeddings)
            
            # Create label array from clusters
            labels = np.zeros(len(embeddings), dtype=int)
            embedding_to_idx = {emb.content_sha: i for i, emb in enumerate(embeddings)}
            
            for cluster_idx, cluster in enumerate(result.clusters):
                for content_sha in cluster.members:
                    if content_sha in embedding_to_idx:
                        labels[embedding_to_idx[content_sha]] = cluster_idx
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1 and len(X) > 1:
                result.silhouette_score = silhouette_score(X, labels)
            
            # Calculate Calinski-Harabasz index
            if len(np.unique(labels)) > 1:
                result.calinski_harabasz_score = calinski_harabasz_score(X, labels)
            
            # Calculate Davies-Bouldin index
            if len(np.unique(labels)) > 1:
                result.davies_bouldin_score = davies_bouldin_score(X, labels)
            
            # Calculate coverage uniformity
            cluster_sizes = [cluster.size for cluster in result.clusters]
            if cluster_sizes:
                size_variance = np.var(cluster_sizes)
                mean_size = np.mean(cluster_sizes)
                # Normalize variance by mean to get coefficient of variation
                result.coverage_uniformity = 1.0 - (size_variance / max(mean_size, 1.0))
                result.coverage_uniformity = max(0.0, result.coverage_uniformity)
        
        except Exception as e:
            # If metric calculation fails, log but don't fail clustering
            print(f"Warning: Could not calculate quality metrics: {e}")


class MiniBatchKMeansClusterer(KMeansClusterer):
    """
    Mini-batch k-means clusterer for large datasets.
    
    Uses scikit-learn's MiniBatchKMeans for improved scalability
    when dealing with large numbers of embeddings.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        **kwargs
    ):
        """
        Initialize mini-batch k-means clusterer.
        
        Args:
            batch_size: Size of mini-batches
            **kwargs: Arguments passed to parent KMeansClusterer
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
    
    def get_method_name(self) -> str:
        """Get the clustering method name."""
        base_name = super().get_method_name()
        return f"minibatch_{base_name}"
    
    def _run_kmeans(
        self,
        embeddings: List[CodeEmbedding],
        k: int,
        package_path: Optional[str]
    ) -> tuple[List[Cluster], float]:
        """
        Run mini-batch k-means clustering.
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            # Fall back to regular k-means
            return super()._run_kmeans(embeddings, k, package_path)
        
        # Prepare embedding matrix
        X = self.prepare_embeddings_matrix(embeddings)
        
        # Run mini-batch k-means
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(self.batch_size, len(embeddings)),
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
        )
        
        labels = kmeans.fit_predict(X)
        
        # Create clusters from results
        clusters = []
        total_inertia = 0.0
        
        for cluster_idx in range(k):
            cluster_mask = labels == cluster_idx
            cluster_embeddings = [emb for i, emb in enumerate(embeddings) if cluster_mask[i]]
            
            if not cluster_embeddings:
                continue
            
            # Create cluster
            cluster_id = f"{package_path or 'global'}_minibatch_kmeans_{cluster_idx}"
            centroid = kmeans.cluster_centers_[cluster_idx]
            
            # Calculate cluster inertia manually (MiniBatchKMeans doesn't provide it)
            cluster_inertia = 0.0
            for emb in cluster_embeddings:
                distance_squared = np.sum((emb.embedding - centroid) ** 2)
                cluster_inertia += distance_squared
            
            total_inertia += cluster_inertia
            
            cluster = Cluster(
                cluster_id=cluster_id,
                centroid=centroid,
                members=[emb.content_sha for emb in cluster_embeddings],
                member_embeddings=cluster_embeddings,
                cluster_type="minibatch_kmeans",
                inertia=cluster_inertia,
                package_path=package_path,
            )
            
            clusters.append(cluster)
        
        return clusters, total_inertia


# Convenience factory functions
def create_package_clusterer(**kwargs) -> KMeansClusterer:
    """Create k-means clusterer optimized for per-package clustering."""
    defaults = {
        'per_package': True,
        'min_k': 1,
        'max_k': 50,  # Reasonable upper limit for package clustering
        'random_state': 42,
    }
    defaults.update(kwargs)
    return KMeansClusterer(**defaults)


def create_global_clusterer(**kwargs) -> KMeansClusterer:
    """Create k-means clusterer for global clustering."""
    defaults = {
        'per_package': False,
        'min_k': 2,
        'max_k': 100,  # Higher limit for global clustering
        'random_state': 42,
    }
    defaults.update(kwargs)
    return KMeansClusterer(**defaults)


def create_scalable_clusterer(**kwargs) -> MiniBatchKMeansClusterer:
    """Create mini-batch k-means clusterer for large datasets."""
    defaults = {
        'per_package': True,
        'batch_size': 100,
        'min_k': 1,
        'max_k': 50,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return MiniBatchKMeansClusterer(**defaults)
"""
Mixed centroid clustering implementation for V2 system.

Combines k-means clustering for dense regions with HNSW medoids for 
long-tail coverage to create comprehensive mixed centroids as specified
in the TODO.md requirements.
"""

import time
import tracemalloc
from typing import List, Dict, Optional, Any, Set
import numpy as np

from .base import Cluster, ClusteringResult, ClusteringProvider
from .kmeans import KMeansClusterer, create_package_clusterer
from .hnsw import HNSWMedoidClusterer, create_density_medoid_clusterer
from ..embeddings.base import CodeEmbedding


class MixedCentroidClusterer(ClusteringProvider):
    """
    Mixed centroid clustering provider combining k-means and HNSW medoids.
    
    Implements the V2 coverage enhancement strategy by:
    1. Using k-means clustering (k≈√N per package) for dense regions
    2. Using HNSW medoids for long-tail coverage of sparse regions  
    3. Combining both approaches for comprehensive mixed centroids
    """
    
    def __init__(
        self,
        kmeans_clusterer: Optional[KMeansClusterer] = None,
        hnsw_clusterer: Optional[HNSWMedoidClusterer] = None,
        tail_threshold: float = 0.5,  # Coverage threshold for tail detection
        overlap_resolution: str = 'prefer_kmeans',  # 'prefer_kmeans', 'prefer_hnsw', 'merge'
        min_tail_ratio: float = 0.1,  # Minimum ratio of embeddings for tail clustering
        max_clusters: Optional[int] = None,  # Maximum total clusters
        random_state: int = 42,
    ):
        """
        Initialize mixed centroid clusterer.
        
        Args:
            kmeans_clusterer: K-means clusterer for dense regions (None for default)
            hnsw_clusterer: HNSW clusterer for tail coverage (None for default)
            tail_threshold: Minimum coverage score to not be considered tail
            overlap_resolution: How to handle embeddings covered by both methods
            min_tail_ratio: Minimum fraction of embeddings to consider tail clustering
            max_clusters: Maximum total clusters from both methods
            random_state: Random seed for reproducibility
        """
        # Initialize clusterers with defaults if not provided
        if kmeans_clusterer is None:
            kmeans_clusterer = create_package_clusterer(random_state=random_state)
        
        if hnsw_clusterer is None:
            hnsw_clusterer = create_density_medoid_clusterer(
                coverage_threshold=tail_threshold,
                random_state=random_state,
            )
        
        self.kmeans_clusterer = kmeans_clusterer
        self.hnsw_clusterer = hnsw_clusterer
        self.tail_threshold = tail_threshold
        self.overlap_resolution = overlap_resolution
        self.min_tail_ratio = min_tail_ratio
        self.max_clusters = max_clusters
        self.random_state = random_state
    
    def get_method_name(self) -> str:
        """Get the clustering method name."""
        kmeans_name = self.kmeans_clusterer.get_method_name()
        hnsw_name = self.hnsw_clusterer.get_method_name()
        return f"mixed_{kmeans_name}_{hnsw_name}"
    
    def fit_predict(self, embeddings: List[CodeEmbedding]) -> ClusteringResult:
        """
        Perform mixed centroid clustering on embeddings.
        
        Args:
            embeddings: List of code embeddings to cluster
            
        Returns:
            ClusteringResult with mixed centroids from k-means and HNSW
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
            # Step 1: Perform k-means clustering for dense regions
            kmeans_result = self.kmeans_clusterer.fit_predict(embeddings)
            
            # Step 2: Identify tail embeddings poorly covered by k-means
            tail_embeddings = self._identify_tail_embeddings(embeddings, kmeans_result)
            
            # Step 3: Perform HNSW medoid clustering on tail embeddings (if sufficient)
            hnsw_result = None
            if len(tail_embeddings) >= len(embeddings) * self.min_tail_ratio:
                hnsw_result = self.hnsw_clusterer.fit_predict(tail_embeddings)
            
            # Step 4: Combine results into mixed centroids
            combined_result = self._combine_clustering_results(
                embeddings, kmeans_result, hnsw_result
            )
            
            # Step 5: Calculate comprehensive quality metrics
            self._calculate_mixed_quality_metrics(combined_result, embeddings, kmeans_result, hnsw_result)
            
            # Add performance metadata
            combined_result.clustering_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            combined_result.memory_peak_mb = peak / 1024 / 1024
            
            return combined_result
            
        finally:
            tracemalloc.stop()
    
    def _identify_tail_embeddings(
        self,
        embeddings: List[CodeEmbedding],
        kmeans_result: ClusteringResult
    ) -> List[CodeEmbedding]:
        """
        Identify embeddings poorly covered by k-means clusters.
        
        Uses distance to nearest k-means centroid as coverage metric.
        Embeddings with poor coverage are candidates for HNSW medoid clustering.
        """
        if not kmeans_result.clusters:
            return embeddings  # If k-means failed, all embeddings are tail
        
        tail_embeddings = []
        
        # Create embedding to content_sha mapping for efficient lookup
        embedding_map = {emb.content_sha: emb for emb in embeddings}
        
        for embedding in embeddings:
            coverage_score = self._calculate_kmeans_coverage(embedding, kmeans_result.clusters)
            
            if coverage_score < self.tail_threshold:
                tail_embeddings.append(embedding)
        
        return tail_embeddings
    
    def _calculate_kmeans_coverage(
        self,
        embedding: CodeEmbedding,
        kmeans_clusters: List[Cluster]
    ) -> float:
        """
        Calculate how well an embedding is covered by k-means clusters.
        
        Returns coverage score based on distance to nearest cluster centroid.
        Higher scores indicate better coverage by k-means.
        """
        if not kmeans_clusters:
            return 0.0
        
        min_distance = float('inf')
        
        for cluster in kmeans_clusters:
            distance = np.linalg.norm(embedding.embedding - cluster.centroid)
            min_distance = min(min_distance, distance)
        
        # Convert distance to coverage score (closer = better coverage)
        coverage_score = 1.0 / (1.0 + min_distance)
        
        return coverage_score
    
    def _combine_clustering_results(
        self,
        embeddings: List[CodeEmbedding],
        kmeans_result: ClusteringResult,
        hnsw_result: Optional[ClusteringResult]
    ) -> ClusteringResult:
        """
        Combine k-means and HNSW clustering results into mixed centroids.
        
        Handles overlap resolution and creates unified clustering result.
        """
        combined_clusters = []
        
        # Add k-means clusters
        for cluster in kmeans_result.clusters:
            # Mark as mixed type to indicate origin
            cluster.cluster_type = "mixed_kmeans"
            combined_clusters.append(cluster)
        
        # Add HNSW medoid clusters if available
        if hnsw_result and hnsw_result.clusters:
            for cluster in hnsw_result.clusters:
                # Mark as mixed type to indicate origin
                cluster.cluster_type = "mixed_hnsw_medoid"
                
                # Handle overlap resolution
                resolved_cluster = self._resolve_cluster_overlap(
                    cluster, combined_clusters
                )
                
                if resolved_cluster is not None:
                    combined_clusters.append(resolved_cluster)
        
        # Apply cluster limit if specified
        if self.max_clusters is not None and len(combined_clusters) > self.max_clusters:
            combined_clusters = self._limit_clusters(combined_clusters, self.max_clusters)
        
        # Create combined result
        return ClusteringResult(
            clusters=combined_clusters,
            total_embeddings=len(embeddings),
            clustering_method=self.get_method_name(),
            clustering_time=kmeans_result.clustering_time + (hnsw_result.clustering_time if hnsw_result else 0.0),
        )
    
    def _resolve_cluster_overlap(
        self,
        hnsw_cluster: Cluster,
        existing_clusters: List[Cluster]
    ) -> Optional[Cluster]:
        """
        Resolve overlap between HNSW cluster and existing k-means clusters.
        
        Implements the overlap resolution strategy specified in constructor.
        """
        if self.overlap_resolution == 'prefer_kmeans':
            # Remove embeddings already covered by k-means clusters
            return self._filter_overlapping_members(hnsw_cluster, existing_clusters)
        
        elif self.overlap_resolution == 'prefer_hnsw':
            # Keep HNSW cluster as-is, potentially duplicating coverage
            return hnsw_cluster
        
        elif self.overlap_resolution == 'merge':
            # Find overlapping k-means cluster and merge if significant overlap
            return self._merge_if_overlapping(hnsw_cluster, existing_clusters)
        
        else:
            raise ValueError(f"Unknown overlap resolution strategy: {self.overlap_resolution}")
    
    def _filter_overlapping_members(
        self,
        hnsw_cluster: Cluster,
        existing_clusters: List[Cluster]
    ) -> Optional[Cluster]:
        """
        Filter out members of HNSW cluster that are already in k-means clusters.
        """
        # Get set of all content SHAs already in existing clusters
        existing_members = set()
        for cluster in existing_clusters:
            existing_members.update(cluster.members)
        
        # Filter HNSW cluster members
        filtered_members = []
        filtered_embeddings = []
        
        for i, content_sha in enumerate(hnsw_cluster.members):
            if content_sha not in existing_members:
                filtered_members.append(content_sha)
                filtered_embeddings.append(hnsw_cluster.member_embeddings[i])
        
        # Return filtered cluster if it has enough members
        if len(filtered_members) >= self.hnsw_clusterer.min_cluster_size:
            # Recalculate centroid and properties
            if filtered_embeddings:
                embedding_matrix = np.stack([emb.embedding for emb in filtered_embeddings])
                new_centroid = np.mean(embedding_matrix, axis=0)
                
                # Calculate new inertia
                new_inertia = 0.0
                for emb in filtered_embeddings:
                    distance_squared = np.sum((emb.embedding - new_centroid) ** 2)
                    new_inertia += distance_squared
                
                return Cluster(
                    cluster_id=hnsw_cluster.cluster_id + "_filtered",
                    centroid=new_centroid,
                    members=filtered_members,
                    member_embeddings=filtered_embeddings,
                    cluster_type="mixed_hnsw_medoid_filtered",
                    inertia=new_inertia,
                    package_path=hnsw_cluster.package_path,
                )
        
        return None  # Cluster too small after filtering
    
    def _merge_if_overlapping(
        self,
        hnsw_cluster: Cluster,
        existing_clusters: List[Cluster]
    ) -> Optional[Cluster]:
        """
        Merge HNSW cluster with overlapping k-means cluster if significant overlap.
        """
        hnsw_members = set(hnsw_cluster.members)
        
        # Find k-means cluster with highest overlap
        best_overlap_ratio = 0.0
        best_cluster_idx = -1
        
        for i, cluster in enumerate(existing_clusters):
            cluster_members = set(cluster.members)
            overlap = hnsw_members.intersection(cluster_members)
            overlap_ratio = len(overlap) / max(len(hnsw_members), 1)
            
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_cluster_idx = i
        
        # Merge if overlap is significant (>30%)
        if best_overlap_ratio > 0.3 and best_cluster_idx >= 0:
            return self._merge_clusters(hnsw_cluster, existing_clusters[best_cluster_idx])
        
        # Otherwise keep as separate cluster
        return hnsw_cluster
    
    def _merge_clusters(self, cluster1: Cluster, cluster2: Cluster) -> Cluster:
        """
        Merge two clusters into a single cluster.
        """
        # Combine members (remove duplicates)
        combined_members = list(set(cluster1.members + cluster2.members))
        combined_embeddings = cluster1.member_embeddings + cluster2.member_embeddings
        
        # Remove duplicate embeddings
        seen_shas = set()
        unique_embeddings = []
        for emb in combined_embeddings:
            if emb.content_sha not in seen_shas:
                unique_embeddings.append(emb)
                seen_shas.add(emb.content_sha)
        
        # Calculate new centroid
        if unique_embeddings:
            embedding_matrix = np.stack([emb.embedding for emb in unique_embeddings])
            new_centroid = np.mean(embedding_matrix, axis=0)
            
            # Calculate new inertia
            new_inertia = 0.0
            for emb in unique_embeddings:
                distance_squared = np.sum((emb.embedding - new_centroid) ** 2)
                new_inertia += distance_squared
            
            return Cluster(
                cluster_id=f"{cluster1.cluster_id}_merged_{cluster2.cluster_id}",
                centroid=new_centroid,
                members=[emb.content_sha for emb in unique_embeddings],
                member_embeddings=unique_embeddings,
                cluster_type="mixed_merged",
                inertia=new_inertia,
                package_path=cluster1.package_path or cluster2.package_path,
            )
        
        # Fallback to cluster1 if merge fails
        return cluster1
    
    def _limit_clusters(self, clusters: List[Cluster], max_clusters: int) -> List[Cluster]:
        """
        Limit the number of clusters by keeping the highest quality ones.
        
        Uses cluster size and inertia as quality metrics.
        """
        if len(clusters) <= max_clusters:
            return clusters
        
        # Score clusters by quality (size / inertia ratio)
        cluster_scores = []
        for cluster in clusters:
            # Higher score = better quality
            quality_score = cluster.size / max(cluster.inertia, 1.0)
            cluster_scores.append((quality_score, cluster))
        
        # Sort by score (descending) and take top clusters
        cluster_scores.sort(key=lambda x: x[0], reverse=True)
        return [cluster for score, cluster in cluster_scores[:max_clusters]]
    
    def _calculate_mixed_quality_metrics(
        self,
        combined_result: ClusteringResult,
        embeddings: List[CodeEmbedding],
        kmeans_result: ClusteringResult,
        hnsw_result: Optional[ClusteringResult]
    ):
        """
        Calculate comprehensive quality metrics for mixed clustering.
        
        Combines metrics from both clustering methods and adds mixed-specific metrics.
        """
        # Basic metrics
        combined_result.inertia = sum(cluster.inertia for cluster in combined_result.clusters)
        
        # Coverage metrics
        cluster_sizes = [cluster.size for cluster in combined_result.clusters]
        if cluster_sizes:
            size_variance = np.var(cluster_sizes)
            mean_size = np.mean(cluster_sizes)
            combined_result.coverage_uniformity = 1.0 - min(1.0, size_variance / max(mean_size, 1.0))
        
        # Mixed-specific metrics
        kmeans_clusters = len([c for c in combined_result.clusters if 'kmeans' in c.cluster_type])
        hnsw_clusters = len([c for c in combined_result.clusters if 'hnsw' in c.cluster_type])
        
        # Calculate coverage ratios
        kmeans_coverage = sum(c.size for c in combined_result.clusters if 'kmeans' in c.cluster_type)
        hnsw_coverage = sum(c.size for c in combined_result.clusters if 'hnsw' in c.cluster_type)
        
        # Store mixed clustering metrics in package_coverage dict
        combined_result.package_coverage.update({
            'kmeans_clusters': kmeans_clusters,
            'hnsw_clusters': hnsw_clusters,
            'kmeans_coverage_ratio': kmeans_coverage / max(len(embeddings), 1),
            'hnsw_coverage_ratio': hnsw_coverage / max(len(embeddings), 1),
            'tail_embeddings_identified': len(embeddings) - kmeans_coverage if hnsw_result else 0,
        })
        
        # Inherit quality metrics from component clustering methods
        if kmeans_result:
            combined_result.silhouette_score = kmeans_result.silhouette_score
            combined_result.calinski_harabasz_score = kmeans_result.calinski_harabasz_score
            combined_result.davies_bouldin_score = kmeans_result.davies_bouldin_score


# Convenience factory functions
def create_mixed_clusterer(**kwargs) -> MixedCentroidClusterer:
    """Create default mixed centroid clusterer."""
    defaults = {
        'tail_threshold': 0.5,
        'overlap_resolution': 'prefer_kmeans',
        'min_tail_ratio': 0.1,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return MixedCentroidClusterer(**defaults)


def create_comprehensive_clusterer(**kwargs) -> MixedCentroidClusterer:
    """Create mixed clusterer optimized for comprehensive coverage."""
    defaults = {
        'tail_threshold': 0.4,  # Lower threshold = more tail coverage
        'overlap_resolution': 'merge',
        'min_tail_ratio': 0.05,  # Allow tail clustering even for small ratios
        'max_clusters': 100,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return MixedCentroidClusterer(**defaults)


def create_efficient_clusterer(**kwargs) -> MixedCentroidClusterer:
    """Create mixed clusterer optimized for efficiency."""
    defaults = {
        'tail_threshold': 0.6,  # Higher threshold = less tail clustering
        'overlap_resolution': 'prefer_kmeans',
        'min_tail_ratio': 0.2,  # Require substantial tail ratio
        'max_clusters': 50,
        'random_state': 42,
    }
    defaults.update(kwargs)
    
    # Use faster component clusterers
    from .kmeans import create_package_clusterer, MiniBatchKMeansClusterer
    from .hnsw import create_fast_medoid_clusterer
    
    kmeans_clusterer = MiniBatchKMeansClusterer(
        per_package=True,
        batch_size=50,
        random_state=42,
    )
    
    hnsw_clusterer = create_fast_medoid_clusterer(random_state=42)
    
    return MixedCentroidClusterer(
        kmeans_clusterer=kmeans_clusterer,
        hnsw_clusterer=hnsw_clusterer,
        **defaults
    )
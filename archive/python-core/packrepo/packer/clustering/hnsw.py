"""
HNSW medoids implementation for V2 system.

Implements approximate nearest neighbor search using HNSW (Hierarchical 
Navigable Small World) graphs for handling long-tail coverage with medoid-based
clustering for improved coverage of sparse regions.
"""

import time
import tracemalloc
from typing import List, Dict, Optional, Any, Set, Tuple
import numpy as np
from collections import defaultdict

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

from .base import Cluster, ClusteringResult, ClusteringProvider
from ..embeddings.base import CodeEmbedding


class HNSWMedoidClusterer(ClusteringProvider):
    """
    HNSW-based medoid clustering provider for long-tail coverage.
    
    Uses approximate nearest neighbor search to identify sparse regions
    and creates medoid-based clusters to improve coverage of tail embeddings
    that are poorly covered by k-means centroids.
    """
    
    def __init__(
        self,
        ef_construction: int = 200,
        ef_search: int = 50,
        M: int = 16,
        max_elements: int = 100000,
        medoid_selection: str = 'density_based',  # 'random', 'density_based', 'distance_based'
        coverage_threshold: float = 0.5,  # Minimum coverage score to not be considered tail
        min_cluster_size: int = 2,
        max_medoids: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize HNSW medoid clusterer.
        
        Args:
            ef_construction: Size of the dynamic candidate list during construction
            ef_search: Size of the dynamic candidate list during search
            M: Number of bi-directional links for every new element added
            max_elements: Maximum number of elements that can be stored
            medoid_selection: Strategy for selecting medoids ('random', 'density_based', 'distance_based')
            coverage_threshold: Threshold below which points are considered tail
            min_cluster_size: Minimum size for medoid clusters
            max_medoids: Maximum number of medoids (None for no limit)
            random_state: Random seed for reproducibility
        """
        if not HNSWLIB_AVAILABLE:
            raise ImportError(
                "hnswlib is required for HNSWMedoidClusterer. "
                "Install with: pip install hnswlib"
            )
        
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        self.max_elements = max_elements
        self.medoid_selection = medoid_selection
        self.coverage_threshold = coverage_threshold
        self.min_cluster_size = min_cluster_size
        self.max_medoids = max_medoids
        self.random_state = random_state
        
        # HNSW index will be created during clustering
        self.index: Optional[hnswlib.Index] = None
    
    def get_method_name(self) -> str:
        """Get the clustering method name."""
        return f"hnsw_medoids_{self.medoid_selection}"
    
    def fit_predict(self, embeddings: List[CodeEmbedding]) -> ClusteringResult:
        """
        Perform HNSW-based medoid clustering on embeddings.
        
        Args:
            embeddings: List of code embeddings to cluster
            
        Returns:
            ClusteringResult with medoid clusters for tail coverage
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
            # Build HNSW index
            self._build_hnsw_index(embeddings)
            
            # Identify tail embeddings (poorly covered regions)
            tail_embeddings = self._identify_tail_embeddings(embeddings)
            
            # Create medoid clusters for tail coverage
            clusters = self._create_medoid_clusters(tail_embeddings, embeddings)
            
            # Create result
            result = ClusteringResult(
                clusters=clusters,
                total_embeddings=len(embeddings),
                clustering_method=self.get_method_name(),
            )
            
            # Calculate quality metrics
            self._calculate_quality_metrics(result, embeddings)
            
            # Add performance metadata
            result.clustering_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            result.memory_peak_mb = peak / 1024 / 1024
            
            return result
            
        finally:
            tracemalloc.stop()
    
    def _build_hnsw_index(self, embeddings: List[CodeEmbedding]):
        """
        Build HNSW index for fast approximate nearest neighbor search.
        """
        if not embeddings:
            return
        
        # Get embedding dimension
        dimension = embeddings[0].dimension
        
        # Initialize HNSW index
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(
            max_elements=min(len(embeddings) * 2, self.max_elements),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.index.set_ef(self.ef_search)
        
        # Add embeddings to index
        embedding_matrix = self.prepare_embeddings_matrix(embeddings)
        ids = list(range(len(embeddings)))
        
        self.index.add_items(embedding_matrix, ids)
    
    def _identify_tail_embeddings(self, embeddings: List[CodeEmbedding]) -> List[CodeEmbedding]:
        """
        Identify tail embeddings that are poorly covered by dense regions.
        
        Uses density-based analysis to find embeddings in sparse regions
        that would benefit from dedicated medoid coverage.
        """
        if not self.index or not embeddings:
            return []
        
        tail_embeddings = []
        embedding_matrix = self.prepare_embeddings_matrix(embeddings)
        
        # Analyze neighborhood density for each embedding
        for i, embedding in enumerate(embeddings):
            coverage_score = self._calculate_coverage_score(embedding_matrix[i], embeddings)
            
            if coverage_score < self.coverage_threshold:
                tail_embeddings.append(embedding)
        
        return tail_embeddings
    
    def _calculate_coverage_score(self, query_embedding: np.ndarray, all_embeddings: List[CodeEmbedding]) -> float:
        """
        Calculate coverage score for an embedding based on neighborhood density.
        
        Higher scores indicate embeddings in dense regions with good coverage.
        Lower scores indicate tail embeddings needing dedicated coverage.
        """
        if not self.index:
            return 0.0
        
        try:
            # Find k nearest neighbors (use √N heuristic)
            k = max(5, min(50, int(np.sqrt(len(all_embeddings)))))
            k = min(k, len(all_embeddings) - 1)
            
            if k <= 0:
                return 1.0
            
            # Query HNSW index
            labels, distances = self.index.knn_query(query_embedding.reshape(1, -1), k=k)
            
            # Calculate coverage score based on neighborhood density
            # Closer neighbors and more uniform distances indicate better coverage
            if len(distances[0]) == 0:
                return 0.0
            
            # Use inverse of mean distance as coverage score
            mean_distance = np.mean(distances[0])
            coverage_score = 1.0 / (1.0 + mean_distance)
            
            # Adjust for neighborhood uniformity
            distance_variance = np.var(distances[0]) if len(distances[0]) > 1 else 0.0
            uniformity_bonus = 1.0 / (1.0 + distance_variance)
            
            return coverage_score * uniformity_bonus
            
        except Exception:
            # If HNSW query fails, assume poor coverage
            return 0.0
    
    def _create_medoid_clusters(
        self, 
        tail_embeddings: List[CodeEmbedding], 
        all_embeddings: List[CodeEmbedding]
    ) -> List[Cluster]:
        """
        Create medoid-based clusters for tail embeddings.
        
        Selects representative medoids and groups nearby embeddings
        to provide focused coverage for sparse regions.
        """
        if not tail_embeddings:
            return []
        
        # Select medoids using specified strategy
        medoids = self._select_medoids(tail_embeddings, all_embeddings)
        
        if not medoids:
            return []
        
        # Create clusters around medoids
        clusters = []
        used_embeddings = set()
        
        for i, medoid in enumerate(medoids):
            cluster_members = self._gather_cluster_members(
                medoid, all_embeddings, used_embeddings
            )
            
            if len(cluster_members) < self.min_cluster_size:
                continue
            
            # Create cluster
            cluster_id = f"hnsw_medoid_{i}"
            cluster = self._create_medoid_cluster(
                cluster_id, medoid, cluster_members
            )
            
            clusters.append(cluster)
            
            # Mark embeddings as used
            for emb in cluster_members:
                used_embeddings.add(emb.content_sha)
        
        return clusters
    
    def _select_medoids(
        self, 
        tail_embeddings: List[CodeEmbedding], 
        all_embeddings: List[CodeEmbedding]
    ) -> List[CodeEmbedding]:
        """
        Select medoids using the specified strategy.
        """
        if not tail_embeddings:
            return []
        
        np.random.seed(self.random_state)
        
        if self.medoid_selection == 'random':
            return self._select_random_medoids(tail_embeddings)
        elif self.medoid_selection == 'density_based':
            return self._select_density_based_medoids(tail_embeddings, all_embeddings)
        elif self.medoid_selection == 'distance_based':
            return self._select_distance_based_medoids(tail_embeddings)
        else:
            raise ValueError(f"Unknown medoid selection strategy: {self.medoid_selection}")
    
    def _select_random_medoids(self, tail_embeddings: List[CodeEmbedding]) -> List[CodeEmbedding]:
        """Select random medoids from tail embeddings."""
        n_medoids = self._calculate_num_medoids(tail_embeddings)
        
        if n_medoids >= len(tail_embeddings):
            return tail_embeddings
        
        indices = np.random.choice(len(tail_embeddings), n_medoids, replace=False)
        return [tail_embeddings[i] for i in indices]
    
    def _select_density_based_medoids(
        self, 
        tail_embeddings: List[CodeEmbedding],
        all_embeddings: List[CodeEmbedding]
    ) -> List[CodeEmbedding]:
        """
        Select medoids based on local density characteristics.
        
        Prefers embeddings in regions with intermediate density -
        not too sparse (would have poor coverage) but not too dense 
        (would overlap with k-means coverage).
        """
        if not self.index:
            return self._select_random_medoids(tail_embeddings)
        
        n_medoids = self._calculate_num_medoids(tail_embeddings)
        
        # Calculate density scores for tail embeddings
        density_scores = []
        embedding_matrix = self.prepare_embeddings_matrix(tail_embeddings)
        
        for i, embedding in enumerate(tail_embeddings):
            try:
                # Find local neighborhood
                k = min(10, len(all_embeddings) - 1)
                if k <= 0:
                    density_scores.append(0.0)
                    continue
                
                labels, distances = self.index.knn_query(embedding_matrix[i].reshape(1, -1), k=k)
                
                # Calculate local density as inverse of mean distance to neighbors
                if len(distances[0]) > 0:
                    mean_distance = np.mean(distances[0])
                    density = 1.0 / (1.0 + mean_distance)
                else:
                    density = 0.0
                
                density_scores.append(density)
                
            except Exception:
                density_scores.append(0.0)
        
        # Select medoids with highest density scores
        if len(density_scores) <= n_medoids:
            return tail_embeddings
        
        top_indices = np.argsort(density_scores)[-n_medoids:]
        return [tail_embeddings[i] for i in top_indices]
    
    def _select_distance_based_medoids(self, tail_embeddings: List[CodeEmbedding]) -> List[CodeEmbedding]:
        """
        Select medoids to maximize distance between selected points.
        
        Uses a greedy approach similar to k-means++ initialization
        to ensure good spatial distribution of medoids.
        """
        if not tail_embeddings:
            return []
        
        n_medoids = self._calculate_num_medoids(tail_embeddings)
        
        if n_medoids >= len(tail_embeddings):
            return tail_embeddings
        
        embedding_matrix = self.prepare_embeddings_matrix(tail_embeddings)
        medoids = []
        
        # Select first medoid randomly
        first_idx = np.random.randint(len(tail_embeddings))
        medoids.append(tail_embeddings[first_idx])
        selected_indices = {first_idx}
        
        # Select remaining medoids greedily
        for _ in range(n_medoids - 1):
            best_idx = -1
            best_distance = -1
            
            for i, embedding in enumerate(tail_embeddings):
                if i in selected_indices:
                    continue
                
                # Calculate minimum distance to existing medoids
                min_distance = float('inf')
                for medoid in medoids:
                    distance = np.linalg.norm(embedding.embedding - medoid.embedding)
                    min_distance = min(min_distance, distance)
                
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_idx = i
            
            if best_idx >= 0:
                medoids.append(tail_embeddings[best_idx])
                selected_indices.add(best_idx)
        
        return medoids
    
    def _calculate_num_medoids(self, tail_embeddings: List[CodeEmbedding]) -> int:
        """Calculate optimal number of medoids using heuristics."""
        n = len(tail_embeddings)
        if n <= 1:
            return n
        
        # Use √N heuristic but with smaller constant for medoids
        # (since we want fewer, more focused medoid clusters)
        n_medoids = max(1, int(np.sqrt(n) / 2))
        
        # Apply maximum limit if specified
        if self.max_medoids is not None:
            n_medoids = min(n_medoids, self.max_medoids)
        
        # Ensure we don't exceed available embeddings
        n_medoids = min(n_medoids, n)
        
        return n_medoids
    
    def _gather_cluster_members(
        self,
        medoid: CodeEmbedding,
        all_embeddings: List[CodeEmbedding],
        used_embeddings: Set[str]
    ) -> List[CodeEmbedding]:
        """
        Gather cluster members around a medoid using HNSW search.
        """
        if not self.index:
            return [medoid]
        
        try:
            # Find nearest neighbors to medoid
            k = min(20, len(all_embeddings))  # Look at up to 20 nearest neighbors
            
            labels, distances = self.index.knn_query(
                medoid.embedding.reshape(1, -1), k=k
            )
            
            cluster_members = []
            
            # Add medoid itself if not already used
            if medoid.content_sha not in used_embeddings:
                cluster_members.append(medoid)
            
            # Add nearby embeddings that aren't already used
            for idx, distance in zip(labels[0], distances[0]):
                if idx < len(all_embeddings):
                    neighbor = all_embeddings[idx]
                    
                    if (neighbor.content_sha != medoid.content_sha and 
                        neighbor.content_sha not in used_embeddings and
                        distance < 0.8):  # Distance threshold for inclusion
                        cluster_members.append(neighbor)
            
            return cluster_members
            
        except Exception:
            # If HNSW search fails, return just the medoid
            return [medoid] if medoid.content_sha not in used_embeddings else []
    
    def _create_medoid_cluster(
        self,
        cluster_id: str,
        medoid: CodeEmbedding,
        members: List[CodeEmbedding]
    ) -> Cluster:
        """
        Create a cluster around a medoid.
        """
        # Use medoid embedding as centroid
        centroid = medoid.embedding.copy()
        
        # Calculate cluster properties
        inertia = 0.0
        for emb in members:
            distance_squared = np.sum((emb.embedding - centroid) ** 2)
            inertia += distance_squared
        
        # Determine package path
        package_paths = set()
        for emb in members:
            from pathlib import Path
            file_path = Path(emb.file_path)
            package_path = str(file_path.parent) if file_path.parent != Path('.') else 'root'
            package_paths.add(package_path)
        
        # Use most common package path
        main_package = max(package_paths, key=lambda p: sum(1 for emb in members 
                                                          if str(Path(emb.file_path).parent) == p)) if package_paths else None
        
        return Cluster(
            cluster_id=cluster_id,
            centroid=centroid,
            members=[emb.content_sha for emb in members],
            member_embeddings=members,
            cluster_type="hnsw_medoid",
            inertia=inertia,
            package_path=main_package,
        )
    
    def _calculate_quality_metrics(self, result: ClusteringResult, embeddings: List[CodeEmbedding]):
        """
        Calculate clustering quality metrics for medoid clusters.
        """
        if result.num_clusters <= 1 or len(embeddings) <= 1:
            return
        
        # Calculate basic metrics
        cluster_sizes = [cluster.size for cluster in result.clusters]
        if cluster_sizes:
            size_variance = np.var(cluster_sizes)
            mean_size = np.mean(cluster_sizes)
            result.coverage_uniformity = 1.0 - min(1.0, size_variance / max(mean_size, 1.0))
        
        # Calculate total inertia
        result.inertia = sum(cluster.inertia for cluster in result.clusters)
        
        # Medoid-specific quality metrics
        tail_coverage = len([emb for cluster in result.clusters for emb in cluster.member_embeddings])
        total_embeddings = len(embeddings)
        
        # Store medoid-specific metrics in coverage statistics
        result.package_coverage['tail_coverage_ratio'] = tail_coverage / max(total_embeddings, 1)
        result.package_coverage['medoid_efficiency'] = tail_coverage / max(result.num_clusters, 1)


# Convenience factory functions
def create_density_medoid_clusterer(**kwargs) -> HNSWMedoidClusterer:
    """Create HNSW medoid clusterer with density-based selection."""
    defaults = {
        'medoid_selection': 'density_based',
        'coverage_threshold': 0.5,
        'min_cluster_size': 2,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return HNSWMedoidClusterer(**defaults)


def create_distance_medoid_clusterer(**kwargs) -> HNSWMedoidClusterer:
    """Create HNSW medoid clusterer with distance-based selection."""
    defaults = {
        'medoid_selection': 'distance_based',
        'coverage_threshold': 0.4,
        'min_cluster_size': 2,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return HNSWMedoidClusterer(**defaults)


def create_fast_medoid_clusterer(**kwargs) -> HNSWMedoidClusterer:
    """Create HNSW medoid clusterer optimized for speed."""
    defaults = {
        'medoid_selection': 'random',
        'ef_construction': 100,  # Faster construction
        'ef_search': 25,         # Faster search
        'M': 8,                  # Smaller graph
        'coverage_threshold': 0.6,
        'min_cluster_size': 1,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return HNSWMedoidClusterer(**defaults)
"""
Base classes for V2 clustering system.

Provides abstract interfaces for clustering providers and core data structures
for k-means and HNSW-based clustering of code embeddings.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from pathlib import Path

from ..embeddings.base import CodeEmbedding


@dataclass
class Cluster:
    """
    Represents a cluster of code embeddings with centroid and metadata.
    
    Used for both k-means clusters and HNSW medoid groups to provide
    uniform interface for mixed centroid coverage calculations.
    """
    
    cluster_id: str
    centroid: np.ndarray  # Cluster center/medoid embedding
    members: List[str]  # Content SHAs of cluster members
    member_embeddings: List[CodeEmbedding]  # Full embeddings for members
    cluster_type: str  # 'kmeans', 'hnsw_medoid', 'file_group', etc.
    
    # Clustering metadata
    inertia: float = 0.0  # Within-cluster sum of squares (k-means)
    radius: float = 0.0  # Maximum distance to centroid
    density: float = 0.0  # Number of members / radius^2
    
    # Package/file context
    package_path: Optional[str] = None  # Package or directory path
    file_paths: Optional[Set[str]] = None  # File paths in cluster
    
    def __post_init__(self):
        """Initialize computed properties."""
        if self.file_paths is None:
            self.file_paths = set()
            for emb in self.member_embeddings:
                self.file_paths.add(emb.file_path)
        
        # Calculate properties if not provided
        if len(self.member_embeddings) > 0:
            if self.radius == 0.0:
                self._calculate_radius()
            if self.density == 0.0:
                self._calculate_density()
    
    @property
    def size(self) -> int:
        """Get number of members in cluster."""
        return len(self.members)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.centroid)
    
    def contains_chunk(self, content_sha: str) -> bool:
        """Check if cluster contains a specific chunk."""
        return content_sha in self.members
    
    def get_distance_to_centroid(self, embedding: np.ndarray) -> float:
        """Calculate distance from embedding to cluster centroid."""
        return float(np.linalg.norm(embedding - self.centroid))
    
    def get_member_distances(self) -> List[float]:
        """Get distances from all members to centroid."""
        distances = []
        for emb in self.member_embeddings:
            distance = self.get_distance_to_centroid(emb.embedding)
            distances.append(distance)
        return distances
    
    def get_coverage_score(self, target_embedding: np.ndarray) -> float:
        """
        Calculate coverage score for target embedding.
        
        Returns higher scores for embeddings close to the cluster centroid,
        useful for facility location algorithms.
        
        Args:
            target_embedding: Embedding to score
            
        Returns:
            Coverage score (higher = better coverage)
        """
        distance = self.get_distance_to_centroid(target_embedding)
        # Convert distance to similarity score (closer = higher score)
        return 1.0 / (1.0 + distance)
    
    def _calculate_radius(self):
        """Calculate cluster radius as maximum distance to centroid."""
        if not self.member_embeddings:
            self.radius = 0.0
            return
        
        distances = self.get_member_distances()
        self.radius = max(distances) if distances else 0.0
    
    def _calculate_density(self):
        """Calculate cluster density as members per unit area."""
        if self.radius == 0.0 or len(self.member_embeddings) <= 1:
            self.density = float(len(self.member_embeddings))
        else:
            # Density = members / radius^2 (approximating 2D area)
            self.density = len(self.member_embeddings) / (self.radius ** 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'centroid': self.centroid.tolist(),
            'members': self.members,
            'cluster_type': self.cluster_type,
            'size': self.size,
            'inertia': self.inertia,
            'radius': self.radius,
            'density': self.density,
            'package_path': self.package_path,
            'file_paths': list(self.file_paths) if self.file_paths else [],
        }


@dataclass
class ClusteringResult:
    """
    Result of clustering operation with clusters and quality metrics.
    
    Provides comprehensive information about clustering quality and
    characteristics for evaluation and optimization.
    """
    
    clusters: List[Cluster]
    total_embeddings: int
    clustering_method: str
    
    # Quality metrics
    silhouette_score: float = 0.0  # Overall clustering quality
    inertia: float = 0.0  # Total within-cluster sum of squares
    calinski_harabasz_score: float = 0.0  # Cluster separation
    davies_bouldin_score: float = 0.0  # Cluster compactness
    
    # Coverage statistics
    coverage_uniformity: float = 0.0  # How evenly coverage is distributed
    package_coverage: Dict[str, int] = None  # Clusters per package
    file_coverage: Dict[str, int] = None  # Clusters per file
    
    # Performance metadata
    clustering_time: float = 0.0  # Time taken for clustering
    memory_peak_mb: float = 0.0  # Peak memory usage
    
    def __post_init__(self):
        """Initialize computed properties."""
        if self.package_coverage is None:
            self.package_coverage = {}
        if self.file_coverage is None:
            self.file_coverage = {}
            
        # Calculate coverage statistics if not provided
        if not self.package_coverage or not self.file_coverage:
            self._calculate_coverage_stats()
    
    @property
    def num_clusters(self) -> int:
        """Get number of clusters."""
        return len(self.clusters)
    
    @property
    def average_cluster_size(self) -> float:
        """Get average cluster size."""
        if not self.clusters:
            return 0.0
        return self.total_embeddings / len(self.clusters)
    
    @property
    def cluster_size_variance(self) -> float:
        """Get variance in cluster sizes."""
        if not self.clusters:
            return 0.0
        
        sizes = [cluster.size for cluster in self.clusters]
        mean_size = np.mean(sizes)
        return float(np.var(sizes)) if len(sizes) > 1 else 0.0
    
    def get_cluster_by_id(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None
    
    def get_clusters_for_package(self, package_path: str) -> List[Cluster]:
        """Get all clusters that contain chunks from a package."""
        return [
            cluster for cluster in self.clusters 
            if cluster.package_path == package_path
        ]
    
    def get_clusters_for_file(self, file_path: str) -> List[Cluster]:
        """Get all clusters that contain chunks from a file."""
        return [
            cluster for cluster in self.clusters
            if cluster.file_paths and file_path in cluster.file_paths
        ]
    
    def find_chunk_cluster(self, content_sha: str) -> Optional[Cluster]:
        """Find the cluster containing a specific chunk."""
        for cluster in self.clusters:
            if cluster.contains_chunk(content_sha):
                return cluster
        return None
    
    def _calculate_coverage_stats(self):
        """Calculate package and file coverage statistics."""
        package_counts = {}
        file_counts = {}
        
        for cluster in self.clusters:
            # Count packages
            if cluster.package_path:
                package_counts[cluster.package_path] = package_counts.get(cluster.package_path, 0) + 1
            
            # Count files
            if cluster.file_paths:
                for file_path in cluster.file_paths:
                    file_counts[file_path] = file_counts.get(file_path, 0) + 1
        
        self.package_coverage = package_counts
        self.file_coverage = file_counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert clustering result to dictionary for serialization."""
        return {
            'clusters': [cluster.to_dict() for cluster in self.clusters],
            'total_embeddings': self.total_embeddings,
            'clustering_method': self.clustering_method,
            'num_clusters': self.num_clusters,
            'average_cluster_size': self.average_cluster_size,
            'cluster_size_variance': self.cluster_size_variance,
            'silhouette_score': self.silhouette_score,
            'inertia': self.inertia,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'davies_bouldin_score': self.davies_bouldin_score,
            'coverage_uniformity': self.coverage_uniformity,
            'package_coverage': self.package_coverage,
            'file_coverage': self.file_coverage,
            'clustering_time': self.clustering_time,
            'memory_peak_mb': self.memory_peak_mb,
        }


class ClusteringProvider(ABC):
    """
    Abstract base class for clustering algorithm implementations.
    
    Defines interface for clustering code embeddings using various
    algorithms (k-means, HNSW, hierarchical, etc.).
    """
    
    @abstractmethod
    def fit_predict(self, embeddings: List[CodeEmbedding]) -> ClusteringResult:
        """
        Perform clustering on embeddings and return results.
        
        Args:
            embeddings: List of code embeddings to cluster
            
        Returns:
            ClusteringResult with clusters and quality metrics
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the clustering method name."""
        pass
    
    def prepare_embeddings_matrix(self, embeddings: List[CodeEmbedding]) -> np.ndarray:
        """
        Convert list of embeddings to matrix format.
        
        Args:
            embeddings: List of code embeddings
            
        Returns:
            Matrix of shape (n_samples, n_features)
        """
        if not embeddings:
            return np.array([])
        
        return np.stack([emb.embedding for emb in embeddings])
    
    def group_embeddings_by_package(self, embeddings: List[CodeEmbedding]) -> Dict[str, List[CodeEmbedding]]:
        """
        Group embeddings by package path for per-package clustering.
        
        Args:
            embeddings: List of code embeddings
            
        Returns:
            Dictionary mapping package path to embeddings
        """
        packages = {}
        
        for emb in embeddings:
            # Extract package path from file path
            file_path = Path(emb.file_path)
            
            # Use parent directory as package
            package_path = str(file_path.parent) if file_path.parent != Path('.') else 'root'
            
            if package_path not in packages:
                packages[package_path] = []
            packages[package_path].append(emb)
        
        return packages
    
    def calculate_optimal_k(self, embeddings: List[CodeEmbedding], max_k: Optional[int] = None) -> int:
        """
        Calculate optimal number of clusters using sqrt(N) heuristic.
        
        Args:
            embeddings: List of embeddings to cluster
            max_k: Maximum number of clusters (None for no limit)
            
        Returns:
            Optimal number of clusters
        """
        n = len(embeddings)
        if n <= 1:
            return 1
        
        # k ≈ √N heuristic from TODO.md
        k = max(1, int(np.sqrt(n)))
        
        # Apply maximum limit if specified
        if max_k is not None:
            k = min(k, max_k)
        
        # Ensure k doesn't exceed number of embeddings
        k = min(k, n)
        
        return k
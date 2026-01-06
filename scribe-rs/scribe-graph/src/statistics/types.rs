//! Type definitions for graph statistics and analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::graph::{GraphStatistics, NodeId};

/// Comprehensive graph analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphAnalysisResults {
    /// Basic graph statistics
    pub basic_stats: GraphStatistics,
    /// Degree distribution analysis
    pub degree_distribution: DegreeDistribution,
    /// Connectivity analysis
    pub connectivity: ConnectivityAnalysis,
    /// Structural patterns
    pub structural_patterns: StructuralPatterns,
    /// Import relationship insights
    pub import_insights: ImportInsights,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata,
}

/// Detailed degree distribution statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DegreeDistribution {
    /// In-degree statistics
    pub in_degree: DegreeStats,
    /// Out-degree statistics
    pub out_degree: DegreeStats,
    /// Total degree statistics
    pub total_degree: DegreeStats,
    /// Degree distribution histogram (degree -> count)
    pub in_degree_histogram: HashMap<usize, usize>,
    pub out_degree_histogram: HashMap<usize, usize>,
    /// Power-law fitting parameters (if applicable)
    pub power_law_alpha: Option<f64>,
    pub power_law_goodness_of_fit: Option<f64>,
}

/// Statistical measures for degree sequences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DegreeStats {
    pub min: usize,
    pub max: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub percentile_90: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
}

/// Graph connectivity and component analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ConnectivityAnalysis {
    /// Number of weakly connected components
    pub weakly_connected_components: usize,
    /// Number of strongly connected components
    pub strongly_connected_components: usize,
    /// Size of largest strongly connected component
    pub largest_scc_size: usize,
    /// Graph density (actual edges / possible edges)
    pub graph_density: f64,
    /// Average clustering coefficient
    pub average_clustering: f64,
    /// Global clustering coefficient (transitivity)
    pub global_clustering: f64,
    /// Average path length (in largest component)
    pub average_path_length: Option<f64>,
    /// Graph diameter (longest shortest path)
    pub diameter: Option<usize>,
    /// Is the graph acyclic (DAG)
    pub is_acyclic: bool,
}

/// Structural patterns and notable nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StructuralPatterns {
    /// Hub nodes (high out-degree)
    pub hubs: Vec<NodeInfo>,
    /// Authority nodes (high in-degree)
    pub authorities: Vec<NodeInfo>,
    /// Bottleneck nodes (high betweenness centrality estimate)
    pub bottlenecks: Vec<NodeInfo>,
    /// Bridge nodes (critical for connectivity)
    pub bridges: Vec<NodeInfo>,
    /// Isolated nodes (no connections)
    pub isolated_nodes: Vec<NodeId>,
    /// Dangling nodes (no outgoing edges)
    pub dangling_nodes: Vec<NodeId>,
    /// Leaf nodes (no incoming edges, but have outgoing)
    pub leaf_nodes: Vec<NodeId>,
}

/// Information about important nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub score: f64,
    pub in_degree: usize,
    pub out_degree: usize,
    pub metadata: Option<String>,
}

/// Import relationship specific insights
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ImportInsights {
    /// Average import depth (distance from root nodes)
    pub average_import_depth: f64,
    /// Maximum import depth
    pub max_import_depth: usize,
    /// Import fan-out distribution (how many files each file imports)
    pub fan_out_distribution: DegreeStats,
    /// Import fan-in distribution (how many files import each file)
    pub fan_in_distribution: DegreeStats,
    /// Circular dependency detection
    pub circular_dependencies: Vec<CircularDependency>,
    /// Dependency layers (topological levels)
    pub dependency_layers: Vec<Vec<NodeId>>,
    /// Critical import paths (most important dependency chains)
    pub critical_paths: Vec<DependencyPath>,
}

/// Circular dependency information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CircularDependency {
    pub nodes: Vec<NodeId>,
    pub cycle_length: usize,
    pub strength: f64,
}

/// Important dependency path
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependencyPath {
    pub path: Vec<NodeId>,
    pub length: usize,
    pub importance_score: f64,
}

/// Performance characteristics of the graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PerformanceProfile {
    /// Estimated memory usage for PageRank computation
    pub pagerank_memory_estimate_mb: f64,
    /// Estimated PageRank computation time
    pub pagerank_time_estimate_ms: u64,
    /// Graph traversal complexity
    pub traversal_complexity: TraversalComplexity,
    /// Storage efficiency metrics
    pub storage_efficiency: StorageEfficiency,
}

/// Complexity analysis for graph algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TraversalComplexity {
    /// BFS/DFS time complexity class
    pub time_complexity_class: String,
    /// Space complexity class
    pub space_complexity_class: String,
    /// Expected iterations for convergence algorithms
    pub expected_iterations: usize,
}

/// Storage and memory efficiency metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StorageEfficiency {
    /// Adjacency list memory usage (bytes)
    pub adjacency_list_size_bytes: usize,
    /// Average edges per node
    pub edges_per_node: f64,
    /// Memory overhead ratio
    pub memory_overhead_ratio: f64,
    /// Sparsity coefficient
    pub sparsity: f64,
}

/// Analysis execution metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// When the analysis was performed
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Analysis duration in milliseconds
    pub analysis_duration_ms: u64,
    /// Whether parallel computation was used
    pub used_parallel: bool,
    /// Analysis configuration used
    pub config: StatisticsConfig,
    /// Version of the statistics module
    pub version: String,
}

impl Default for AnalysisMetadata {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            analysis_duration_ms: 0,
            used_parallel: false,
            config: StatisticsConfig::default(),
            version: "1.0.0".to_string(),
        }
    }
}

/// Configuration for statistics computation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Whether to compute expensive metrics (clustering, path length)
    pub compute_expensive_metrics: bool,
    /// Whether to use parallel computation where possible
    pub use_parallel: bool,
    /// Maximum number of nodes to analyze for expensive operations
    pub max_nodes_for_expensive_ops: usize,
    /// Number of top nodes to include in structural patterns
    pub top_nodes_count: usize,
    /// Minimum score threshold for pattern detection
    pub pattern_threshold: f64,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            compute_expensive_metrics: true,
            use_parallel: true,
            max_nodes_for_expensive_ops: 10000,
            top_nodes_count: 10,
            pattern_threshold: 0.1,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degree_distribution_default() {
        let dist = DegreeDistribution::default();
        assert!(dist.in_degree_histogram.is_empty());
        assert!(dist.out_degree_histogram.is_empty());
        assert!(dist.power_law_alpha.is_none());
    }

    #[test]
    fn test_degree_distribution_clone() {
        let dist = DegreeDistribution::default();
        let cloned = dist.clone();
        assert_eq!(dist, cloned);
    }

    #[test]
    fn test_degree_distribution_serialize() {
        let dist = DegreeDistribution::default();
        let json = serde_json::to_string(&dist).unwrap();
        let deserialized: DegreeDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(dist, deserialized);
    }

    #[test]
    fn test_degree_stats_default() {
        let stats = DegreeStats::default();
        assert_eq!(stats.min, 0);
        assert_eq!(stats.max, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.median, 0.0);
        assert_eq!(stats.std_dev, 0.0);
    }

    #[test]
    fn test_degree_stats_custom() {
        let stats = DegreeStats {
            min: 1,
            max: 100,
            mean: 25.0,
            median: 20.0,
            std_dev: 15.0,
            percentile_25: 10.0,
            percentile_75: 40.0,
            percentile_90: 60.0,
            percentile_95: 80.0,
            percentile_99: 95.0,
        };

        assert_eq!(stats.min, 1);
        assert_eq!(stats.max, 100);
        assert_eq!(stats.mean, 25.0);
    }

    #[test]
    fn test_connectivity_analysis_default() {
        let analysis = ConnectivityAnalysis::default();
        assert_eq!(analysis.weakly_connected_components, 0);
        assert_eq!(analysis.strongly_connected_components, 0);
        assert_eq!(analysis.largest_scc_size, 0);
        assert!(!analysis.is_acyclic);
    }

    #[test]
    fn test_connectivity_analysis_clone() {
        let analysis = ConnectivityAnalysis {
            weakly_connected_components: 5,
            strongly_connected_components: 3,
            largest_scc_size: 100,
            graph_density: 0.05,
            average_clustering: 0.3,
            global_clustering: 0.25,
            average_path_length: Some(4.5),
            diameter: Some(10),
            is_acyclic: true,
        };

        let cloned = analysis.clone();
        assert_eq!(analysis, cloned);
    }

    #[test]
    fn test_structural_patterns_default() {
        let patterns = StructuralPatterns::default();
        assert!(patterns.hubs.is_empty());
        assert!(patterns.authorities.is_empty());
        assert!(patterns.bottlenecks.is_empty());
        assert!(patterns.bridges.is_empty());
        assert!(patterns.isolated_nodes.is_empty());
    }

    #[test]
    fn test_node_info_creation() {
        let info = NodeInfo {
            node_id: "test.rs".to_string(),
            score: 0.85,
            in_degree: 10,
            out_degree: 5,
            metadata: Some("important file".to_string()),
        };

        assert_eq!(info.node_id, "test.rs");
        assert_eq!(info.score, 0.85);
        assert_eq!(info.in_degree, 10);
        assert_eq!(info.out_degree, 5);
    }

    #[test]
    fn test_node_info_serialize() {
        let info = NodeInfo {
            node_id: "main.rs".to_string(),
            score: 0.5,
            in_degree: 5,
            out_degree: 3,
            metadata: None,
        };

        let json = serde_json::to_string(&info).unwrap();
        let deserialized: NodeInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, deserialized);
    }

    #[test]
    fn test_import_insights_default() {
        let insights = ImportInsights::default();
        assert_eq!(insights.average_import_depth, 0.0);
        assert_eq!(insights.max_import_depth, 0);
        assert!(insights.circular_dependencies.is_empty());
        assert!(insights.dependency_layers.is_empty());
    }

    #[test]
    fn test_circular_dependency_creation() {
        let dep = CircularDependency {
            nodes: vec![
                "a.rs".to_string(),
                "b.rs".to_string(),
                "a.rs".to_string(),
            ],
            cycle_length: 2,
            strength: 1.0,
        };

        assert_eq!(dep.nodes.len(), 3);
        assert_eq!(dep.cycle_length, 2);
        assert_eq!(dep.strength, 1.0);
    }

    #[test]
    fn test_circular_dependency_serialize() {
        let dep = CircularDependency {
            nodes: vec!["x.rs".to_string()],
            cycle_length: 1,
            strength: 0.5,
        };

        let json = serde_json::to_string(&dep).unwrap();
        let deserialized: CircularDependency = serde_json::from_str(&json).unwrap();
        assert_eq!(dep, deserialized);
    }

    #[test]
    fn test_dependency_path_creation() {
        let path = DependencyPath {
            path: vec![
                "main.rs".to_string(),
                "lib.rs".to_string(),
                "utils.rs".to_string(),
            ],
            length: 3,
            importance_score: 0.9,
        };

        assert_eq!(path.path.len(), 3);
        assert_eq!(path.length, 3);
        assert_eq!(path.importance_score, 0.9);
    }

    #[test]
    fn test_performance_profile_default() {
        let profile = PerformanceProfile::default();
        assert_eq!(profile.pagerank_memory_estimate_mb, 0.0);
        assert_eq!(profile.pagerank_time_estimate_ms, 0);
    }

    #[test]
    fn test_performance_profile_custom() {
        let profile = PerformanceProfile {
            pagerank_memory_estimate_mb: 50.0,
            pagerank_time_estimate_ms: 1000,
            traversal_complexity: TraversalComplexity {
                time_complexity_class: "O(V + E)".to_string(),
                space_complexity_class: "O(V)".to_string(),
                expected_iterations: 20,
            },
            storage_efficiency: StorageEfficiency {
                adjacency_list_size_bytes: 1_000_000,
                edges_per_node: 5.0,
                memory_overhead_ratio: 1.2,
                sparsity: 0.95,
            },
        };

        assert_eq!(profile.pagerank_memory_estimate_mb, 50.0);
        assert_eq!(profile.traversal_complexity.time_complexity_class, "O(V + E)");
    }

    #[test]
    fn test_traversal_complexity_default() {
        let complexity = TraversalComplexity::default();
        assert!(complexity.time_complexity_class.is_empty());
        assert!(complexity.space_complexity_class.is_empty());
        assert_eq!(complexity.expected_iterations, 0);
    }

    #[test]
    fn test_storage_efficiency_default() {
        let efficiency = StorageEfficiency::default();
        assert_eq!(efficiency.adjacency_list_size_bytes, 0);
        assert_eq!(efficiency.edges_per_node, 0.0);
        assert_eq!(efficiency.memory_overhead_ratio, 0.0);
        assert_eq!(efficiency.sparsity, 0.0);
    }

    #[test]
    fn test_analysis_metadata_default() {
        let metadata = AnalysisMetadata::default();
        assert_eq!(metadata.analysis_duration_ms, 0);
        assert!(!metadata.used_parallel);
        assert_eq!(metadata.version, "1.0.0");
    }

    #[test]
    fn test_analysis_metadata_custom() {
        let metadata = AnalysisMetadata {
            timestamp: chrono::Utc::now(),
            analysis_duration_ms: 500,
            used_parallel: true,
            config: StatisticsConfig::default(),
            version: "2.0.0".to_string(),
        };

        assert_eq!(metadata.analysis_duration_ms, 500);
        assert!(metadata.used_parallel);
        assert_eq!(metadata.version, "2.0.0");
    }

    #[test]
    fn test_statistics_config_default() {
        let config = StatisticsConfig::default();
        assert!(config.compute_expensive_metrics);
        assert!(config.use_parallel);
        assert_eq!(config.max_nodes_for_expensive_ops, 10000);
        assert_eq!(config.top_nodes_count, 10);
        assert_eq!(config.pattern_threshold, 0.1);
    }

    #[test]
    fn test_statistics_config_custom() {
        let config = StatisticsConfig {
            compute_expensive_metrics: false,
            use_parallel: false,
            max_nodes_for_expensive_ops: 5000,
            top_nodes_count: 20,
            pattern_threshold: 0.05,
        };

        assert!(!config.compute_expensive_metrics);
        assert!(!config.use_parallel);
        assert_eq!(config.max_nodes_for_expensive_ops, 5000);
    }

    #[test]
    fn test_statistics_config_clone() {
        let config = StatisticsConfig::default();
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_statistics_config_serialize() {
        let config = StatisticsConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: StatisticsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_degree_stats_serialize() {
        let stats = DegreeStats {
            min: 1,
            max: 50,
            mean: 10.0,
            median: 8.0,
            std_dev: 5.0,
            percentile_25: 5.0,
            percentile_75: 15.0,
            percentile_90: 25.0,
            percentile_95: 35.0,
            percentile_99: 45.0,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: DegreeStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_connectivity_analysis_serialize() {
        let analysis = ConnectivityAnalysis {
            weakly_connected_components: 1,
            strongly_connected_components: 1,
            largest_scc_size: 100,
            graph_density: 0.1,
            average_clustering: 0.5,
            global_clustering: 0.4,
            average_path_length: Some(3.5),
            diameter: Some(8),
            is_acyclic: false,
        };

        let json = serde_json::to_string(&analysis).unwrap();
        let deserialized: ConnectivityAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(analysis, deserialized);
    }

    #[test]
    fn test_import_insights_serialize() {
        let insights = ImportInsights {
            average_import_depth: 3.5,
            max_import_depth: 10,
            fan_out_distribution: DegreeStats::default(),
            fan_in_distribution: DegreeStats::default(),
            circular_dependencies: vec![],
            dependency_layers: vec![vec!["a.rs".to_string()]],
            critical_paths: vec![],
        };

        let json = serde_json::to_string(&insights).unwrap();
        let deserialized: ImportInsights = serde_json::from_str(&json).unwrap();
        assert_eq!(insights, deserialized);
    }

    #[test]
    fn test_dependency_path_serialize() {
        let path = DependencyPath {
            path: vec!["a.rs".to_string(), "b.rs".to_string()],
            length: 2,
            importance_score: 0.75,
        };

        let json = serde_json::to_string(&path).unwrap();
        let deserialized: DependencyPath = serde_json::from_str(&json).unwrap();
        assert_eq!(path, deserialized);
    }

    #[test]
    fn test_performance_profile_serialize() {
        let profile = PerformanceProfile::default();
        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: PerformanceProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile, deserialized);
    }

    #[test]
    fn test_traversal_complexity_serialize() {
        let complexity = TraversalComplexity {
            time_complexity_class: "O(V log V)".to_string(),
            space_complexity_class: "O(V)".to_string(),
            expected_iterations: 15,
        };

        let json = serde_json::to_string(&complexity).unwrap();
        let deserialized: TraversalComplexity = serde_json::from_str(&json).unwrap();
        assert_eq!(complexity, deserialized);
    }

    #[test]
    fn test_storage_efficiency_serialize() {
        let efficiency = StorageEfficiency {
            adjacency_list_size_bytes: 500_000,
            edges_per_node: 3.5,
            memory_overhead_ratio: 1.1,
            sparsity: 0.98,
        };

        let json = serde_json::to_string(&efficiency).unwrap();
        let deserialized: StorageEfficiency = serde_json::from_str(&json).unwrap();
        assert_eq!(efficiency, deserialized);
    }

    #[test]
    fn test_structural_patterns_with_data() {
        let patterns = StructuralPatterns {
            hubs: vec![NodeInfo {
                node_id: "hub.rs".to_string(),
                score: 0.9,
                in_degree: 5,
                out_degree: 50,
                metadata: None,
            }],
            authorities: vec![],
            bottlenecks: vec![],
            bridges: vec![],
            isolated_nodes: vec!["isolated.rs".to_string()],
            dangling_nodes: vec![],
            leaf_nodes: vec![],
        };

        assert_eq!(patterns.hubs.len(), 1);
        assert_eq!(patterns.isolated_nodes.len(), 1);
    }

    #[test]
    fn test_node_info_debug() {
        let info = NodeInfo {
            node_id: "test.rs".to_string(),
            score: 0.5,
            in_degree: 3,
            out_degree: 2,
            metadata: None,
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("NodeInfo"));
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_statistics_config_debug() {
        let config = StatisticsConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("StatisticsConfig"));
        assert!(debug_str.contains("compute_expensive_metrics"));
    }
}

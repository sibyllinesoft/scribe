//! # Graph Statistics and Analysis for Dependency Graphs
//!
//! Comprehensive statistical analysis and metrics computation for code dependency graphs.
//! This module provides detailed insights into graph structure, connectivity patterns,
//! and characteristics that are essential for understanding codebase architecture.
//!
//! ## Key Features
//! - **Degree Distribution Analysis**: In-degree, out-degree, and total degree statistics
//! - **Connectivity Metrics**: Strongly connected components, graph density, clustering
//! - **Structural Patterns**: Hub detection, authority identification, bottleneck analysis
//! - **Import Relationship Insights**: Dependency depth, fan-in/fan-out analysis
//! - **Performance Characteristics**: Memory usage, computation complexity estimates
//! - **Comparative Analysis**: Before/after graph evolution tracking

mod analyzer;
mod types;

pub use analyzer::GraphStatisticsAnalyzer;
pub use types::{
    AnalysisMetadata, CircularDependency, ConnectivityAnalysis, DegreeDistribution, DegreeStats,
    DependencyPath, GraphAnalysisResults, ImportInsights, NodeInfo, PerformanceProfile,
    StatisticsConfig, StorageEfficiency, StructuralPatterns, TraversalComplexity,
};

/// Utility functions for graph analysis results
impl GraphAnalysisResults {
    /// Generate a comprehensive summary report
    pub fn summary_report(&self) -> String {
        format!(
            "Graph Analysis Summary Report\n\
             ============================\n\
             \n\
             Basic Statistics:\n\
             - Nodes: {} | Edges: {} | Density: {:.4}\n\
             - Average in-degree: {:.2} | Average out-degree: {:.2}\n\
             - Strongly connected components: {}\n\
             \n\
             Degree Distribution:\n\
             - In-degree range: [{}, {}] (mean: {:.2}, std: {:.2})\n\
             - Out-degree range: [{}, {}] (mean: {:.2}, std: {:.2})\n\
             - Power-law alpha: {}\n\
             \n\
             Connectivity:\n\
             - Average clustering: {:.4}\n\
             - Average path length: {}\n\
             - Diameter: {} | Is acyclic: {}\n\
             \n\
             Structural Patterns:\n\
             - Top hubs: {} | Top authorities: {}\n\
             - Isolated nodes: {} | Dangling nodes: {} | Leaf nodes: {}\n\
             \n\
             Import Insights:\n\
             - Average import depth: {:.2} | Max depth: {}\n\
             - Circular dependencies: {}\n\
             \n\
             Performance Profile:\n\
             - PageRank memory estimate: {:.1} MB\n\
             - PageRank time estimate: {} ms\n\
             - Graph sparsity: {:.4}\n\
             \n\
             Analysis completed in {} ms (parallel: {})",
            self.basic_stats.total_nodes,
            self.basic_stats.total_edges,
            self.basic_stats.graph_density,
            self.basic_stats.in_degree_avg,
            self.basic_stats.out_degree_avg,
            self.basic_stats.strongly_connected_components,
            self.degree_distribution.in_degree.min,
            self.degree_distribution.in_degree.max,
            self.degree_distribution.in_degree.mean,
            self.degree_distribution.in_degree.std_dev,
            self.degree_distribution.out_degree.min,
            self.degree_distribution.out_degree.max,
            self.degree_distribution.out_degree.mean,
            self.degree_distribution.out_degree.std_dev,
            self.degree_distribution
                .power_law_alpha
                .map_or("N/A".to_string(), |a| format!("{:.3}", a)),
            self.connectivity.average_clustering,
            self.connectivity
                .average_path_length
                .map_or("N/A".to_string(), |l| format!("{:.2}", l)),
            self.connectivity
                .diameter
                .map_or("N/A".to_string(), |d| d.to_string()),
            self.connectivity.is_acyclic,
            self.structural_patterns.hubs.len(),
            self.structural_patterns.authorities.len(),
            self.structural_patterns.isolated_nodes.len(),
            self.structural_patterns.dangling_nodes.len(),
            self.structural_patterns.leaf_nodes.len(),
            self.import_insights.average_import_depth,
            self.import_insights.max_import_depth,
            self.import_insights.circular_dependencies.len(),
            self.performance_profile.pagerank_memory_estimate_mb,
            self.performance_profile.pagerank_time_estimate_ms,
            self.performance_profile.storage_efficiency.sparsity,
            self.analysis_metadata.analysis_duration_ms,
            self.analysis_metadata.used_parallel,
        )
    }

    /// Get a list of the most important nodes by various metrics
    pub fn important_nodes_summary(&self) -> Vec<(String, Vec<String>)> {
        vec![
            (
                "Top Hubs (High Out-degree)".to_string(),
                self.structural_patterns
                    .hubs
                    .iter()
                    .map(|node| format!("{} (out: {})", node.node_id, node.out_degree))
                    .collect(),
            ),
            (
                "Top Authorities (High In-degree)".to_string(),
                self.structural_patterns
                    .authorities
                    .iter()
                    .map(|node| format!("{} (in: {})", node.node_id, node.in_degree))
                    .collect(),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DependencyGraph;

    fn create_test_graph() -> DependencyGraph {
        let mut graph = DependencyGraph::new();

        graph
            .add_edge("main.py".to_string(), "utils.py".to_string())
            .unwrap();
        graph
            .add_edge("main.py".to_string(), "config.py".to_string())
            .unwrap();
        graph
            .add_edge("utils.py".to_string(), "config.py".to_string())
            .unwrap();
        graph
            .add_edge("test.py".to_string(), "main.py".to_string())
            .unwrap();
        graph
            .add_edge("test.py".to_string(), "utils.py".to_string())
            .unwrap();
        graph.add_node("isolated.py".to_string()).unwrap();

        graph
    }

    #[test]
    fn test_statistics_analyzer_creation() {
        // Test that analyzers can be created with different configurations
        let analyzer = GraphStatisticsAnalyzer::new();
        let graph = create_test_graph();
        // Default analyzer should be able to analyze
        let result = analyzer.analyze(&graph);
        assert!(result.is_ok());

        let large_graph_analyzer = GraphStatisticsAnalyzer::for_large_graphs();
        // Large graph analyzer should also work
        let result = large_graph_analyzer.analyze(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_degree_statistics() {
        let analyzer = GraphStatisticsAnalyzer::new();
        let degrees = vec![0, 1, 1, 2, 3, 5, 8];

        let stats = analyzer.compute_degree_stats(&degrees);

        assert_eq!(stats.min, 0);
        assert_eq!(stats.max, 8);
        assert!((stats.mean - 2.857).abs() < 0.01);
        assert_eq!(stats.median, 2.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let analyzer = GraphStatisticsAnalyzer::new();
        let sorted_values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        assert_eq!(analyzer.percentile(&sorted_values, 0.0), 1.0);
        assert_eq!(analyzer.percentile(&sorted_values, 50.0), 5.5);
        assert_eq!(analyzer.percentile(&sorted_values, 100.0), 10.0);
        assert!((analyzer.percentile(&sorted_values, 25.0) - 3.25).abs() < 0.01);
    }

    #[test]
    fn test_histogram_computation() {
        let analyzer = GraphStatisticsAnalyzer::new();
        let degrees = vec![1, 1, 2, 2, 2, 3];

        let histogram = analyzer.compute_histogram(&degrees);

        assert_eq!(histogram[&1], 2);
        assert_eq!(histogram[&2], 3);
        assert_eq!(histogram[&3], 1);
    }

    #[test]
    fn test_clustering_coefficient() {
        let mut graph = DependencyGraph::new();

        graph.add_edge("A".to_string(), "B".to_string()).unwrap();
        graph.add_edge("B".to_string(), "C".to_string()).unwrap();
        graph.add_edge("C".to_string(), "A".to_string()).unwrap();

        let analyzer = GraphStatisticsAnalyzer::new();
        let clustering = analyzer.local_clustering_coefficient(&graph, &"A".to_string());

        assert!(clustering >= 0.0);
        assert!(clustering <= 1.0);
    }

    #[test]
    fn test_bfs_distances() {
        let graph = create_test_graph();
        let analyzer = GraphStatisticsAnalyzer::new();

        let distances = analyzer.bfs_distances(&graph, &"main.py".to_string());

        assert_eq!(distances["main.py"], 0);
        assert!(distances.contains_key("utils.py"));
        assert!(distances.contains_key("config.py"));
        assert!(distances["utils.py"] <= distances["config.py"]);
    }

    #[test]
    fn test_full_analysis() {
        let graph = create_test_graph();
        let analyzer = GraphStatisticsAnalyzer::new();

        let analysis = analyzer.analyze(&graph).unwrap();

        assert_eq!(analysis.basic_stats.total_nodes, 5);
        assert_eq!(analysis.basic_stats.total_edges, 5);
        assert!(analysis.degree_distribution.in_degree.mean >= 0.0);
        assert!(analysis.degree_distribution.out_degree.mean >= 0.0);
        assert!(analysis.connectivity.graph_density >= 0.0);
        assert!(analysis.connectivity.graph_density <= 1.0);
        assert!(analysis.performance_profile.pagerank_memory_estimate_mb >= 0.0);
        assert!(analysis.performance_profile.pagerank_time_estimate_ms >= 0);
        assert!(analysis.analysis_metadata.analysis_duration_ms >= 0);
        assert_eq!(
            analysis.analysis_metadata.version,
            env!("CARGO_PKG_VERSION")
        );
    }

    #[test]
    fn test_summary_report() {
        let graph = create_test_graph();
        let analyzer = GraphStatisticsAnalyzer::new();
        let analysis = analyzer.analyze(&graph).unwrap();

        let summary = analysis.summary_report();

        assert!(summary.contains("Graph Analysis Summary Report"));
        assert!(summary.contains("Basic Statistics"));
        assert!(summary.contains("Degree Distribution"));
        assert!(summary.contains("Connectivity"));
        assert!(summary.contains("Structural Patterns"));
        assert!(summary.contains("Import Insights"));
        assert!(summary.contains("Performance Profile"));
    }

    #[test]
    fn test_important_nodes_summary() {
        let graph = create_test_graph();
        let analyzer = GraphStatisticsAnalyzer::new();
        let analysis = analyzer.analyze(&graph).unwrap();

        let important_nodes = analysis.important_nodes_summary();

        assert!(!important_nodes.is_empty());
        assert_eq!(important_nodes.len(), 2);
    }

    #[test]
    fn test_empty_graph_analysis() {
        let graph = DependencyGraph::new();
        let analyzer = GraphStatisticsAnalyzer::new();

        let analysis = analyzer.analyze(&graph).unwrap();

        assert_eq!(analysis.basic_stats.total_nodes, 0);
        assert_eq!(analysis.basic_stats.total_edges, 0);
        assert_eq!(analysis.degree_distribution.in_degree.mean, 0.0);
        assert!(analysis.structural_patterns.hubs.is_empty());
        assert!(analysis.structural_patterns.authorities.is_empty());
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = GraphStatisticsAnalyzer::default();
        let graph = create_test_graph();
        let result = analyzer.analyze(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_graph_power_law() {
        // Create a graph with power-law-like degree distribution
        let mut graph = DependencyGraph::new();

        // Create hub node with many dependencies
        for i in 0..20 {
            graph
                .add_edge("hub".to_string(), format!("dep{}", i))
                .unwrap();
        }

        // Create nodes with varying degrees
        for i in 0..15 {
            graph
                .add_edge(format!("src{}", i), "hub".to_string())
                .unwrap();
        }

        for i in 0..10 {
            graph
                .add_edge(format!("mod{}", i), format!("dep{}", i % 5))
                .unwrap();
        }

        let analyzer = GraphStatisticsAnalyzer::new();
        let analysis = analyzer.analyze(&graph).unwrap();

        // Check that analysis completes for a larger graph
        assert!(analysis.basic_stats.total_nodes > 30);
        assert!(analysis.basic_stats.total_edges > 40);
        // Power law alpha may or may not be computed
        let _ = analysis.degree_distribution.power_law_alpha;
    }

    #[test]
    fn test_config_custom_values() {
        let config = StatisticsConfig {
            compute_expensive_metrics: false,
            max_nodes_for_expensive_ops: 100,
            top_nodes_count: 5,
            pattern_threshold: 0.2,
            use_parallel: false,
        };

        let analyzer = GraphStatisticsAnalyzer::with_config(config);
        let graph = create_test_graph();
        let analysis = analyzer.analyze(&graph).unwrap();

        // Should complete without expensive metrics
        assert!(analysis.connectivity.average_clustering >= 0.0);
    }

    #[test]
    fn test_analysis_metadata() {
        let graph = create_test_graph();
        let analyzer = GraphStatisticsAnalyzer::new();
        let analysis = analyzer.analyze(&graph).unwrap();

        assert!(!analysis.analysis_metadata.version.is_empty());
        assert!(analysis.analysis_metadata.analysis_duration_ms >= 0);
        // Timestamp should exist
        let _ = analysis.analysis_metadata.timestamp;
    }
}

//! Graph statistics analyzer implementation.

use scribe_core::Result;
use std::collections::{HashMap, VecDeque};

use crate::graph::{DependencyGraph, GraphStatistics, NodeId};

use super::types::{
    AnalysisMetadata, ConnectivityAnalysis, DegreeDistribution, DegreeStats, GraphAnalysisResults,
    ImportInsights, NodeInfo, PerformanceProfile, StatisticsConfig, StorageEfficiency,
    StructuralPatterns, TraversalComplexity,
};

/// Main graph statistics analyzer
#[derive(Debug)]
pub struct GraphStatisticsAnalyzer {
    config: StatisticsConfig,
}

impl GraphStatisticsAnalyzer {
    /// Create a new analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: StatisticsConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StatisticsConfig) -> Self {
        Self { config }
    }

    /// Create analyzer optimized for large graphs
    pub fn for_large_graphs() -> Self {
        Self {
            config: StatisticsConfig {
                compute_expensive_metrics: false,
                max_nodes_for_expensive_ops: 5000,
                top_nodes_count: 5,
                ..StatisticsConfig::default()
            },
        }
    }

    /// Perform comprehensive analysis of the dependency graph
    pub fn analyze(&self, graph: &DependencyGraph) -> Result<GraphAnalysisResults> {
        let start_time = std::time::Instant::now();

        let basic_stats = self.compute_basic_statistics(graph);
        let degree_distribution = self.analyze_degree_distribution(graph)?;
        let connectivity = self.analyze_connectivity(graph)?;
        let structural_patterns = self.identify_structural_patterns(graph)?;
        let import_insights = self.analyze_import_patterns(graph)?;
        let performance_profile = self.estimate_performance_characteristics(graph);

        let analysis_metadata = AnalysisMetadata {
            timestamp: chrono::Utc::now(),
            analysis_duration_ms: start_time.elapsed().as_millis() as u64,
            used_parallel: self.config.use_parallel,
            config: self.config.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        Ok(GraphAnalysisResults {
            basic_stats,
            degree_distribution,
            connectivity,
            structural_patterns,
            import_insights,
            performance_profile,
            analysis_metadata,
        })
    }

    /// Compute basic graph statistics
    fn compute_basic_statistics(&self, graph: &DependencyGraph) -> GraphStatistics {
        let total_nodes = graph.node_count();
        let total_edges = graph.edge_count();

        if total_nodes == 0 {
            return GraphStatistics::empty();
        }

        let degrees: Vec<_> = graph
            .nodes()
            .map(|node| (graph.in_degree(node), graph.out_degree(node)))
            .collect();

        let in_degrees: Vec<_> = degrees.iter().map(|(in_deg, _)| *in_deg).collect();
        let out_degrees: Vec<_> = degrees.iter().map(|(_, out_deg)| *out_deg).collect();

        let in_degree_avg = in_degrees.iter().sum::<usize>() as f64 / total_nodes as f64;
        let in_degree_max = *in_degrees.iter().max().unwrap_or(&0);
        let out_degree_avg = out_degrees.iter().sum::<usize>() as f64 / total_nodes as f64;
        let out_degree_max = *out_degrees.iter().max().unwrap_or(&0);

        let isolated_nodes = degrees
            .iter()
            .filter(|(in_deg, out_deg)| *in_deg == 0 && *out_deg == 0)
            .count();
        let dangling_nodes = degrees.iter().filter(|(_, out_deg)| *out_deg == 0).count();

        let max_possible_edges = total_nodes * (total_nodes - 1);
        let graph_density = if max_possible_edges > 0 {
            total_edges as f64 / max_possible_edges as f64
        } else {
            0.0
        };

        GraphStatistics {
            total_nodes,
            total_edges,
            in_degree_avg,
            in_degree_max,
            out_degree_avg,
            out_degree_max,
            strongly_connected_components: graph.estimate_scc_count(),
            graph_density,
            isolated_nodes,
            dangling_nodes,
        }
    }

    /// Analyze degree distributions in detail
    fn analyze_degree_distribution(&self, graph: &DependencyGraph) -> Result<DegreeDistribution> {
        let degrees: Vec<_> = graph
            .nodes()
            .map(|node| (graph.in_degree(node), graph.out_degree(node)))
            .collect();

        let in_degrees: Vec<_> = degrees.iter().map(|(in_deg, _)| *in_deg).collect();
        let out_degrees: Vec<_> = degrees.iter().map(|(_, out_deg)| *out_deg).collect();
        let total_degrees: Vec<_> = degrees
            .iter()
            .map(|(in_deg, out_deg)| in_deg + out_deg)
            .collect();

        Ok(DegreeDistribution {
            in_degree: self.compute_degree_stats(&in_degrees),
            out_degree: self.compute_degree_stats(&out_degrees),
            total_degree: self.compute_degree_stats(&total_degrees),
            in_degree_histogram: self.compute_histogram(&in_degrees),
            out_degree_histogram: self.compute_histogram(&out_degrees),
            power_law_alpha: self.estimate_power_law_alpha(&in_degrees),
            power_law_goodness_of_fit: None,
        })
    }

    /// Compute detailed statistics for a degree sequence
    pub fn compute_degree_stats(&self, degrees: &[usize]) -> DegreeStats {
        if degrees.is_empty() {
            return DegreeStats::default();
        }

        let mut sorted_degrees = degrees.to_vec();
        sorted_degrees.sort();

        let min = sorted_degrees[0];
        let max = sorted_degrees[sorted_degrees.len() - 1];
        let mean = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;

        let median = self.percentile(&sorted_degrees, 50.0);
        let percentile_25 = self.percentile(&sorted_degrees, 25.0);
        let percentile_75 = self.percentile(&sorted_degrees, 75.0);
        let percentile_90 = self.percentile(&sorted_degrees, 90.0);
        let percentile_95 = self.percentile(&sorted_degrees, 95.0);
        let percentile_99 = self.percentile(&sorted_degrees, 99.0);

        let variance = degrees
            .iter()
            .map(|&deg| (deg as f64 - mean).powi(2))
            .sum::<f64>()
            / degrees.len() as f64;
        let std_dev = variance.sqrt();

        DegreeStats {
            min,
            max,
            mean,
            median,
            std_dev,
            percentile_25,
            percentile_75,
            percentile_90,
            percentile_95,
            percentile_99,
        }
    }

    /// Calculate percentile from sorted array
    pub fn percentile(&self, sorted_values: &[usize], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_values[lower] as f64
        } else {
            let weight = index - lower as f64;
            (1.0 - weight) * sorted_values[lower] as f64 + weight * sorted_values[upper] as f64
        }
    }

    /// Compute degree histogram
    pub fn compute_histogram(&self, degrees: &[usize]) -> HashMap<usize, usize> {
        let mut histogram = HashMap::new();
        for &degree in degrees {
            *histogram.entry(degree).or_insert(0) += 1;
        }
        histogram
    }

    /// Estimate power-law exponent alpha using linear regression on log-log plot
    fn estimate_power_law_alpha(&self, degrees: &[usize]) -> Option<f64> {
        let non_zero_degrees: Vec<_> = degrees.iter().filter(|&&d| d > 0).collect();

        if non_zero_degrees.len() < 10 {
            return None;
        }

        let mut log_points = Vec::new();
        let histogram =
            self.compute_histogram(&non_zero_degrees.into_iter().copied().collect::<Vec<_>>());

        for (&degree, &count) in &histogram {
            if count > 0 {
                log_points.push((degree as f64, count as f64));
            }
        }

        if log_points.len() < 5 {
            return None;
        }

        let n = log_points.len() as f64;
        let sum_log_x: f64 = log_points.iter().map(|(x, _)| x.ln()).sum();
        let sum_log_y: f64 = log_points.iter().map(|(_, y)| y.ln()).sum();
        let sum_log_x_log_y: f64 = log_points.iter().map(|(x, y)| x.ln() * y.ln()).sum();
        let sum_log_x_squared: f64 = log_points.iter().map(|(x, _)| x.ln().powi(2)).sum();

        let denominator = n * sum_log_x_squared - sum_log_x.powi(2);
        if denominator.abs() < 1e-10 {
            return None;
        }

        let alpha = -(n * sum_log_x_log_y - sum_log_x * sum_log_y) / denominator;

        if alpha > 0.0 {
            Some(alpha)
        } else {
            None
        }
    }

    /// Analyze graph connectivity properties
    fn analyze_connectivity(&self, graph: &DependencyGraph) -> Result<ConnectivityAnalysis> {
        let strongly_connected_components = graph.estimate_scc_count();
        let largest_scc_size = self.estimate_largest_scc_size(graph);
        let weakly_connected_components = strongly_connected_components;

        let graph_density = if graph.node_count() > 1 {
            let max_edges = graph.node_count() * (graph.node_count() - 1);
            graph.edge_count() as f64 / max_edges as f64
        } else {
            0.0
        };

        let (average_clustering, global_clustering, average_path_length, diameter) =
            if self.config.compute_expensive_metrics
                && graph.node_count() <= self.config.max_nodes_for_expensive_ops
            {
                (
                    Some(self.estimate_average_clustering(graph)),
                    Some(self.estimate_global_clustering(graph)),
                    self.estimate_average_path_length(graph),
                    self.estimate_diameter(graph),
                )
            } else {
                (None, None, None, None)
            };

        let is_acyclic = self.estimate_is_acyclic(graph);

        Ok(ConnectivityAnalysis {
            weakly_connected_components,
            strongly_connected_components,
            largest_scc_size,
            graph_density,
            average_clustering: average_clustering.unwrap_or(0.0),
            global_clustering: global_clustering.unwrap_or(0.0),
            average_path_length,
            diameter,
            is_acyclic,
        })
    }

    /// Estimate size of largest strongly connected component
    fn estimate_largest_scc_size(&self, graph: &DependencyGraph) -> usize {
        graph
            .nodes()
            .map(|node| std::cmp::min(graph.in_degree(node) + 1, graph.out_degree(node) + 1))
            .max()
            .unwrap_or(0)
    }

    /// Estimate average clustering coefficient
    fn estimate_average_clustering(&self, graph: &DependencyGraph) -> f64 {
        let clustering_coefficients: Vec<f64> = graph
            .nodes()
            .map(|node| self.local_clustering_coefficient(graph, node))
            .collect();

        if clustering_coefficients.is_empty() {
            0.0
        } else {
            clustering_coefficients.iter().sum::<f64>() / clustering_coefficients.len() as f64
        }
    }

    /// Compute local clustering coefficient for a node
    pub fn local_clustering_coefficient(&self, graph: &DependencyGraph, node: &NodeId) -> f64 {
        let neighbors = graph.all_neighbors(node);
        let k = neighbors.len();

        if k < 2 {
            return 0.0;
        }

        let mut edges_between_neighbors = 0;
        let neighbor_vec: Vec<_> = neighbors.iter().collect();

        for i in 0..neighbor_vec.len() {
            for j in (i + 1)..neighbor_vec.len() {
                if graph.contains_edge(neighbor_vec[i], neighbor_vec[j])
                    || graph.contains_edge(neighbor_vec[j], neighbor_vec[i])
                {
                    edges_between_neighbors += 1;
                }
            }
        }

        let max_possible_edges = k * (k - 1) / 2;
        edges_between_neighbors as f64 / max_possible_edges as f64
    }

    /// Estimate global clustering coefficient (transitivity)
    fn estimate_global_clustering(&self, graph: &DependencyGraph) -> f64 {
        let mut triangles = 0;
        let mut triplets = 0;

        for node in graph.nodes() {
            let neighbors = graph.all_neighbors(node);
            if neighbors.len() < 2 {
                continue;
            }

            let neighbor_vec: Vec<_> = neighbors.iter().collect();

            for i in 0..neighbor_vec.len() {
                for j in (i + 1)..neighbor_vec.len() {
                    triplets += 1;
                    if graph.contains_edge(neighbor_vec[i], neighbor_vec[j])
                        || graph.contains_edge(neighbor_vec[j], neighbor_vec[i])
                    {
                        triangles += 1;
                    }
                }
            }
        }

        if triplets > 0 {
            3.0 * triangles as f64 / triplets as f64
        } else {
            0.0
        }
    }

    /// Estimate average path length using sampling
    pub fn estimate_average_path_length(&self, graph: &DependencyGraph) -> Option<f64> {
        let nodes: Vec<_> = graph.nodes().collect();
        if nodes.len() < 2 {
            return None;
        }

        let sample_size = std::cmp::min(100, nodes.len());
        let mut total_path_length = 0.0;
        let mut valid_paths = 0;

        for i in 0..sample_size {
            let start_node = &nodes[i % nodes.len()];
            let distances = self.bfs_distances(graph, start_node);

            for distance in distances.values() {
                if *distance > 0 && *distance < usize::MAX {
                    total_path_length += *distance as f64;
                    valid_paths += 1;
                }
            }
        }

        if valid_paths > 0 {
            Some(total_path_length / valid_paths as f64)
        } else {
            None
        }
    }

    /// Estimate graph diameter
    pub fn estimate_diameter(&self, graph: &DependencyGraph) -> Option<usize> {
        let nodes: Vec<_> = graph.nodes().collect();
        if nodes.is_empty() {
            return None;
        }

        let mut max_distance = 0;
        let sample_size = std::cmp::min(20, nodes.len());

        for i in 0..sample_size {
            let start_node = &nodes[i % nodes.len()];
            let distances = self.bfs_distances(graph, start_node);

            for &distance in distances.values() {
                if distance != usize::MAX && distance > max_distance {
                    max_distance = distance;
                }
            }
        }

        if max_distance > 0 {
            Some(max_distance)
        } else {
            None
        }
    }

    /// BFS to compute distances from a source node
    pub fn bfs_distances(
        &self,
        graph: &DependencyGraph,
        source: &NodeId,
    ) -> HashMap<NodeId, usize> {
        let mut distances = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(source.clone(), 0);
        queue.push_back(source.clone());

        while let Some(current) = queue.pop_front() {
            let current_distance = distances[&current];

            if let Some(outgoing) = graph.outgoing_neighbors(&current) {
                for neighbor in outgoing {
                    if !distances.contains_key(neighbor) {
                        distances.insert(neighbor.clone(), current_distance + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }

            if let Some(incoming) = graph.incoming_neighbors(&current) {
                for neighbor in incoming {
                    if !distances.contains_key(neighbor) {
                        distances.insert(neighbor.clone(), current_distance + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        distances
    }

    /// Check if graph is likely acyclic (DAG)
    fn estimate_is_acyclic(&self, graph: &DependencyGraph) -> bool {
        let nodes_with_both = graph
            .nodes()
            .filter(|&node| graph.in_degree(node) > 0 && graph.out_degree(node) > 0)
            .count();

        let total_nodes = graph.node_count();
        if total_nodes == 0 {
            return true;
        }

        let bidirectional_ratio = nodes_with_both as f64 / total_nodes as f64;
        bidirectional_ratio < 0.3
    }

    /// Identify structural patterns in the graph
    fn identify_structural_patterns(&self, graph: &DependencyGraph) -> Result<StructuralPatterns> {
        let mut hub_candidates: Vec<_> = graph
            .nodes()
            .map(|node| NodeInfo {
                node_id: node.clone(),
                score: graph.out_degree(node) as f64,
                in_degree: graph.in_degree(node),
                out_degree: graph.out_degree(node),
                metadata: None,
            })
            .collect();
        hub_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let hubs = hub_candidates
            .into_iter()
            .take(self.config.top_nodes_count)
            .collect();

        let mut authority_candidates: Vec<_> = graph
            .nodes()
            .map(|node| NodeInfo {
                node_id: node.clone(),
                score: graph.in_degree(node) as f64,
                in_degree: graph.in_degree(node),
                out_degree: graph.out_degree(node),
                metadata: None,
            })
            .collect();
        authority_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let authorities = authority_candidates
            .into_iter()
            .take(self.config.top_nodes_count)
            .collect();

        let isolated_nodes: Vec<_> = graph
            .nodes()
            .filter(|&node| graph.degree(node) == 0)
            .cloned()
            .collect();

        let dangling_nodes = graph.dangling_nodes().into_iter().cloned().collect();

        let leaf_nodes: Vec<_> = graph
            .nodes()
            .filter(|&node| graph.in_degree(node) == 0 && graph.out_degree(node) > 0)
            .cloned()
            .collect();

        let bottlenecks = Vec::new();
        let bridges = Vec::new();

        Ok(StructuralPatterns {
            hubs,
            authorities,
            bottlenecks,
            bridges,
            isolated_nodes,
            dangling_nodes,
            leaf_nodes,
        })
    }

    /// Analyze import-specific patterns
    fn analyze_import_patterns(&self, graph: &DependencyGraph) -> Result<ImportInsights> {
        let fan_out_stats = {
            let out_degrees: Vec<_> = graph.nodes().map(|node| graph.out_degree(node)).collect();
            self.compute_degree_stats(&out_degrees)
        };

        let fan_in_stats = {
            let in_degrees: Vec<_> = graph.nodes().map(|node| graph.in_degree(node)).collect();
            self.compute_degree_stats(&in_degrees)
        };

        let circular_dependencies = Vec::new();
        let dependency_layers = Vec::new();
        let critical_paths = Vec::new();

        let max_import_depth = self.estimate_diameter(graph).unwrap_or(0);
        let average_import_depth = self.estimate_average_path_length(graph).unwrap_or(0.0);

        Ok(ImportInsights {
            average_import_depth,
            max_import_depth,
            fan_out_distribution: fan_out_stats,
            fan_in_distribution: fan_in_stats,
            circular_dependencies,
            dependency_layers,
            critical_paths,
        })
    }

    /// Estimate performance characteristics for various graph algorithms
    fn estimate_performance_characteristics(&self, graph: &DependencyGraph) -> PerformanceProfile {
        let n = graph.node_count();
        let m = graph.edge_count();

        let pagerank_memory_mb = if n > 0 {
            let score_vector_size = n * std::mem::size_of::<f64>();
            let graph_overhead = m * (std::mem::size_of::<String>() + std::mem::size_of::<usize>());
            ((score_vector_size * 2 + graph_overhead) as f64) / (1024.0 * 1024.0)
        } else {
            0.0
        };

        let pagerank_time_estimate_ms = if n > 0 {
            let base_time = 1;
            let per_iteration_time = (n + m) / 10000;
            let estimated_iterations = if n < 1000 { 10 } else { 20 };
            base_time + estimated_iterations * per_iteration_time.max(1)
        } else {
            0
        } as u64;

        let traversal_complexity = TraversalComplexity {
            time_complexity_class: format!("O(V + E) = O({} + {})", n, m),
            space_complexity_class: format!("O(V) = O({})", n),
            expected_iterations: if n < 1000 { 10 } else { 20 },
        };

        let edges_per_node = if n > 0 { m as f64 / n as f64 } else { 0.0 };
        let max_possible_edges = if n > 1 { n * (n - 1) } else { 1 };
        let sparsity = 1.0 - (m as f64 / max_possible_edges as f64);

        let storage_efficiency = StorageEfficiency {
            adjacency_list_size_bytes: m
                * (std::mem::size_of::<String>() + std::mem::size_of::<usize>()),
            edges_per_node,
            memory_overhead_ratio: 1.5,
            sparsity,
        };

        PerformanceProfile {
            pagerank_memory_estimate_mb: pagerank_memory_mb,
            pagerank_time_estimate_ms,
            traversal_complexity,
            storage_efficiency,
        }
    }
}

impl Default for GraphStatisticsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

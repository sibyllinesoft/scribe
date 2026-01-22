//! Type definitions for centrality calculation
//!
//! Contains all the configuration and result types used by the centrality
//! calculation system for PageRank-based code importance analysis.

use crate::pagerank::{PageRankConfig, PageRankResults};
use crate::statistics::GraphAnalysisResults;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete centrality calculation results with comprehensive metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentralityResults {
    /// PageRank scores (file path -> centrality score)
    pub pagerank_scores: HashMap<String, f64>,

    /// Graph analysis results
    pub graph_analysis: GraphAnalysisResults,

    /// PageRank computation details
    pub pagerank_details: PageRankResults,

    /// Import detection statistics
    pub import_stats: ImportDetectionStats,

    /// Integration metadata
    pub integration_metadata: IntegrationMetadata,
}

/// Statistics about import detection and graph construction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportDetectionStats {
    /// Number of files processed for import detection
    pub files_processed: usize,

    /// Number of import relationships detected
    pub imports_detected: usize,

    /// Number of resolved imports (mapped to actual files)
    pub imports_resolved: usize,

    /// Import resolution success rate
    pub resolution_rate: f64,

    /// Language breakdown of processed files
    pub language_breakdown: HashMap<String, usize>,

    /// Import patterns by language
    pub import_patterns: HashMap<String, ImportPatternStats>,
}

/// Import pattern statistics for a specific language
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportPatternStats {
    /// Total imports found
    pub total_imports: usize,

    /// Relative imports (./,../)
    pub relative_imports: usize,

    /// Absolute imports
    pub absolute_imports: usize,

    /// Standard library imports
    pub stdlib_imports: usize,

    /// Third-party imports
    pub third_party_imports: usize,
}

/// Metadata about centrality-heuristics integration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrationMetadata {
    /// When the analysis was performed
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Total computation time
    pub computation_time_ms: u64,

    /// Whether centrality was successfully integrated
    pub integration_successful: bool,

    /// Centrality weight used in integration
    pub centrality_weight: f64,

    /// Number of files with centrality scores
    pub files_with_centrality: usize,

    /// Configuration used
    pub config: CentralityConfig,
}

/// Configuration for centrality calculation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentralityConfig {
    /// PageRank algorithm configuration
    pub pagerank_config: PageRankConfig,

    /// Whether to perform expensive graph analysis
    pub analyze_graph_structure: bool,

    /// Import resolution configuration
    pub import_resolution: ImportResolutionConfig,

    /// Integration parameters
    pub integration: IntegrationConfig,
}

/// Configuration for import resolution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportResolutionConfig {
    /// Maximum search depth for import resolution
    pub max_search_depth: usize,

    /// Whether to resolve relative imports
    pub resolve_relative_imports: bool,

    /// Whether to resolve absolute imports
    pub resolve_absolute_imports: bool,

    /// Whether to exclude standard library imports
    pub exclude_stdlib_imports: bool,

    /// Custom import path mappings
    pub path_mappings: HashMap<String, String>,
}

/// Configuration for heuristics integration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Weight for centrality in final score
    pub centrality_weight: f64,

    /// Normalization method for centrality scores
    pub normalization_method: NormalizationMethod,

    /// Minimum centrality score threshold
    pub min_centrality_threshold: f64,

    /// Whether to boost entrypoint centrality
    pub boost_entrypoints: bool,

    /// Entrypoint boost factor
    pub entrypoint_boost_factor: f64,
}

/// Methods for normalizing centrality scores
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Normalize to \[0,1\] range
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Rank-based normalization
    Rank,
    /// No normalization
    None,
}

impl Default for CentralityConfig {
    fn default() -> Self {
        Self {
            pagerank_config: PageRankConfig::for_code_analysis(),
            analyze_graph_structure: true,
            import_resolution: ImportResolutionConfig::default(),
            integration: IntegrationConfig::default(),
        }
    }
}

impl Default for ImportResolutionConfig {
    fn default() -> Self {
        Self {
            max_search_depth: 3,
            resolve_relative_imports: true,
            resolve_absolute_imports: true,
            exclude_stdlib_imports: true,
            path_mappings: HashMap::new(),
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            centrality_weight: 0.15, // 15% weight in V2 scoring
            normalization_method: NormalizationMethod::MinMax,
            min_centrality_threshold: 1e-6,
            boost_entrypoints: true,
            entrypoint_boost_factor: 1.5,
        }
    }
}

/// Utility functions for centrality results analysis
impl CentralityResults {
    /// Get files sorted by centrality score (descending)
    pub fn top_files_by_centrality(&self, k: usize) -> Vec<(String, f64)> {
        let mut scored_files: Vec<_> = self
            .pagerank_scores
            .iter()
            .map(|(path, &score)| (path.clone(), score))
            .collect();

        scored_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_files.into_iter().take(k).collect()
    }

    /// Get summary statistics about centrality computation
    pub fn summary(&self) -> String {
        format!(
            "Centrality Analysis Summary:\n\
             - Files with centrality scores: {}\n\
             - PageRank iterations: {} (converged: {})\n\
             - Graph: {} nodes, {} edges (density: {:.4})\n\
             - Import resolution: {:.1}% ({}/{})\n\
             - Top languages: {}\n\
             - Computation time: {}ms\n\
             - Integration weight: {:.2}",
            self.pagerank_scores.len(),
            self.pagerank_details.iterations_converged,
            self.pagerank_details.converged(),
            self.graph_analysis.basic_stats.total_nodes,
            self.graph_analysis.basic_stats.total_edges,
            self.graph_analysis.basic_stats.graph_density,
            self.import_stats.resolution_rate * 100.0,
            self.import_stats.imports_resolved,
            self.import_stats.imports_detected,
            self.import_stats
                .language_breakdown
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(lang, count)| format!("{} ({})", lang, count))
                .unwrap_or_else(|| "None".to_string()),
            self.integration_metadata.computation_time_ms,
            self.integration_metadata.centrality_weight,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphStatistics;
    use crate::pagerank::PerformanceMetrics;

    #[test]
    fn test_centrality_config_default() {
        let config = CentralityConfig::default();
        assert!(config.analyze_graph_structure);
        assert_eq!(config.integration.centrality_weight, 0.15);
    }

    #[test]
    fn test_import_resolution_config_default() {
        let config = ImportResolutionConfig::default();
        assert_eq!(config.max_search_depth, 3);
        assert!(config.resolve_relative_imports);
        assert!(config.resolve_absolute_imports);
        assert!(config.exclude_stdlib_imports);
        assert!(config.path_mappings.is_empty());
    }

    #[test]
    fn test_integration_config_default() {
        let config = IntegrationConfig::default();
        assert_eq!(config.centrality_weight, 0.15);
        assert_eq!(config.normalization_method, NormalizationMethod::MinMax);
        assert!(config.min_centrality_threshold > 0.0);
        assert!(config.boost_entrypoints);
        assert_eq!(config.entrypoint_boost_factor, 1.5);
    }

    #[test]
    fn test_normalization_method_equality() {
        assert_eq!(NormalizationMethod::MinMax, NormalizationMethod::MinMax);
        assert_eq!(NormalizationMethod::ZScore, NormalizationMethod::ZScore);
        assert_eq!(NormalizationMethod::Rank, NormalizationMethod::Rank);
        assert_eq!(NormalizationMethod::None, NormalizationMethod::None);
        assert_ne!(NormalizationMethod::MinMax, NormalizationMethod::ZScore);
    }

    #[test]
    fn test_normalization_method_clone() {
        let method = NormalizationMethod::MinMax;
        let cloned = method.clone();
        assert_eq!(method, cloned);
    }

    #[test]
    fn test_import_pattern_stats() {
        let stats = ImportPatternStats {
            total_imports: 100,
            relative_imports: 30,
            absolute_imports: 50,
            stdlib_imports: 15,
            third_party_imports: 5,
        };

        assert_eq!(stats.total_imports, 100);
        assert_eq!(stats.relative_imports, 30);
        assert_eq!(stats.stdlib_imports, 15);
    }

    #[test]
    fn test_import_pattern_stats_clone() {
        let stats = ImportPatternStats {
            total_imports: 50,
            relative_imports: 10,
            absolute_imports: 30,
            stdlib_imports: 5,
            third_party_imports: 5,
        };

        let cloned = stats.clone();
        assert_eq!(stats.total_imports, cloned.total_imports);
        assert_eq!(stats.relative_imports, cloned.relative_imports);
    }

    #[test]
    fn test_import_detection_stats() {
        let mut language_breakdown = HashMap::new();
        language_breakdown.insert("Rust".to_string(), 50);
        language_breakdown.insert("Python".to_string(), 30);

        let stats = ImportDetectionStats {
            files_processed: 100,
            imports_detected: 200,
            imports_resolved: 180,
            resolution_rate: 0.9,
            language_breakdown,
            import_patterns: HashMap::new(),
        };

        assert_eq!(stats.files_processed, 100);
        assert_eq!(stats.imports_detected, 200);
        assert_eq!(stats.imports_resolved, 180);
        assert!((stats.resolution_rate - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_integration_metadata() {
        let metadata = IntegrationMetadata {
            timestamp: chrono::Utc::now(),
            computation_time_ms: 150,
            integration_successful: true,
            centrality_weight: 0.15,
            files_with_centrality: 500,
            config: CentralityConfig::default(),
        };

        assert!(metadata.integration_successful);
        assert_eq!(metadata.computation_time_ms, 150);
        assert_eq!(metadata.centrality_weight, 0.15);
        assert_eq!(metadata.files_with_centrality, 500);
    }

    #[test]
    fn test_integration_metadata_clone() {
        let metadata = IntegrationMetadata {
            timestamp: chrono::Utc::now(),
            computation_time_ms: 100,
            integration_successful: true,
            centrality_weight: 0.2,
            files_with_centrality: 200,
            config: CentralityConfig::default(),
        };

        let cloned = metadata.clone();
        assert_eq!(metadata.computation_time_ms, cloned.computation_time_ms);
        assert_eq!(metadata.centrality_weight, cloned.centrality_weight);
    }

    #[test]
    fn test_centrality_config_clone() {
        let config = CentralityConfig::default();
        let cloned = config.clone();

        assert_eq!(
            config.analyze_graph_structure,
            cloned.analyze_graph_structure
        );
        assert_eq!(
            config.integration.centrality_weight,
            cloned.integration.centrality_weight
        );
    }

    #[test]
    fn test_import_resolution_config_clone() {
        let config = ImportResolutionConfig::default();
        let cloned = config.clone();

        assert_eq!(config.max_search_depth, cloned.max_search_depth);
        assert_eq!(
            config.resolve_relative_imports,
            cloned.resolve_relative_imports
        );
    }

    #[test]
    fn test_integration_config_clone() {
        let config = IntegrationConfig::default();
        let cloned = config.clone();

        assert_eq!(config.centrality_weight, cloned.centrality_weight);
        assert_eq!(config.normalization_method, cloned.normalization_method);
    }

    fn create_mock_centrality_results() -> CentralityResults {
        use crate::statistics::{AnalysisMetadata, PerformanceProfile};

        let mut pagerank_scores = HashMap::new();
        pagerank_scores.insert("main.rs".to_string(), 0.9);
        pagerank_scores.insert("lib.rs".to_string(), 0.7);
        pagerank_scores.insert("utils.rs".to_string(), 0.5);

        let mut language_breakdown = HashMap::new();
        language_breakdown.insert("Rust".to_string(), 3);

        CentralityResults {
            pagerank_scores,
            graph_analysis: GraphAnalysisResults {
                basic_stats: GraphStatistics::empty(),
                degree_distribution: Default::default(),
                connectivity: Default::default(),
                structural_patterns: Default::default(),
                import_insights: Default::default(),
                performance_profile: PerformanceProfile::default(),
                analysis_metadata: AnalysisMetadata::default(),
            },
            pagerank_details: crate::pagerank::PageRankResults {
                scores: HashMap::new(),
                iterations_converged: 10,
                convergence_epsilon: 1e-8,
                graph_stats: GraphStatistics::empty(),
                parameters: PageRankConfig::default(),
                performance_metrics: PerformanceMetrics::default(),
            },
            import_stats: ImportDetectionStats {
                files_processed: 3,
                imports_detected: 10,
                imports_resolved: 8,
                resolution_rate: 0.8,
                language_breakdown,
                import_patterns: HashMap::new(),
            },
            integration_metadata: IntegrationMetadata {
                timestamp: chrono::Utc::now(),
                computation_time_ms: 50,
                integration_successful: true,
                centrality_weight: 0.15,
                files_with_centrality: 3,
                config: CentralityConfig::default(),
            },
        }
    }

    #[test]
    fn test_centrality_results_top_files() {
        let results = create_mock_centrality_results();

        let top_2 = results.top_files_by_centrality(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].0, "main.rs");
        assert!((top_2[0].1 - 0.9).abs() < 0.01);
        assert_eq!(top_2[1].0, "lib.rs");
    }

    #[test]
    fn test_centrality_results_top_files_more_than_available() {
        let results = create_mock_centrality_results();

        let top_10 = results.top_files_by_centrality(10);
        assert_eq!(top_10.len(), 3); // Only 3 files available
    }

    #[test]
    fn test_centrality_results_summary() {
        let results = create_mock_centrality_results();
        let summary = results.summary();

        assert!(summary.contains("Centrality Analysis Summary"));
        assert!(summary.contains("Files with centrality scores: 3"));
        assert!(summary.contains("PageRank iterations: 10"));
        assert!(summary.contains("Rust"));
    }

    #[test]
    fn test_centrality_results_clone() {
        let results = create_mock_centrality_results();
        let cloned = results.clone();

        assert_eq!(results.pagerank_scores.len(), cloned.pagerank_scores.len());
        assert_eq!(
            results.import_stats.files_processed,
            cloned.import_stats.files_processed
        );
    }

    #[test]
    fn test_centrality_results_equality() {
        let results1 = create_mock_centrality_results();
        let results2 = results1.clone();

        assert_eq!(results1.pagerank_scores, results2.pagerank_scores);
        assert_eq!(results1.import_stats, results2.import_stats);
    }
}

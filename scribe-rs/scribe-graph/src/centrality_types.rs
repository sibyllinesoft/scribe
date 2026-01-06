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

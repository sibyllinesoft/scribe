//! Core type definitions for the Scribe library.
//!
//! Provides fundamental data structures for scoring, analysis, and
//! code representation used throughout the Scribe ecosystem.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

use crate::file::{FileInfo, Language};

/// Position in a source file (line and column)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    /// Line number (0-based)
    pub line: usize,
    /// Column number (0-based, UTF-8 bytes)
    pub column: usize,
}

impl Position {
    /// Create a new position
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }

    /// Create position at start of file
    pub fn zero() -> Self {
        Self::new(0, 0)
    }

    /// Convert to 1-based line/column for display
    pub fn to_display(&self) -> (usize, usize) {
        (self.line + 1, self.column + 1)
    }
}

/// Range in a source file (start and end positions)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Range {
    /// Start position (inclusive)
    pub start: Position,
    /// End position (exclusive)
    pub end: Position,
}

impl Range {
    /// Create a new range
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }

    /// Create a single-character range
    pub fn single(pos: Position) -> Self {
        Self::new(pos, Position::new(pos.line, pos.column + 1))
    }

    /// Check if this range contains a position
    pub fn contains(&self, pos: Position) -> bool {
        (pos.line > self.start.line || 
         (pos.line == self.start.line && pos.column >= self.start.column)) &&
        (pos.line < self.end.line || 
         (pos.line == self.end.line && pos.column < self.end.column))
    }

    /// Get the length in characters (approximation)
    pub fn length(&self) -> usize {
        if self.start.line == self.end.line {
            self.end.column.saturating_sub(self.start.column)
        } else {
            // Multi-line range - approximate
            (self.end.line - self.start.line) * 80 + self.end.column
        }
    }
}

/// Individual components of the heuristic scoring system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreComponents {
    /// Documentation score (presence of comments, docstrings)
    pub doc_score: f64,
    
    /// README/documentation file score
    pub readme_score: f64,
    
    /// Import/dependency score (how much this file is imported)
    pub import_score: f64,
    
    /// Path depth score (penalize deeply nested files)
    pub path_score: f64,
    
    /// Test linkage score (proximity to tests)
    pub test_link_score: f64,
    
    /// Code churn score (git activity)
    pub churn_score: f64,
    
    /// Final weighted score
    pub final_score: f64,
    
    // V2 advanced features (optional)
    
    /// PageRank centrality score
    pub centrality_score: f64,
    
    /// Entrypoint detection score
    pub entrypoint_score: f64,
    
    /// Examples/usage score
    pub examples_score: f64,
}

impl ScoreComponents {
    /// Create a new ScoreComponents with all scores at zero
    pub fn zero() -> Self {
        Self {
            doc_score: 0.0,
            readme_score: 0.0,
            import_score: 0.0,
            path_score: 0.0,
            test_link_score: 0.0,
            churn_score: 0.0,
            final_score: 0.0,
            centrality_score: 0.0,
            entrypoint_score: 0.0,
            examples_score: 0.0,
        }
    }

    /// Compute final score using provided weights
    pub fn compute_final_score(&mut self, weights: &HeuristicWeights) {
        self.final_score = self.doc_score * weights.doc
            + self.readme_score * weights.readme
            + self.import_score * weights.import_deg
            + self.path_score * weights.path
            + self.test_link_score * weights.test_link
            + self.churn_score * weights.churn
            + self.centrality_score * weights.centrality
            + self.entrypoint_score * weights.entrypoint
            + self.examples_score * weights.examples;
    }

    /// Get a breakdown of score contributions
    pub fn breakdown(&self, weights: &HeuristicWeights) -> Vec<(String, f64, f64)> {
        vec![
            ("doc".to_string(), self.doc_score, self.doc_score * weights.doc),
            ("readme".to_string(), self.readme_score, self.readme_score * weights.readme),
            ("import".to_string(), self.import_score, self.import_score * weights.import_deg),
            ("path".to_string(), self.path_score, self.path_score * weights.path),
            ("test_link".to_string(), self.test_link_score, self.test_link_score * weights.test_link),
            ("churn".to_string(), self.churn_score, self.churn_score * weights.churn),
            ("centrality".to_string(), self.centrality_score, self.centrality_score * weights.centrality),
            ("entrypoint".to_string(), self.entrypoint_score, self.entrypoint_score * weights.entrypoint),
            ("examples".to_string(), self.examples_score, self.examples_score * weights.examples),
        ]
    }
}

/// Configurable weights for different scoring components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeuristicWeights {
    /// Documentation weight
    pub doc: f64,
    
    /// README file weight
    pub readme: f64,
    
    /// Import degree weight
    pub import_deg: f64,
    
    /// Path depth weight
    pub path: f64,
    
    /// Test linkage weight
    pub test_link: f64,
    
    /// Code churn weight
    pub churn: f64,
    
    // V2 feature weights
    
    /// PageRank centrality weight
    pub centrality: f64,
    
    /// Entrypoint detection weight
    pub entrypoint: f64,
    
    /// Examples/usage weight
    pub examples: f64,
}

impl Default for HeuristicWeights {
    fn default() -> Self {
        Self {
            doc: 0.3,
            readme: 0.25,
            import_deg: 0.15,
            path: 0.1,
            test_link: 0.1,
            churn: 0.1,
            centrality: 0.0,
            entrypoint: 0.0,
            examples: 0.0,
        }
    }
}

impl HeuristicWeights {
    /// Create weights with V2 features enabled
    pub fn with_v2_features() -> Self {
        let mut weights = Self::default();
        
        // Enable V2 features
        weights.centrality = 0.15;
        weights.entrypoint = 0.10;
        weights.examples = 0.05;
        
        // Reduce other weights proportionally
        let reduction_factor = 0.7;
        weights.doc *= reduction_factor;
        weights.readme *= reduction_factor;
        weights.import_deg *= reduction_factor;
        weights.path *= reduction_factor;
        weights.test_link *= reduction_factor;
        weights.churn *= reduction_factor;
        
        weights.normalize()
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(mut self) -> Self {
        let total = self.doc + self.readme + self.import_deg + self.path + 
                   self.test_link + self.churn + self.centrality + 
                   self.entrypoint + self.examples;
        
        if total > 0.0 {
            self.doc /= total;
            self.readme /= total;
            self.import_deg /= total;
            self.path /= total;
            self.test_link /= total;
            self.churn /= total;
            self.centrality /= total;
            self.entrypoint /= total;
            self.examples /= total;
        }
        
        self
    }

    /// Check if V2 features are enabled
    pub fn has_v2_features(&self) -> bool {
        self.centrality > 0.0 || self.entrypoint > 0.0 || self.examples > 0.0
    }
}

/// Repository-level information and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryInfo {
    /// Repository root path
    pub root_path: PathBuf,
    
    /// Repository name (from directory name or git remote)
    pub name: String,
    
    /// Git remote URL (if available)
    pub remote_url: Option<String>,
    
    /// Current branch name
    pub branch: Option<String>,
    
    /// Last commit hash
    pub last_commit: Option<String>,
    
    /// Repository creation/initialization time
    pub created: Option<SystemTime>,
    
    /// Last modification time
    pub modified: Option<SystemTime>,
    
    /// Total number of files in repository
    pub total_files: usize,
    
    /// Number of files included in analysis
    pub analyzed_files: usize,
    
    /// File size statistics
    pub size_stats: SizeStatistics,
    
    /// Language breakdown
    pub languages: HashMap<Language, LanguageStats>,
    
    /// File type breakdown
    pub file_types: FileTypeStats,
    
    /// Git statistics (if available)
    pub git_stats: Option<GitStatistics>,
}

/// File size statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizeStatistics {
    /// Total size in bytes
    pub total_bytes: u64,
    
    /// Average file size
    pub avg_bytes: f64,
    
    /// Median file size
    pub median_bytes: u64,
    
    /// Largest file size
    pub max_bytes: u64,
    
    /// Smallest file size
    pub min_bytes: u64,
    
    /// Standard deviation of file sizes
    pub std_dev: f64,
}

impl SizeStatistics {
    /// Create statistics from a list of file sizes
    pub fn from_sizes(sizes: &[u64]) -> Self {
        if sizes.is_empty() {
            return Self::zero();
        }

        let total_bytes: u64 = sizes.iter().sum();
        let avg_bytes = total_bytes as f64 / sizes.len() as f64;
        
        let mut sorted_sizes = sizes.to_vec();
        sorted_sizes.sort_unstable();
        let median_bytes = sorted_sizes[sizes.len() / 2];
        let max_bytes = sorted_sizes[sizes.len() - 1];
        let min_bytes = sorted_sizes[0];
        
        // Calculate standard deviation
        let variance = sizes.iter()
            .map(|&size| {
                let diff = size as f64 - avg_bytes;
                diff * diff
            })
            .sum::<f64>() / sizes.len() as f64;
        let std_dev = variance.sqrt();

        Self {
            total_bytes,
            avg_bytes,
            median_bytes,
            max_bytes,
            min_bytes,
            std_dev,
        }
    }

    /// Create zero statistics
    pub fn zero() -> Self {
        Self {
            total_bytes: 0,
            avg_bytes: 0.0,
            median_bytes: 0,
            max_bytes: 0,
            min_bytes: 0,
            std_dev: 0.0,
        }
    }
}

/// Language-specific statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageStats {
    /// Number of files in this language
    pub file_count: usize,
    
    /// Total lines of code (if available)
    pub total_lines: Option<usize>,
    
    /// Total size in bytes
    pub total_bytes: u64,
    
    /// Percentage of repository (by file count)
    pub percentage: f64,
}

/// File type statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileTypeStats {
    /// Source code files
    pub source_files: usize,
    
    /// Test files
    pub test_files: usize,
    
    /// Documentation files
    pub doc_files: usize,
    
    /// Configuration files
    pub config_files: usize,
    
    /// Binary files (excluded)
    pub binary_files: usize,
    
    /// Generated files (excluded)
    pub generated_files: usize,
    
    /// Unknown file types
    pub unknown_files: usize,
}

impl FileTypeStats {
    /// Create empty statistics
    pub fn new() -> Self {
        Self {
            source_files: 0,
            test_files: 0,
            doc_files: 0,
            config_files: 0,
            binary_files: 0,
            generated_files: 0,
            unknown_files: 0,
        }
    }

    /// Get total file count
    pub fn total(&self) -> usize {
        self.source_files + self.test_files + self.doc_files + 
        self.config_files + self.binary_files + self.generated_files + 
        self.unknown_files
    }

    /// Get analyzed file count (excluding binary and generated)
    pub fn analyzed(&self) -> usize {
        self.source_files + self.test_files + self.doc_files + self.config_files
    }
}

impl Default for FileTypeStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Git repository statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GitStatistics {
    /// Total number of commits
    pub commit_count: usize,
    
    /// Number of contributors
    pub contributor_count: usize,
    
    /// Repository age in days
    pub age_days: Option<u64>,
    
    /// Files with most commits (churn analysis)
    pub high_churn_files: Vec<ChurnInfo>,
    
    /// Recent activity (commits in last 30 days)
    pub recent_commits: usize,
    
    /// Branch count
    pub branch_count: usize,
    
    /// Tag count
    pub tag_count: usize,
}

/// File churn information
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChurnInfo {
    /// File path
    pub path: String,
    
    /// Number of commits affecting this file
    pub commit_count: usize,
    
    /// Lines added (total)
    pub lines_added: usize,
    
    /// Lines deleted (total)
    pub lines_deleted: usize,
    
    /// Last modification date
    pub last_modified: Option<SystemTime>,
}

/// PageRank centrality computation results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentralityScores {
    /// PageRank scores for each file
    pub pagerank_scores: HashMap<String, f64>,
    
    /// Number of iterations until convergence
    pub iterations_converged: usize,
    
    /// Convergence epsilon achieved
    pub convergence_epsilon: f64,
    
    /// Graph statistics
    pub graph_stats: GraphStats,
}

/// Dependency graph statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of nodes (files)
    pub total_nodes: usize,
    
    /// Total number of edges (dependencies)
    pub total_edges: usize,
    
    /// Average in-degree (how many files depend on average file)
    pub in_degree_avg: f64,
    
    /// Maximum in-degree
    pub in_degree_max: usize,
    
    /// Average out-degree (how many files average file depends on)
    pub out_degree_avg: f64,
    
    /// Maximum out-degree
    pub out_degree_max: usize,
    
    /// Number of strongly connected components
    pub strongly_connected_components: usize,
    
    /// Graph density (edges / possible_edges)
    pub graph_density: f64,
}

impl GraphStats {
    /// Create empty graph statistics
    pub fn empty() -> Self {
        Self {
            total_nodes: 0,
            total_edges: 0,
            in_degree_avg: 0.0,
            in_degree_max: 0,
            out_degree_avg: 0.0,
            out_degree_max: 0,
            strongly_connected_components: 0,
            graph_density: 0.0,
        }
    }
}

/// Analysis result for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// File information
    pub file_info: FileInfo,
    
    /// Heuristic scores
    pub scores: ScoreComponents,
    
    /// Analysis duration in milliseconds
    pub analysis_duration_ms: u64,
    
    /// Any warnings or issues encountered
    pub warnings: Vec<String>,
    
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Metadata about the analysis process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Timestamp when analysis was performed
    pub timestamp: SystemTime,
    
    /// Version of Scribe that performed the analysis
    pub scribe_version: String,
    
    /// Features enabled during analysis
    pub features_enabled: Vec<String>,
    
    /// Configuration hash (for cache invalidation)
    pub config_hash: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position() {
        let pos = Position::new(5, 10);
        assert_eq!(pos.line, 5);
        assert_eq!(pos.column, 10);
        assert_eq!(pos.to_display(), (6, 11));

        let zero = Position::zero();
        assert_eq!(zero.line, 0);
        assert_eq!(zero.column, 0);
    }

    #[test]
    fn test_range() {
        let start = Position::new(1, 5);
        let end = Position::new(1, 10);
        let range = Range::new(start, end);

        assert!(range.contains(Position::new(1, 7)));
        assert!(!range.contains(Position::new(1, 10))); // end is exclusive
        assert!(!range.contains(Position::new(0, 7)));
        assert_eq!(range.length(), 5);

        let single = Range::single(Position::new(2, 3));
        assert_eq!(single.length(), 1);
    }

    #[test]
    fn test_score_components() {
        let mut scores = ScoreComponents::zero();
        scores.doc_score = 0.8;
        scores.import_score = 0.6;

        let weights = HeuristicWeights::default();
        scores.compute_final_score(&weights);

        assert!(scores.final_score > 0.0);
        assert!(scores.final_score < 1.0);

        let breakdown = scores.breakdown(&weights);
        assert_eq!(breakdown.len(), 9); // All score components
    }

    #[test]
    fn test_heuristic_weights() {
        let default_weights = HeuristicWeights::default();
        assert!(!default_weights.has_v2_features());

        let v2_weights = HeuristicWeights::with_v2_features();
        assert!(v2_weights.has_v2_features());
        assert!(v2_weights.centrality > 0.0);

        // Test normalization
        let total = v2_weights.doc + v2_weights.readme + v2_weights.import_deg + 
                   v2_weights.path + v2_weights.test_link + v2_weights.churn + 
                   v2_weights.centrality + v2_weights.entrypoint + v2_weights.examples;
        assert!((total - 1.0).abs() < 1e-10); // Should sum to 1.0
    }

    #[test]
    fn test_size_statistics() {
        let sizes = vec![100, 200, 150, 300, 250];
        let stats = SizeStatistics::from_sizes(&sizes);

        assert_eq!(stats.total_bytes, 1000);
        assert_eq!(stats.avg_bytes, 200.0);
        assert_eq!(stats.median_bytes, 200);
        assert_eq!(stats.max_bytes, 300);
        assert_eq!(stats.min_bytes, 100);
        assert!(stats.std_dev > 0.0);

        let empty_stats = SizeStatistics::from_sizes(&[]);
        assert_eq!(empty_stats.total_bytes, 0);
    }

    #[test]
    fn test_file_type_stats() {
        let mut stats = FileTypeStats::new();
        stats.source_files = 10;
        stats.test_files = 5;
        stats.binary_files = 3;

        assert_eq!(stats.total(), 18);
        assert_eq!(stats.analyzed(), 15); // Excludes binary
    }

    #[test]
    fn test_graph_stats() {
        let stats = GraphStats::empty();
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.graph_density, 0.0);
    }
}
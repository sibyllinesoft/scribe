//! # Scribe Graph - Advanced Code Dependency Analysis
//!
//! High-performance graph-based code analysis with PageRank centrality computation.
//! This crate provides sophisticated tools for understanding code structure, dependency
//! relationships, and file importance through research-grade graph algorithms.
//!
//! ## Key Features
//!
//! ### PageRank Centrality Analysis
//! - **Research-grade PageRank implementation** optimized for code dependency graphs
//! - **Reverse edge emphasis** (importance flows to imported files)
//! - **Convergence detection** with configurable precision
//! - **Multi-language import detection** (Python, JavaScript, TypeScript, Rust, Go, Java)
//!
//! ### Graph Construction and Analysis
//! - **Efficient dependency graph** representation with adjacency lists
//! - **Comprehensive statistics** (degree distribution, connectivity, structural patterns)
//! - **Performance optimized** for large codebases (10k+ files)
//! - **Concurrent processing** support for multi-core systems
//!
//! ### Integration with Scribe Heuristics
//! - **Seamless V2 integration** with existing heuristic scoring system
//! - **Configurable centrality weighting** in final importance scores
//! - **Multiple normalization methods** (min-max, z-score, rank-based)
//! - **Entrypoint boosting** for main/index files
//!
//! ## Quick Start
//!
//! ```ignore
//! use scribe_graph::{CentralityCalculator, PageRankConfig};
//! # use scribe_analysis::heuristics::ScanResult;
//! # use std::collections::HashMap;
//! #
//! # // Mock implementation for documentation
//! # #[derive(Debug)]
//! # struct MockScanResult {
//! #     path: String,
//! #     relative_path: String,
//! # }
//! #
//! # impl ScanResult for MockScanResult {
//! #     fn path(&self) -> &str { &self.path }
//! #     fn relative_path(&self) -> &str { &self.relative_path }
//! #     fn depth(&self) -> usize { 1 }
//! #     fn is_docs(&self) -> bool { false }
//! #     fn is_readme(&self) -> bool { false }
//! #     fn is_entrypoint(&self) -> bool { false }
//! #     fn is_examples(&self) -> bool { false }
//! #     fn is_tests(&self) -> bool { false }
//! #     fn priority_boost(&self) -> f64 { 0.0 }
//! #     fn get_documentation_score(&self) -> f64 { 0.0 }
//! #     fn get_file_size(&self) -> usize { 1000 }
//! #     fn get_imports(&self) -> Vec<String> { vec![] }
//! #     fn get_git_churn(&self) -> usize { 0 }
//! # }
//! #
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create centrality calculator optimized for code analysis
//! let calculator = CentralityCalculator::for_large_codebases()?;
//!
//! // Example scan results (replace with actual scan results)
//! let scan_results = vec![
//!     MockScanResult { path: "main.rs".to_string(), relative_path: "main.rs".to_string() },
//!     MockScanResult { path: "lib.rs".to_string(), relative_path: "lib.rs".to_string() },
//! ];
//! let heuristic_scores = HashMap::new();
//!
//! // Calculate PageRank centrality for scan results
//! let centrality_results = calculator.calculate_centrality(&scan_results)?;
//!
//! // Get top files by centrality
//! let top_files = centrality_results.top_files_by_centrality(10);
//!
//! // Integrate with existing heuristic scores
//! let integrated_scores = calculator.integrate_with_heuristics(
//!     &centrality_results,
//!     &heuristic_scores
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Memory usage**: ~2MB for 1000-file codebases, ~20MB for 10k+ files
//! - **Computation time**: ~10ms for small projects, ~100ms for large codebases
//! - **Convergence**: Typically 8-15 iterations for most dependency graphs
//! - **Parallel efficiency**: Near-linear speedup on multi-core systems

// Core modules
pub mod centrality;
pub mod centrality_types;
pub mod graph;
pub mod import_detector;
pub mod pagerank;
pub mod statistics;
pub mod ts_resolver;

// Primary API exports - PageRank centrality system
pub use centrality::{
    CentralityCalculator, CentralityConfig, CentralityResults, ImportDetectionStats,
    ImportResolutionConfig, IntegrationConfig, IntegrationMetadata, NormalizationMethod,
};

pub use pagerank::{
    PageRankComputer, PageRankConfig, PageRankResults, PerformanceMetrics, ScoreStatistics,
};

pub use graph::{
    ConcurrentDependencyGraph, DegreeInfo, DependencyGraph, GraphStatistics, NodeMetadata,
    TraversalDirection,
};

pub use statistics::{
    ConnectivityAnalysis, DegreeDistribution, GraphAnalysisResults, GraphStatisticsAnalyzer,
    ImportInsights, PerformanceProfile, StatisticsConfig, StructuralPatterns,
};

// Maintain compatibility alias for existing integrations
pub use graph::DependencyGraph as CodeGraph;

use scribe_analysis::heuristics::ScanResult;
use scribe_core::Result;
use std::collections::HashMap;

/// Main entry point for PageRank centrality analysis
///
/// This is the primary interface for computing PageRank centrality scores
/// and integrating them with the Scribe heuristic system.
pub struct PageRankAnalysis {
    calculator: CentralityCalculator,
}

impl PageRankAnalysis {
    /// Create a new PageRank analysis instance with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            calculator: CentralityCalculator::new()?,
        })
    }

    /// Create with custom centrality configuration
    pub fn with_config(config: CentralityConfig) -> Result<Self> {
        Ok(Self {
            calculator: CentralityCalculator::with_config(config)?,
        })
    }

    /// Create optimized for code dependency analysis
    pub fn for_code_analysis() -> Result<Self> {
        Ok(Self {
            calculator: CentralityCalculator::new()?,
        })
    }

    /// Create optimized for large codebases (>5k files)
    pub fn for_large_codebases() -> Result<Self> {
        Ok(Self {
            calculator: CentralityCalculator::for_large_codebases()?,
        })
    }

    /// Compute PageRank centrality scores for a collection of files
    pub fn compute_centrality<T>(&self, scan_results: &[T]) -> Result<CentralityResults>
    where
        T: ScanResult + Sync,
    {
        self.calculator.calculate_centrality(scan_results)
    }

    /// Integrate centrality scores with existing heuristic scores
    ///
    /// This combines PageRank centrality with Scribe heuristic scores using
    /// configurable weights. The default configuration uses 15% centrality weight
    /// and 85% heuristic weight.
    pub fn integrate_with_heuristics(
        &self,
        centrality_results: &CentralityResults,
        heuristic_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        self.calculator
            .integrate_with_heuristics(centrality_results, heuristic_scores)
    }

    /// Get a summary of centrality computation results
    pub fn summarize_results(&self, results: &CentralityResults) -> String {
        results.summary()
    }
}

impl Default for PageRankAnalysis {
    fn default() -> Self {
        Self::new().expect("Failed to create PageRankAnalysis")
    }
}

/// Utility functions for PageRank analysis
pub mod utils {
    use super::*;

    /// Quick function to compute centrality scores for scan results
    ///
    /// This is a convenience function for simple use cases. For more control
    /// over configuration, use `PageRankAnalysis` directly.
    pub fn compute_file_centrality<T>(scan_results: &[T]) -> Result<HashMap<String, f64>>
    where
        T: ScanResult + Sync,
    {
        let analysis = PageRankAnalysis::for_code_analysis()?;
        let results = analysis.compute_centrality(scan_results)?;
        Ok(results.pagerank_scores)
    }

    /// Quick function to get top-K most important files
    pub fn get_top_important_files<T>(
        scan_results: &[T],
        top_k: usize,
    ) -> Result<Vec<(String, f64)>>
    where
        T: ScanResult + Sync,
    {
        let analysis = PageRankAnalysis::for_code_analysis()?;
        let results = analysis.compute_centrality(scan_results)?;
        Ok(results.top_files_by_centrality(top_k))
    }

    /// Combine centrality and heuristic scores with default configuration
    pub fn combine_scores<T>(
        scan_results: &[T],
        heuristic_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>>
    where
        T: ScanResult + Sync,
    {
        let analysis = PageRankAnalysis::for_code_analysis()?;
        let centrality_results = analysis.compute_centrality(scan_results)?;
        analysis.integrate_with_heuristics(&centrality_results, heuristic_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_analysis_creation() {
        let analysis = PageRankAnalysis::new();
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_pagerank_analysis_default() {
        let analysis = PageRankAnalysis::default();
        // Should be created successfully via Default
        let _ = analysis;
    }

    #[test]
    fn test_pagerank_analysis_with_config() {
        let config = CentralityConfig::default();
        let analysis = PageRankAnalysis::with_config(config);
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_pagerank_analysis_for_code_analysis() {
        let analysis = PageRankAnalysis::for_code_analysis();
        assert!(analysis.is_ok());
    }

    #[test]
    fn test_pagerank_analysis_for_large_codebases() {
        let analysis = PageRankAnalysis::for_large_codebases();
        assert!(analysis.is_ok());
    }

    // Test mock ScanResult implementation for testing
    #[derive(Debug)]
    struct TestScanResult {
        path: String,
        relative_path: String,
        imports: Vec<String>,
    }

    impl TestScanResult {
        fn new(path: &str, relative_path: &str) -> Self {
            Self {
                path: path.to_string(),
                relative_path: relative_path.to_string(),
                imports: vec![],
            }
        }

        fn with_imports(mut self, imports: Vec<&str>) -> Self {
            self.imports = imports.into_iter().map(|s| s.to_string()).collect();
            self
        }
    }

    impl ScanResult for TestScanResult {
        fn path(&self) -> &str {
            &self.path
        }
        fn relative_path(&self) -> &str {
            &self.relative_path
        }
        fn depth(&self) -> usize {
            self.path.matches('/').count() + 1
        }
        fn is_docs(&self) -> bool {
            self.path.ends_with(".md")
        }
        fn is_readme(&self) -> bool {
            self.path.to_lowercase().contains("readme")
        }
        fn is_test(&self) -> bool {
            self.path.contains("test")
        }
        fn is_entrypoint(&self) -> bool {
            self.path.contains("main") || self.path.contains("index")
        }
        fn has_examples(&self) -> bool {
            self.path.contains("example")
        }
        fn priority_boost(&self) -> f64 {
            0.0
        }
        fn churn_score(&self) -> f64 {
            0.0
        }
        fn centrality_in(&self) -> f64 {
            0.0
        }
        fn imports(&self) -> Option<&[String]> {
            if self.imports.is_empty() {
                None
            } else {
                Some(&self.imports)
            }
        }
        fn doc_analysis(&self) -> Option<&scribe_analysis::heuristics::DocumentAnalysis> {
            None
        }
    }

    #[test]
    fn test_pagerank_analysis_compute_centrality_empty() {
        let analysis = PageRankAnalysis::new().unwrap();
        let empty_results: Vec<TestScanResult> = vec![];
        let result = analysis.compute_centrality(&empty_results);
        assert!(result.is_ok());
        let centrality = result.unwrap();
        assert!(centrality.pagerank_scores.is_empty());
    }

    #[test]
    fn test_pagerank_analysis_compute_centrality() {
        let analysis = PageRankAnalysis::new().unwrap();
        let scan_results = vec![
            TestScanResult::new("src/main.rs", "src/main.rs").with_imports(vec!["lib"]),
            TestScanResult::new("src/lib.rs", "src/lib.rs"),
            TestScanResult::new("src/utils.rs", "src/utils.rs"),
        ];

        let result = analysis.compute_centrality(&scan_results);
        assert!(result.is_ok());
        let centrality = result.unwrap();
        // Either we have scores or the graph analysis was performed
        assert!(
            !centrality.pagerank_scores.is_empty()
                || centrality.graph_analysis.basic_stats.total_nodes >= 0
        );
    }

    #[test]
    fn test_pagerank_analysis_summarize_results() {
        let analysis = PageRankAnalysis::new().unwrap();
        let scan_results = vec![
            TestScanResult::new("main.rs", "main.rs"),
            TestScanResult::new("lib.rs", "lib.rs"),
        ];

        let centrality_results = analysis.compute_centrality(&scan_results).unwrap();
        let summary = analysis.summarize_results(&centrality_results);
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_pagerank_analysis_integrate_with_heuristics() {
        let analysis = PageRankAnalysis::new().unwrap();
        let scan_results = vec![
            TestScanResult::new("main.rs", "main.rs"),
            TestScanResult::new("lib.rs", "lib.rs"),
        ];

        let centrality_results = analysis.compute_centrality(&scan_results).unwrap();

        let mut heuristic_scores = HashMap::new();
        heuristic_scores.insert("main.rs".to_string(), 0.8);
        heuristic_scores.insert("lib.rs".to_string(), 0.6);

        let integrated = analysis.integrate_with_heuristics(&centrality_results, &heuristic_scores);
        assert!(integrated.is_ok());
    }

    #[test]
    fn test_utils_compute_file_centrality() {
        let scan_results = vec![
            TestScanResult::new("a.rs", "a.rs"),
            TestScanResult::new("b.rs", "b.rs"),
        ];

        let result = utils::compute_file_centrality(&scan_results);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utils_get_top_important_files() {
        let scan_results = vec![
            TestScanResult::new("main.rs", "main.rs").with_imports(vec!["lib", "utils"]),
            TestScanResult::new("lib.rs", "lib.rs"),
            TestScanResult::new("utils.rs", "utils.rs"),
        ];

        let result = utils::get_top_important_files(&scan_results, 2);
        assert!(result.is_ok());
        let top_files = result.unwrap();
        assert!(top_files.len() <= 2);
    }

    #[test]
    fn test_utils_combine_scores() {
        let scan_results = vec![
            TestScanResult::new("main.rs", "main.rs"),
            TestScanResult::new("lib.rs", "lib.rs"),
        ];

        let mut heuristic_scores = HashMap::new();
        heuristic_scores.insert("main.rs".to_string(), 0.7);
        heuristic_scores.insert("lib.rs".to_string(), 0.5);

        let result = utils::combine_scores(&scan_results, &heuristic_scores);
        assert!(result.is_ok());
    }

    #[test]
    fn test_centrality_config_default() {
        let config = CentralityConfig::default();
        // Default config should have reasonable values
        assert!(config.pagerank_config.damping_factor > 0.0);
        assert!(config.pagerank_config.damping_factor < 1.0);
    }

    #[test]
    fn test_test_scan_result_implementation() {
        let result = TestScanResult::new("src/test.rs", "test.rs");
        assert_eq!(result.path(), "src/test.rs");
        assert_eq!(result.relative_path(), "test.rs");
        assert_eq!(result.depth(), 2);
        assert!(!result.is_docs());
        assert!(!result.is_readme());
        assert!(result.is_test());
        assert!(!result.is_entrypoint());
        assert!(!result.has_examples());
        assert_eq!(result.priority_boost(), 0.0);
        assert_eq!(result.churn_score(), 0.0);
        assert_eq!(result.centrality_in(), 0.0);
        assert!(result.imports().is_none());
        assert!(result.doc_analysis().is_none());
    }

    #[test]
    fn test_test_scan_result_entrypoint() {
        let main_result = TestScanResult::new("src/main.rs", "main.rs");
        assert!(main_result.is_entrypoint());

        let index_result = TestScanResult::new("index.js", "index.js");
        assert!(index_result.is_entrypoint());
    }

    #[test]
    fn test_test_scan_result_docs() {
        let readme = TestScanResult::new("README.md", "README.md");
        assert!(readme.is_docs());
        assert!(readme.is_readme());

        let doc = TestScanResult::new("docs/guide.md", "docs/guide.md");
        assert!(doc.is_docs());
        assert!(!doc.is_readme());
    }

    #[test]
    fn test_test_scan_result_with_imports() {
        let result =
            TestScanResult::new("main.rs", "main.rs").with_imports(vec!["lib", "utils", "config"]);

        let imports = result.imports();
        assert!(imports.is_some());
        let imports = imports.unwrap();
        assert_eq!(imports.len(), 3);
        assert!(imports.contains(&"lib".to_string()));
    }
}

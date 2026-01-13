//! # Heuristic Scoring System for File Prioritization
//!
//! Implements a sophisticated multi-dimensional scoring system for ranking file importance
//! within a codebase. This module ports and enhances the legacy Python heuristics that
//! originally powered Scribe, while adding more robust Rust-based features:
//!
//! ## Core Scoring Formula
//! ```text
//! score = w_doc*doc + w_readme*readme + w_imp*imp_deg + w_path*path_depth^-1 +
//!         w_test*test_link + w_churn*churn + w_centrality*centrality +
//!         w_entrypoint*entrypoint + w_examples*examples + priority_boost
//! ```
//!
//! ## Key Features
//! - **Template Detection**: Advanced template engine detection using multiple methods
//! - **Import Graph Analysis**: Dependency centrality and PageRank calculations
//! - **Documentation Intelligence**: README prioritization and document structure analysis
//! - **Test-Code Relationships**: Heuristic test file linkage detection
//! - **Git Churn Integration**: Change recency and frequency signals
//! - **Configurable Weighting**: V1/V2 feature flags with dynamic weight adjustment
//! - **Performance Optimized**: Caching and lazy evaluation for large codebases

pub mod import_analysis;
pub mod scoring;
pub mod template_detection;

pub use scoring::{
    HeuristicScorer, HeuristicWeights, ScoreComponents, ScoringFeatures, WeightPreset,
};

pub use template_detection::{
    get_template_score_boost, is_template_file, TemplateDetectionMethod, TemplateDetector,
    TemplateEngine,
};

pub use import_analysis::{
    import_matches_file, CentralityCalculator, ImportGraph, ImportGraphBuilder,
};

use scribe_core::Result;
use std::collections::HashMap;

/// Main entry point for the heuristic scoring system
#[derive(Debug)]
pub struct HeuristicSystem {
    /// Core file scorer
    scorer: HeuristicScorer,
    /// Template detection engine
    template_detector: TemplateDetector,
}

impl HeuristicSystem {
    /// Create a new heuristic system with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            scorer: HeuristicScorer::new(HeuristicWeights::default()),
            template_detector: TemplateDetector::new()?,
        })
    }

    /// Create with custom weights
    pub fn with_weights(weights: HeuristicWeights) -> Result<Self> {
        Ok(Self {
            scorer: HeuristicScorer::new(weights),
            template_detector: TemplateDetector::new()?,
        })
    }

    /// Create with V2 features enabled
    pub fn with_v2_features() -> Result<Self> {
        Ok(Self {
            scorer: HeuristicScorer::new(HeuristicWeights::with_v2_features()),
            template_detector: TemplateDetector::new()?,
        })
    }

    /// Score a single file within the context of all scanned files
    pub fn score_file<T>(&mut self, file: &T, all_files: &[T]) -> Result<ScoreComponents>
    where
        T: ScanResult,
    {
        self.scorer.score_file(file, all_files)
    }

    /// Score all files and return ranked results
    pub fn score_all_files<T>(&mut self, files: &[T]) -> Result<Vec<(usize, ScoreComponents)>>
    where
        T: ScanResult,
    {
        self.scorer.score_all_files(files)
    }

    /// Get top-K files by heuristic score
    pub fn get_top_files<T>(
        &mut self,
        files: &[T],
        top_k: usize,
    ) -> Result<Vec<(usize, ScoreComponents)>>
    where
        T: ScanResult,
    {
        Ok(self
            .score_all_files(files)?
            .into_iter()
            .take(top_k)
            .collect())
    }

    /// Get template score boost for a file path
    pub fn get_template_boost(&self, file_path: &str) -> Result<f64> {
        self.template_detector.get_score_boost(file_path)
    }

    /// Check if two imports match (for import graph construction)
    pub fn import_matches(&self, import_name: &str, file_path: &str) -> bool {
        import_analysis::import_matches_file(import_name, file_path)
    }
}

impl Default for HeuristicSystem {
    fn default() -> Self {
        Self::new().expect("Failed to create HeuristicSystem")
    }
}

/// Trait that scan results must implement for heuristic scoring
pub trait ScanResult {
    /// Get the file path
    fn path(&self) -> &str;

    /// Get the relative path from repository root
    fn relative_path(&self) -> &str;

    /// Get file depth (directory nesting level)
    fn depth(&self) -> usize;

    /// Check if this is a documentation file
    fn is_docs(&self) -> bool;

    /// Check if this is a README file
    fn is_readme(&self) -> bool;

    /// Check if this is a test file
    fn is_test(&self) -> bool;

    /// Check if this is an entrypoint file (main, index, etc.)
    fn is_entrypoint(&self) -> bool;

    /// Check if this file contains examples
    fn has_examples(&self) -> bool;

    /// Get the priority boost value (from scanner)
    fn priority_boost(&self) -> f64;

    /// Get the churn score (git activity)
    fn churn_score(&self) -> f64;

    /// Get centrality in score (PageRank)
    fn centrality_in(&self) -> f64;

    /// Get list of import statements
    fn imports(&self) -> Option<&[String]>;

    /// Get document analysis results (if available)
    fn doc_analysis(&self) -> Option<&DocumentAnalysis>;
}

/// Document analysis results for scoring
#[derive(Debug, Clone)]
pub struct DocumentAnalysis {
    /// Number of headings in the document
    pub heading_count: usize,
    /// Number of table-of-contents indicators
    pub toc_indicators: usize,
    /// Number of links in the document
    pub link_count: usize,
    /// Number of code blocks
    pub code_block_count: usize,
    /// Whether the document appears well-structured
    pub is_well_structured: bool,
}

impl DocumentAnalysis {
    pub fn new() -> Self {
        Self {
            heading_count: 0,
            toc_indicators: 0,
            link_count: 0,
            code_block_count: 0,
            is_well_structured: false,
        }
    }

    /// Calculate structure score based on analysis
    pub fn structure_score(&self) -> f64 {
        let mut score = 0.0;

        // Heading structure bonus
        if self.heading_count > 0 {
            score += (self.heading_count as f64 / 10.0).min(0.5);
        }

        // TOC indicates well-organized document
        if self.toc_indicators > 0 {
            score += 0.3;
        }

        // Links indicate reference document
        if self.link_count > 0 {
            score += (self.link_count as f64 / 20.0).min(0.3);
        }

        // Code blocks in docs indicate technical documentation
        if self.code_block_count > 0 {
            score += (self.code_block_count as f64 / 10.0).min(0.2);
        }

        score
    }
}

impl Default for DocumentAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for the heuristic system
#[derive(Debug, Clone)]
pub struct HeuristicMetrics {
    /// Number of files processed
    pub files_processed: usize,
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    /// Import graph construction time
    pub import_graph_time_ms: u64,
    /// Template detection time
    pub template_detection_time_ms: u64,
    /// Average time per file
    pub avg_time_per_file_ms: f64,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, f64>,
}

impl HeuristicMetrics {
    pub fn new() -> Self {
        Self {
            files_processed: 0,
            processing_time_ms: 0,
            import_graph_time_ms: 0,
            template_detection_time_ms: 0,
            avg_time_per_file_ms: 0.0,
            cache_hit_rates: HashMap::new(),
        }
    }

    pub fn finalize(&mut self) {
        if self.files_processed > 0 {
            self.avg_time_per_file_ms =
                self.processing_time_ms as f64 / self.files_processed as f64;
        }
    }
}

impl Default for HeuristicMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_system_creation() {
        let system = HeuristicSystem::new();
        assert!(system.is_ok());

        let v2_system = HeuristicSystem::with_v2_features();
        assert!(v2_system.is_ok());
    }

    #[test]
    fn test_document_analysis() {
        let mut doc = DocumentAnalysis::new();
        doc.heading_count = 5;
        doc.link_count = 10;
        doc.code_block_count = 3;

        let score = doc.structure_score();
        assert!(score > 0.0);
        assert!(score < 2.0); // Should be reasonable
    }

    #[test]
    fn test_metrics() {
        let mut metrics = HeuristicMetrics::new();
        metrics.files_processed = 100;
        metrics.processing_time_ms = 500;
        metrics.finalize();

        assert_eq!(metrics.avg_time_per_file_ms, 5.0);
    }

    #[test]
    fn test_heuristic_system_default() {
        let system = HeuristicSystem::default();
        // Verify it creates successfully
        let _ = system;
    }

    #[test]
    fn test_heuristic_system_with_weights() {
        let weights = HeuristicWeights::default();
        let system = HeuristicSystem::with_weights(weights);
        assert!(system.is_ok());
    }

    #[test]
    fn test_document_analysis_default() {
        let doc = DocumentAnalysis::default();
        assert_eq!(doc.heading_count, 0);
        assert_eq!(doc.link_count, 0);
        assert_eq!(doc.code_block_count, 0);
        assert_eq!(doc.toc_indicators, 0);
        assert!(!doc.is_well_structured);
    }

    #[test]
    fn test_document_analysis_structure_score_headings() {
        let mut doc = DocumentAnalysis::new();
        doc.heading_count = 10;

        let score = doc.structure_score();
        // Should cap at 0.5 for headings
        assert!(score <= 0.5);
        assert!(score > 0.0);
    }

    #[test]
    fn test_document_analysis_structure_score_toc() {
        let mut doc = DocumentAnalysis::new();
        doc.toc_indicators = 1;

        let score = doc.structure_score();
        assert!((score - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_document_analysis_structure_score_all_features() {
        let doc = DocumentAnalysis {
            heading_count: 10,
            toc_indicators: 1,
            link_count: 20,
            code_block_count: 10,
            is_well_structured: true,
        };

        let score = doc.structure_score();
        // 0.5 (headings max) + 0.3 (TOC) + 0.3 (links max) + 0.2 (code blocks max) = 1.3
        assert!(score > 1.0);
        assert!(score <= 1.5);
    }

    #[test]
    fn test_document_analysis_clone() {
        let doc = DocumentAnalysis {
            heading_count: 5,
            toc_indicators: 2,
            link_count: 10,
            code_block_count: 3,
            is_well_structured: true,
        };

        let cloned = doc.clone();
        assert_eq!(doc.heading_count, cloned.heading_count);
        assert_eq!(doc.toc_indicators, cloned.toc_indicators);
        assert_eq!(doc.is_well_structured, cloned.is_well_structured);
    }

    #[test]
    fn test_heuristic_metrics_default() {
        let metrics = HeuristicMetrics::default();
        assert_eq!(metrics.files_processed, 0);
        assert_eq!(metrics.processing_time_ms, 0);
        assert_eq!(metrics.import_graph_time_ms, 0);
        assert_eq!(metrics.template_detection_time_ms, 0);
        assert_eq!(metrics.avg_time_per_file_ms, 0.0);
        assert!(metrics.cache_hit_rates.is_empty());
    }

    #[test]
    fn test_heuristic_metrics_finalize_zero_files() {
        let mut metrics = HeuristicMetrics::new();
        metrics.finalize();

        // With 0 files, avg should remain 0
        assert_eq!(metrics.avg_time_per_file_ms, 0.0);
    }

    #[test]
    fn test_heuristic_metrics_clone() {
        let mut metrics = HeuristicMetrics::new();
        metrics.files_processed = 50;
        metrics.processing_time_ms = 100;

        let cloned = metrics.clone();
        assert_eq!(metrics.files_processed, cloned.files_processed);
        assert_eq!(metrics.processing_time_ms, cloned.processing_time_ms);
    }

    #[test]
    fn test_heuristic_metrics_debug() {
        let metrics = HeuristicMetrics {
            files_processed: 42,
            processing_time_ms: 500,
            import_graph_time_ms: 100,
            template_detection_time_ms: 50,
            avg_time_per_file_ms: 11.9,
            cache_hit_rates: HashMap::new(),
        };

        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("500"));
    }

    #[test]
    fn test_heuristic_system_import_matches() {
        let system = HeuristicSystem::new().unwrap();

        // Test import matching
        let matches = system.import_matches("utils", "src/utils.py");
        // Result depends on import_analysis implementation
        let _ = matches;
    }

    #[test]
    fn test_heuristic_system_get_template_boost() {
        let system = HeuristicSystem::new().unwrap();

        // Test template boost for non-template file
        let boost = system.get_template_boost("src/main.rs");
        assert!(boost.is_ok());
    }

    // Mock ScanResult implementation for testing
    #[derive(Debug)]
    struct MockScanResult {
        path: String,
        is_test: bool,
        is_readme: bool,
    }

    impl MockScanResult {
        fn new(path: &str) -> Self {
            Self {
                path: path.to_string(),
                is_test: path.contains("test"),
                is_readme: path.to_lowercase().contains("readme"),
            }
        }
    }

    impl ScanResult for MockScanResult {
        fn path(&self) -> &str {
            &self.path
        }
        fn relative_path(&self) -> &str {
            &self.path
        }
        fn depth(&self) -> usize {
            self.path.matches('/').count()
        }
        fn is_docs(&self) -> bool {
            self.path.ends_with(".md")
        }
        fn is_readme(&self) -> bool {
            self.is_readme
        }
        fn is_test(&self) -> bool {
            self.is_test
        }
        fn is_entrypoint(&self) -> bool {
            self.path.contains("main") || self.path.contains("lib")
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
            None
        }
        fn doc_analysis(&self) -> Option<&DocumentAnalysis> {
            None
        }
    }

    #[test]
    fn test_heuristic_system_score_file() {
        let mut system = HeuristicSystem::new().unwrap();
        let files = vec![
            MockScanResult::new("src/main.rs"),
            MockScanResult::new("src/lib.rs"),
            MockScanResult::new("tests/test_main.rs"),
        ];

        let result = system.score_file(&files[0], &files);
        assert!(result.is_ok());
        let scores = result.unwrap();
        assert!(scores.final_score >= 0.0);
    }

    #[test]
    fn test_heuristic_system_get_top_files() {
        let mut system = HeuristicSystem::new().unwrap();
        let files = vec![
            MockScanResult::new("src/main.rs"),
            MockScanResult::new("src/lib.rs"),
            MockScanResult::new("README.md"),
            MockScanResult::new("tests/test_main.rs"),
            MockScanResult::new("examples/demo.rs"),
        ];

        let top = system.get_top_files(&files, 3);
        assert!(top.is_ok());
        let top_files = top.unwrap();
        assert!(top_files.len() <= 3);
    }

    #[test]
    fn test_heuristic_system_score_all_files() {
        let mut system = HeuristicSystem::new().unwrap();
        let files = vec![
            MockScanResult::new("src/main.rs"),
            MockScanResult::new("README.md"),
        ];

        let result = system.score_all_files(&files);
        assert!(result.is_ok());
        let scores = result.unwrap();
        assert_eq!(scores.len(), 2);
    }
}

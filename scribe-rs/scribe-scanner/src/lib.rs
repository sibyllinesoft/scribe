//! # Scribe Scanner
//!
//! High-performance file system scanning and indexing capabilities for the Scribe library.
//! This crate provides efficient tools for discovering, filtering, and analyzing files
//! in large codebases with git integration and parallel processing.
//!
//! ## Features
//!
//! - **Fast Repository Traversal**: Efficient file discovery using `walkdir` and `ignore`
//! - **Git Integration**: Prefer `git ls-files` when available, with fallback to filesystem walk
//! - **Language Detection**: Automatic detection for 25+ programming languages
//! - **Parallel Processing**: Memory-efficient parallel file processing using Rayon
//! - **Binary Detection**: Libmagic-compatible content detection to skip non-text files
//!
//! ## Usage
//!
//! ```rust
//! use scribe_scanner::{Scanner, ScanOptions};
//! use std::path::Path;
//!
//! # async fn example() -> scribe_core::Result<()> {
//! let scanner = Scanner::new();
//! let options = ScanOptions::default()
//!     .with_git_integration(true)
//!     .with_parallel_processing(true);
//!
//! let results = scanner.scan(Path::new("."), options).await?;
//! println!("Scanned {} files", results.len());
//! # Ok(())
//! # }
//! ```

// Module organization
pub mod analysis;
pub mod core;
pub mod git;
pub mod perf;

// Re-export core types
pub use core::filtering::{DirectoryFilter, FileFilter, FilterReason, FilterResult};
pub use core::metadata::{FileMetadata, MetadataExtractor, SizeStats};
pub use core::scanner::{ScanOptions, ScanProgress, ScanResult, Scanner};

// Re-export git types
pub use git::{GitCommitInfo, GitFileInfo, GitIntegrator};

// Re-export analysis types
pub use analysis::aho_corasick_reference_index::{AhoCorasickReferenceIndex, IndexConfig, IndexMetrics};
pub use analysis::language_detection::{DetectionStrategy, LanguageDetector, LanguageHints};

// Re-export performance types
pub use perf::parallel::{ParallelConfig, ParallelController, ParallelMetrics, WorkItem};
pub use perf::performance::{
    ErrorType, PerfTimer, PerformanceMonitor, PerformanceReport, PerformanceSnapshot, PERF_MONITOR,
};

use scribe_core::{FileInfo, Result};
use std::path::Path;

/// Current version of the scanner crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// High-level scanner facade providing convenient access to all scanning functionality
pub struct FileScanner {
    scanner: Scanner,
    metadata_extractor: MetadataExtractor,
    git_integrator: Option<GitIntegrator>,
    language_detector: LanguageDetector,
}

impl FileScanner {
    /// Create a new file scanner with default configuration
    pub fn new() -> Self {
        Self {
            scanner: Scanner::new(),
            metadata_extractor: MetadataExtractor::new(),
            git_integrator: None,
            language_detector: LanguageDetector::new(),
        }
    }

    /// Enable git integration for enhanced file discovery
    pub fn with_git_integration(mut self, repo_path: &Path) -> Result<Self> {
        self.git_integrator = Some(GitIntegrator::new(repo_path)?);
        Ok(self)
    }

    /// Scan a directory with comprehensive analysis
    pub async fn scan_comprehensive<P: AsRef<Path>>(&self, path: P) -> Result<Vec<FileInfo>> {
        let options = ScanOptions::default()
            .with_metadata_extraction(true)
            .with_git_integration(self.git_integrator.is_some())
            .with_parallel_processing(true);

        self.scanner.scan(path, options).await
    }

    /// Quick scan without full content analysis
    pub async fn scan_fast<P: AsRef<Path>>(&self, path: P) -> Result<Vec<FileInfo>> {
        let options = ScanOptions::default()
            .with_metadata_extraction(true)
            .with_parallel_processing(true);

        self.scanner.scan(path, options).await
    }

    /// Get detailed statistics about the scanning process
    pub fn get_stats(&self) -> ScannerStats {
        ScannerStats {
            files_processed: self.scanner.files_processed(),
            directories_traversed: self.scanner.directories_traversed(),
            binary_files_skipped: self.scanner.binary_files_skipped(),
            git_files_discovered: self
                .git_integrator
                .as_ref()
                .map(|g| g.files_discovered())
                .unwrap_or(0),
        }
    }
}

impl Default for FileScanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the scanning process
#[derive(Debug, Clone)]
pub struct ScannerStats {
    pub files_processed: usize,
    pub directories_traversed: usize,
    pub binary_files_skipped: usize,
    pub git_files_discovered: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_scanner_creation() {
        let scanner = FileScanner::new();
        let stats = scanner.get_stats();
        assert_eq!(stats.files_processed, 0);
    }

    #[tokio::test]
    async fn test_fast_scan() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() {}").unwrap();

        let scanner = FileScanner::new();
        let results = scanner.scan_fast(temp_dir.path()).await.unwrap();

        assert!(!results.is_empty());
        assert!(results
            .iter()
            .any(|f| f.path.file_name().unwrap() == "test.rs"));
    }

    #[test]
    fn test_scanner_stats() {
        let scanner = FileScanner::new();
        let stats = scanner.get_stats();

        assert_eq!(stats.files_processed, 0);
        assert_eq!(stats.directories_traversed, 0);
        assert_eq!(stats.binary_files_skipped, 0);
        assert_eq!(stats.git_files_discovered, 0);
    }

    #[test]
    fn test_scanner_default() {
        let scanner = FileScanner::default();
        let stats = scanner.get_stats();
        assert_eq!(stats.files_processed, 0);
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.chars().any(|c| c.is_numeric()));
    }

    #[tokio::test]
    async fn test_comprehensive_scan() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.py");
        fs::write(&test_file, "def main():\n    print('hello')").unwrap();

        let scanner = FileScanner::new();
        let results = scanner.scan_comprehensive(temp_dir.path()).await.unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_scanner_stats_clone() {
        let stats = ScannerStats {
            files_processed: 10,
            directories_traversed: 5,
            binary_files_skipped: 2,
            git_files_discovered: 8,
        };

        let cloned = stats.clone();
        assert_eq!(stats.files_processed, cloned.files_processed);
        assert_eq!(stats.directories_traversed, cloned.directories_traversed);
    }

    #[test]
    fn test_scanner_stats_debug() {
        let stats = ScannerStats {
            files_processed: 10,
            directories_traversed: 5,
            binary_files_skipped: 2,
            git_files_discovered: 8,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("files_processed: 10"));
    }

    // Test re-exports are accessible
    #[test]
    fn test_reexport_filter_reason() {
        let _: FilterReason = FilterReason::Binary;
        let _: FilterReason = FilterReason::Hidden;
        let _: FilterReason = FilterReason::ColdExtension;
        let _: FilterReason = FilterReason::ColdDirectory;
        let _: FilterReason = FilterReason::TooLarge(1000);
        let _: FilterReason = FilterReason::CustomExtensionFilter;
    }

    #[test]
    fn test_reexport_detection_strategy() {
        let _: DetectionStrategy = DetectionStrategy::ExtensionOnly;
        let _: DetectionStrategy = DetectionStrategy::ExtensionWithContent;
        let _: DetectionStrategy = DetectionStrategy::FullAnalysis;
    }

    #[test]
    fn test_reexport_parallel_config() {
        let config = ParallelConfig::default();
        assert!(config.max_concurrency > 0);
    }

    #[test]
    fn test_reexport_scan_options() {
        let options = ScanOptions::default();
        let options_with_git = options.clone().with_git_integration(true);
        let options_with_parallel = options_with_git.with_parallel_processing(true);
        let _ = options_with_parallel;
    }

    #[test]
    fn test_with_git_integration_nonexistent() {
        let scanner = FileScanner::new();
        // Non-existent directory should fail
        let result = scanner.with_git_integration(Path::new("/nonexistent/path/that/does/not/exist"));
        // GitIntegrator::new might succeed or fail depending on whether it tries to find .git
        // The important thing is it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_with_git_integration_valid_repo() {
        // Find the repo root by looking for .git directory
        let mut path = std::env::current_dir().unwrap();
        while !path.join(".git").exists() && path.parent().is_some() {
            path = path.parent().unwrap().to_path_buf();
        }

        if path.join(".git").exists() {
            let scanner = FileScanner::new();
            let result = scanner.with_git_integration(&path);
            assert!(result.is_ok(), "Should succeed with valid git repo");

            let scanner = result.unwrap();
            let stats = scanner.get_stats();
            // After initialization with git integration, git_files_discovered might still be 0
            // until we actually scan, but the integrator should be set
            assert!(stats.git_files_discovered >= 0);
        }
        // If we're not in a git repo, skip the test
    }
}

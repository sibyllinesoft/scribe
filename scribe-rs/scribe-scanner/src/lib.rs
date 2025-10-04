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

// Core modules
pub mod git_integration;
pub mod language_detection;
pub mod metadata;
pub mod scanner;

// Performance optimization modules
pub mod aho_corasick_reference_index;
pub mod filtering;
pub mod parallel;
pub mod performance;

// Re-export main types for convenience
pub use git_integration::{GitCommitInfo, GitFileInfo, GitIntegrator};
pub use language_detection::{DetectionStrategy, LanguageDetector, LanguageHints};
pub use metadata::{FileMetadata, MetadataExtractor, SizeStats};
pub use scanner::{ScanOptions, ScanProgress, ScanResult, Scanner};

// Re-export performance optimization types
pub use aho_corasick_reference_index::{AhoCorasickReferenceIndex, IndexConfig, IndexMetrics};
pub use filtering::{DirectoryFilter, FileFilter, FilterReason, FilterResult};
pub use parallel::{ParallelConfig, ParallelController, ParallelMetrics, WorkItem};
pub use performance::{
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
}

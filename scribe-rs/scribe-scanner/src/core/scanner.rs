//! Core scanning functionality for efficient file system traversal.
//!
//! This module provides the main Scanner implementation with support for
//! parallel processing, git integration, and advanced filtering.

use crate::{GitIntegrator, LanguageDetector, MetadataExtractor};
use scribe_core::{
    FileInfo, FileWeight, GitFileStatus, GitStatus, Language, RenderDecision, Result, ScribeError,
};

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures::stream::{self, StreamExt};
use ignore::{DirEntry as IgnoreDirEntry, WalkBuilder, WalkState};
use rayon::prelude::*;
use tokio::sync::{RwLock, Semaphore};
use walkdir::{DirEntry, WalkDir};

/// High-performance file system scanner with parallel processing
#[derive(Debug)]
pub struct Scanner {
    stats: Arc<ScannerStats>,
    semaphore: Arc<Semaphore>,
}

/// Internal statistics tracking for the scanner
#[derive(Debug, Default)]
pub struct ScannerStats {
    files_processed: AtomicUsize,
    directories_traversed: AtomicUsize,
    binary_files_skipped: AtomicUsize,
    errors_encountered: AtomicUsize,
}

/// Configuration options for scanning operations
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// Enable parallel processing using Rayon
    pub parallel_processing: bool,
    /// Maximum number of concurrent file operations
    pub max_concurrency: usize,
    /// Extract detailed file metadata
    pub metadata_extraction: bool,
    /// Use git integration when available
    pub git_integration: bool,
    /// Follow symbolic links
    pub follow_symlinks: bool,
    /// Include hidden files and directories
    pub include_hidden: bool,
    /// Maximum file size to process (bytes)
    pub max_file_size: Option<u64>,
    /// Custom file extensions to include
    pub include_extensions: Option<Vec<String>>,
    /// Custom file extensions to exclude
    pub exclude_extensions: Option<Vec<String>>,
}

/// Result of a scanning operation
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub files: Vec<FileInfo>,
    pub stats: ScanProgress,
    pub duration: std::time::Duration,
    pub errors: Vec<String>,
}

/// Progress information during scanning
#[derive(Debug, Clone)]
pub struct ScanProgress {
    pub files_processed: usize,
    pub directories_traversed: usize,
    pub binary_files_skipped: usize,
    pub errors_encountered: usize,
    pub bytes_processed: u64,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            parallel_processing: true,
            max_concurrency: num_cpus::get().min(16), // Cap at 16 for memory efficiency
            metadata_extraction: true,
            git_integration: false,
            follow_symlinks: false,
            include_hidden: false,
            max_file_size: Some(50 * 1024 * 1024), // 50MB
            include_extensions: None,
            exclude_extensions: None,
        }
    }
}

impl ScanOptions {
    /// Enable parallel processing
    pub fn with_parallel_processing(mut self, enabled: bool) -> Self {
        self.parallel_processing = enabled;
        self
    }

    /// Set maximum concurrency level
    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = max;
        self
    }

    /// Enable metadata extraction
    pub fn with_metadata_extraction(mut self, enabled: bool) -> Self {
        self.metadata_extraction = enabled;
        self
    }

    /// Enable git integration
    pub fn with_git_integration(mut self, enabled: bool) -> Self {
        self.git_integration = enabled;
        self
    }

    /// Follow symbolic links
    pub fn with_follow_symlinks(mut self, enabled: bool) -> Self {
        self.follow_symlinks = enabled;
        self
    }

    /// Include hidden files
    pub fn with_include_hidden(mut self, enabled: bool) -> Self {
        self.include_hidden = enabled;
        self
    }

    /// Set maximum file size limit
    pub fn with_max_file_size(mut self, size: Option<u64>) -> Self {
        self.max_file_size = size;
        self
    }

    /// Set extensions to include
    pub fn with_include_extensions(mut self, extensions: Vec<String>) -> Self {
        self.include_extensions = Some(extensions);
        self
    }

    /// Set extensions to exclude
    pub fn with_exclude_extensions(mut self, extensions: Vec<String>) -> Self {
        self.exclude_extensions = Some(extensions);
        self
    }
}

impl Scanner {
    /// Create a new scanner with default configuration
    pub fn new() -> Self {
        Self {
            stats: Arc::new(ScannerStats::default()),
            semaphore: Arc::new(Semaphore::new(16)), // Default concurrency limit
        }
    }

    /// Scan a directory with the given options
    pub async fn scan<P: AsRef<Path>>(
        &self,
        path: P,
        options: ScanOptions,
    ) -> Result<Vec<FileInfo>> {
        let start_time = Instant::now();
        let path = path.as_ref();

        // Validate input path
        if !path.exists() {
            return Err(ScribeError::path(
                format!("Path does not exist: {}", path.display()),
                path,
            ));
        }

        if !path.is_dir() {
            return Err(ScribeError::path(
                format!("Path is not a directory: {}", path.display()),
                path,
            ));
        }

        // Initialize components
        let metadata_extractor = if options.metadata_extraction {
            Some(MetadataExtractor::new())
        } else {
            None
        };

        let git_integrator = if options.git_integration {
            GitIntegrator::new(path).ok()
        } else {
            None
        };

        let language_detector = LanguageDetector::new();

        // Try git-based discovery first if enabled
        let file_paths = if let Some(ref git) = git_integrator {
            match git.list_tracked_files().await {
                Ok(paths) => {
                    log::debug!(
                        "Using git ls-files for file discovery: {} files",
                        paths.len()
                    );
                    paths
                }
                Err(_) => {
                    log::debug!("Git discovery failed, falling back to filesystem walk");
                    self.discover_files_filesystem(path, &options).await?
                }
            }
        } else {
            self.discover_files_filesystem(path, &options).await?
        };

        log::info!("Discovered {} files for processing", file_paths.len());

        // Load batch git status for performance if git integration is enabled
        if let Some(ref git) = git_integrator {
            if let Err(e) = git.load_batch_file_statuses().await {
                log::debug!("Failed to load batch git statuses: {}", e);
            }
        }

        // Process files with appropriate strategy
        let files = if options.parallel_processing {
            log::debug!(
                "Processing files in parallel with concurrency={}",
                options.max_concurrency
            );
            self.process_files_parallel(
                file_paths,
                &options,
                metadata_extractor.as_ref(),
                git_integrator.as_ref(),
                &language_detector,
            )
            .await?
        } else {
            log::debug!("Processing files sequentially");
            self.process_files_sequential(
                file_paths,
                &options,
                metadata_extractor.as_ref(),
                git_integrator.as_ref(),
                &language_detector,
            )
            .await?
        };

        log::info!(
            "Scanning completed in {:.2}s: {} files processed",
            start_time.elapsed().as_secs_f64(),
            files.len()
        );

        Ok(files)
    }

    /// Discover files using filesystem traversal with ignore patterns
    async fn discover_files_filesystem(
        &self,
        root: &Path,
        options: &ScanOptions,
    ) -> Result<Vec<PathBuf>> {
        let mut builder = WalkBuilder::new(root);

        builder
            .follow_links(options.follow_symlinks)
            .hidden(!options.include_hidden)
            .git_ignore(true)
            .git_exclude(true)
            .require_git(false);

        let mut files = Vec::new();

        // Use the ignore crate for efficient traversal with gitignore support
        builder.build().for_each(|entry| {
            match entry {
                Ok(entry) => {
                    if entry.file_type().map_or(false, |ft| ft.is_file()) {
                        let path = entry.path().to_path_buf();

                        // Apply extension filters
                        if self.should_include_file(&path, options) {
                            files.push(path);
                        }
                    }

                    if entry.file_type().map_or(false, |ft| ft.is_dir()) {
                        self.stats
                            .directories_traversed
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(err) => {
                    log::warn!("Error during filesystem traversal: {}", err);
                    self.stats
                        .errors_encountered
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            // Continue walking
        });

        Ok(files)
    }

    /// Process files in parallel using Rayon
    async fn process_files_parallel(
        &self,
        file_paths: Vec<PathBuf>,
        options: &ScanOptions,
        metadata_extractor: Option<&MetadataExtractor>,
        git_integrator: Option<&GitIntegrator>,
        language_detector: &LanguageDetector,
    ) -> Result<Vec<FileInfo>> {
        let semaphore = Arc::new(Semaphore::new(options.max_concurrency));
        let results = Arc::new(RwLock::new(Vec::new()));

        // Process files in chunks to manage memory usage
        let chunk_size = 1000;
        for chunk in file_paths.chunks(chunk_size) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|path| {
                    let semaphore = Arc::clone(&semaphore);
                    let results = Arc::clone(&results);
                    let path = path.clone();

                    async move {
                        let _permit = semaphore.acquire().await.unwrap();

                        match self
                            .process_single_file(
                                &path,
                                options,
                                metadata_extractor,
                                git_integrator,
                                language_detector,
                            )
                            .await
                        {
                            Ok(Some(file_info)) => {
                                results.write().await.push(file_info);
                            }
                            Ok(None) => {
                                // File was filtered out or is binary
                            }
                            Err(err) => {
                                log::debug!("Error processing file {}: {}", path.display(), err);
                                self.stats
                                    .errors_encountered
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                })
                .collect();

            // Process chunk concurrently
            stream::iter(futures)
                .buffer_unordered(options.max_concurrency)
                .collect::<Vec<_>>()
                .await;
        }

        let results = results.read().await;
        Ok(results.clone())
    }

    /// Process files sequentially
    async fn process_files_sequential(
        &self,
        file_paths: Vec<PathBuf>,
        options: &ScanOptions,
        metadata_extractor: Option<&MetadataExtractor>,
        git_integrator: Option<&GitIntegrator>,
        language_detector: &LanguageDetector,
    ) -> Result<Vec<FileInfo>> {
        let mut results = Vec::new();

        for path in file_paths {
            match self
                .process_single_file(
                    &path,
                    options,
                    metadata_extractor,
                    git_integrator,
                    language_detector,
                )
                .await
            {
                Ok(Some(file_info)) => {
                    results.push(file_info);
                }
                Ok(None) => {
                    // File was filtered out or is binary
                }
                Err(err) => {
                    log::debug!("Error processing file {}: {}", path.display(), err);
                    self.stats
                        .errors_encountered
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        Ok(results)
    }

    /// Process a single file and extract its information
    async fn process_single_file(
        &self,
        path: &Path,
        options: &ScanOptions,
        metadata_extractor: Option<&MetadataExtractor>,
        git_integrator: Option<&GitIntegrator>,
        language_detector: &LanguageDetector,
    ) -> Result<Option<FileInfo>> {
        // Basic file validation
        if !path.exists() {
            return Ok(None);
        }

        let metadata = tokio::fs::metadata(path).await?;

        // Skip if file is too large
        if let Some(max_size) = options.max_file_size {
            if metadata.len() > max_size {
                log::debug!(
                    "Skipping large file: {} ({} bytes)",
                    path.display(),
                    metadata.len()
                );
                return Ok(None);
            }
        }

        // Basic language detection
        let language = language_detector.detect_language(path);

        // Skip binary files unless specifically included
        if self.is_likely_binary(path, &language) {
            self.stats
                .binary_files_skipped
                .fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }

        // Create base FileInfo
        let relative_path = path.to_string_lossy().to_string();

        let file_type = FileInfo::classify_file_type(
            &relative_path,
            &language,
            path.extension().and_then(|e| e.to_str()).unwrap_or(""),
        );

        let mut file_info = FileInfo {
            path: path.to_path_buf(),
            relative_path,
            size: metadata.len(),
            modified: metadata.modified().ok(),
            decision: RenderDecision::include("scanned file"),
            file_type,
            language,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false, // Will be determined by binary detection
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None, // Will be calculated during analysis phase
        };

        // Extract metadata if requested
        if let Some(extractor) = metadata_extractor {
            if let Ok(file_metadata) = extractor.extract_metadata(path).await {
                file_info.size = file_metadata.size;
                // Copy over other metadata fields as needed
            }
        }

        // Get git information if available
        if let Some(git) = git_integrator {
            if let Ok(git_info) = git.get_file_info(path).await {
                // Add git status and commit info
                file_info.git_status = Some(GitStatus {
                    working_tree: git_info.status,
                    index: GitFileStatus::Unmodified,
                });
            }
        }

        self.stats.files_processed.fetch_add(1, Ordering::Relaxed);
        Ok(Some(file_info))
    }

    /// Check if a file should be included based on extension filters
    fn should_include_file(&self, path: &Path, options: &ScanOptions) -> bool {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Check exclusion list first
        if let Some(ref exclude) = options.exclude_extensions {
            if exclude.iter().any(|ext| ext.to_lowercase() == extension) {
                return false;
            }
        }

        // Check inclusion list if specified
        if let Some(ref include) = options.include_extensions {
            return include.iter().any(|ext| ext.to_lowercase() == extension);
        }

        true
    }

    /// Binary file detection backed by libmagic-style signatures with sensible fallbacks.
    fn is_likely_binary(&self, path: &Path, _language: &Language) -> bool {
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        FileInfo::detect_binary_with_hint(path, extension)
    }

    /// Get current processing statistics
    pub fn files_processed(&self) -> usize {
        self.stats.files_processed.load(Ordering::Relaxed)
    }

    /// Get number of directories traversed
    pub fn directories_traversed(&self) -> usize {
        self.stats.directories_traversed.load(Ordering::Relaxed)
    }

    /// Get number of binary files skipped
    pub fn binary_files_skipped(&self) -> usize {
        self.stats.binary_files_skipped.load(Ordering::Relaxed)
    }

    /// Get number of errors encountered
    pub fn errors_encountered(&self) -> usize {
        self.stats.errors_encountered.load(Ordering::Relaxed)
    }
}

impl Default for Scanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use tokio::fs as async_fs;

    #[tokio::test]
    async fn test_scanner_creation() {
        let scanner = Scanner::new();
        assert_eq!(scanner.files_processed(), 0);
        assert_eq!(scanner.directories_traversed(), 0);
    }

    #[tokio::test]
    async fn test_scan_empty_directory() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        let options = ScanOptions::default();
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_scan_with_files() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        let rust_file = temp_dir.path().join("test.rs");
        let python_file = temp_dir.path().join("test.py");
        let binary_file = temp_dir.path().join("test.bin");

        fs::write(&rust_file, "fn main() { println!(\"Hello, world!\"); }").unwrap();
        fs::write(&python_file, "print('Hello, world!')").unwrap();
        fs::write(&binary_file, &[0u8; 256]).unwrap(); // Binary content

        let options = ScanOptions::default();
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        // Should find the text files but skip the binary
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .any(|f| f.path.file_name().unwrap() == "test.rs"));
        assert!(results
            .iter()
            .any(|f| f.path.file_name().unwrap() == "test.py"));

        // Check language detection
        let rust_file_info = results
            .iter()
            .find(|f| f.path.file_name().unwrap() == "test.rs")
            .unwrap();
        assert_eq!(rust_file_info.language, Language::Rust);

        let python_file_info = results
            .iter()
            .find(|f| f.path.file_name().unwrap() == "test.py")
            .unwrap();
        assert_eq!(python_file_info.language, Language::Python);
    }

    #[tokio::test]
    async fn test_scan_options_extension_filtering() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        // Create test files with different extensions
        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();
        fs::write(temp_dir.path().join("test.py"), "print('hello')").unwrap();
        fs::write(temp_dir.path().join("test.js"), "console.log('hello')").unwrap();

        // Test include filter
        let options = ScanOptions::default()
            .with_include_extensions(vec!["rs".to_string(), "py".to_string()]);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|f| f.path.extension().unwrap() == "rs"));
        assert!(results.iter().any(|f| f.path.extension().unwrap() == "py"));
        assert!(!results.iter().any(|f| f.path.extension().unwrap() == "js"));
    }

    #[tokio::test]
    async fn test_parallel_processing() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        // Create multiple test files to trigger parallel processing
        for i in 0..150 {
            let file_path = temp_dir.path().join(format!("test_{}.rs", i));
            fs::write(&file_path, format!("fn main_{i}() {{}}")).unwrap();
        }

        let options = ScanOptions::default()
            .with_parallel_processing(true)
            .with_max_concurrency(4);

        let start = Instant::now();
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), 150);
        log::info!("Parallel scan of 150 files took: {:?}", duration);

        // Verify all files were processed correctly
        for i in 0..150 {
            assert!(results
                .iter()
                .any(|f| { f.path.file_name().unwrap() == format!("test_{}.rs", i).as_str() }));
        }
    }

    #[test]
    fn test_scan_options_builder() {
        let options = ScanOptions::default()
            .with_parallel_processing(true)
            .with_max_concurrency(8)
            .with_metadata_extraction(true)
            .with_git_integration(false)
            .with_follow_symlinks(false)
            .with_include_hidden(true)
            .with_max_file_size(Some(1024 * 1024));

        assert_eq!(options.parallel_processing, true);
        assert_eq!(options.max_concurrency, 8);
        assert_eq!(options.metadata_extraction, true);
        assert_eq!(options.git_integration, false);
        assert_eq!(options.follow_symlinks, false);
        assert_eq!(options.include_hidden, true);
        assert_eq!(options.max_file_size, Some(1024 * 1024));
    }

    #[test]
    fn test_binary_file_detection() {
        let scanner = Scanner::new();
        let temp_dir = tempfile::TempDir::new().unwrap();

        let text_path = temp_dir.path().join("test.rs");
        std::fs::write(&text_path, "fn main() {}\n").unwrap();

        let binary_path = temp_dir.path().join("image.png");
        std::fs::write(&binary_path, &[0u8, 159, 146, 150, 0, 1]).unwrap();

        assert!(scanner.is_likely_binary(&binary_path, &Language::Unknown));
        assert!(!scanner.is_likely_binary(&text_path, &Language::Rust));
    }

    #[test]
    fn test_scanner_default() {
        let scanner = Scanner::default();
        assert_eq!(scanner.files_processed(), 0);
        assert_eq!(scanner.binary_files_skipped(), 0);
        assert_eq!(scanner.errors_encountered(), 0);
    }

    #[test]
    fn test_scan_options_default() {
        let options = ScanOptions::default();
        assert!(options.parallel_processing);
        assert!(options.max_concurrency > 0);
        assert!(options.metadata_extraction);
        assert!(!options.git_integration);
        assert!(!options.follow_symlinks);
        assert!(!options.include_hidden);
        assert!(options.max_file_size.is_some());
        assert!(options.include_extensions.is_none());
        assert!(options.exclude_extensions.is_none());
    }

    #[test]
    fn test_should_include_file_no_filters() {
        let scanner = Scanner::new();
        let options = ScanOptions::default();

        assert!(scanner.should_include_file(Path::new("test.rs"), &options));
        assert!(scanner.should_include_file(Path::new("test.py"), &options));
        assert!(scanner.should_include_file(Path::new("test.js"), &options));
        assert!(scanner.should_include_file(Path::new("no_extension"), &options));
    }

    #[test]
    fn test_should_include_file_with_include_filter() {
        let scanner = Scanner::new();
        let options = ScanOptions::default()
            .with_include_extensions(vec!["rs".to_string(), "py".to_string()]);

        assert!(scanner.should_include_file(Path::new("test.rs"), &options));
        assert!(scanner.should_include_file(Path::new("test.py"), &options));
        assert!(!scanner.should_include_file(Path::new("test.js"), &options));
        assert!(!scanner.should_include_file(Path::new("test.txt"), &options));
    }

    #[test]
    fn test_should_include_file_with_exclude_filter() {
        let scanner = Scanner::new();
        let options = ScanOptions::default()
            .with_exclude_extensions(vec!["txt".to_string(), "log".to_string()]);

        assert!(scanner.should_include_file(Path::new("test.rs"), &options));
        assert!(scanner.should_include_file(Path::new("test.py"), &options));
        assert!(!scanner.should_include_file(Path::new("test.txt"), &options));
        assert!(!scanner.should_include_file(Path::new("test.log"), &options));
    }

    #[test]
    fn test_should_include_file_exclude_takes_precedence() {
        let scanner = Scanner::new();
        let options = ScanOptions::default()
            .with_include_extensions(vec!["rs".to_string(), "txt".to_string()])
            .with_exclude_extensions(vec!["txt".to_string()]);

        assert!(scanner.should_include_file(Path::new("test.rs"), &options));
        assert!(!scanner.should_include_file(Path::new("test.txt"), &options)); // Excluded takes precedence
    }

    #[test]
    fn test_should_include_file_case_insensitive() {
        let scanner = Scanner::new();
        let options = ScanOptions::default()
            .with_include_extensions(vec!["RS".to_string()]);

        assert!(scanner.should_include_file(Path::new("test.rs"), &options));
        assert!(scanner.should_include_file(Path::new("test.RS"), &options));
        assert!(scanner.should_include_file(Path::new("test.Rs"), &options));
    }

    #[test]
    fn test_scan_progress_clone() {
        let progress = ScanProgress {
            files_processed: 100,
            directories_traversed: 20,
            binary_files_skipped: 5,
            errors_encountered: 2,
            bytes_processed: 1024 * 1024,
        };

        let cloned = progress.clone();
        assert_eq!(progress.files_processed, cloned.files_processed);
        assert_eq!(progress.bytes_processed, cloned.bytes_processed);
    }

    #[test]
    fn test_scan_result_clone() {
        let result = ScanResult {
            files: vec![],
            stats: ScanProgress {
                files_processed: 10,
                directories_traversed: 2,
                binary_files_skipped: 1,
                errors_encountered: 0,
                bytes_processed: 1024,
            },
            duration: std::time::Duration::from_secs(1),
            errors: vec!["test error".to_string()],
        };

        let cloned = result.clone();
        assert_eq!(result.errors.len(), cloned.errors.len());
        assert_eq!(result.stats.files_processed, cloned.stats.files_processed);
    }

    #[test]
    fn test_scan_options_debug() {
        let options = ScanOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("parallel_processing"));
    }

    #[test]
    fn test_scanner_debug() {
        let scanner = Scanner::new();
        let debug_str = format!("{:?}", scanner);
        assert!(debug_str.contains("Scanner"));
    }

    #[test]
    fn test_scanner_stats_debug() {
        let stats = ScannerStats::default();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ScannerStats"));
    }

    #[test]
    fn test_scan_progress_debug() {
        let progress = ScanProgress {
            files_processed: 100,
            directories_traversed: 20,
            binary_files_skipped: 5,
            errors_encountered: 2,
            bytes_processed: 1024,
        };
        let debug_str = format!("{:?}", progress);
        assert!(debug_str.contains("100"));
    }

    #[tokio::test]
    async fn test_scan_nonexistent_path() {
        let scanner = Scanner::new();
        let result = scanner.scan("/nonexistent/path/12345", ScanOptions::default()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scan_file_instead_of_directory() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.rs");
        fs::write(&file_path, "fn main() {}").unwrap();

        // Trying to scan a file instead of directory should fail
        let result = scanner.scan(&file_path, ScanOptions::default()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scan_sequential_processing() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("test1.rs"), "fn main() {}").unwrap();
        fs::write(temp_dir.path().join("test2.rs"), "fn other() {}").unwrap();

        let options = ScanOptions::default().with_parallel_processing(false);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_scan_with_hidden_files() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("visible.rs"), "fn main() {}").unwrap();
        fs::write(temp_dir.path().join(".hidden.rs"), "fn hidden() {}").unwrap();

        // Without hidden files
        let options = ScanOptions::default().with_include_hidden(false);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();
        assert_eq!(results.len(), 1);

        // With hidden files
        let options = ScanOptions::default().with_include_hidden(true);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_scan_with_max_file_size() {
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        // Small file
        fs::write(temp_dir.path().join("small.rs"), "fn main() {}").unwrap();

        // Large file (1MB)
        let large_content = "fn main() {}".repeat(100000);
        fs::write(temp_dir.path().join("large.rs"), &large_content).unwrap();

        // With small max size limit (100 bytes)
        let options = ScanOptions::default().with_max_file_size(Some(100));
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        // Only small file should be included
        assert_eq!(results.len(), 1);
        assert!(results[0].path.to_str().unwrap().contains("small"));
    }

    #[tokio::test]
    async fn test_scan_metadata_extraction_disabled() {
        // Tests line 189: metadata_extraction = false path
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();

        let options = ScanOptions::default()
            .with_metadata_extraction(false);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_scan_with_git_integration_disabled() {
        // Tests line 195: git_integration = false path
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();

        let options = ScanOptions::default()
            .with_git_integration(false);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        assert_eq!(results.len(), 1);
        // Git status should be None when git integration is disabled
        assert!(results[0].git_status.is_none());
    }

    #[tokio::test]
    async fn test_scan_parallel_many_files() {
        // Tests lines 231-232, 255-257: parallel processing with logging
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        // Create many files to ensure we hit the parallel path with chunking
        for i in 0..200 {
            fs::write(temp_dir.path().join(format!("file_{}.rs", i)), format!("fn f{}() {{}}", i)).unwrap();
        }

        let options = ScanOptions::default()
            .with_parallel_processing(true)
            .with_max_concurrency(4);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        assert_eq!(results.len(), 200);
    }

    #[tokio::test]
    async fn test_scan_git_integration_enabled_no_repo() {
        // Tests lines 202-206, 208, 210-212: git integration enabled but fails
        let scanner = Scanner::new();
        let temp_dir = TempDir::new().unwrap();

        // Create a file but don't initialize git repo
        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();

        let options = ScanOptions::default()
            .with_git_integration(true);
        let results = scanner.scan(temp_dir.path(), options).await.unwrap();

        // Should still work and fall back to filesystem discovery
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_scan_options_with_follow_symlinks() {
        let options = ScanOptions::default().with_follow_symlinks(true);
        assert!(options.follow_symlinks);
    }

    #[test]
    fn test_scan_stats_atomic_operations() {
        // Tests the atomic counters in ScannerStats
        let stats = ScannerStats::default();

        stats.files_processed.fetch_add(5, Ordering::Relaxed);
        assert_eq!(stats.files_processed.load(Ordering::Relaxed), 5);

        stats.directories_traversed.fetch_add(2, Ordering::Relaxed);
        assert_eq!(stats.directories_traversed.load(Ordering::Relaxed), 2);

        stats.binary_files_skipped.fetch_add(1, Ordering::Relaxed);
        assert_eq!(stats.binary_files_skipped.load(Ordering::Relaxed), 1);

        stats.errors_encountered.fetch_add(3, Ordering::Relaxed);
        assert_eq!(stats.errors_encountered.load(Ordering::Relaxed), 3);
    }
}

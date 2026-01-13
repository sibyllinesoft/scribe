//! Progressive loading and streaming for memory-efficient file processing.
//!
//! This module provides true streaming file discovery and processing, avoiding
//! the memory bottleneck of loading all file metadata at once.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::SystemTime;

use futures::{Stream, StreamExt};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{debug, info, warn};

use crate::core::error::{ScalingError, ScalingResult};
use crate::core::utils::classify_file_type_string;
use scribe_core::file;

/// File metadata for streaming operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileMetadata {
    /// File path
    pub path: PathBuf,

    /// File size in bytes
    pub size: u64,

    /// Last modified time
    pub modified: SystemTime,

    /// Detected programming language
    pub language: String,

    /// File type classification
    pub file_type: String,
}

/// Configuration for streaming operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Whether to enable streaming (vs loading all at once)
    pub enable_streaming: bool,

    /// Number of files to process concurrently
    pub concurrency_limit: usize,

    /// Memory limit for streaming operations (bytes)
    pub memory_limit: usize,

    /// Maximum files to hold in selection heap
    pub selection_heap_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            concurrency_limit: num_cpus::get() * 2,
            memory_limit: 100 * 1024 * 1024, // 100MB
            selection_heap_size: 10000,      // Maximum files in selection heap
        }
    }
}

/// Scored file for heap-based selection
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredFile {
    pub metadata: FileMetadata,
    pub score: f64,
    pub tokens: usize,
}

impl Eq for ScoredFile {}

impl PartialOrd for ScoredFile {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredFile {
    fn cmp(&self, other: &Self) -> Ordering {
        // For min-heap: higher scores should be "less" so they get removed last
        // We want to keep highest scores, so lower scores should be removed first
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.tokens.cmp(&self.tokens)) // Prefer smaller token files when scores equal
    }
}

/// Streaming file selector with heap-based optimization
pub struct StreamingSelector {
    config: StreamingConfig,
}

impl StreamingSelector {
    /// Create new streaming selector
    pub fn new(config: StreamingConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(StreamingConfig::default())
    }

    /// Stream files from directory with intelligent selection
    ///
    /// This uses O(N log K) complexity instead of O(N log N) where:
    /// - N = total files in repository
    /// - K = target number of files to select
    pub async fn select_files_streaming(
        &self,
        repo_path: &Path,
        target_count: usize,
        token_budget: usize,
        score_fn: impl Fn(&FileMetadata) -> f64 + Send + Sync + 'static,
        token_fn: impl Fn(&FileMetadata) -> usize + Send + Sync + 'static,
    ) -> ScalingResult<Vec<ScoredFile>> {
        info!("Starting streaming file selection for: {:?}", repo_path);
        info!(
            "Target: {} files, Budget: {} tokens",
            target_count, token_budget
        );

        if !repo_path.exists() {
            return Err(ScalingError::path(
                "Repository path does not exist",
                repo_path,
            ));
        }

        if !repo_path.is_dir() {
            return Err(ScalingError::path(
                "Repository path is not a directory",
                repo_path,
            ));
        }

        // Use min-heap to keep only the best K files in memory
        let mut selection_heap: BinaryHeap<Reverse<ScoredFile>> = BinaryHeap::new();
        let mut total_files_seen = 0usize;
        let mut total_tokens_used = 0usize;

        // Create file discovery stream
        let file_stream = self.create_file_stream(repo_path).await?;

        // Process files in parallel batches
        let mut file_stream = Box::pin(file_stream);

        while let Some(file_batch) = file_stream.next().await {
            total_files_seen += file_batch.len();

            // Score files in parallel
            let scored_batch: Vec<ScoredFile> = file_batch
                .into_par_iter()
                .filter_map(|metadata| {
                    let score = score_fn(&metadata);
                    let tokens = token_fn(&metadata);

                    // Skip files that would exceed budget immediately
                    if tokens > token_budget {
                        return None;
                    }

                    Some(ScoredFile {
                        metadata,
                        score,
                        tokens,
                    })
                })
                .collect();

            // Update selection heap with O(log K) insertions
            for scored_file in scored_batch {
                if selection_heap.len() < target_count {
                    // Heap not full, add directly
                    total_tokens_used += scored_file.tokens;
                    selection_heap.push(Reverse(scored_file));
                } else if let Some(worst) = selection_heap.peek() {
                    // Check if this file is better than the worst in heap
                    if scored_file.score > worst.0.score {
                        // Remove worst file
                        if let Some(Reverse(removed)) = selection_heap.pop() {
                            total_tokens_used = total_tokens_used.saturating_sub(removed.tokens);
                        }

                        // Add new file if it fits in budget
                        if total_tokens_used + scored_file.tokens <= token_budget {
                            total_tokens_used += scored_file.tokens;
                            selection_heap.push(Reverse(scored_file));
                        } else {
                            // Try to fit by removing files from heap
                            self.optimize_heap_for_budget(
                                &mut selection_heap,
                                &mut total_tokens_used,
                                token_budget,
                            );
                            if total_tokens_used + scored_file.tokens <= token_budget {
                                total_tokens_used += scored_file.tokens;
                                selection_heap.push(Reverse(scored_file));
                            }
                        }
                    }
                }
            }

            // Log progress every 10k files
            if total_files_seen % 10000 == 0 {
                debug!(
                    "Processed {} files, selected {} candidates",
                    total_files_seen,
                    selection_heap.len()
                );
            }
        }

        info!(
            "Streaming selection complete: {} files processed, {} selected",
            total_files_seen,
            selection_heap.len()
        );
        info!(
            "Token utilization: {}/{} ({:.1}%)",
            total_tokens_used,
            token_budget,
            (total_tokens_used as f64 / token_budget as f64) * 100.0
        );

        // Convert heap to sorted vec (highest scores first)
        let mut selected: Vec<ScoredFile> =
            selection_heap.into_iter().map(|Reverse(sf)| sf).collect();

        // Sort by score descending for final output
        selected.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        Ok(selected)
    }

    /// Create async stream of file metadata from directory
    async fn create_file_stream(
        &self,
        repo_path: &Path,
    ) -> ScalingResult<impl Stream<Item = Vec<FileMetadata>> + use<'_>> {
        let walkdir_iter = walkdir::WalkDir::new(repo_path)
            .follow_links(false)
            .max_depth(20) // Reasonable depth limit
            .into_iter();

        // Convert walkdir iterator to async stream
        let concurrency_limit = self.config.concurrency_limit;
        let file_stream = futures::stream::iter(walkdir_iter)
            .filter_map(move |entry| async move {
                match entry {
                    Ok(entry) if entry.file_type().is_file() => {
                        Some(Self::create_file_metadata_static(entry).await)
                    }
                    Ok(_) => None, // Skip directories
                    Err(e) => {
                        warn!("Skipping file due to error: {}", e);
                        None
                    }
                }
            })
            .filter_map(|result| async move {
                match result {
                    Ok(metadata) => Some(metadata),
                    Err(e) => {
                        warn!("Failed to create file metadata: {}", e);
                        None
                    }
                }
            })
            .chunks(concurrency_limit); // Batch for parallel processing

        Ok(file_stream)
    }

    /// Create file metadata from walkdir entry (static version)
    async fn create_file_metadata_static(entry: walkdir::DirEntry) -> ScalingResult<FileMetadata> {
        let path = entry.path().to_path_buf();

        let (size, modified) = match entry.metadata() {
            Ok(metadata) => {
                let size = metadata.len();
                let modified = metadata.modified().unwrap_or_else(|_| SystemTime::now());
                (size, modified)
            }
            Err(_) => (0, SystemTime::now()),
        };

        let language = detect_language(&path);
        let file_type = classify_file_type(&path);

        Ok(FileMetadata {
            path,
            size,
            modified,
            language,
            file_type,
        })
    }

    /// Optimize heap to fit within token budget by removing lowest-value files
    fn optimize_heap_for_budget(
        &self,
        heap: &mut BinaryHeap<Reverse<ScoredFile>>,
        current_tokens: &mut usize,
        budget: usize,
    ) {
        while *current_tokens > budget && !heap.is_empty() {
            if let Some(Reverse(removed)) = heap.pop() {
                *current_tokens = current_tokens.saturating_sub(removed.tokens);
            }
        }
    }
}

/// Fast language detection based on file extension
fn detect_language(path: &Path) -> String {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    if matches!(extension.as_deref(), Some("h" | "hpp" | "hxx")) {
        return "Header".to_string();
    }

    if path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("dockerfile"))
        .unwrap_or(false)
    {
        return "Dockerfile".to_string();
    }

    let language = file::detect_language_from_path(path);
    file::language_display_name(&language).to_string()
}

/// Fast file type classification - delegates to shared utility
fn classify_file_type(path: &Path) -> String {
    classify_file_type_string(path)
}

/// Legacy file chunk for backwards compatibility
#[derive(Debug, Clone)]
pub struct FileChunk {
    /// Files in this chunk
    pub files: Vec<FileMetadata>,

    /// Chunk index
    pub index: usize,

    /// Total number of chunks
    pub total_chunks: usize,
}

impl FileChunk {
    /// Create a new file chunk
    pub fn new(files: Vec<FileMetadata>, index: usize, total_chunks: usize) -> Self {
        Self {
            files,
            index,
            total_chunks,
        }
    }

    /// Get the number of files in this chunk
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Check if the chunk is empty
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Get total size of all files in this chunk
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_streaming_selector_creation() {
        let selector = StreamingSelector::with_defaults();
        assert!(selector.config.enable_streaming);
        assert!(selector.config.concurrency_limit > 0);
    }

    #[tokio::test]
    async fn test_streaming_file_selection() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create test files
        fs::create_dir_all(repo_path.join("src")).unwrap();
        for i in 0..100 {
            let content = format!("// File {}\nfn main() {{ println!(\"Hello {}\"); }}", i, i);
            fs::write(
                repo_path.join("src").join(format!("file_{}.rs", i)),
                content,
            )
            .unwrap();
        }

        let selector = StreamingSelector::with_defaults();

        // Simple scoring function
        let score_fn = |file: &FileMetadata| {
            if file.path.to_string_lossy().contains("file_1") {
                2.0 // Boost files with "1" in name
            } else {
                1.0
            }
        };

        // Simple token estimation
        let token_fn = |file: &FileMetadata| (file.size / 4) as usize;

        let selected = selector
            .select_files_streaming(repo_path, 10, 10000, score_fn, token_fn)
            .await
            .unwrap();

        // Should select some files
        assert!(!selected.is_empty());
        assert!(selected.len() <= 10);

        // Files should be sorted by score (highest first)
        for i in 1..selected.len() {
            assert!(selected[i - 1].score >= selected[i].score);
        }
    }

    #[test]
    fn test_scored_file_ordering() {
        let file1 = FileMetadata {
            path: PathBuf::from("test1.rs"),
            size: 100,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let file2 = file1.clone();

        let scored1 = ScoredFile {
            metadata: file1,
            score: 2.0,
            tokens: 100,
        };
        let scored2 = ScoredFile {
            metadata: file2,
            score: 1.0,
            tokens: 50,
        };

        // Higher score should be "greater" in our ordering
        assert!(scored1 > scored2);

        // Test in heap
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(scored1.clone()));
        heap.push(Reverse(scored2.clone()));

        // Min-heap with Reverse should give us the lowest score first (for removal)
        assert_eq!(heap.pop().unwrap().0.score, 1.0);
        assert_eq!(heap.pop().unwrap().0.score, 2.0);
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(detect_language(&PathBuf::from("test.rs")), "Rust");
        assert_eq!(detect_language(&PathBuf::from("test.py")), "Python");
        assert_eq!(detect_language(&PathBuf::from("test.js")), "JavaScript");
        assert_eq!(detect_language(&PathBuf::from("test.unknown")), "Unknown");
    }

    #[test]
    fn test_file_type_classification() {
        assert_eq!(classify_file_type(&PathBuf::from("main.rs")), "Source");
        assert_eq!(
            classify_file_type(&PathBuf::from("README.md")),
            "Documentation"
        );
        assert_eq!(
            classify_file_type(&PathBuf::from("config.json")),
            "Configuration"
        );
        assert_eq!(classify_file_type(&PathBuf::from("style.css")), "Web");
        // PNG files are classified as Binary by FileInfo, then we check extension for Image
        // In practice Binary is returned because scribe_core detects .png as binary first
        assert_eq!(classify_file_type(&PathBuf::from("image.png")), "Binary");
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert!(config.enable_streaming);
        assert!(config.concurrency_limit > 0);
        assert_eq!(config.memory_limit, 100 * 1024 * 1024);
        assert_eq!(config.selection_heap_size, 10000);
    }

    #[test]
    fn test_streaming_config_custom() {
        let config = StreamingConfig {
            enable_streaming: false,
            concurrency_limit: 4,
            memory_limit: 50 * 1024 * 1024,
            selection_heap_size: 5000,
        };

        assert!(!config.enable_streaming);
        assert_eq!(config.concurrency_limit, 4);
        assert_eq!(config.memory_limit, 50 * 1024 * 1024);
    }

    #[test]
    fn test_file_metadata_clone() {
        let metadata = FileMetadata {
            path: PathBuf::from("src/lib.rs"),
            size: 1024,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let cloned = metadata.clone();
        assert_eq!(metadata.path, cloned.path);
        assert_eq!(metadata.size, cloned.size);
        assert_eq!(metadata.language, cloned.language);
    }

    #[test]
    fn test_file_metadata_equality() {
        let now = SystemTime::now();
        let m1 = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: now,
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let m2 = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: now,
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_scored_file_clone() {
        let metadata = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let scored = ScoredFile {
            metadata,
            score: 1.5,
            tokens: 50,
        };

        let cloned = scored.clone();
        assert_eq!(scored.score, cloned.score);
        assert_eq!(scored.tokens, cloned.tokens);
    }

    #[test]
    fn test_scored_file_equal_scores_prefer_smaller_tokens() {
        let file1 = FileMetadata {
            path: PathBuf::from("large.rs"),
            size: 1000,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let file2 = FileMetadata {
            path: PathBuf::from("small.rs"),
            size: 100,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let scored1 = ScoredFile {
            metadata: file1,
            score: 1.0,
            tokens: 250, // More tokens
        };
        let scored2 = ScoredFile {
            metadata: file2,
            score: 1.0,
            tokens: 25, // Fewer tokens - should be preferred
        };

        // With equal scores, smaller token files are preferred (greater in ordering)
        assert!(scored2 > scored1);
    }

    #[test]
    fn test_file_chunk_creation() {
        let files = vec![
            FileMetadata {
                path: PathBuf::from("a.rs"),
                size: 100,
                modified: SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            FileMetadata {
                path: PathBuf::from("b.rs"),
                size: 200,
                modified: SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
        ];

        let chunk = FileChunk::new(files, 0, 5);

        assert_eq!(chunk.len(), 2);
        assert!(!chunk.is_empty());
        assert_eq!(chunk.total_size(), 300);
        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.total_chunks, 5);
    }

    #[test]
    fn test_file_chunk_empty() {
        let chunk = FileChunk::new(vec![], 0, 1);

        assert_eq!(chunk.len(), 0);
        assert!(chunk.is_empty());
        assert_eq!(chunk.total_size(), 0);
    }

    #[test]
    fn test_language_detection_more_types() {
        assert_eq!(detect_language(&PathBuf::from("test.ts")), "TypeScript");
        assert_eq!(detect_language(&PathBuf::from("test.go")), "Go");
        assert_eq!(detect_language(&PathBuf::from("Test.java")), "Java");
        assert_eq!(detect_language(&PathBuf::from("test.h")), "Header");
        assert_eq!(detect_language(&PathBuf::from("test.hpp")), "Header");
        assert_eq!(detect_language(&PathBuf::from("test.hxx")), "Header");
    }

    #[test]
    fn test_language_detection_dockerfile() {
        assert_eq!(detect_language(&PathBuf::from("Dockerfile")), "Dockerfile");
        assert_eq!(detect_language(&PathBuf::from("dockerfile")), "Dockerfile");
        assert_eq!(detect_language(&PathBuf::from("DOCKERFILE")), "Dockerfile");
    }

    #[test]
    fn test_language_detection_no_extension() {
        // Files without extensions that aren't Dockerfile
        assert_eq!(detect_language(&PathBuf::from("Makefile")), "Unknown");
        assert_eq!(detect_language(&PathBuf::from("LICENSE")), "Unknown");
    }

    #[tokio::test]
    async fn test_streaming_selector_nonexistent_path() {
        let selector = StreamingSelector::with_defaults();
        let score_fn = |_: &FileMetadata| 1.0;
        let token_fn = |_: &FileMetadata| 10;

        let result = selector
            .select_files_streaming(
                Path::new("/nonexistent/path/that/does/not/exist"),
                10,
                1000,
                score_fn,
                token_fn,
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_streaming_selector_file_not_dir() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "test content").unwrap();

        let selector = StreamingSelector::with_defaults();
        let score_fn = |_: &FileMetadata| 1.0;
        let token_fn = |_: &FileMetadata| 10;

        let result = selector
            .select_files_streaming(&file_path, 10, 1000, score_fn, token_fn)
            .await;

        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_selector_new() {
        let config = StreamingConfig {
            enable_streaming: true,
            concurrency_limit: 8,
            memory_limit: 200 * 1024 * 1024,
            selection_heap_size: 20000,
        };

        let selector = StreamingSelector::new(config);
        assert_eq!(selector.config.concurrency_limit, 8);
        assert_eq!(selector.config.selection_heap_size, 20000);
    }

    #[test]
    fn test_file_chunk_clone() {
        let files = vec![
            FileMetadata {
                path: PathBuf::from("test.rs"),
                size: 50,
                modified: SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
        ];

        let chunk = FileChunk::new(files, 2, 10);
        let cloned = chunk.clone();

        assert_eq!(chunk.index, cloned.index);
        assert_eq!(chunk.total_chunks, cloned.total_chunks);
        assert_eq!(chunk.len(), cloned.len());
    }

    #[test]
    fn test_scored_file_partial_ord() {
        let metadata = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let scored1 = ScoredFile {
            metadata: metadata.clone(),
            score: 1.0,
            tokens: 10,
        };
        let scored2 = ScoredFile {
            metadata: metadata.clone(),
            score: 2.0,
            tokens: 10,
        };

        // Test partial_cmp
        assert!(scored1.partial_cmp(&scored2).is_some());
        assert_eq!(scored1.partial_cmp(&scored2), Some(Ordering::Less));
        assert_eq!(scored2.partial_cmp(&scored1), Some(Ordering::Greater));
    }

    #[test]
    fn test_file_metadata_debug() {
        let metadata = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let debug_str = format!("{:?}", metadata);
        assert!(debug_str.contains("FileMetadata"));
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_scored_file_debug() {
        let metadata = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let scored = ScoredFile {
            metadata,
            score: 1.5,
            tokens: 25,
        };

        let debug_str = format!("{:?}", scored);
        assert!(debug_str.contains("ScoredFile"));
        assert!(debug_str.contains("score"));
    }

    #[test]
    fn test_file_chunk_debug() {
        let chunk = FileChunk::new(vec![], 0, 1);
        let debug_str = format!("{:?}", chunk);
        assert!(debug_str.contains("FileChunk"));
    }

    #[test]
    fn test_optimize_heap_for_budget() {
        let selector = StreamingSelector::with_defaults();
        let mut heap: BinaryHeap<Reverse<ScoredFile>> = BinaryHeap::new();
        let mut current_tokens = 0usize;

        // Add some files to the heap
        for i in 0..5 {
            let metadata = FileMetadata {
                path: PathBuf::from(format!("file_{}.rs", i)),
                size: 100,
                modified: SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            };
            let scored = ScoredFile {
                metadata,
                score: (i + 1) as f64,
                tokens: 100,
            };
            current_tokens += scored.tokens;
            heap.push(Reverse(scored));
        }

        assert_eq!(current_tokens, 500);
        assert_eq!(heap.len(), 5);

        // Optimize to fit within budget of 300 tokens
        selector.optimize_heap_for_budget(&mut heap, &mut current_tokens, 300);

        // Should have removed some files
        assert!(current_tokens <= 300);
        assert!(heap.len() < 5);
    }

    #[test]
    fn test_optimize_heap_for_budget_empty_heap() {
        let selector = StreamingSelector::with_defaults();
        let mut heap: BinaryHeap<Reverse<ScoredFile>> = BinaryHeap::new();
        let mut current_tokens = 100usize;

        // Optimize an empty heap - should not crash
        selector.optimize_heap_for_budget(&mut heap, &mut current_tokens, 50);

        // Tokens should stay at 100 since heap was empty (nothing to remove)
        assert_eq!(current_tokens, 100);
    }

    #[test]
    fn test_optimize_heap_for_budget_already_fits() {
        let selector = StreamingSelector::with_defaults();
        let mut heap: BinaryHeap<Reverse<ScoredFile>> = BinaryHeap::new();

        let metadata = FileMetadata {
            path: PathBuf::from("small.rs"),
            size: 50,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        let scored = ScoredFile {
            metadata,
            score: 1.0,
            tokens: 50,
        };
        let mut current_tokens = scored.tokens;
        heap.push(Reverse(scored));

        // Already fits budget - no changes
        selector.optimize_heap_for_budget(&mut heap, &mut current_tokens, 100);

        assert_eq!(current_tokens, 50);
        assert_eq!(heap.len(), 1);
    }

    #[tokio::test]
    async fn test_streaming_with_large_files_exceeding_budget() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create test files - one large, one small
        fs::create_dir_all(repo_path.join("src")).unwrap();
        fs::write(
            repo_path.join("src").join("small.rs"),
            "fn main() {}",
        ).unwrap();
        fs::write(
            repo_path.join("src").join("large.rs"),
            "x".repeat(10000), // Large file
        ).unwrap();

        let selector = StreamingSelector::with_defaults();

        // Token function that makes large files exceed budget
        let score_fn = |_: &FileMetadata| 1.0;
        let token_fn = |file: &FileMetadata| file.size as usize;

        // Very small budget - should only fit small file
        let selected = selector
            .select_files_streaming(repo_path, 10, 100, score_fn, token_fn)
            .await
            .unwrap();

        // Should have selected only the small file
        assert!(selected.len() >= 1);
        for file in &selected {
            assert!(file.tokens <= 100);
        }
    }

    #[tokio::test]
    async fn test_streaming_selection_replaces_lower_score() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create 20 files
        fs::create_dir_all(repo_path.join("src")).unwrap();
        for i in 0..20 {
            let content = format!("// file {}\nfn func{}() {{}}", i, i);
            fs::write(
                repo_path.join("src").join(format!("file_{:02}.rs", i)),
                content,
            ).unwrap();
        }

        let selector = StreamingSelector::with_defaults();

        // Score function that gives higher scores to higher numbered files
        let score_fn = |file: &FileMetadata| {
            let path_str = file.path.to_string_lossy();
            if path_str.contains("file_1") {
                10.0 // Files with 1 (10-19) get higher score
            } else {
                1.0
            }
        };
        let token_fn = |_: &FileMetadata| 10;

        // Select only 5 files - should prefer higher scored ones
        let selected = selector
            .select_files_streaming(repo_path, 5, 10000, score_fn, token_fn)
            .await
            .unwrap();

        assert_eq!(selected.len(), 5);
        // Higher scored files should be selected
        let high_score_count = selected.iter().filter(|f| f.score >= 10.0).count();
        assert!(high_score_count > 0);
    }

    #[test]
    fn test_scored_file_equality() {
        let now = SystemTime::now();
        let m1 = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: now,
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        let m2 = m1.clone();

        let s1 = ScoredFile { metadata: m1, score: 1.0, tokens: 10 };
        let s2 = ScoredFile { metadata: m2, score: 1.0, tokens: 10 };

        // Test Eq trait
        assert!(s1 == s2);
    }

    #[test]
    fn test_file_metadata_serialize_deserialize() {
        let metadata = FileMetadata {
            path: PathBuf::from("test.rs"),
            size: 100,
            modified: SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: FileMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata.path, deserialized.path);
        assert_eq!(metadata.size, deserialized.size);
        assert_eq!(metadata.language, deserialized.language);
    }

    #[test]
    fn test_streaming_config_serialize_deserialize() {
        let config = StreamingConfig::default();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: StreamingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.enable_streaming, deserialized.enable_streaming);
        assert_eq!(config.concurrency_limit, deserialized.concurrency_limit);
    }
}

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
use scribe_core::{file, FileInfo, FileType};

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

/// Fast file type classification
fn classify_file_type(path: &Path) -> String {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let language = file::detect_language_from_path(path);
    let file_type =
        FileInfo::classify_file_type(path.to_string_lossy().as_ref(), &language, &extension);

    match file_type {
        FileType::Test { .. } => "Test".to_string(),
        FileType::Documentation { .. } => "Documentation".to_string(),
        FileType::Configuration { .. } => "Configuration".to_string(),
        FileType::Binary => "Binary".to_string(),
        FileType::Generated => "Generated".to_string(),
        FileType::Source { .. } => match extension.as_str() {
            "jsx" | "tsx" | "vue" | "svelte" => "Frontend".to_string(),
            "html" | "htm" | "css" | "scss" | "sass" | "less" => "Web".to_string(),
            "sh" | "bash" | "bat" | "ps1" => "Script".to_string(),
            _ => "Source".to_string(),
        },
        FileType::Unknown => match extension.as_str() {
            "png" | "jpg" | "jpeg" | "gif" | "svg" | "ico" => "Image".to_string(),
            "pdf" | "doc" | "docx" | "ppt" | "pptx" => "Document".to_string(),
            "sql" => "Database".to_string(),
            "xml" | "xsd" | "xsl" => "Markup".to_string(),
            "json" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" => {
                "Configuration".to_string()
            }
            _ => "Other".to_string(),
        },
    }
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
}

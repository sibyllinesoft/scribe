//! Type definitions for incremental scanning.

use fxhash::FxHashMap;
use serde::{Deserialize, Serialize};

/// File manifest for incremental scanning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileManifest {
    /// Version of the manifest format
    pub version: u32,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
    /// Repository root path (for validation)
    pub repo_root: String,
    /// Git commit hash (if in git repo)
    pub git_commit: Option<String>,
    /// File entries keyed by relative path
    pub entries: FxHashMap<String, ManifestEntry>,
    /// Manifest statistics
    pub stats: ManifestStats,
}

/// Individual file entry in the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Relative path from repository root
    pub path: String,
    /// File size in bytes
    pub size: u64,
    /// Last modified timestamp (seconds since epoch)
    pub modified: u64,
    /// Device ID (for detecting moved files)
    pub device: u64,
    /// Inode number (Unix) or file index (Windows)
    pub inode: u64,
    /// Content hash (for detecting content changes with same timestamp)
    pub content_hash: u64,
    /// Git blob object ID (if available)
    pub git_blob_id: Option<String>,
    /// Last scan timestamp
    pub scanned_at: u64,
    /// Scan results (cached)
    pub cached_results: Option<CachedScanResults>,
}

/// Cached scan results to avoid re-analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedScanResults {
    /// Language detection result
    pub language: u8, // Packed language enum
    /// File type classification
    pub file_type: u8,
    /// Line count
    pub line_count: u32,
    /// Character count
    pub char_count: u32,
    /// Token estimate
    pub token_estimate: u16,
    /// Binary detection result
    pub is_binary: bool,
    /// Content analysis version (to invalidate on algorithm changes)
    pub analysis_version: u32,
}

/// Statistics about the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestStats {
    /// Total files tracked
    pub total_files: u64,
    /// Files with cached results
    pub cached_files: u64,
    /// Total bytes tracked
    pub total_bytes: u64,
    /// Manifest size on disk
    pub manifest_size_bytes: u64,
    /// Last full scan duration (seconds)
    pub last_scan_duration_secs: f64,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
}

/// Configuration for incremental scanning
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Manifest file name
    pub manifest_name: String,
    /// Maximum manifest age before full rescan (hours)
    pub max_manifest_age_hours: u64,
    /// Content analysis version (increment to invalidate caches)
    pub analysis_version: u32,
    /// Whether to calculate content hashes
    pub enable_content_hashing: bool,
    /// Hash chunk size for large files
    pub hash_chunk_size: usize,
    /// Maximum file size to hash (bytes)
    pub max_hash_file_size: u64,
    /// Git integration enabled
    pub git_integration: bool,
}

/// Runtime scanning metrics
#[derive(Debug, Default)]
pub struct ScanMetrics {
    /// Files scanned (new analysis)
    pub files_scanned: u64,
    /// Files loaded from cache
    pub files_cached: u64,
    /// Files updated (metadata changed)
    pub files_updated: u64,
    /// Files removed (no longer exist)
    pub files_removed: u64,
    /// Time spent on I/O operations (microseconds)
    pub io_time_us: u64,
    /// Time spent on content hashing (microseconds)
    pub hash_time_us: u64,
    /// Cache hit rate for this session
    pub session_cache_hit_rate: f64,
}

/// File change detection result
#[derive(Debug, Clone, PartialEq)]
pub enum FileChange {
    /// File is unchanged (use cached data)
    Unchanged,
    /// File content changed
    ContentChanged,
    /// File metadata changed (size, modified time)
    MetadataChanged,
    /// New file (not in manifest)
    NewFile,
    /// File was moved/renamed
    Moved(String), // old path
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            manifest_name: ".scribe-manifest".to_string(),
            max_manifest_age_hours: 24,
            analysis_version: 1,
            enable_content_hashing: true,
            hash_chunk_size: 8192,
            max_hash_file_size: 10 * 1024 * 1024, // 10MB
            git_integration: true,
        }
    }
}

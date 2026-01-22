//! # scribe-cache
//!
//! Persistent caching layer for scribe computations.
//!
//! This crate provides:
//! - Content-hash based caching for file computations
//! - Granular cache invalidation based on file changes
//! - In-memory hot cache with disk persistence
//! - Cache versioning for format migrations
//!
//! ## Example
//!
//! ```no_run
//! use scribe_cache::{ScribeCache, CachedFileData};
//! use std::path::Path;
//!
//! let cache = ScribeCache::open(Path::new("/path/to/repo")).unwrap();
//!
//! // Check what files changed
//! let files = vec![/* discovered files */];
//! let diff = cache.diff_files(&files);
//!
//! // Process only changed files
//! for changed_file in &diff.changed {
//!     // Access path, content, and hash from the ChangedFile struct
//!     let path = &changed_file.path;
//!     let hash = changed_file.hash;
//!     // ... compute expensive data ...
//!     // cache.store_file_data(hash, &data);
//! }
//! ```

pub mod error;
pub mod invalidation;
pub mod keys;
pub mod store;
pub mod tables;
pub mod version;

pub use error::{CacheError, CacheResult};
pub use invalidation::{ChangedFile, FileDiff};
pub use keys::ContentHash;
pub use store::ScribeCache;
pub use version::CACHE_VERSION;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Cached data for a single file, keyed by content hash
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedFileData {
    /// Estimated token count for the file
    pub token_count: usize,
    /// Extracted symbols (function names, class names, etc.)
    pub symbols: Vec<String>,
    /// Import statements / dependencies
    pub imports: Vec<String>,
    /// Detected programming language
    pub language: String,
    /// File size in bytes
    pub size: u64,
}

impl CachedFileData {
    pub fn new(language: String, size: u64) -> Self {
        Self {
            token_count: 0,
            symbols: Vec::new(),
            imports: Vec::new(),
            language,
            size,
        }
    }
}

/// Cached graph-level data for a repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedGraphData {
    /// PageRank scores per file
    pub pagerank: Vec<(PathBuf, f64)>,
    /// Centrality scores per file
    pub centrality: Vec<(PathBuf, f64)>,
    /// Hash of the edge set for invalidation
    pub edges_hash: u64,
}

/// Metadata about the cache itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Cache format version
    pub version: u32,
    /// Repository identifier
    pub repo_id: String,
    /// When the cache was created
    pub created_at: u64,
    /// When the cache was last updated
    pub updated_at: u64,
}

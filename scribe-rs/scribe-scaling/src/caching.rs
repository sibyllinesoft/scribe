//! Intelligent caching system with persistent storage.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Configuration for caching system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether to enable persistent caching
    pub enable_persistent_cache: bool,
    
    /// Size of in-memory cache
    pub memory_cache_size: usize,
    
    /// Whether to enable compression for cached data
    pub compression_enabled: bool,
    
    /// Directory for cache storage (None = use default)
    pub cache_dir: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_persistent_cache: true,
            memory_cache_size: 1000,
            compression_enabled: false,
            cache_dir: None,
        }
    }
}
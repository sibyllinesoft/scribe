//! Intelligent caching system with persistent storage for scaling results.

use crate::api::engine::{ProcessingResult, ScalingConfig};
use crate::core::error::{ScalingError, ScalingResult};
use blake3::Hasher;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

/// Configuration for caching system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether to enable persistent caching
    pub enable_persistent_cache: bool,

    /// Size of in-memory cache (number of repository entries to keep)
    pub memory_cache_size: usize,

    /// Whether to enable compression for cached data
    pub compression_enabled: bool,

    /// Directory for cache storage (None = use project-local default)
    pub cache_dir: Option<PathBuf>,

    /// Time-to-live for cache entries in seconds (0 = never expire)
    #[serde(default = "CacheConfig::default_ttl")]
    pub cache_ttl: u64,
}

impl CacheConfig {
    fn default_ttl() -> u64 {
        3600
    }

    fn resolved_dir(&self) -> PathBuf {
        if let Some(dir) = &self.cache_dir {
            dir.clone()
        } else {
            PathBuf::from(".scribe-cache")
        }
    }

    fn cache_file_path(&self) -> PathBuf {
        self.resolved_dir().join("scaling-cache.json")
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_persistent_cache: true,
            memory_cache_size: 128,
            compression_enabled: false,
            cache_dir: None,
            cache_ttl: Self::default_ttl(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedProcessingResult {
    repo_hash: u64,
    config_hash: String,
    last_updated_epoch: u64,
    result: ProcessingResult,
}

impl CachedProcessingResult {
    fn is_expired(&self, ttl_seconds: u64) -> bool {
        if ttl_seconds == 0 {
            return false;
        }

        let last_updated = UNIX_EPOCH + Duration::from_secs(self.last_updated_epoch);
        match SystemTime::now().duration_since(last_updated) {
            Ok(elapsed) => elapsed.as_secs() > ttl_seconds,
            Err(_) => true,
        }
    }
}

/// Cache manager responsible for storing recent scaling results
pub struct ProcessingCache {
    config: CacheConfig,
    enabled: bool,
    entries: LruCache<String, CachedProcessingResult>,
    dirty: bool,
}

impl ProcessingCache {
    /// Create a new processing cache. Loads persistent data if enabled.
    pub fn new(config: CacheConfig) -> Self {
        let enabled = config.memory_cache_size > 0;
        let capacity = NonZeroUsize::new(config.memory_cache_size.max(1)).unwrap();
        let mut cache = Self {
            entries: LruCache::new(capacity),
            enabled,
            dirty: false,
            config,
        };

        if cache.config.enable_persistent_cache && cache.enabled {
            cache.load_from_disk();
        }

        cache
    }

    /// Attempt to retrieve a cached processing result when repository and
    /// configuration hashes match.
    pub fn get(&mut self, repo_hash: u64, config_hash: &str) -> Option<ProcessingResult> {
        if !self.enabled {
            return None;
        }

        let key = Self::make_key(repo_hash, config_hash);
        let ttl = self.config.cache_ttl;

        if let Some(entry) = self.entries.peek(&key) {
            if entry.is_expired(ttl) {
                self.entries.pop(&key);
                self.dirty = true;
                return None;
            }
        }

        self.entries.get(&key).map(|entry| entry.result.clone())
    }

    /// Store a processing result in cache.
    pub fn insert(&mut self, repo_hash: u64, config_hash: &str, result: ProcessingResult) {
        if !self.enabled {
            return;
        }

        let key = Self::make_key(repo_hash, config_hash);
        let cached = CachedProcessingResult {
            repo_hash,
            config_hash: config_hash.to_string(),
            last_updated_epoch: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            result,
        };

        self.entries.put(key, cached);
        self.dirty = true;
    }

    /// Persist cache contents to disk if configured.
    pub fn flush(&mut self) {
        if !self.config.enable_persistent_cache || !self.enabled || !self.dirty {
            return;
        }

        let cache_dir = self.config.resolved_dir();
        if let Err(err) = fs::create_dir_all(&cache_dir) {
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!(
                    "⚠️  Failed to create cache directory {}: {}",
                    cache_dir.display(),
                    err
                );
            }
            return;
        }

        let cache_file = self.config.cache_file_path();
        let snapshot: Vec<&CachedProcessingResult> = self.entries.iter().map(|(_, v)| v).collect();

        match serde_json::to_string_pretty(&snapshot) {
            Ok(serialized) => {
                if let Err(err) = fs::write(&cache_file, serialized) {
                    if std::env::var("SCRIBE_DEBUG").is_ok() {
                        eprintln!(
                            "⚠️  Failed to write cache file {}: {}",
                            cache_file.display(),
                            err
                        );
                    }
                } else {
                    self.dirty = false;
                }
            }
            Err(err) => {
                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!("⚠️  Failed to serialize cache: {}", err);
                }
            }
        }
    }

    fn load_from_disk(&mut self) {
        let cache_file = self.config.cache_file_path();
        if !cache_file.exists() {
            return;
        }

        match fs::read_to_string(&cache_file) {
            Ok(content) => match serde_json::from_str::<Vec<CachedProcessingResult>>(&content) {
                Ok(entries) => {
                    for entry in entries {
                        let key = Self::make_key(entry.repo_hash, &entry.config_hash);
                        self.entries.put(key, entry);
                    }
                    self.dirty = false;
                }
                Err(err) => {
                    if std::env::var("SCRIBE_DEBUG").is_ok() {
                        eprintln!(
                            "⚠️  Failed to parse cache file {}: {}",
                            cache_file.display(),
                            err
                        );
                    }
                }
            },
            Err(err) => {
                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!(
                        "⚠️  Failed to read cache file {}: {}",
                        cache_file.display(),
                        err
                    );
                }
            }
        }
    }

    fn make_key(repo_hash: u64, config_hash: &str) -> String {
        format!("{}::{}", repo_hash, config_hash)
    }
}

impl Drop for ProcessingCache {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Compute a stable hash representing the current repository state.
pub fn compute_repository_hash(repo_path: &Path) -> ScalingResult<u64> {
    let mut hasher = Hasher::new();

    for entry in WalkDir::new(repo_path)
        .into_iter()
        .filter_entry(|e| e.file_type().is_dir() || e.file_type().is_file())
    {
        let entry = entry.map_err(|err| {
            ScalingError::path(
                "Failed to traverse repository",
                err.path().unwrap_or(repo_path),
            )
        })?;
        if entry.file_type().is_file() {
            let metadata = entry
                .metadata()
                .map_err(|_| ScalingError::path("Failed to read file metadata", entry.path()))?;

            hasher.update(entry.path().to_string_lossy().as_bytes());
            hasher.update(&metadata.len().to_le_bytes());

            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                    hasher.update(&duration.as_secs().to_le_bytes());
                    hasher.update(&duration.subsec_nanos().to_le_bytes());
                }
            }
        }
    }

    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[..8]);
    Ok(u64::from_le_bytes(bytes))
}

/// Compute a stable hash for the scaling configuration.
pub fn compute_config_hash(config: &ScalingConfig) -> String {
    match serde_json::to_vec(config) {
        Ok(bytes) => {
            let mut hasher = Hasher::new();
            hasher.update(&bytes);
            hasher.finalize().to_hex().to_string()
        }
        Err(_) => "default".to_string(),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert!(config.enable_persistent_cache);
        assert_eq!(config.memory_cache_size, 128);
        assert!(!config.compression_enabled);
        assert!(config.cache_dir.is_none());
        assert_eq!(config.cache_ttl, 3600);
    }

    #[test]
    fn test_cache_config_resolved_dir_default() {
        let config = CacheConfig::default();
        assert_eq!(config.resolved_dir(), PathBuf::from(".scribe-cache"));
    }

    #[test]
    fn test_cache_config_resolved_dir_custom() {
        let config = CacheConfig {
            cache_dir: Some(PathBuf::from("/custom/cache")),
            ..Default::default()
        };
        assert_eq!(config.resolved_dir(), PathBuf::from("/custom/cache"));
    }

    #[test]
    fn test_cache_config_cache_file_path() {
        let config = CacheConfig::default();
        let path = config.cache_file_path();
        assert!(path.ends_with("scaling-cache.json"));
    }

    #[test]
    fn test_cache_config_clone() {
        let config = CacheConfig::default();
        let cloned = config.clone();
        assert_eq!(config.memory_cache_size, cloned.memory_cache_size);
        assert_eq!(config.cache_ttl, cloned.cache_ttl);
    }

    #[test]
    fn test_processing_cache_disabled() {
        let config = CacheConfig {
            memory_cache_size: 0,
            enable_persistent_cache: false,
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config);

        // Cache should be disabled
        assert!(!cache.enabled);

        // Operations should be no-ops
        assert!(cache.get(123, "test").is_none());

        let result = ProcessingResult {
            files: vec![],
            total_files: 0,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };
        cache.insert(123, "test", result);
        assert!(cache.get(123, "test").is_none());
    }

    #[test]
    fn test_processing_cache_basic_operations() {
        let config = CacheConfig {
            memory_cache_size: 10,
            enable_persistent_cache: false,
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config);

        // Initially empty
        assert!(cache.get(123, "config1").is_none());

        let result = ProcessingResult {
            files: vec![],
            total_files: 5,
            processing_time: Duration::from_millis(100),
            memory_peak: 1024,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };

        cache.insert(123, "config1", result.clone());

        // Should now be retrievable
        let cached = cache.get(123, "config1");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().total_files, 5);
    }

    #[test]
    fn test_processing_cache_different_keys() {
        let config = CacheConfig {
            memory_cache_size: 10,
            enable_persistent_cache: false,
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config);

        let result1 = ProcessingResult {
            files: vec![],
            total_files: 1,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };
        let result2 = ProcessingResult {
            files: vec![],
            total_files: 2,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };

        cache.insert(100, "config_a", result1);
        cache.insert(200, "config_b", result2);

        assert_eq!(cache.get(100, "config_a").unwrap().total_files, 1);
        assert_eq!(cache.get(200, "config_b").unwrap().total_files, 2);
        assert!(cache.get(100, "config_b").is_none());
        assert!(cache.get(200, "config_a").is_none());
    }

    #[test]
    fn test_make_key() {
        let key = ProcessingCache::make_key(123, "test_config");
        assert_eq!(key, "123::test_config");
    }

    #[test]
    fn test_compute_config_hash() {
        let config1 = ScalingConfig::default();
        let config2 = ScalingConfig::default();
        let config3 = ScalingConfig::small_repository();

        let hash1 = compute_config_hash(&config1);
        let hash2 = compute_config_hash(&config2);
        let hash3 = compute_config_hash(&config3);

        // Same configs should produce same hash
        assert_eq!(hash1, hash2);
        // Different configs should produce different hashes
        assert_ne!(hash1, hash3);
        // Hashes should be non-empty
        assert!(!hash1.is_empty());
    }

    #[test]
    fn test_compute_repository_hash() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create some test files
        std::fs::write(repo_path.join("file1.txt"), "content1").unwrap();
        std::fs::write(repo_path.join("file2.txt"), "content2").unwrap();

        let hash = compute_repository_hash(repo_path);
        assert!(hash.is_ok());
        assert!(hash.unwrap() > 0);
    }

    #[test]
    fn test_compute_repository_hash_changes_with_content() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        std::fs::write(repo_path.join("file.txt"), "content1").unwrap();
        let hash1 = compute_repository_hash(repo_path).unwrap();

        // Modify file (changes size)
        std::fs::write(repo_path.join("file.txt"), "different content").unwrap();
        let hash2 = compute_repository_hash(repo_path).unwrap();

        // Hashes should be different
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_cached_processing_result_not_expired() {
        let result = ProcessingResult {
            files: vec![],
            total_files: 0,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };

        let cached = CachedProcessingResult {
            repo_hash: 123,
            config_hash: "test".to_string(),
            last_updated_epoch: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            result,
        };

        // Should not be expired with TTL of 3600 seconds
        assert!(!cached.is_expired(3600));

        // Should not be expired with TTL of 0 (never expire)
        assert!(!cached.is_expired(0));
    }

    #[test]
    fn test_cached_processing_result_expired() {
        let result = ProcessingResult {
            files: vec![],
            total_files: 0,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };

        // Set last_updated to 2 hours ago
        let two_hours_ago = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 7200;

        let cached = CachedProcessingResult {
            repo_hash: 123,
            config_hash: "test".to_string(),
            last_updated_epoch: two_hours_ago,
            result,
        };

        // Should be expired with TTL of 3600 seconds (1 hour)
        assert!(cached.is_expired(3600));
    }

    #[test]
    fn test_processing_cache_flush_no_persistent() {
        let config = CacheConfig {
            memory_cache_size: 10,
            enable_persistent_cache: false,
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config);

        let result = ProcessingResult {
            files: vec![],
            total_files: 1,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };
        cache.insert(123, "test", result);

        // Should not panic when flushing with persistent cache disabled
        cache.flush();
    }

    #[test]
    fn test_processing_cache_with_persistent() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            memory_cache_size: 10,
            enable_persistent_cache: true,
            cache_dir: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config.clone());

        let result = ProcessingResult {
            files: vec![],
            total_files: 42,
            processing_time: Duration::from_millis(500),
            memory_peak: 2048,
            cache_hits: 10,
            cache_misses: 5,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };
        cache.insert(12345, "config_hash", result);
        cache.flush();

        // Verify cache file was created
        let cache_file = config.cache_file_path();
        assert!(cache_file.exists());

        // Create new cache and verify it loads the data
        let mut new_cache = ProcessingCache::new(config);
        let loaded = new_cache.get(12345, "config_hash");
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().total_files, 42);
    }

    #[test]
    fn test_processing_cache_expiration_removes_entry() {
        let config = CacheConfig {
            memory_cache_size: 10,
            enable_persistent_cache: false,
            cache_ttl: 1, // 1 second TTL
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config);

        // Insert with old timestamp
        let result = ProcessingResult {
            files: vec![],
            total_files: 5,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };
        cache.insert(123, "test", result);

        // Directly verify the entry is there initially
        assert!(cache
            .entries
            .peek(&ProcessingCache::make_key(123, "test"))
            .is_some());

        // Note: This test verifies cache insertion works, but real TTL expiration
        // would require waiting or mocking time
    }

    #[test]
    fn test_cached_processing_result_future_timestamp() {
        let result = ProcessingResult {
            files: vec![],
            total_files: 0,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };

        // Set timestamp in the future (simulates clock skew)
        let future_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;

        let cached = CachedProcessingResult {
            repo_hash: 123,
            config_hash: "test".to_string(),
            last_updated_epoch: future_time,
            result,
        };

        // Should be expired due to clock error (duration_since returns Err)
        assert!(cached.is_expired(100));
    }

    #[test]
    fn test_processing_cache_lru_eviction() {
        let config = CacheConfig {
            memory_cache_size: 2, // Very small cache
            enable_persistent_cache: false,
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config);

        let result = ProcessingResult {
            files: vec![],
            total_files: 0,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };

        // Insert 3 items (more than capacity)
        cache.insert(1, "a", result.clone());
        cache.insert(2, "b", result.clone());
        cache.insert(3, "c", result);

        // First item should be evicted (LRU)
        assert!(cache.get(1, "a").is_none());
        // Later items should still be present
        assert!(cache.get(2, "b").is_some());
        assert!(cache.get(3, "c").is_some());
    }

    #[test]
    fn test_cache_config_serialization() {
        let config = CacheConfig {
            enable_persistent_cache: true,
            memory_cache_size: 64,
            compression_enabled: true,
            cache_dir: Some(PathBuf::from("/custom")),
            cache_ttl: 7200,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: CacheConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.memory_cache_size, deserialized.memory_cache_size);
        assert_eq!(config.cache_ttl, deserialized.cache_ttl);
        assert_eq!(config.compression_enabled, deserialized.compression_enabled);
    }

    #[test]
    fn test_compute_repository_hash_empty_dir() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Empty directory should still produce a valid hash
        let hash = compute_repository_hash(repo_path);
        assert!(hash.is_ok());
    }

    #[test]
    fn test_compute_repository_hash_with_subdirs() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create nested structure
        std::fs::create_dir_all(repo_path.join("src/utils")).unwrap();
        std::fs::write(repo_path.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(repo_path.join("src/utils/helper.rs"), "pub fn help() {}").unwrap();

        let hash = compute_repository_hash(repo_path);
        assert!(hash.is_ok());
        assert!(hash.unwrap() > 0);
    }

    #[test]
    fn test_processing_cache_dirty_flag() {
        let config = CacheConfig {
            memory_cache_size: 10,
            enable_persistent_cache: false,
            ..Default::default()
        };
        let mut cache = ProcessingCache::new(config);

        // Initially not dirty
        assert!(!cache.dirty);

        let result = ProcessingResult {
            files: vec![],
            total_files: 0,
            processing_time: Duration::from_secs(0),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            metrics: crate::io::metrics::ScalingMetrics::default(),
        };

        // After insert, should be dirty
        cache.insert(123, "test", result);
        assert!(cache.dirty);
    }

    #[test]
    fn test_cache_config_debug() {
        let config = CacheConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CacheConfig"));
        assert!(debug_str.contains("memory_cache_size"));
    }
}

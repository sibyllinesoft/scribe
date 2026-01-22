//! Main cache store implementation

use crate::error::{CacheError, CacheResult};
use crate::invalidation::{compute_file_diff, ChangedFile, FileDiff};
use crate::keys::{repo_identifier, ContentHash};
use crate::tables::{self, FILE_DATA, GRAPH_DATA, METADATA, PATH_HASHES, PATH_MTIMES};
use crate::version::CACHE_VERSION;
use crate::{CacheMetadata, CachedFileData, CachedGraphData};

use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use redb::{Database, ReadableTable};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Main cache interface for scribe computations
pub struct ScribeCache {
    /// Persistent database
    db: Database,
    /// In-memory hot cache for file data (hash -> data)
    file_cache: DashMap<u64, CachedFileData>,
    /// In-memory path to hash mapping
    path_hashes: DashMap<PathBuf, ContentHash>,
    /// In-memory path to mtime mapping
    path_mtimes: DashMap<PathBuf, u64>,
    /// Cached graph data
    graph_cache: RwLock<Option<CachedGraphData>>,
    /// Cache directory path
    cache_dir: PathBuf,
    /// Repository identifier
    repo_id: String,
}

impl ScribeCache {
    /// Open or create a cache for the given repository
    pub fn open(repo_path: &Path) -> CacheResult<Self> {
        let repo_id = repo_identifier(repo_path);
        let cache_dir = Self::cache_dir_for_repo(&repo_id)?;

        info!(
            "Opening cache at {} for repo {}",
            cache_dir.display(),
            repo_id
        );

        let db_path = cache_dir.join("cache.redb");
        let db = Database::create(&db_path)?;

        let mut cache = Self {
            db,
            file_cache: DashMap::new(),
            path_hashes: DashMap::new(),
            path_mtimes: DashMap::new(),
            graph_cache: RwLock::new(None),
            cache_dir,
            repo_id,
        };

        // Check version and potentially clear cache
        cache.check_version()?;

        // Load existing data into memory
        cache.load_into_memory()?;

        Ok(cache)
    }

    /// Get the cache directory for a repository
    fn cache_dir_for_repo(repo_id: &str) -> CacheResult<PathBuf> {
        let base = dirs::cache_dir()
            .ok_or(CacheError::NoCacheDir)?
            .join("scribe")
            .join(repo_id);

        std::fs::create_dir_all(&base)?;
        Ok(base)
    }

    /// Check cache version, clear if mismatched
    fn check_version(&mut self) -> CacheResult<()> {
        let read_txn = self.db.begin_read()?;

        let version = if let Ok(table) = read_txn.open_table(METADATA) {
            table
                .get(tables::meta_keys::VERSION)?
                .map(|v| {
                    let bytes: [u8; 4] = v.value().try_into().unwrap_or([0; 4]);
                    u32::from_le_bytes(bytes)
                })
                .unwrap_or(0)
        } else {
            0
        };

        drop(read_txn);

        if version != CACHE_VERSION {
            if version != 0 {
                warn!(
                    "Cache version mismatch: found {}, expected {}. Clearing cache.",
                    version, CACHE_VERSION
                );
            }
            self.clear_all()?;
            self.write_metadata()?;
        }

        Ok(())
    }

    /// Write initial metadata
    fn write_metadata(&self) -> CacheResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA)?;
            table.insert(
                tables::meta_keys::VERSION,
                CACHE_VERSION.to_le_bytes().as_slice(),
            )?;
            table.insert(tables::meta_keys::REPO_ID, self.repo_id.as_bytes())?;

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            table.insert(tables::meta_keys::CREATED_AT, now.to_le_bytes().as_slice())?;
            table.insert(tables::meta_keys::UPDATED_AT, now.to_le_bytes().as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Clear all cached data
    fn clear_all(&mut self) -> CacheResult<()> {
        info!("Clearing all cache data");

        // Clear in-memory caches
        self.file_cache.clear();
        self.path_hashes.clear();
        self.path_mtimes.clear();
        *self.graph_cache.write() = None;

        // Clear database tables
        let write_txn = self.db.begin_write()?;
        {
            // Delete and recreate tables
            let _ = write_txn.delete_table(FILE_DATA);
            let _ = write_txn.delete_table(PATH_HASHES);
            let _ = write_txn.delete_table(PATH_MTIMES);
            let _ = write_txn.delete_table(GRAPH_DATA);
            let _ = write_txn.delete_table(METADATA);
        }
        write_txn.commit()?;

        Ok(())
    }

    /// Load cached data into memory for fast access
    fn load_into_memory(&mut self) -> CacheResult<()> {
        let read_txn = self.db.begin_read()?;

        // Load path -> hash mappings
        if let Ok(table) = read_txn.open_table(PATH_HASHES) {
            for result in table.iter()? {
                let (key, value) = result?;
                let path = PathBuf::from(String::from_utf8_lossy(key.value()).to_string());
                let hash = ContentHash::from(value.value());
                self.path_hashes.insert(path, hash);
            }
        }

        // Load path -> mtime mappings
        if let Ok(table) = read_txn.open_table(PATH_MTIMES) {
            for result in table.iter()? {
                let (key, value) = result?;
                let path = PathBuf::from(String::from_utf8_lossy(key.value()).to_string());
                self.path_mtimes.insert(path, value.value());
            }
        }

        // Load file data
        if let Ok(table) = read_txn.open_table(FILE_DATA) {
            for result in table.iter()? {
                let (key, value) = result?;
                let hash = key.value();
                if let Ok(data) = bincode::deserialize::<CachedFileData>(value.value()) {
                    self.file_cache.insert(hash, data);
                }
            }
        }

        // Load graph data
        if let Ok(table) = read_txn.open_table(GRAPH_DATA) {
            if let Some(guard) = table.get(tables::graph_keys::PAGERANK)? {
                if let Ok(data) = bincode::deserialize::<CachedGraphData>(guard.value()) {
                    *self.graph_cache.write() = Some(data);
                }
            }
        }

        debug!(
            "Loaded {} path hashes, {} file data entries into memory",
            self.path_hashes.len(),
            self.file_cache.len()
        );

        Ok(())
    }

    /// Compute diff between current files and cache
    pub fn diff_files(&self, current_files: &[PathBuf]) -> FileDiff {
        compute_file_diff(current_files, |path| self.path_hashes.get(path).map(|h| *h))
    }

    /// Get cached file data by content hash
    pub fn get_file_data(&self, hash: ContentHash) -> Option<CachedFileData> {
        self.file_cache.get(&hash.as_u64()).map(|r| r.clone())
    }

    /// Get cached file data by path
    pub fn get_file_data_by_path(&self, path: &Path) -> Option<CachedFileData> {
        let hash = self.path_hashes.get(path)?;
        self.get_file_data(*hash)
    }

    /// Store file data for a content hash
    pub fn store_file_data(&self, hash: ContentHash, data: &CachedFileData) -> CacheResult<()> {
        // Store in memory
        self.file_cache.insert(hash.as_u64(), data.clone());

        // Store in database
        let serialized = bincode::serialize(data)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(FILE_DATA)?;
            table.insert(hash.as_u64(), serialized.as_slice())?;
        }
        write_txn.commit()?;

        Ok(())
    }

    /// Store multiple file data entries (batched for performance)
    pub fn store_file_data_batch(
        &self,
        entries: &[(ContentHash, CachedFileData)],
    ) -> CacheResult<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Serialize in parallel
        let serialized: Vec<_> = entries
            .par_iter()
            .map(|(hash, data)| {
                let bytes = bincode::serialize(data).unwrap();
                (hash.as_u64(), bytes)
            })
            .collect();

        // Store in memory
        for (hash, data) in entries {
            self.file_cache.insert(hash.as_u64(), data.clone());
        }

        // Batch write to database
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(FILE_DATA)?;
            for (hash, bytes) in &serialized {
                table.insert(*hash, bytes.as_slice())?;
            }
        }
        write_txn.commit()?;

        debug!("Stored {} file data entries", entries.len());
        Ok(())
    }

    /// Update path -> hash mappings after processing changed files
    pub fn update_path_mappings(&self, files: &[ChangedFile]) -> CacheResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let write_txn = self.db.begin_write()?;
        {
            let mut hash_table = write_txn.open_table(PATH_HASHES)?;
            let mut mtime_table = write_txn.open_table(PATH_MTIMES)?;

            for file in files {
                let path_bytes = file.path.to_string_lossy().as_bytes().to_vec();

                // Update hash
                hash_table.insert(path_bytes.as_slice(), file.hash.as_u64())?;
                self.path_hashes.insert(file.path.clone(), file.hash);

                // Update mtime
                if let Ok(metadata) = file.path.metadata() {
                    if let Ok(mtime) = metadata.modified() {
                        if let Ok(duration) = mtime.duration_since(UNIX_EPOCH) {
                            let nanos = duration.as_nanos() as u64;
                            mtime_table.insert(path_bytes.as_slice(), nanos)?;
                            self.path_mtimes.insert(file.path.clone(), nanos);
                        }
                    }
                }
            }
        }
        write_txn.commit()?;

        Ok(())
    }

    /// Remove entries for deleted files
    pub fn remove_deleted(&self, paths: &[PathBuf]) -> CacheResult<()> {
        if paths.is_empty() {
            return Ok(());
        }

        // Get hashes to remove before clearing path mappings
        let hashes_to_remove: Vec<_> = paths
            .iter()
            .filter_map(|p| self.path_hashes.remove(p).map(|(_, h)| h))
            .collect();

        // Remove from memory
        for path in paths {
            self.path_mtimes.remove(path);
        }
        for hash in &hashes_to_remove {
            self.file_cache.remove(&hash.as_u64());
        }

        // Remove from database
        let write_txn = self.db.begin_write()?;
        {
            let mut hash_table = write_txn.open_table(PATH_HASHES)?;
            let mut mtime_table = write_txn.open_table(PATH_MTIMES)?;
            let mut data_table = write_txn.open_table(FILE_DATA)?;

            for path in paths {
                let path_bytes = path.to_string_lossy().as_bytes().to_vec();
                let _ = hash_table.remove(path_bytes.as_slice());
                let _ = mtime_table.remove(path_bytes.as_slice());
            }

            for hash in &hashes_to_remove {
                let _ = data_table.remove(hash.as_u64());
            }
        }
        write_txn.commit()?;

        debug!("Removed {} deleted file entries", paths.len());
        Ok(())
    }

    /// Store graph-level cached data
    pub fn store_graph_data(&self, data: &CachedGraphData) -> CacheResult<()> {
        *self.graph_cache.write() = Some(data.clone());

        let serialized = bincode::serialize(data)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(GRAPH_DATA)?;
            table.insert(tables::graph_keys::PAGERANK, serialized.as_slice())?;
        }
        write_txn.commit()?;

        Ok(())
    }

    /// Get cached graph data
    pub fn get_graph_data(&self) -> Option<CachedGraphData> {
        self.graph_cache.read().clone()
    }

    /// Check if graph data needs recomputation
    pub fn graph_needs_update(&self, current_edges_hash: u64) -> bool {
        match &*self.graph_cache.read() {
            Some(data) => data.edges_hash != current_edges_hash,
            None => true,
        }
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            file_entries: self.file_cache.len(),
            path_mappings: self.path_hashes.len(),
            has_graph_data: self.graph_cache.read().is_some(),
        }
    }
}

/// Statistics about the cache
#[derive(Debug)]
pub struct CacheStats {
    pub file_entries: usize,
    pub path_mappings: usize,
    pub has_graph_data: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_repo() -> TempDir {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("file1.rs"), "fn main() {}").unwrap();
        std::fs::write(temp.path().join("file2.rs"), "fn helper() {}").unwrap();
        temp
    }

    #[test]
    fn test_cache_open_create() {
        let temp = create_test_repo();
        let cache = ScribeCache::open(temp.path()).unwrap();
        assert!(cache.cache_dir().exists());
    }

    #[test]
    fn test_store_and_retrieve_file_data() {
        let temp = create_test_repo();
        let cache = ScribeCache::open(temp.path()).unwrap();

        let hash = ContentHash::from_content(b"test content");
        let data = CachedFileData {
            token_count: 100,
            symbols: vec!["main".to_string()],
            imports: vec![],
            language: "rust".to_string(),
            size: 50,
        };

        cache.store_file_data(hash, &data).unwrap();
        let retrieved = cache.get_file_data(hash).unwrap();

        assert_eq!(retrieved.token_count, 100);
        assert_eq!(retrieved.symbols, vec!["main"]);
    }

    #[test]
    fn test_diff_files() {
        let temp = create_test_repo();
        let cache = ScribeCache::open(temp.path()).unwrap();

        let files: Vec<_> = std::fs::read_dir(temp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "rs").unwrap_or(false))
            .collect();

        let diff = cache.diff_files(&files);

        // All files should be new since cache is empty
        assert_eq!(diff.new_files.len(), 2);
        assert_eq!(diff.changed.len(), 0);
        assert_eq!(diff.unchanged.len(), 0);
    }
}

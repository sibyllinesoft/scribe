//! Incremental scanning with persistent file manifest and smart caching.
//!
//! This module implements an incremental scanning system that maintains a
//! persistent manifest of file metadata to avoid re-scanning unchanged files,
//! dramatically improving performance for large repositories on subsequent runs.

use crate::perf::compact_data::{CompactFileCollection, PackedFileType, PackedLanguage};
use bincode;
use fxhash::{FxHashMap, FxHashSet};
use scribe_core::{FileInfo, FileWeight, Language, RenderDecision, Result, ScribeError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use xxhash_rust::xxh3::xxh3_64;

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

/// Incremental scanner with persistent caching
#[derive(Debug)]
pub struct IncrementalScanner {
    config: IncrementalConfig,
    manifest: Option<FileManifest>,
    manifest_path: PathBuf,
    repo_root: PathBuf,
    metrics: ScanMetrics,
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

impl IncrementalScanner {
    /// Create a new incremental scanner
    pub async fn new<P: AsRef<Path>>(repo_root: P, config: IncrementalConfig) -> Result<Self> {
        let repo_root = repo_root.as_ref().to_path_buf();
        let manifest_path = repo_root.join(&config.manifest_name);

        let mut scanner = Self {
            config,
            manifest: None,
            manifest_path,
            repo_root,
            metrics: ScanMetrics::default(),
        };

        // Try to load existing manifest
        scanner.load_manifest().await?;

        Ok(scanner)
    }

    /// Perform incremental scan of the repository
    pub async fn scan_incremental(&mut self) -> Result<CompactFileCollection> {
        let start_time = std::time::Instant::now();

        log::info!("Starting incremental scan of {}", self.repo_root.display());

        // Initialize or validate manifest
        let manifest = match &self.manifest {
            Some(manifest) => {
                if self.should_full_rescan(manifest).await? {
                    log::info!("Manifest expired or invalid, performing full rescan");
                    self.create_new_manifest().await?
                } else {
                    manifest.clone()
                }
            }
            None => {
                log::info!("No manifest found, performing initial scan");
                self.create_new_manifest().await?
            }
        };

        // Discover current files
        let current_files = self.discover_files().await?;
        log::debug!("Discovered {} files in repository", current_files.len());

        // Detect changes
        let changes = self.detect_changes(&manifest, &current_files).await?;
        log::info!("Detected {} changes", changes.len());

        // Process changes and build result collection
        let mut collection = CompactFileCollection::new();
        let mut new_manifest = manifest.clone();
        new_manifest.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for (path, change) in &changes {
            match change {
                FileChange::Unchanged => {
                    // Use cached data
                    if let Some(entry) = manifest.entries.get(path) {
                        if let Some(cached) = &entry.cached_results {
                            let cached_info = self.file_info_from_cache(entry, cached)?;
                            collection.add_file(&cached_info);
                            self.metrics.files_cached += 1;
                        }
                    }
                }

                FileChange::NewFile | FileChange::ContentChanged | FileChange::MetadataChanged => {
                    // Scan the file
                    let file_path = self.repo_root.join(path);
                    let SingleFileScanResult {
                        manifest_entry,
                        file_info,
                    } = self.scan_single_file(&file_path).await?;

                    collection.add_file(&file_info);
                    new_manifest.entries.insert(path.clone(), manifest_entry);

                    match change {
                        FileChange::NewFile => self.metrics.files_scanned += 1,
                        _ => self.metrics.files_updated += 1,
                    }
                }

                FileChange::Moved(old_path) => {
                    // Handle moved file
                    if let Some(old_entry) = new_manifest.entries.remove(old_path) {
                        let mut new_entry = old_entry;
                        new_entry.path = path.clone();
                        new_entry.scanned_at = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs();

                        new_manifest.entries.insert(path.clone(), new_entry);
                        self.metrics.files_updated += 1;
                    }
                }
            }
        }

        // Remove deleted files from manifest
        let current_paths: FxHashSet<_> = current_files
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        new_manifest.entries.retain(|path, _| {
            if current_paths.contains(path) {
                true
            } else {
                self.metrics.files_removed += 1;
                false
            }
        });

        // Update manifest statistics
        self.update_manifest_stats(&mut new_manifest, start_time.elapsed());

        // Save updated manifest
        self.save_manifest(&new_manifest).await?;
        self.manifest = Some(new_manifest);

        // Calculate session metrics
        let total_files =
            self.metrics.files_cached + self.metrics.files_scanned + self.metrics.files_updated;
        if total_files > 0 {
            self.metrics.session_cache_hit_rate =
                self.metrics.files_cached as f64 / total_files as f64;
        }

        log::info!(
            "Incremental scan completed in {:.2}s: {}/{} files cached ({:.1}% hit rate)",
            start_time.elapsed().as_secs_f64(),
            self.metrics.files_cached,
            total_files,
            self.metrics.session_cache_hit_rate * 100.0
        );

        Ok(collection)
    }

    /// Load existing manifest from disk
    async fn load_manifest(&mut self) -> Result<()> {
        match fs::read(&self.manifest_path).await {
            Ok(data) => {
                match bincode::deserialize::<FileManifest>(&data) {
                    Ok(manifest) => {
                        // Validate manifest
                        if manifest.repo_root == self.repo_root.to_string_lossy() {
                            log::debug!("Loaded manifest with {} entries", manifest.entries.len());
                            self.manifest = Some(manifest);
                        } else {
                            log::warn!("Manifest repository root mismatch, ignoring");
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to deserialize manifest: {}", e);
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                log::debug!("No existing manifest found");
            }
            Err(e) => {
                log::warn!("Failed to read manifest: {}", e);
            }
        }

        Ok(())
    }

    /// Save manifest to disk
    async fn save_manifest(&self, manifest: &FileManifest) -> Result<()> {
        let data = bincode::serialize(manifest)
            .map_err(|e| ScribeError::io(format!("Failed to serialize manifest: {}", e)))?;

        // Write atomically using temporary file
        let temp_path = self.manifest_path.with_extension("tmp");
        fs::write(&temp_path, &data)
            .await
            .map_err(|e| ScribeError::io(format!("Failed to write manifest: {}", e)))?;

        fs::rename(&temp_path, &self.manifest_path)
            .await
            .map_err(|e| ScribeError::io(format!("Failed to finalize manifest: {}", e)))?;

        log::debug!("Saved manifest with {} bytes", data.len());
        Ok(())
    }

    /// Check if a full rescan is needed
    async fn should_full_rescan(&self, manifest: &FileManifest) -> Result<bool> {
        // Check manifest age
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let age_hours = (now - manifest.updated_at) / 3600;

        if age_hours > self.config.max_manifest_age_hours {
            return Ok(true);
        }

        // Check analysis version
        if manifest.entries.values().any(|entry| {
            entry
                .cached_results
                .as_ref()
                .map(|r| r.analysis_version != self.config.analysis_version)
                .unwrap_or(true)
        }) {
            return Ok(true);
        }

        // Check git commit (if enabled)
        if self.config.git_integration {
            if let Ok(current_commit) = self.get_current_git_commit().await {
                if manifest.git_commit.as_ref() != Some(&current_commit) {
                    log::info!("Git commit changed, full rescan needed");
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Create a new empty manifest
    async fn create_new_manifest(&self) -> Result<FileManifest> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let git_commit = if self.config.git_integration {
            self.get_current_git_commit().await.ok()
        } else {
            None
        };

        Ok(FileManifest {
            version: 1,
            created_at: now,
            updated_at: now,
            repo_root: self.repo_root.to_string_lossy().to_string(),
            git_commit,
            entries: FxHashMap::default(),
            stats: ManifestStats {
                total_files: 0,
                cached_files: 0,
                total_bytes: 0,
                manifest_size_bytes: 0,
                last_scan_duration_secs: 0.0,
                cache_hit_rate: 0.0,
            },
        })
    }

    /// Discover all files in the repository
    async fn discover_files(&self) -> Result<Vec<PathBuf>> {
        // Use ignore crate for efficient traversal
        use ignore::{DirEntry, WalkBuilder, WalkState};

        let mut builder = WalkBuilder::new(&self.repo_root);
        builder
            .git_ignore(true)
            .git_exclude(true)
            .hidden(false)
            .follow_links(false);

        let mut files = Vec::new();

        builder.build().for_each(|entry| match entry {
            Ok(entry) => {
                if entry.file_type().map_or(false, |ft| ft.is_file()) {
                    if let Ok(relative) = entry.path().strip_prefix(&self.repo_root) {
                        files.push(relative.to_path_buf());
                    }
                }
            }
            Err(err) => {
                log::debug!("Walk error: {}", err);
            }
        });

        Ok(files)
    }

    /// Detect changes between manifest and current files
    async fn detect_changes(
        &self,
        manifest: &FileManifest,
        current_files: &[PathBuf],
    ) -> Result<FxHashMap<String, FileChange>> {
        let mut changes = FxHashMap::default();
        let mut processed_inodes = FxHashSet::default();

        for file_path in current_files {
            let path_str = file_path.to_string_lossy().to_string();
            let full_path = self.repo_root.join(file_path);

            let metadata = match fs::metadata(&full_path).await {
                Ok(metadata) => metadata,
                Err(_) => {
                    // File disappeared between discovery and check
                    continue;
                }
            };

            let change = if let Some(entry) = manifest.entries.get(&path_str) {
                self.detect_file_change(entry, &metadata, &full_path)
                    .await?
            } else {
                // Check if this might be a moved file
                let inode = self.get_inode(&metadata);
                let device = self.get_device(&metadata);

                if let Some(old_entry) = manifest.entries.values().find(|e| {
                    e.inode == inode
                        && e.device == device
                        && e.size == metadata.len()
                        && !processed_inodes.contains(&inode)
                }) {
                    processed_inodes.insert(inode);
                    FileChange::Moved(old_entry.path.clone())
                } else {
                    FileChange::NewFile
                }
            };

            changes.insert(path_str, change);
        }

        Ok(changes)
    }

    /// Detect changes to a single file
    async fn detect_file_change(
        &self,
        entry: &ManifestEntry,
        metadata: &Metadata,
        file_path: &Path,
    ) -> Result<FileChange> {
        let modified = metadata
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Quick checks first
        if metadata.len() != entry.size {
            return Ok(FileChange::ContentChanged);
        }

        if modified != entry.modified {
            // Modified time changed, check content hash if enabled
            if self.config.enable_content_hashing {
                let content_hash = self.calculate_file_hash(file_path).await?;
                if content_hash != entry.content_hash {
                    return Ok(FileChange::ContentChanged);
                }
            }
            return Ok(FileChange::MetadataChanged);
        }

        // Check inode (file system level)
        let inode = self.get_inode(metadata);
        if inode != entry.inode {
            return Ok(FileChange::ContentChanged);
        }

        // Check if analysis is outdated
        if let Some(cached) = &entry.cached_results {
            if cached.analysis_version != self.config.analysis_version {
                return Ok(FileChange::ContentChanged);
            }
        }

        Ok(FileChange::Unchanged)
    }

    /// Scan a single file and create manifest entry
    async fn scan_single_file(&mut self, file_path: &Path) -> Result<SingleFileScanResult> {
        use crate::analysis::language_detection::LanguageDetector;

        let io_start = std::time::Instant::now();
        let metadata = fs::metadata(file_path).await?;
        self.metrics.io_time_us += io_start.elapsed().as_micros() as u64;

        let relative_path = file_path
            .strip_prefix(&self.repo_root)
            .unwrap()
            .to_string_lossy()
            .to_string();

        // Calculate content hash if enabled
        let hash_start = std::time::Instant::now();
        let content_hash = if self.config.enable_content_hashing {
            self.calculate_file_hash(file_path).await?
        } else {
            0
        };
        self.metrics.hash_time_us += hash_start.elapsed().as_micros() as u64;

        // Analyze file content
        let language_detector = LanguageDetector::new();
        let language = language_detector.detect_language(file_path);

        let raw_bytes = fs::read(file_path).await?;
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_string();
        let is_binary = FileInfo::detect_binary_from_bytes(&raw_bytes, Some(extension.as_str()));
        let (line_count, char_count, token_estimate) = if is_binary {
            (0u32, 0u32, 0u16)
        } else {
            let content = String::from_utf8_lossy(&raw_bytes);
            let lines = content.lines().count() as u32;
            let chars = content.chars().count() as u32;
            let tokens = scribe_core::FileInfo::estimate_tokens_with_path(
                content.as_ref(),
                file_path,
            );
            let tokens_u32 = tokens.min(u32::MAX as usize) as u32;
            (
                lines,
                chars,
                tokens_u32.min(u16::MAX as u32) as u16,
            )
        };

        let mut file_type = if is_binary {
            scribe_core::FileType::Binary
        } else {
            scribe_core::FileInfo::classify_file_type_with_binary(
                &relative_path,
                &language,
                extension.as_str(),
                is_binary,
            )
        };

        // If classification yielded generated for binary scenario adjust accordingly
        if is_binary {
            file_type = scribe_core::FileType::Binary;
        }

        let packed_file_type = PackedFileType::from_file_type(&file_type);

        let cached_results = CachedScanResults {
            language: PackedLanguage::from(language.clone()).as_u8(),
            file_type: packed_file_type.as_u8(),
            line_count,
            char_count,
            token_estimate,
            is_binary,
            analysis_version: self.config.analysis_version,
        };

        let file_info = FileInfo {
            path: file_path.to_path_buf(),
            relative_path: relative_path.clone(),
            size: metadata.len(),
            modified: metadata.modified().ok(),
            decision: RenderDecision::include("scanned"),
            file_type,
            language: language.clone(),
            content: None,
            token_estimate: Some(token_estimate as usize),
            line_count: Some(line_count as usize),
            char_count: Some(char_count as usize),
            is_binary,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let manifest_entry = ManifestEntry {
            path: relative_path.clone(),
            size: metadata.len(),
            modified: metadata
                .modified()
                .unwrap_or(UNIX_EPOCH)
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            device: self.get_device(&metadata),
            inode: self.get_inode(&metadata),
            content_hash,
            git_blob_id: None, // Would need git integration
            scanned_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cached_results: Some(cached_results),
        };

        Ok(SingleFileScanResult {
            manifest_entry,
            file_info,
        })
    }

    fn file_info_from_cache(
        &self,
        entry: &ManifestEntry,
        cached: &CachedScanResults,
    ) -> Result<FileInfo> {
        let absolute_path = self.repo_root.join(&entry.path);
        let packed_language = PackedLanguage::from_u8(cached.language);
        let language = Language::from(packed_language);
        let extension = std::path::Path::new(&entry.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        let packed_file_type = PackedFileType::from_u8(cached.file_type);
        let cached_binary = cached.is_binary || matches!(packed_file_type, PackedFileType::Binary);

        let file_type = FileInfo::classify_file_type_with_binary(
            &entry.path,
            &language,
            extension,
            cached_binary,
        );

        let is_binary = cached_binary || matches!(file_type, scribe_core::FileType::Binary);

        let modified = if entry.modified > 0 {
            Some(UNIX_EPOCH + Duration::from_secs(entry.modified))
        } else {
            None
        };

        Ok(FileInfo {
            path: absolute_path,
            relative_path: entry.path.clone(),
            size: entry.size,
            modified,
            decision: RenderDecision::include("cached"),
            file_type,
            language,
            content: None,
            token_estimate: Some(cached.token_estimate as usize),
            line_count: Some(cached.line_count as usize),
            char_count: Some(cached.char_count as usize),
            is_binary,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        })
    }

    /// Calculate file hash for content change detection
    async fn calculate_file_hash(&self, file_path: &Path) -> Result<u64> {
        let metadata = fs::metadata(file_path).await?;

        // Skip large files
        if metadata.len() > self.config.max_hash_file_size {
            return Ok(0); // Use 0 to indicate "not hashed"
        }

        let mut file = fs::File::open(file_path).await?;
        let mut hasher = xxhash_rust::xxh3::Xxh3::new();
        let mut buffer = vec![0u8; self.config.hash_chunk_size];

        use tokio::io::AsyncReadExt;

        loop {
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hasher.digest())
    }

    /// Get current git commit hash
    async fn get_current_git_commit(&self) -> Result<String> {
        use tokio::process::Command;

        let output = Command::new("git")
            .arg("rev-parse")
            .arg("HEAD")
            .current_dir(&self.repo_root)
            .output()
            .await?;

        if output.status.success() {
            let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
            Ok(commit)
        } else {
            Err(ScribeError::git("Failed to get git commit".to_string()))
        }
    }

    /// Get file inode number (platform specific)
    fn get_inode(&self, metadata: &Metadata) -> u64 {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            metadata.ino()
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            metadata.file_index().unwrap_or(0)
        }
        #[cfg(not(any(unix, windows)))]
        {
            0 // Fallback
        }
    }

    /// Get device ID (platform specific)
    fn get_device(&self, metadata: &Metadata) -> u64 {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            metadata.dev()
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            metadata.volume_serial_number().unwrap_or(0) as u64
        }
        #[cfg(not(any(unix, windows)))]
        {
            0 // Fallback
        }
    }

    /// Update manifest statistics
    fn update_manifest_stats(&self, manifest: &mut FileManifest, scan_duration: Duration) {
        manifest.stats.total_files = manifest.entries.len() as u64;
        manifest.stats.cached_files = manifest
            .entries
            .values()
            .filter(|e| e.cached_results.is_some())
            .count() as u64;
        manifest.stats.total_bytes = manifest.entries.values().map(|e| e.size).sum();
        manifest.stats.last_scan_duration_secs = scan_duration.as_secs_f64();

        if manifest.stats.total_files > 0 {
            manifest.stats.cache_hit_rate =
                manifest.stats.cached_files as f64 / manifest.stats.total_files as f64;
        }
    }

    /// Get scanning metrics
    pub fn metrics(&self) -> &ScanMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = ScanMetrics::default();
    }
}

/// Result of scanning a single file
struct SingleFileScanResult {
    manifest_entry: ManifestEntry,
    file_info: scribe_core::FileInfo,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    async fn create_test_repo() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Create some test files
        fs::write(root.join("main.rs"), "fn main() {}")
            .await
            .unwrap();
        fs::write(root.join("lib.rs"), "pub fn hello() {}")
            .await
            .unwrap();

        // Create subdirectory
        fs::create_dir(root.join("src")).await.unwrap();
        fs::write(root.join("src/module.rs"), "mod test;")
            .await
            .unwrap();

        temp_dir
    }

    #[tokio::test]
    async fn test_incremental_scanner_creation() {
        let temp_dir = create_test_repo().await;
        let config = IncrementalConfig::default();

        let scanner = IncrementalScanner::new(temp_dir.path(), config).await;
        assert!(scanner.is_ok());
    }

    #[tokio::test]
    async fn test_file_discovery() {
        let temp_dir = create_test_repo().await;
        let config = IncrementalConfig::default();
        let scanner = IncrementalScanner::new(temp_dir.path(), config)
            .await
            .unwrap();

        let files = scanner.discover_files().await.unwrap();
        assert!(files.len() >= 3); // main.rs, lib.rs, src/module.rs

        let file_names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert!(file_names.contains(&"main.rs"));
        assert!(file_names.contains(&"lib.rs"));
        assert!(file_names.contains(&"module.rs"));
    }

    #[tokio::test]
    async fn test_manifest_serialization() {
        let manifest = FileManifest {
            version: 1,
            created_at: 1640995200,
            updated_at: 1640995200,
            repo_root: "/test/repo".to_string(),
            git_commit: Some("abcdef123456".to_string()),
            entries: FxHashMap::default(),
            stats: ManifestStats {
                total_files: 0,
                cached_files: 0,
                total_bytes: 0,
                manifest_size_bytes: 0,
                last_scan_duration_secs: 0.0,
                cache_hit_rate: 0.0,
            },
        };

        let serialized = bincode::serialize(&manifest).unwrap();
        let deserialized: FileManifest = bincode::deserialize(&serialized).unwrap();

        assert_eq!(manifest.version, deserialized.version);
        assert_eq!(manifest.repo_root, deserialized.repo_root);
        assert_eq!(manifest.git_commit, deserialized.git_commit);
    }

    #[tokio::test]
    async fn test_content_hashing() {
        let temp_dir = create_test_repo().await;
        let config = IncrementalConfig {
            enable_content_hashing: true,
            max_hash_file_size: 1024,
            hash_chunk_size: 256,
            ..Default::default()
        };
        let mut scanner = IncrementalScanner::new(temp_dir.path(), config)
            .await
            .unwrap();

        let test_file = temp_dir.path().join("main.rs");
        let hash1 = scanner.calculate_file_hash(&test_file).await.unwrap();

        // Hash should be consistent
        let hash2 = scanner.calculate_file_hash(&test_file).await.unwrap();
        assert_eq!(hash1, hash2);

        // Modify file and check hash changes
        fs::write(&test_file, "fn main() { println!(\"modified\"); }")
            .await
            .unwrap();
        let hash3 = scanner.calculate_file_hash(&test_file).await.unwrap();
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_file_change_detection() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let temp_dir = create_test_repo().await;
        let config = IncrementalConfig::default();
        let scanner = IncrementalScanner::new(temp_dir.path(), config)
            .await
            .unwrap();

        let test_file = temp_dir.path().join("main.rs");
        let metadata = fs::metadata(&test_file).await.unwrap();

        // Create manifest entry
        let entry = ManifestEntry {
            path: "main.rs".to_string(),
            size: metadata.len(),
            modified: metadata
                .modified()
                .unwrap_or(UNIX_EPOCH)
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            device: scanner.get_device(&metadata),
            inode: scanner.get_inode(&metadata),
            content_hash: 0,
            git_blob_id: None,
            scanned_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cached_results: None,
        };

        // Unchanged file
        let change = scanner
            .detect_file_change(&entry, &metadata, &test_file)
            .await
            .unwrap();
        assert_eq!(change, FileChange::Unchanged);

        // Modify file
        fs::write(&test_file, "fn main() { println!(\"changed\"); }")
            .await
            .unwrap();
        let new_metadata = fs::metadata(&test_file).await.unwrap();
        let change = scanner
            .detect_file_change(&entry, &new_metadata, &test_file)
            .await
            .unwrap();
        assert_eq!(change, FileChange::ContentChanged);
    }

    #[tokio::test]
    async fn test_manifest_persistence() {
        let temp_dir = create_test_repo().await;
        let config = IncrementalConfig::default();
        let mut scanner = IncrementalScanner::new(temp_dir.path(), config)
            .await
            .unwrap();

        // Create a manifest
        let manifest = scanner.create_new_manifest().await.unwrap();
        scanner.save_manifest(&manifest).await.unwrap();

        // Load manifest in new scanner instance
        let config2 = IncrementalConfig::default();
        let mut scanner2 = IncrementalScanner::new(temp_dir.path(), config2)
            .await
            .unwrap();

        assert!(scanner2.manifest.is_some());
        let loaded_manifest = scanner2.manifest.unwrap();
        assert_eq!(loaded_manifest.repo_root, manifest.repo_root);
        assert_eq!(loaded_manifest.version, manifest.version);
    }
}

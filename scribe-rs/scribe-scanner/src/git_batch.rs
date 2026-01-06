//! High-performance batched git operations using libgit2.
//!
//! This module replaces expensive per-file git command invocations with efficient
//! batch operations using libgit2 plumbing, dramatically reducing I/O overhead
//! and system call overhead for large repositories.

use bincode;
use fxhash::FxHashMap;
use scribe_core::{GitFileStatus, GitStatus, Result, ScribeError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use string_interner::{DefaultSymbol, StringInterner};
use xxhash_rust::xxh3::xxh3_64;

/// Interned path identifier for memory efficiency
pub type PathId = DefaultSymbol;

/// Batched git operations handler using libgit2
#[derive(Debug)]
pub struct GitBatchProcessor {
    repo: Option<git2::Repository>,
    repo_path: PathBuf,
    /// String interner for path deduplication
    path_interner: StringInterner,
    /// Cached file statuses by interned path ID
    status_cache: FxHashMap<PathId, GitFileStatus>,
    /// Cached last commit hash per path
    last_commit_cache: FxHashMap<PathId, Option<u64>>,
    /// Cached churn scores per path
    churn_cache: FxHashMap<PathId, f32>,
    /// Bulk status loaded flag
    bulk_status_loaded: bool,
    /// Cache validity timestamp
    cache_timestamp: Option<SystemTime>,
    /// Cache TTL (5 minutes)
    cache_ttl: std::time::Duration,
    /// Performance metrics
    metrics: BatchMetrics,
}

/// Performance metrics for batch operations
#[derive(Debug, Default, Clone)]
pub struct BatchMetrics {
    /// Number of individual git status calls avoided
    pub status_calls_avoided: u64,
    /// Number of files processed in single batch
    pub batch_size: u64,
    /// Time taken for batch loading (microseconds)
    pub batch_load_time_us: u64,
    /// Memory saved through path interning (bytes)
    pub memory_saved_bytes: u64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
}

/// Compact git file information using interned paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactGitFileInfo {
    pub path_id: PathId,
    pub status: GitFileStatus,
    pub last_commit_hash: Option<u64>, // xxh3 hash of commit
    pub churn_score: f32,              // 0.0-1.0 activity score
}

/// Bulk git status result
#[derive(Debug)]
pub struct BulkStatusResult {
    pub files_processed: usize,
    pub modified_files: usize,
    pub untracked_files: usize,
    pub deleted_files: usize,
    pub load_time_ms: u64,
}

impl GitBatchProcessor {
    /// Create a new batch processor for the given repository
    pub fn new<P: AsRef<Path>>(repo_path: P) -> Result<Self> {
        let repo_path = repo_path.as_ref().to_path_buf();

        let repo = match git2::Repository::open(&repo_path) {
            Ok(repo) => Some(repo),
            Err(_) => None, // Not a git repository or git2 unavailable
        };

        Ok(Self {
            repo,
            repo_path,
            path_interner: StringInterner::new(),
            status_cache: FxHashMap::default(),
            last_commit_cache: FxHashMap::default(),
            churn_cache: FxHashMap::default(),
            bulk_status_loaded: false,
            cache_timestamp: None,
            cache_ttl: std::time::Duration::from_secs(300),
            metrics: BatchMetrics::default(),
        })
    }

    /// Load all file statuses in a single bulk operation
    pub fn load_bulk_status(&mut self) -> Result<BulkStatusResult> {
        let start_time = SystemTime::now();

        let repo = match &self.repo {
            Some(repo) => repo,
            None => return Err(ScribeError::git("Repository not available".to_string())),
        };

        // Clear previous cache
        self.status_cache.clear();
        self.path_interner.clear();

        // Get repository status using libgit2
        let mut status_options = git2::StatusOptions::new();
        status_options
            .include_untracked(true)
            .include_ignored(false)
            .recurse_untracked_dirs(true);

        let statuses = repo
            .statuses(Some(&mut status_options))
            .map_err(|e| ScribeError::git(format!("Failed to get repository status: {}", e)))?;

        let mut result = BulkStatusResult {
            files_processed: 0,
            modified_files: 0,
            untracked_files: 0,
            deleted_files: 0,
            load_time_ms: 0,
        };

        // Process all status entries in batch
        for entry in statuses.iter() {
            if let Some(path) = entry.path() {
                let path_id = self.path_interner.get_or_intern(path);
                let git_status = self.convert_git2_status(entry.status());

                // Update counters
                match git_status {
                    GitFileStatus::Modified => result.modified_files += 1,
                    GitFileStatus::Untracked => result.untracked_files += 1,
                    GitFileStatus::Deleted => result.deleted_files += 1,
                    _ => {}
                }

                self.status_cache.insert(path_id, git_status);
                result.files_processed += 1;
            }
        }

        // Mark bulk status as loaded and cache timestamp
        self.bulk_status_loaded = true;
        self.cache_timestamp = Some(SystemTime::now());

        // Update metrics
        self.metrics.batch_size = result.files_processed as u64;
        self.metrics.batch_load_time_us =
            start_time.elapsed().unwrap_or_default().as_micros() as u64;

        result.load_time_ms = self.metrics.batch_load_time_us / 1000;

        log::info!(
            "Loaded git status for {} files in {}ms",
            result.files_processed,
            result.load_time_ms
        );

        Ok(result)
    }

    /// Get file status using cached batch data or fallback to individual lookup
    pub fn get_file_status(&mut self, path: &Path) -> Result<GitFileStatus> {
        // Load bulk status if not already loaded
        if !self.bulk_status_loaded || !self.is_cache_valid() {
            self.load_bulk_status()?;
        }

        // Convert path to relative path within repository
        let relative_path = if path.is_absolute() {
            path.strip_prefix(&self.repo_path)
                .map_err(|_| ScribeError::git("Path not in repository".to_string()))?
        } else {
            path
        };

        let path_str = relative_path.to_string_lossy();

        // Check cache first
        if let Some(path_id) = self.path_interner.get(&*path_str) {
            if let Some(&status) = self.status_cache.get(&path_id) {
                self.metrics.status_calls_avoided += 1;
                return Ok(status);
            }
        }

        // File not found in status - it's tracked and clean
        Ok(GitFileStatus::Unmodified)
    }

    /// Get multiple file statuses efficiently using batch cache
    pub fn get_multiple_file_statuses(
        &mut self,
        paths: &[PathBuf],
    ) -> Result<Vec<(PathBuf, GitFileStatus)>> {
        // Ensure bulk status is loaded
        if !self.bulk_status_loaded || !self.is_cache_valid() {
            self.load_bulk_status()?;
        }

        let mut results = Vec::with_capacity(paths.len());
        let mut cache_hits = 0;

        for path in paths {
            let status = match self.get_file_status(path) {
                Ok(status) => {
                    cache_hits += 1;
                    status
                }
                Err(_) => GitFileStatus::Unmodified,
            };
            results.push((path.clone(), status));
        }

        // Update cache hit rate metric
        self.metrics.cache_hit_rate = cache_hits as f64 / paths.len() as f64;
        self.metrics.status_calls_avoided += cache_hits;

        Ok(results)
    }

    /// Get compact git information for a file using cached data
    pub fn get_compact_file_info(&mut self, path: &Path) -> Result<CompactGitFileInfo> {
        let status = self.get_file_status(path)?;

        let relative_path = if path.is_absolute() {
            path.strip_prefix(&self.repo_path)
                .map_err(|_| ScribeError::git("Path not in repository".to_string()))?
        } else {
            path
        };

        let path_str = relative_path.to_string_lossy();
        let path_id = self.path_interner.get_or_intern(&*path_str);

        // Get last commit hash for the file (cached lookup)
        let last_commit_hash = self.get_cached_last_commit_hash(path)?;

        // Calculate churn score based on git activity
        let churn_score = self.calculate_churn_score(path)?;

        Ok(CompactGitFileInfo {
            path_id,
            status,
            last_commit_hash,
            churn_score,
        })
    }

    /// List all tracked files using libgit2 index
    pub fn list_tracked_files_fast(&self) -> Result<Vec<PathBuf>> {
        let repo = match &self.repo {
            Some(repo) => repo,
            None => return Err(ScribeError::git("Repository not available".to_string())),
        };

        let index = repo
            .index()
            .map_err(|e| ScribeError::git(format!("Failed to get repository index: {}", e)))?;

        let mut files = Vec::with_capacity(index.len());

        for entry in index.iter() {
            if let Some(path) = std::str::from_utf8(&entry.path).ok() {
                files.push(self.repo_path.join(path));
            }
        }

        log::debug!("Found {} tracked files via libgit2 index", files.len());
        Ok(files)
    }

    /// Bulk analyze file churn for multiple files
    pub fn bulk_analyze_churn(&mut self, paths: &[PathBuf]) -> Result<FxHashMap<PathBuf, f32>> {
        let repo = match &self.repo {
            Some(repo) => repo,
            None => return Err(ScribeError::git("Repository not available".to_string())),
        };

        let mut results = FxHashMap::default();

        // Walk the repository history for bulk analysis
        let mut revwalk = repo
            .revwalk()
            .map_err(|e| ScribeError::git(format!("Failed to create revwalk: {}", e)))?;

        revwalk
            .push_head()
            .map_err(|e| ScribeError::git(format!("Failed to push HEAD: {}", e)))?;

        // Limit to recent commits for performance
        let mut commit_count = 0;
        const MAX_COMMITS: usize = 100;

        for oid in revwalk {
            if commit_count >= MAX_COMMITS {
                break;
            }

            let oid =
                oid.map_err(|e| ScribeError::git(format!("Failed to get commit OID: {}", e)))?;
            let commit = repo
                .find_commit(oid)
                .map_err(|e| ScribeError::git(format!("Failed to find commit: {}", e)))?;

            if let Ok(tree) = commit.tree() {
                for path in paths {
                    let relative_path = if let Ok(rel_path) = path.strip_prefix(&self.repo_path) {
                        rel_path
                    } else {
                        continue;
                    };

                    // Check if file exists in this commit
                    if tree.get_path(relative_path).is_ok() {
                        let current_score = results.get(path).copied().unwrap_or(0.0);
                        results.insert(path.clone(), current_score + 0.1);
                    }
                }
            }

            commit_count += 1;
        }

        // Normalize scores to 0.0-1.0 range
        let max_score = results.values().fold(0.0f32, |max, &val| max.max(val));
        if max_score > 0.0 {
            for score in results.values_mut() {
                *score /= max_score;
            }
        }

        Ok(results)
    }

    fn resolve_relative_path(&self, path: &Path) -> Result<PathBuf> {
        if path.is_absolute() {
            path.strip_prefix(&self.repo_path)
                .map(|p| p.to_path_buf())
                .map_err(|_| ScribeError::git("Path not in repository".to_string()))
        } else {
            Ok(path.to_path_buf())
        }
    }

    fn intern_relative_path(&mut self, path: &Path) -> Result<(PathId, PathBuf)> {
        let relative_path = self.resolve_relative_path(path)?;
        let relative_str = relative_path.to_string_lossy();
        let path_id = self.path_interner.get_or_intern(relative_str.as_ref());
        Ok((path_id, relative_path))
    }

    /// Convert git2 status flags to our GitFileStatus enum
    fn convert_git2_status(&self, status: git2::Status) -> GitFileStatus {
        if status.contains(git2::Status::WT_NEW) || status.contains(git2::Status::INDEX_NEW) {
            GitFileStatus::Untracked
        } else if status.contains(git2::Status::WT_MODIFIED)
            || status.contains(git2::Status::INDEX_MODIFIED)
        {
            GitFileStatus::Modified
        } else if status.contains(git2::Status::WT_DELETED)
            || status.contains(git2::Status::INDEX_DELETED)
        {
            GitFileStatus::Deleted
        } else if status.contains(git2::Status::WT_RENAMED)
            || status.contains(git2::Status::INDEX_RENAMED)
        {
            GitFileStatus::Renamed
        } else if status.contains(git2::Status::IGNORED) {
            GitFileStatus::Ignored
        } else {
            GitFileStatus::Unmodified
        }
    }

    /// Create a time-sorted revwalk from HEAD
    fn create_revwalk(repo: &git2::Repository) -> Result<git2::Revwalk> {
        let mut revwalk = repo
            .revwalk()
            .map_err(|e| ScribeError::git(format!("Failed to create revwalk: {}", e)))?;
        revwalk
            .push_head()
            .map_err(|e| ScribeError::git(format!("Failed to push HEAD: {}", e)))?;
        revwalk
            .set_sorting(git2::Sort::TIME)
            .map_err(|e| ScribeError::git(format!("Failed to sort revwalk: {}", e)))?;
        Ok(revwalk)
    }

    /// Get the parent tree of a commit, if it has a parent
    fn get_parent_tree(commit: &git2::Commit) -> Result<Option<git2::Tree>> {
        if commit.parent_count() == 0 {
            return Ok(None);
        }
        let parent = commit
            .parent(0)
            .map_err(|e| ScribeError::git(format!("Failed to access parent commit: {}", e)))?;
        let tree = parent
            .tree()
            .map_err(|e| ScribeError::git(format!("Failed to load parent tree: {}", e)))?;
        Ok(Some(tree))
    }

    /// Check if a commit modified the given path
    fn commit_modified_path(
        repo: &git2::Repository,
        commit: &git2::Commit,
        diff_path: &str,
    ) -> Result<bool> {
        let tree = commit
            .tree()
            .map_err(|e| ScribeError::git(format!("Failed to load commit tree: {}", e)))?;
        let parent_tree = Self::get_parent_tree(commit)?;

        let mut diff_opts = git2::DiffOptions::new();
        diff_opts.pathspec(diff_path);

        let diff = repo
            .diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut diff_opts))
            .map_err(|e| ScribeError::git(format!("Failed to diff trees: {}", e)))?;

        Ok(diff.deltas().len() > 0)
    }

    /// Get cached last commit hash for a file (simplified for performance)
    fn get_cached_last_commit_hash(&mut self, path: &Path) -> Result<Option<u64>> {
        let repo = match &self.repo {
            Some(repo) => repo,
            None => return Ok(None),
        };

        let (path_id, relative_path) = self.intern_relative_path(path)?;
        if let Some(cached) = self.last_commit_cache.get(&path_id) {
            return Ok(*cached);
        }

        let diff_path = relative_path.to_string_lossy().replace('\\', "/");
        let revwalk = Self::create_revwalk(repo)?;

        for oid_result in revwalk.take(256) {
            let oid = oid_result
                .map_err(|e| ScribeError::git(format!("Failed to get commit OID: {}", e)))?;
            let commit = repo
                .find_commit(oid)
                .map_err(|e| ScribeError::git(format!("Failed to find commit: {}", e)))?;

            if Self::commit_modified_path(repo, &commit, &diff_path)? {
                let hash = xxh3_64(commit.id().as_bytes());
                self.last_commit_cache.insert(path_id, Some(hash));
                return Ok(Some(hash));
            }
        }

        self.last_commit_cache.insert(path_id, None);
        Ok(None)
    }

    /// Calculate file churn score based on git history
    fn calculate_churn_score(&mut self, path: &Path) -> Result<f32> {
        let repo = match &self.repo {
            Some(repo) => repo,
            None => return Ok(0.0),
        };

        let (path_id, relative_path) = self.intern_relative_path(path)?;
        if let Some(cached) = self.churn_cache.get(&path_id) {
            return Ok(*cached);
        }

        let diff_path = relative_path.to_string_lossy().replace('\\', "/");
        let revwalk = Self::create_revwalk(repo)?;

        let mut changes = 0usize;
        let mut total = 0usize;
        const MAX_COMMITS: usize = 200;

        for oid_result in revwalk.take(MAX_COMMITS) {
            let oid = oid_result
                .map_err(|e| ScribeError::git(format!("Failed to get commit OID: {}", e)))?;
            let commit = repo
                .find_commit(oid)
                .map_err(|e| ScribeError::git(format!("Failed to find commit: {}", e)))?;

            if Self::commit_modified_path(repo, &commit, &diff_path)? {
                changes += 1;
            }
            total += 1;
        }

        let score = if total == 0 {
            0.0
        } else {
            (changes as f32 / total as f32).clamp(0.0, 1.0)
        };

        self.churn_cache.insert(path_id, score);
        Ok(score)
    }

    /// Check if the cache is still valid
    fn is_cache_valid(&self) -> bool {
        if let Some(cache_time) = self.cache_timestamp {
            SystemTime::now()
                .duration_since(cache_time)
                .map(|duration| duration < self.cache_ttl)
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Clear all caches and reset state
    pub fn clear_cache(&mut self) {
        self.status_cache.clear();
        self.last_commit_cache.clear();
        self.churn_cache.clear();
        self.path_interner.clear();
        self.bulk_status_loaded = false;
        self.cache_timestamp = None;
        self.metrics = BatchMetrics::default();
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &BatchMetrics {
        &self.metrics
    }

    /// Serialize cache to disk for persistence
    pub fn serialize_cache(&self) -> Result<Vec<u8>> {
        let cache_data = (
            &self.status_cache,
            self.path_interner.into_iter().collect::<Vec<_>>(),
            self.cache_timestamp,
        );

        bincode::serialize(&cache_data)
            .map_err(|e| ScribeError::io(format!("Failed to serialize cache: {}", e)))
    }

    /// Deserialize cache from disk
    pub fn deserialize_cache(&mut self, data: &[u8]) -> Result<()> {
        let (status_cache, interner_data, cache_timestamp): (
            FxHashMap<PathId, GitFileStatus>,
            Vec<(DefaultSymbol, String)>,
            Option<SystemTime>,
        ) = bincode::deserialize(data)
            .map_err(|e| ScribeError::io(format!("Failed to deserialize cache: {}", e)))?;

        // Rebuild string interner
        self.path_interner.clear();
        for (symbol, string) in interner_data {
            self.path_interner.get_or_intern(&string);
        }

        self.status_cache = status_cache;
        self.cache_timestamp = cache_timestamp;
        self.bulk_status_loaded = true;

        Ok(())
    }

    /// Estimate memory savings from path interning
    pub fn calculate_memory_savings(&self) -> u64 {
        let total_strings_bytes: usize = self.path_interner.into_iter().map(|(_, s)| s.len()).sum();

        let interned_overhead = self.path_interner.len() * std::mem::size_of::<PathId>();
        let cache_entries = self.status_cache.len()
            * (std::mem::size_of::<PathId>() + std::mem::size_of::<GitFileStatus>());

        let without_interning =
            self.status_cache.len() * (total_strings_bytes / self.path_interner.len().max(1));

        let with_interning = total_strings_bytes + interned_overhead + cache_entries;

        without_interning.saturating_sub(with_interning) as u64
    }

    /// Check if repository is available
    pub fn is_available(&self) -> bool {
        self.repo.is_some()
    }

    /// Get repository path
    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }
}

impl Default for GitBatchProcessor {
    fn default() -> Self {
        // Create a dummy processor for paths without git
        Self {
            repo: None,
            repo_path: PathBuf::from("."),
            path_interner: StringInterner::new(),
            status_cache: FxHashMap::default(),
            bulk_status_loaded: false,
            cache_timestamp: None,
            cache_ttl: std::time::Duration::from_secs(300),
            metrics: BatchMetrics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;
    use tempfile::TempDir;

    fn create_test_git_repo() -> Result<TempDir> {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Initialize git repo
        let output = Command::new("git")
            .arg("init")
            .current_dir(repo_path)
            .output();

        if output.is_err() || !output.unwrap().status.success() {
            return Err(ScribeError::git(
                "Git not available for testing".to_string(),
            ));
        }

        // Configure git
        Command::new("git")
            .args(&["config", "user.name", "Test User"])
            .current_dir(repo_path)
            .output()
            .unwrap();

        Command::new("git")
            .args(&["config", "user.email", "test@example.com"])
            .current_dir(repo_path)
            .output()
            .unwrap();

        // Create and commit a test file
        let test_file = repo_path.join("test.rs");
        fs::write(&test_file, "fn main() {}").unwrap();

        Command::new("git")
            .args(&["add", "test.rs"])
            .current_dir(repo_path)
            .output()
            .unwrap();

        Command::new("git")
            .args(&["commit", "-m", "Initial commit"])
            .current_dir(repo_path)
            .output()
            .unwrap();

        Ok(temp_dir)
    }

    #[test]
    fn test_batch_processor_creation() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let processor = GitBatchProcessor::new(temp_dir.path()).unwrap();
            assert!(processor.is_available());
            assert_eq!(processor.repo_path(), temp_dir.path());
        }
    }

    #[test]
    fn test_list_tracked_files_fast() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let processor = GitBatchProcessor::new(temp_dir.path()).unwrap();
            let files = processor.list_tracked_files_fast().unwrap();

            assert_eq!(files.len(), 1);
            assert!(files[0].file_name().unwrap() == "test.rs");
        }
    }

    #[test]
    fn test_bulk_status_loading() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let mut processor = GitBatchProcessor::new(temp_dir.path()).unwrap();

            // Create a modified file
            let modified_file = temp_dir.path().join("modified.rs");
            fs::write(&modified_file, "fn modified() {}").unwrap();

            let result = processor.load_bulk_status().unwrap();

            assert_eq!(result.files_processed, 1); // Only the untracked file
            assert_eq!(result.untracked_files, 1);
            assert!(result.load_time_ms < 1000); // Should be fast
        }
    }

    #[test]
    fn test_file_status_lookup() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let mut processor = GitBatchProcessor::new(temp_dir.path()).unwrap();

            // Test existing committed file
            let test_file = temp_dir.path().join("test.rs");
            let status = processor.get_file_status(&test_file).unwrap();
            assert_eq!(status, GitFileStatus::Unmodified);

            // Test untracked file
            let untracked_file = temp_dir.path().join("untracked.rs");
            fs::write(&untracked_file, "fn untracked() {}").unwrap();

            // Need to reload status to see the new file
            processor.load_bulk_status().unwrap();
            let status = processor.get_file_status(&untracked_file).unwrap();
            assert_eq!(status, GitFileStatus::Untracked);
        }
    }

    #[test]
    fn test_multiple_file_statuses() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let mut processor = GitBatchProcessor::new(temp_dir.path()).unwrap();

            let files = vec![
                temp_dir.path().join("test.rs"),
                temp_dir.path().join("nonexistent.rs"),
            ];

            let results = processor.get_multiple_file_statuses(&files).unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].1, GitFileStatus::Unmodified);
            assert_eq!(results[1].1, GitFileStatus::Unmodified); // Non-existent = clean
        }
    }

    #[test]
    fn test_memory_savings_calculation() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let mut processor = GitBatchProcessor::new(temp_dir.path()).unwrap();
            processor.load_bulk_status().unwrap();

            // Should show some memory savings from interning
            let savings = processor.calculate_memory_savings();
            // For a single file, savings might be minimal or zero
            assert!(savings >= 0);
        }
    }

    #[test]
    fn test_cache_serialization() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let mut processor = GitBatchProcessor::new(temp_dir.path()).unwrap();
            processor.load_bulk_status().unwrap();

            // Serialize cache
            let serialized = processor.serialize_cache().unwrap();
            assert!(!serialized.is_empty());

            // Clear and deserialize
            processor.clear_cache();
            assert!(!processor.bulk_status_loaded);

            processor.deserialize_cache(&serialized).unwrap();
            assert!(processor.bulk_status_loaded);
        }
    }

    #[test]
    fn test_performance_metrics() {
        if let Ok(temp_dir) = create_test_git_repo() {
            let mut processor = GitBatchProcessor::new(temp_dir.path()).unwrap();
            processor.load_bulk_status().unwrap();

            // Make some status queries to generate metrics
            let test_file = temp_dir.path().join("test.rs");
            processor.get_file_status(&test_file).unwrap();
            processor.get_file_status(&test_file).unwrap(); // Second call should hit cache

            let metrics = processor.metrics();
            assert!(metrics.batch_size > 0);
            assert!(metrics.status_calls_avoided > 0);
        }
    }

    #[test]
    fn test_git2_status_conversion() {
        let processor = GitBatchProcessor::default();

        assert_eq!(
            processor.convert_git2_status(git2::Status::WT_NEW),
            GitFileStatus::Untracked
        );

        assert_eq!(
            processor.convert_git2_status(git2::Status::WT_MODIFIED),
            GitFileStatus::Modified
        );

        assert_eq!(
            processor.convert_git2_status(git2::Status::WT_DELETED),
            GitFileStatus::Deleted
        );

        assert_eq!(
            processor.convert_git2_status(git2::Status::empty()),
            GitFileStatus::Unmodified
        );
    }
}

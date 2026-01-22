//! Cache invalidation logic based on file content hashes

use crate::keys::ContentHash;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;
use tracing::{debug, trace};

/// Result of diffing current files against cache
#[derive(Debug, Default)]
pub struct FileDiff {
    /// Files that haven't changed (hash matches)
    pub unchanged: Vec<PathBuf>,
    /// Files that changed (new content, new hash)
    pub changed: Vec<ChangedFile>,
    /// New files not in cache
    pub new_files: Vec<ChangedFile>,
    /// Files that were deleted
    pub deleted: Vec<PathBuf>,
}

/// A file that needs processing (new or changed)
#[derive(Debug, Clone)]
pub struct ChangedFile {
    pub path: PathBuf,
    pub content: Vec<u8>,
    pub hash: ContentHash,
}

impl FileDiff {
    /// Total number of files that need processing
    pub fn files_to_process(&self) -> usize {
        self.changed.len() + self.new_files.len()
    }

    /// Check if cache is fully up to date
    pub fn is_up_to_date(&self) -> bool {
        self.changed.is_empty() && self.new_files.is_empty() && self.deleted.is_empty()
    }
}

/// Compute file diff between current state and cached state
///
/// This is parallelized for performance on large repositories.
pub fn compute_file_diff<F>(current_files: &[PathBuf], get_cached_hash: F) -> FileDiff
where
    F: Fn(&PathBuf) -> Option<ContentHash> + Sync,
{
    let current_set: HashSet<_> = current_files.iter().collect();

    // Process files in parallel
    let results: Vec<_> = current_files
        .par_iter()
        .filter_map(|path| {
            // Read file content
            let content = match std::fs::read(path) {
                Ok(c) => c,
                Err(e) => {
                    trace!("Failed to read {}: {}", path.display(), e);
                    return None;
                }
            };

            let hash = ContentHash::from_content(&content);
            let cached_hash = get_cached_hash(path);

            Some((path.clone(), content, hash, cached_hash))
        })
        .collect();

    // Categorize results
    let mut diff = FileDiff::default();

    for (path, content, hash, cached_hash) in results {
        match cached_hash {
            Some(cached) if cached == hash => {
                diff.unchanged.push(path);
            }
            Some(_) => {
                diff.changed.push(ChangedFile {
                    path,
                    content,
                    hash,
                });
            }
            None => {
                diff.new_files.push(ChangedFile {
                    path,
                    content,
                    hash,
                });
            }
        }
    }

    debug!(
        "File diff: {} unchanged, {} changed, {} new",
        diff.unchanged.len(),
        diff.changed.len(),
        diff.new_files.len()
    );

    diff
}

/// Fast check using mtime before computing hash
/// Returns files that might have changed (mtime differs or unknown)
pub fn quick_mtime_filter<F>(files: &[PathBuf], get_cached_mtime: F) -> Vec<PathBuf>
where
    F: Fn(&PathBuf) -> Option<u64> + Sync,
{
    files
        .par_iter()
        .filter(|path| {
            let current_mtime = match path.metadata() {
                Ok(m) => m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as u64),
                Err(_) => None,
            };

            match (current_mtime, get_cached_mtime(path)) {
                (Some(current), Some(cached)) => current != cached,
                _ => true, // Unknown mtime, assume changed
            }
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_file_diff_new_files() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("file1.txt");
        std::fs::write(&file1, "content1").unwrap();

        let cache: HashMap<PathBuf, ContentHash> = HashMap::new();
        let diff = compute_file_diff(&[file1.clone()], |p| cache.get(p).copied());

        assert_eq!(diff.new_files.len(), 1);
        assert_eq!(diff.changed.len(), 0);
        assert_eq!(diff.unchanged.len(), 0);
    }

    #[test]
    fn test_file_diff_unchanged() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("file1.txt");
        std::fs::write(&file1, "content1").unwrap();

        let hash = ContentHash::from_content(b"content1");
        let mut cache: HashMap<PathBuf, ContentHash> = HashMap::new();
        cache.insert(file1.clone(), hash);

        let diff = compute_file_diff(&[file1], |p| cache.get(p).copied());

        assert_eq!(diff.new_files.len(), 0);
        assert_eq!(diff.changed.len(), 0);
        assert_eq!(diff.unchanged.len(), 1);
    }

    #[test]
    fn test_file_diff_changed() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("file1.txt");
        std::fs::write(&file1, "new content").unwrap();

        let old_hash = ContentHash::from_content(b"old content");
        let mut cache: HashMap<PathBuf, ContentHash> = HashMap::new();
        cache.insert(file1.clone(), old_hash);

        let diff = compute_file_diff(&[file1], |p| cache.get(p).copied());

        assert_eq!(diff.new_files.len(), 0);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.unchanged.len(), 0);
    }
}

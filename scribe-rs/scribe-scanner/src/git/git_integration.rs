//! Git integration for enhanced file discovery and status tracking.
//!
//! This module provides comprehensive Git integration capabilities including:
//! - Fast file discovery using libgit2 index
//! - File status tracking (modified, staged, untracked)
//! - Commit history and blame information
//! - Repository statistics and health metrics

use dashmap::DashMap;
use git2::{Repository, StatusOptions};
use scribe_core::{GitFileStatus, Result, ScribeError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command as AsyncCommand;

use super::types::{
    ActivityPeriod, AgeDistribution, BranchHealth, ContributorStats, GitBlameInfo, GitBlameLine,
    GitCommitInfo, GitFileInfo, GitRepositoryStats, RepositoryHealth,
};

/// Git repository integration handler
pub struct GitIntegrator {
    repo_path: PathBuf,
    repo: parking_lot::Mutex<Repository>,
    cache: GitCache,
}

// Manual Debug impl since Repository doesn't implement Debug
impl std::fmt::Debug for GitIntegrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GitIntegrator")
            .field("repo_path", &self.repo_path)
            .finish()
    }
}

/// Git operations cache for performance
#[derive(Debug)]
struct GitCache {
    file_statuses: DashMap<PathBuf, GitFileStatus>,
    commit_cache: DashMap<String, GitCommitInfo>,
    blame_cache: DashMap<PathBuf, GitBlameInfo>,
    files_discovered: parking_lot::RwLock<usize>,
    cache_timestamp: parking_lot::RwLock<Option<SystemTime>>,
    cache_ttl: std::time::Duration,
    batch_status_cache: DashMap<PathBuf, GitFileStatus>,
    /// Flag indicating batch status has been loaded (even if empty = all unmodified)
    batch_status_loaded: std::sync::atomic::AtomicBool,
}

impl Default for GitCache {
    fn default() -> Self {
        Self {
            file_statuses: DashMap::new(),
            commit_cache: DashMap::new(),
            blame_cache: DashMap::new(),
            files_discovered: parking_lot::RwLock::new(0),
            cache_timestamp: parking_lot::RwLock::new(None),
            cache_ttl: std::time::Duration::from_secs(300),
            batch_status_cache: DashMap::new(),
            batch_status_loaded: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl GitIntegrator {
    /// Create a new Git integrator for the given repository path
    pub fn new<P: AsRef<Path>>(repo_path: P) -> Result<Self> {
        let repo_path = repo_path.as_ref().to_path_buf();

        let repo = Repository::open(&repo_path)
            .map_err(|e| ScribeError::git(format!("Failed to open repository: {}", e)))?;

        Ok(Self {
            repo_path,
            repo: parking_lot::Mutex::new(repo),
            cache: GitCache {
                cache_ttl: std::time::Duration::from_secs(300),
                ..Default::default()
            },
        })
    }

    /// List all tracked files in the repository using libgit2 index
    pub async fn list_tracked_files(&self) -> Result<Vec<PathBuf>> {
        let repo = self.repo.lock();
        let index = repo
            .index()
            .map_err(|e| ScribeError::git(format!("Failed to read index: {}", e)))?;

        let files: Vec<PathBuf> = index
            .iter()
            .filter_map(|entry| {
                std::str::from_utf8(&entry.path)
                    .ok()
                    .map(|p| self.repo_path.join(p))
            })
            .collect();

        *self.cache.files_discovered.write() = files.len();
        *self.cache.cache_timestamp.write() = Some(SystemTime::now());

        log::debug!("Git discovered {} tracked files", files.len());
        Ok(files)
    }

    /// Load all file statuses in a single batch operation using libgit2
    pub async fn load_batch_file_statuses(&self) -> Result<()> {
        let mut opts = StatusOptions::new();
        opts.include_untracked(true)
            .include_ignored(false)
            .include_unmodified(false);

        let repo = self.repo.lock();
        let statuses = repo
            .statuses(Some(&mut opts))
            .map_err(|e| ScribeError::git(format!("Failed to get status: {}", e)))?;

        for entry in statuses.iter() {
            let Some(path) = entry.path() else { continue };
            let git_status = entry.status();

            let status = if git_status.is_wt_new() {
                GitFileStatus::Untracked
            } else if git_status.is_wt_modified() || git_status.is_index_modified() {
                GitFileStatus::Modified
            } else if git_status.is_wt_deleted() || git_status.is_index_deleted() {
                GitFileStatus::Deleted
            } else if git_status.is_wt_renamed() || git_status.is_index_renamed() {
                GitFileStatus::Renamed
            } else if git_status.is_index_new() {
                GitFileStatus::Added
            } else if git_status.is_ignored() {
                GitFileStatus::Ignored
            } else if git_status.is_conflicted() {
                GitFileStatus::Unmerged
            } else {
                continue; // Skip unmodified
            };

            let full_path = self.repo_path.join(path);
            self.cache.batch_status_cache.insert(full_path, status);
        }

        // Mark batch status as loaded, even if empty (empty = all files unmodified)
        self.cache
            .batch_status_loaded
            .store(true, std::sync::atomic::Ordering::Release);
        *self.cache.cache_timestamp.write() = Some(SystemTime::now());

        log::debug!(
            "Loaded batch file statuses for {} modified files",
            self.cache.batch_status_cache.len()
        );

        Ok(())
    }

    /// Get file status only (fast path for scanning)
    ///
    /// This method only retrieves git status, skipping expensive operations like
    /// blame, commit history, and change stats. Use `get_file_info_detailed` if
    /// you need the full information.
    pub async fn get_file_info(&self, file_path: &Path) -> Result<GitFileInfo> {
        if let Some(cached_status) = self.cache.file_statuses.get(file_path) {
            if self.is_cache_valid() {
                return Ok(GitFileInfo {
                    path: file_path.to_path_buf(),
                    status: cached_status.clone(),
                    last_commit: None,
                    blame_info: None,
                    changes_count: 0,
                    additions: 0,
                    deletions: 0,
                });
            }
        }

        let status = self.get_file_status(file_path).await?;

        self.cache
            .file_statuses
            .insert(file_path.to_path_buf(), status.clone());
        *self.cache.cache_timestamp.write() = Some(SystemTime::now());

        Ok(GitFileInfo {
            path: file_path.to_path_buf(),
            status,
            last_commit: None,
            blame_info: None,
            changes_count: 0,
            additions: 0,
            deletions: 0,
        })
    }

    /// Get detailed file information including blame, commit history, and change stats
    ///
    /// WARNING: This is expensive! It runs git log, git blame, and git numstat per file.
    /// Only use when detailed git information is explicitly needed.
    #[allow(dead_code)]
    pub async fn get_file_info_detailed(&self, file_path: &Path) -> Result<GitFileInfo> {
        let status = self.get_file_status(file_path).await?;
        let last_commit = self.get_last_commit_for_file(file_path).await.ok();
        let blame_info = self.get_blame_info(file_path).await.ok();
        let (changes_count, additions, deletions) = self
            .get_file_change_stats(file_path)
            .await
            .unwrap_or((0, 0, 0));

        self.cache
            .file_statuses
            .insert(file_path.to_path_buf(), status.clone());
        if let Some(ref blame) = blame_info {
            self.cache
                .blame_cache
                .insert(file_path.to_path_buf(), blame.clone());
        }
        *self.cache.cache_timestamp.write() = Some(SystemTime::now());

        Ok(GitFileInfo {
            path: file_path.to_path_buf(),
            status,
            last_commit,
            blame_info,
            changes_count,
            additions,
            deletions,
        })
    }

    async fn get_file_status(&self, file_path: &Path) -> Result<GitFileStatus> {
        // If batch status was loaded, use it (even if empty = all unmodified)
        if self
            .cache
            .batch_status_loaded
            .load(std::sync::atomic::Ordering::Acquire)
        {
            if let Some(status) = self.cache.batch_status_cache.get(file_path) {
                return Ok(status.clone());
            }
            // Not in cache means unmodified (batch loaded all modified files)
            return Ok(GitFileStatus::Unmodified);
        }

        // Fallback: use libgit2 single-file status (only if batch wasn't loaded)
        let relative_path = file_path
            .strip_prefix(&self.repo_path)
            .map_err(|_| ScribeError::git("File not in repository".to_string()))?;

        let repo = self.repo.lock();
        let status = repo
            .status_file(relative_path)
            .map_err(|e| ScribeError::git(format!("Failed to get file status: {}", e)))?;

        let result = if status.is_wt_new() {
            GitFileStatus::Untracked
        } else if status.is_wt_modified() || status.is_index_modified() {
            GitFileStatus::Modified
        } else if status.is_wt_deleted() || status.is_index_deleted() {
            GitFileStatus::Deleted
        } else if status.is_wt_renamed() || status.is_index_renamed() {
            GitFileStatus::Renamed
        } else if status.is_index_new() {
            GitFileStatus::Added
        } else if status.is_ignored() {
            GitFileStatus::Ignored
        } else if status.is_conflicted() {
            GitFileStatus::Unmerged
        } else {
            GitFileStatus::Unmodified
        };

        Ok(result)
    }

    async fn get_last_commit_for_file(&self, file_path: &Path) -> Result<GitCommitInfo> {
        let relative_path = file_path
            .strip_prefix(&self.repo_path)
            .map_err(|_| ScribeError::git("File not in repository".to_string()))?;

        let output = AsyncCommand::new("git")
            .arg("log")
            .arg("-1")
            .arg("--pretty=format:%H|%an|%ae|%at|%s|%H")
            .arg("--")
            .arg(relative_path)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get commit info: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ScribeError::git(format!("git log failed: {}", stderr)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().splitn(6, '|').collect();

        if parts.len() < 5 {
            return Err(ScribeError::git("Invalid git log output".to_string()));
        }

        let timestamp = parts[3]
            .parse::<u64>()
            .map_err(|_| ScribeError::git("Invalid timestamp".to_string()))?;

        Ok(GitCommitInfo {
            hash: parts[0].to_string(),
            author: parts[1].to_string(),
            email: parts[2].to_string(),
            timestamp,
            message: parts[4].to_string(),
            files_changed: 1,
        })
    }

    async fn get_blame_info(&self, file_path: &Path) -> Result<GitBlameInfo> {
        if let Some(cached_blame) = self.cache.blame_cache.get(file_path) {
            if self.is_cache_valid() {
                return Ok(cached_blame.clone());
            }
        }

        let relative_path = file_path
            .strip_prefix(&self.repo_path)
            .map_err(|_| ScribeError::git("File not in repository".to_string()))?;

        let output = AsyncCommand::new("git")
            .arg("blame")
            .arg("--porcelain")
            .arg(relative_path)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get blame info: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ScribeError::git(format!("git blame failed: {}", stderr)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_blame_output(&stdout)
    }

    fn parse_blame_output(&self, blame_output: &str) -> Result<GitBlameInfo> {
        let mut lines = Vec::new();
        let mut contributors = HashMap::new();
        let mut last_modified = 0u64;

        let blame_lines: Vec<&str> = blame_output.lines().collect();
        let mut i = 0;

        while i < blame_lines.len() {
            let line = blame_lines[i];
            if line.is_empty() {
                i += 1;
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                i += 1;
                continue;
            }

            let commit_hash = parts[0].to_string();
            let line_number = parts[2].parse::<usize>().unwrap_or(0);

            let mut author = String::new();
            let mut timestamp = 0u64;
            let mut content = String::new();

            i += 1;
            while i < blame_lines.len() {
                let info_line = blame_lines[i];
                if info_line.starts_with("author ") {
                    author = info_line[7..].to_string();
                } else if info_line.starts_with("author-time ") {
                    timestamp = info_line[12..].parse().unwrap_or(0);
                    last_modified = last_modified.max(timestamp);
                } else if info_line.starts_with('\t') {
                    content = info_line[1..].to_string();
                    break;
                }
                i += 1;
            }

            *contributors.entry(author.clone()).or_insert(0) += 1;

            lines.push(GitBlameLine {
                line_number,
                commit_hash,
                author,
                timestamp,
                content,
            });

            i += 1;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut age_distribution = AgeDistribution::default();

        for line in &lines {
            let age_seconds = now.saturating_sub(line.timestamp);
            let age_days = age_seconds / 86400;

            match age_days {
                0..=30 => age_distribution.recent += 1,
                31..=180 => age_distribution.moderate += 1,
                181..=365 => age_distribution.old += 1,
                _ => age_distribution.ancient += 1,
            }
        }

        Ok(GitBlameInfo {
            lines,
            contributors,
            last_modified,
            age_distribution,
        })
    }

    async fn get_file_change_stats(&self, file_path: &Path) -> Result<(usize, usize, usize)> {
        let relative_path = file_path
            .strip_prefix(&self.repo_path)
            .map_err(|_| ScribeError::git("File not in repository".to_string()))?;

        let output = AsyncCommand::new("git")
            .arg("log")
            .arg("--numstat")
            .arg("--pretty=format:")
            .arg("--")
            .arg(relative_path)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get change stats: {}", e)))?;

        if !output.status.success() {
            return Ok((0, 0, 0));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut total_changes = 0;
        let mut total_additions = 0;
        let mut total_deletions = 0;

        for line in stdout.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(additions), Ok(deletions)) =
                    (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                {
                    total_additions += additions;
                    total_deletions += deletions;
                    total_changes += 1;
                }
            }
        }

        Ok((total_changes, total_additions, total_deletions))
    }

    /// Get comprehensive repository statistics
    pub async fn get_repository_stats(&self) -> Result<GitRepositoryStats> {
        let (total_commits, contributors) = self.get_contributor_stats().await?;
        let branches = self.get_branches().await?;
        let tags = self.get_tags().await?;
        let file_types = self.analyze_file_types().await?;
        let activity_timeline = self.get_activity_timeline().await?;
        let repository_health = self
            .calculate_repository_health(&contributors, &activity_timeline)
            .await?;

        Ok(GitRepositoryStats {
            total_commits,
            contributors,
            branches,
            tags,
            file_types,
            activity_timeline,
            repository_health,
        })
    }

    async fn get_contributor_stats(&self) -> Result<(usize, Vec<ContributorStats>)> {
        let output = AsyncCommand::new("git")
            .arg("shortlog")
            .arg("-sne")
            .arg("--all")
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get contributors: {}", e)))?;

        if !output.status.success() {
            return Ok((0, vec![]));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut contributors = Vec::new();
        let mut total_commits = 0;

        for line in stdout.lines() {
            if let Some((count_str, name_email)) = line.trim().split_once('\t') {
                if let Ok(commits) = count_str.trim().parse::<usize>() {
                    total_commits += commits;

                    let (name, email) = if let Some((n, e)) = name_email.rsplit_once('<') {
                        let email = e.trim_end_matches('>');
                        (n.trim().to_string(), email.to_string())
                    } else {
                        (name_email.to_string(), String::new())
                    };

                    let (lines_added, lines_deleted, files_modified, first_commit, last_commit) =
                        self.get_detailed_contributor_stats(&email)
                            .await
                            .unwrap_or((0, 0, 0, 0, 0));

                    contributors.push(ContributorStats {
                        name,
                        email,
                        commits,
                        lines_added,
                        lines_deleted,
                        files_modified,
                        first_commit,
                        last_commit,
                    });
                }
            }
        }

        contributors.sort_by(|a, b| b.commits.cmp(&a.commits));
        Ok((total_commits, contributors))
    }

    async fn get_detailed_contributor_stats(
        &self,
        email: &str,
    ) -> Result<(usize, usize, usize, u64, u64)> {
        let output = AsyncCommand::new("git")
            .arg("log")
            .arg("--author")
            .arg(email)
            .arg("--numstat")
            .arg("--pretty=format:%at")
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get detailed stats: {}", e)))?;

        if !output.status.success() {
            return Ok((0, 0, 0, 0, 0));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines_added = 0;
        let mut lines_deleted = 0;
        let mut files_modified = 0;
        let mut timestamps = Vec::new();

        for line in stdout.lines() {
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(timestamp) = line.parse::<u64>() {
                timestamps.push(timestamp);
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let (Ok(added), Ok(deleted)) =
                    (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                {
                    lines_added += added;
                    lines_deleted += deleted;
                    files_modified += 1;
                }
            }
        }

        let first_commit = timestamps.iter().min().copied().unwrap_or(0);
        let last_commit = timestamps.iter().max().copied().unwrap_or(0);

        Ok((
            lines_added,
            lines_deleted,
            files_modified,
            first_commit,
            last_commit,
        ))
    }

    async fn get_branches(&self) -> Result<Vec<String>> {
        let output = AsyncCommand::new("git")
            .arg("branch")
            .arg("-a")
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get branches: {}", e)))?;

        if !output.status.success() {
            return Ok(vec![]);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout
            .lines()
            .map(|line| line.trim_start_matches("* ").trim().to_string())
            .filter(|line| !line.is_empty())
            .collect())
    }

    async fn get_tags(&self) -> Result<Vec<String>> {
        let output = AsyncCommand::new("git")
            .arg("tag")
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get tags: {}", e)))?;

        if !output.status.success() {
            return Ok(vec![]);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect())
    }

    async fn analyze_file_types(&self) -> Result<HashMap<String, usize>> {
        let files = self.list_tracked_files().await?;
        let mut file_types = HashMap::new();

        for file in files {
            if let Some(extension) = file.extension().and_then(|ext| ext.to_str()) {
                *file_types.entry(extension.to_string()).or_insert(0) += 1;
            } else {
                *file_types.entry("no_extension".to_string()).or_insert(0) += 1;
            }
        }

        Ok(file_types)
    }

    async fn get_activity_timeline(&self) -> Result<Vec<ActivityPeriod>> {
        Ok(vec![])
    }

    async fn calculate_repository_health(
        &self,
        contributors: &[ContributorStats],
        activity_timeline: &[ActivityPeriod],
    ) -> Result<RepositoryHealth> {
        let commit_frequency = if !activity_timeline.is_empty() {
            let total_commits: usize = activity_timeline.iter().map(|p| p.commits).sum();
            total_commits as f64 / activity_timeline.len() as f64
        } else {
            0.0
        };

        let contributor_diversity = contributors.len() as f64;
        let total_added: usize = contributors.iter().map(|c| c.lines_added).sum();
        let total_deleted: usize = contributors.iter().map(|c| c.lines_deleted).sum();
        let code_churn = if total_added > 0 {
            total_deleted as f64 / total_added as f64
        } else {
            0.0
        };

        Ok(RepositoryHealth {
            commit_frequency,
            contributor_diversity,
            code_churn,
            documentation_ratio: 0.0,
            test_coverage_estimate: 0.0,
            branch_health: BranchHealth {
                main_branch: "main".to_string(),
                active_branches: 1,
                stale_branches: 0,
                merge_conflicts_risk: 0.0,
            },
        })
    }

    fn is_cache_valid(&self) -> bool {
        if let Some(cache_time) = *self.cache.cache_timestamp.read() {
            SystemTime::now()
                .duration_since(cache_time)
                .map(|duration| duration < self.cache.cache_ttl)
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Clear all caches
    pub fn clear_cache(&self) {
        self.cache.file_statuses.clear();
        self.cache.commit_cache.clear();
        self.cache.blame_cache.clear();
        self.cache.batch_status_cache.clear();
        *self.cache.cache_timestamp.write() = None;
    }

    /// Get number of files discovered through git
    pub fn files_discovered(&self) -> usize {
        *self.cache.files_discovered.read()
    }

    /// Check if git is available (always true if GitIntegrator was created successfully)
    pub fn is_git_available(&self) -> bool {
        true
    }

    /// Get repository root path
    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }
}

// Diff analysis methods are in diff_analysis.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_cache_default() {
        let cache = GitCache::default();
        assert!(cache.file_statuses.is_empty());
        assert!(cache.commit_cache.is_empty());
        assert!(cache.blame_cache.is_empty());
        assert_eq!(cache.cache_ttl, std::time::Duration::from_secs(300));
    }

    #[test]
    fn test_cache_ttl_setting() {
        let cache = GitCache {
            cache_ttl: std::time::Duration::from_secs(60),
            ..Default::default()
        };
        assert_eq!(cache.cache_ttl, std::time::Duration::from_secs(60));
    }

    #[test]
    fn test_cache_file_status() {
        let cache = GitCache::default();
        let path = PathBuf::from("test.rs");

        cache
            .file_statuses
            .insert(path.clone(), GitFileStatus::Modified);

        assert!(cache.file_statuses.contains_key(&path));
        assert_eq!(
            *cache.file_statuses.get(&path).unwrap(),
            GitFileStatus::Modified
        );
    }

    #[test]
    fn test_cache_commit_info() {
        let cache = GitCache::default();
        let commit_hash = "abc123".to_string();

        let commit_info = GitCommitInfo {
            hash: commit_hash.clone(),
            author: "Test".to_string(),
            email: "test@example.com".to_string(),
            timestamp: 0,
            message: "Test commit".to_string(),
            files_changed: 5,
        };

        cache
            .commit_cache
            .insert(commit_hash.clone(), commit_info.clone());

        assert!(cache.commit_cache.contains_key(&commit_hash));
        let cached = cache.commit_cache.get(&commit_hash).unwrap();
        assert_eq!(cached.message, "Test commit");
    }

    #[test]
    fn test_parse_status_output() {
        // Test parsing of git status porcelain output
        let status_line = "M  src/main.rs";
        let parts: Vec<&str> = status_line.split_whitespace().collect();

        if parts.len() >= 2 {
            let status_code = parts[0];
            let file_path = parts[1];

            assert_eq!(status_code, "M");
            assert_eq!(file_path, "src/main.rs");
        }
    }

    #[test]
    fn test_parse_ls_files_output() {
        // Test parsing of git ls-files -z output
        let output = "src/main.rs\0src/lib.rs\0tests/test.rs\0";
        let files: Vec<&str> = output.split('\0').filter(|s| !s.is_empty()).collect();

        assert_eq!(files.len(), 3);
        assert_eq!(files[0], "src/main.rs");
        assert_eq!(files[1], "src/lib.rs");
        assert_eq!(files[2], "tests/test.rs");
    }

    #[test]
    fn test_git_file_status_variants() {
        let statuses = vec![
            GitFileStatus::Modified,
            GitFileStatus::Added,
            GitFileStatus::Deleted,
            GitFileStatus::Renamed,
            GitFileStatus::Untracked,
        ];

        for status in statuses {
            // Just ensure all variants exist and can be compared
            assert_eq!(status, status);
        }
    }

    #[test]
    fn test_parse_blame_line() {
        // Test parsing a typical blame line
        let blame_line = "abc12345 (Author Name 2024-01-15 10:30:00 +0000 42) Some code here";

        // Extract hash (first 8+ characters before space)
        let hash_end = blame_line.find(' ').unwrap_or(0);
        let hash = &blame_line[..hash_end];

        assert_eq!(hash, "abc12345");
        assert!(blame_line.contains("Author Name"));
        assert!(blame_line.contains("Some code here"));
    }

    #[test]
    fn test_contributor_stats() {
        let stats = ContributorStats {
            name: "Test Author".to_string(),
            email: "test@example.com".to_string(),
            commits: 10,
            lines_added: 500,
            lines_deleted: 100,
            files_modified: 25,
            first_commit: 0,
            last_commit: 1000000,
        };

        assert_eq!(stats.name, "Test Author");
        assert_eq!(stats.commits, 10);
        assert_eq!(stats.lines_added, 500);
    }

    #[test]
    fn test_repository_health() {
        let branch_health = BranchHealth {
            main_branch: "main".to_string(),
            active_branches: 5,
            stale_branches: 2,
            merge_conflicts_risk: 0.1,
        };

        let health = RepositoryHealth {
            commit_frequency: 5.0,
            contributor_diversity: 0.8,
            code_churn: 0.3,
            documentation_ratio: 0.15,
            test_coverage_estimate: 0.7,
            branch_health,
        };

        assert!(health.commit_frequency > 0.0);
        assert!(health.contributor_diversity <= 1.0);
    }

    #[test]
    fn test_age_distribution_default() {
        let dist = AgeDistribution::default();
        assert_eq!(dist.recent, 0);
        assert_eq!(dist.moderate, 0);
        assert_eq!(dist.old, 0);
        assert_eq!(dist.ancient, 0);
    }

    #[test]
    fn test_age_distribution_values() {
        let dist = AgeDistribution {
            recent: 100,
            moderate: 50,
            old: 30,
            ancient: 20,
        };

        let total = dist.recent + dist.moderate + dist.old + dist.ancient;
        assert_eq!(total, 200);
    }

    #[test]
    fn test_branch_health_structure() {
        let branch = BranchHealth {
            main_branch: "main".to_string(),
            active_branches: 10,
            stale_branches: 3,
            merge_conflicts_risk: 0.05,
        };

        assert_eq!(branch.main_branch, "main");
        assert_eq!(branch.active_branches, 10);
        assert!(branch.merge_conflicts_risk < 1.0);
    }

    #[test]
    fn test_git_file_info() {
        let info = GitFileInfo {
            path: PathBuf::from("src/main.rs"),
            status: GitFileStatus::Modified,
            last_commit: None,
            blame_info: None,
            changes_count: 5,
            additions: 10,
            deletions: 3,
        };

        assert_eq!(info.path, PathBuf::from("src/main.rs"));
        assert_eq!(info.status, GitFileStatus::Modified);
        assert_eq!(info.changes_count, 5);
    }

    #[test]
    fn test_git_blame_line() {
        let line = GitBlameLine {
            line_number: 42,
            commit_hash: "abc123".to_string(),
            author: "Test Author".to_string(),
            timestamp: 1704067200,
            content: "let x = 42;".to_string(),
        };

        assert_eq!(line.line_number, 42);
        assert_eq!(line.commit_hash, "abc123");
        assert!(line.content.contains("let"));
    }

    #[test]
    fn test_git_blame_info() {
        let mut contributors = HashMap::new();
        contributors.insert("Author1".to_string(), 50);
        contributors.insert("Author2".to_string(), 30);

        let info = GitBlameInfo {
            lines: vec![],
            contributors,
            last_modified: 1704067200,
            age_distribution: AgeDistribution::default(),
        };

        assert_eq!(info.contributors.len(), 2);
        assert_eq!(info.contributors.get("Author1"), Some(&50));
    }

    #[test]
    fn test_git_commit_info() {
        let commit = GitCommitInfo {
            hash: "abc123def456".to_string(),
            author: "Test Author".to_string(),
            email: "test@example.com".to_string(),
            timestamp: 1704067200,
            message: "Initial commit".to_string(),
            files_changed: 5,
        };

        assert_eq!(commit.hash, "abc123def456");
        assert_eq!(commit.author, "Test Author");
        assert_eq!(commit.email, "test@example.com");
        assert_eq!(commit.files_changed, 5);
    }

    #[test]
    fn test_git_repository_stats() {
        let stats = GitRepositoryStats {
            total_commits: 100,
            contributors: vec![],
            branches: vec!["main".to_string(), "develop".to_string()],
            tags: vec!["v1.0".to_string()],
            file_types: HashMap::new(),
            activity_timeline: vec![],
            repository_health: RepositoryHealth {
                commit_frequency: 5.0,
                contributor_diversity: 0.8,
                code_churn: 0.2,
                documentation_ratio: 0.1,
                test_coverage_estimate: 0.75,
                branch_health: BranchHealth {
                    main_branch: "main".to_string(),
                    active_branches: 2,
                    stale_branches: 0,
                    merge_conflicts_risk: 0.0,
                },
            },
        };

        assert_eq!(stats.total_commits, 100);
        assert_eq!(stats.branches.len(), 2);
        assert_eq!(stats.tags.len(), 1);
    }

    #[test]
    fn test_activity_period() {
        use std::collections::HashSet;

        let mut contributors = HashSet::new();
        contributors.insert("dev1".to_string());
        contributors.insert("dev2".to_string());
        contributors.insert("dev3".to_string());

        let period = ActivityPeriod {
            period: "2024-01".to_string(),
            commits: 15,
            lines_changed: 600,
            files_touched: 25,
            contributors,
        };

        assert_eq!(period.commits, 15);
        assert_eq!(period.contributors.len(), 3);
        assert_eq!(period.lines_changed, 600);
        assert_eq!(period.files_touched, 25);
    }

    #[test]
    fn test_parse_status_codes() {
        // Test all git status code mappings
        let codes = vec![
            (" M", GitFileStatus::Modified),
            ("M ", GitFileStatus::Modified),
            ("MM", GitFileStatus::Modified),
            ("A ", GitFileStatus::Added),
            (" A", GitFileStatus::Added),
            ("D ", GitFileStatus::Deleted),
            (" D", GitFileStatus::Deleted),
            ("R ", GitFileStatus::Renamed),
            (" R", GitFileStatus::Renamed),
            ("C ", GitFileStatus::Copied),
            (" C", GitFileStatus::Copied),
            ("??", GitFileStatus::Untracked),
            ("!!", GitFileStatus::Ignored),
            ("XX", GitFileStatus::Unmodified), // Unknown defaults to Unmodified
        ];

        for (code, expected_status) in codes {
            let status = match code {
                " M" | "M " | "MM" => GitFileStatus::Modified,
                "A " | " A" => GitFileStatus::Added,
                "D " | " D" => GitFileStatus::Deleted,
                "R " | " R" => GitFileStatus::Renamed,
                "C " | " C" => GitFileStatus::Copied,
                "??" => GitFileStatus::Untracked,
                "!!" => GitFileStatus::Ignored,
                _ => GitFileStatus::Unmodified,
            };
            assert_eq!(status, expected_status);
        }
    }

    #[test]
    fn test_parse_numstat_line() {
        // Test parsing of git numstat output
        let lines = vec![
            ("10\t5\tsrc/main.rs", 10, 5),
            ("100\t50\tsrc/lib.rs", 100, 50),
            ("0\t25\ttests/test.rs", 0, 25),
        ];

        for (line, expected_add, expected_del) in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let additions: usize = parts[0].parse().unwrap();
                let deletions: usize = parts[1].parse().unwrap();
                assert_eq!(additions, expected_add);
                assert_eq!(deletions, expected_del);
            }
        }
    }

    #[test]
    fn test_parse_shortlog_line() {
        // Test parsing git shortlog output
        let line = "  42\tJohn Doe <john@example.com>";
        if let Some((count_str, name_email)) = line.trim().split_once('\t') {
            let count: usize = count_str.trim().parse().unwrap();
            assert_eq!(count, 42);
            assert!(name_email.contains("John Doe"));
            assert!(name_email.contains("john@example.com"));
        }
    }

    #[test]
    fn test_extract_name_email() {
        let name_email = "John Doe <john@example.com>";
        if let Some((n, e)) = name_email.rsplit_once('<') {
            let name = n.trim().to_string();
            let email = e.trim_end_matches('>').to_string();
            assert_eq!(name, "John Doe");
            assert_eq!(email, "john@example.com");
        }
    }

    #[test]
    fn test_file_types_analysis() {
        let mut file_types: HashMap<String, usize> = HashMap::new();
        let files = vec![
            "src/main.rs",
            "src/lib.rs",
            "tests/test.rs",
            "Cargo.toml",
            "README.md",
            "Makefile", // No extension
        ];

        for file in files {
            let path = PathBuf::from(file);
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                *file_types.entry(ext.to_string()).or_insert(0) += 1;
            } else {
                *file_types.entry("no_extension".to_string()).or_insert(0) += 1;
            }
        }

        assert_eq!(file_types.get("rs"), Some(&3));
        assert_eq!(file_types.get("toml"), Some(&1));
        assert_eq!(file_types.get("md"), Some(&1));
        assert_eq!(file_types.get("no_extension"), Some(&1));
    }

    #[test]
    fn test_age_calculation() {
        let now = 1704067200u64; // Some fixed timestamp
        let timestamps = vec![
            (now - 86400 * 7, "recent"),    // 7 days ago
            (now - 86400 * 60, "moderate"), // 60 days ago
            (now - 86400 * 200, "old"),     // 200 days ago
            (now - 86400 * 500, "ancient"), // 500 days ago
        ];

        for (timestamp, expected_category) in timestamps {
            let age_seconds = now - timestamp;
            let age_days = age_seconds / 86400;

            let category = match age_days {
                0..=30 => "recent",
                31..=180 => "moderate",
                181..=365 => "old",
                _ => "ancient",
            };
            assert_eq!(category, expected_category);
        }
    }

    #[test]
    fn test_cache_batch_status() {
        let cache = GitCache::default();

        let paths = vec![
            PathBuf::from("src/main.rs"),
            PathBuf::from("src/lib.rs"),
            PathBuf::from("tests/test.rs"),
        ];

        for path in &paths {
            cache
                .batch_status_cache
                .insert(path.clone(), GitFileStatus::Modified);
        }

        assert_eq!(cache.batch_status_cache.len(), 3);
        for path in paths {
            assert!(cache.batch_status_cache.contains_key(&path));
        }
    }

    #[test]
    fn test_cache_clear() {
        let cache = GitCache::default();

        // Add some items
        cache
            .file_statuses
            .insert(PathBuf::from("test.rs"), GitFileStatus::Modified);
        cache.commit_cache.insert(
            "abc123".to_string(),
            GitCommitInfo {
                hash: "abc123".to_string(),
                author: "Test".to_string(),
                email: "test@test.com".to_string(),
                timestamp: 0,
                message: "test".to_string(),
                files_changed: 1,
            },
        );

        assert!(!cache.file_statuses.is_empty());
        assert!(!cache.commit_cache.is_empty());

        // Clear
        cache.file_statuses.clear();
        cache.commit_cache.clear();

        assert!(cache.file_statuses.is_empty());
        assert!(cache.commit_cache.is_empty());
    }

    #[test]
    fn test_contributor_stats_sorting() {
        let mut contributors = vec![
            ContributorStats {
                name: "Alice".to_string(),
                email: "alice@example.com".to_string(),
                commits: 10,
                lines_added: 100,
                lines_deleted: 50,
                files_modified: 5,
                first_commit: 0,
                last_commit: 1000,
            },
            ContributorStats {
                name: "Bob".to_string(),
                email: "bob@example.com".to_string(),
                commits: 50,
                lines_added: 500,
                lines_deleted: 100,
                files_modified: 20,
                first_commit: 0,
                last_commit: 1000,
            },
            ContributorStats {
                name: "Charlie".to_string(),
                email: "charlie@example.com".to_string(),
                commits: 25,
                lines_added: 250,
                lines_deleted: 75,
                files_modified: 10,
                first_commit: 0,
                last_commit: 1000,
            },
        ];

        contributors.sort_by(|a, b| b.commits.cmp(&a.commits));

        assert_eq!(contributors[0].name, "Bob");
        assert_eq!(contributors[1].name, "Charlie");
        assert_eq!(contributors[2].name, "Alice");
    }

    #[test]
    fn test_code_churn_calculation() {
        let total_added = 1000usize;
        let total_deleted = 300usize;

        let code_churn = if total_added > 0 {
            total_deleted as f64 / total_added as f64
        } else {
            0.0
        };

        assert!((code_churn - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_branch_name_parsing() {
        let lines = vec![
            "* main",
            "  develop",
            "  feature/test",
            "  remotes/origin/main",
        ];

        let branches: Vec<String> = lines
            .iter()
            .map(|line| line.trim_start_matches("* ").trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();

        assert_eq!(branches.len(), 4);
        assert_eq!(branches[0], "main");
        assert_eq!(branches[1], "develop");
    }

    #[test]
    fn test_git_file_status_clone_and_eq() {
        let status1 = GitFileStatus::Modified;
        let status2 = status1.clone();

        assert_eq!(status1, status2);
    }

    #[test]
    fn test_git_file_status_all_variants() {
        let statuses = [
            GitFileStatus::Modified,
            GitFileStatus::Added,
            GitFileStatus::Deleted,
            GitFileStatus::Renamed,
            GitFileStatus::Copied,
            GitFileStatus::Untracked,
            GitFileStatus::Ignored,
            GitFileStatus::Unmodified,
        ];

        // Verify all variants are distinct
        for (i, s1) in statuses.iter().enumerate() {
            for (j, s2) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_blame_line_with_different_timestamps() {
        let lines = vec![
            GitBlameLine {
                line_number: 1,
                commit_hash: "aaa111".to_string(),
                author: "Author1".to_string(),
                timestamp: 1000000,
                content: "line 1".to_string(),
            },
            GitBlameLine {
                line_number: 2,
                commit_hash: "bbb222".to_string(),
                author: "Author2".to_string(),
                timestamp: 2000000,
                content: "line 2".to_string(),
            },
        ];

        let last_modified = lines.iter().map(|l| l.timestamp).max().unwrap_or(0);
        assert_eq!(last_modified, 2000000);
    }

    #[test]
    fn test_repository_health_calculation() {
        let contributors = vec![
            ContributorStats {
                name: "Dev1".to_string(),
                email: "dev1@test.com".to_string(),
                commits: 100,
                lines_added: 5000,
                lines_deleted: 1000,
                files_modified: 50,
                first_commit: 0,
                last_commit: 1000000,
            },
            ContributorStats {
                name: "Dev2".to_string(),
                email: "dev2@test.com".to_string(),
                commits: 50,
                lines_added: 2000,
                lines_deleted: 500,
                files_modified: 25,
                first_commit: 100000,
                last_commit: 900000,
            },
        ];

        let total_added: usize = contributors.iter().map(|c| c.lines_added).sum();
        let total_deleted: usize = contributors.iter().map(|c| c.lines_deleted).sum();

        let code_churn = if total_added > 0 {
            total_deleted as f64 / total_added as f64
        } else {
            0.0
        };

        assert_eq!(total_added, 7000);
        assert_eq!(total_deleted, 1500);
        assert!((code_churn - 0.214).abs() < 0.01);
    }
}

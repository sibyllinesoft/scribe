//! Git integration for enhanced file discovery and status tracking.
//!
//! This module provides comprehensive Git integration capabilities including:
//! - Fast file discovery using `git ls-files`
//! - File status tracking (modified, staged, untracked)
//! - Commit history and blame information
//! - Repository statistics and health metrics

use dashmap::DashMap;
use scribe_core::{GitFileStatus, Result, ScribeError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command as AsyncCommand;

use super::types::{
    ActivityPeriod, AgeDistribution, BranchHealth, ContributorStats, GitBlameInfo, GitBlameLine,
    GitCommitInfo, GitFileInfo, GitRepositoryStats, RepositoryHealth,
};

/// Git repository integration handler
#[derive(Debug)]
pub struct GitIntegrator {
    repo_path: PathBuf,
    git_available: bool,
    cache: GitCache,
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
        }
    }
}

impl GitIntegrator {
    /// Create a new Git integrator for the given repository path
    pub fn new<P: AsRef<Path>>(repo_path: P) -> Result<Self> {
        let repo_path = repo_path.as_ref().to_path_buf();

        let git_dir = repo_path.join(".git");
        if !git_dir.exists() {
            return Err(ScribeError::git("Not a git repository".to_string()));
        }

        let git_available = Command::new("git")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        if !git_available {
            log::warn!("Git command not available, falling back to filesystem scanning");
        }

        Ok(Self {
            repo_path,
            git_available,
            cache: GitCache {
                cache_ttl: std::time::Duration::from_secs(300),
                ..Default::default()
            },
        })
    }

    /// List all tracked files in the repository
    pub async fn list_tracked_files(&self) -> Result<Vec<PathBuf>> {
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

        let output = AsyncCommand::new("git")
            .arg("ls-files")
            .arg("-z")
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to run git ls-files: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ScribeError::git(format!("git ls-files failed: {}", stderr)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let files: Vec<PathBuf> = stdout
            .split('\0')
            .filter(|s| !s.is_empty())
            .map(|s| self.repo_path.join(s))
            .collect();

        *self.cache.files_discovered.write() = files.len();
        *self.cache.cache_timestamp.write() = Some(SystemTime::now());

        log::debug!("Git discovered {} tracked files", files.len());
        Ok(files)
    }

    /// Load all file statuses in a single batch operation for better performance
    pub async fn load_batch_file_statuses(&self) -> Result<()> {
        if !self.git_available {
            return Ok(());
        }

        let output = AsyncCommand::new("git")
            .arg("status")
            .arg("--porcelain")
            .arg("-z")
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get batch file status: {}", e)))?;

        if !output.status.success() {
            log::warn!("Git status failed, batch status unavailable");
            return Ok(());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        for line in stdout.split('\0') {
            if line.len() < 3 {
                continue;
            }

            let status_code = &line[..2];
            let file_path = &line[3..];

            if file_path.is_empty() {
                continue;
            }

            let status = match status_code {
                " M" | "M " | "MM" => GitFileStatus::Modified,
                "A " | " A" => GitFileStatus::Added,
                "D " | " D" => GitFileStatus::Deleted,
                "R " | " R" => GitFileStatus::Renamed,
                "C " | " C" => GitFileStatus::Copied,
                "??" => GitFileStatus::Untracked,
                "!!" => GitFileStatus::Ignored,
                _ => GitFileStatus::Unmodified,
            };

            let full_path = self.repo_path.join(file_path);
            self.cache.batch_status_cache.insert(full_path, status);
        }

        *self.cache.cache_timestamp.write() = Some(SystemTime::now());

        log::debug!(
            "Loaded batch file statuses for {} files",
            self.cache.batch_status_cache.len()
        );

        Ok(())
    }

    /// Get detailed file information including git status
    pub async fn get_file_info(&self, file_path: &Path) -> Result<GitFileInfo> {
        if let Some(cached_status) = self.cache.file_statuses.get(file_path) {
            if self.is_cache_valid() {
                return Ok(GitFileInfo {
                    path: file_path.to_path_buf(),
                    status: cached_status.clone(),
                    last_commit: None,
                    blame_info: self.cache.blame_cache.get(file_path).map(|entry| entry.clone()),
                    changes_count: 0,
                    additions: 0,
                    deletions: 0,
                });
            }
        }

        let status = self.get_file_status(file_path).await?;
        let last_commit = self.get_last_commit_for_file(file_path).await.ok();
        let blame_info = self.get_blame_info(file_path).await.ok();
        let (changes_count, additions, deletions) = self.get_file_change_stats(file_path).await.unwrap_or((0, 0, 0));

        self.cache.file_statuses.insert(file_path.to_path_buf(), status.clone());
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
        if !self.git_available {
            return Ok(GitFileStatus::Untracked);
        }

        if !self.cache.batch_status_cache.is_empty() {
            if let Some(status) = self.cache.batch_status_cache.get(file_path) {
                return Ok(status.clone());
            }
            return Ok(GitFileStatus::Unmodified);
        }

        let relative_path = file_path
            .strip_prefix(&self.repo_path)
            .map_err(|_| ScribeError::git("File not in repository".to_string()))?;

        let output = AsyncCommand::new("git")
            .arg("status")
            .arg("--porcelain")
            .arg(relative_path)
            .current_dir(&self.repo_path)
            .output()
            .await
            .map_err(|e| ScribeError::git(format!("Failed to get file status: {}", e)))?;

        if !output.status.success() {
            return Ok(GitFileStatus::Unmodified);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let status = if stdout.is_empty() {
            GitFileStatus::Unmodified
        } else {
            let status_code = stdout.chars().take(2).collect::<String>();
            match status_code.as_str() {
                " M" | "M " | "MM" => GitFileStatus::Modified,
                "A " => GitFileStatus::Added,
                "D " => GitFileStatus::Deleted,
                "R " => GitFileStatus::Renamed,
                "C " => GitFileStatus::Copied,
                "??" => GitFileStatus::Untracked,
                "!!" => GitFileStatus::Ignored,
                _ => GitFileStatus::Unmodified,
            }
        };

        Ok(status)
    }

    async fn get_last_commit_for_file(&self, file_path: &Path) -> Result<GitCommitInfo> {
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

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
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

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

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

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
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

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
                if let (Ok(additions), Ok(deletions)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
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
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

        let (total_commits, contributors) = self.get_contributor_stats().await?;
        let branches = self.get_branches().await?;
        let tags = self.get_tags().await?;
        let file_types = self.analyze_file_types().await?;
        let activity_timeline = self.get_activity_timeline().await?;
        let repository_health = self.calculate_repository_health(&contributors, &activity_timeline).await?;

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
                        self.get_detailed_contributor_stats(&email).await.unwrap_or((0, 0, 0, 0, 0));

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

    async fn get_detailed_contributor_stats(&self, email: &str) -> Result<(usize, usize, usize, u64, u64)> {
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
                if let (Ok(added), Ok(deleted)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    lines_added += added;
                    lines_deleted += deleted;
                    files_modified += 1;
                }
            }
        }

        let first_commit = timestamps.iter().min().copied().unwrap_or(0);
        let last_commit = timestamps.iter().max().copied().unwrap_or(0);

        Ok((lines_added, lines_deleted, files_modified, first_commit, last_commit))
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
        Ok(stdout.lines().map(|line| line.trim_start_matches("* ").trim().to_string()).filter(|line| !line.is_empty()).collect())
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
        Ok(stdout.lines().filter(|line| !line.trim().is_empty()).map(|line| line.trim().to_string()).collect())
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

    async fn calculate_repository_health(&self, contributors: &[ContributorStats], activity_timeline: &[ActivityPeriod]) -> Result<RepositoryHealth> {
        let commit_frequency = if !activity_timeline.is_empty() {
            let total_commits: usize = activity_timeline.iter().map(|p| p.commits).sum();
            total_commits as f64 / activity_timeline.len() as f64
        } else {
            0.0
        };

        let contributor_diversity = contributors.len() as f64;
        let total_added: usize = contributors.iter().map(|c| c.lines_added).sum();
        let total_deleted: usize = contributors.iter().map(|c| c.lines_deleted).sum();
        let code_churn = if total_added > 0 { total_deleted as f64 / total_added as f64 } else { 0.0 };

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
            SystemTime::now().duration_since(cache_time).map(|duration| duration < self.cache.cache_ttl).unwrap_or(false)
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

    /// Check if git is available
    pub fn is_git_available(&self) -> bool {
        self.git_available
    }

    /// Get repository root path
    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }
}

// Diff analysis methods are in diff_analysis.rs

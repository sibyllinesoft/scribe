//! Git integration for enhanced file discovery and status tracking.
//!
//! This module provides comprehensive Git integration capabilities including:
//! - Fast file discovery using `git ls-files`
//! - File status tracking (modified, staged, untracked)
//! - Commit history and blame information
//! - Repository statistics and health metrics

use scribe_core::{Result, ScribeError, GitStatus, GitFileStatus};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use std::cell::RefCell;
use serde::{Serialize, Deserialize};
use tokio::process::Command as AsyncCommand;

/// Git repository integration handler
#[derive(Debug)]
pub struct GitIntegrator {
    repo_path: PathBuf,
    git_available: bool,
    cache: GitCache,
}

/// Git file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitFileInfo {
    pub path: PathBuf,
    pub status: GitFileStatus,
    pub last_commit: Option<GitCommitInfo>,
    pub blame_info: Option<GitBlameInfo>,
    pub changes_count: usize,
    pub additions: usize,
    pub deletions: usize,
}

/// Git commit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCommitInfo {
    pub hash: String,
    pub author: String,
    pub email: String,
    pub timestamp: u64,
    pub message: String,
    pub files_changed: usize,
}

/// Git blame information for a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitBlameInfo {
    pub lines: Vec<GitBlameLine>,
    pub contributors: HashMap<String, usize>, // author -> line count
    pub last_modified: u64,
    pub age_distribution: AgeDistribution,
}

/// Individual line blame information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitBlameLine {
    pub line_number: usize,
    pub commit_hash: String,
    pub author: String,
    pub timestamp: u64,
    pub content: String,
}

/// Age distribution of code lines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeDistribution {
    pub recent: usize,    // < 1 month
    pub moderate: usize,  // 1-6 months
    pub old: usize,       // 6-12 months
    pub ancient: usize,   // > 1 year
}

/// Git repository statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRepositoryStats {
    pub total_commits: usize,
    pub contributors: Vec<ContributorStats>,
    pub branches: Vec<String>,
    pub tags: Vec<String>,
    pub file_types: HashMap<String, usize>,
    pub activity_timeline: Vec<ActivityPeriod>,
    pub repository_health: RepositoryHealth,
}

/// Contributor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributorStats {
    pub name: String,
    pub email: String,
    pub commits: usize,
    pub lines_added: usize,
    pub lines_deleted: usize,
    pub files_modified: usize,
    pub first_commit: u64,
    pub last_commit: u64,
}

/// Activity period statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityPeriod {
    pub period: String, // e.g., "2024-01", "2024-W15"
    pub commits: usize,
    pub lines_changed: usize,
    pub files_touched: usize,
    pub contributors: HashSet<String>,
}

/// Repository health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryHealth {
    pub commit_frequency: f64,        // commits per day
    pub contributor_diversity: f64,   // number of active contributors
    pub code_churn: f64,             // lines changed / lines total
    pub documentation_ratio: f64,     // docs files / code files
    pub test_coverage_estimate: f64,  // test files / code files
    pub branch_health: BranchHealth,
}

/// Branch health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchHealth {
    pub main_branch: String,
    pub active_branches: usize,
    pub stale_branches: usize,
    pub merge_conflicts_risk: f64,
}

/// Git operations cache for performance
#[derive(Debug)]
struct GitCache {
    file_statuses: RefCell<HashMap<PathBuf, GitFileStatus>>,
    commit_cache: RefCell<HashMap<String, GitCommitInfo>>,
    blame_cache: RefCell<HashMap<PathBuf, GitBlameInfo>>,
    files_discovered: RefCell<usize>,
    cache_timestamp: RefCell<Option<SystemTime>>,
    cache_ttl: std::time::Duration,
}

impl Default for GitCache {
    fn default() -> Self {
        Self {
            file_statuses: RefCell::new(HashMap::new()),
            commit_cache: RefCell::new(HashMap::new()),
            blame_cache: RefCell::new(HashMap::new()),
            files_discovered: RefCell::new(0),
            cache_timestamp: RefCell::new(None),
            cache_ttl: std::time::Duration::from_secs(300),
        }
    }
}

impl GitIntegrator {
    /// Create a new Git integrator for the given repository path
    pub fn new<P: AsRef<Path>>(repo_path: P) -> Result<Self> {
        let repo_path = repo_path.as_ref().to_path_buf();
        
        // Verify this is a Git repository
        let git_dir = repo_path.join(".git");
        if !git_dir.exists() {
            return Err(ScribeError::git("Not a git repository".to_string()));
        }

        // Check if git command is available
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
                cache_ttl: std::time::Duration::from_secs(300), // 5 minutes
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
            .arg("-z") // null-separated output for safety
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

        // Update cache
        *self.cache.files_discovered.borrow_mut() = files.len();
        *self.cache.cache_timestamp.borrow_mut() = Some(SystemTime::now());

        log::debug!("Git discovered {} tracked files", files.len());
        Ok(files)
    }

    /// Get detailed file information including git status
    pub async fn get_file_info(&self, file_path: &Path) -> Result<GitFileInfo> {
        // Check cache first
        if let Some(cached_status) = self.cache.file_statuses.borrow().get(file_path) {
            if self.is_cache_valid() {
                return Ok(GitFileInfo {
                    path: file_path.to_path_buf(),
                    status: cached_status.clone(),
                    last_commit: None, // Would need to implement commit lookup
                    blame_info: self.cache.blame_cache.borrow().get(file_path).cloned(),
                    changes_count: 0,
                    additions: 0,
                    deletions: 0,
                });
            }
        }

        let status = self.get_file_status(file_path).await?;
        let last_commit = self.get_last_commit_for_file(file_path).await.ok();
        let blame_info = self.get_blame_info(file_path).await.ok();

        // Get file change statistics
        let (changes_count, additions, deletions) = self.get_file_change_stats(file_path).await
            .unwrap_or((0, 0, 0));

        // Cache the status and update timestamp
        self.cache.file_statuses.borrow_mut().insert(file_path.to_path_buf(), status.clone());
        *self.cache.cache_timestamp.borrow_mut() = Some(SystemTime::now());

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

    /// Get the current status of a file
    async fn get_file_status(&self, file_path: &Path) -> Result<GitFileStatus> {
        if !self.git_available {
            return Ok(GitFileStatus::Untracked);
        }

        let relative_path = file_path.strip_prefix(&self.repo_path)
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
                " M" => GitFileStatus::Modified,
                "M " => GitFileStatus::Modified,
                "MM" => GitFileStatus::Modified, // Modified after staging
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

    /// Get the last commit information for a file
    async fn get_last_commit_for_file(&self, file_path: &Path) -> Result<GitCommitInfo> {
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

        let relative_path = file_path.strip_prefix(&self.repo_path)
            .map_err(|_| ScribeError::git("File not in repository".to_string()))?;

        let output = AsyncCommand::new("git")
            .arg("log")
            .arg("-1")
            .arg("--pretty=format:%H|%an|%ae|%at|%s|%H") // hash|author|email|timestamp|subject|hash_again
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

        let timestamp = parts[3].parse::<u64>()
            .map_err(|_| ScribeError::git("Invalid timestamp".to_string()))?;

        Ok(GitCommitInfo {
            hash: parts[0].to_string(),
            author: parts[1].to_string(),
            email: parts[2].to_string(),
            timestamp,
            message: parts[4].to_string(),
            files_changed: 1, // Would need additional command to get accurate count
        })
    }

    /// Get blame information for a file
    async fn get_blame_info(&self, file_path: &Path) -> Result<GitBlameInfo> {
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

        // Check cache first
        if let Some(cached_blame) = self.cache.blame_cache.borrow().get(file_path) {
            if self.is_cache_valid() {
                return Ok(cached_blame.clone());
            }
        }

        let relative_path = file_path.strip_prefix(&self.repo_path)
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
        let blame_info = self.parse_blame_output(&stdout)?;

        Ok(blame_info)
    }

    /// Parse git blame porcelain output
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

            // Parse commit hash and line number from first line
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                i += 1;
                continue;
            }

            let commit_hash = parts[0].to_string();
            let line_number = parts[2].parse::<usize>().unwrap_or(0);

            // Parse additional information
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

            // Count lines per author
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

        // Calculate age distribution
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut age_distribution = AgeDistribution {
            recent: 0,
            moderate: 0,
            old: 0,
            ancient: 0,
        };

        for line in &lines {
            let age_seconds = now.saturating_sub(line.timestamp);
            let age_days = age_seconds / 86400; // seconds per day

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

    /// Get file change statistics (additions/deletions count)
    async fn get_file_change_stats(&self, file_path: &Path) -> Result<(usize, usize, usize)> {
        if !self.git_available {
            return Err(ScribeError::git("Git not available".to_string()));
        }

        let relative_path = file_path.strip_prefix(&self.repo_path)
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

    /// Get contributor statistics
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
                    
                    // Parse name and email
                    let (name, email) = if let Some((n, e)) = name_email.rsplit_once('<') {
                        let email = e.trim_end_matches('>');
                        (n.trim().to_string(), email.to_string())
                    } else {
                        (name_email.to_string(), String::new())
                    };

                    // Get additional stats for this contributor
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

        // Sort by commit count descending
        contributors.sort_by(|a, b| b.commits.cmp(&a.commits));

        Ok((total_commits, contributors))
    }

    /// Get detailed statistics for a specific contributor
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

            // Check if it's a timestamp line
            if let Ok(timestamp) = line.parse::<u64>() {
                timestamps.push(timestamp);
                continue;
            }

            // Check if it's a numstat line
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

    /// Get list of branches
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
        let branches = stdout
            .lines()
            .map(|line| line.trim_start_matches("* ").trim())
            .filter(|line| !line.is_empty())
            .map(|line| line.to_string())
            .collect();

        Ok(branches)
    }

    /// Get list of tags
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
        let tags = stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();

        Ok(tags)
    }

    /// Analyze file types in the repository
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

    /// Get activity timeline
    async fn get_activity_timeline(&self) -> Result<Vec<ActivityPeriod>> {
        // This would implement more sophisticated timeline analysis
        // For now, returning empty vector as placeholder
        Ok(vec![])
    }

    /// Calculate repository health metrics
    async fn calculate_repository_health(
        &self,
        contributors: &[ContributorStats],
        activity_timeline: &[ActivityPeriod],
    ) -> Result<RepositoryHealth> {
        // Calculate basic health metrics
        let commit_frequency = if !activity_timeline.is_empty() {
            let total_commits: usize = activity_timeline.iter().map(|p| p.commits).sum();
            total_commits as f64 / activity_timeline.len() as f64
        } else {
            0.0
        };

        let contributor_diversity = contributors.len() as f64;
        
        // Basic code churn calculation (would need more sophisticated analysis)
        let total_added: usize = contributors.iter().map(|c| c.lines_added).sum();
        let total_deleted: usize = contributors.iter().map(|c| c.lines_deleted).sum();
        let code_churn = if total_added > 0 {
            total_deleted as f64 / total_added as f64
        } else {
            0.0
        };

        // Placeholder values for other metrics
        let documentation_ratio = 0.0;
        let test_coverage_estimate = 0.0;

        let branch_health = BranchHealth {
            main_branch: "main".to_string(),
            active_branches: 1,
            stale_branches: 0,
            merge_conflicts_risk: 0.0,
        };

        Ok(RepositoryHealth {
            commit_frequency,
            contributor_diversity,
            code_churn,
            documentation_ratio,
            test_coverage_estimate,
            branch_health,
        })
    }

    /// Check if cache is still valid
    fn is_cache_valid(&self) -> bool {
        if let Some(cache_time) = *self.cache.cache_timestamp.borrow() {
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
        self.cache.file_statuses.borrow_mut().clear();
        self.cache.commit_cache.borrow_mut().clear();
        self.cache.blame_cache.borrow_mut().clear();
        *self.cache.cache_timestamp.borrow_mut() = None;
    }

    /// Get number of files discovered through git
    pub fn files_discovered(&self) -> usize {
        *self.cache.files_discovered.borrow()
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

impl Default for AgeDistribution {
    fn default() -> Self {
        Self {
            recent: 0,
            moderate: 0,
            old: 0,
            ancient: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use std::process::Command;

    async fn create_test_git_repo() -> Result<TempDir> {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Initialize git repo
        let output = Command::new("git")
            .arg("init")
            .current_dir(repo_path)
            .output();

        if output.is_err() || !output.unwrap().status.success() {
            // Skip tests if git is not available
            return Err(ScribeError::git("Git not available for testing".to_string()));
        }

        // Configure git for testing
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
        fs::write(&test_file, "fn main() { println!(\"Hello, world!\"); }").unwrap();

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

    #[tokio::test]
    async fn test_git_integrator_creation() {
        if let Ok(temp_dir) = create_test_git_repo().await {
            let integrator = GitIntegrator::new(temp_dir.path()).unwrap();
            assert!(integrator.is_git_available());
            assert_eq!(integrator.repo_path(), temp_dir.path());
        }
    }

    #[tokio::test]
    async fn test_list_tracked_files() {
        if let Ok(temp_dir) = create_test_git_repo().await {
            let integrator = GitIntegrator::new(temp_dir.path()).unwrap();
            let files = integrator.list_tracked_files().await.unwrap();
            
            assert_eq!(files.len(), 1);
            assert!(files[0].file_name().unwrap() == "test.rs");
            assert_eq!(integrator.files_discovered(), 1);
        }
    }

    #[tokio::test]
    async fn test_get_file_info() {
        if let Ok(temp_dir) = create_test_git_repo().await {
            let integrator = GitIntegrator::new(temp_dir.path()).unwrap();
            let test_file = temp_dir.path().join("test.rs");
            
            let file_info = integrator.get_file_info(&test_file).await.unwrap();
            
            assert_eq!(file_info.path, test_file);
            assert_eq!(file_info.status, GitFileStatus::Unmodified);
            assert!(file_info.last_commit.is_some());
        }
    }

    #[tokio::test]
    async fn test_get_repository_stats() {
        if let Ok(temp_dir) = create_test_git_repo().await {
            let integrator = GitIntegrator::new(temp_dir.path()).unwrap();
            let stats = integrator.get_repository_stats().await.unwrap();
            
            assert!(stats.total_commits >= 1);
            assert!(!stats.contributors.is_empty());
            assert!(stats.contributors[0].name == "Test User");
            assert!(stats.file_types.contains_key("rs"));
        }
    }

    #[tokio::test]
    async fn test_file_status_detection() {
        if let Ok(temp_dir) = create_test_git_repo().await {
            let integrator = GitIntegrator::new(temp_dir.path()).unwrap();
            let test_file = temp_dir.path().join("test.rs");
            
            // File should be tracked initially
            let status = integrator.get_file_status(&test_file).await.unwrap();
            assert_eq!(status, GitFileStatus::Unmodified);
            
            // Modify the file
            fs::write(&test_file, "fn main() { println!(\"Modified!\"); }").unwrap();
            
            let status = integrator.get_file_status(&test_file).await.unwrap();
            assert_eq!(status, GitFileStatus::Modified);
            
            // Create untracked file
            let new_file = temp_dir.path().join("untracked.rs");
            fs::write(&new_file, "// untracked").unwrap();
            
            let status = integrator.get_file_status(&new_file).await.unwrap();
            assert_eq!(status, GitFileStatus::Untracked);
        }
    }

    #[tokio::test]
    async fn test_blame_info() {
        if let Ok(temp_dir) = create_test_git_repo().await {
            let integrator = GitIntegrator::new(temp_dir.path()).unwrap();
            let test_file = temp_dir.path().join("test.rs");
            
            let blame_info = integrator.get_blame_info(&test_file).await.unwrap();
            
            assert_eq!(blame_info.lines.len(), 1);
            assert!(!blame_info.contributors.is_empty());
            assert!(blame_info.contributors.contains_key("Test User"));
            assert!(blame_info.last_modified > 0);
        }
    }

    #[test]
    fn test_age_distribution_calculation() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut age_dist = AgeDistribution::default();
        
        // Simulate line ages
        let recent_timestamp = now - (15 * 24 * 3600); // 15 days ago
        let moderate_timestamp = now - (90 * 24 * 3600); // 90 days ago
        let old_timestamp = now - (300 * 24 * 3600); // 300 days ago
        let ancient_timestamp = now - (400 * 24 * 3600); // 400 days ago
        
        let timestamps = vec![recent_timestamp, moderate_timestamp, old_timestamp, ancient_timestamp];
        
        for timestamp in timestamps {
            let age_seconds = now.saturating_sub(timestamp);
            let age_days = age_seconds / 86400;
            
            match age_days {
                0..=30 => age_dist.recent += 1,
                31..=180 => age_dist.moderate += 1,
                181..=365 => age_dist.old += 1,
                _ => age_dist.ancient += 1,
            }
        }
        
        assert_eq!(age_dist.recent, 1);
        assert_eq!(age_dist.moderate, 1);
        assert_eq!(age_dist.old, 1);
        assert_eq!(age_dist.ancient, 1);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        if let Ok(temp_dir) = create_test_git_repo().await {
            let mut integrator = GitIntegrator::new(temp_dir.path()).unwrap();
            let test_file = temp_dir.path().join("test.rs");
            
            // First call should populate cache
            let _ = integrator.get_file_info(&test_file).await.unwrap();
            assert!(integrator.is_cache_valid());
            
            // Clear cache
            integrator.clear_cache();
            assert!(!integrator.is_cache_valid());
        }
    }
}
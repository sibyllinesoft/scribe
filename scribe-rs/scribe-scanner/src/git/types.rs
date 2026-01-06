//! Git type definitions and data structures

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use scribe_core::GitFileStatus;

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
    pub contributors: HashMap<String, usize>,
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
    pub recent: usize,
    pub moderate: usize,
    pub old: usize,
    pub ancient: usize,
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
    pub period: String,
    pub commits: usize,
    pub lines_changed: usize,
    pub files_touched: usize,
    pub contributors: HashSet<String>,
}

/// Repository health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryHealth {
    pub commit_frequency: f64,
    pub contributor_diversity: f64,
    pub code_churn: f64,
    pub documentation_ratio: f64,
    pub test_coverage_estimate: f64,
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

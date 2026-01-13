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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_file_info_creation() {
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
        assert!(info.last_commit.is_none());
        assert!(info.blame_info.is_none());
        assert_eq!(info.changes_count, 5);
        assert_eq!(info.additions, 10);
        assert_eq!(info.deletions, 3);
    }

    #[test]
    fn test_git_file_info_clone() {
        let info = GitFileInfo {
            path: PathBuf::from("test.rs"),
            status: GitFileStatus::Added,
            last_commit: None,
            blame_info: None,
            changes_count: 1,
            additions: 100,
            deletions: 0,
        };

        let cloned = info.clone();
        assert_eq!(info.path, cloned.path);
        assert_eq!(info.status, cloned.status);
        assert_eq!(info.additions, cloned.additions);
    }

    #[test]
    fn test_git_file_info_debug() {
        let info = GitFileInfo {
            path: PathBuf::from("lib.rs"),
            status: GitFileStatus::Untracked,
            last_commit: None,
            blame_info: None,
            changes_count: 0,
            additions: 0,
            deletions: 0,
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("GitFileInfo"));
        assert!(debug_str.contains("lib.rs"));
    }

    #[test]
    fn test_git_file_info_serialize() {
        let info = GitFileInfo {
            path: PathBuf::from("src/lib.rs"),
            status: GitFileStatus::Deleted,
            last_commit: None,
            blame_info: None,
            changes_count: 2,
            additions: 0,
            deletions: 50,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("src/lib.rs"));
        assert!(json.contains("Deleted"));

        let deserialized: GitFileInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.path, info.path);
        assert_eq!(deserialized.deletions, 50);
    }

    #[test]
    fn test_git_commit_info_creation() {
        let commit = GitCommitInfo {
            hash: "abc123def456".to_string(),
            author: "John Doe".to_string(),
            email: "john@example.com".to_string(),
            timestamp: 1234567890,
            message: "Initial commit".to_string(),
            files_changed: 3,
        };

        assert_eq!(commit.hash, "abc123def456");
        assert_eq!(commit.author, "John Doe");
        assert_eq!(commit.email, "john@example.com");
        assert_eq!(commit.timestamp, 1234567890);
        assert_eq!(commit.message, "Initial commit");
        assert_eq!(commit.files_changed, 3);
    }

    #[test]
    fn test_git_commit_info_clone() {
        let commit = GitCommitInfo {
            hash: "123456".to_string(),
            author: "Dev".to_string(),
            email: "dev@test.com".to_string(),
            timestamp: 0,
            message: "Test".to_string(),
            files_changed: 1,
        };

        let cloned = commit.clone();
        assert_eq!(commit.hash, cloned.hash);
        assert_eq!(commit.author, cloned.author);
    }

    #[test]
    fn test_git_commit_info_serialize() {
        let commit = GitCommitInfo {
            hash: "abcdef".to_string(),
            author: "Test Author".to_string(),
            email: "test@test.com".to_string(),
            timestamp: 1000000,
            message: "Test message".to_string(),
            files_changed: 5,
        };

        let json = serde_json::to_string(&commit).unwrap();
        assert!(json.contains("abcdef"));
        assert!(json.contains("Test Author"));

        let deserialized: GitCommitInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.hash, commit.hash);
    }

    #[test]
    fn test_git_blame_info_creation() {
        let blame = GitBlameInfo {
            lines: vec![],
            contributors: HashMap::new(),
            last_modified: 1234567890,
            age_distribution: AgeDistribution::default(),
        };

        assert!(blame.lines.is_empty());
        assert!(blame.contributors.is_empty());
        assert_eq!(blame.last_modified, 1234567890);
    }

    #[test]
    fn test_git_blame_info_with_data() {
        let mut contributors = HashMap::new();
        contributors.insert("dev@test.com".to_string(), 100);

        let lines = vec![GitBlameLine {
            line_number: 1,
            commit_hash: "abc".to_string(),
            author: "dev".to_string(),
            timestamp: 123,
            content: "fn main()".to_string(),
        }];

        let blame = GitBlameInfo {
            lines,
            contributors,
            last_modified: 123456,
            age_distribution: AgeDistribution {
                recent: 10,
                moderate: 5,
                old: 3,
                ancient: 1,
            },
        };

        assert_eq!(blame.lines.len(), 1);
        assert_eq!(blame.contributors.get("dev@test.com"), Some(&100));
        assert_eq!(blame.age_distribution.recent, 10);
    }

    #[test]
    fn test_git_blame_line_creation() {
        let line = GitBlameLine {
            line_number: 42,
            commit_hash: "def789".to_string(),
            author: "Developer".to_string(),
            timestamp: 9999,
            content: "let x = 5;".to_string(),
        };

        assert_eq!(line.line_number, 42);
        assert_eq!(line.commit_hash, "def789");
        assert_eq!(line.author, "Developer");
        assert_eq!(line.timestamp, 9999);
        assert_eq!(line.content, "let x = 5;");
    }

    #[test]
    fn test_git_blame_line_serialize() {
        let line = GitBlameLine {
            line_number: 1,
            commit_hash: "hash".to_string(),
            author: "author".to_string(),
            timestamp: 100,
            content: "code".to_string(),
        };

        let json = serde_json::to_string(&line).unwrap();
        assert!(json.contains("hash"));
        assert!(json.contains("author"));

        let deserialized: GitBlameLine = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.line_number, 1);
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
    fn test_age_distribution_custom() {
        let dist = AgeDistribution {
            recent: 100,
            moderate: 50,
            old: 25,
            ancient: 10,
        };

        assert_eq!(dist.recent, 100);
        assert_eq!(dist.moderate, 50);
        assert_eq!(dist.old, 25);
        assert_eq!(dist.ancient, 10);
    }

    #[test]
    fn test_age_distribution_clone() {
        let dist = AgeDistribution {
            recent: 1,
            moderate: 2,
            old: 3,
            ancient: 4,
        };

        let cloned = dist.clone();
        assert_eq!(dist.recent, cloned.recent);
        assert_eq!(dist.moderate, cloned.moderate);
        assert_eq!(dist.old, cloned.old);
        assert_eq!(dist.ancient, cloned.ancient);
    }

    #[test]
    fn test_age_distribution_serialize() {
        let dist = AgeDistribution {
            recent: 10,
            moderate: 20,
            old: 30,
            ancient: 40,
        };

        let json = serde_json::to_string(&dist).unwrap();
        assert!(json.contains("10"));
        assert!(json.contains("40"));

        let deserialized: AgeDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.recent, 10);
        assert_eq!(deserialized.ancient, 40);
    }

    #[test]
    fn test_git_repository_stats_creation() {
        let stats = GitRepositoryStats {
            total_commits: 1000,
            contributors: vec![],
            branches: vec!["main".to_string()],
            tags: vec!["v1.0.0".to_string()],
            file_types: HashMap::new(),
            activity_timeline: vec![],
            repository_health: RepositoryHealth {
                commit_frequency: 5.0,
                contributor_diversity: 0.8,
                code_churn: 0.3,
                documentation_ratio: 0.15,
                test_coverage_estimate: 0.75,
                branch_health: BranchHealth {
                    main_branch: "main".to_string(),
                    active_branches: 3,
                    stale_branches: 1,
                    merge_conflicts_risk: 0.1,
                },
            },
        };

        assert_eq!(stats.total_commits, 1000);
        assert_eq!(stats.branches.len(), 1);
        assert_eq!(stats.tags.len(), 1);
        assert_eq!(stats.repository_health.commit_frequency, 5.0);
    }

    #[test]
    fn test_contributor_stats_creation() {
        let stats = ContributorStats {
            name: "Jane Developer".to_string(),
            email: "jane@example.com".to_string(),
            commits: 150,
            lines_added: 5000,
            lines_deleted: 1500,
            files_modified: 100,
            first_commit: 1000000,
            last_commit: 2000000,
        };

        assert_eq!(stats.name, "Jane Developer");
        assert_eq!(stats.email, "jane@example.com");
        assert_eq!(stats.commits, 150);
        assert_eq!(stats.lines_added, 5000);
        assert_eq!(stats.lines_deleted, 1500);
        assert_eq!(stats.files_modified, 100);
    }

    #[test]
    fn test_contributor_stats_serialize() {
        let stats = ContributorStats {
            name: "Dev".to_string(),
            email: "dev@test.com".to_string(),
            commits: 10,
            lines_added: 100,
            lines_deleted: 50,
            files_modified: 5,
            first_commit: 100,
            last_commit: 200,
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("Dev"));
        assert!(json.contains("dev@test.com"));

        let deserialized: ContributorStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "Dev");
        assert_eq!(deserialized.commits, 10);
    }

    #[test]
    fn test_activity_period_creation() {
        let mut contributors = HashSet::new();
        contributors.insert("dev1@test.com".to_string());
        contributors.insert("dev2@test.com".to_string());

        let period = ActivityPeriod {
            period: "2024-01".to_string(),
            commits: 50,
            lines_changed: 2500,
            files_touched: 30,
            contributors,
        };

        assert_eq!(period.period, "2024-01");
        assert_eq!(period.commits, 50);
        assert_eq!(period.lines_changed, 2500);
        assert_eq!(period.files_touched, 30);
        assert_eq!(period.contributors.len(), 2);
    }

    #[test]
    fn test_activity_period_serialize() {
        let period = ActivityPeriod {
            period: "2024-Q1".to_string(),
            commits: 100,
            lines_changed: 5000,
            files_touched: 50,
            contributors: HashSet::new(),
        };

        let json = serde_json::to_string(&period).unwrap();
        assert!(json.contains("2024-Q1"));

        let deserialized: ActivityPeriod = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.period, "2024-Q1");
        assert_eq!(deserialized.commits, 100);
    }

    #[test]
    fn test_repository_health_creation() {
        let health = RepositoryHealth {
            commit_frequency: 10.5,
            contributor_diversity: 0.9,
            code_churn: 0.2,
            documentation_ratio: 0.2,
            test_coverage_estimate: 0.85,
            branch_health: BranchHealth {
                main_branch: "master".to_string(),
                active_branches: 5,
                stale_branches: 2,
                merge_conflicts_risk: 0.05,
            },
        };

        assert_eq!(health.commit_frequency, 10.5);
        assert_eq!(health.contributor_diversity, 0.9);
        assert_eq!(health.code_churn, 0.2);
        assert_eq!(health.documentation_ratio, 0.2);
        assert_eq!(health.test_coverage_estimate, 0.85);
        assert_eq!(health.branch_health.main_branch, "master");
    }

    #[test]
    fn test_repository_health_serialize() {
        let health = RepositoryHealth {
            commit_frequency: 5.0,
            contributor_diversity: 0.5,
            code_churn: 0.1,
            documentation_ratio: 0.1,
            test_coverage_estimate: 0.5,
            branch_health: BranchHealth {
                main_branch: "main".to_string(),
                active_branches: 1,
                stale_branches: 0,
                merge_conflicts_risk: 0.0,
            },
        };

        let json = serde_json::to_string(&health).unwrap();
        assert!(json.contains("commit_frequency"));
        assert!(json.contains("main"));

        let deserialized: RepositoryHealth = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.commit_frequency, 5.0);
    }

    #[test]
    fn test_branch_health_creation() {
        let health = BranchHealth {
            main_branch: "develop".to_string(),
            active_branches: 10,
            stale_branches: 5,
            merge_conflicts_risk: 0.3,
        };

        assert_eq!(health.main_branch, "develop");
        assert_eq!(health.active_branches, 10);
        assert_eq!(health.stale_branches, 5);
        assert_eq!(health.merge_conflicts_risk, 0.3);
    }

    #[test]
    fn test_branch_health_clone() {
        let health = BranchHealth {
            main_branch: "main".to_string(),
            active_branches: 3,
            stale_branches: 1,
            merge_conflicts_risk: 0.1,
        };

        let cloned = health.clone();
        assert_eq!(health.main_branch, cloned.main_branch);
        assert_eq!(health.active_branches, cloned.active_branches);
    }

    #[test]
    fn test_branch_health_serialize() {
        let health = BranchHealth {
            main_branch: "trunk".to_string(),
            active_branches: 7,
            stale_branches: 3,
            merge_conflicts_risk: 0.15,
        };

        let json = serde_json::to_string(&health).unwrap();
        assert!(json.contains("trunk"));

        let deserialized: BranchHealth = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.main_branch, "trunk");
        assert_eq!(deserialized.active_branches, 7);
    }

    #[test]
    fn test_git_file_info_with_commit() {
        let commit = GitCommitInfo {
            hash: "abc123".to_string(),
            author: "Developer".to_string(),
            email: "dev@test.com".to_string(),
            timestamp: 123456,
            message: "Fix bug".to_string(),
            files_changed: 1,
        };

        let info = GitFileInfo {
            path: PathBuf::from("bugfix.rs"),
            status: GitFileStatus::Modified,
            last_commit: Some(commit),
            blame_info: None,
            changes_count: 1,
            additions: 5,
            deletions: 3,
        };

        assert!(info.last_commit.is_some());
        assert_eq!(info.last_commit.as_ref().unwrap().hash, "abc123");
    }

    #[test]
    fn test_repository_stats_with_file_types() {
        let mut file_types = HashMap::new();
        file_types.insert("rs".to_string(), 50);
        file_types.insert("toml".to_string(), 5);
        file_types.insert("md".to_string(), 10);

        let stats = GitRepositoryStats {
            total_commits: 500,
            contributors: vec![],
            branches: vec!["main".to_string(), "develop".to_string()],
            tags: vec!["v1.0.0".to_string(), "v1.0.1".to_string()],
            file_types,
            activity_timeline: vec![],
            repository_health: RepositoryHealth {
                commit_frequency: 1.0,
                contributor_diversity: 0.5,
                code_churn: 0.1,
                documentation_ratio: 0.1,
                test_coverage_estimate: 0.5,
                branch_health: BranchHealth {
                    main_branch: "main".to_string(),
                    active_branches: 2,
                    stale_branches: 0,
                    merge_conflicts_risk: 0.0,
                },
            },
        };

        assert_eq!(stats.file_types.get("rs"), Some(&50));
        assert_eq!(stats.branches.len(), 2);
        assert_eq!(stats.tags.len(), 2);
    }
}

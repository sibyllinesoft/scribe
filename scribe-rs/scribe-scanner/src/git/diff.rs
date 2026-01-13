//! Git diff analysis types and configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Git diff entry representing a single change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitDiffEntry {
    pub file_path: PathBuf,
    pub change_type: DiffChangeType,
    pub diff_content: String,
    pub line_additions: usize,
    pub line_deletions: usize,
    pub commit_hash: Option<String>,
    pub commit_message: Option<String>,
    pub author: Option<String>,
    pub timestamp: Option<u64>,
    pub old_file_path: Option<PathBuf>,
}

/// Type of diff change
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DiffChangeType {
    Added,
    Modified,
    Deleted,
    Renamed,
    Copied,
}

/// Configuration for diff-based analysis
#[derive(Debug, Clone)]
pub struct DiffAnalysisConfig {
    pub include_staged: bool,
    pub include_unstaged: bool,
    pub include_commits: Option<Vec<String>>,
    pub commit_range: Option<String>,
    pub branch_comparison: Option<String>,
    pub max_commits: usize,
    pub max_diff_size_kb: usize,
    pub ignore_patterns: Vec<String>,
    pub relevance_threshold: f64,
    pub include_binary_diffs: bool,
    pub include_generated_files: bool,
    pub max_lines_per_diff: usize,
}

impl Default for DiffAnalysisConfig {
    fn default() -> Self {
        Self {
            include_staged: true,
            include_unstaged: true,
            include_commits: None,
            commit_range: None,
            branch_comparison: None,
            max_commits: 50,
            max_diff_size_kb: 100,
            ignore_patterns: vec![
                "*.lock".to_string(),
                "*.log".to_string(),
                "*.tmp".to_string(),
                "*.cache".to_string(),
                "node_modules/*".to_string(),
                ".git/*".to_string(),
                "__pycache__/*".to_string(),
                "*.min.js".to_string(),
                "*.min.css".to_string(),
                "build/*".to_string(),
                "dist/*".to_string(),
            ],
            relevance_threshold: 0.1,
            include_binary_diffs: false,
            include_generated_files: false,
            max_lines_per_diff: 1000,
        }
    }
}

/// Diff analysis result containing all extracted changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffAnalysisResult {
    pub diffs: Vec<GitDiffEntry>,
    pub total_files_changed: usize,
    pub total_additions: usize,
    pub total_deletions: usize,
    pub commit_range_analyzed: Option<String>,
    pub analysis_timestamp: u64,
}

/// Source of diff information
#[derive(Debug)]
pub enum DiffSource {
    Staged,
    Unstaged,
    BranchComparison,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_diff_entry_creation() {
        let entry = GitDiffEntry {
            file_path: PathBuf::from("src/main.rs"),
            change_type: DiffChangeType::Modified,
            diff_content: "+fn main() {}".to_string(),
            line_additions: 5,
            line_deletions: 2,
            commit_hash: Some("abc123".to_string()),
            commit_message: Some("Initial commit".to_string()),
            author: Some("developer".to_string()),
            timestamp: Some(1234567890),
            old_file_path: None,
        };

        assert_eq!(entry.file_path, PathBuf::from("src/main.rs"));
        assert_eq!(entry.change_type, DiffChangeType::Modified);
        assert_eq!(entry.line_additions, 5);
        assert_eq!(entry.line_deletions, 2);
        assert_eq!(entry.commit_hash, Some("abc123".to_string()));
        assert_eq!(entry.commit_message, Some("Initial commit".to_string()));
        assert_eq!(entry.author, Some("developer".to_string()));
        assert_eq!(entry.timestamp, Some(1234567890));
        assert!(entry.old_file_path.is_none());
    }

    #[test]
    fn test_git_diff_entry_with_rename() {
        let entry = GitDiffEntry {
            file_path: PathBuf::from("src/new_name.rs"),
            change_type: DiffChangeType::Renamed,
            diff_content: String::new(),
            line_additions: 0,
            line_deletions: 0,
            commit_hash: None,
            commit_message: None,
            author: None,
            timestamp: None,
            old_file_path: Some(PathBuf::from("src/old_name.rs")),
        };

        assert_eq!(entry.change_type, DiffChangeType::Renamed);
        assert_eq!(entry.old_file_path, Some(PathBuf::from("src/old_name.rs")));
    }

    #[test]
    fn test_git_diff_entry_clone() {
        let entry = GitDiffEntry {
            file_path: PathBuf::from("test.rs"),
            change_type: DiffChangeType::Added,
            diff_content: "content".to_string(),
            line_additions: 10,
            line_deletions: 0,
            commit_hash: None,
            commit_message: None,
            author: None,
            timestamp: None,
            old_file_path: None,
        };

        let cloned = entry.clone();
        assert_eq!(entry.file_path, cloned.file_path);
        assert_eq!(entry.change_type, cloned.change_type);
        assert_eq!(entry.line_additions, cloned.line_additions);
    }

    #[test]
    fn test_git_diff_entry_debug() {
        let entry = GitDiffEntry {
            file_path: PathBuf::from("test.rs"),
            change_type: DiffChangeType::Deleted,
            diff_content: String::new(),
            line_additions: 0,
            line_deletions: 15,
            commit_hash: None,
            commit_message: None,
            author: None,
            timestamp: None,
            old_file_path: None,
        };

        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("GitDiffEntry"));
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_git_diff_entry_serialize() {
        let entry = GitDiffEntry {
            file_path: PathBuf::from("src/lib.rs"),
            change_type: DiffChangeType::Modified,
            diff_content: "+new line".to_string(),
            line_additions: 1,
            line_deletions: 0,
            commit_hash: Some("hash123".to_string()),
            commit_message: None,
            author: None,
            timestamp: None,
            old_file_path: None,
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("src/lib.rs"));
        assert!(json.contains("Modified"));
        assert!(json.contains("hash123"));

        let deserialized: GitDiffEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.file_path, entry.file_path);
        assert_eq!(deserialized.change_type, entry.change_type);
    }

    #[test]
    fn test_diff_change_type_variants() {
        assert_eq!(DiffChangeType::Added, DiffChangeType::Added);
        assert_eq!(DiffChangeType::Modified, DiffChangeType::Modified);
        assert_eq!(DiffChangeType::Deleted, DiffChangeType::Deleted);
        assert_eq!(DiffChangeType::Renamed, DiffChangeType::Renamed);
        assert_eq!(DiffChangeType::Copied, DiffChangeType::Copied);

        assert_ne!(DiffChangeType::Added, DiffChangeType::Modified);
        assert_ne!(DiffChangeType::Deleted, DiffChangeType::Renamed);
    }

    #[test]
    fn test_diff_change_type_clone() {
        let change_type = DiffChangeType::Modified;
        let cloned = change_type.clone();
        assert_eq!(change_type, cloned);
    }

    #[test]
    fn test_diff_change_type_debug() {
        let debug_added = format!("{:?}", DiffChangeType::Added);
        assert_eq!(debug_added, "Added");

        let debug_modified = format!("{:?}", DiffChangeType::Modified);
        assert_eq!(debug_modified, "Modified");

        let debug_deleted = format!("{:?}", DiffChangeType::Deleted);
        assert_eq!(debug_deleted, "Deleted");

        let debug_renamed = format!("{:?}", DiffChangeType::Renamed);
        assert_eq!(debug_renamed, "Renamed");

        let debug_copied = format!("{:?}", DiffChangeType::Copied);
        assert_eq!(debug_copied, "Copied");
    }

    #[test]
    fn test_diff_change_type_serialize() {
        let json = serde_json::to_string(&DiffChangeType::Added).unwrap();
        assert_eq!(json, "\"Added\"");

        let json = serde_json::to_string(&DiffChangeType::Modified).unwrap();
        assert_eq!(json, "\"Modified\"");

        let deserialized: DiffChangeType = serde_json::from_str("\"Deleted\"").unwrap();
        assert_eq!(deserialized, DiffChangeType::Deleted);
    }

    #[test]
    fn test_diff_analysis_config_default() {
        let config = DiffAnalysisConfig::default();

        assert!(config.include_staged);
        assert!(config.include_unstaged);
        assert!(config.include_commits.is_none());
        assert!(config.commit_range.is_none());
        assert!(config.branch_comparison.is_none());
        assert_eq!(config.max_commits, 50);
        assert_eq!(config.max_diff_size_kb, 100);
        assert_eq!(config.relevance_threshold, 0.1);
        assert!(!config.include_binary_diffs);
        assert!(!config.include_generated_files);
        assert_eq!(config.max_lines_per_diff, 1000);
    }

    #[test]
    fn test_diff_analysis_config_ignore_patterns() {
        let config = DiffAnalysisConfig::default();

        assert!(config.ignore_patterns.contains(&"*.lock".to_string()));
        assert!(config.ignore_patterns.contains(&"*.log".to_string()));
        assert!(config.ignore_patterns.contains(&"*.tmp".to_string()));
        assert!(config.ignore_patterns.contains(&"*.cache".to_string()));
        assert!(config.ignore_patterns.contains(&"node_modules/*".to_string()));
        assert!(config.ignore_patterns.contains(&".git/*".to_string()));
        assert!(config.ignore_patterns.contains(&"__pycache__/*".to_string()));
        assert!(config.ignore_patterns.contains(&"*.min.js".to_string()));
        assert!(config.ignore_patterns.contains(&"*.min.css".to_string()));
        assert!(config.ignore_patterns.contains(&"build/*".to_string()));
        assert!(config.ignore_patterns.contains(&"dist/*".to_string()));
    }

    #[test]
    fn test_diff_analysis_config_clone() {
        let config = DiffAnalysisConfig::default();
        let cloned = config.clone();

        assert_eq!(config.include_staged, cloned.include_staged);
        assert_eq!(config.max_commits, cloned.max_commits);
        assert_eq!(config.ignore_patterns.len(), cloned.ignore_patterns.len());
    }

    #[test]
    fn test_diff_analysis_config_debug() {
        let config = DiffAnalysisConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("DiffAnalysisConfig"));
        assert!(debug_str.contains("include_staged"));
        assert!(debug_str.contains("max_commits"));
    }

    #[test]
    fn test_diff_analysis_config_custom() {
        let config = DiffAnalysisConfig {
            include_staged: false,
            include_unstaged: true,
            include_commits: Some(vec!["abc123".to_string(), "def456".to_string()]),
            commit_range: Some("HEAD~5..HEAD".to_string()),
            branch_comparison: Some("main".to_string()),
            max_commits: 100,
            max_diff_size_kb: 500,
            ignore_patterns: vec!["*.log".to_string()],
            relevance_threshold: 0.5,
            include_binary_diffs: true,
            include_generated_files: true,
            max_lines_per_diff: 5000,
        };

        assert!(!config.include_staged);
        assert!(config.include_unstaged);
        assert_eq!(config.include_commits.as_ref().unwrap().len(), 2);
        assert_eq!(config.commit_range, Some("HEAD~5..HEAD".to_string()));
        assert_eq!(config.branch_comparison, Some("main".to_string()));
        assert_eq!(config.max_commits, 100);
        assert_eq!(config.max_diff_size_kb, 500);
        assert_eq!(config.ignore_patterns.len(), 1);
        assert_eq!(config.relevance_threshold, 0.5);
        assert!(config.include_binary_diffs);
        assert!(config.include_generated_files);
        assert_eq!(config.max_lines_per_diff, 5000);
    }

    #[test]
    fn test_diff_analysis_result_creation() {
        let result = DiffAnalysisResult {
            diffs: vec![],
            total_files_changed: 5,
            total_additions: 100,
            total_deletions: 50,
            commit_range_analyzed: Some("abc..def".to_string()),
            analysis_timestamp: 1234567890,
        };

        assert!(result.diffs.is_empty());
        assert_eq!(result.total_files_changed, 5);
        assert_eq!(result.total_additions, 100);
        assert_eq!(result.total_deletions, 50);
        assert_eq!(result.commit_range_analyzed, Some("abc..def".to_string()));
        assert_eq!(result.analysis_timestamp, 1234567890);
    }

    #[test]
    fn test_diff_analysis_result_with_diffs() {
        let entry = GitDiffEntry {
            file_path: PathBuf::from("main.rs"),
            change_type: DiffChangeType::Modified,
            diff_content: "diff content".to_string(),
            line_additions: 10,
            line_deletions: 5,
            commit_hash: None,
            commit_message: None,
            author: None,
            timestamp: None,
            old_file_path: None,
        };

        let result = DiffAnalysisResult {
            diffs: vec![entry],
            total_files_changed: 1,
            total_additions: 10,
            total_deletions: 5,
            commit_range_analyzed: None,
            analysis_timestamp: 0,
        };

        assert_eq!(result.diffs.len(), 1);
        assert_eq!(result.diffs[0].file_path, PathBuf::from("main.rs"));
    }

    #[test]
    fn test_diff_analysis_result_clone() {
        let result = DiffAnalysisResult {
            diffs: vec![],
            total_files_changed: 3,
            total_additions: 30,
            total_deletions: 10,
            commit_range_analyzed: None,
            analysis_timestamp: 12345,
        };

        let cloned = result.clone();
        assert_eq!(result.total_files_changed, cloned.total_files_changed);
        assert_eq!(result.total_additions, cloned.total_additions);
        assert_eq!(result.analysis_timestamp, cloned.analysis_timestamp);
    }

    #[test]
    fn test_diff_analysis_result_debug() {
        let result = DiffAnalysisResult {
            diffs: vec![],
            total_files_changed: 0,
            total_additions: 0,
            total_deletions: 0,
            commit_range_analyzed: None,
            analysis_timestamp: 0,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("DiffAnalysisResult"));
        assert!(debug_str.contains("total_files_changed"));
    }

    #[test]
    fn test_diff_analysis_result_serialize() {
        let result = DiffAnalysisResult {
            diffs: vec![],
            total_files_changed: 2,
            total_additions: 20,
            total_deletions: 10,
            commit_range_analyzed: Some("main..feature".to_string()),
            analysis_timestamp: 1000000,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("total_files_changed"));
        assert!(json.contains("main..feature"));

        let deserialized: DiffAnalysisResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_files_changed, result.total_files_changed);
        assert_eq!(deserialized.commit_range_analyzed, result.commit_range_analyzed);
    }

    #[test]
    fn test_diff_source_variants() {
        let _staged = DiffSource::Staged;
        let _unstaged = DiffSource::Unstaged;
        let _branch_comparison = DiffSource::BranchComparison;
    }

    #[test]
    fn test_diff_source_debug() {
        let staged_str = format!("{:?}", DiffSource::Staged);
        assert_eq!(staged_str, "Staged");

        let unstaged_str = format!("{:?}", DiffSource::Unstaged);
        assert_eq!(unstaged_str, "Unstaged");

        let branch_str = format!("{:?}", DiffSource::BranchComparison);
        assert_eq!(branch_str, "BranchComparison");
    }
}

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

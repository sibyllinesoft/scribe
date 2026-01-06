//! Internal analyzer types used during repository analysis.

use scribe_analysis::DocumentAnalysis;
use scribe_core::{file::is_test_path, FileInfo, FileType, GitFileStatus};

use crate::scoring::compute_priority_boost;

/// Context collected for a file during analysis
#[derive(Debug, Clone)]
pub(crate) struct AnalyzerContext {
    pub imports: Vec<String>,
    pub doc_analysis: Option<DocumentAnalysis>,
    pub has_examples: bool,
    pub is_entrypoint: bool,
    pub priority_boost: f64,
    pub content: Option<String>,
}

/// Internal representation of a file for the analyzer
#[derive(Debug, Clone)]
pub(crate) struct AnalyzerFile {
    pub path: String,
    pub relative_path: String,
    pub depth: usize,
    pub is_docs: bool,
    pub is_readme: bool,
    pub is_test: bool,
    pub is_entrypoint: bool,
    pub has_examples: bool,
    pub priority_boost: f64,
    pub churn_score: f64,
    pub centrality_score: f64,
    pub imports: Vec<String>,
    pub doc_analysis: Option<DocumentAnalysis>,
}

impl AnalyzerFile {
    pub fn from_file_info(file: &FileInfo, context: &AnalyzerContext) -> Self {
        let path_string = file.path.to_string_lossy().to_string();
        let relative = if file.relative_path.is_empty() {
            path_string.clone()
        } else {
            file.relative_path.clone()
        };
        let normalized_path = relative.replace('\\', "/");
        let depth = normalized_path.matches('/').count();
        let is_docs = matches!(file.file_type, FileType::Documentation { .. });
        let is_readme = normalized_path.to_lowercase().contains("readme");
        let is_test =
            matches!(file.file_type, FileType::Test { .. }) || is_test_path(&file.path);

        Self {
            path: path_string,
            relative_path: normalized_path,
            depth,
            is_docs,
            is_readme,
            is_test,
            is_entrypoint: context.is_entrypoint,
            has_examples: context.has_examples,
            priority_boost: context.priority_boost.min(1.0),
            churn_score: compute_churn_score(file),
            centrality_score: 0.0,
            imports: context.imports.clone(),
            doc_analysis: context.doc_analysis.clone(),
        }
    }
}

impl scribe_analysis::heuristics::ScanResult for AnalyzerFile {
    fn path(&self) -> &str {
        &self.path
    }

    fn relative_path(&self) -> &str {
        &self.relative_path
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn is_docs(&self) -> bool {
        self.is_docs
    }

    fn is_readme(&self) -> bool {
        self.is_readme
    }

    fn is_test(&self) -> bool {
        self.is_test
    }

    fn is_entrypoint(&self) -> bool {
        self.is_entrypoint
    }

    fn has_examples(&self) -> bool {
        self.has_examples
    }

    fn priority_boost(&self) -> f64 {
        self.priority_boost
    }

    fn churn_score(&self) -> f64 {
        self.churn_score
    }

    fn centrality_in(&self) -> f64 {
        self.centrality_score
    }

    fn imports(&self) -> Option<&[String]> {
        if self.imports.is_empty() {
            None
        } else {
            Some(&self.imports)
        }
    }

    fn doc_analysis(&self) -> Option<&DocumentAnalysis> {
        self.doc_analysis.as_ref()
    }
}

/// Compute churn score from git status
pub(crate) fn compute_churn_score(file: &FileInfo) -> f64 {
    match &file.git_status {
        Some(status) => match status.working_tree {
            GitFileStatus::Modified => 0.6,
            GitFileStatus::Added => 0.8,
            GitFileStatus::Deleted => 0.4,
            GitFileStatus::Renamed => 0.5,
            GitFileStatus::Copied => 0.45,
            GitFileStatus::Unmerged => 0.9,
            GitFileStatus::Untracked => 0.3,
            _ => 0.1,
        },
        None => 0.0,
    }
}

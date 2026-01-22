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
        let is_test = matches!(file.file_type, FileType::Test { .. }) || is_test_path(&file.path);

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

#[cfg(test)]
mod tests {
    use super::*;
    use scribe_analysis::heuristics::ScanResult;

    #[test]
    fn test_analyzer_context_creation() {
        let ctx = AnalyzerContext {
            imports: vec!["module".to_string()],
            doc_analysis: None,
            has_examples: true,
            is_entrypoint: true,
            priority_boost: 0.8,
            content: Some("content".to_string()),
        };

        assert_eq!(ctx.imports.len(), 1);
        assert!(ctx.has_examples);
        assert!(ctx.is_entrypoint);
        assert_eq!(ctx.priority_boost, 0.8);
    }

    #[test]
    fn test_analyzer_context_empty() {
        let ctx = AnalyzerContext {
            imports: vec![],
            doc_analysis: None,
            has_examples: false,
            is_entrypoint: false,
            priority_boost: 0.0,
            content: None,
        };

        assert!(ctx.imports.is_empty());
        assert!(!ctx.has_examples);
        assert!(!ctx.is_entrypoint);
        assert_eq!(ctx.priority_boost, 0.0);
        assert!(ctx.content.is_none());
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_path() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert_eq!(file.path(), "src/lib.rs");
        assert_eq!(file.relative_path(), "src/lib.rs");
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_depth() {
        let file = AnalyzerFile {
            path: "src/core/utils.rs".to_string(),
            relative_path: "src/core/utils.rs".to_string(),
            depth: 2,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert_eq!(file.depth(), 2);
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_docs() {
        let file = AnalyzerFile {
            path: "docs/guide.md".to_string(),
            relative_path: "docs/guide.md".to_string(),
            depth: 1,
            is_docs: true,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.3,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert!(file.is_docs());
        assert!(!file.is_readme());
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_readme() {
        let file = AnalyzerFile {
            path: "README.md".to_string(),
            relative_path: "README.md".to_string(),
            depth: 0,
            is_docs: true,
            is_readme: true,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.9,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert!(file.is_docs());
        assert!(file.is_readme());
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_test() {
        let file = AnalyzerFile {
            path: "tests/test_main.rs".to_string(),
            relative_path: "tests/test_main.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: true,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.4,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert!(file.is_test());
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_entrypoint() {
        let file = AnalyzerFile {
            path: "src/main.rs".to_string(),
            relative_path: "src/main.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: true,
            has_examples: false,
            priority_boost: 0.9,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert!(file.is_entrypoint());
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_examples() {
        let file = AnalyzerFile {
            path: "examples/basic.rs".to_string(),
            relative_path: "examples/basic.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: true,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert!(file.has_examples());
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_priority() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.75,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert_eq!(file.priority_boost(), 0.75);
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_churn() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.6,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert_eq!(file.churn_score(), 0.6);
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_centrality() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.8,
            imports: vec![],
            doc_analysis: None,
        };

        assert_eq!(file.centrality_in(), 0.8);
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_imports_some() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec!["std::collections".to_string()],
            doc_analysis: None,
        };

        assert!(file.imports().is_some());
        assert_eq!(file.imports().unwrap().len(), 1);
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_imports_none() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert!(file.imports().is_none());
    }

    #[test]
    fn test_analyzer_file_scan_result_trait_doc_analysis() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        assert!(file.doc_analysis().is_none());
    }

    #[test]
    fn test_analyzer_file_clone() {
        let file = AnalyzerFile {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            is_docs: false,
            is_readme: false,
            is_test: false,
            is_entrypoint: true,
            has_examples: false,
            priority_boost: 0.9,
            churn_score: 0.5,
            centrality_score: 0.3,
            imports: vec!["std".to_string()],
            doc_analysis: None,
        };

        let cloned = file.clone();
        assert_eq!(file.path, cloned.path);
        assert_eq!(file.is_entrypoint, cloned.is_entrypoint);
        assert_eq!(file.priority_boost, cloned.priority_boost);
    }

    #[test]
    fn test_compute_churn_score_modified() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/lib.rs"),
            relative_path: "src/lib.rs".to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Modified,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.6);
    }

    #[test]
    fn test_compute_churn_score_added() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/new.rs"),
            relative_path: "src/new.rs".to_string(),
            size: 50,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Added,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.8);
    }

    #[test]
    fn test_compute_churn_score_deleted() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/old.rs"),
            relative_path: "src/old.rs".to_string(),
            size: 0,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Deleted,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.4);
    }

    #[test]
    fn test_compute_churn_score_renamed() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/renamed.rs"),
            relative_path: "src/renamed.rs".to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Renamed,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.5);
    }

    #[test]
    fn test_compute_churn_score_copied() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/copy.rs"),
            relative_path: "src/copy.rs".to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Copied,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.45);
    }

    #[test]
    fn test_compute_churn_score_unmerged() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/conflict.rs"),
            relative_path: "src/conflict.rs".to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Unmerged,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.9);
    }

    #[test]
    fn test_compute_churn_score_untracked() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/new_untracked.rs"),
            relative_path: "src/new_untracked.rs".to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Untracked,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.3);
    }

    #[test]
    fn test_compute_churn_score_unmodified() {
        use scribe_core::{file::FileWeight, FileType, GitStatus, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/unchanged.rs"),
            relative_path: "src/unchanged.rs".to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Unmodified,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.1);
    }

    #[test]
    fn test_compute_churn_score_no_status() {
        use scribe_core::{file::FileWeight, FileType, Language, RenderDecision};
        use std::path::PathBuf;

        let file_info = FileInfo {
            path: PathBuf::from("src/no_git.rs"),
            relative_path: "src/no_git.rs".to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        };

        let score = compute_churn_score(&file_info);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_analyzer_file_debug() {
        let file = AnalyzerFile {
            path: "test.rs".to_string(),
            relative_path: "test.rs".to_string(),
            depth: 0,
            is_docs: false,
            is_readme: false,
            is_test: true,
            is_entrypoint: false,
            has_examples: false,
            priority_boost: 0.5,
            churn_score: 0.0,
            centrality_score: 0.0,
            imports: vec![],
            doc_analysis: None,
        };

        let debug_str = format!("{:?}", file);
        assert!(debug_str.contains("AnalyzerFile"));
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_analyzer_context_clone() {
        let ctx = AnalyzerContext {
            imports: vec!["std".to_string()],
            doc_analysis: None,
            has_examples: true,
            is_entrypoint: true,
            priority_boost: 0.9,
            content: Some("code".to_string()),
        };

        let cloned = ctx.clone();
        assert_eq!(ctx.imports, cloned.imports);
        assert_eq!(ctx.has_examples, cloned.has_examples);
        assert_eq!(ctx.is_entrypoint, cloned.is_entrypoint);
    }

    #[test]
    fn test_analyzer_context_debug() {
        let ctx = AnalyzerContext {
            imports: vec![],
            doc_analysis: None,
            has_examples: false,
            is_entrypoint: false,
            priority_boost: 0.0,
            content: None,
        };

        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("AnalyzerContext"));
    }
}

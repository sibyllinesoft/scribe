//! Code selector module that wraps the token budget selection pipeline.
//! The selector consumes `FileInfo` records produced by the scanner and applies
//! the multi-tier token budget logic from the legacy CLI.

use crate::budget::token_budget::{apply_token_budget_selection, SelectionConfig};
use crate::budget::weighting::FileWeights;
use scribe_core::{Config, FileInfo, Result};

/// Input parameters for running the selector.
#[derive(Debug, Clone)]
pub struct SelectionCriteria<'a> {
    /// Files produced by the scanner that are eligible for selection.
    pub files: Vec<FileInfo>,
    /// Maximum number of tokens that may be emitted across the selected files.
    pub token_budget: usize,
    /// Analyzer configuration that influences downstream decisions (e.g. demotion options).
    pub config: &'a Config,
    /// Optional external weights for file prioritization.
    pub weights: Option<&'a FileWeights>,
}

/// Result returned by the selector.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Files selected within the provided budget (with content/token metadata loaded).
    pub files: Vec<FileInfo>,
    /// Total tokens consumed by the selected files.
    pub total_tokens_used: usize,
    /// Budget that was provided to the selector.
    pub budget: usize,
    /// Tokens left unused after selection completed.
    pub unused_tokens: usize,
    /// Total number of files that were considered when running selection.
    pub total_files_considered: usize,
}

impl SelectionResult {
    /// Convenience helper returning the relative paths of all selected files.
    pub fn file_paths(&self) -> Vec<String> {
        self.files.iter().map(|f| f.relative_path.clone()).collect()
    }
}

pub struct CodeSelector;

impl CodeSelector {
    pub fn new() -> Self {
        Self
    }

    pub async fn select(&self, criteria: SelectionCriteria<'_>) -> Result<SelectionResult> {
        let total_files_considered = criteria.files.len();
        let budget = criteria.token_budget;

        let files = apply_token_budget_selection(
            criteria.files,
            criteria.token_budget,
            criteria.config,
            criteria.weights,
            &SelectionConfig::default(),
        )
        .await?;

        let total_tokens_used: usize = files.iter().filter_map(|f| f.token_estimate).sum();
        let unused_tokens = budget.saturating_sub(total_tokens_used);

        Ok(SelectionResult {
            files,
            total_tokens_used,
            budget,
            unused_tokens,
            total_files_considered,
        })
    }
}

impl Default for CodeSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scribe_core::{FileType, Language, RenderDecision};
    use std::path::PathBuf;

    fn create_test_file(path: &str, tokens: usize) -> FileInfo {
        FileInfo {
            path: PathBuf::from(format!("/repo/{}", path)),
            relative_path: path.to_string(),
            content: Some(format!("// content of {}", path)),
            token_estimate: Some(tokens),
            size: 100,
            is_binary: false,
            language: Language::Rust,
            modified: None,
            file_type: FileType::Source { language: Language::Rust },
            decision: RenderDecision::include("test"),
            centrality_score: None,
            line_count: None,
            char_count: None,
            git_status: None,
            weight: Default::default(),
        }
    }

    #[test]
    fn test_selection_result_file_paths() {
        let result = SelectionResult {
            files: vec![
                create_test_file("src/main.rs", 100),
                create_test_file("src/lib.rs", 50),
            ],
            total_tokens_used: 150,
            budget: 200,
            unused_tokens: 50,
            total_files_considered: 5,
        };

        let paths = result.file_paths();
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&"src/main.rs".to_string()));
        assert!(paths.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn test_selection_result_file_paths_empty() {
        let result = SelectionResult {
            files: vec![],
            total_tokens_used: 0,
            budget: 100,
            unused_tokens: 100,
            total_files_considered: 0,
        };

        assert!(result.file_paths().is_empty());
    }

    #[test]
    fn test_code_selector_default() {
        let selector = CodeSelector::default();
        let _ = selector;
    }

    #[test]
    fn test_code_selector_new() {
        let selector = CodeSelector::new();
        let _ = selector;
    }

    #[test]
    fn test_selection_result_structure() {
        let result = SelectionResult {
            files: vec![create_test_file("test.rs", 50)],
            total_tokens_used: 50,
            budget: 100,
            unused_tokens: 50,
            total_files_considered: 10,
        };

        assert_eq!(result.files.len(), 1);
        assert_eq!(result.total_tokens_used, 50);
        assert_eq!(result.budget, 100);
        assert_eq!(result.unused_tokens, 50);
        assert_eq!(result.total_files_considered, 10);
    }

    #[test]
    fn test_selection_criteria_structure() {
        let config = Config::default();
        let files = vec![create_test_file("test.rs", 100)];

        let criteria = SelectionCriteria {
            files,
            token_budget: 1000,
            config: &config,
            weights: None,
        };

        assert_eq!(criteria.files.len(), 1);
        assert_eq!(criteria.token_budget, 1000);
        assert!(criteria.weights.is_none());
    }

    #[test]
    fn test_selection_criteria_clone() {
        let config = Config::default();
        let files = vec![create_test_file("test.rs", 100)];

        let criteria = SelectionCriteria {
            files,
            token_budget: 1000,
            config: &config,
            weights: None,
        };

        let cloned = criteria.clone();
        assert_eq!(cloned.token_budget, 1000);
    }

    #[test]
    fn test_selection_criteria_debug() {
        let config = Config::default();
        let criteria = SelectionCriteria {
            files: vec![],
            token_budget: 500,
            config: &config,
            weights: None,
        };

        let debug_str = format!("{:?}", criteria);
        assert!(debug_str.contains("SelectionCriteria"));
    }

    #[test]
    fn test_selection_result_clone() {
        let result = SelectionResult {
            files: vec![create_test_file("test.rs", 50)],
            total_tokens_used: 50,
            budget: 100,
            unused_tokens: 50,
            total_files_considered: 10,
        };

        let cloned = result.clone();
        assert_eq!(result.budget, cloned.budget);
        assert_eq!(result.total_tokens_used, cloned.total_tokens_used);
    }

    #[test]
    fn test_selection_result_debug() {
        let result = SelectionResult {
            files: vec![],
            total_tokens_used: 0,
            budget: 100,
            unused_tokens: 100,
            total_files_considered: 0,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("SelectionResult"));
    }

    #[tokio::test]
    async fn test_code_selector_select_empty() {
        let selector = CodeSelector::new();
        let config = Config::default();

        let criteria = SelectionCriteria {
            files: vec![],
            token_budget: 1000,
            config: &config,
            weights: None,
        };

        let result = selector.select(criteria).await.unwrap();
        assert!(result.files.is_empty());
        assert_eq!(result.total_tokens_used, 0);
        assert_eq!(result.unused_tokens, 1000);
    }

    #[tokio::test]
    async fn test_code_selector_select_with_files() {
        let selector = CodeSelector::new();
        let config = Config::default();

        let files = vec![
            create_test_file("src/main.rs", 100),
            create_test_file("src/lib.rs", 200),
        ];

        let criteria = SelectionCriteria {
            files,
            token_budget: 500,
            config: &config,
            weights: None,
        };

        let result = selector.select(criteria).await.unwrap();
        // Result should have selected files within budget
        assert!(result.total_tokens_used <= 500);
        assert_eq!(result.budget, 500);
    }

    #[tokio::test]
    async fn test_code_selector_select_budget_constraint() {
        let selector = CodeSelector::new();
        let config = Config::default();

        // Create files that exceed budget
        let files = vec![
            create_test_file("big1.rs", 400),
            create_test_file("big2.rs", 400),
            create_test_file("big3.rs", 400),
        ];

        let criteria = SelectionCriteria {
            files,
            token_budget: 500,
            config: &config,
            weights: None,
        };

        let result = selector.select(criteria).await.unwrap();
        // Should not exceed budget
        assert!(result.total_tokens_used <= 500);
        assert_eq!(result.total_files_considered, 3);
    }
}

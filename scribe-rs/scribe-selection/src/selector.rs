//! Code selector module that wraps the token budget selection pipeline.
//! The selector consumes `FileInfo` records produced by the scanner and applies
//! the multi-tier token budget logic from the legacy CLI.

use crate::token_budget::apply_token_budget_selection;
use crate::weighting::FileWeights;
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

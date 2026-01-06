//! Context extraction module that prepares selected files for bundling.

use crate::core::selector::SelectionResult;
use scribe_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOptions {
    /// When true, dependency metadata will be included if available.
    pub include_dependencies: bool,
    /// Maximum depth for dependency expansion (currently unused placeholder).
    pub max_depth: usize,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            include_dependencies: true,
            max_depth: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFile {
    /// Path relative to the repository root.
    pub path: String,
    /// File contents if they were loaded during selection.
    pub contents: Option<String>,
    /// Token estimate computed for the file.
    pub token_estimate: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContext {
    /// Files that will be bundled.
    pub files: Vec<ContextFile>,
    /// Dependency metadata (currently empty until graph extraction lands).
    pub dependencies: Vec<String>,
    /// Total tokens consumed by the selected files.
    pub total_tokens: usize,
}

pub struct ContextExtractor;

impl ContextExtractor {
    pub fn new() -> Self {
        Self
    }

    pub async fn extract(
        &self,
        selection: &SelectionResult,
        _options: &ContextOptions,
    ) -> Result<CodeContext> {
        let files = selection
            .files
            .iter()
            .map(|file| ContextFile {
                path: file.relative_path.clone(),
                contents: file.content.clone(),
                token_estimate: file.token_estimate,
            })
            .collect();

        Ok(CodeContext {
            files,
            dependencies: Vec::new(),
            total_tokens: selection.total_tokens_used,
        })
    }
}

impl Default for ContextExtractor {
    fn default() -> Self {
        Self::new()
    }
}

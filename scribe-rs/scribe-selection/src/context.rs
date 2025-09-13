//! Context extraction module - stub implementation

use scribe_core::Result;
use crate::selector::SelectionResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOptions {
    pub include_dependencies: bool,
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
pub struct CodeContext {
    pub files: Vec<String>,
    pub dependencies: Vec<String>,
}

pub struct ContextExtractor;

impl ContextExtractor {
    pub fn new() -> Self {
        Self
    }

    pub async fn extract(
        &self,
        _selection: &SelectionResult,
        _options: &ContextOptions
    ) -> Result<CodeContext> {
        // Stub implementation
        Ok(CodeContext {
            files: vec![],
            dependencies: vec![],
        })
    }
}

impl Default for ContextExtractor {
    fn default() -> Self {
        Self::new()
    }
}
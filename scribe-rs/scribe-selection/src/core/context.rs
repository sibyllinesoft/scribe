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

#[cfg(test)]
mod tests {
    use super::*;
    use scribe_core::{FileInfo, FileType, Language, RenderDecision};
    use std::path::PathBuf;

    fn create_test_file(path: &str, rel_path: &str, content: &str, tokens: usize, size: u64) -> FileInfo {
        FileInfo {
            path: PathBuf::from(path),
            relative_path: rel_path.to_string(),
            content: Some(content.to_string()),
            token_estimate: Some(tokens),
            size,
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

    fn create_test_selection_result() -> SelectionResult {
        SelectionResult {
            files: vec![
                create_test_file("/repo/src/main.rs", "src/main.rs", "fn main() {}", 10, 12),
                create_test_file("/repo/src/lib.rs", "src/lib.rs", "pub mod utils;", 5, 15),
            ],
            total_tokens_used: 15,
            budget: 100,
            unused_tokens: 85,
            total_files_considered: 5,
        }
    }

    #[test]
    fn test_context_options_default() {
        let options = ContextOptions::default();
        assert!(options.include_dependencies);
        assert_eq!(options.max_depth, 3);
    }

    #[test]
    fn test_context_file_structure() {
        let file = ContextFile {
            path: "src/main.rs".to_string(),
            contents: Some("fn main() {}".to_string()),
            token_estimate: Some(100),
        };

        assert_eq!(file.path, "src/main.rs");
        assert_eq!(file.contents, Some("fn main() {}".to_string()));
        assert_eq!(file.token_estimate, Some(100));
    }

    #[test]
    fn test_code_context_structure() {
        let context = CodeContext {
            files: vec![ContextFile {
                path: "test.rs".to_string(),
                contents: None,
                token_estimate: None,
            }],
            dependencies: vec!["dep1".to_string(), "dep2".to_string()],
            total_tokens: 50,
        };

        assert_eq!(context.files.len(), 1);
        assert_eq!(context.dependencies.len(), 2);
        assert_eq!(context.total_tokens, 50);
    }

    #[test]
    fn test_context_extractor_default() {
        let extractor = ContextExtractor::default();
        let _ = extractor;
    }

    #[test]
    fn test_context_extractor_new() {
        let extractor = ContextExtractor::new();
        let _ = extractor;
    }

    #[tokio::test]
    async fn test_extract_context() {
        let extractor = ContextExtractor::new();
        let selection = create_test_selection_result();
        let options = ContextOptions::default();

        let context = extractor.extract(&selection, &options).await.unwrap();

        assert_eq!(context.files.len(), 2);
        assert_eq!(context.total_tokens, 15);
        assert!(context.dependencies.is_empty()); // Currently always empty

        // Check first file
        assert_eq!(context.files[0].path, "src/main.rs");
        assert_eq!(context.files[0].contents, Some("fn main() {}".to_string()));
        assert_eq!(context.files[0].token_estimate, Some(10));

        // Check second file
        assert_eq!(context.files[1].path, "src/lib.rs");
        assert_eq!(context.files[1].contents, Some("pub mod utils;".to_string()));
        assert_eq!(context.files[1].token_estimate, Some(5));
    }

    #[tokio::test]
    async fn test_extract_empty_selection() {
        let extractor = ContextExtractor::new();
        let selection = SelectionResult {
            files: vec![],
            total_tokens_used: 0,
            budget: 100,
            unused_tokens: 100,
            total_files_considered: 0,
        };
        let options = ContextOptions::default();

        let context = extractor.extract(&selection, &options).await.unwrap();

        assert!(context.files.is_empty());
        assert_eq!(context.total_tokens, 0);
    }
}

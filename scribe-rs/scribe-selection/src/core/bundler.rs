//! Code bundler module responsible for turning a selection context into
//! consumable artifacts. The implementation focuses on two lightweight
//! formats (JSON and plain-text) to unblock the web service and CLI.

use crate::core::context::CodeContext;
use scribe_core::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleOptions {
    /// Output format (`json` or `plain`). Defaults to JSON.
    pub format: String,
    /// Whether to include metadata describing the produced bundle.
    pub include_metadata: bool,
}

impl Default for BundleOptions {
    fn default() -> Self {
        Self {
            format: "json".to_string(),
            include_metadata: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBundle {
    /// Serialized bundle in the requested format.
    pub content: String,
    /// Additional metadata describing the bundle.
    pub metadata: HashMap<String, String>,
}

pub struct CodeBundler;

impl CodeBundler {
    pub fn new() -> Self {
        Self
    }

    pub async fn bundle(
        &self,
        context: &CodeContext,
        options: &BundleOptions,
    ) -> Result<CodeBundle> {
        let content = match options.format.as_str() {
            "plain" => render_plain(context),
            _ => render_json(context)?,
        };

        let metadata = if options.include_metadata {
            build_metadata(context, options)
        } else {
            HashMap::new()
        };

        Ok(CodeBundle { content, metadata })
    }
}

impl Default for CodeBundler {
    fn default() -> Self {
        Self::new()
    }
}

fn render_json(context: &CodeContext) -> Result<String> {
    let files: Vec<_> = context
        .files
        .iter()
        .map(|file| {
            json!({
                "path": file.path,
                "token_estimate": file.token_estimate,
                "contents": file.contents,
            })
        })
        .collect();

    Ok(serde_json::to_string_pretty(&json!({
        "total_tokens": context.total_tokens,
        "files": files,
    }))?)
}

fn render_plain(context: &CodeContext) -> String {
    let mut body = String::new();

    for file in &context.files {
        body.push_str("===== ");
        body.push_str(&file.path);
        body.push_str(" =====\n");

        match &file.contents {
            Some(contents) => body.push_str(contents),
            None => body.push_str("[content not loaded]\n"),
        }

        body.push_str("\n\n");
    }

    body
}

fn build_metadata(context: &CodeContext, options: &BundleOptions) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), options.format.clone());
    metadata.insert("file_count".to_string(), context.files.len().to_string());
    metadata.insert("total_tokens".to_string(), context.total_tokens.to_string());
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context::ContextFile;

    fn create_test_context() -> CodeContext {
        CodeContext {
            files: vec![
                ContextFile {
                    path: "src/main.rs".to_string(),
                    contents: Some("fn main() {}".to_string()),
                    token_estimate: Some(10),
                },
                ContextFile {
                    path: "src/lib.rs".to_string(),
                    contents: Some("pub mod utils;".to_string()),
                    token_estimate: Some(5),
                },
            ],
            dependencies: vec!["dep1".to_string()],
            total_tokens: 15,
        }
    }

    #[test]
    fn test_bundle_options_default() {
        let options = BundleOptions::default();
        assert_eq!(options.format, "json");
        assert!(options.include_metadata);
    }

    #[test]
    fn test_code_bundler_default() {
        let bundler = CodeBundler::default();
        // Just verify it can be created
        let _ = bundler;
    }

    #[test]
    fn test_code_bundler_new() {
        let bundler = CodeBundler::new();
        let _ = bundler;
    }

    #[test]
    fn test_render_json() {
        let context = create_test_context();
        let result = render_json(&context).unwrap();

        assert!(result.contains("\"total_tokens\": 15"));
        assert!(result.contains("\"path\": \"src/main.rs\""));
        assert!(result.contains("\"path\": \"src/lib.rs\""));
        assert!(result.contains("fn main()"));
        assert!(result.contains("pub mod utils"));
    }

    #[test]
    fn test_render_plain() {
        let context = create_test_context();
        let result = render_plain(&context);

        assert!(result.contains("===== src/main.rs ====="));
        assert!(result.contains("fn main() {}"));
        assert!(result.contains("===== src/lib.rs ====="));
        assert!(result.contains("pub mod utils;"));
    }

    #[test]
    fn test_render_plain_no_content() {
        let context = CodeContext {
            files: vec![ContextFile {
                path: "src/empty.rs".to_string(),
                contents: None,
                token_estimate: None,
            }],
            dependencies: vec![],
            total_tokens: 0,
        };

        let result = render_plain(&context);
        assert!(result.contains("[content not loaded]"));
    }

    #[test]
    fn test_build_metadata() {
        let context = create_test_context();
        let options = BundleOptions::default();

        let metadata = build_metadata(&context, &options);

        assert_eq!(metadata.get("format"), Some(&"json".to_string()));
        assert_eq!(metadata.get("file_count"), Some(&"2".to_string()));
        assert_eq!(metadata.get("total_tokens"), Some(&"15".to_string()));
    }

    #[tokio::test]
    async fn test_bundle_json_format() {
        let bundler = CodeBundler::new();
        let context = create_test_context();
        let options = BundleOptions {
            format: "json".to_string(),
            include_metadata: true,
        };

        let bundle = bundler.bundle(&context, &options).await.unwrap();

        assert!(bundle.content.contains("\"total_tokens\""));
        assert!(bundle.content.contains("src/main.rs"));
        assert_eq!(bundle.metadata.get("format"), Some(&"json".to_string()));
    }

    #[tokio::test]
    async fn test_bundle_plain_format() {
        let bundler = CodeBundler::new();
        let context = create_test_context();
        let options = BundleOptions {
            format: "plain".to_string(),
            include_metadata: true,
        };

        let bundle = bundler.bundle(&context, &options).await.unwrap();

        assert!(bundle.content.contains("====="));
        assert!(bundle.content.contains("fn main()"));
        assert_eq!(bundle.metadata.get("format"), Some(&"plain".to_string()));
    }

    #[tokio::test]
    async fn test_bundle_no_metadata() {
        let bundler = CodeBundler::new();
        let context = create_test_context();
        let options = BundleOptions {
            format: "json".to_string(),
            include_metadata: false,
        };

        let bundle = bundler.bundle(&context, &options).await.unwrap();

        assert!(bundle.metadata.is_empty());
    }
}

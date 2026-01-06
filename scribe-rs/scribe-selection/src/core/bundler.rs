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

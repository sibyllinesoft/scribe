//! Shared utility functions for file classification and processing.

use scribe_core::file::{self, FileInfo, FileType};
use std::path::Path;

/// Classify a file type to a string category based on its path and extension.
///
/// This is a shared implementation used across the scaling crate to ensure
/// consistent file categorization.
pub fn classify_file_type_string(path: &Path) -> String {
    let extension = get_lowercase_extension(path);
    let language = file::detect_language_from_path(path);
    let file_type =
        FileInfo::classify_file_type(path.to_string_lossy().as_ref(), &language, &extension);

    file_type_to_category(&file_type, &extension)
}

/// Get the lowercase extension from a path.
fn get_lowercase_extension(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default()
}

/// Convert a FileType to a category string.
fn file_type_to_category(file_type: &FileType, extension: &str) -> String {
    match file_type {
        FileType::Test { .. } => "Test".to_string(),
        FileType::Documentation { .. } => "Documentation".to_string(),
        FileType::Configuration { .. } => "Configuration".to_string(),
        FileType::Binary => "Binary".to_string(),
        FileType::Generated => "Generated".to_string(),
        FileType::Source { .. } => classify_source_extension(extension),
        FileType::Unknown => classify_unknown_extension(extension),
    }
}

/// Classify source files by extension for more specific categorization.
fn classify_source_extension(extension: &str) -> String {
    match extension {
        "jsx" | "tsx" | "vue" | "svelte" => "Frontend".to_string(),
        "html" | "htm" | "css" | "scss" | "sass" | "less" => "Web".to_string(),
        "sh" | "bash" | "bat" | "ps1" => "Script".to_string(),
        "h" | "hpp" | "hxx" => "Header".to_string(),
        _ => "Source".to_string(),
    }
}

/// Classify unknown file types by extension.
fn classify_unknown_extension(extension: &str) -> String {
    match extension {
        "md" | "txt" | "rst" | "adoc" => "Documentation".to_string(),
        "json" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" => "Configuration".to_string(),
        "png" | "jpg" | "jpeg" | "gif" | "svg" | "ico" => "Image".to_string(),
        "pdf" | "doc" | "docx" | "ppt" | "pptx" => "Document".to_string(),
        "sql" => "Database".to_string(),
        "xml" | "xsd" | "xsl" => "Markup".to_string(),
        _ => "Other".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_classify_rust_file() {
        let path = PathBuf::from("src/main.rs");
        let result = classify_file_type_string(&path);
        assert_eq!(result, "Source");
    }

    #[test]
    fn test_classify_test_file() {
        let path = PathBuf::from("tests/integration_test.rs");
        let result = classify_file_type_string(&path);
        assert_eq!(result, "Test");
    }

    #[test]
    fn test_classify_config_file() {
        let path = PathBuf::from("config.yaml");
        let result = classify_file_type_string(&path);
        assert_eq!(result, "Configuration");
    }

    #[test]
    fn test_classify_js_file() {
        let path = PathBuf::from("src/index.js");
        let result = classify_file_type_string(&path);
        // JavaScript files should be classified as Source
        assert_eq!(result, "Source");
    }
}

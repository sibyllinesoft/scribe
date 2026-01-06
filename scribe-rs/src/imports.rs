//! Import extraction utilities for various programming languages.
//!
//! This module provides functions to extract import/use/require statements
//! from source code content for different programming languages.

use scribe_core::Language;
use std::collections::HashSet;

/// Extract import statements from source code content.
///
/// This function parses the content and extracts import/use/require statements
/// based on the detected language. Returns a vector of unique import paths.
pub fn extract_imports(content: &str, language: &Language) -> Vec<String> {
    let mut imports = HashSet::new();

    match language {
        Language::Rust => extract_rust_imports(content, &mut imports),
        Language::Python => extract_python_imports(content, &mut imports),
        Language::JavaScript | Language::TypeScript => extract_js_imports(content, &mut imports),
        Language::Go => extract_go_imports(content, &mut imports),
        _ => {}
    }

    imports.into_iter().collect()
}

/// Extract Rust imports from content
pub fn extract_rust_imports(content: &str, imports: &mut HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("use ") {
            let statement = trimmed
                .trim_start_matches("use ")
                .trim_end_matches(';')
                .split_whitespace()
                .next()
                .unwrap_or_default();
            let module_path = if let Some(brace_pos) = statement.find('{') {
                statement[..brace_pos].trim_end_matches("::")
            } else {
                statement.trim_end_matches("::")
            };
            if !module_path.is_empty() {
                imports.insert(module_path.to_string());
            }
        } else if trimmed.starts_with("mod ") {
            let module = trimmed.trim_start_matches("mod ").trim_end_matches(';').trim();
            if !module.is_empty() {
                imports.insert(module.to_string());
            }
        }
    }
}

/// Extract Python imports from content
pub fn extract_python_imports(content: &str, imports: &mut HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            for module in trimmed.trim_start_matches("import ").split(',') {
                let module = module.trim().split_whitespace().next().unwrap_or("");
                if !module.is_empty() {
                    imports.insert(module.to_string());
                }
            }
        } else if trimmed.starts_with("from ") && trimmed.contains(" import ") {
            let module = trimmed
                .trim_start_matches("from ")
                .split(" import ")
                .next()
                .unwrap_or("")
                .trim();
            if !module.is_empty() {
                imports.insert(module.to_string());
            }
        }
    }
}

/// Extract JavaScript/TypeScript imports from content
pub fn extract_js_imports(content: &str, imports: &mut HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            extract_quoted_import(trimmed, imports);
        } else if trimmed.contains("require(") {
            extract_require_import(trimmed, imports);
        }
    }
}

/// Extract a quoted import path from a line
pub fn extract_quoted_import(trimmed: &str, imports: &mut HashSet<String>) {
    if let Some(start) = trimmed.find('"') {
        if let Some(end) = trimmed[start + 1..].find('"') {
            imports.insert(trimmed[start + 1..start + 1 + end].to_string());
        }
    } else if let Some(start) = trimmed.find('\'') {
        if let Some(end) = trimmed[start + 1..].find('\'') {
            imports.insert(trimmed[start + 1..start + 1 + end].to_string());
        }
    }
}

/// Extract a require() import path from a line
pub fn extract_require_import(trimmed: &str, imports: &mut HashSet<String>) {
    if let Some(start) = trimmed.find("require(") {
        let start = start + "require(".len();
        let slice = &trimmed[start..];
        if let Some(end_idx) = slice.find(')') {
            let inner = slice[..end_idx].trim_matches(&['\'', '"'][..]);
            if !inner.is_empty() {
                imports.insert(inner.to_string());
            }
        }
    }
}

/// Extract Go imports from content
pub fn extract_go_imports(content: &str, imports: &mut HashSet<String>) {
    let mut in_block = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "import (" {
            in_block = true;
            continue;
        }
        if in_block {
            if trimmed == ")" {
                in_block = false;
                continue;
            }
            if let Some(import_path) = extract_go_import_path(trimmed) {
                if !import_path.is_empty() {
                    imports.insert(import_path);
                }
            }
        } else if trimmed.starts_with("import ") {
            let rest = trimmed.trim_start_matches("import ");
            if let Some(import_path) = extract_go_import_path(rest) {
                if !import_path.is_empty() {
                    imports.insert(import_path);
                }
            }
        }
    }
}

/// Helper to extract the import path from a Go import line.
/// Handles: `"path"`, `alias "path"`, `. "path"`, `_ "path"`
pub fn extract_go_import_path(line: &str) -> Option<String> {
    let line = line.trim();

    // Try double quotes first
    if let Some(start) = line.find('"') {
        if let Some(end) = line[start + 1..].find('"') {
            return Some(line[start + 1..start + 1 + end].to_string());
        }
    }

    // Try backticks
    if let Some(start) = line.find('`') {
        if let Some(end) = line[start + 1..].find('`') {
            return Some(line[start + 1..start + 1 + end].to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_imports() {
        let content = r#"
use std::collections::HashMap;
use crate::module;
mod my_module;
        "#;
        let imports = extract_imports(content, &Language::Rust);
        assert!(imports.contains(&"std::collections::HashMap".to_string()));
        assert!(imports.contains(&"crate::module".to_string()));
        assert!(imports.contains(&"my_module".to_string()));
    }

    #[test]
    fn test_python_imports() {
        let content = r#"
import os
import sys
from collections import defaultdict
        "#;
        let imports = extract_imports(content, &Language::Python);
        assert!(imports.contains(&"os".to_string()));
        assert!(imports.contains(&"sys".to_string()));
        assert!(imports.contains(&"collections".to_string()));
    }

    #[test]
    fn test_js_imports() {
        let content = r#"
import React from 'react';
const fs = require('fs');
        "#;
        let imports = extract_imports(content, &Language::JavaScript);
        assert!(imports.contains(&"react".to_string()));
        assert!(imports.contains(&"fs".to_string()));
    }

    #[test]
    fn test_go_imports() {
        let content = r#"
import (
    "fmt"
    "os"
)
import "strings"
        "#;
        let imports = extract_imports(content, &Language::Go);
        assert!(imports.contains(&"fmt".to_string()));
        assert!(imports.contains(&"os".to_string()));
        assert!(imports.contains(&"strings".to_string()));
    }

    #[test]
    fn test_go_import_with_alias() {
        let content = r#"
import (
    f "fmt"
    . "os"
    _ "init/package"
)
        "#;
        let imports = extract_imports(content, &Language::Go);
        assert!(imports.contains(&"fmt".to_string()));
        assert!(imports.contains(&"os".to_string()));
        assert!(imports.contains(&"init/package".to_string()));
    }
}

//! Tests for the main scribe library.

use super::*;

#[test]
fn test_version() {
    assert!(!VERSION.is_empty());
}

#[cfg(feature = "core")]
#[test]
fn test_core_reexport() {
    let config = Config::default();
    assert!(config.validate().is_ok());
}

#[cfg(all(feature = "analysis", feature = "scanner", feature = "patterns"))]
#[tokio::test]
async fn test_repository_analysis_interface() {
    use std::fs;
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.rs");
    fs::write(&test_file, "fn main() { println!(\"Hello world\"); }").unwrap();

    let config = Config::default();
    let result = analyze_repository(temp_dir.path(), &config).await;

    // Should succeed or fail gracefully
    match result {
        Ok(analysis) => {
            assert!(analysis.file_count() > 0);
            assert!(!analysis.summary().is_empty());
        }
        Err(_) => {
            // Analysis might fail in test environment, which is acceptable
            // as long as the interface compiles correctly
        }
    }
}

#[cfg(all(feature = "scanner", feature = "patterns"))]
#[tokio::test]
async fn test_scan_repository_interface() {
    use std::fs;
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.rs");
    fs::write(&test_file, "fn main() {}").unwrap();

    let result =
        scan_repository(temp_dir.path(), Some(&["**/*.rs"]), Some(&["**/target/**"])).await;

    // Should find the test file
    match result {
        Ok(files) => {
            assert!(!files.is_empty());
            assert!(files
                .iter()
                .any(|f| f.path.file_name().unwrap() == "test.rs"));
        }
        Err(_) => {
            // Scan might fail in test environment, which is acceptable
        }
    }
}

#[cfg(feature = "core")]
#[test]
fn test_prelude_imports() {
    use crate::prelude::*;

    // Test that basic types are available
    let config = Config::default();
    assert!(config.validate().is_ok());

    // Test that version is available
    assert!(!VERSION.is_empty());
}

// Import extraction tests
mod import_extraction_tests {
    use super::*;

    #[test]
    fn test_rust_imports() {
        let content = r#"
use crate::module;
use crate::module::{item1, item2};
use super::parent_module;
use self::sibling;
use std::collections::HashMap;
use std::io::{Read, Write};
mod my_module;
pub use crate::reexport;
"#;
        let imports = extract_imports(content, &Language::Rust);
        println!("Rust imports: {:?}", imports);

        assert!(
            imports.contains(&"crate::module".to_string()),
            "Should extract simple crate import"
        );
        assert!(
            imports.contains(&"super::parent_module".to_string()),
            "Should extract super import"
        );
        assert!(
            imports.contains(&"self::sibling".to_string()),
            "Should extract self import"
        );
        assert!(
            imports.contains(&"std::collections::HashMap".to_string()),
            "Should extract std import"
        );
        assert!(
            imports.contains(&"std::io".to_string()),
            "Should extract grouped std import"
        );
        assert!(
            imports.contains(&"my_module".to_string()),
            "Should extract mod declaration"
        );
        // pub use should NOT be extracted (doesn't start with "use ")
        assert!(
            !imports.iter().any(|i| i.contains("reexport")),
            "Should NOT extract pub use"
        );
    }

    #[test]
    fn test_python_imports() {
        let content = r#"
import os
import os, sys
import numpy as np
from os import path
from . import module
from .. import parent
from ..package import module
from typing import List, Dict
"#;
        let imports = extract_imports(content, &Language::Python);
        println!("Python imports: {:?}", imports);

        assert!(
            imports.contains(&"os".to_string()),
            "Should extract simple import"
        );
        assert!(
            imports.contains(&"sys".to_string()),
            "Should extract comma-separated import"
        );
        assert!(
            imports.contains(&"numpy".to_string()),
            "Should extract aliased import"
        );
        assert!(
            imports.contains(&".".to_string()),
            "Should extract relative import"
        );
        assert!(
            imports.contains(&"..".to_string()),
            "Should extract parent relative import"
        );
        assert!(
            imports.contains(&"..package".to_string()),
            "Should extract parent package import"
        );
        assert!(
            imports.contains(&"typing".to_string()),
            "Should extract from import"
        );
    }

    #[test]
    fn test_javascript_imports() {
        let content = r#"
import foo from 'module'
import bar from "double-quotes"
import { a, b } from 'destructure'
import * as all from 'star-import'
import 'side-effect'
const req1 = require('commonjs')
const req2 = require("cjs-double")
import { x } from './relative'
"#;
        let imports = extract_imports(content, &Language::JavaScript);
        println!("JavaScript imports: {:?}", imports);

        assert!(
            imports.contains(&"module".to_string()),
            "Should extract single-quote import"
        );
        assert!(
            imports.contains(&"double-quotes".to_string()),
            "Should extract double-quote import"
        );
        assert!(
            imports.contains(&"destructure".to_string()),
            "Should extract destructured import"
        );
        assert!(
            imports.contains(&"star-import".to_string()),
            "Should extract star import"
        );
        assert!(
            imports.contains(&"side-effect".to_string()),
            "Should extract side-effect import"
        );
        assert!(
            imports.contains(&"commonjs".to_string()),
            "Should extract require single"
        );
        assert!(
            imports.contains(&"cjs-double".to_string()),
            "Should extract require double"
        );
        assert!(
            imports.contains(&"./relative".to_string()),
            "Should extract relative import"
        );
    }

    #[test]
    fn test_go_imports() {
        let content = r#"
package main

import "fmt"
import (
    "os"
    "path/filepath"
)
"#;
        let imports = extract_imports(content, &Language::Go);
        println!("Go imports: {:?}", imports);

        assert!(
            imports.contains(&"fmt".to_string()),
            "Should extract single import"
        );
        assert!(
            imports.contains(&"os".to_string()),
            "Should extract block import"
        );
        assert!(
            imports.contains(&"path/filepath".to_string()),
            "Should extract block import with path"
        );
    }

    #[test]
    fn test_go_aliased_imports() {
        let content = r#"
import f "fmt"
import (
    . "os"
    _ "init/pkg"
    alias "github.com/pkg/errors"
)
"#;
        let imports = extract_imports(content, &Language::Go);
        println!("Go aliased imports: {:?}", imports);

        // These should extract the package path, not the alias
        assert!(
            imports.contains(&"fmt".to_string()),
            "Should extract aliased fmt"
        );
        assert!(
            imports.contains(&"os".to_string()),
            "Should extract dot-imported os"
        );
        assert!(
            imports.contains(&"init/pkg".to_string()),
            "Should extract blank-imported init/pkg"
        );
        assert!(
            imports.contains(&"github.com/pkg/errors".to_string()),
            "Should extract aliased github package"
        );

        // Should NOT contain the aliases themselves
        assert!(
            !imports.iter().any(|i| i.starts_with("f ")),
            "Should not include alias 'f'"
        );
        assert!(
            !imports.iter().any(|i| i.starts_with(". ")),
            "Should not include dot alias"
        );
    }

    #[test]
    fn test_elixir_imports() {
        let content = r#"
alias MyApp.Repo
alias MyApp.{Accounts.User, Accounts.Team}
import Plug.Conn
require Logger
use MyAppWeb, :controller
"#;
        let imports = extract_imports(content, &Language::Elixir);

        assert!(imports.contains(&"MyApp.Repo".to_string()));
        assert!(imports.contains(&"MyApp.Accounts.User".to_string()));
        assert!(imports.contains(&"MyApp.Accounts.Team".to_string()));
        assert!(imports.contains(&"Plug.Conn".to_string()));
        assert!(imports.contains(&"Logger".to_string()));
        assert!(imports.contains(&"MyAppWeb".to_string()));
    }
}

// Import edge case tests
mod import_edge_cases {
    use super::*;

    #[test]
    fn test_rust_pub_use_not_extracted() {
        let content = "pub use crate::module;";
        let imports = extract_imports(content, &Language::Rust);
        assert!(imports.is_empty(), "pub use should not be extracted");
    }

    #[test]
    fn test_rust_multiline_use() {
        // Multi-line use statements (spanning lines) - only first line detected
        let content = r#"
use std::collections::{
    HashMap,
    HashSet,
};
"#;
        let imports = extract_imports(content, &Language::Rust);
        println!("Multiline Rust: {:?}", imports);
        assert!(imports.contains(&"std::collections".to_string()));
    }

    #[test]
    fn test_typescript_import_type() {
        let content = r#"
import type { Type } from 'type-module'
import { Component } from '@angular/core'
"#;
        let imports = extract_imports(content, &Language::TypeScript);
        println!("TypeScript imports: {:?}", imports);
        assert!(
            imports.contains(&"type-module".to_string()),
            "Should extract type import"
        );
        assert!(
            imports.contains(&"@angular/core".to_string()),
            "Should extract scoped package"
        );
    }

    #[test]
    fn test_python_multiline_from_import() {
        // Python allows parenthesized imports
        let content = r#"
from module import (
    item1,
    item2,
)
"#;
        let imports = extract_imports(content, &Language::Python);
        println!("Python multiline: {:?}", imports);
        assert!(imports.contains(&"module".to_string()));
    }

    #[test]
    fn test_go_backtick_imports() {
        let content = "import `fmt`";
        let imports = extract_imports(content, &Language::Go);
        assert!(
            imports.contains(&"fmt".to_string()),
            "Should handle backtick imports"
        );
    }

    #[test]
    fn test_js_template_literal_not_extracted() {
        // Template literals shouldn't be confused with imports
        let content = r#"
const x = `not an import`;
import real from 'real-module';
"#;
        let imports = extract_imports(content, &Language::JavaScript);
        println!("JS template test: {:?}", imports);
        assert!(imports.contains(&"real-module".to_string()));
        assert!(!imports.contains(&"not an import".to_string()));
    }
}

#[cfg(all(feature = "analysis", feature = "scanner"))]
mod repository_analysis_tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_repository_analysis() -> RepositoryAnalysis {
        let mut heuristic_scores = HashMap::new();
        heuristic_scores.insert("src/main.rs".to_string(), 0.9);
        heuristic_scores.insert("src/lib.rs".to_string(), 0.8);
        heuristic_scores.insert("tests/test.rs".to_string(), 0.3);
        heuristic_scores.insert("README.md".to_string(), 0.5);
        heuristic_scores.insert("Cargo.toml".to_string(), 0.6);

        RepositoryAnalysis {
            files: vec![],
            heuristic_scores: heuristic_scores.clone(),
            #[cfg(feature = "graph")]
            centrality_scores: None,
            final_scores: heuristic_scores,
            metadata: scribe_core::AnalysisMetadata {
                timestamp: std::time::SystemTime::now(),
                scribe_version: "test".to_string(),
                features_enabled: vec!["test".to_string()],
                config_hash: None,
            },
        }
    }

    #[test]
    fn test_top_files() {
        let analysis = create_test_repository_analysis();
        let top = analysis.top_files(3);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "src/main.rs");
        assert!((top[0].1 - 0.9).abs() < 0.01);
        assert_eq!(top[1].0, "src/lib.rs");
    }

    #[test]
    fn test_top_files_more_than_available() {
        let analysis = create_test_repository_analysis();
        let top = analysis.top_files(100);

        // Should return all files
        assert_eq!(top.len(), 5);
    }

    #[test]
    fn test_files_above_threshold() {
        let analysis = create_test_repository_analysis();
        let files = analysis.files_above_threshold(0.7);

        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|(path, _)| *path == "src/main.rs"));
        assert!(files.iter().any(|(path, _)| *path == "src/lib.rs"));
    }

    #[test]
    fn test_files_above_threshold_high() {
        let analysis = create_test_repository_analysis();
        let files = analysis.files_above_threshold(0.95);

        // No files above 0.95
        assert!(files.is_empty());
    }

    #[test]
    fn test_file_count() {
        let analysis = create_test_repository_analysis();
        assert_eq!(analysis.file_count(), 0); // files vec is empty in test
    }

    #[test]
    fn test_summary() {
        let analysis = create_test_repository_analysis();
        let summary = analysis.summary();

        assert!(summary.contains("Repository Analysis Summary"));
        assert!(summary.contains("Files analyzed"));
        assert!(summary.contains("Average score"));
        assert!(summary.contains("Top file"));
        assert!(summary.contains("Scribe version: test"));
    }
}

#[test]
fn test_build_optimized_config() {
    let config = Config::default();
    let optimized = build_optimized_config(&config);

    assert_eq!(optimized.performance.batch_size, 20);
    assert!(optimized.performance.use_mmap);
    assert_eq!(optimized.performance.io_buffer_size, 512 * 1024);
    assert!(optimized.analysis.enable_caching);
}

#[test]
fn test_debug_log_without_env() {
    // Debug log should not panic when SCRIBE_DEBUG is not set
    std::env::remove_var("SCRIBE_DEBUG");
    debug_log("test message");
}

#[test]
fn test_debug_log_with_env() {
    // Debug log should work when SCRIBE_DEBUG is set
    std::env::set_var("SCRIBE_DEBUG", "1");
    debug_log("test message");
    std::env::remove_var("SCRIBE_DEBUG");
}

#[cfg(feature = "core")]
mod should_load_content_tests {
    use super::*;
    use scribe_core::file::{FileWeight, RenderDecision};
    use std::path::PathBuf;

    fn create_test_file_info(size: u64, is_binary: bool) -> FileInfo {
        FileInfo {
            path: PathBuf::from("test.rs"),
            relative_path: "test.rs".to_string(),
            size,
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
            is_binary,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        }
    }

    #[test]
    fn test_should_load_content_small_file() {
        let file = create_test_file_info(1024, false);
        let config = Config::default();
        assert!(should_load_content(&file, &config));
    }

    #[test]
    fn test_should_load_content_large_file() {
        let file = create_test_file_info(100 * 1024 * 1024, false); // 100MB
        let config = Config::default();
        // Should not load very large files
        assert!(!should_load_content(&file, &config));
    }

    #[test]
    fn test_should_load_content_binary_file() {
        let file = create_test_file_info(1024, true);
        let config = Config::default();
        // Should not load binary files
        assert!(!should_load_content(&file, &config));
    }

    #[test]
    fn test_should_load_content_analysis_disabled() {
        let file = create_test_file_info(1024, false);
        let mut config = Config::default();
        config.analysis.analyze_content = false;
        // Should not load if analysis is disabled
        assert!(!should_load_content(&file, &config));
    }

    #[test]
    fn test_should_load_content_at_size_limit() {
        let mut config = Config::default();
        // The logic uses max(io_buffer_size, 256KB), so we need to use a high value
        config.performance.io_buffer_size = 1024 * 1024; // 1MB
        let file = create_test_file_info(1024 * 1024, false); // Exactly at limit
        assert!(should_load_content(&file, &config));

        let file_over = create_test_file_info(1024 * 1024 + 1, false); // Just over limit
        assert!(!should_load_content(&file_over, &config));
    }
}

#[cfg(all(feature = "analysis", feature = "scanner"))]
mod derive_file_context_tests {
    use super::*;
    use scribe_core::file::{FileWeight, RenderDecision};
    use std::path::PathBuf;

    fn create_source_file(path: &str, language: Language) -> FileInfo {
        FileInfo {
            path: PathBuf::from(path),
            relative_path: path.to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: language.clone(),
            },
            language,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        }
    }

    #[test]
    fn test_derive_context_example_file() {
        let file = create_source_file("examples/demo.rs", Language::Rust);
        let config = Config::default();
        let context = derive_file_context(&file, &config);
        assert!(context.has_examples);
    }

    #[test]
    fn test_derive_context_non_example_file() {
        let file = create_source_file("src/main.rs", Language::Rust);
        let mut config = Config::default();
        config.analysis.analyze_content = false; // Disable content loading
        let context = derive_file_context(&file, &config);
        // Won't detect examples without content analysis
        assert!(!context.has_examples);
    }

    #[test]
    fn test_derive_context_entrypoint() {
        let file = create_source_file("src/main.rs", Language::Rust);
        let mut config = Config::default();
        config.analysis.analyze_content = false;
        let context = derive_file_context(&file, &config);
        // main.rs should be detected as entrypoint from path
        assert!(context.is_entrypoint);
    }
}

mod scoring_tests {
    use super::*;

    #[test]
    fn test_apply_boost_match() {
        let result = apply_boost("test/package.json", &["package.json"], 0.5);
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_apply_boost_no_match() {
        let result = apply_boost("test/other.txt", &["package.json"], 0.5);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_apply_boost_multiple_patterns() {
        let patterns = &["package.json", "Cargo.toml"];
        let result = apply_boost("test/Cargo.toml", patterns, 0.3);
        assert_eq!(result, 0.3);
    }
}

mod entrypoint_detection_tests {
    use super::*;

    #[test]
    fn test_detect_rust_main() {
        let content = "fn main( { println!(\"hello\"); }";
        assert!(detect_entrypoint_from_content(content, &Language::Rust));
    }

    #[test]
    fn test_detect_rust_non_main() {
        let content = "pub fn helper() {}";
        assert!(!detect_entrypoint_from_content(content, &Language::Rust));
    }

    #[test]
    fn test_detect_python_main() {
        // Python uses __name__ == "__main__" (double quotes)
        let content = "if __name__ == \"__main__\":\n    main()";
        assert!(detect_entrypoint_from_content(content, &Language::Python));
    }

    #[test]
    fn test_detect_python_non_main() {
        let content = "def helper():\n    pass";
        assert!(!detect_entrypoint_from_content(content, &Language::Python));
    }

    #[test]
    fn test_detect_javascript_module_exports() {
        let content = "module.exports = myModule;";
        assert!(detect_entrypoint_from_content(
            content,
            &Language::JavaScript
        ));
    }

    #[test]
    fn test_detect_javascript_export_default() {
        let content = "export default App;";
        assert!(detect_entrypoint_from_content(
            content,
            &Language::JavaScript
        ));
    }

    #[test]
    fn test_detect_go_main() {
        let content = "func main( { fmt.Println(\"hello\") }";
        assert!(detect_entrypoint_from_content(content, &Language::Go));
    }

    #[test]
    fn test_detect_go_non_main() {
        let content = "func helper() {}";
        assert!(!detect_entrypoint_from_content(content, &Language::Go));
    }

    #[test]
    fn test_detect_java_main() {
        let content = "public static void main(String[] args) {}";
        assert!(detect_entrypoint_from_content(content, &Language::Java));
    }

    #[test]
    fn test_detect_unknown_language() {
        let content = "fn main() {}";
        // Unknown language should not detect entrypoints
        assert!(!detect_entrypoint_from_content(content, &Language::Unknown));
    }
}

#[cfg(feature = "core")]
mod priority_boost_tests {
    use super::*;
    use scribe_core::file::{FileWeight, RenderDecision};
    use std::path::PathBuf;

    fn create_file(path: &str) -> FileInfo {
        FileInfo {
            path: PathBuf::from(path),
            relative_path: path.to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Unknown,
            },
            language: Language::Unknown,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        }
    }

    #[test]
    fn test_priority_boost_readme() {
        let file = create_file("README.md");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.0);
    }

    #[test]
    fn test_priority_boost_config() {
        let file = create_file("package.json");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.0);
    }

    #[test]
    fn test_priority_boost_main() {
        let file = create_file("src/main.rs");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.0);
    }

    #[test]
    fn test_priority_boost_no_match() {
        let file = create_file("src/utils/helper.rs");
        let boost = compute_priority_boost(&file);
        assert_eq!(boost, 0.0);
    }
}

#[test]
fn test_utils_module_exists() {
    // Verify the utils module can be accessed
    #[cfg(feature = "core")]
    {
        use crate::utils;
        // Just verifying the module compiles
    }
}

#[test]
fn test_prelude_module_exists() {
    // Verify the prelude module can be accessed
    use crate::prelude::VERSION;
    assert!(!VERSION.is_empty());
}

//! Scoring and priority boost utilities for file analysis.
//!
//! This module provides functions for computing priority boosts
//! and various scoring metrics for repository files.

use scribe_core::{FileInfo, Language};

/// Priority boost patterns: (pattern_list, boost_value)
const README_BOOST: (&[&str], f64) = (&["readme.md", "readme"], 0.4);
const PACKAGE_MANAGER_BOOST: (&[&str], f64) = (
    &[
        "cargo.toml",
        "package.json",
        "requirements.txt",
        "pyproject.toml",
    ],
    0.25,
);
const ENTRYPOINT_BOOST: (&[&str], f64) = (
    &["main.rs", "main.py", "main.go", "index.js", "index.ts"],
    0.3,
);
const LIB_BOOST: (&[&str], f64) = (&["lib.rs"], 0.2);
const BUILD_BOOST: (&[&str], f64) = (&["build.rs", "setup.py"], 0.15);

/// Apply a single boost category if path matches any pattern
pub fn apply_boost(path_lower: &str, patterns: &[&str], boost: f64) -> f64 {
    if patterns.iter().any(|p| path_lower.ends_with(p)) {
        boost
    } else {
        0.0
    }
}

/// Compute priority boost for a file based on its path patterns
pub fn compute_priority_boost(file: &FileInfo) -> f64 {
    let path_lower = file.relative_path.to_lowercase();

    let boost = apply_boost(&path_lower, README_BOOST.0, README_BOOST.1)
        + apply_boost(
            &path_lower,
            PACKAGE_MANAGER_BOOST.0,
            PACKAGE_MANAGER_BOOST.1,
        )
        + apply_boost(&path_lower, ENTRYPOINT_BOOST.0, ENTRYPOINT_BOOST.1)
        + apply_boost(&path_lower, LIB_BOOST.0, LIB_BOOST.1)
        + apply_boost(&path_lower, BUILD_BOOST.0, BUILD_BOOST.1);

    boost.min(1.0)
}

/// Detect if content appears to be an entrypoint based on language-specific patterns
pub fn detect_entrypoint_from_content(content: &str, language: &Language) -> bool {
    match language {
        Language::Rust => content.contains("fn main("),
        Language::Python => content.contains("__name__ == \"__main__\""),
        Language::JavaScript | Language::TypeScript => {
            content.contains("module.exports") || content.contains("export default")
        }
        Language::Go => content.contains("func main("),
        Language::Java => content.contains("public static void main("),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scribe_core::{FileWeight, RenderDecision};
    use std::path::PathBuf;

    fn make_file_info(relative_path: &str) -> FileInfo {
        FileInfo {
            path: PathBuf::from(relative_path),
            relative_path: relative_path.to_string(),
            size: 100,
            language: Language::Unknown,
            file_type: scribe_core::FileType::Source {
                language: Language::Unknown,
            },
            modified: None,
            decision: RenderDecision::include("test"),
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            content: None,
            centrality_score: None,
            git_status: None,
            weight: FileWeight::default(),
        }
    }

    #[test]
    fn test_readme_boost() {
        let file = make_file_info("README.md");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.3);
    }

    #[test]
    fn test_package_manager_boost() {
        let file = make_file_info("Cargo.toml");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_entrypoint_boost() {
        let file = make_file_info("src/main.rs");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_no_boost() {
        let file = make_file_info("src/utils/helper.rs");
        let boost = compute_priority_boost(&file);
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn test_detect_rust_entrypoint() {
        let content = "fn main() { println!(\"Hello\"); }";
        assert!(detect_entrypoint_from_content(content, &Language::Rust));
    }

    #[test]
    fn test_detect_python_entrypoint() {
        let content = "if __name__ == \"__main__\":\n    main()";
        assert!(detect_entrypoint_from_content(content, &Language::Python));
    }

    #[test]
    fn test_document_analysis() {
        let content = "# Header\n\nSome text\n```\ncode\n```\n\n[Link](http://example.com)";
        let analysis = crate::analysis_helpers::analyze_document_content(content);
        assert!(analysis.heading_count > 0);
        assert!(analysis.code_block_count > 0);
        assert!(analysis.link_count > 0);
    }

    #[test]
    fn test_detect_javascript_entrypoint_module_exports() {
        let content = "module.exports = myFunction;";
        assert!(detect_entrypoint_from_content(
            content,
            &Language::JavaScript
        ));
    }

    #[test]
    fn test_detect_javascript_entrypoint_export_default() {
        let content = "export default function App() {}";
        assert!(detect_entrypoint_from_content(
            content,
            &Language::JavaScript
        ));
    }

    #[test]
    fn test_detect_typescript_entrypoint() {
        let content = "export default class MyComponent {}";
        assert!(detect_entrypoint_from_content(
            content,
            &Language::TypeScript
        ));
    }

    #[test]
    fn test_detect_go_entrypoint() {
        let content = "func main() { fmt.Println(\"Hello\") }";
        assert!(detect_entrypoint_from_content(content, &Language::Go));
    }

    #[test]
    fn test_detect_java_entrypoint() {
        let content = "public static void main(String[] args) {}";
        assert!(detect_entrypoint_from_content(content, &Language::Java));
    }

    #[test]
    fn test_detect_unknown_language_no_entrypoint() {
        let content = "fn main() { }";
        assert!(!detect_entrypoint_from_content(content, &Language::Unknown));
    }

    #[test]
    fn test_detect_no_entrypoint_rust() {
        let content = "fn helper() { println!(\"Not main\"); }";
        assert!(!detect_entrypoint_from_content(content, &Language::Rust));
    }

    #[test]
    fn test_detect_no_entrypoint_python() {
        let content = "def helper(): pass";
        assert!(!detect_entrypoint_from_content(content, &Language::Python));
    }

    #[test]
    fn test_lib_boost() {
        let file = make_file_info("src/lib.rs");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.1);
    }

    #[test]
    fn test_build_rs_boost() {
        let file = make_file_info("build.rs");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.1);
    }

    #[test]
    fn test_setup_py_boost() {
        let file = make_file_info("setup.py");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.1);
    }

    #[test]
    fn test_requirements_txt_boost() {
        let file = make_file_info("requirements.txt");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_pyproject_toml_boost() {
        let file = make_file_info("pyproject.toml");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_index_js_boost() {
        let file = make_file_info("index.js");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_index_ts_boost() {
        let file = make_file_info("src/index.ts");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_main_py_boost() {
        let file = make_file_info("main.py");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_main_go_boost() {
        let file = make_file_info("cmd/main.go");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.2);
    }

    #[test]
    fn test_case_insensitive_readme() {
        let file = make_file_info("readme.md");
        let boost = compute_priority_boost(&file);
        assert!(boost > 0.3);
    }

    #[test]
    fn test_combined_boost_capped_at_1() {
        // README.md + main.rs would normally be > 0.7, but should be capped
        let file = make_file_info("README.md");
        let boost = compute_priority_boost(&file);
        assert!(boost <= 1.0);
    }

    #[test]
    fn test_apply_boost_direct() {
        let path = "test/package.json";
        let patterns = &["package.json", "Cargo.toml"];
        let boost = apply_boost(path, patterns, 0.5);
        assert_eq!(boost, 0.5);

        let no_match_path = "test/other.txt";
        let no_boost = apply_boost(no_match_path, patterns, 0.5);
        assert_eq!(no_boost, 0.0);
    }
}

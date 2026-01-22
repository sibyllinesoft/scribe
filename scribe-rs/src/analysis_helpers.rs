//! Helper functions for repository analysis.

use scribe_analysis::DocumentAnalysis;
use scribe_core::Language;

/// Analyze document content for structural indicators
pub(crate) fn analyze_document_content(content: &str) -> DocumentAnalysis {
    let mut analysis = DocumentAnalysis::new();
    let mut in_code_block = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            if !in_code_block {
                analysis.code_block_count += 1;
            }
            in_code_block = !in_code_block;
            continue;
        }

        if trimmed.starts_with('#') {
            analysis.heading_count += 1;
            if trimmed.to_lowercase().contains("table of contents") {
                analysis.toc_indicators += 1;
            }
        }

        if trimmed.contains("](") {
            analysis.link_count += trimmed.matches("](").count();
        }
    }

    analysis.is_well_structured = analysis.heading_count > 0 && analysis.link_count > 0;
    analysis
}

/// Convert a language identifier string to a Language enum
pub(crate) fn language_from_identifier(language: &str, path: &std::path::Path) -> Language {
    if !language.is_empty() {
        match language.to_lowercase().as_str() {
            "rust" => return Language::Rust,
            "python" => return Language::Python,
            "javascript" => return Language::JavaScript,
            "typescript" => return Language::TypeScript,
            "go" => return Language::Go,
            "java" => return Language::Java,
            "c" => return Language::C,
            "cpp" | "c++" => return Language::Cpp,
            "kotlin" => return Language::Kotlin,
            "swift" => return Language::Swift,
            "php" => return Language::PHP,
            "ruby" => return Language::Ruby,
            _ => {}
        }
    }

    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    Language::from_extension(extension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_analyze_document_content_empty() {
        let analysis = analyze_document_content("");
        assert_eq!(analysis.heading_count, 0);
        assert_eq!(analysis.code_block_count, 0);
        assert_eq!(analysis.link_count, 0);
        assert!(!analysis.is_well_structured);
    }

    #[test]
    fn test_analyze_document_content_headings() {
        let content = "# Title\n## Subtitle\n### Section\nSome text";
        let analysis = analyze_document_content(content);
        assert_eq!(analysis.heading_count, 3);
        assert!(!analysis.is_well_structured); // No links
    }

    #[test]
    fn test_analyze_document_content_links() {
        let content = "Check out [link1](http://example.com) and [link2](http://test.com)";
        let analysis = analyze_document_content(content);
        assert_eq!(analysis.link_count, 2);
    }

    #[test]
    fn test_analyze_document_content_code_blocks() {
        let content = "```rust\nfn main() {}\n```\n\n```python\nprint('hello')\n```";
        let analysis = analyze_document_content(content);
        assert_eq!(analysis.code_block_count, 2);
    }

    #[test]
    fn test_analyze_document_content_toc() {
        let content = "# Table of Contents\n- [Section 1](link)\n- [Section 2](link)";
        let analysis = analyze_document_content(content);
        assert_eq!(analysis.toc_indicators, 1);
    }

    #[test]
    fn test_analyze_document_content_well_structured() {
        let content = "# My README\n\nCheck out [this link](http://example.com)";
        let analysis = analyze_document_content(content);
        assert!(analysis.is_well_structured);
        assert_eq!(analysis.heading_count, 1);
        assert_eq!(analysis.link_count, 1);
    }

    #[test]
    fn test_language_from_identifier_rust() {
        let path = Path::new("test.rs");
        assert_eq!(language_from_identifier("rust", path), Language::Rust);
        assert_eq!(language_from_identifier("RUST", path), Language::Rust);
    }

    #[test]
    fn test_language_from_identifier_python() {
        let path = Path::new("test.py");
        assert_eq!(language_from_identifier("python", path), Language::Python);
    }

    #[test]
    fn test_language_from_identifier_javascript() {
        let path = Path::new("test.js");
        assert_eq!(
            language_from_identifier("javascript", path),
            Language::JavaScript
        );
    }

    #[test]
    fn test_language_from_identifier_typescript() {
        let path = Path::new("test.ts");
        assert_eq!(
            language_from_identifier("typescript", path),
            Language::TypeScript
        );
    }

    #[test]
    fn test_language_from_identifier_go() {
        let path = Path::new("test.go");
        assert_eq!(language_from_identifier("go", path), Language::Go);
    }

    #[test]
    fn test_language_from_identifier_java() {
        let path = Path::new("Test.java");
        assert_eq!(language_from_identifier("java", path), Language::Java);
    }

    #[test]
    fn test_language_from_identifier_c() {
        let path = Path::new("test.c");
        assert_eq!(language_from_identifier("c", path), Language::C);
    }

    #[test]
    fn test_language_from_identifier_cpp() {
        let path = Path::new("test.cpp");
        assert_eq!(language_from_identifier("cpp", path), Language::Cpp);
        assert_eq!(language_from_identifier("c++", path), Language::Cpp);
    }

    #[test]
    fn test_language_from_identifier_kotlin() {
        let path = Path::new("Test.kt");
        assert_eq!(language_from_identifier("kotlin", path), Language::Kotlin);
    }

    #[test]
    fn test_language_from_identifier_swift() {
        let path = Path::new("test.swift");
        assert_eq!(language_from_identifier("swift", path), Language::Swift);
    }

    #[test]
    fn test_language_from_identifier_php() {
        let path = Path::new("test.php");
        assert_eq!(language_from_identifier("php", path), Language::PHP);
    }

    #[test]
    fn test_language_from_identifier_ruby() {
        let path = Path::new("test.rb");
        assert_eq!(language_from_identifier("ruby", path), Language::Ruby);
    }

    #[test]
    fn test_language_from_identifier_fallback_to_extension() {
        let path = Path::new("test.rs");
        // Empty identifier should fall back to extension
        assert_eq!(language_from_identifier("", path), Language::Rust);
    }

    #[test]
    fn test_language_from_identifier_unknown() {
        let path = Path::new("test.unknown");
        // Unknown identifier and extension
        let lang = language_from_identifier("foobar", path);
        assert_eq!(lang, Language::Unknown);
    }
}

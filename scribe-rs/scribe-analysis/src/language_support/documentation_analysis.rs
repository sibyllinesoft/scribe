//! # Documentation Coverage Analysis
//!
//! Analyzes documentation coverage in source code files, including docstrings,
//! comments, and inline documentation patterns.

use super::ast_language::AstLanguage;
use scribe_core::Result;
use serde::{Deserialize, Serialize};

/// Documentation coverage analysis results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentationCoverage {
    /// Total number of functions/methods
    pub total_functions: usize,
    /// Number of documented functions
    pub documented_functions: usize,
    /// Total number of classes
    pub total_classes: usize,
    /// Number of documented classes
    pub documented_classes: usize,
    /// Documentation coverage percentage
    pub coverage_percentage: f64,
    /// Lines of documentation
    pub documentation_lines: usize,
    /// Total lines of code
    pub total_lines: usize,
}

/// Documentation analyzer for specific languages
#[derive(Debug)]
pub struct DocumentationAnalyzer {
    language: AstLanguage,
}

impl DocumentationAnalyzer {
    /// Create a new documentation analyzer
    pub fn new(language: AstLanguage) -> Result<Self> {
        Ok(Self { language })
    }

    /// Analyze documentation coverage in source code
    pub fn analyze_coverage(&self, content: &str) -> Result<DocumentationCoverage> {
        // Basic implementation - can be enhanced with AST analysis
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let documentation_lines = self.count_documentation_lines(&lines);

        let coverage_percentage = if total_lines > 0 {
            (documentation_lines as f64 / total_lines as f64) * 100.0
        } else {
            0.0
        };

        Ok(DocumentationCoverage {
            total_functions: 0,      // TODO: Extract from AST
            documented_functions: 0, // TODO: Extract from AST
            total_classes: 0,        // TODO: Extract from AST
            documented_classes: 0,   // TODO: Extract from AST
            coverage_percentage,
            documentation_lines,
            total_lines,
        })
    }

    /// Count lines that contain documentation
    fn count_documentation_lines(&self, lines: &[&str]) -> usize {
        lines
            .iter()
            .filter(|line| self.is_documentation_line(line))
            .count()
    }

    /// Get documentation prefixes for a language
    fn doc_prefixes(language: &AstLanguage) -> &'static [&'static str] {
        match language {
            AstLanguage::Python => &["#", "\"\"\"", "'''"],
            AstLanguage::JavaScript | AstLanguage::TypeScript => &["//", "/*", "*", "/**"],
            AstLanguage::Rust => &["//", "///", "//!"],
            AstLanguage::Go => &["//"],
            _ => &["//", "#", "/*", "*"], // Generic fallback
        }
    }

    /// Check if a line contains documentation
    fn is_documentation_line(&self, line: &str) -> bool {
        let trimmed = line.trim();
        Self::doc_prefixes(&self.language)
            .iter()
            .any(|prefix| trimmed.starts_with(prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_documentation_analysis() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::Python).unwrap();
        let python_code = r#"
# This is a comment
def hello():
    """This is a docstring."""
    print("Hello")

def world():
    # Another comment
    print("World")
"#;

        let coverage = analyzer.analyze_coverage(python_code).unwrap();
        assert!(coverage.documentation_lines > 0);
        assert!(coverage.coverage_percentage > 0.0);
    }

    #[test]
    fn test_rust_documentation_analysis() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::Rust).unwrap();
        let rust_code = r#"
//! Module documentation
//! More module docs

/// Function documentation
fn hello() {
    // Inline comment
    println!("Hello");
}

/// Another documented function
fn world() {}
"#;

        let coverage = analyzer.analyze_coverage(rust_code).unwrap();
        assert!(coverage.documentation_lines > 0);
        assert!(coverage.coverage_percentage > 0.0);
    }

    #[test]
    fn test_javascript_documentation_analysis() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::JavaScript).unwrap();
        let js_code = r#"
// This is a comment
/**
 * JSDoc comment
 */
function hello() {
    /* Block comment */
    console.log("Hello");
}
"#;

        let coverage = analyzer.analyze_coverage(js_code).unwrap();
        assert!(coverage.documentation_lines > 0);
    }

    #[test]
    fn test_typescript_documentation_analysis() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::TypeScript).unwrap();
        let ts_code = r#"
// TypeScript comment
/** JSDoc style */
function hello(): void {}
"#;

        let coverage = analyzer.analyze_coverage(ts_code).unwrap();
        assert!(coverage.documentation_lines > 0);
    }

    #[test]
    fn test_go_documentation_analysis() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::Go).unwrap();
        let go_code = r#"
// Package comment
package main

// Function comment
func main() {
    // Inline comment
    fmt.Println("Hello")
}
"#;

        let coverage = analyzer.analyze_coverage(go_code).unwrap();
        assert!(coverage.documentation_lines > 0);
    }

    #[test]
    fn test_empty_content() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::Python).unwrap();
        let coverage = analyzer.analyze_coverage("").unwrap();
        assert_eq!(coverage.documentation_lines, 0);
        assert_eq!(coverage.total_lines, 0);
        assert_eq!(coverage.coverage_percentage, 0.0);
    }

    #[test]
    fn test_no_documentation() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::Rust).unwrap();
        let code = r#"
fn main() {
    println!("Hello");
}
"#;

        let coverage = analyzer.analyze_coverage(code).unwrap();
        // No documentation comments (// without /)
        assert_eq!(coverage.documentation_lines, 0);
    }

    #[test]
    fn test_documentation_coverage_default() {
        let coverage = DocumentationCoverage::default();
        assert_eq!(coverage.total_functions, 0);
        assert_eq!(coverage.documented_functions, 0);
        assert_eq!(coverage.total_classes, 0);
        assert_eq!(coverage.documented_classes, 0);
        assert_eq!(coverage.coverage_percentage, 0.0);
        assert_eq!(coverage.documentation_lines, 0);
        assert_eq!(coverage.total_lines, 0);
    }

    #[test]
    fn test_documentation_coverage_clone() {
        let coverage = DocumentationCoverage {
            total_functions: 10,
            documented_functions: 5,
            total_classes: 3,
            documented_classes: 2,
            coverage_percentage: 50.0,
            documentation_lines: 20,
            total_lines: 100,
        };

        let cloned = coverage.clone();
        assert_eq!(coverage.total_functions, cloned.total_functions);
        assert_eq!(coverage.documented_functions, cloned.documented_functions);
        assert_eq!(coverage.coverage_percentage, cloned.coverage_percentage);
    }

    #[test]
    fn test_documentation_coverage_serialize() {
        let coverage = DocumentationCoverage {
            total_functions: 5,
            documented_functions: 3,
            total_classes: 2,
            documented_classes: 1,
            coverage_percentage: 60.0,
            documentation_lines: 15,
            total_lines: 50,
        };

        let json = serde_json::to_string(&coverage).unwrap();
        let deserialized: DocumentationCoverage = serde_json::from_str(&json).unwrap();
        assert_eq!(coverage.total_functions, deserialized.total_functions);
        assert_eq!(
            coverage.coverage_percentage,
            deserialized.coverage_percentage
        );
    }

    #[test]
    fn test_documentation_coverage_debug() {
        let coverage = DocumentationCoverage::default();
        let debug_str = format!("{:?}", coverage);
        assert!(debug_str.contains("DocumentationCoverage"));
    }

    #[test]
    fn test_documentation_analyzer_debug() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::Rust).unwrap();
        let debug_str = format!("{:?}", analyzer);
        assert!(debug_str.contains("DocumentationAnalyzer"));
    }

    #[test]
    fn test_doc_prefixes_python() {
        let prefixes = DocumentationAnalyzer::doc_prefixes(&AstLanguage::Python);
        assert!(prefixes.contains(&"#"));
        assert!(prefixes.contains(&"\"\"\""));
        assert!(prefixes.contains(&"'''"));
    }

    #[test]
    fn test_doc_prefixes_javascript() {
        let prefixes = DocumentationAnalyzer::doc_prefixes(&AstLanguage::JavaScript);
        assert!(prefixes.contains(&"//"));
        assert!(prefixes.contains(&"/*"));
        assert!(prefixes.contains(&"/**"));
    }

    #[test]
    fn test_doc_prefixes_rust() {
        let prefixes = DocumentationAnalyzer::doc_prefixes(&AstLanguage::Rust);
        assert!(prefixes.contains(&"//"));
        assert!(prefixes.contains(&"///"));
        assert!(prefixes.contains(&"//!"));
    }

    #[test]
    fn test_doc_prefixes_go() {
        let prefixes = DocumentationAnalyzer::doc_prefixes(&AstLanguage::Go);
        assert!(prefixes.contains(&"//"));
    }

    #[test]
    fn test_mixed_code_and_docs() {
        let analyzer = DocumentationAnalyzer::new(AstLanguage::Python).unwrap();
        let code = r#"
# Header comment
import os

# Function docs
def foo():
    x = 1
    # Inline comment
    return x

class Bar:
    """Class docstring."""
    pass
"#;

        let coverage = analyzer.analyze_coverage(code).unwrap();
        assert!(coverage.documentation_lines >= 4); // At least 4 comment lines
        assert!(coverage.total_lines > coverage.documentation_lines);
    }
}

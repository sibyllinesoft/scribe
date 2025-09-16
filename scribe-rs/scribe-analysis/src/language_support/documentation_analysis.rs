//! # Documentation Coverage Analysis
//!
//! Analyzes documentation coverage in source code files, including docstrings,
//! comments, and inline documentation patterns.

use serde::{Deserialize, Serialize};
use scribe_core::Result;
use super::ast_language::AstLanguage;

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
            total_functions: 0, // TODO: Extract from AST
            documented_functions: 0, // TODO: Extract from AST
            total_classes: 0, // TODO: Extract from AST
            documented_classes: 0, // TODO: Extract from AST
            coverage_percentage,
            documentation_lines,
            total_lines,
        })
    }
    
    /// Count lines that contain documentation
    fn count_documentation_lines(&self, lines: &[&str]) -> usize {
        lines.iter()
            .filter(|line| self.is_documentation_line(line))
            .count()
    }
    
    /// Check if a line contains documentation
    fn is_documentation_line(&self, line: &str) -> bool {
        let trimmed = line.trim();
        
        match self.language {
            AstLanguage::Python => {
                trimmed.starts_with('#') ||
                trimmed.starts_with("\"\"\"") ||
                trimmed.starts_with("'''")
            }
            AstLanguage::JavaScript | AstLanguage::TypeScript => {
                trimmed.starts_with("//") ||
                trimmed.starts_with("/*") ||
                trimmed.starts_with("*") ||
                trimmed.starts_with("/**")
            }
            AstLanguage::Rust => {
                trimmed.starts_with("//") ||
                trimmed.starts_with("///") ||
                trimmed.starts_with("//!")
            }
            AstLanguage::Go => {
                trimmed.starts_with("//")
            }
            _ => {
                // Generic comment detection
                trimmed.starts_with("//") ||
                trimmed.starts_with("#") ||
                trimmed.starts_with("/*") ||
                trimmed.starts_with("*")
            }
        }
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
}
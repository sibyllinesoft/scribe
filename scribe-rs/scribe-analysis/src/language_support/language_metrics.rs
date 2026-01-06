//! # Language-Specific Metrics
//!
//! Calculates language-specific complexity and quality metrics that are
//! tailored to each programming language's characteristics.

use super::ast_language::AstLanguage;
use scribe_core::Result;
use serde::{Deserialize, Serialize};

/// Language-specific complexity factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSpecificComplexity {
    /// Base complexity score
    pub base_complexity: f64,
    /// Language-specific factors (e.g., async/await, generics)
    pub language_factors: f64,
    /// Idiomatic patterns bonus/penalty
    pub idiom_score: f64,
    /// Framework/library usage complexity
    pub framework_complexity: f64,
}

/// Comprehensive language metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageMetrics {
    /// Programming language
    pub language: AstLanguage,
    /// Lines of code
    pub lines_of_code: usize,
    /// Number of functions
    pub function_count: usize,
    /// Number of classes
    pub class_count: usize,
    /// Language-specific complexity
    pub complexity: LanguageSpecificComplexity,
    /// Estimated maintainability score
    pub maintainability_score: f64,
}

impl LanguageMetrics {
    /// Calculate language metrics for source code
    pub fn calculate(content: &str, language: AstLanguage) -> Result<Self> {
        let lines: Vec<&str> = content.lines().collect();
        let lines_of_code = lines.len();

        // Basic pattern counting - can be enhanced with AST analysis
        let function_count = Self::count_functions(content, language);
        let class_count = Self::count_classes(content, language);

        let complexity = Self::calculate_language_complexity(content, language);
        let maintainability_score = Self::calculate_maintainability(
            lines_of_code,
            function_count,
            class_count,
            &complexity,
        );

        Ok(Self {
            language,
            lines_of_code,
            function_count,
            class_count,
            complexity,
            maintainability_score,
        })
    }

    /// Count functions using language-specific patterns
    fn count_functions(content: &str, language: AstLanguage) -> usize {
        match language {
            AstLanguage::Python => content
                .lines()
                .filter(|line| {
                    line.trim().starts_with("def ") || line.trim().starts_with("async def ")
                })
                .count(),
            AstLanguage::JavaScript | AstLanguage::TypeScript => content
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    trimmed.starts_with("function ")
                        || trimmed.contains("=> ")
                        || trimmed.contains("function(")
                })
                .count(),
            AstLanguage::Rust => content
                .lines()
                .filter(|line| line.trim().starts_with("fn ") || line.trim().starts_with("pub fn "))
                .count(),
            AstLanguage::Go => content
                .lines()
                .filter(|line| line.trim().starts_with("func "))
                .count(),
            _ => 0,
        }
    }

    /// Count classes using language-specific patterns
    fn count_classes(content: &str, language: AstLanguage) -> usize {
        match language {
            AstLanguage::Python => content
                .lines()
                .filter(|line| line.trim().starts_with("class "))
                .count(),
            AstLanguage::JavaScript | AstLanguage::TypeScript => content
                .lines()
                .filter(|line| line.trim().starts_with("class "))
                .count(),
            AstLanguage::Rust => content
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    trimmed.starts_with("struct ")
                        || trimmed.starts_with("pub struct ")
                        || trimmed.starts_with("enum ")
                        || trimmed.starts_with("pub enum ")
                })
                .count(),
            AstLanguage::Go => content
                .lines()
                .filter(|line| line.trim().starts_with("type ") && line.contains("struct"))
                .count(),
            _ => 0,
        }
    }

    /// Calculate language-specific complexity factors
    fn calculate_language_complexity(
        content: &str,
        language: AstLanguage,
    ) -> LanguageSpecificComplexity {
        let (language_factors, idiom_score, framework_complexity) = match language {
            AstLanguage::Python => Self::python_complexity(content),
            AstLanguage::Rust => Self::rust_complexity(content),
            AstLanguage::JavaScript | AstLanguage::TypeScript => Self::js_ts_complexity(content, language),
            AstLanguage::Go => Self::go_complexity(content),
            _ => (0.1, 0.0, 0.0),
        };

        LanguageSpecificComplexity {
            base_complexity: 1.0,
            language_factors,
            idiom_score,
            framework_complexity,
        }
    }

    /// Calculate Python-specific complexity factors
    fn python_complexity(content: &str) -> (f64, f64, f64) {
        let mut factors = 0.0;
        let mut framework = 0.0;

        if content.contains("async def") || content.contains("await ") {
            factors += 0.3;
        }
        if content.contains("@") {
            factors += 0.2;
        }
        if content.contains("[") && content.contains("for ") && content.contains("in ") {
            factors += 0.1;
        }
        if content.contains("import django") || content.contains("from django") {
            framework += 0.2;
        }
        if content.contains("import flask") || content.contains("from flask") {
            framework += 0.1;
        }

        (factors, 0.0, framework)
    }

    /// Calculate Rust-specific complexity factors
    fn rust_complexity(content: &str) -> (f64, f64, f64) {
        let mut factors = 0.0;
        let mut idiom = 0.0;

        if content.contains("'") && content.contains("&") {
            factors += 0.4;
        }
        if content.contains("match ") || content.contains("if let ") {
            factors += 0.1;
        }
        if content.contains("macro_rules!") || content.contains("!") {
            factors += 0.3;
        }
        if content.contains("Result<") || content.contains("Option<") {
            idiom += 0.2;
        }

        (factors, idiom, 0.0)
    }

    /// Calculate JavaScript/TypeScript complexity factors
    fn js_ts_complexity(content: &str, language: AstLanguage) -> (f64, f64, f64) {
        let mut factors = 0.0;
        let mut framework = 0.0;

        if content.contains("async ") || content.contains("await ") {
            factors += 0.2;
        }
        if content.contains("Promise") {
            factors += 0.1;
        }
        if language == AstLanguage::TypeScript && content.contains("<") && content.contains(">") {
            factors += 0.2;
        }
        if content.contains("import React") || content.contains("from 'react'") {
            framework += 0.1;
        }

        (factors, 0.0, framework)
    }

    /// Calculate Go-specific complexity factors
    fn go_complexity(content: &str) -> (f64, f64, f64) {
        let mut factors = 0.0;

        if content.contains("go ") && content.contains("()") {
            factors += 0.2;
        }
        if content.contains("chan ") || content.contains("<-") {
            factors += 0.3;
        }
        if content.contains("defer ") {
            factors += 0.1;
        }

        (factors, 0.0, 0.0)
    }

    /// Calculate maintainability score
    fn calculate_maintainability(
        lines_of_code: usize,
        function_count: usize,
        class_count: usize,
        complexity: &LanguageSpecificComplexity,
    ) -> f64 {
        let mut score = 100.0;

        // Penalize large files
        if lines_of_code > 500 {
            score -= (lines_of_code as f64 - 500.0) * 0.01;
        }

        // Reward modular code
        if function_count > 0 {
            let avg_lines_per_function = lines_of_code as f64 / function_count as f64;
            if avg_lines_per_function < 20.0 {
                score += 5.0; // Small functions are good
            } else if avg_lines_per_function > 100.0 {
                score -= 10.0; // Large functions are bad
            }
        }

        // Adjust for language complexity
        score -= complexity.language_factors * 10.0;
        score += complexity.idiom_score * 5.0;
        score -= complexity.framework_complexity * 5.0;

        score.max(0.0).min(100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_metrics() {
        let python_code = r#"
def hello():
    print("Hello")

async def async_hello():
    await some_async_function()

class Calculator:
    def add(self, a, b):
        return a + b
"#;

        let metrics = LanguageMetrics::calculate(python_code, AstLanguage::Python).unwrap();
        assert_eq!(metrics.language, AstLanguage::Python);
        assert!(metrics.function_count >= 2);
        assert_eq!(metrics.class_count, 1);
        assert!(metrics.complexity.language_factors > 0.0); // Should detect async
    }

    #[test]
    fn test_rust_metrics() {
        let rust_code = r#"
fn main() {
    println!("Hello, world!");
}

struct Calculator {
    value: f64,
}

impl Calculator {
    fn new() -> Self {
        Calculator { value: 0.0 }
    }
}
"#;

        let metrics = LanguageMetrics::calculate(rust_code, AstLanguage::Rust).unwrap();
        assert_eq!(metrics.language, AstLanguage::Rust);
        assert!(metrics.function_count >= 1);
        assert!(metrics.class_count >= 1); // struct counts as class
    }
}

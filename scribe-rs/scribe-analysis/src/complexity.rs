//! # Code Complexity Analysis Module
//!
//! This module provides comprehensive code complexity metrics to enhance the heuristic
//! scoring system with deeper analysis of code structure and maintainability.
//!
//! ## Complexity Metrics
//!
//! - **Cyclomatic Complexity**: Measures the number of linearly independent paths through code
//! - **Nesting Depth**: Maximum depth of nested control structures
//! - **Function Count**: Number of functions/methods in a file
//! - **Line Complexity**: Analysis of code lines vs comment lines vs blank lines
//! - **Cognitive Complexity**: Human-focused complexity measure
//! - **Maintainability Index**: Composite metric for code maintainability
//!
//! ## Usage
//!
//! ```rust
//! use scribe_analysis::complexity::{ComplexityAnalyzer, ComplexityMetrics};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let analyzer = ComplexityAnalyzer::new();
//! let content = "fn main() { if x > 0 { println!(\"positive\"); } }";
//! let metrics = analyzer.analyze_content(content, "rust")?;
//!
//! println!("Cyclomatic Complexity: {}", metrics.cyclomatic_complexity);
//! println!("Maintainability Index: {:.2}", metrics.maintainability_index);
//! # Ok(())
//! # }
//! ```

use scribe_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive complexity metrics for a code file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity (McCabe complexity)
    pub cyclomatic_complexity: usize,

    /// Maximum nesting depth
    pub max_nesting_depth: usize,

    /// Total number of functions/methods
    pub function_count: usize,

    /// Number of logical lines of code (excluding comments and blanks)
    pub logical_lines: usize,

    /// Number of comment lines
    pub comment_lines: usize,

    /// Number of blank lines
    pub blank_lines: usize,

    /// Total physical lines
    pub total_lines: usize,

    /// Cognitive complexity (easier to understand than cyclomatic)
    pub cognitive_complexity: usize,

    /// Maintainability index (0-100, higher is better)
    pub maintainability_index: f64,

    /// Average function length
    pub average_function_length: f64,

    /// Code density (logical lines / total lines)
    pub code_density: f64,

    /// Comment ratio (comment lines / logical lines)
    pub comment_ratio: f64,

    /// Language-specific metrics
    pub language_metrics: LanguageSpecificMetrics,
}

/// Language-specific complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSpecificMetrics {
    /// Language identifier
    pub language: String,

    /// Language-specific complexity factors
    pub complexity_factors: HashMap<String, f64>,

    /// Number of imports/includes
    pub import_count: usize,

    /// Number of exported symbols
    pub export_count: usize,

    /// Estimated API surface area
    pub api_surface_area: usize,
}

/// Core complexity analyzer
#[derive(Debug)]
pub struct ComplexityAnalyzer {
    /// Configuration for complexity analysis
    config: ComplexityConfig,
}

/// Configuration for complexity analysis
#[derive(Debug, Clone)]
pub struct ComplexityConfig {
    /// Enable cognitive complexity calculation
    pub enable_cognitive_complexity: bool,

    /// Enable maintainability index calculation
    pub enable_maintainability_index: bool,

    /// Language-specific analysis
    pub enable_language_specific: bool,

    /// Thresholds for complexity warnings
    pub thresholds: ComplexityThresholds,
}

/// Thresholds for complexity warnings
#[derive(Debug, Clone)]
pub struct ComplexityThresholds {
    /// Cyclomatic complexity warning threshold
    pub cyclomatic_warning: usize,

    /// Nesting depth warning threshold
    pub nesting_warning: usize,

    /// Function length warning threshold
    pub function_length_warning: usize,

    /// Maintainability index warning threshold (below this is concerning)
    pub maintainability_warning: f64,
}

impl Default for ComplexityConfig {
    fn default() -> Self {
        Self {
            enable_cognitive_complexity: true,
            enable_maintainability_index: true,
            enable_language_specific: true,
            thresholds: ComplexityThresholds::default(),
        }
    }
}

impl Default for ComplexityThresholds {
    fn default() -> Self {
        Self {
            cyclomatic_warning: 10,        // Standard McCabe threshold
            nesting_warning: 4,            // Deep nesting becomes hard to follow
            function_length_warning: 50,   // Functions longer than 50 lines
            maintainability_warning: 20.0, // Below 20 is concerning
        }
    }
}

impl ComplexityAnalyzer {
    /// Create a new complexity analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ComplexityConfig::default(),
        }
    }

    /// Create a new complexity analyzer with custom configuration
    pub fn with_config(config: ComplexityConfig) -> Self {
        Self { config }
    }

    /// Analyze complexity of file content
    pub fn analyze_content(&self, content: &str, language: &str) -> Result<ComplexityMetrics> {
        let line_metrics = self.analyze_lines(content);
        let cyclomatic_complexity = self.calculate_cyclomatic_complexity(content, language);
        let max_nesting_depth = self.calculate_max_nesting_depth(content, language);
        let function_count = self.count_functions(content, language);

        let cognitive_complexity = if self.config.enable_cognitive_complexity {
            self.calculate_cognitive_complexity(content, language)
        } else {
            0
        };

        let maintainability_index = if self.config.enable_maintainability_index {
            self.calculate_maintainability_index(
                &line_metrics,
                cyclomatic_complexity,
                function_count,
            )
        } else {
            0.0
        };

        let average_function_length = if function_count > 0 {
            line_metrics.logical_lines as f64 / function_count as f64
        } else {
            0.0
        };

        let code_density = if line_metrics.total_lines > 0 {
            line_metrics.logical_lines as f64 / line_metrics.total_lines as f64
        } else {
            0.0
        };

        let comment_ratio = if line_metrics.logical_lines > 0 {
            line_metrics.comment_lines as f64 / line_metrics.logical_lines as f64
        } else {
            0.0
        };

        let language_metrics = if self.config.enable_language_specific {
            self.analyze_language_specific(content, language)
        } else {
            LanguageSpecificMetrics::default(language)
        };

        Ok(ComplexityMetrics {
            cyclomatic_complexity,
            max_nesting_depth,
            function_count,
            logical_lines: line_metrics.logical_lines,
            comment_lines: line_metrics.comment_lines,
            blank_lines: line_metrics.blank_lines,
            total_lines: line_metrics.total_lines,
            cognitive_complexity,
            maintainability_index,
            average_function_length,
            code_density,
            comment_ratio,
            language_metrics,
        })
    }

    /// Analyze line-based metrics
    fn analyze_lines(&self, content: &str) -> LineMetrics {
        let mut logical_lines = 0;
        let mut comment_lines = 0;
        let mut blank_lines = 0;
        let total_lines = content.lines().count();

        for line in content.lines() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                blank_lines += 1;
            } else if self.is_comment_line(trimmed) {
                comment_lines += 1;
            } else {
                logical_lines += 1;
            }
        }

        LineMetrics {
            logical_lines,
            comment_lines,
            blank_lines,
            total_lines,
        }
    }

    /// Check if a line is primarily a comment
    fn is_comment_line(&self, line: &str) -> bool {
        let trimmed = line.trim();

        // Common comment patterns
        trimmed.starts_with("//")
            || trimmed.starts_with("#")
            || trimmed.starts_with("/*")
            || trimmed.starts_with("*")
            || trimmed.starts_with("*/")
            || trimmed.starts_with("<!--")
            || trimmed.starts_with("--")
            || trimmed.starts_with("%")
            || trimmed.starts_with(";")
    }

    /// Calculate cyclomatic complexity (FAST version - no regex, minimal string ops)
    fn calculate_cyclomatic_complexity(&self, content: &str, language: &str) -> usize {
        let mut complexity = 1; // Base complexity

        // Use simple byte-based counting instead of expensive string matching
        match language.to_lowercase().as_str() {
            "rust" => {
                // Count key complexity indicators quickly
                complexity += content.matches(" if ").count();
                complexity += content.matches(" else ").count();
                complexity += content.matches(" match ").count();
                complexity += content.matches(" while ").count();
                complexity += content.matches(" for ").count();
                complexity += content.matches("?").count(); // Error handling
                complexity += content.matches("&&").count();
                complexity += content.matches("||").count();
            }
            "python" => {
                complexity += content.matches(" if ").count();
                complexity += content.matches(" elif ").count();
                complexity += content.matches(" while ").count();
                complexity += content.matches(" for ").count();
                complexity += content.matches(" except ").count();
                complexity += content.matches(" and ").count();
                complexity += content.matches(" or ").count();
            }
            "javascript" | "typescript" => {
                complexity += content.matches(" if ").count();
                complexity += content.matches(" while ").count();
                complexity += content.matches(" for ").count();
                complexity += content.matches(" catch ").count();
                complexity += content.matches("&&").count();
                complexity += content.matches("||").count();
                complexity += content.matches("?").count();
            }
            _ => {
                // Generic fallback - just count some basic patterns
                complexity += content.matches(" if ").count();
                complexity += content.matches(" while ").count();
                complexity += content.matches(" for ").count();
            }
        }

        complexity.max(1) // Ensure minimum complexity of 1
    }

    /// Get complexity-increasing keywords for a language
    fn get_complexity_keywords(&self, language: &str) -> Vec<&'static str> {
        match language.to_lowercase().as_str() {
            "rust" => vec![
                "if", "else if", "match", "while", "for", "loop", "catch", "?", "&&", "||",
                "break", "continue",
            ],
            "python" => vec![
                "if", "elif", "while", "for", "except", "and", "or", "break", "continue", "return",
                "yield",
            ],
            "javascript" | "typescript" => vec![
                "if", "else if", "while", "for", "catch", "case", "&&", "||", "?", "break",
                "continue", "return",
            ],
            "java" | "c#" => vec![
                "if", "else if", "while", "for", "foreach", "catch", "case", "&&", "||", "?",
                "break", "continue", "return",
            ],
            "go" => vec![
                "if", "else if", "for", "switch", "case", "select", "&&", "||", "break",
                "continue", "return",
            ],
            "c" | "cpp" | "c++" => vec![
                "if", "else if", "while", "for", "switch", "case", "&&", "||", "?", "break",
                "continue", "return",
            ],
            _ => vec![
                "if", "else", "while", "for", "switch", "case", "&&", "||", "?", "break",
                "continue", "return",
            ],
        }
    }

    /// Calculate maximum nesting depth (FAST version - no char iteration)
    fn calculate_max_nesting_depth(&self, content: &str, _language: &str) -> usize {
        let mut max_depth: usize = 0;
        let mut current_depth: usize = 0;

        // Simple heuristic: count braces on each line (much faster than char-by-char)
        for line in content.lines() {
            // Count opening and closing braces in one pass
            let opens = line.matches('{').count();
            let closes = line.matches('}').count();

            current_depth += opens;
            max_depth = max_depth.max(current_depth);
            current_depth = current_depth.saturating_sub(closes);
        }

        max_depth
    }

    /// Get nesting characters for a language
    fn get_nesting_chars(&self, language: &str) -> (Vec<char>, Vec<char>) {
        match language.to_lowercase().as_str() {
            "python" => {
                // Python uses indentation, but we can still track some nesting
                (vec!['{', '[', '('], vec!['}', ']', ')'])
            }
            _ => {
                // Most C-style languages
                (vec!['{', '[', '('], vec!['}', ']', ')'])
            }
        }
    }

    /// Count functions in the content
    fn count_functions(&self, content: &str, language: &str) -> usize {
        let function_keywords = self.get_function_keywords(language);
        let mut count = 0;

        for line in content.lines() {
            let line = line.trim();

            for keyword in &function_keywords {
                if line.starts_with(keyword) || line.contains(&format!(" {}", keyword)) {
                    count += 1;
                    break; // Only count once per line
                }
            }
        }

        count
    }

    /// Get function declaration keywords for a language
    fn get_function_keywords(&self, language: &str) -> Vec<&'static str> {
        match language.to_lowercase().as_str() {
            "rust" => vec!["fn ", "pub fn ", "async fn ", "pub async fn "],
            "python" => vec!["def ", "async def ", "class "],
            "javascript" | "typescript" => {
                vec!["function ", "const ", "let ", "var ", "async function "]
            }
            "java" => vec!["public ", "private ", "protected ", "static "],
            "c#" => vec!["public ", "private ", "protected ", "internal ", "static "],
            "go" => vec!["func "],
            "c" | "cpp" | "c++" => vec!["int ", "void ", "char ", "float ", "double ", "static "],
            _ => vec!["function ", "def ", "fn "],
        }
    }

    /// Calculate cognitive complexity (more intuitive than cyclomatic)
    fn calculate_cognitive_complexity(&self, content: &str, language: &str) -> usize {
        let mut complexity: usize = 0;
        let mut nesting_level: usize = 0;

        for line in content.lines() {
            let line = line.trim().to_lowercase();

            // Track nesting level changes
            if line.contains('{') {
                nesting_level += 1;
            }
            if line.contains('}') {
                nesting_level = nesting_level.saturating_sub(1);
            }

            // Cognitive complexity increments
            if line.contains("if") || line.contains("while") || line.contains("for") {
                complexity += 1 + nesting_level; // Base increment + nesting penalty
            }

            if line.contains("else if") || line.contains("elif") {
                complexity += 1;
            }

            if line.contains("catch") || line.contains("except") {
                complexity += 1 + nesting_level;
            }

            if line.contains("switch") || line.contains("match") {
                complexity += 1 + nesting_level;
            }

            // Recursive calls add complexity
            if self.has_recursive_call(&line, language) {
                complexity += 1;
            }
        }

        complexity
    }

    /// Check if a line contains a recursive call
    fn has_recursive_call(&self, line: &str, _language: &str) -> bool {
        // Simple heuristic: look for function calls that might be recursive
        line.contains("self.")
            || line.contains("this.")
            || line.contains("recursive")
            || line.contains("recurse")
    }

    /// Calculate maintainability index (Halstead-based)
    fn calculate_maintainability_index(
        &self,
        line_metrics: &LineMetrics,
        cyclomatic: usize,
        functions: usize,
    ) -> f64 {
        // Simplified maintainability index calculation
        // Real calculation would use Halstead metrics

        let volume = (line_metrics.logical_lines as f64).ln();
        let complexity = cyclomatic as f64;
        let lloc = line_metrics.logical_lines as f64;

        // Simplified formula (real one is more complex)
        let mi = 171.0 - 5.2 * volume - 0.23 * complexity - 16.2 * lloc.ln();

        // Normalize to 0-100 range
        mi.max(0.0).min(100.0)
    }

    /// Analyze language-specific metrics
    fn analyze_language_specific(&self, content: &str, language: &str) -> LanguageSpecificMetrics {
        let mut complexity_factors = HashMap::new();
        let import_count = self.count_imports(content, language);
        let export_count = self.count_exports(content, language);
        let api_surface_area = self.estimate_api_surface_area(content, language);

        // Language-specific complexity factors
        match language.to_lowercase().as_str() {
            "rust" => {
                complexity_factors.insert(
                    "ownership_complexity".to_string(),
                    self.calculate_ownership_complexity(content),
                );
                complexity_factors.insert(
                    "trait_complexity".to_string(),
                    self.count_trait_usage(content) as f64,
                );
                complexity_factors.insert(
                    "macro_complexity".to_string(),
                    self.count_macro_usage(content) as f64,
                );
            }
            "python" => {
                complexity_factors.insert(
                    "decorator_complexity".to_string(),
                    self.count_decorators(content) as f64,
                );
                complexity_factors.insert(
                    "comprehension_complexity".to_string(),
                    self.count_comprehensions(content) as f64,
                );
            }
            "javascript" | "typescript" => {
                complexity_factors.insert(
                    "closure_complexity".to_string(),
                    self.count_closures(content) as f64,
                );
                complexity_factors.insert(
                    "promise_complexity".to_string(),
                    self.count_async_patterns(content) as f64,
                );
            }
            _ => {
                // Generic complexity factors
                complexity_factors.insert("generic_complexity".to_string(), 1.0);
            }
        }

        LanguageSpecificMetrics {
            language: language.to_string(),
            complexity_factors,
            import_count,
            export_count,
            api_surface_area,
        }
    }

    /// Count import statements
    fn count_imports(&self, content: &str, language: &str) -> usize {
        let import_patterns = match language.to_lowercase().as_str() {
            "rust" => vec!["use ", "extern crate "],
            "python" => vec!["import ", "from "],
            "javascript" | "typescript" => vec!["import ", "require(", "const ", "let "],
            "java" => vec!["import "],
            "go" => vec!["import "],
            _ => vec!["import ", "include ", "use "],
        };

        content
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                import_patterns
                    .iter()
                    .any(|pattern| trimmed.starts_with(pattern))
            })
            .count()
    }

    /// Count export statements
    fn count_exports(&self, content: &str, language: &str) -> usize {
        let export_patterns = match language.to_lowercase().as_str() {
            "rust" => vec!["pub fn ", "pub struct ", "pub enum ", "pub trait "],
            "python" => vec!["def ", "class "], // Python exports everything by default
            "javascript" | "typescript" => vec!["export ", "module.exports"],
            "java" => vec!["public class ", "public interface ", "public enum "],
            _ => vec!["public ", "export "],
        };

        content
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                export_patterns
                    .iter()
                    .any(|pattern| trimmed.contains(pattern))
            })
            .count()
    }

    /// Estimate API surface area
    fn estimate_api_surface_area(&self, content: &str, language: &str) -> usize {
        // Simple heuristic: count public functions, classes, etc.
        let public_items = self.count_exports(content, language);
        let function_count = self.count_functions(content, language);

        // API surface area is roughly the number of publicly accessible items
        public_items.min(function_count)
    }

    // Language-specific complexity calculations
    fn calculate_ownership_complexity(&self, content: &str) -> f64 {
        let ownership_keywords = ["&", "&mut", "Box<", "Rc<", "Arc<", "RefCell<", "Mutex<"];
        let mut complexity = 0.0;

        for line in content.lines() {
            for keyword in &ownership_keywords {
                complexity += line.matches(keyword).count() as f64 * 0.5;
            }
        }

        complexity
    }

    fn count_trait_usage(&self, content: &str) -> usize {
        content
            .lines()
            .filter(|line| line.contains("trait ") || line.contains("impl "))
            .count()
    }

    fn count_macro_usage(&self, content: &str) -> usize {
        content
            .lines()
            .filter(|line| line.contains("macro_rules!") || line.contains("!"))
            .count()
    }

    fn count_decorators(&self, content: &str) -> usize {
        content
            .lines()
            .filter(|line| line.trim().starts_with("@"))
            .count()
    }

    fn count_comprehensions(&self, content: &str) -> usize {
        content
            .lines()
            .filter(|line| {
                line.contains("[") && line.contains("for ") && line.contains("in ")
                    || line.contains("{") && line.contains("for ") && line.contains("in ")
            })
            .count()
    }

    fn count_closures(&self, content: &str) -> usize {
        content
            .lines()
            .filter(|line| line.contains("=>") || line.contains("function("))
            .count()
    }

    fn count_async_patterns(&self, content: &str) -> usize {
        content
            .lines()
            .filter(|line| {
                line.contains("async")
                    || line.contains("await")
                    || line.contains("Promise")
                    || line.contains(".then(")
            })
            .count()
    }
}

/// Line-based metrics
#[derive(Debug)]
struct LineMetrics {
    logical_lines: usize,
    comment_lines: usize,
    blank_lines: usize,
    total_lines: usize,
}

impl Default for LanguageSpecificMetrics {
    fn default() -> Self {
        Self::default("unknown")
    }
}

impl LanguageSpecificMetrics {
    fn default(language: &str) -> Self {
        Self {
            language: language.to_string(),
            complexity_factors: HashMap::new(),
            import_count: 0,
            export_count: 0,
            api_surface_area: 0,
        }
    }
}

impl ComplexityMetrics {
    /// Get a complexity score (0-1, where higher means more complex)
    pub fn complexity_score(&self) -> f64 {
        // Normalize various complexity metrics to a 0-1 score
        let cyclomatic_score = (self.cyclomatic_complexity as f64 / 20.0).min(1.0);
        let nesting_score = (self.max_nesting_depth as f64 / 8.0).min(1.0);
        let cognitive_score = (self.cognitive_complexity as f64 / 15.0).min(1.0);
        let maintainability_score = (100.0 - self.maintainability_index) / 100.0;

        // Weighted average
        (cyclomatic_score * 0.3
            + nesting_score * 0.2
            + cognitive_score * 0.3
            + maintainability_score * 0.2)
            .min(1.0)
    }

    /// Check if any complexity thresholds are exceeded
    pub fn exceeds_thresholds(&self, thresholds: &ComplexityThresholds) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.cyclomatic_complexity > thresholds.cyclomatic_warning {
            warnings.push(format!(
                "High cyclomatic complexity: {}",
                self.cyclomatic_complexity
            ));
        }

        if self.max_nesting_depth > thresholds.nesting_warning {
            warnings.push(format!("Deep nesting: {}", self.max_nesting_depth));
        }

        if self.average_function_length > thresholds.function_length_warning as f64 {
            warnings.push(format!(
                "Long functions: avg {:.1} lines",
                self.average_function_length
            ));
        }

        if self.maintainability_index < thresholds.maintainability_warning {
            warnings.push(format!(
                "Low maintainability: {:.1}",
                self.maintainability_index
            ));
        }

        warnings
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Complexity: CC={}, Depth={}, Functions={}, MI={:.1}, Cognitive={}",
            self.cyclomatic_complexity,
            self.max_nesting_depth,
            self.function_count,
            self.maintainability_index,
            self.cognitive_complexity
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ComplexityAnalyzer::new();
        assert!(analyzer.config.enable_cognitive_complexity);
        assert!(analyzer.config.enable_maintainability_index);
    }

    #[test]
    fn test_simple_rust_analysis() {
        let analyzer = ComplexityAnalyzer::new();
        let content = r#"
fn main() {
    if x > 0 {
        println!("positive");
    } else {
        println!("negative");
    }
}
"#;

        let metrics = analyzer.analyze_content(content, "rust").unwrap();

        assert!(metrics.cyclomatic_complexity >= 2); // if-else adds complexity
        assert!(metrics.function_count >= 1);
        assert!(metrics.max_nesting_depth >= 1);
        assert!(metrics.total_lines > 0);
    }

    #[test]
    fn test_complex_code_analysis() {
        let analyzer = ComplexityAnalyzer::new();
        let content = r#"
fn complex_function() {
    for i in 0..10 {
        if i % 2 == 0 {
            while some_condition() {
                if another_condition() {
                    match value {
                        1 => do_something(),
                        2 => do_something_else(),
                        _ => default_action(),
                    }
                }
            }
        } else {
            continue;
        }
    }
}
"#;

        let metrics = analyzer.analyze_content(content, "rust").unwrap();

        assert!(metrics.cyclomatic_complexity > 5); // Multiple branches
        assert!(metrics.max_nesting_depth > 3); // Deep nesting
        assert!(metrics.cognitive_complexity > metrics.cyclomatic_complexity); // Cognitive should be higher due to nesting
    }

    #[test]
    fn test_line_analysis() {
        let analyzer = ComplexityAnalyzer::new();
        let content = r#"
// This is a comment
fn test() {
    // Another comment
    let x = 5;
    
    // More comments
    println!("Hello");
}
"#;

        let metrics = analyzer.analyze_content(content, "rust").unwrap();

        assert!(metrics.comment_lines > 0);
        assert!(metrics.blank_lines > 0);
        assert!(metrics.logical_lines > 0);
        assert!(metrics.comment_ratio > 0.0);
        assert!(metrics.code_density > 0.0);
    }

    #[test]
    fn test_language_specific_analysis() {
        let analyzer = ComplexityAnalyzer::new();

        // Test Rust-specific features
        let rust_content = r#"
use std::collections::HashMap;
pub fn test() -> Result<(), Box<dyn Error>> {
    let mut data: Vec<&str> = vec![];
    Ok(())
}
"#;

        let rust_metrics = analyzer.analyze_content(rust_content, "rust").unwrap();
        assert_eq!(rust_metrics.language_metrics.language, "rust");
        assert!(rust_metrics.language_metrics.import_count > 0);

        // Test Python-specific features
        let python_content = r#"
import os
from typing import List

@decorator
def test_function():
    result = [x for x in range(10) if x % 2 == 0]
    return result
"#;

        let python_metrics = analyzer.analyze_content(python_content, "python").unwrap();
        assert_eq!(python_metrics.language_metrics.language, "python");
        assert!(python_metrics.language_metrics.import_count > 0);
    }

    #[test]
    fn test_complexity_score() {
        let analyzer = ComplexityAnalyzer::new();

        // Simple code should have low complexity score
        let simple_content = "fn main() { println!(\"hello\"); }";
        let simple_metrics = analyzer.analyze_content(simple_content, "rust").unwrap();
        let simple_score = simple_metrics.complexity_score();

        // Complex code should have higher complexity score
        let complex_content = r#"
fn complex() {
    for i in 0..100 {
        if i % 2 == 0 {
            while condition() {
                match value {
                    1 => { if nested() { deep(); } },
                    2 => { if more_nested() { deeper(); } },
                    _ => { if even_more() { deepest(); } },
                }
            }
        }
    }
}
"#;
        let complex_metrics = analyzer.analyze_content(complex_content, "rust").unwrap();
        let complex_score = complex_metrics.complexity_score();

        assert!(complex_score > simple_score);
        assert!(simple_score >= 0.0 && simple_score <= 1.0);
        assert!(complex_score >= 0.0 && complex_score <= 1.0);
    }

    #[test]
    fn test_threshold_warnings() {
        let analyzer = ComplexityAnalyzer::new();
        let thresholds = ComplexityThresholds {
            cyclomatic_warning: 5,
            nesting_warning: 2,
            function_length_warning: 10,
            maintainability_warning: 50.0,
        };

        let complex_content = r#"
fn complex_function() {
    for i in 0..10 {
        if i % 2 == 0 {
            while some_condition() {
                if another_condition() {
                    if yet_another() {
                        do_something();
                    }
                }
            }
        }
    }
}
"#;

        let metrics = analyzer.analyze_content(complex_content, "rust").unwrap();
        let warnings = metrics.exceeds_thresholds(&thresholds);

        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.contains("complexity")));
    }

    #[test]
    fn test_metrics_summary() {
        let analyzer = ComplexityAnalyzer::new();
        let content = "fn test() { if x > 0 { return 1; } else { return 0; } }";
        let metrics = analyzer.analyze_content(content, "rust").unwrap();

        let summary = metrics.summary();
        assert!(summary.contains("CC="));
        assert!(summary.contains("Depth="));
        assert!(summary.contains("Functions="));
        assert!(summary.contains("MI="));
        assert!(summary.contains("Cognitive="));
    }
}

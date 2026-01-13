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

    #[test]
    fn test_javascript_metrics() {
        let js_code = r#"
function hello() {
    console.log("Hello");
}

const greet = (name) => {
    return `Hello, ${name}`;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}

async function fetchData() {
    await fetch('/api/data');
}
"#;

        let metrics = LanguageMetrics::calculate(js_code, AstLanguage::JavaScript).unwrap();
        assert_eq!(metrics.language, AstLanguage::JavaScript);
        assert!(metrics.function_count >= 2);
        assert_eq!(metrics.class_count, 1);
        assert!(metrics.complexity.language_factors > 0.0); // Should detect async
    }

    #[test]
    fn test_typescript_metrics() {
        let ts_code = r#"
function hello(): void {
    console.log("Hello");
}

const greet = async (name: string): Promise<string> => {
    return `Hello, ${name}`;
}

class Calculator<T> {
    add(a: T, b: T): T {
        return a + b;
    }
}
"#;

        let metrics = LanguageMetrics::calculate(ts_code, AstLanguage::TypeScript).unwrap();
        assert_eq!(metrics.language, AstLanguage::TypeScript);
        assert!(metrics.function_count >= 1);
        assert!(metrics.class_count >= 1);
        // TypeScript with generics should have higher complexity
        assert!(metrics.complexity.language_factors > 0.0);
    }

    #[test]
    fn test_go_metrics() {
        let go_code = r#"
package main

func main() {
    fmt.Println("Hello")
}

func hello(name string) string {
    return "Hello, " + name
}

type Calculator struct {
    value float64
}

func (c *Calculator) Add(a, b float64) float64 {
    return a + b
}
"#;

        let metrics = LanguageMetrics::calculate(go_code, AstLanguage::Go).unwrap();
        assert_eq!(metrics.language, AstLanguage::Go);
        assert!(metrics.function_count >= 2);
        assert!(metrics.class_count >= 1);
    }

    #[test]
    fn test_go_goroutine_complexity() {
        let go_code = r#"
func main() {
    go doWork()
    ch := make(chan int)
    ch <- 42
    defer cleanup()
}
"#;

        let metrics = LanguageMetrics::calculate(go_code, AstLanguage::Go).unwrap();
        // Should detect goroutine, channel, and defer patterns
        assert!(metrics.complexity.language_factors > 0.3);
    }

    #[test]
    fn test_rust_complexity_factors() {
        let rust_code = r#"
fn process<'a>(data: &'a str) -> Result<Option<String>, Error> {
    match data.parse() {
        Ok(v) => Some(v),
        Err(_) => None,
    }
}

macro_rules! my_macro {
    () => { println!("Hello"); }
}
"#;

        let metrics = LanguageMetrics::calculate(rust_code, AstLanguage::Rust).unwrap();
        // Should detect lifetimes, match, Result, Option, and macros
        assert!(metrics.complexity.language_factors > 0.5);
        assert!(metrics.complexity.idiom_score > 0.0);
    }

    #[test]
    fn test_python_django_framework() {
        let python_code = r#"
from django.db import models
from django.views import View

class MyModel(models.Model):
    name = models.CharField(max_length=100)

@login_required
def my_view(request):
    return render(request, 'template.html')
"#;

        let metrics = LanguageMetrics::calculate(python_code, AstLanguage::Python).unwrap();
        // Should detect Django framework and decorators
        assert!(metrics.complexity.framework_complexity > 0.0);
        assert!(metrics.complexity.language_factors > 0.0); // decorators
    }

    #[test]
    fn test_python_flask_framework() {
        let python_code = r#"
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"
"#;

        let metrics = LanguageMetrics::calculate(python_code, AstLanguage::Python).unwrap();
        // Should detect Flask framework and decorators
        assert!(metrics.complexity.framework_complexity > 0.0);
    }

    #[test]
    fn test_python_list_comprehension() {
        let python_code = r#"
def process_data(items):
    return [x * 2 for x in items if x > 0]
"#;

        let metrics = LanguageMetrics::calculate(python_code, AstLanguage::Python).unwrap();
        // Should detect list comprehension
        assert!(metrics.complexity.language_factors >= 0.1);
    }

    #[test]
    fn test_javascript_react_framework() {
        let js_code = r#"
import React from 'react';

function MyComponent() {
    return <div>Hello</div>;
}
"#;

        let metrics = LanguageMetrics::calculate(js_code, AstLanguage::JavaScript).unwrap();
        // Should detect React
        assert!(metrics.complexity.framework_complexity > 0.0);
    }

    #[test]
    fn test_javascript_promise_complexity() {
        let js_code = r#"
function fetchData() {
    return new Promise((resolve, reject) => {
        resolve(data);
    });
}
"#;

        let metrics = LanguageMetrics::calculate(js_code, AstLanguage::JavaScript).unwrap();
        // Should detect Promise
        assert!(metrics.complexity.language_factors >= 0.1);
    }

    #[test]
    fn test_maintainability_large_file() {
        // Create a file with >500 lines but without many functions
        // so it doesn't get the "small functions" bonus
        let large_code = "let x = 1;\n".repeat(600);

        let metrics = LanguageMetrics::calculate(&large_code, AstLanguage::Rust).unwrap();
        assert!(metrics.lines_of_code > 500);
        // Large file with no functions should have lower maintainability due to size penalty
        assert!(metrics.maintainability_score < 100.0);
    }

    #[test]
    fn test_maintainability_small_functions() {
        let modular_code = r#"
fn a() {}
fn b() {}
fn c() {}
fn d() {}
fn e() {}
"#;

        let metrics = LanguageMetrics::calculate(modular_code, AstLanguage::Rust).unwrap();
        // Many small functions should have good maintainability
        assert!(metrics.maintainability_score >= 90.0);
    }

    #[test]
    fn test_maintainability_large_functions() {
        // Create code with one function and many lines
        let mut lines = vec!["fn very_long_function() {"];
        for _ in 0..200 {
            lines.push("    let x = 1;");
        }
        lines.push("}");
        let long_function_code = lines.join("\n");

        let metrics = LanguageMetrics::calculate(&long_function_code, AstLanguage::Rust).unwrap();
        // Large functions should have lower maintainability
        assert!(metrics.maintainability_score < 95.0);
    }

    #[test]
    fn test_unsupported_language() {
        let code = "some code here";

        // Using a language that doesn't have specific metrics (Html uses the fallthrough)
        let metrics = LanguageMetrics::calculate(code, AstLanguage::Html).unwrap();
        assert_eq!(metrics.function_count, 0);
        assert_eq!(metrics.class_count, 0);
        assert!(metrics.complexity.language_factors < 0.2); // Minimal factors
    }

    #[test]
    fn test_language_specific_complexity_struct() {
        let complexity = LanguageSpecificComplexity {
            base_complexity: 1.0,
            language_factors: 0.5,
            idiom_score: 0.2,
            framework_complexity: 0.1,
        };

        let cloned = complexity.clone();
        assert_eq!(complexity.base_complexity, cloned.base_complexity);
        assert_eq!(complexity.language_factors, cloned.language_factors);
        assert_eq!(complexity.idiom_score, cloned.idiom_score);
        assert_eq!(complexity.framework_complexity, cloned.framework_complexity);
    }

    #[test]
    fn test_language_metrics_debug() {
        let code = "fn main() {}";
        let metrics = LanguageMetrics::calculate(code, AstLanguage::Rust).unwrap();
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("LanguageMetrics"));
        assert!(debug_str.contains("Rust"));
    }

    #[test]
    fn test_language_metrics_clone() {
        let code = "fn main() {}";
        let metrics = LanguageMetrics::calculate(code, AstLanguage::Rust).unwrap();
        let cloned = metrics.clone();
        assert_eq!(metrics.language, cloned.language);
        assert_eq!(metrics.lines_of_code, cloned.lines_of_code);
        assert_eq!(metrics.function_count, cloned.function_count);
    }

    #[test]
    fn test_rust_pub_fn() {
        let rust_code = r#"
pub fn public_function() {}
fn private_function() {}
pub async fn async_pub_function() {}
"#;

        let metrics = LanguageMetrics::calculate(rust_code, AstLanguage::Rust).unwrap();
        assert!(metrics.function_count >= 2); // Should count both pub and private fn
    }

    #[test]
    fn test_rust_enum() {
        let rust_code = r#"
enum Color {
    Red,
    Green,
    Blue,
}

pub enum Status {
    Active,
    Inactive,
}
"#;

        let metrics = LanguageMetrics::calculate(rust_code, AstLanguage::Rust).unwrap();
        assert!(metrics.class_count >= 2); // enums count as classes
    }

    #[test]
    fn test_rust_if_let() {
        let rust_code = r#"
fn process(value: Option<i32>) -> i32 {
    if let Some(v) = value {
        v
    } else {
        0
    }
}
"#;

        let metrics = LanguageMetrics::calculate(rust_code, AstLanguage::Rust).unwrap();
        // Should detect if let pattern
        assert!(metrics.complexity.language_factors > 0.0);
    }

    #[test]
    fn test_empty_code() {
        let empty_code = "";
        let metrics = LanguageMetrics::calculate(empty_code, AstLanguage::Python).unwrap();
        assert_eq!(metrics.lines_of_code, 0);
        assert_eq!(metrics.function_count, 0);
        assert_eq!(metrics.class_count, 0);
    }

    #[test]
    fn test_complexity_struct_serialize() {
        let complexity = LanguageSpecificComplexity {
            base_complexity: 1.0,
            language_factors: 0.5,
            idiom_score: 0.2,
            framework_complexity: 0.1,
        };

        let json = serde_json::to_string(&complexity).unwrap();
        let deserialized: LanguageSpecificComplexity = serde_json::from_str(&json).unwrap();
        assert!((complexity.base_complexity - deserialized.base_complexity).abs() < 0.001);
    }

    #[test]
    fn test_metrics_serialize() {
        let code = "fn main() {}";
        let metrics = LanguageMetrics::calculate(code, AstLanguage::Rust).unwrap();

        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: LanguageMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics.language, deserialized.language);
        assert_eq!(metrics.lines_of_code, deserialized.lines_of_code);
    }

    #[test]
    fn test_typescript_react_with_generics() {
        let ts_code = r#"
import React from 'react';

async function fetchData<T>(): Promise<T> {
    const response = await fetch('/api');
    return response.json();
}

class Component<Props> {
    render() {
        return <div></div>;
    }
}
"#;

        let metrics = LanguageMetrics::calculate(ts_code, AstLanguage::TypeScript).unwrap();
        // Should detect async, Promise, generics, and React
        assert!(metrics.complexity.language_factors > 0.2);
        assert!(metrics.complexity.framework_complexity > 0.0);
    }

    #[test]
    fn test_go_channel_operations() {
        let go_code = r#"
func producer(ch chan int) {
    ch <- 42
}

func consumer(ch chan int) {
    value := <-ch
}
"#;

        let metrics = LanguageMetrics::calculate(go_code, AstLanguage::Go).unwrap();
        // Should detect channel operations
        assert!(metrics.complexity.language_factors >= 0.3);
    }

    #[test]
    fn test_maintainability_score_bounds() {
        // Test that maintainability score stays within 0-100 bounds
        let huge_code = "fn f() { let x = 1; }\n".repeat(10000);
        let metrics = LanguageMetrics::calculate(&huge_code, AstLanguage::Rust).unwrap();
        assert!(metrics.maintainability_score >= 0.0);
        assert!(metrics.maintainability_score <= 100.0);
    }

    #[test]
    fn test_python_import_django() {
        let python_code = r#"
import django.urls
"#;

        let metrics = LanguageMetrics::calculate(python_code, AstLanguage::Python).unwrap();
        assert!(metrics.complexity.framework_complexity >= 0.2);
    }
}

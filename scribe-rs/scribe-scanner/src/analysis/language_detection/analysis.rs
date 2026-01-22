//! AST-based analysis methods for language detection.

use super::detection::get_likely_languages_from_content;
use super::types::SyntaxAnalyzer;
use scribe_core::Language;
use std::collections::HashMap;
use tree_sitter::{Node, Parser};

/// Count import-related AST nodes for a specific language
pub fn count_import_nodes(node: &Node, language: &Language) -> usize {
    let mut count = 0;
    let import_types: &[&str] = match language {
        Language::Python => &["import_statement", "import_from_statement"],
        Language::JavaScript | Language::TypeScript => &["import_statement", "import_declaration"],
        Language::Rust => &["use_declaration"],
        Language::Go => &["import_spec", "import_declaration"],
        Language::Java => &["import_declaration"],
        _ => &[],
    };

    count_nodes_recursive(node, import_types, &mut count);
    count
}

/// Calculate structural score based on AST node patterns
pub fn calculate_structural_score(node: &Node, analyzer: &SyntaxAnalyzer) -> f32 {
    let mut score = 0.0;

    for pattern in &analyzer.structural_patterns {
        let count = count_specific_nodes(node, pattern);
        if count > 0 {
            let weight = analyzer.confidence_weights.get(pattern).unwrap_or(&0.5);
            score += (count as f32) * weight;
        }
    }

    (score / 10.0).min(1.0)
}

/// Recursively count nodes of specific types
pub fn count_nodes_recursive(node: &Node, target_types: &[&str], count: &mut usize) {
    if target_types.contains(&node.kind()) {
        *count += 1;
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            count_nodes_recursive(&child, target_types, count);
        }
    }
}

/// Count specific node types in AST
pub fn count_specific_nodes(node: &Node, target_type: &str) -> usize {
    let mut count = 0;
    count_nodes_recursive(node, &[target_type], &mut count);
    count
}

/// Analyze import patterns using AST parsing with extension-first optimization
pub fn analyze_import_patterns(
    content: &str,
    ast_parsers: &mut HashMap<Language, Parser>,
) -> Vec<(Language, f32)> {
    let mut results = Vec::new();

    let likely_languages = get_likely_languages_from_content(content);

    for language in likely_languages {
        if let Some(parser) = ast_parsers.get_mut(&language) {
            if let Some(tree) = parser.parse(content, None) {
                let root_node = tree.root_node();
                let import_count = count_import_nodes(&root_node, &language);

                if import_count > 0 {
                    let confidence = (import_count as f32 / 10.0).min(0.9);
                    results.push((language, confidence));

                    if confidence > 0.7 {
                        break;
                    }
                }
            }
        }
    }

    results
}

/// Perform AST-based structural analysis of content with extension-first optimization
pub fn statistical_analysis(
    content: &str,
    ast_parsers: &mut HashMap<Language, Parser>,
    syntax_analyzers: &HashMap<Language, SyntaxAnalyzer>,
) -> Vec<(Language, f32)> {
    let mut results = Vec::new();

    let likely_languages = get_likely_languages_from_content(content);

    for language in likely_languages {
        if let Some(analyzer) = syntax_analyzers.get(&language) {
            if let Some(parser) = ast_parsers.get_mut(&language) {
                if let Some(tree) = parser.parse(content, None) {
                    let root_node = tree.root_node();
                    let structural_score = calculate_structural_score(&root_node, analyzer);

                    if structural_score > 0.0 {
                        results.push((language, structural_score));

                        if structural_score > 0.8 {
                            break;
                        }
                    }
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_rust_parser() -> Parser {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_rust::language()).unwrap();
        parser
    }

    fn create_python_parser() -> Parser {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_python::language()).unwrap();
        parser
    }

    fn create_javascript_parser() -> Parser {
        let mut parser = Parser::new();
        parser
            .set_language(tree_sitter_javascript::language())
            .unwrap();
        parser
    }

    fn create_go_parser() -> Parser {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_go::language()).unwrap();
        parser
    }

    #[test]
    fn test_count_import_nodes_rust() {
        let mut parser = create_rust_parser();
        let content = "use std::io;\nuse std::fs::File;";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Rust);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_import_nodes_rust_no_imports() {
        let mut parser = create_rust_parser();
        let content = "fn main() { println!(\"Hello\"); }";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Rust);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_import_nodes_python() {
        let mut parser = create_python_parser();
        let content = "import os\nfrom pathlib import Path";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Python);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_import_nodes_python_no_imports() {
        let mut parser = create_python_parser();
        let content = "def hello():\n    print('Hello')";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Python);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_import_nodes_javascript() {
        let mut parser = create_javascript_parser();
        let content = "import React from 'react';\nimport { useState } from 'react';";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::JavaScript);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_import_nodes_typescript() {
        let mut parser = create_javascript_parser();
        let content = "import React from 'react';";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        // TypeScript uses the same import types as JavaScript
        let count = count_import_nodes(&root, &Language::TypeScript);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_count_import_nodes_go() {
        let mut parser = create_go_parser();
        let content = "package main\n\nimport \"fmt\"\nimport \"os\"";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Go);
        // Each import statement has import_declaration AND import_spec, so 2 imports = 4 nodes
        assert_eq!(count, 4);
    }

    #[test]
    fn test_count_import_nodes_java_empty() {
        // Java parser not available in this crate, test the language match arm
        let mut parser = create_rust_parser();
        let content = "fn main() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        // Java returns import_declaration types, but with Rust parser won't match
        let count = count_import_nodes(&root, &Language::Java);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_import_nodes_unknown_language() {
        let mut parser = create_rust_parser();
        let content = "use std::io;";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        // Unknown language should return 0
        let count = count_import_nodes(&root, &Language::Unknown);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_nodes_recursive_basic() {
        let mut parser = create_rust_parser();
        let content = "fn main() {}\nfn helper() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let mut count = 0;
        count_nodes_recursive(&root, &["function_item"], &mut count);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_nodes_recursive_no_match() {
        let mut parser = create_rust_parser();
        let content = "let x = 1;";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let mut count = 0;
        count_nodes_recursive(&root, &["function_item"], &mut count);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_nodes_recursive_multiple_types() {
        let mut parser = create_rust_parser();
        let content = "use std::io;\nfn main() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let mut count = 0;
        count_nodes_recursive(&root, &["use_declaration", "function_item"], &mut count);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_specific_nodes() {
        let mut parser = create_rust_parser();
        let content = "fn one() {}\nfn two() {}\nfn three() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_specific_nodes(&root, "function_item");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_count_specific_nodes_nested() {
        let mut parser = create_rust_parser();
        let content = "mod inner { fn nested() {} }\nfn outer() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_specific_nodes(&root, "function_item");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_calculate_structural_score_no_patterns() {
        let mut parser = create_rust_parser();
        let content = "fn main() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let analyzer = SyntaxAnalyzer {
            language: Language::Rust,
            keywords: vec![],
            structural_patterns: vec![],
            confidence_weights: HashMap::new(),
        };

        let score = calculate_structural_score(&root, &analyzer);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_calculate_structural_score_with_patterns() {
        let mut parser = create_rust_parser();
        let content = "fn main() {}\nfn helper() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let mut weights = HashMap::new();
        weights.insert("function_item".to_string(), 0.8_f32);

        let analyzer = SyntaxAnalyzer {
            language: Language::Rust,
            keywords: vec![],
            structural_patterns: vec!["function_item".to_string()],
            confidence_weights: weights,
        };

        let score = calculate_structural_score(&root, &analyzer);
        // 2 functions * 0.8 weight / 10.0 = 0.16
        assert!((score - 0.16).abs() < 0.01);
    }

    #[test]
    fn test_calculate_structural_score_capped_at_one() {
        let mut parser = create_rust_parser();
        // Create many functions to exceed 1.0 score
        let content = "fn a() {}\nfn b() {}\nfn c() {}\nfn d() {}\nfn e() {}\nfn f() {}\nfn g() {}\nfn h() {}\nfn i() {}\nfn j() {}\nfn k() {}\nfn l() {}\nfn m() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let mut weights = HashMap::new();
        weights.insert("function_item".to_string(), 1.0_f32);

        let analyzer = SyntaxAnalyzer {
            language: Language::Rust,
            keywords: vec![],
            structural_patterns: vec!["function_item".to_string()],
            confidence_weights: weights,
        };

        let score = calculate_structural_score(&root, &analyzer);
        // Score should be capped at 1.0
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_structural_score_default_weight() {
        let mut parser = create_rust_parser();
        let content = "fn main() {}";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        // No weight defined for function_item, should use default 0.5
        let analyzer = SyntaxAnalyzer {
            language: Language::Rust,
            keywords: vec![],
            structural_patterns: vec!["function_item".to_string()],
            confidence_weights: HashMap::new(),
        };

        let score = calculate_structural_score(&root, &analyzer);
        // 1 function * 0.5 default weight / 10.0 = 0.05
        assert!((score - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_analyze_import_patterns_empty_content() {
        let mut parsers = HashMap::new();

        let results = analyze_import_patterns("", &mut parsers);
        assert!(results.is_empty());
    }

    #[test]
    fn test_analyze_import_patterns_no_matching_parser() {
        let mut parsers = HashMap::new();

        // Content that looks like Rust but no parser available
        let results = analyze_import_patterns("use std::io;", &mut parsers);
        assert!(results.is_empty());
    }

    #[test]
    fn test_statistical_analysis_empty_content() {
        let mut parsers = HashMap::new();
        let analyzers = HashMap::new();

        let results = statistical_analysis("", &mut parsers, &analyzers);
        assert!(results.is_empty());
    }

    #[test]
    fn test_statistical_analysis_no_matching_parser() {
        let mut parsers = HashMap::new();
        let analyzers = HashMap::new();

        let results = statistical_analysis("fn main() {}", &mut parsers, &analyzers);
        assert!(results.is_empty());
    }

    #[test]
    fn test_statistical_analysis_no_matching_analyzer() {
        let mut parsers = HashMap::new();
        parsers.insert(Language::Rust, create_rust_parser());
        let analyzers = HashMap::new(); // Empty analyzers

        let results = statistical_analysis("fn main() {}", &mut parsers, &analyzers);
        assert!(results.is_empty());
    }

    #[test]
    fn test_count_import_nodes_nested_rust() {
        let mut parser = create_rust_parser();
        let content = "mod inner {\n    use std::io;\n}\nuse std::fs;";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Rust);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_import_nodes_python_multiple_from() {
        let mut parser = create_python_parser();
        let content = "from os import path, getcwd\nfrom sys import argv";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Python);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_import_nodes_go_grouped() {
        let mut parser = create_go_parser();
        let content = "package main\n\nimport (\n    \"fmt\"\n    \"os\"\n)";
        let tree = parser.parse(content, None).unwrap();
        let root = tree.root_node();

        let count = count_import_nodes(&root, &Language::Go);
        // Grouped imports are still separate import_spec nodes
        assert!(count >= 2);
    }
}

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
        Language::JavaScript | Language::TypeScript => {
            &["import_statement", "import_declaration"]
        }
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

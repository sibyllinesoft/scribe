//! Tests for import graph analysis module.

use super::super::DocumentAnalysis;
use super::*;

// Mock scan result for testing
#[derive(Debug)]
struct MockScanResult {
    path: String,
    relative_path: String,
    depth: usize,
    imports: Option<Vec<String>>,
}

impl MockScanResult {
    fn new(path: &str, imports: Vec<&str>) -> Self {
        Self {
            path: path.to_string(),
            relative_path: path.to_string(),
            depth: path.matches('/').count(),
            imports: Some(imports.iter().map(|s| s.to_string()).collect()),
        }
    }
}

impl ScanResult for MockScanResult {
    fn path(&self) -> &str {
        &self.path
    }
    fn relative_path(&self) -> &str {
        &self.relative_path
    }
    fn depth(&self) -> usize {
        self.depth
    }
    fn is_docs(&self) -> bool {
        false
    }
    fn is_readme(&self) -> bool {
        false
    }
    fn is_test(&self) -> bool {
        false
    }
    fn is_entrypoint(&self) -> bool {
        false
    }
    fn has_examples(&self) -> bool {
        false
    }
    fn priority_boost(&self) -> f64 {
        0.0
    }
    fn churn_score(&self) -> f64 {
        0.0
    }
    fn centrality_in(&self) -> f64 {
        0.0
    }
    fn imports(&self) -> Option<&[String]> {
        self.imports.as_deref()
    }
    fn doc_analysis(&self) -> Option<&DocumentAnalysis> {
        None
    }
}

#[test]
fn test_import_graph_creation() {
    let mut graph = ImportGraph::new();

    let idx1 = graph.add_node("file1.rs".to_string());
    let idx2 = graph.add_node("file2.rs".to_string());

    assert_eq!(idx1, 0);
    assert_eq!(idx2, 1);
    assert_eq!(graph.nodes.len(), 2);

    graph.add_edge(idx1, idx2);
    assert_eq!(graph.dependencies[idx1].len(), 1);
    assert_eq!(graph.dependents[idx2].len(), 1);
}

#[test]
fn test_graph_builder() {
    let files = vec![
        MockScanResult::new("src/main.rs", vec!["src/lib.rs", "src/utils.rs"]),
        MockScanResult::new("src/lib.rs", vec!["src/utils.rs"]),
        MockScanResult::new("src/utils.rs", vec![]),
    ];

    let mut builder = ImportGraphBuilder::new().unwrap();
    let result = builder.build_graph(&files);
    assert!(result.is_ok());

    let graph = result.unwrap();
    assert_eq!(graph.nodes.len(), 3);

    let stats = graph.stats();
    assert_eq!(stats.node_count, 3);
    assert!(stats.edge_count > 0);
}

#[test]
fn test_import_matching() {
    // Direct matches
    assert!(import_matches_file("src/utils", "src/utils.rs"));
    assert!(import_matches_file("./lib", "src/lib.js"));

    // Module-style matches
    assert!(import_matches_file(
        "std::collections::HashMap",
        "std/collections/hash_map.rs"
    ));

    // Index file matches
    assert!(import_matches_file(
        "src/components",
        "src/components/index.js"
    ));

    // Non-matches
    assert!(!import_matches_file("completely_different", "src/utils.rs"));
}

#[test]
fn test_path_normalization() {
    let mut builder = ImportGraphBuilder::new().unwrap();

    // Test various import formats
    let normalized1 = builder.normalize_import_path("\"./utils.js\"");
    assert_eq!(normalized1, "utils");

    let normalized2 = builder.normalize_import_path("../lib/helper.ts");
    assert!(normalized2.contains("helper"));

    let normalized3 = builder.normalize_import_path("@/components/Button");
    assert!(normalized3.contains("src/components/Button"));
}

#[test]
fn test_pagerank_calculation() {
    let mut graph = ImportGraph::new();

    // Create a simple graph: A -> B -> C, A -> C
    let idx_a = graph.add_node("A".to_string());
    let idx_b = graph.add_node("B".to_string());
    let idx_c = graph.add_node("C".to_string());

    graph.add_edge(idx_a, idx_b);
    graph.add_edge(idx_b, idx_c);
    graph.add_edge(idx_a, idx_c);

    let scores = graph.get_pagerank_scores();
    assert!(scores.is_ok());

    let scores = scores.unwrap();
    assert_eq!(scores.len(), 3);

    // C should have the highest score (most depended upon)
    assert!(scores[idx_c] > scores[idx_a]);
    assert!(scores[idx_c] > scores[idx_b]);
}

#[test]
fn test_import_extraction() {
    let builder = ImportGraphBuilder::new().unwrap();

    // JavaScript content
    let js_content = r#"
        import { Component } from 'react';
        import utils from './utils.js';
        const fs = require('fs');
    "#;

    let imports = builder.extract_imports(js_content, "test.js");
    assert!(imports.len() >= 2);
    assert!(imports.contains(&"react".to_string()));
    assert!(imports.contains(&"./utils.js".to_string()));

    // Rust content
    let rust_content = r#"
        use std::collections::HashMap;
        use crate::utils::helper;
        mod tests;
    "#;

    let imports = builder.extract_imports(rust_content, "test.rs");
    assert!(imports.len() >= 2);
    assert!(imports.contains(&"std::collections::HashMap".to_string()));
    assert!(imports.contains(&"crate::utils::helper".to_string()));
}

#[test]
fn test_centrality_calculator() {
    let mut graph = ImportGraph::new();

    // Create a star graph with central node
    let center = graph.add_node("center.rs".to_string());
    for i in 1..=5 {
        let node = graph.add_node(format!("node{}.rs", i));
        graph.add_edge(node, center); // All nodes depend on center
    }

    let calculator = CentralityCalculator::default();
    let scores = calculator.calculate_pagerank(&graph);
    assert!(scores.is_ok());

    let scores = scores.unwrap();
    assert_eq!(scores.len(), 6);

    // Center node should have highest PageRank
    assert!(scores[center] > scores[1]); // Higher than any leaf node
}

#[test]
fn test_graph_statistics() {
    let mut graph = ImportGraph::new();

    for i in 0..5 {
        graph.add_node(format!("file{}.rs", i));
    }

    // Add some edges
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 3);
    graph.add_edge(0, 4);

    let stats = graph.stats();
    assert_eq!(stats.node_count, 5);
    assert_eq!(stats.edge_count, 4);
    assert!(stats.avg_out_degree > 0.0);
    assert!(stats.density > 0.0);
    assert!(stats.density < 1.0);
}

#[test]
fn test_import_graph_default() {
    let graph = ImportGraph::default();
    assert_eq!(graph.nodes.len(), 0);
    assert!(graph.dependencies.is_empty());
    assert!(graph.dependents.is_empty());
}

#[test]
fn test_import_graph_add_duplicate_node() {
    let mut graph = ImportGraph::new();

    let idx1 = graph.add_node("file.rs".to_string());
    let idx2 = graph.add_node("file.rs".to_string()); // Same file

    assert_eq!(idx1, idx2);
    assert_eq!(graph.nodes.len(), 1);
}

#[test]
fn test_import_graph_add_edge_self_loop() {
    let mut graph = ImportGraph::new();

    let idx = graph.add_node("file.rs".to_string());
    graph.add_edge(idx, idx); // Self loop - should be ignored

    assert!(graph.dependencies[idx].is_empty());
    assert!(graph.dependents[idx].is_empty());
}

#[test]
fn test_import_graph_add_duplicate_edge() {
    let mut graph = ImportGraph::new();

    let idx1 = graph.add_node("file1.rs".to_string());
    let idx2 = graph.add_node("file2.rs".to_string());

    graph.add_edge(idx1, idx2);
    graph.add_edge(idx1, idx2); // Duplicate edge

    assert_eq!(graph.dependencies[idx1].len(), 1);
    assert_eq!(graph.dependents[idx2].len(), 1);
}

#[test]
fn test_import_graph_get_node_degrees() {
    let mut graph = ImportGraph::new();

    let idx1 = graph.add_node("center.rs".to_string());
    let idx2 = graph.add_node("client1.rs".to_string());
    let idx3 = graph.add_node("client2.rs".to_string());

    graph.add_edge(idx2, idx1);
    graph.add_edge(idx3, idx1);

    let degrees = graph.get_node_degrees("center.rs");
    assert!(degrees.is_some());
    let (in_degree, out_degree) = degrees.unwrap();
    assert_eq!(in_degree, 2); // Two files depend on it
    assert_eq!(out_degree, 0); // Doesn't depend on anything

    // Non-existent file
    let degrees = graph.get_node_degrees("nonexistent.rs");
    assert!(degrees.is_none());
}

#[test]
fn test_import_graph_get_pagerank_score_single() {
    let mut graph = ImportGraph::new();

    let idx = graph.add_node("file.rs".to_string());
    let score = graph.get_pagerank_score("file.rs").unwrap();
    assert!(score > 0.0);

    // Non-existent file should return 0 or first index score
    let score = graph.get_pagerank_score("nonexistent.rs").unwrap();
    assert!(score >= 0.0);
}

#[test]
fn test_centrality_calculator_with_params() {
    let calc = CentralityCalculator::with_params(0.9, 50, 1e-4);
    assert!((calc.damping_factor - 0.9).abs() < 0.001);
    assert_eq!(calc.max_iterations, 50);
    assert!((calc.tolerance - 1e-4).abs() < 1e-5);
}

#[test]
fn test_pagerank_empty_graph() {
    let graph = ImportGraph::new();
    let calc = CentralityCalculator::default();
    let scores = calc.calculate_pagerank(&graph).unwrap();
    assert!(scores.is_empty());
}

#[test]
fn test_graph_stats_empty() {
    let graph = ImportGraph::new();
    let stats = graph.stats();

    assert_eq!(stats.node_count, 0);
    assert_eq!(stats.edge_count, 0);
    assert_eq!(stats.avg_in_degree, 0.0);
    assert_eq!(stats.avg_out_degree, 0.0);
    assert_eq!(stats.max_in_degree, 0);
    assert_eq!(stats.max_out_degree, 0);
    assert_eq!(stats.density, 0.0);
}

#[test]
fn test_graph_stats_single_node() {
    let mut graph = ImportGraph::new();
    graph.add_node("single.rs".to_string());

    let stats = graph.stats();
    assert_eq!(stats.node_count, 1);
    assert_eq!(stats.edge_count, 0);
    assert_eq!(stats.density, 0.0); // Can't compute density with 1 node
}

#[test]
fn test_graph_stats_clone() {
    let mut graph = ImportGraph::new();
    graph.add_node("file.rs".to_string());
    let stats = graph.stats();
    let cloned = stats.clone();
    assert_eq!(stats.node_count, cloned.node_count);
}

#[test]
fn test_import_matches_file_index_files() {
    // Python __init__.py
    assert!(import_matches_file("mypackage", "mypackage/__init__.py"));

    // Rust mod.rs
    assert!(import_matches_file("mymodule", "mymodule/mod.rs"));

    // JavaScript index.js
    assert!(import_matches_file("components", "components/index.js"));

    // Main file
    assert!(import_matches_file("app", "app/main.py"));
}

#[test]
fn test_import_matches_file_std_library() {
    // Rust std:: imports
    assert!(import_matches_file(
        "std::io::Read",
        "std/io/read.rs"
    ));

    assert!(import_matches_file(
        "std::fs::File",
        "std/fs/file.rs"
    ));
}

#[test]
fn test_is_index_file() {
    assert!(is_index_file("src/index.js"));
    assert!(is_index_file("package/__init__.py"));
    assert!(is_index_file("module/mod.rs"));
    assert!(is_index_file("app/main.go"));

    assert!(!is_index_file("src/utils.js"));
    assert!(!is_index_file("lib/helper.py"));
}

#[test]
fn test_normalize_for_matching() {
    let result = normalize_for_matching("./src/utils.rs");
    assert!(!result.contains("./"));
    assert!(!result.ends_with(".rs"));

    let result = normalize_for_matching("src/lib.ts");
    assert!(!result.starts_with("src/"));
    assert!(!result.ends_with(".ts"));

    let result = normalize_for_matching("path\\with\\backslash.js");
    assert!(!result.contains('\\'));
}

#[test]
fn test_import_graph_builder_language_detection() {
    let builder = ImportGraphBuilder::new().unwrap();

    // Test various file extensions
    let imports = builder.extract_imports("import os", "test.py");
    assert!(!imports.is_empty());

    let imports = builder.extract_imports("use std::io;", "test.rs");
    assert!(!imports.is_empty());

    // Unknown extension should return empty
    let imports = builder.extract_imports("some content", "test.unknown");
    assert!(imports.is_empty());
}

#[test]
fn test_import_graph_builder_normalize_various_formats() {
    let mut builder = ImportGraphBuilder::new().unwrap();

    // Test normalization with basic paths
    let normalized = builder.normalize_import_path("./utils.js");
    assert!(normalized.contains("utils"));

    // Test with module path
    let normalized = builder.normalize_import_path("react");
    assert!(normalized.contains("react"));

    // @ alias transforms
    let normalized = builder.normalize_import_path("@/store/index");
    assert!(normalized.contains("src/store/index"));

    // ~ alias transforms
    let normalized = builder.normalize_import_path("~/components/Button");
    assert!(normalized.contains("src/components/Button"));
}

#[test]
fn test_import_key_map() {
    let files = vec![
        MockScanResult::new("src/main.rs", vec![]),
        MockScanResult::new("src/lib.rs", vec![]),
        MockScanResult::new("src/utils/helper.rs", vec![]),
    ];

    let key_map = ImportKeyMap::new(&files);

    // Should resolve exact paths
    let result = key_map.resolve_import("src/main.rs");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), 0);

    // Should resolve by suffix
    let result = key_map.resolve_import("main");
    assert!(result.is_some());
}

#[test]
fn test_import_graph_clone() {
    let mut graph = ImportGraph::new();
    graph.add_node("file1.rs".to_string());
    graph.add_node("file2.rs".to_string());
    graph.add_edge(0, 1);

    let cloned = graph.clone();
    assert_eq!(cloned.nodes.len(), 2);
    assert_eq!(cloned.dependencies[0].len(), 1);
}

#[test]
fn test_mock_scan_result_impl() {
    let mock = MockScanResult::new("src/test.rs", vec!["import1", "import2"]);

    assert_eq!(mock.path(), "src/test.rs");
    assert_eq!(mock.relative_path(), "src/test.rs");
    assert_eq!(mock.depth(), 1);
    assert!(!mock.is_docs());
    assert!(!mock.is_readme());
    assert!(!mock.is_test());
    assert!(!mock.is_entrypoint());
    assert!(!mock.has_examples());
    assert_eq!(mock.priority_boost(), 0.0);
    assert_eq!(mock.churn_score(), 0.0);
    assert_eq!(mock.centrality_in(), 0.0);
    assert!(mock.imports().is_some());
    assert_eq!(mock.imports().unwrap().len(), 2);
    assert!(mock.doc_analysis().is_none());
}

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

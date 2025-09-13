//! # Import Graph Analysis and Centrality Calculation
//!
//! Implements dependency graph construction and PageRank centrality calculation
//! for the heuristic scoring system. This module provides:
//!
//! - AST-based import statement parsing and normalization
//! - Dependency graph construction with intelligent matching
//! - PageRank centrality computation
//! - Import-to-file path matching heuristics
//!
//! The centrality scores help identify important files that are heavily depended upon.
//! Uses tree-sitter AST parsing instead of regex patterns for accurate import extraction.

use std::collections::HashMap;
use std::path::Path;
use scribe_core::Result;
use super::ScanResult;
use crate::ast_import_parser::{SimpleAstParser, ImportLanguage, SimpleImport};

/// Dependency graph for centrality calculation
#[derive(Debug, Clone)]
pub struct ImportGraph {
    /// Node index to file path mapping
    nodes: Vec<String>,
    /// File path to node index mapping
    path_to_index: HashMap<String, usize>,
    /// Adjacency list: node -> list of nodes it depends on
    dependencies: Vec<Vec<usize>>,
    /// Reverse adjacency list: node -> list of nodes that depend on it
    dependents: Vec<Vec<usize>>,
    /// Cached PageRank scores
    pagerank_scores: Option<Vec<f64>>,
}

/// Builder for constructing import graphs from scan results
#[derive(Debug)]
pub struct ImportGraphBuilder {
    /// Import path normalization cache
    normalization_cache: HashMap<String, String>,
    /// AST parser for extracting imports
    ast_parser: SimpleAstParser,
}


/// Centrality calculator implementing PageRank algorithm
#[derive(Debug)]
pub struct CentralityCalculator {
    /// PageRank damping factor (typically 0.85)
    damping_factor: f64,
    /// Maximum iterations for convergence
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl Default for ImportGraphBuilder {
    fn default() -> Self {
        Self::new().expect("Failed to create ImportGraphBuilder")
    }
}

impl ImportGraphBuilder {
    /// Create a new import graph builder
    pub fn new() -> Result<Self> {
        Ok(Self {
            normalization_cache: HashMap::new(),
            ast_parser: SimpleAstParser::new()?,
        })
    }
    
    /// Build import graph from scan results
    pub fn build_graph<T>(&mut self, files: &[T]) -> Result<ImportGraph> 
    where 
        T: ScanResult,
    {
        let mut graph = ImportGraph::new();
        
        // Add all files as nodes
        for file in files {
            graph.add_node(file.path().to_string());
        }
        
        // Build edges based on import relationships
        for (file_idx, file) in files.iter().enumerate() {
            if let Some(imports) = file.imports() {
                for import_path in imports {
                    if let Some(target_idx) = self.find_matching_file(import_path, files) {
                        graph.add_edge(file_idx, target_idx);
                    }
                }
            }
        }
        
        Ok(graph)
    }
    
    /// Find which file an import statement refers to
    fn find_matching_file<T>(&mut self, import_path: &str, files: &[T]) -> Option<usize>
    where 
        T: ScanResult,
    {
        let normalized_import = self.normalize_import_path(import_path);
        
        for (idx, file) in files.iter().enumerate() {
            if import_matches_file(&normalized_import, file.path()) {
                return Some(idx);
            }
        }
        
        None
    }
    
    /// Normalize import path for matching
    fn normalize_import_path(&mut self, import_path: &str) -> String {
        // Check cache first
        if let Some(cached) = self.normalization_cache.get(import_path) {
            return cached.clone();
        }
        
        let normalized = self.perform_normalization(import_path);
        self.normalization_cache.insert(import_path.to_string(), normalized.clone());
        normalized
    }
    
    /// Perform actual import path normalization
    fn perform_normalization(&self, import_path: &str) -> String {
        let mut normalized = import_path.to_string();
        
        // Remove quotes and whitespace
        normalized = normalized.trim_matches(|c| c == '"' || c == '\'' || c == ' ').to_string();
        
        // Normalize path separators
        normalized = normalized.replace('\\', "/");
        
        // Remove file extensions for language-agnostic matching
        if let Some(dot_pos) = normalized.rfind('.') {
            // Keep extension only for specific cases
            let extension = &normalized[dot_pos..];
            match extension {
                ".json" | ".xml" | ".yaml" | ".toml" | ".css" | ".scss" => {
                    // Keep these extensions as they're often imported directly
                }
                _ => {
                    // Remove programming language extensions
                    normalized.truncate(dot_pos);
                }
            }
        }
        
        // Handle relative imports
        if normalized.starts_with("./") || normalized.starts_with("../") {
            // Keep relative structure but normalize
            normalized = self.normalize_relative_path(&normalized);
        }
        
        // Convert module-style imports to file paths
        normalized = normalized.replace("::", "/").replace('.', "/");
        
        // Handle common import aliases
        normalized = self.resolve_import_aliases(&normalized);
        
        normalized
    }
    
    /// Normalize relative path imports
    fn normalize_relative_path(&self, path: &str) -> String {
        // Simple path normalization - in practice, this would be more sophisticated
        let components: Vec<&str> = path.split('/').collect();
        let mut normalized = Vec::new();
        
        for component in components {
            match component {
                "" | "." => continue,
                ".." => {
                    normalized.pop();
                }
                _ => normalized.push(component),
            }
        }
        
        normalized.join("/")
    }
    
    /// Resolve common import aliases and patterns
    fn resolve_import_aliases(&self, import_path: &str) -> String {
        // Handle common aliases
        match import_path {
            // Node.js built-ins
            "fs" | "path" | "http" | "https" | "url" | "crypto" | "os" => {
                return import_path.to_string(); // Keep as-is for built-ins
            }
            // Package imports (usually not files in the repo)
            path if !path.contains('/') && !path.starts_with('.') => {
                return import_path.to_string(); // Likely external package
            }
            _ => {}
        }
        
        let mut resolved = import_path.to_string();
        
        // Common alias patterns
        if resolved.starts_with("@/") {
            resolved = resolved.replacen("@/", "src/", 1);
        }
        
        if resolved.starts_with("~") {
            resolved = resolved.replacen("~", "src", 1);
        }
        
        resolved
    }
    
    /// Detect language from file extension for AST parsing
    fn detect_language_from_path(&self, file_path: &str) -> Option<ImportLanguage> {
        let extension = Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        ImportLanguage::from_extension(extension)
    }
    
    /// Extract imports from file content using AST parsing
    pub fn extract_imports(&self, content: &str, file_path: &str) -> Vec<String> {
        // Detect language from file path
        let language = match self.detect_language_from_path(file_path) {
            Some(lang) => lang,
            None => return Vec::new(),
        };
        
        // Use AST parser to extract imports
        match self.ast_parser.extract_imports(content, language) {
            Ok(simple_imports) => {
                simple_imports.into_iter()
                    .map(|import| import.module)
                    .collect()
            }
            Err(_) => {
                // Fall back to empty list if parsing fails
                Vec::new()
            }
        }
    }
}

impl ImportGraph {
    /// Create a new empty import graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            path_to_index: HashMap::new(),
            dependencies: Vec::new(),
            dependents: Vec::new(),
            pagerank_scores: None,
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, path: String) -> usize {
        if let Some(&existing_idx) = self.path_to_index.get(&path) {
            return existing_idx;
        }
        
        let index = self.nodes.len();
        self.nodes.push(path.clone());
        self.path_to_index.insert(path, index);
        self.dependencies.push(Vec::new());
        self.dependents.push(Vec::new());
        
        // Invalidate cached PageRank scores
        self.pagerank_scores = None;
        
        index
    }
    
    /// Add an edge from source to target (source depends on target)
    pub fn add_edge(&mut self, source: usize, target: usize) {
        if source < self.dependencies.len() && target < self.dependents.len() && source != target {
            // Avoid duplicate edges
            if !self.dependencies[source].contains(&target) {
                self.dependencies[source].push(target);
                self.dependents[target].push(source);
                
                // Invalidate cached PageRank scores
                self.pagerank_scores = None;
            }
        }
    }
    
    /// Get node degrees (in-degree, out-degree)
    pub fn get_node_degrees(&self, path: &str) -> Option<(usize, usize)> {
        let index = *self.path_to_index.get(path)?;
        let in_degree = self.dependents[index].len();
        let out_degree = self.dependencies[index].len();
        Some((in_degree, out_degree))
    }
    
    /// Get PageRank scores for all nodes
    pub fn get_pagerank_scores(&mut self) -> Result<&[f64]> {
        if self.pagerank_scores.is_none() {
            let calculator = CentralityCalculator::default();
            let scores = calculator.calculate_pagerank(self)?;
            self.pagerank_scores = Some(scores);
        }
        
        Ok(self.pagerank_scores.as_ref().unwrap())
    }
    
    /// Get PageRank score for a specific file
    pub fn get_pagerank_score(&mut self, path: &str) -> Result<f64> {
        let index = *self.path_to_index.get(path).unwrap_or(&0);
        let scores = self.get_pagerank_scores()?;
        Ok(*scores.get(index).unwrap_or(&0.0))
    }
    
    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let node_count = self.nodes.len();
        let edge_count: usize = self.dependencies.iter().map(|deps| deps.len()).sum();
        
        let in_degrees: Vec<usize> = self.dependents.iter().map(|deps| deps.len()).collect();
        let out_degrees: Vec<usize> = self.dependencies.iter().map(|deps| deps.len()).collect();
        
        let avg_in_degree = if node_count > 0 {
            in_degrees.iter().sum::<usize>() as f64 / node_count as f64
        } else {
            0.0
        };
        
        let avg_out_degree = if node_count > 0 {
            out_degrees.iter().sum::<usize>() as f64 / node_count as f64
        } else {
            0.0
        };
        
        let max_in_degree = in_degrees.iter().max().cloned().unwrap_or(0);
        let max_out_degree = out_degrees.iter().max().cloned().unwrap_or(0);
        
        let density = if node_count > 1 {
            edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };
        
        GraphStats {
            node_count,
            edge_count,
            avg_in_degree,
            avg_out_degree,
            max_in_degree,
            max_out_degree,
            density,
        }
    }
}

impl Default for ImportGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CentralityCalculator {
    /// Create a new centrality calculator with default parameters
    pub fn new() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
    
    /// Create with custom parameters
    pub fn with_params(damping_factor: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            damping_factor,
            max_iterations,
            tolerance,
        }
    }
    
    /// Calculate PageRank scores for the graph
    pub fn calculate_pagerank(&self, graph: &ImportGraph) -> Result<Vec<f64>> {
        let n = graph.nodes.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        
        // Initialize scores uniformly
        let mut scores = vec![1.0 / n as f64; n];
        let mut new_scores = vec![0.0; n];
        
        for _iteration in 0..self.max_iterations {
            // Calculate new scores
            for (i, new_score) in new_scores.iter_mut().enumerate() {
                let mut rank_sum = 0.0;
                
                // Sum contributions from nodes pointing to this one
                for &source in &graph.dependents[i] {
                    let out_degree = graph.dependencies[source].len();
                    if out_degree > 0 {
                        rank_sum += scores[source] / out_degree as f64;
                    }
                }
                
                // Apply PageRank formula
                *new_score = (1.0 - self.damping_factor) / n as f64 + self.damping_factor * rank_sum;
            }
            
            // Check for convergence
            let mut max_diff = 0.0;
            for i in 0..n {
                let diff = (new_scores[i] - scores[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            
            // Swap score vectors
            std::mem::swap(&mut scores, &mut new_scores);
            
            if max_diff < self.tolerance {
                break;
            }
        }
        
        Ok(scores)
    }
}

impl Default for CentralityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics for analysis
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_in_degree: f64,
    pub avg_out_degree: f64,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
    pub density: f64,
}

/// Check if an import statement matches a file path
pub fn import_matches_file(import_path: &str, file_path: &str) -> bool {
    // Normalize both paths for comparison
    let normalized_import = normalize_for_matching(import_path);
    let normalized_file = normalize_for_matching(file_path);
    
    // Direct match
    if normalized_import == normalized_file {
        return true;
    }
    
    // Check if import is a substring of file path (for module imports)
    if normalized_file.contains(&normalized_import) {
        return true;
    }
    
    // Check if file path ends with import (common for relative imports)
    if normalized_file.ends_with(&normalized_import) {
        return true;
    }
    
    // Check module-style imports (convert :: to /)
    let module_import = normalized_import.replace("::", "/");
    if normalized_file.contains(&module_import) || 
       module_import.contains(&normalized_file) {
        return true;
    }
    
    // Special handling for std library and common patterns
    if normalized_import.starts_with("std::") {
        let std_path = normalized_import.replace("::", "/");
        if normalized_file.replace("_", "").contains(&std_path.replace("_", "")) {
            return true;
        }
    }
    
    // Handle index files
    if is_index_file(file_path) {
        let directory = std::path::Path::new(file_path)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or("");
        let normalized_dir = normalize_for_matching(directory);
        if normalized_import == normalized_dir {
            return true;
        }
    }
    
    false
}

/// Normalize paths for matching comparison
fn normalize_for_matching(path: &str) -> String {
    let mut normalized = path.to_lowercase();
    
    // Remove common prefixes
    if normalized.starts_with("./") {
        normalized = normalized[2..].to_string();
    }
    
    if normalized.starts_with("src/") {
        normalized = normalized[4..].to_string();
    }
    
    // Remove file extensions
    if let Some(dot_pos) = normalized.rfind('.') {
        normalized.truncate(dot_pos);
    }
    
    // Normalize separators
    normalized = normalized.replace('\\', "/");
    
    // Remove trailing slashes
    normalized.trim_end_matches('/').to_string()
}

/// Check if a file is an index file (index.js, __init__.py, mod.rs, etc.)
fn is_index_file(file_path: &str) -> bool {
    let file_name = std::path::Path::new(file_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("");
    
    matches!(file_name, "index" | "__init__" | "mod" | "main")
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::DocumentAnalysis;
    
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
        fn path(&self) -> &str { &self.path }
        fn relative_path(&self) -> &str { &self.relative_path }
        fn depth(&self) -> usize { self.depth }
        fn is_docs(&self) -> bool { false }
        fn is_readme(&self) -> bool { false }
        fn is_test(&self) -> bool { false }
        fn is_entrypoint(&self) -> bool { false }
        fn has_examples(&self) -> bool { false }
        fn priority_boost(&self) -> f64 { 0.0 }
        fn churn_score(&self) -> f64 { 0.0 }
        fn centrality_in(&self) -> f64 { 0.0 }
        fn imports(&self) -> Option<&[String]> { self.imports.as_deref() }
        fn doc_analysis(&self) -> Option<&DocumentAnalysis> { None }
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
        assert!(import_matches_file("std::collections::HashMap", "std/collections/hash_map.rs"));
        
        // Index file matches
        assert!(import_matches_file("src/components", "src/components/index.js"));
        
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
}
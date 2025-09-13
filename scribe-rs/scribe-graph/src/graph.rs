//! # Graph Data Structures for PageRank Centrality
//!
//! Efficient graph representation optimized for dependency analysis and PageRank computation.
//! This module implements the core graph structures used in the PageRank centrality algorithm
//! with emphasis on reverse edges for code dependency analysis.
//!
//! ## Design Philosophy
//! - **Forward edges**: A imports B (A -> B)  
//! - **Reverse edges**: B is imported by A (B <- A)
//! - **PageRank flows along reverse edges** (importance flows to imported files)
//! - **Memory-efficient adjacency list representation** for large graphs (10k+ nodes)
//! - **Fast degree queries** and statistics calculation

use std::collections::{HashMap, HashSet, BTreeSet};
use serde::{Deserialize, Serialize};
use indexmap::IndexMap;
use dashmap::DashMap;
use parking_lot::RwLock;
use scribe_core::{Result, error::ScribeError};

/// Node identifier type for the dependency graph
pub type NodeId = String;

/// Edge weight type (unused in unweighted PageRank, but reserved for extensions)
pub type EdgeWeight = f64;

/// Efficient dependency graph representation optimized for PageRank computation
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Forward adjacency list: file -> files it imports
    forward_edges: IndexMap<NodeId, BTreeSet<NodeId>>,
    
    /// Reverse adjacency list: file -> files that import it (for PageRank)
    reverse_edges: IndexMap<NodeId, BTreeSet<NodeId>>,
    
    /// All nodes in the graph (includes isolated nodes)
    nodes: BTreeSet<NodeId>,
    
    /// Node metadata cache
    node_cache: HashMap<NodeId, NodeMetadata>,
    
    /// Graph statistics cache (invalidated on mutations)
    stats_cache: Option<GraphStatistics>,
}

/// Metadata associated with each node in the graph
#[derive(Debug, Clone, PartialEq)]
pub struct NodeMetadata {
    /// File path of the node
    pub file_path: String,
    /// Programming language detected
    pub language: Option<String>,
    /// Whether this is an entrypoint file
    pub is_entrypoint: bool,
    /// Whether this is a test file
    pub is_test: bool,
    /// File size in bytes (for statistics)
    pub size_bytes: u64,
}

impl NodeMetadata {
    /// Create new node metadata
    pub fn new(file_path: String) -> Self {
        let language = detect_language_from_extension(&file_path);
        let is_entrypoint = is_entrypoint_file(&file_path);
        let is_test = is_test_file(&file_path);
        
        Self {
            file_path,
            language,
            is_entrypoint,
            is_test,
            size_bytes: 0,
        }
    }
    
    /// Create with size information
    pub fn with_size(mut self, size_bytes: u64) -> Self {
        self.size_bytes = size_bytes;
        self
    }
}

/// Graph construction and manipulation operations
impl DependencyGraph {
    /// Create a new empty dependency graph
    pub fn new() -> Self {
        Self {
            forward_edges: IndexMap::new(),
            reverse_edges: IndexMap::new(),
            nodes: BTreeSet::new(),
            node_cache: HashMap::new(),
            stats_cache: None,
        }
    }
    
    /// Create with initial capacity hint for performance optimization
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            forward_edges: IndexMap::with_capacity(capacity),
            reverse_edges: IndexMap::with_capacity(capacity),
            nodes: BTreeSet::new(),
            node_cache: HashMap::with_capacity(capacity),
            stats_cache: None,
        }
    }
    
    /// Add a node to the graph (can exist without edges)
    pub fn add_node(&mut self, node_id: NodeId) -> Result<()> {
        self.nodes.insert(node_id.clone());
        
        // Initialize empty adjacency lists if not present
        self.forward_edges.entry(node_id.clone()).or_insert_with(BTreeSet::new);
        self.reverse_edges.entry(node_id.clone()).or_insert_with(BTreeSet::new);
        
        // Add default metadata if not present
        if !self.node_cache.contains_key(&node_id) {
            self.node_cache.insert(node_id.clone(), NodeMetadata::new(node_id));
        }
        
        // Invalidate cache
        self.stats_cache = None;
        
        Ok(())
    }
    
    /// Add a node with metadata
    pub fn add_node_with_metadata(&mut self, node_id: NodeId, metadata: NodeMetadata) -> Result<()> {
        self.add_node(node_id.clone())?;
        self.node_cache.insert(node_id, metadata);
        Ok(())
    }
    
    /// Add an import edge: from_file imports to_file
    pub fn add_edge(&mut self, from_node: NodeId, to_node: NodeId) -> Result<()> {
        // Ensure both nodes exist
        self.add_node(from_node.clone())?;
        self.add_node(to_node.clone())?;
        
        // Add forward edge: from_node -> to_node
        if let Some(forward_set) = self.forward_edges.get_mut(&from_node) {
            forward_set.insert(to_node.clone());
        }
        
        // Add reverse edge: to_node <- from_node
        if let Some(reverse_set) = self.reverse_edges.get_mut(&to_node) {
            reverse_set.insert(from_node);
        }
        
        // Invalidate cache
        self.stats_cache = None;
        
        Ok(())
    }
    
    /// Add multiple edges efficiently (batch operation)
    pub fn add_edges(&mut self, edges: &[(NodeId, NodeId)]) -> Result<()> {
        for (from_node, to_node) in edges {
            self.add_edge(from_node.clone(), to_node.clone())?;
        }
        Ok(())
    }
    
    /// Remove a node and all its edges
    pub fn remove_node(&mut self, node_id: &NodeId) -> Result<bool> {
        if !self.nodes.contains(node_id) {
            return Ok(false);
        }
        
        // Remove from nodes set
        self.nodes.remove(node_id);
        
        // Get outgoing edges to clean up reverse references
        if let Some(outgoing) = self.forward_edges.shift_remove(node_id) {
            for target in &outgoing {
                if let Some(reverse_set) = self.reverse_edges.get_mut(target) {
                    reverse_set.remove(node_id);
                }
            }
        }
        
        // Get incoming edges to clean up forward references  
        if let Some(incoming) = self.reverse_edges.shift_remove(node_id) {
            for source in &incoming {
                if let Some(forward_set) = self.forward_edges.get_mut(source) {
                    forward_set.remove(node_id);
                }
            }
        }
        
        // Remove metadata
        self.node_cache.remove(node_id);
        
        // Invalidate cache
        self.stats_cache = None;
        
        Ok(true)
    }
    
    /// Remove an edge between two nodes
    pub fn remove_edge(&mut self, from_node: &NodeId, to_node: &NodeId) -> Result<bool> {
        let forward_removed = if let Some(forward_set) = self.forward_edges.get_mut(from_node) {
            forward_set.remove(to_node)
        } else {
            false
        };
        
        let reverse_removed = if let Some(reverse_set) = self.reverse_edges.get_mut(to_node) {
            reverse_set.remove(from_node)
        } else {
            false
        };
        
        if forward_removed || reverse_removed {
            self.stats_cache = None;
        }
        
        Ok(forward_removed || reverse_removed)
    }
    
    /// Check if a node exists in the graph
    pub fn contains_node(&self, node_id: &NodeId) -> bool {
        self.nodes.contains(node_id)
    }
    
    /// Check if an edge exists between two nodes
    pub fn contains_edge(&self, from_node: &NodeId, to_node: &NodeId) -> bool {
        self.forward_edges
            .get(from_node)
            .map_or(false, |edges| edges.contains(to_node))
    }
    
    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get the total number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.forward_edges.values().map(|edges| edges.len()).sum()
    }
    
    /// Get all nodes in the graph
    pub fn nodes(&self) -> impl Iterator<Item = &NodeId> {
        self.nodes.iter()
    }
    
    /// Get all edges in the graph as (from, to) pairs
    pub fn edges(&self) -> impl Iterator<Item = (&NodeId, &NodeId)> {
        self.forward_edges.iter()
            .flat_map(|(from, targets)| targets.iter().map(move |to| (from, to)))
    }
}

/// Degree and neighbor query operations
impl DependencyGraph {
    /// Get in-degree of a node (number of files that import this node)
    pub fn in_degree(&self, node_id: &NodeId) -> usize {
        self.reverse_edges.get(node_id).map_or(0, |edges| edges.len())
    }
    
    /// Get out-degree of a node (number of files this node imports)
    pub fn out_degree(&self, node_id: &NodeId) -> usize {
        self.forward_edges.get(node_id).map_or(0, |edges| edges.len())
    }
    
    /// Get total degree of a node (in + out)
    pub fn degree(&self, node_id: &NodeId) -> usize {
        self.in_degree(node_id) + self.out_degree(node_id)
    }
    
    /// Get nodes that this node imports (outgoing edges)
    pub fn outgoing_neighbors(&self, node_id: &NodeId) -> Option<&BTreeSet<NodeId>> {
        self.forward_edges.get(node_id)
    }
    
    /// Get nodes that import this node (incoming edges) - important for PageRank
    pub fn incoming_neighbors(&self, node_id: &NodeId) -> Option<&BTreeSet<NodeId>> {
        self.reverse_edges.get(node_id)
    }
    
    /// Get both incoming and outgoing neighbors
    pub fn all_neighbors(&self, node_id: &NodeId) -> HashSet<&NodeId> {
        let mut neighbors = HashSet::new();
        
        if let Some(outgoing) = self.forward_edges.get(node_id) {
            neighbors.extend(outgoing.iter());
        }
        
        if let Some(incoming) = self.reverse_edges.get(node_id) {
            neighbors.extend(incoming.iter());
        }
        
        neighbors
    }
    
    /// Get degree information for a node
    pub fn get_degree_info(&self, node_id: &NodeId) -> Option<DegreeInfo> {
        if !self.contains_node(node_id) {
            return None;
        }
        
        Some(DegreeInfo {
            node_id: node_id.clone(),
            in_degree: self.in_degree(node_id),
            out_degree: self.out_degree(node_id),
            total_degree: self.degree(node_id),
        })
    }
}

/// Node metadata and information queries
impl DependencyGraph {
    /// Get metadata for a node
    pub fn node_metadata(&self, node_id: &NodeId) -> Option<&NodeMetadata> {
        self.node_cache.get(node_id)
    }
    
    /// Set metadata for a node
    pub fn set_node_metadata(&mut self, node_id: NodeId, metadata: NodeMetadata) -> Result<()> {
        if !self.contains_node(&node_id) {
            return Err(ScribeError::invalid_operation(
                format!("Node {} does not exist in graph", node_id),
                "set_node_metadata"
            ));
        }
        
        self.node_cache.insert(node_id, metadata);
        Ok(())
    }
    
    /// Get all entrypoint nodes
    pub fn entrypoint_nodes(&self) -> Vec<&NodeId> {
        self.node_cache.iter()
            .filter_map(|(id, meta)| if meta.is_entrypoint { Some(id) } else { None })
            .collect()
    }
    
    /// Get all test nodes
    pub fn test_nodes(&self) -> Vec<&NodeId> {
        self.node_cache.iter()
            .filter_map(|(id, meta)| if meta.is_test { Some(id) } else { None })
            .collect()
    }
    
    /// Get nodes by language
    pub fn nodes_by_language(&self, language: &str) -> Vec<&NodeId> {
        self.node_cache.iter()
            .filter_map(|(id, meta)| {
                if meta.language.as_deref() == Some(language) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Specialized operations for PageRank computation
impl DependencyGraph {
    /// Get all nodes with their reverse edge neighbors (for PageRank iteration)
    pub fn pagerank_iterator(&self) -> impl Iterator<Item = (&NodeId, Option<&BTreeSet<NodeId>>)> {
        self.nodes.iter().map(move |node| {
            (node, self.reverse_edges.get(node))
        })
    }
    
    /// Get dangling nodes (nodes with no outgoing edges)
    pub fn dangling_nodes(&self) -> Vec<&NodeId> {
        self.nodes.iter()
            .filter(|&node| self.out_degree(node) == 0)
            .collect()
    }
    
    /// Get strongly connected components (simplified estimation for statistics)
    pub fn estimate_scc_count(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        
        // Count nodes with both in and out edges (likely in cycles)
        let potential_scc_nodes = self.nodes.iter()
            .filter(|&node| self.in_degree(node) > 0 && self.out_degree(node) > 0)
            .count();
        
        // Rough estimate: most SCCs are small, assume average size of 3
        let estimated_scc = if potential_scc_nodes > 0 {
            std::cmp::max(1, potential_scc_nodes / 3)
        } else {
            0
        };
        
        // Add isolated nodes and simple chains
        let isolated_nodes = self.nodes.len() - potential_scc_nodes;
        estimated_scc + isolated_nodes
    }
    
    /// Check if the graph is strongly connected (simplified check)
    pub fn is_strongly_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }
        
        // Simplified check: all nodes have both in and out edges
        self.nodes.iter().all(|node| {
            self.in_degree(node) > 0 && self.out_degree(node) > 0
        })
    }
}

/// Concurrent graph operations for performance
impl DependencyGraph {
    /// Create a thread-safe concurrent graph for parallel operations
    pub fn into_concurrent(self) -> ConcurrentDependencyGraph {
        ConcurrentDependencyGraph {
            forward_edges: DashMap::from_iter(self.forward_edges),
            reverse_edges: DashMap::from_iter(self.reverse_edges),
            nodes: RwLock::new(self.nodes),
            node_cache: DashMap::from_iter(self.node_cache),
            stats_cache: RwLock::new(self.stats_cache),
        }
    }
}

/// Thread-safe concurrent version of DependencyGraph
#[derive(Debug)]
pub struct ConcurrentDependencyGraph {
    forward_edges: DashMap<NodeId, BTreeSet<NodeId>>,
    reverse_edges: DashMap<NodeId, BTreeSet<NodeId>>,
    nodes: RwLock<BTreeSet<NodeId>>,
    node_cache: DashMap<NodeId, NodeMetadata>,
    stats_cache: RwLock<Option<GraphStatistics>>,
}

impl ConcurrentDependencyGraph {
    /// Add a node concurrently
    pub fn add_node(&self, node_id: NodeId) -> Result<()> {
        {
            let mut nodes = self.nodes.write();
            nodes.insert(node_id.clone());
        }
        
        self.forward_edges.entry(node_id.clone()).or_insert_with(BTreeSet::new);
        self.reverse_edges.entry(node_id.clone()).or_insert_with(BTreeSet::new);
        
        if !self.node_cache.contains_key(&node_id) {
            self.node_cache.insert(node_id.clone(), NodeMetadata::new(node_id));
        }
        
        // Invalidate stats cache
        *self.stats_cache.write() = None;
        
        Ok(())
    }
    
    /// Get in-degree concurrently
    pub fn in_degree(&self, node_id: &NodeId) -> usize {
        self.reverse_edges.get(node_id).map_or(0, |entry| entry.len())
    }
    
    /// Get out-degree concurrently  
    pub fn out_degree(&self, node_id: &NodeId) -> usize {
        self.forward_edges.get(node_id).map_or(0, |entry| entry.len())
    }
    
    /// Convert back to single-threaded graph
    pub fn into_sequential(self) -> DependencyGraph {
        let nodes = self.nodes.into_inner();
        let stats_cache = self.stats_cache.into_inner();
        
        DependencyGraph {
            forward_edges: self.forward_edges.into_iter().collect(),
            reverse_edges: self.reverse_edges.into_iter().collect(),
            nodes,
            node_cache: self.node_cache.into_iter().collect(),
            stats_cache,
        }
    }
}

/// Degree information for a node
#[derive(Debug, Clone, PartialEq)]
pub struct DegreeInfo {
    pub node_id: NodeId,
    pub in_degree: usize,
    pub out_degree: usize,
    pub total_degree: usize,
}

/// Graph statistics computed lazily and cached
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Total number of edges
    pub total_edges: usize,
    /// Average in-degree
    pub in_degree_avg: f64,
    /// Maximum in-degree
    pub in_degree_max: usize,
    /// Average out-degree
    pub out_degree_avg: f64,
    /// Maximum out-degree
    pub out_degree_max: usize,
    /// Estimated number of strongly connected components
    pub strongly_connected_components: usize,
    /// Graph density (actual_edges / possible_edges)
    pub graph_density: f64,
    /// Number of isolated nodes (no edges)
    pub isolated_nodes: usize,
    /// Number of dangling nodes (no outgoing edges)
    pub dangling_nodes: usize,
}

impl GraphStatistics {
    /// Create empty statistics
    pub fn empty() -> Self {
        Self {
            total_nodes: 0,
            total_edges: 0,
            in_degree_avg: 0.0,
            in_degree_max: 0,
            out_degree_avg: 0.0,
            out_degree_max: 0,
            strongly_connected_components: 0,
            graph_density: 0.0,
            isolated_nodes: 0,
            dangling_nodes: 0,
        }
    }
}

/// Utility functions for node classification
fn detect_language_from_extension(file_path: &str) -> Option<String> {
    let ext = std::path::Path::new(file_path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())?;
    
    match ext.as_str() {
        "py" => Some("python".to_string()),
        "js" | "jsx" | "mjs" => Some("javascript".to_string()),
        "ts" | "tsx" => Some("typescript".to_string()),
        "rs" => Some("rust".to_string()),
        "go" => Some("go".to_string()),
        "java" | "kt" => Some("java".to_string()),
        "cpp" | "cc" | "cxx" | "hpp" | "h" => Some("cpp".to_string()),
        "c" => Some("c".to_string()),
        "rb" => Some("ruby".to_string()),
        "php" => Some("php".to_string()),
        "cs" => Some("csharp".to_string()),
        "swift" => Some("swift".to_string()),
        _ => None,
    }
}

fn is_entrypoint_file(file_path: &str) -> bool {
    let file_name = std::path::Path::new(file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    matches!(file_name.as_str(), "main.py" | "main.rs" | "main.go" | "main.js" | "main.ts" | 
                                 "index.py" | "index.rs" | "index.go" | "index.js" | "index.ts" |
                                 "app.py" | "app.rs" | "app.go" | "app.js" | "app.ts" |
                                 "server.py" | "server.rs" | "server.go" | "server.js" | "server.ts" |
                                 "lib.rs" | "__init__.py")
}

fn is_test_file(file_path: &str) -> bool {
    let path_lower = file_path.to_lowercase();
    path_lower.contains("test") || 
    path_lower.contains("spec") ||
    path_lower.contains("__tests__") ||
    path_lower.ends_with("_test.py") ||
    path_lower.ends_with("_test.rs") ||
    path_lower.ends_with("_test.go") ||
    path_lower.ends_with(".test.js") ||
    path_lower.ends_with(".test.ts") ||
    path_lower.ends_with("_spec.rb")
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_graph_creation() {
        let graph = DependencyGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
    
    #[test]
    fn test_node_operations() {
        let mut graph = DependencyGraph::new();
        
        // Add nodes
        graph.add_node("main.py".to_string()).unwrap();
        graph.add_node("utils.py".to_string()).unwrap();
        
        assert_eq!(graph.node_count(), 2);
        assert!(graph.contains_node(&"main.py".to_string()));
        assert!(graph.contains_node(&"utils.py".to_string()));
        
        // Remove node
        let removed = graph.remove_node(&"utils.py".to_string()).unwrap();
        assert!(removed);
        assert_eq!(graph.node_count(), 1);
        assert!(!graph.contains_node(&"utils.py".to_string()));
    }
    
    #[test]
    fn test_edge_operations() {
        let mut graph = DependencyGraph::new();
        
        // Add edge (automatically creates nodes)
        graph.add_edge("main.py".to_string(), "utils.py".to_string()).unwrap();
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.contains_edge(&"main.py".to_string(), &"utils.py".to_string()));
        
        // Check degrees
        assert_eq!(graph.out_degree(&"main.py".to_string()), 1);
        assert_eq!(graph.in_degree(&"utils.py".to_string()), 1);
        assert_eq!(graph.in_degree(&"main.py".to_string()), 0);
        assert_eq!(graph.out_degree(&"utils.py".to_string()), 0);
    }
    
    #[test]
    fn test_multiple_edges() {
        let mut graph = DependencyGraph::new();
        
        let edges = vec![
            ("main.py".to_string(), "utils.py".to_string()),
            ("main.py".to_string(), "config.py".to_string()),
            ("utils.py".to_string(), "config.py".to_string()),
        ];
        
        graph.add_edges(&edges).unwrap();
        
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        
        // main.py should have out-degree 2
        assert_eq!(graph.out_degree(&"main.py".to_string()), 2);
        
        // config.py should have in-degree 2
        assert_eq!(graph.in_degree(&"config.py".to_string()), 2);
    }
    
    #[test]
    fn test_node_metadata() {
        let mut graph = DependencyGraph::new();
        
        let metadata = NodeMetadata::new("main.py".to_string()).with_size(1024);
        graph.add_node_with_metadata("main.py".to_string(), metadata).unwrap();
        
        let retrieved = graph.node_metadata(&"main.py".to_string()).unwrap();
        assert_eq!(retrieved.file_path, "main.py");
        assert_eq!(retrieved.language, Some("python".to_string()));
        assert!(retrieved.is_entrypoint);
        assert!(!retrieved.is_test);
        assert_eq!(retrieved.size_bytes, 1024);
    }
    
    #[test]
    fn test_language_detection() {
        assert_eq!(detect_language_from_extension("main.py"), Some("python".to_string()));
        assert_eq!(detect_language_from_extension("app.js"), Some("javascript".to_string()));
        assert_eq!(detect_language_from_extension("lib.rs"), Some("rust".to_string()));
        assert_eq!(detect_language_from_extension("server.go"), Some("go".to_string()));
        assert_eq!(detect_language_from_extension("component.tsx"), Some("typescript".to_string()));
        assert_eq!(detect_language_from_extension("unknown.xyz"), None);
    }
    
    #[test]
    fn test_file_classification() {
        assert!(is_entrypoint_file("main.py"));
        assert!(is_entrypoint_file("index.js"));
        assert!(is_entrypoint_file("lib.rs"));
        assert!(!is_entrypoint_file("utils.py"));
        
        assert!(is_test_file("test_utils.py"));
        assert!(is_test_file("utils.test.js"));
        assert!(is_test_file("integration_test.rs"));
        assert!(!is_test_file("utils.py"));
    }
    
    #[test]
    fn test_pagerank_iterator() {
        let mut graph = DependencyGraph::new();
        
        // Build a small graph: A -> B -> C, C -> A (creates cycle)
        graph.add_edge("A".to_string(), "B".to_string()).unwrap();
        graph.add_edge("B".to_string(), "C".to_string()).unwrap();
        graph.add_edge("C".to_string(), "A".to_string()).unwrap();
        
        let pagerank_data: Vec<_> = graph.pagerank_iterator().collect();
        assert_eq!(pagerank_data.len(), 3);
        
        // Each node should have incoming edges (reverse edges)
        for (node, reverse_edges) in pagerank_data {
            assert!(reverse_edges.is_some());
            assert!(!reverse_edges.unwrap().is_empty());
        }
    }
    
    #[test]
    fn test_dangling_nodes() {
        let mut graph = DependencyGraph::new();
        
        // A -> B, C is isolated, D has no outgoing edges
        graph.add_edge("A".to_string(), "B".to_string()).unwrap();
        graph.add_node("C".to_string()).unwrap();
        graph.add_edge("B".to_string(), "D".to_string()).unwrap();
        
        let dangling = graph.dangling_nodes();
        
        // C and D should be dangling (no outgoing edges)
        assert_eq!(dangling.len(), 2);
        assert!(dangling.contains(&&"C".to_string()));
        assert!(dangling.contains(&&"D".to_string()));
    }
    
    #[test]
    fn test_concurrent_graph() {
        let mut graph = DependencyGraph::new();
        graph.add_edge("A".to_string(), "B".to_string()).unwrap();
        graph.add_edge("B".to_string(), "C".to_string()).unwrap();
        
        let concurrent = graph.into_concurrent();
        
        // Test concurrent operations
        assert_eq!(concurrent.in_degree(&"B".to_string()), 1);
        assert_eq!(concurrent.out_degree(&"B".to_string()), 1);
        
        // Add node concurrently
        concurrent.add_node("D".to_string()).unwrap();
        
        // Convert back to sequential
        let sequential = concurrent.into_sequential();
        assert_eq!(sequential.node_count(), 4); // A, B, C, D
    }
    
    #[test]
    fn test_scc_estimation() {
        let mut graph = DependencyGraph::new();
        
        // Create a graph with potential cycles: A <-> B, C -> D
        graph.add_edge("A".to_string(), "B".to_string()).unwrap();
        graph.add_edge("B".to_string(), "A".to_string()).unwrap();
        graph.add_edge("C".to_string(), "D".to_string()).unwrap();
        graph.add_node("E".to_string()).unwrap(); // Isolated
        
        let scc_count = graph.estimate_scc_count();
        
        // Should estimate: 1 SCC (A,B have both in/out), plus isolated/chain nodes (C,D,E)
        assert!(scc_count >= 3); // At least C, D, E as separate components
    }
    
    #[test]
    fn test_nodes_by_language() {
        let mut graph = DependencyGraph::new();
        
        graph.add_node("main.py".to_string()).unwrap();
        graph.add_node("utils.py".to_string()).unwrap();
        graph.add_node("app.js".to_string()).unwrap();
        graph.add_node("lib.rs".to_string()).unwrap();
        
        let python_nodes = graph.nodes_by_language("python");
        let js_nodes = graph.nodes_by_language("javascript");
        let rust_nodes = graph.nodes_by_language("rust");
        
        assert_eq!(python_nodes.len(), 2);
        assert_eq!(js_nodes.len(), 1);
        assert_eq!(rust_nodes.len(), 1);
        
        assert!(python_nodes.contains(&&"main.py".to_string()));
        assert!(python_nodes.contains(&&"utils.py".to_string()));
    }
}
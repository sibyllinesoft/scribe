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

use dashmap::DashMap;
use parking_lot::RwLock;
use scribe_core::{error::ScribeError, file, Language, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Internal node identifier type for efficient graph operations (usize for array indexing)
pub type InternalNodeId = usize;

/// External node identifier type for the dependency graph (file paths)
pub type NodeId = String;

/// Edge weight type (unused in unweighted PageRank, but reserved for extensions)
pub type EdgeWeight = f64;

/// Efficient dependency graph representation optimized for PageRank computation
/// Uses integer-based internal representation for massive performance improvements
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Forward adjacency list: internal_id -> set of internal_ids it imports
    forward_edges: Vec<HashSet<InternalNodeId>>,

    /// Reverse adjacency list: internal_id -> set of internal_ids that import it (for PageRank)
    reverse_edges: Vec<HashSet<InternalNodeId>>,

    /// Mapping from file path to internal node ID
    path_to_id: HashMap<NodeId, InternalNodeId>,

    /// Mapping from internal node ID to file path
    id_to_path: Vec<NodeId>,

    /// Node metadata cache (indexed by internal ID)
    node_metadata: Vec<Option<NodeMetadata>>,

    /// Graph statistics cache (invalidated on mutations)
    stats_cache: Option<GraphStatistics>,

    /// Next available internal node ID
    next_id: InternalNodeId,
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
        let path = std::path::Path::new(&file_path);
        let language_enum = file::detect_language_from_path(path);
        let language = if matches!(language_enum, Language::Unknown) {
            None
        } else {
            Some(file::language_display_name(&language_enum).to_lowercase())
        };
        let is_entrypoint = file::is_entrypoint_path(path, &language_enum);
        let is_test = file::is_test_path(path);

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
            forward_edges: Vec::new(),
            reverse_edges: Vec::new(),
            path_to_id: HashMap::new(),
            id_to_path: Vec::new(),
            node_metadata: Vec::new(),
            stats_cache: None,
            next_id: 0,
        }
    }

    /// Create with initial capacity hint for performance optimization
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            forward_edges: Vec::with_capacity(capacity),
            reverse_edges: Vec::with_capacity(capacity),
            path_to_id: HashMap::with_capacity(capacity),
            id_to_path: Vec::with_capacity(capacity),
            node_metadata: Vec::with_capacity(capacity),
            stats_cache: None,
            next_id: 0,
        }
    }

    /// Add a node to the graph (can exist without edges)
    pub fn add_node(&mut self, node_id: NodeId) -> Result<InternalNodeId> {
        // Check if node already exists
        if let Some(&existing_id) = self.path_to_id.get(&node_id) {
            return Ok(existing_id);
        }

        let internal_id = self.next_id;
        self.next_id += 1;

        // Add to mappings
        self.path_to_id.insert(node_id.clone(), internal_id);
        self.id_to_path.push(node_id.clone());

        // Initialize empty adjacency lists
        self.forward_edges.push(HashSet::new());
        self.reverse_edges.push(HashSet::new());

        // Add default metadata
        self.node_metadata.push(Some(NodeMetadata::new(node_id)));

        // Invalidate cache
        self.stats_cache = None;

        Ok(internal_id)
    }

    /// Add a node with metadata
    pub fn add_node_with_metadata(
        &mut self,
        node_id: NodeId,
        metadata: NodeMetadata,
    ) -> Result<InternalNodeId> {
        let internal_id = self.add_node(node_id)?;
        self.node_metadata[internal_id] = Some(metadata);
        Ok(internal_id)
    }

    /// Add an import edge: from_file imports to_file
    pub fn add_edge(&mut self, from_node: NodeId, to_node: NodeId) -> Result<()> {
        // Ensure both nodes exist and get their internal IDs
        let from_id = self.add_node(from_node)?;
        let to_id = self.add_node(to_node)?;

        // Add forward edge: from_id -> to_id
        self.forward_edges[from_id].insert(to_id);

        // Add reverse edge: to_id <- from_id
        self.reverse_edges[to_id].insert(from_id);

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
        let internal_id = match self.path_to_id.get(node_id) {
            Some(&id) => id,
            None => return Ok(false),
        };

        // Get outgoing edges to clean up reverse references
        let outgoing = self.forward_edges[internal_id].clone();
        for target_id in &outgoing {
            self.reverse_edges[*target_id].remove(&internal_id);
        }

        // Get incoming edges to clean up forward references
        let incoming = self.reverse_edges[internal_id].clone();
        for source_id in &incoming {
            self.forward_edges[*source_id].remove(&internal_id);
        }

        // Clear the adjacency lists for this node
        self.forward_edges[internal_id].clear();
        self.reverse_edges[internal_id].clear();

        // Remove metadata
        self.node_metadata[internal_id] = None;

        // Remove from path mapping (but keep internal_id for consistency)
        self.path_to_id.remove(node_id);

        // Note: We don't remove from id_to_path to maintain index consistency
        // Instead, we'll need to handle None cases when iterating

        // Invalidate cache
        self.stats_cache = None;

        Ok(true)
    }

    /// Remove an edge between two nodes
    pub fn remove_edge(&mut self, from_node: &NodeId, to_node: &NodeId) -> Result<bool> {
        let from_id = match self.path_to_id.get(from_node) {
            Some(&id) => id,
            None => return Ok(false),
        };

        let to_id = match self.path_to_id.get(to_node) {
            Some(&id) => id,
            None => return Ok(false),
        };

        let forward_removed = self.forward_edges[from_id].remove(&to_id);
        let reverse_removed = self.reverse_edges[to_id].remove(&from_id);

        if forward_removed || reverse_removed {
            self.stats_cache = None;
        }

        Ok(forward_removed || reverse_removed)
    }

    /// Check if a node exists in the graph
    pub fn contains_node(&self, node_id: &NodeId) -> bool {
        self.path_to_id.contains_key(node_id)
    }

    /// Check if an edge exists between two nodes
    pub fn contains_edge(&self, from_node: &NodeId, to_node: &NodeId) -> bool {
        match (self.path_to_id.get(from_node), self.path_to_id.get(to_node)) {
            (Some(&from_id), Some(&to_id)) => self.forward_edges[from_id].contains(&to_id),
            _ => false,
        }
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.path_to_id.len()
    }

    /// Get the total number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.forward_edges.iter().map(|edges| edges.len()).sum()
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> impl Iterator<Item = &NodeId> {
        self.path_to_id.keys()
    }

    /// Get all edges in the graph as (from, to) pairs
    pub fn edges(&self) -> impl Iterator<Item = (String, String)> + '_ {
        self.forward_edges
            .iter()
            .enumerate()
            .flat_map(move |(from_id, targets)| {
                let from_path = self.id_to_path[from_id].clone();
                targets.iter().map(move |&to_id| {
                    let to_path = self.id_to_path[to_id].clone();
                    (from_path.clone(), to_path)
                })
            })
    }
}

/// Degree and neighbor query operations
impl DependencyGraph {
    /// Get in-degree of a node (number of files that import this node)
    pub fn in_degree(&self, node_id: &NodeId) -> usize {
        match self.path_to_id.get(node_id) {
            Some(&internal_id) => self.reverse_edges[internal_id].len(),
            None => 0,
        }
    }

    /// Get out-degree of a node (number of files this node imports)
    pub fn out_degree(&self, node_id: &NodeId) -> usize {
        match self.path_to_id.get(node_id) {
            Some(&internal_id) => self.forward_edges[internal_id].len(),
            None => 0,
        }
    }

    /// Get total degree of a node (in + out)
    pub fn degree(&self, node_id: &NodeId) -> usize {
        self.in_degree(node_id) + self.out_degree(node_id)
    }

    /// Get nodes that this node imports (outgoing edges)
    pub fn outgoing_neighbors(&self, node_id: &NodeId) -> Option<Vec<&NodeId>> {
        match self.path_to_id.get(node_id) {
            Some(&internal_id) => {
                let neighbors: Vec<&NodeId> = self.forward_edges[internal_id]
                    .iter()
                    .map(|&target_id| &self.id_to_path[target_id])
                    .collect();
                Some(neighbors)
            }
            None => None,
        }
    }

    /// Get nodes that import this node (incoming edges) - important for PageRank
    pub fn incoming_neighbors(&self, node_id: &NodeId) -> Option<Vec<&NodeId>> {
        match self.path_to_id.get(node_id) {
            Some(&internal_id) => {
                let neighbors: Vec<&NodeId> = self.reverse_edges[internal_id]
                    .iter()
                    .map(|&source_id| &self.id_to_path[source_id])
                    .collect();
                Some(neighbors)
            }
            None => None,
        }
    }

    /// Get both incoming and outgoing neighbors
    pub fn all_neighbors(&self, node_id: &NodeId) -> HashSet<&NodeId> {
        let mut neighbors = HashSet::new();

        if let Some(&internal_id) = self.path_to_id.get(node_id) {
            // Add outgoing neighbors
            for &target_id in &self.forward_edges[internal_id] {
                neighbors.insert(&self.id_to_path[target_id]);
            }

            // Add incoming neighbors
            for &source_id in &self.reverse_edges[internal_id] {
                neighbors.insert(&self.id_to_path[source_id]);
            }
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

    /// Internal API: Get internal node ID for path (for PageRank optimization)
    pub(crate) fn get_internal_id(&self, node_id: &NodeId) -> Option<InternalNodeId> {
        self.path_to_id.get(node_id).copied()
    }

    /// Internal API: Get path for internal ID
    pub(crate) fn get_path(&self, internal_id: InternalNodeId) -> Option<&NodeId> {
        self.id_to_path.get(internal_id)
    }

    /// Internal API: Get incoming neighbors by internal ID (for PageRank)
    pub(crate) fn incoming_neighbors_by_id(
        &self,
        internal_id: InternalNodeId,
    ) -> Option<&HashSet<InternalNodeId>> {
        self.reverse_edges.get(internal_id)
    }

    /// Internal API: Get out-degree by internal ID (for PageRank)
    pub(crate) fn out_degree_by_id(&self, internal_id: InternalNodeId) -> usize {
        self.forward_edges
            .get(internal_id)
            .map_or(0, |edges| edges.len())
    }

    /// Internal API: Get total number of active nodes (for PageRank)
    pub(crate) fn internal_node_count(&self) -> usize {
        self.path_to_id.len()
    }

    /// Internal API: Iterator over all internal node IDs with their paths
    pub(crate) fn internal_nodes(&self) -> impl Iterator<Item = (InternalNodeId, &NodeId)> {
        self.path_to_id.iter().map(|(path, &id)| (id, path))
    }
}

/// Node metadata and information queries
impl DependencyGraph {
    /// Get metadata for a node
    pub fn node_metadata(&self, node_id: &NodeId) -> Option<&NodeMetadata> {
        match self.path_to_id.get(node_id) {
            Some(&internal_id) => self.node_metadata[internal_id].as_ref(),
            None => None,
        }
    }

    /// Set metadata for a node
    pub fn set_node_metadata(&mut self, node_id: NodeId, metadata: NodeMetadata) -> Result<()> {
        match self.path_to_id.get(&node_id) {
            Some(&internal_id) => {
                self.node_metadata[internal_id] = Some(metadata);
                Ok(())
            }
            None => Err(ScribeError::invalid_operation(
                format!("Node {} does not exist in graph", node_id),
                "set_node_metadata".to_string(),
            )),
        }
    }

    /// Get all entrypoint nodes
    pub fn entrypoint_nodes(&self) -> Vec<&NodeId> {
        self.node_metadata
            .iter()
            .enumerate()
            .filter_map(|(internal_id, meta_opt)| {
                if let Some(meta) = meta_opt {
                    if meta.is_entrypoint {
                        return Some(&self.id_to_path[internal_id]);
                    }
                }
                None
            })
            .collect()
    }

    /// Get all test nodes
    pub fn test_nodes(&self) -> Vec<&NodeId> {
        self.node_metadata
            .iter()
            .enumerate()
            .filter_map(|(internal_id, meta_opt)| {
                if let Some(meta) = meta_opt {
                    if meta.is_test {
                        return Some(&self.id_to_path[internal_id]);
                    }
                }
                None
            })
            .collect()
    }

    /// Get nodes by language
    pub fn nodes_by_language(&self, language: &str) -> Vec<&NodeId> {
        self.node_metadata
            .iter()
            .enumerate()
            .filter_map(|(internal_id, meta_opt)| {
                if let Some(meta) = meta_opt {
                    if meta.language.as_deref() == Some(language) {
                        return Some(&self.id_to_path[internal_id]);
                    }
                }
                None
            })
            .collect()
    }
}

/// Specialized operations for PageRank computation
impl DependencyGraph {
    /// Get all nodes with their reverse edge neighbors (for PageRank iteration)
    pub fn pagerank_iterator(&self) -> impl Iterator<Item = (&NodeId, Option<Vec<&NodeId>>)> + '_ {
        self.path_to_id.iter().map(|(node_path, &internal_id)| {
            let incoming: Option<Vec<&NodeId>> = if !self.reverse_edges[internal_id].is_empty() {
                Some(
                    self.reverse_edges[internal_id]
                        .iter()
                        .map(|&source_id| &self.id_to_path[source_id])
                        .collect(),
                )
            } else {
                Some(Vec::new())
            };
            (node_path, incoming)
        })
    }

    /// Get dangling nodes (nodes with no outgoing edges)
    pub fn dangling_nodes(&self) -> Vec<&NodeId> {
        self.path_to_id
            .iter()
            .filter(|(_, &internal_id)| self.forward_edges[internal_id].is_empty())
            .map(|(node_path, _)| node_path)
            .collect()
    }

    /// Get strongly connected components (simplified estimation for statistics)
    pub fn estimate_scc_count(&self) -> usize {
        if self.path_to_id.is_empty() {
            return 0;
        }

        // Count nodes with both in and out edges (likely in cycles)
        let potential_scc_nodes = self
            .path_to_id
            .iter()
            .filter(|(_, &internal_id)| {
                !self.reverse_edges[internal_id].is_empty()
                    && !self.forward_edges[internal_id].is_empty()
            })
            .count();

        // Rough estimate: most SCCs are small, assume average size of 3
        let estimated_scc = if potential_scc_nodes > 0 {
            std::cmp::max(1, potential_scc_nodes / 3)
        } else {
            0
        };

        // Add isolated nodes and simple chains
        let isolated_nodes = self.path_to_id.len() - potential_scc_nodes;
        estimated_scc + isolated_nodes
    }

    /// Check if the graph is strongly connected (simplified check)
    pub fn is_strongly_connected(&self) -> bool {
        if self.path_to_id.is_empty() {
            return true;
        }

        // Simplified check: all nodes have both in and out edges
        self.path_to_id.iter().all(|(_, &internal_id)| {
            !self.reverse_edges[internal_id].is_empty()
                && !self.forward_edges[internal_id].is_empty()
        })
    }
}

/// Concurrent graph operations for performance
impl DependencyGraph {
    /// Create a thread-safe concurrent graph for parallel operations
    pub fn into_concurrent(self) -> ConcurrentDependencyGraph {
        // Convert Vec<HashSet> to DashMap representation for concurrency
        let forward_edges = DashMap::new();
        let reverse_edges = DashMap::new();

        for (internal_id, edge_set) in self.forward_edges.into_iter().enumerate() {
            forward_edges.insert(internal_id, edge_set);
        }

        for (internal_id, edge_set) in self.reverse_edges.into_iter().enumerate() {
            reverse_edges.insert(internal_id, edge_set);
        }

        ConcurrentDependencyGraph {
            forward_edges,
            reverse_edges,
            path_to_id: DashMap::from_iter(self.path_to_id),
            id_to_path: RwLock::new(self.id_to_path),
            node_metadata: RwLock::new(self.node_metadata),
            stats_cache: RwLock::new(self.stats_cache),
            next_id: RwLock::new(self.next_id),
        }
    }
}

/// Thread-safe concurrent version of DependencyGraph
#[derive(Debug)]
pub struct ConcurrentDependencyGraph {
    forward_edges: DashMap<InternalNodeId, HashSet<InternalNodeId>>,
    reverse_edges: DashMap<InternalNodeId, HashSet<InternalNodeId>>,
    path_to_id: DashMap<NodeId, InternalNodeId>,
    id_to_path: RwLock<Vec<NodeId>>,
    node_metadata: RwLock<Vec<Option<NodeMetadata>>>,
    stats_cache: RwLock<Option<GraphStatistics>>,
    next_id: RwLock<InternalNodeId>,
}

impl ConcurrentDependencyGraph {
    /// Add a node concurrently
    pub fn add_node(&self, node_id: NodeId) -> Result<InternalNodeId> {
        // Check if node already exists
        if let Some(existing_id) = self.path_to_id.get(&node_id) {
            return Ok(*existing_id);
        }

        let internal_id = {
            let mut next_id = self.next_id.write();
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Add to mappings
        self.path_to_id.insert(node_id.clone(), internal_id);
        {
            let mut id_to_path = self.id_to_path.write();
            id_to_path.push(node_id.clone());
        }

        // Initialize empty adjacency lists
        self.forward_edges.insert(internal_id, HashSet::new());
        self.reverse_edges.insert(internal_id, HashSet::new());

        // Add default metadata
        {
            let mut metadata = self.node_metadata.write();
            metadata.push(Some(NodeMetadata::new(node_id)));
        }

        // Invalidate stats cache
        *self.stats_cache.write() = None;

        Ok(internal_id)
    }

    /// Get in-degree concurrently
    pub fn in_degree(&self, node_id: &NodeId) -> usize {
        match self.path_to_id.get(node_id) {
            Some(internal_id) => self
                .reverse_edges
                .get(&internal_id)
                .map_or(0, |entry| entry.len()),
            None => 0,
        }
    }

    /// Get out-degree concurrently  
    pub fn out_degree(&self, node_id: &NodeId) -> usize {
        match self.path_to_id.get(node_id) {
            Some(internal_id) => self
                .forward_edges
                .get(&internal_id)
                .map_or(0, |entry| entry.len()),
            None => 0,
        }
    }

    /// Convert back to single-threaded graph
    pub fn into_sequential(self) -> DependencyGraph {
        let id_to_path = self.id_to_path.into_inner();
        let node_metadata = self.node_metadata.into_inner();
        let stats_cache = self.stats_cache.into_inner();
        let next_id = self.next_id.into_inner();

        // Convert DashMap back to Vec
        let mut forward_edges = vec![HashSet::new(); next_id];
        let mut reverse_edges = vec![HashSet::new(); next_id];

        for (internal_id, edge_set) in self.forward_edges.into_iter() {
            if internal_id < forward_edges.len() {
                forward_edges[internal_id] = edge_set;
            }
        }

        for (internal_id, edge_set) in self.reverse_edges.into_iter() {
            if internal_id < reverse_edges.len() {
                reverse_edges[internal_id] = edge_set;
            }
        }

        DependencyGraph {
            forward_edges,
            reverse_edges,
            path_to_id: self.path_to_id.into_iter().collect(),
            id_to_path,
            node_metadata,
            stats_cache,
            next_id,
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
        graph
            .add_edge("main.py".to_string(), "utils.py".to_string())
            .unwrap();

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
        graph
            .add_node_with_metadata("main.py".to_string(), metadata)
            .unwrap();

        let retrieved = graph.node_metadata(&"main.py".to_string()).unwrap();
        assert_eq!(retrieved.file_path, "main.py");
        assert_eq!(retrieved.language, Some("python".to_string()));
        assert!(retrieved.is_entrypoint);
        assert!(!retrieved.is_test);
        assert_eq!(retrieved.size_bytes, 1024);
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(
            detect_language_from_extension("main.py"),
            Some("python".to_string())
        );
        assert_eq!(
            detect_language_from_extension("app.js"),
            Some("javascript".to_string())
        );
        assert_eq!(
            detect_language_from_extension("lib.rs"),
            Some("rust".to_string())
        );
        assert_eq!(
            detect_language_from_extension("server.go"),
            Some("go".to_string())
        );
        assert_eq!(
            detect_language_from_extension("component.tsx"),
            Some("typescript".to_string())
        );
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

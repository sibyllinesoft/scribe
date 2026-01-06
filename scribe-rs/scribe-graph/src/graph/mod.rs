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

mod concurrent;
#[cfg(test)]
mod tests;
mod types;

pub use concurrent::ConcurrentDependencyGraph;
pub use types::{
    DegreeInfo, EdgeWeight, GraphStatistics, InternalNodeId, NodeId, NodeMetadata,
    TraversalDirection,
};

use dashmap::DashMap;
use scribe_core::{error::ScribeError, Result};
use std::collections::{HashMap, HashSet};

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

    /// Internal BFS traversal helper for transitive reachability
    fn bfs_traverse(
        &self,
        node_id: &NodeId,
        max_depth: Option<usize>,
        use_outgoing: bool,
    ) -> HashSet<NodeId> {
        use std::collections::VecDeque;

        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if !self.contains_node(node_id) {
            return result;
        }

        queue.push_back((node_id.clone(), 0));
        visited.insert(node_id.clone());

        while let Some((current, depth)) = queue.pop_front() {
            if max_depth.is_some_and(|max_d| depth >= max_d) {
                continue;
            }

            let neighbors = if use_outgoing {
                self.outgoing_neighbors(&current)
            } else {
                self.incoming_neighbors(&current)
            };

            if let Some(neighbors) = neighbors {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        result.insert(neighbor.clone());
                        queue.push_back((neighbor.clone(), depth + 1));
                    }
                }
            }
        }

        result
    }

    /// Get all transitive dependencies of a node (files it depends on, transitively)
    pub fn transitive_dependencies(
        &self,
        node_id: &NodeId,
        max_depth: Option<usize>,
    ) -> HashSet<NodeId> {
        self.bfs_traverse(node_id, max_depth, true)
    }

    /// Get all transitive dependents of a node (files that depend on it, transitively)
    pub fn transitive_dependents(
        &self,
        node_id: &NodeId,
        max_depth: Option<usize>,
    ) -> HashSet<NodeId> {
        self.bfs_traverse(node_id, max_depth, false)
    }

    /// Compute the closure of a set of seed nodes
    pub fn compute_closure(
        &self,
        seeds: &[NodeId],
        direction: TraversalDirection,
        max_depth: Option<usize>,
    ) -> HashSet<NodeId> {
        let mut result: HashSet<NodeId> = seeds.iter().cloned().collect();

        for seed in seeds {
            let reachable = match direction {
                TraversalDirection::Dependencies => self.transitive_dependencies(seed, max_depth),
                TraversalDirection::Dependents => self.transitive_dependents(seed, max_depth),
                TraversalDirection::Both => {
                    let mut combined = self.transitive_dependencies(seed, max_depth);
                    combined.extend(self.transitive_dependents(seed, max_depth));
                    combined
                }
            };
            result.extend(reachable);
        }

        result
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

        ConcurrentDependencyGraph::from_components(
            forward_edges,
            reverse_edges,
            self.path_to_id,
            self.id_to_path,
            self.node_metadata,
            self.stats_cache,
            self.next_id,
        )
    }
}

impl ConcurrentDependencyGraph {
    /// Convert back to single-threaded graph
    pub fn into_sequential(self) -> DependencyGraph {
        let (forward_dashmap, reverse_dashmap, path_to_id, id_to_path, node_metadata, stats_cache, next_id) =
            self.into_components();

        // Convert DashMap back to Vec
        let mut forward_edges = vec![HashSet::new(); next_id];
        let mut reverse_edges = vec![HashSet::new(); next_id];

        for (internal_id, edge_set) in forward_dashmap.into_iter() {
            if internal_id < forward_edges.len() {
                forward_edges[internal_id] = edge_set;
            }
        }

        for (internal_id, edge_set) in reverse_dashmap.into_iter() {
            if internal_id < reverse_edges.len() {
                reverse_edges[internal_id] = edge_set;
            }
        }

        DependencyGraph {
            forward_edges,
            reverse_edges,
            path_to_id: path_to_id.into_iter().collect(),
            id_to_path,
            node_metadata,
            stats_cache,
            next_id,
        }
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

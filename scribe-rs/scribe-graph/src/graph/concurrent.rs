//! Thread-safe concurrent version of DependencyGraph.

use dashmap::DashMap;
use parking_lot::RwLock;
use scribe_core::Result;
use std::collections::{HashMap, HashSet};

use super::types::{GraphStatistics, InternalNodeId, NodeId, NodeMetadata};

/// Thread-safe concurrent version of DependencyGraph
#[derive(Debug)]
pub struct ConcurrentDependencyGraph {
    pub(crate) forward_edges: DashMap<InternalNodeId, HashSet<InternalNodeId>>,
    pub(crate) reverse_edges: DashMap<InternalNodeId, HashSet<InternalNodeId>>,
    pub(crate) path_to_id: DashMap<NodeId, InternalNodeId>,
    pub(crate) id_to_path: RwLock<Vec<NodeId>>,
    pub(crate) node_metadata: RwLock<Vec<Option<NodeMetadata>>>,
    pub(crate) stats_cache: RwLock<Option<GraphStatistics>>,
    pub(crate) next_id: RwLock<InternalNodeId>,
}

impl ConcurrentDependencyGraph {
    /// Create a new concurrent graph from components
    pub(crate) fn from_components(
        forward_edges: DashMap<InternalNodeId, HashSet<InternalNodeId>>,
        reverse_edges: DashMap<InternalNodeId, HashSet<InternalNodeId>>,
        path_to_id: HashMap<NodeId, InternalNodeId>,
        id_to_path: Vec<NodeId>,
        node_metadata: Vec<Option<NodeMetadata>>,
        stats_cache: Option<GraphStatistics>,
        next_id: InternalNodeId,
    ) -> Self {
        Self {
            forward_edges,
            reverse_edges,
            path_to_id: DashMap::from_iter(path_to_id),
            id_to_path: RwLock::new(id_to_path),
            node_metadata: RwLock::new(node_metadata),
            stats_cache: RwLock::new(stats_cache),
            next_id: RwLock::new(next_id),
        }
    }

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

    /// Decompose into components for conversion back to sequential
    pub(crate) fn into_components(
        self,
    ) -> (
        DashMap<InternalNodeId, HashSet<InternalNodeId>>,
        DashMap<InternalNodeId, HashSet<InternalNodeId>>,
        DashMap<NodeId, InternalNodeId>,
        Vec<NodeId>,
        Vec<Option<NodeMetadata>>,
        Option<GraphStatistics>,
        InternalNodeId,
    ) {
        (
            self.forward_edges,
            self.reverse_edges,
            self.path_to_id,
            self.id_to_path.into_inner(),
            self.node_metadata.into_inner(),
            self.stats_cache.into_inner(),
            self.next_id.into_inner(),
        )
    }
}

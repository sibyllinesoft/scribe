//! Type definitions for the dependency graph module.

use scribe_core::{file, Language};
use serde::{Deserialize, Serialize};

/// Internal node identifier type for efficient graph operations (usize for array indexing)
pub type InternalNodeId = usize;

/// External node identifier type for the dependency graph (file paths)
pub type NodeId = String;

/// Edge weight type (unused in unweighted PageRank, but reserved for extensions)
pub type EdgeWeight = f64;

/// Direction for graph traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    /// Traverse along outgoing edges (dependencies)
    Dependencies,
    /// Traverse along incoming edges (dependents)
    Dependents,
    /// Traverse in both directions
    Both,
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

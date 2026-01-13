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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traversal_direction_dependencies() {
        let dir = TraversalDirection::Dependencies;
        assert_eq!(dir, TraversalDirection::Dependencies);
        assert_ne!(dir, TraversalDirection::Dependents);
    }

    #[test]
    fn test_traversal_direction_dependents() {
        let dir = TraversalDirection::Dependents;
        assert_eq!(dir, TraversalDirection::Dependents);
        assert_ne!(dir, TraversalDirection::Both);
    }

    #[test]
    fn test_traversal_direction_both() {
        let dir = TraversalDirection::Both;
        assert_eq!(dir, TraversalDirection::Both);
        assert_ne!(dir, TraversalDirection::Dependencies);
    }

    #[test]
    fn test_traversal_direction_clone() {
        let dir = TraversalDirection::Dependencies;
        let cloned = dir.clone();
        assert_eq!(dir, cloned);
    }

    #[test]
    fn test_traversal_direction_copy() {
        let dir = TraversalDirection::Both;
        let copied = dir;
        assert_eq!(dir, copied);
    }

    #[test]
    fn test_traversal_direction_debug() {
        let debug = format!("{:?}", TraversalDirection::Dependencies);
        assert_eq!(debug, "Dependencies");

        let debug = format!("{:?}", TraversalDirection::Dependents);
        assert_eq!(debug, "Dependents");

        let debug = format!("{:?}", TraversalDirection::Both);
        assert_eq!(debug, "Both");
    }

    #[test]
    fn test_node_metadata_new_rust() {
        let meta = NodeMetadata::new("src/main.rs".to_string());
        assert_eq!(meta.file_path, "src/main.rs");
        assert_eq!(meta.language, Some("rust".to_string()));
        assert!(meta.is_entrypoint);
        assert!(!meta.is_test);
        assert_eq!(meta.size_bytes, 0);
    }

    #[test]
    fn test_node_metadata_new_python() {
        let meta = NodeMetadata::new("app.py".to_string());
        assert_eq!(meta.file_path, "app.py");
        assert_eq!(meta.language, Some("python".to_string()));
        assert!(meta.is_entrypoint);
    }

    #[test]
    fn test_node_metadata_new_test_file() {
        let meta = NodeMetadata::new("tests/test_utils.py".to_string());
        assert!(meta.is_test);
    }

    #[test]
    fn test_node_metadata_new_unknown() {
        let meta = NodeMetadata::new("file.xyz".to_string());
        assert!(meta.language.is_none());
        assert!(!meta.is_entrypoint);
    }

    #[test]
    fn test_node_metadata_with_size() {
        let meta = NodeMetadata::new("lib.rs".to_string()).with_size(1024);
        assert_eq!(meta.size_bytes, 1024);
        assert_eq!(meta.file_path, "lib.rs");
    }

    #[test]
    fn test_node_metadata_clone() {
        let meta = NodeMetadata::new("utils.rs".to_string()).with_size(500);
        let cloned = meta.clone();

        assert_eq!(meta.file_path, cloned.file_path);
        assert_eq!(meta.language, cloned.language);
        assert_eq!(meta.size_bytes, cloned.size_bytes);
    }

    #[test]
    fn test_node_metadata_partial_eq() {
        let meta1 = NodeMetadata::new("file.rs".to_string());
        let meta2 = NodeMetadata::new("file.rs".to_string());
        let meta3 = NodeMetadata::new("other.rs".to_string());

        assert_eq!(meta1, meta2);
        assert_ne!(meta1, meta3);
    }

    #[test]
    fn test_node_metadata_debug() {
        let meta = NodeMetadata::new("test.rs".to_string());
        let debug = format!("{:?}", meta);

        assert!(debug.contains("NodeMetadata"));
        assert!(debug.contains("test.rs"));
    }

    #[test]
    fn test_degree_info_creation() {
        let info = DegreeInfo {
            node_id: "module.rs".to_string(),
            in_degree: 5,
            out_degree: 10,
            total_degree: 15,
        };

        assert_eq!(info.node_id, "module.rs");
        assert_eq!(info.in_degree, 5);
        assert_eq!(info.out_degree, 10);
        assert_eq!(info.total_degree, 15);
    }

    #[test]
    fn test_degree_info_clone() {
        let info = DegreeInfo {
            node_id: "main.rs".to_string(),
            in_degree: 3,
            out_degree: 7,
            total_degree: 10,
        };

        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    #[test]
    fn test_degree_info_partial_eq() {
        let info1 = DegreeInfo {
            node_id: "a.rs".to_string(),
            in_degree: 1,
            out_degree: 2,
            total_degree: 3,
        };

        let info2 = DegreeInfo {
            node_id: "a.rs".to_string(),
            in_degree: 1,
            out_degree: 2,
            total_degree: 3,
        };

        let info3 = DegreeInfo {
            node_id: "b.rs".to_string(),
            in_degree: 1,
            out_degree: 2,
            total_degree: 3,
        };

        assert_eq!(info1, info2);
        assert_ne!(info1, info3);
    }

    #[test]
    fn test_degree_info_debug() {
        let info = DegreeInfo {
            node_id: "test.rs".to_string(),
            in_degree: 0,
            out_degree: 0,
            total_degree: 0,
        };

        let debug = format!("{:?}", info);
        assert!(debug.contains("DegreeInfo"));
        assert!(debug.contains("test.rs"));
    }

    #[test]
    fn test_graph_statistics_empty() {
        let stats = GraphStatistics::empty();

        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.in_degree_avg, 0.0);
        assert_eq!(stats.in_degree_max, 0);
        assert_eq!(stats.out_degree_avg, 0.0);
        assert_eq!(stats.out_degree_max, 0);
        assert_eq!(stats.strongly_connected_components, 0);
        assert_eq!(stats.graph_density, 0.0);
        assert_eq!(stats.isolated_nodes, 0);
        assert_eq!(stats.dangling_nodes, 0);
    }

    #[test]
    fn test_graph_statistics_custom() {
        let stats = GraphStatistics {
            total_nodes: 100,
            total_edges: 250,
            in_degree_avg: 2.5,
            in_degree_max: 15,
            out_degree_avg: 2.5,
            out_degree_max: 20,
            strongly_connected_components: 3,
            graph_density: 0.025,
            isolated_nodes: 5,
            dangling_nodes: 10,
        };

        assert_eq!(stats.total_nodes, 100);
        assert_eq!(stats.total_edges, 250);
        assert_eq!(stats.in_degree_avg, 2.5);
        assert_eq!(stats.in_degree_max, 15);
    }

    #[test]
    fn test_graph_statistics_clone() {
        let stats = GraphStatistics {
            total_nodes: 50,
            total_edges: 100,
            in_degree_avg: 2.0,
            in_degree_max: 10,
            out_degree_avg: 2.0,
            out_degree_max: 10,
            strongly_connected_components: 1,
            graph_density: 0.04,
            isolated_nodes: 2,
            dangling_nodes: 5,
        };

        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[test]
    fn test_graph_statistics_partial_eq() {
        let stats1 = GraphStatistics::empty();
        let stats2 = GraphStatistics::empty();
        let stats3 = GraphStatistics {
            total_nodes: 1,
            ..GraphStatistics::empty()
        };

        assert_eq!(stats1, stats2);
        assert_ne!(stats1, stats3);
    }

    #[test]
    fn test_graph_statistics_debug() {
        let stats = GraphStatistics::empty();
        let debug = format!("{:?}", stats);

        assert!(debug.contains("GraphStatistics"));
        assert!(debug.contains("total_nodes"));
    }

    #[test]
    fn test_graph_statistics_serialize() {
        let stats = GraphStatistics {
            total_nodes: 25,
            total_edges: 50,
            in_degree_avg: 2.0,
            in_degree_max: 8,
            out_degree_avg: 2.0,
            out_degree_max: 6,
            strongly_connected_components: 2,
            graph_density: 0.08,
            isolated_nodes: 1,
            dangling_nodes: 3,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: GraphStatistics = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_graph_statistics_serialize_empty() {
        let stats = GraphStatistics::empty();
        let json = serde_json::to_string(&stats).unwrap();

        assert!(json.contains("total_nodes"));
        assert!(json.contains(":0"));
    }

    #[test]
    fn test_node_metadata_javascript() {
        let meta = NodeMetadata::new("index.js".to_string());
        assert_eq!(meta.language, Some("javascript".to_string()));
        assert!(meta.is_entrypoint);
    }

    #[test]
    fn test_node_metadata_typescript() {
        let meta = NodeMetadata::new("index.ts".to_string());
        assert_eq!(meta.language, Some("typescript".to_string()));
        assert!(meta.is_entrypoint);
    }

    #[test]
    fn test_node_metadata_go() {
        let meta = NodeMetadata::new("main.go".to_string());
        assert_eq!(meta.language, Some("go".to_string()));
        assert!(meta.is_entrypoint);
    }
}

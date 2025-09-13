//! # Legacy Graph Traversal (Stub Implementation)
//! 
//! This module provides stub implementations for backward compatibility.
//! New code should use the PageRank centrality system directly.

/// Legacy graph traversal maintained for backward compatibility
#[derive(Debug)]
pub struct GraphTraversal;

impl Default for GraphTraversal {
    fn default() -> Self {
        Self
    }
}

/// Legacy traversal order
#[derive(Debug, Clone)]
pub enum TraversalOrder {
    DepthFirst,
    BreadthFirst,
}

impl Default for TraversalOrder {
    fn default() -> Self {
        Self::DepthFirst
    }
}
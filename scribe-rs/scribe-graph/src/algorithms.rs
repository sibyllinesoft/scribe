//! # Legacy Graph Algorithms (Stub Implementation)
//! 
//! This module provides stub implementations for backward compatibility.
//! New code should use the PageRank centrality system directly.

use scribe_core::Result;
use crate::graph::DependencyGraph;

/// Legacy graph algorithms maintained for backward compatibility
#[derive(Debug)]
pub struct GraphAlgorithms;

impl GraphAlgorithms {
    pub fn new() -> Self {
        Self
    }
    
    pub fn find_dependencies(&self, _graph: &DependencyGraph) -> Result<Vec<String>> {
        // Stub implementation
        Ok(Vec::new())
    }
}

impl Default for GraphAlgorithms {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy path finder
#[derive(Debug)]
pub struct PathFinder;

impl Default for PathFinder {
    fn default() -> Self {
        Self
    }
}
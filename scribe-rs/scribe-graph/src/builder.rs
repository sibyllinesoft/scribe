//! # Legacy Graph Builder (Stub Implementation)
//!
//! This module provides stub implementations for backward compatibility.
//! New code should use the PageRank centrality system directly.

use crate::graph::DependencyGraph;
use scribe_analysis::AnalysisResult;
use scribe_core::Result;

/// Legacy graph builder maintained for backward compatibility
#[derive(Debug)]
pub struct GraphBuilder;

impl GraphBuilder {
    pub fn new() -> Self {
        Self
    }

    pub async fn build_from_analysis(&self, _analysis: &AnalysisResult) -> Result<DependencyGraph> {
        // Stub implementation - return empty graph
        Ok(DependencyGraph::new())
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy build options
#[derive(Debug, Clone)]
pub struct BuildOptions;

impl Default for BuildOptions {
    fn default() -> Self {
        Self
    }
}

//! Legacy PageRank iteration methods.
//!
//! These methods are kept for compatibility but are superseded by the optimized
//! versions that use internal node IDs for O(1) lookups.

use crate::graph::{DependencyGraph, NodeId};
use scribe_core::Result;
use std::collections::HashMap;

/// Legacy iteration methods using HashMap-based score storage
pub struct LegacyIterations;

impl LegacyIterations {
    /// Compute a single PageRank iteration (sequential version) - LEGACY
    pub fn compute_iteration_sequential(
        damping_factor: f64,
        graph: &DependencyGraph,
        nodes: &[NodeId],
        previous_scores: &HashMap<NodeId, f64>,
        current_scores: &mut HashMap<NodeId, f64>,
    ) -> Result<()> {
        let num_nodes = nodes.len() as f64;
        let teleport_prob = (1.0 - damping_factor) / num_nodes;

        for node in nodes {
            let mut new_score = teleport_prob;

            // Sum contributions from nodes that link to this node (reverse edges)
            if let Some(incoming_neighbors) = graph.incoming_neighbors(node) {
                for linking_node in incoming_neighbors {
                    if let Some(&linking_score) = previous_scores.get(linking_node) {
                        let linking_out_degree = graph.out_degree(linking_node).max(1) as f64;
                        new_score += damping_factor * (linking_score / linking_out_degree);
                    }
                }
            }

            current_scores.insert(node.clone(), new_score);
        }

        Ok(())
    }

    /// Compute a single PageRank iteration (parallel version) - LEGACY
    pub fn compute_iteration_parallel(
        damping_factor: f64,
        graph: &DependencyGraph,
        nodes: &[NodeId],
        previous_scores: &HashMap<NodeId, f64>,
        current_scores: &mut HashMap<NodeId, f64>,
    ) -> Result<()> {
        use rayon::prelude::*;

        let num_nodes = nodes.len() as f64;
        let teleport_prob = (1.0 - damping_factor) / num_nodes;

        // Parallel computation of new scores
        let new_scores: Vec<(NodeId, f64)> = nodes
            .par_iter()
            .map(|node| {
                let mut new_score = teleport_prob;

                // Sum contributions from nodes that link to this node (reverse edges)
                if let Some(incoming_neighbors) = graph.incoming_neighbors(node) {
                    for linking_node in incoming_neighbors {
                        if let Some(&linking_score) = previous_scores.get(linking_node) {
                            let linking_out_degree = graph.out_degree(linking_node).max(1) as f64;
                            new_score += damping_factor * (linking_score / linking_out_degree);
                        }
                    }
                }

                (node.clone(), new_score)
            })
            .collect();

        // Update scores map
        for (node, score) in new_scores {
            current_scores.insert(node, score);
        }

        Ok(())
    }

    /// Compute L1 norm difference between score vectors (convergence metric) - LEGACY
    pub fn compute_convergence_diff(
        current: &HashMap<NodeId, f64>,
        previous: &HashMap<NodeId, f64>,
    ) -> f64 {
        current
            .iter()
            .map(|(node, &current_score)| {
                let previous_score = previous.get(node).copied().unwrap_or(0.0);
                (current_score - previous_score).abs()
            })
            .sum()
    }
}

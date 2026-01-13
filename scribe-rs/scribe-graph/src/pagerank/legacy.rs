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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> DependencyGraph {
        let mut graph = DependencyGraph::new();
        graph.add_node("A".to_string()).unwrap();
        graph.add_node("B".to_string()).unwrap();
        graph.add_node("C".to_string()).unwrap();
        graph.add_edge("A".to_string(), "B".to_string()).unwrap();
        graph.add_edge("B".to_string(), "C".to_string()).unwrap();
        graph.add_edge("C".to_string(), "A".to_string()).unwrap();
        graph
    }

    #[test]
    fn test_compute_convergence_diff_identical() {
        let mut scores = HashMap::new();
        scores.insert("A".to_string(), 0.5);
        scores.insert("B".to_string(), 0.3);
        scores.insert("C".to_string(), 0.2);

        let diff = LegacyIterations::compute_convergence_diff(&scores, &scores);
        assert!((diff - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_convergence_diff_different() {
        let mut current = HashMap::new();
        current.insert("A".to_string(), 0.5);
        current.insert("B".to_string(), 0.3);

        let mut previous = HashMap::new();
        previous.insert("A".to_string(), 0.4);
        previous.insert("B".to_string(), 0.2);

        let diff = LegacyIterations::compute_convergence_diff(&current, &previous);
        // |0.5 - 0.4| + |0.3 - 0.2| = 0.1 + 0.1 = 0.2
        assert!((diff - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_convergence_diff_missing_previous() {
        let mut current = HashMap::new();
        current.insert("A".to_string(), 0.5);

        let previous = HashMap::new(); // Empty

        let diff = LegacyIterations::compute_convergence_diff(&current, &previous);
        // |0.5 - 0.0| = 0.5
        assert!((diff - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_iteration_sequential() {
        let graph = create_test_graph();
        let nodes: Vec<NodeId> = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let mut previous_scores = HashMap::new();
        for node in &nodes {
            previous_scores.insert(node.clone(), 1.0 / 3.0);
        }

        let mut current_scores = HashMap::new();

        LegacyIterations::compute_iteration_sequential(
            0.85,
            &graph,
            &nodes,
            &previous_scores,
            &mut current_scores,
        )
        .unwrap();

        // All nodes should have scores
        assert_eq!(current_scores.len(), 3);

        // Scores should sum to approximately 1.0
        let sum: f64 = current_scores.values().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_iteration_parallel() {
        let graph = create_test_graph();
        let nodes: Vec<NodeId> = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let mut previous_scores = HashMap::new();
        for node in &nodes {
            previous_scores.insert(node.clone(), 1.0 / 3.0);
        }

        let mut current_scores = HashMap::new();

        LegacyIterations::compute_iteration_parallel(
            0.85,
            &graph,
            &nodes,
            &previous_scores,
            &mut current_scores,
        )
        .unwrap();

        // All nodes should have scores
        assert_eq!(current_scores.len(), 3);

        // Scores should sum to approximately 1.0
        let sum: f64 = current_scores.values().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_convergence_after_iterations() {
        let graph = create_test_graph();
        let nodes: Vec<NodeId> = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let mut previous_scores = HashMap::new();
        for node in &nodes {
            previous_scores.insert(node.clone(), 1.0 / 3.0);
        }

        let mut current_scores = HashMap::new();

        // Run a few iterations
        for _ in 0..10 {
            LegacyIterations::compute_iteration_sequential(
                0.85,
                &graph,
                &nodes,
                &previous_scores,
                &mut current_scores,
            )
            .unwrap();

            std::mem::swap(&mut previous_scores, &mut current_scores);
            current_scores.clear();
        }

        // After several iterations, scores should be relatively stable
        LegacyIterations::compute_iteration_sequential(
            0.85,
            &graph,
            &nodes,
            &previous_scores,
            &mut current_scores,
        )
        .unwrap();

        let diff = LegacyIterations::compute_convergence_diff(&current_scores, &previous_scores);
        // Diff should be small after convergence
        assert!(diff < 0.01);
    }
}

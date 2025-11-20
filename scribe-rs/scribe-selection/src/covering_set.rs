//! Covering set computation for targeted code selection.
//!
//! This module provides functionality to compute minimal covering sets of files
//! needed to understand a specific code entity (function, class, module, etc.).
//! It integrates AST parsing, dependency graph traversal, and intelligent selection.

use crate::ast_parser::{AstParser, EntityLocation, EntityQuery};
use scribe_core::{Result, ScribeError};
use scribe_graph::{DependencyGraph, TraversalDirection};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Options for covering set computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSetOptions {
    /// Include files that the target depends on
    pub include_dependencies: bool,
    /// Include files that depend on the target
    pub include_dependents: bool,
    /// Maximum depth for dependency traversal (None = unlimited)
    pub max_depth: Option<usize>,
    /// Maximum number of files to include (None = unlimited)
    pub max_files: Option<usize>,
    /// Minimum importance score to include a file (0.0-1.0)
    pub min_importance: Option<f64>,
}

impl Default for CoveringSetOptions {
    fn default() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: None,
            max_files: None,
            min_importance: None,
        }
    }
}

impl CoveringSetOptions {
    /// Create default options optimized for understanding a target
    pub fn for_understanding() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: None, // Get all dependencies
            max_files: Some(100), // Reasonable limit
            min_importance: Some(0.3), // Exclude very low importance files
        }
    }

    /// Create options optimized for impact analysis
    pub fn for_impact_analysis() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: true, // Include what depends on this
            max_depth: Some(2), // Don't go too deep
            max_files: Some(50),
            min_importance: Some(0.4),
        }
    }

    /// Create options for minimal covering set
    pub fn minimal() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: Some(1), // Only direct dependencies
            max_files: Some(20),
            min_importance: Some(0.5),
        }
    }
}

/// Result of covering set computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSetResult {
    /// The target entity that was located
    pub target_entity: Option<EntityLocation>,
    /// Files included in the covering set
    pub files: Vec<CoveringSetFile>,
    /// Statistics about the computation
    pub statistics: CoveringSetStatistics,
    /// Explanation of why files were included
    pub inclusion_reasons: HashMap<String, String>,
}

/// Information about a file in the covering set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSetFile {
    /// File path (relative or absolute)
    pub path: String,
    /// Why this file was included
    pub reason: InclusionReason,
    /// Distance from target (0 = target file, 1 = direct dependency, etc.)
    pub distance: usize,
    /// Importance score if available
    pub importance: Option<f64>,
}

/// Reason a file was included in the covering set
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InclusionReason {
    /// File contains the target entity
    TargetFile,
    /// File is a direct dependency of the target
    DirectDependency,
    /// File is a transitive dependency
    TransitiveDependency,
    /// File directly depends on the target
    DirectDependent,
    /// File transitively depends on the target
    TransitiveDependent,
}

/// Statistics about the covering set computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSetStatistics {
    /// Total files examined
    pub files_examined: usize,
    /// Files in the final covering set
    pub files_selected: usize,
    /// Files excluded due to limits
    pub files_excluded: usize,
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Whether any limits were hit
    pub limits_reached: bool,
}

/// Computes covering sets for targeted code selection
pub struct CoveringSetComputer {
    ast_parser: AstParser,
}

impl CoveringSetComputer {
    /// Create a new covering set computer
    pub fn new() -> Result<Self> {
        Ok(Self {
            ast_parser: AstParser::new()?,
        })
    }

    /// Compute a covering set for a target entity
    ///
    /// # Arguments
    /// * `query` - Query to locate the target entity
    /// * `file_contents` - Map of file paths to their contents
    /// * `graph` - Dependency graph for the project
    /// * `options` - Options controlling the computation
    ///
    /// # Returns
    /// A covering set result containing the target and all related files
    pub fn compute_covering_set(
        &mut self,
        query: &EntityQuery,
        file_contents: &HashMap<String, String>,
        graph: &DependencyGraph,
        options: &CoveringSetOptions,
    ) -> Result<CoveringSetResult> {
        let mut statistics = CoveringSetStatistics {
            files_examined: file_contents.len(),
            files_selected: 0,
            files_excluded: 0,
            max_depth_reached: 0,
            limits_reached: false,
        };

        // Step 1: Find the target entity across all files
        let target_entity = self.find_target_entity(query, file_contents)?;

        if target_entity.is_none() {
            return Ok(CoveringSetResult {
                target_entity: None,
                files: Vec::new(),
                statistics,
                inclusion_reasons: HashMap::new(),
            });
        }

        let target = target_entity.clone().unwrap();
        let target_file = &target.file_path;

        // Step 2: Compute file closure using dependency graph
        let mut seed_files = vec![target_file.clone()];
        let direction = self.get_traversal_direction(options);
        let closure_files = graph.compute_closure(&seed_files, direction, options.max_depth);

        // Step 3: Build covering set with metadata
        let mut covering_set = Vec::new();
        let mut inclusion_reasons = HashMap::new();

        // Add target file first
        covering_set.push(CoveringSetFile {
            path: target_file.clone(),
            reason: InclusionReason::TargetFile,
            distance: 0,
            importance: None,
        });
        inclusion_reasons.insert(
            target_file.clone(),
            "Contains the target entity".to_string(),
        );

        // Add dependency/dependent files
        for file in closure_files {
            if file == *target_file {
                continue; // Already added
            }

            let (reason, distance) = self.compute_inclusion_info(
                &file,
                target_file,
                graph,
                options,
            );

            covering_set.push(CoveringSetFile {
                path: file.clone(),
                reason: reason.clone(),
                distance,
                importance: None,
            });

            let reason_text = self.format_inclusion_reason(&reason, distance);
            inclusion_reasons.insert(file, reason_text);
        }

        // Step 4: Apply limits and filters
        let original_count = covering_set.len();
        self.apply_limits(&mut covering_set, options, &mut statistics);

        if covering_set.len() < original_count {
            statistics.limits_reached = true;
            statistics.files_excluded = original_count - covering_set.len();
        }

        statistics.files_selected = covering_set.len();
        statistics.max_depth_reached = covering_set
            .iter()
            .map(|f| f.distance)
            .max()
            .unwrap_or(0);

        Ok(CoveringSetResult {
            target_entity,
            files: covering_set,
            statistics,
            inclusion_reasons,
        })
    }

    /// Find the target entity across all files
    fn find_target_entity(
        &mut self,
        query: &EntityQuery,
        file_contents: &HashMap<String, String>,
    ) -> Result<Option<EntityLocation>> {
        for (file_path, content) in file_contents {
            let entities = self.ast_parser.find_entities(content, file_path, query)?;
            if let Some(entity) = entities.first() {
                return Ok(Some(entity.clone()));
            }
        }
        Ok(None)
    }

    /// Determine traversal direction based on options
    fn get_traversal_direction(&self, options: &CoveringSetOptions) -> TraversalDirection {
        match (options.include_dependencies, options.include_dependents) {
            (true, true) => TraversalDirection::Both,
            (true, false) => TraversalDirection::Dependencies,
            (false, true) => TraversalDirection::Dependents,
            (false, false) => TraversalDirection::Dependencies, // Default to dependencies
        }
    }

    /// Compute inclusion reason and distance for a file
    fn compute_inclusion_info(
        &self,
        file: &str,
        target_file: &str,
        graph: &DependencyGraph,
        options: &CoveringSetOptions,
    ) -> (InclusionReason, usize) {
        let file_string = file.to_string();
        let target_string = target_file.to_string();

        // Check if it's a direct dependency
        if graph.contains_edge(&target_string, &file_string) {
            return (InclusionReason::DirectDependency, 1);
        }

        // Check if it's a direct dependent
        if graph.contains_edge(&file_string, &target_string) {
            return (InclusionReason::DirectDependent, 1);
        }

        // Otherwise it's transitive - compute distance
        let deps = graph.transitive_dependencies(&target_string, options.max_depth);
        if deps.contains(&file_string) {
            let distance = self.compute_distance(target_file, file, graph);
            return (InclusionReason::TransitiveDependency, distance);
        }

        let dependents = graph.transitive_dependents(&target_string, options.max_depth);
        if dependents.contains(&file_string) {
            let distance = self.compute_distance(file, target_file, graph);
            return (InclusionReason::TransitiveDependent, distance);
        }

        // Fallback
        (InclusionReason::TransitiveDependency, 2)
    }

    /// Compute distance between two files in the graph (simple BFS)
    fn compute_distance(&self, from: &str, to: &str, graph: &DependencyGraph) -> usize {
        use std::collections::{HashSet, VecDeque};

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((from.to_string(), 0));
        visited.insert(from.to_string());

        while let Some((current, dist)) = queue.pop_front() {
            if current == to {
                return dist;
            }

            if let Some(neighbors) = graph.outgoing_neighbors(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.to_string());
                        queue.push_back((neighbor.to_string(), dist + 1));
                    }
                }
            }
        }

        // Not reachable
        999
    }

    /// Format inclusion reason as human-readable text
    fn format_inclusion_reason(&self, reason: &InclusionReason, distance: usize) -> String {
        match reason {
            InclusionReason::TargetFile => "Contains the target entity".to_string(),
            InclusionReason::DirectDependency => "Direct dependency of target".to_string(),
            InclusionReason::TransitiveDependency => {
                format!("Transitive dependency (distance: {})", distance)
            }
            InclusionReason::DirectDependent => "Directly depends on target".to_string(),
            InclusionReason::TransitiveDependent => {
                format!("Transitively depends on target (distance: {})", distance)
            }
        }
    }

    /// Apply limits and filters to the covering set
    fn apply_limits(
        &self,
        covering_set: &mut Vec<CoveringSetFile>,
        options: &CoveringSetOptions,
        statistics: &mut CoveringSetStatistics,
    ) {
        // Sort by distance (closer files first) and importance
        covering_set.sort_by_key(|f| (f.distance, f.path.clone()));

        // Apply max_files limit
        if let Some(max_files) = options.max_files {
            if covering_set.len() > max_files {
                covering_set.truncate(max_files);
                statistics.limits_reached = true;
            }
        }

        // Filter by importance if specified
        if let Some(min_importance) = options.min_importance {
            covering_set.retain(|f| {
                // Always keep target file
                if matches!(f.reason, InclusionReason::TargetFile) {
                    return true;
                }
                // Keep if importance is above threshold (or unknown)
                f.importance.map_or(true, |imp| imp >= min_importance)
            });
        }
    }
}

impl Default for CoveringSetComputer {
    fn default() -> Self {
        Self::new().expect("Failed to create CoveringSetComputer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_parser::EntityQuery;

    #[test]
    fn test_covering_set_options() {
        let opts = CoveringSetOptions::default();
        assert!(opts.include_dependencies);
        assert!(!opts.include_dependents);

        let minimal = CoveringSetOptions::minimal();
        assert_eq!(minimal.max_depth, Some(1));
        assert_eq!(minimal.max_files, Some(20));
    }

    #[test]
    fn test_covering_set_computer_creation() {
        let computer = CoveringSetComputer::new();
        assert!(computer.is_ok());
    }

    #[test]
    fn test_inclusion_reason_formatting() {
        let computer = CoveringSetComputer::new().unwrap();

        let reason = computer.format_inclusion_reason(&InclusionReason::TargetFile, 0);
        assert_eq!(reason, "Contains the target entity");

        let reason = computer.format_inclusion_reason(&InclusionReason::DirectDependency, 1);
        assert_eq!(reason, "Direct dependency of target");

        let reason = computer.format_inclusion_reason(&InclusionReason::TransitiveDependency, 3);
        assert!(reason.contains("distance: 3"));
    }
}

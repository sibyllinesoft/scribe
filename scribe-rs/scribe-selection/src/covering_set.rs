//! Covering set computation for targeted code selection.
//!
//! This module provides functionality to compute minimal covering sets of code
//! needed to understand a specific code entity (function, class, module, etc.).
//! It supports both file-level and entity-level granularity.
//!
//! ## Granularity Modes
//!
//! - **File-level**: Returns whole files based on dependency graph traversal
//! - **Entity-level**: Returns only the specific functions/classes needed,
//!   extracted from their source files
//!
//! Entity-level mode analyzes what symbols a target entity uses and resolves
//! them to their source definitions, returning only the relevant code chunks.

use crate::ast_parser::{AstChunk, AstLanguage, AstParser, EntityLocation, EntityQuery};
use scribe_core::{Result, ScribeError};
use scribe_graph::{DependencyGraph, TraversalDirection};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Granularity level for covering set results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CoveringSetGranularity {
    /// Return whole files (traditional behavior)
    #[default]
    File,
    /// Return only the specific entities (functions, classes) needed
    Entity,
}

/// Options for covering set computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSetOptions {
    /// Include files/entities that the target depends on
    pub include_dependencies: bool,
    /// Include files/entities that depend on the target
    pub include_dependents: bool,
    /// Maximum depth for dependency traversal (None = unlimited)
    pub max_depth: Option<usize>,
    /// Maximum number of files/entities to include (None = unlimited)
    pub max_files: Option<usize>,
    /// Minimum importance score to include (0.0-1.0)
    pub min_importance: Option<f64>,
    /// Granularity level for results
    pub granularity: CoveringSetGranularity,
}

impl Default for CoveringSetOptions {
    fn default() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: None,
            max_files: None,
            min_importance: None,
            granularity: CoveringSetGranularity::default(),
        }
    }
}

impl CoveringSetOptions {
    /// Create default options optimized for understanding a target (file-level)
    pub fn for_understanding() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: None,
            max_files: Some(100),
            min_importance: Some(0.3),
            granularity: CoveringSetGranularity::File,
        }
    }

    /// Create options optimized for impact analysis (file-level)
    pub fn for_impact_analysis() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: true,
            max_depth: Some(2),
            max_files: Some(50),
            min_importance: Some(0.4),
            granularity: CoveringSetGranularity::File,
        }
    }

    /// Create options for minimal covering set (file-level)
    pub fn minimal() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: Some(1),
            max_files: Some(20),
            min_importance: Some(0.5),
            granularity: CoveringSetGranularity::File,
        }
    }

    /// Create options for entity-level covering set (functions/classes only)
    pub fn entity_level() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: Some(3),
            max_files: Some(50),
            min_importance: None,
            granularity: CoveringSetGranularity::Entity,
        }
    }

    /// Create focused entity-level options for understanding a single function/class
    pub fn entity_focused() -> Self {
        Self {
            include_dependencies: true,
            include_dependents: false,
            max_depth: Some(2),
            max_files: Some(30),
            min_importance: None,
            granularity: CoveringSetGranularity::Entity,
        }
    }
}

/// Result of covering set computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSetResult {
    /// The target entity that was located
    pub target_entity: Option<EntityLocation>,
    /// Files included in the covering set (populated for File granularity)
    pub files: Vec<CoveringSetFile>,
    /// Entities included in the covering set (populated for Entity granularity)
    pub entities: Vec<CoveringSetEntity>,
    /// Statistics about the computation
    pub statistics: CoveringSetStatistics,
    /// Explanation of why files/entities were included
    pub inclusion_reasons: HashMap<String, String>,
}

/// Information about an entity (function, class, etc.) in the covering set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringSetEntity {
    /// File path containing this entity
    pub file_path: String,
    /// Name of the entity
    pub name: String,
    /// Type of entity (function, class, method, etc.)
    pub entity_type: String,
    /// The extracted code content
    pub content: String,
    /// Start line (1-indexed)
    pub start_line: usize,
    /// End line (1-indexed)
    pub end_line: usize,
    /// Why this entity was included
    pub reason: InclusionReason,
    /// Distance from target (0 = target, 1 = direct dependency, etc.)
    pub distance: usize,
    /// Names of symbols this entity references (for debugging/inspection)
    pub references: Vec<String>,
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
    /// Relevant line ranges (inclusive, 1-indexed)
    pub line_ranges: Vec<LineRange>,
}

/// Line range information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LineRange {
    pub start_line: usize,
    pub end_line: usize,
}
/// Reason a file was included in the covering set
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InclusionReason {
    /// File contains the target entity
    TargetFile,
    /// File was directly changed in a diff
    ChangedFile,
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
    /// Files in the final covering set (file-level mode)
    pub files_selected: usize,
    /// Files excluded due to limits
    pub files_excluded: usize,
    /// Entities in the final covering set (entity-level mode)
    pub entities_selected: usize,
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

    /// Compute a covering set for a target file or entity
    ///
    /// # Arguments
    /// * `query` - Query specifying file (required) and optionally entity
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
            entities_selected: 0,
            max_depth_reached: 0,
            limits_reached: false,
        };

        // File pattern is required
        let file_pattern = match &query.file_pattern {
            Some(p) => p,
            None => {
                return Err(ScribeError::invalid_operation(
                    "File pattern is required. Use 'file' or 'file:entity' format.",
                    "compute_covering_set",
                ));
            }
        };

        // Find the target file that matches the pattern
        let target_file = self.find_matching_file(file_pattern, file_contents)?;

        if target_file.is_none() {
            return Ok(CoveringSetResult {
                target_entity: None,
                files: Vec::new(),
                entities: Vec::new(),
                statistics,
                inclusion_reasons: HashMap::new(),
            });
        }

        let target_file_path = target_file.unwrap();

        // If entity name is specified, find the specific entity
        let target_entity = if query.name_pattern.is_some() {
            if let Some(content) = file_contents.get(&target_file_path) {
                let entities = self.ast_parser.find_entities(content, &target_file_path, query)?;
                entities.into_iter().next()
            } else {
                None
            }
        } else {
            None
        };

        // Branch based on granularity (only for entity-level when entity is specified)
        if options.granularity == CoveringSetGranularity::Entity && target_entity.is_some() {
            return self.compute_entity_level_covering_set(
                target_entity,
                file_contents,
                graph,
                options,
                statistics,
            );
        }

        // Use target file for file-level covering set
        let target_file = target_file_path;

        // Step 2: Compute file closure using dependency graph
        let seed_files = vec![target_file.clone()];
        let direction = self.get_traversal_direction(options);
        let closure_files = graph.compute_closure(&seed_files, direction, options.max_depth);

        // Step 3: Build covering set with metadata
        let mut covering_set = Vec::new();
        let mut inclusion_reasons = HashMap::new();

        // Determine line ranges for target file
        let target_line_ranges = if let Some(ref entity) = target_entity {
            vec![LineRange {
                start_line: entity.start_line,
                end_line: entity.end_line,
            }]
        } else {
            Vec::new() // Whole file mode - no specific line ranges
        };

        let reason_text = if target_entity.is_some() {
            "Contains the target entity".to_string()
        } else {
            "Target file".to_string()
        };

        // Add target file first
        covering_set.push(CoveringSetFile {
            path: target_file.clone(),
            reason: InclusionReason::TargetFile,
            distance: 0,
            importance: None,
            line_ranges: target_line_ranges,
        });
        inclusion_reasons.insert(target_file.clone(), reason_text);

        // Add dependency/dependent files
        for file in closure_files {
            if file == target_file {
                continue; // Already added
            }

            let (reason, distance) = self.compute_inclusion_info(
                &file,
                &target_file,
                graph,
                options,
            );

            covering_set.push(CoveringSetFile {
                path: file.clone(),
                reason: reason.clone(),
                distance,
                importance: None,
                line_ranges: Vec::new(),
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
            entities: Vec::new(), // Empty for file-level granularity
            statistics,
            inclusion_reasons,
        })
    }

    /// Compute a covering set starting from a set of changed files (git diff)
    ///
    /// This variant is useful for code review or impact analysis scenarios where
    /// you want the minimal set of files that explain or are affected by a diff
    /// without needing to name a specific entity.
    pub fn compute_covering_set_for_files(
        &self,
        changed_files: &[String],
        graph: &DependencyGraph,
        line_map: Option<&HashMap<String, Vec<LineRange>>>,
        options: &CoveringSetOptions,
    ) -> Result<CoveringSetResult> {
        let mut statistics = CoveringSetStatistics {
            files_examined: changed_files.len(),
            files_selected: 0,
            files_excluded: 0,
            entities_selected: 0,
            max_depth_reached: 0,
            limits_reached: false,
        };

        if changed_files.is_empty() {
            return Ok(CoveringSetResult {
                target_entity: None,
                files: Vec::new(),
                entities: Vec::new(),
                statistics,
                inclusion_reasons: HashMap::new(),
            });
        }

        let direction = self.get_traversal_direction(options);
        let closure_files = graph.compute_closure(changed_files, direction, options.max_depth);

        let mut covering_set = Vec::new();
        let mut inclusion_reasons = HashMap::new();

        // Add the changed files as seeds
        for changed in changed_files {
            covering_set.push(CoveringSetFile {
                path: changed.clone(),
                reason: InclusionReason::ChangedFile,
                distance: 0,
                importance: None,
                line_ranges: line_map
                    .and_then(|m| m.get(changed))
                    .cloned()
                    .unwrap_or_default(),
            });
            inclusion_reasons.insert(changed.clone(), "Changed in diff".to_string());
        }

        // Add dependency/dependent files reachable from any changed file
        for file in closure_files {
            if changed_files.contains(&file) {
                continue;
            }

            // Pick a representative changed file to compute distance/reason.
            // We use the first changed file that connects; fallback to the first seed.
            let reference = changed_files
                .iter()
                .find(|target| {
                    graph.contains_edge(target, &file) || graph.contains_edge(&file, target)
                })
                .unwrap_or(&changed_files[0]);

            let (reason, distance) = self.compute_inclusion_info(&file, reference, graph, options);

            covering_set.push(CoveringSetFile {
                path: file.clone(),
                reason: reason.clone(),
                distance,
                importance: None,
                line_ranges: Vec::new(),
            });

            let reason_text = self.format_inclusion_reason(&reason, distance);
            inclusion_reasons.insert(file, reason_text);
        }

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
            target_entity: None,
            files: covering_set,
            entities: Vec::new(),
            statistics,
            inclusion_reasons,
        })
    }

    /// Compute entity-level covering set (returns specific functions/classes, not whole files)
    fn compute_entity_level_covering_set(
        &mut self,
        target_entity: Option<EntityLocation>,
        file_contents: &HashMap<String, String>,
        graph: &DependencyGraph,
        options: &CoveringSetOptions,
        mut statistics: CoveringSetStatistics,
    ) -> Result<CoveringSetResult> {
        let target = target_entity.clone().unwrap();
        let target_file = &target.file_path;

        let mut entities = Vec::new();
        let mut inclusion_reasons = HashMap::new();
        let mut visited_entities: HashSet<String> = HashSet::new();

        // Add target entity first
        let target_id = format!("{}::{}", target.file_path, target.entity_name);
        visited_entities.insert(target_id.clone());

        // Extract references from target entity's content
        let target_references = self.extract_symbol_references(&target.content, &target.file_path)?;

        entities.push(CoveringSetEntity {
            file_path: target.file_path.clone(),
            name: target.entity_name.clone(),
            entity_type: target.entity_type.clone(),
            content: target.content.clone(),
            start_line: target.start_line,
            end_line: target.end_line,
            reason: InclusionReason::TargetFile,
            distance: 0,
            references: target_references.clone(),
        });
        inclusion_reasons.insert(target_id, "Target entity".to_string());

        // Get file-level dependencies from graph
        let direction = self.get_traversal_direction(options);
        let closure_files = graph.compute_closure(&[target_file.clone()], direction, options.max_depth);

        // For each referenced symbol, try to find its definition in dependency files
        let mut pending_refs: Vec<(String, usize)> = target_references
            .iter()
            .map(|r| (r.clone(), 1))
            .collect();

        while let Some((ref_name, distance)) = pending_refs.pop() {
            // Check depth limit
            if let Some(max_depth) = options.max_depth {
                if distance > max_depth {
                    continue;
                }
            }

            // Check entity limit
            if let Some(max_files) = options.max_files {
                if entities.len() >= max_files {
                    statistics.limits_reached = true;
                    break;
                }
            }

            // Search for the referenced entity in dependency files
            for dep_file in &closure_files {
                if let Some(content) = file_contents.get(dep_file) {
                    // Try to find entity definition matching the reference name
                    if let Some(found_entity) = self.find_entity_by_name(content, dep_file, &ref_name)? {
                        let entity_id = format!("{}::{}", found_entity.file_path, found_entity.entity_name);

                        if visited_entities.contains(&entity_id) {
                            continue;
                        }
                        visited_entities.insert(entity_id.clone());

                        // Extract references from this entity too (for transitive deps)
                        let nested_refs = self.extract_symbol_references(&found_entity.content, dep_file)?;

                        let reason = if distance == 1 {
                            InclusionReason::DirectDependency
                        } else {
                            InclusionReason::TransitiveDependency
                        };

                        entities.push(CoveringSetEntity {
                            file_path: found_entity.file_path.clone(),
                            name: found_entity.entity_name.clone(),
                            entity_type: found_entity.entity_type.clone(),
                            content: found_entity.content.clone(),
                            start_line: found_entity.start_line,
                            end_line: found_entity.end_line,
                            reason: reason.clone(),
                            distance,
                            references: nested_refs.clone(),
                        });

                        inclusion_reasons.insert(
                            entity_id,
                            self.format_inclusion_reason(&reason, distance),
                        );

                        // Add nested references to pending (for next depth level)
                        for nested_ref in nested_refs {
                            if !visited_entities.iter().any(|id| id.ends_with(&format!("::{}", nested_ref))) {
                                pending_refs.push((nested_ref, distance + 1));
                            }
                        }

                        break; // Found the entity, move to next reference
                    }
                }
            }
        }

        statistics.entities_selected = entities.len();
        statistics.max_depth_reached = entities.iter().map(|e| e.distance).max().unwrap_or(0);

        Ok(CoveringSetResult {
            target_entity,
            files: Vec::new(), // Empty for entity-level granularity
            entities,
            statistics,
            inclusion_reasons,
        })
    }

    /// Extract symbol references from code content (function/method calls, class instantiations)
    fn extract_symbol_references(&self, content: &str, file_path: &str) -> Result<Vec<String>> {
        let mut references = Vec::new();

        // Determine language from file extension
        let language = Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(AstLanguage::from_extension);

        if language.is_none() {
            return Ok(references);
        }

        // Simple heuristic extraction of identifiers that look like references
        // This is a simplified approach - a full implementation would use tree-sitter
        // to properly identify call expressions and type references

        // Extract potential function calls: identifier followed by (
        let call_pattern = regex::Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
            .map_err(|e| ScribeError::parse(format!("Regex error: {}", e)))?;

        for cap in call_pattern.captures_iter(content) {
            if let Some(name) = cap.get(1) {
                let name_str = name.as_str();
                // Filter out common keywords and builtins
                if !is_common_keyword(name_str) && !references.contains(&name_str.to_string()) {
                    references.push(name_str.to_string());
                }
            }
        }

        // Extract potential class instantiations (new Foo, Foo())
        let class_pattern = regex::Regex::new(r"\bnew\s+([A-Z][a-zA-Z0-9_]*)")
            .map_err(|e| ScribeError::parse(format!("Regex error: {}", e)))?;

        for cap in class_pattern.captures_iter(content) {
            if let Some(name) = cap.get(1) {
                let name_str = name.as_str();
                if !references.contains(&name_str.to_string()) {
                    references.push(name_str.to_string());
                }
            }
        }

        // Extract type annotations (Python type hints, TypeScript types)
        let type_pattern = regex::Regex::new(r":\s*([A-Z][a-zA-Z0-9_]*)")
            .map_err(|e| ScribeError::parse(format!("Regex error: {}", e)))?;

        for cap in type_pattern.captures_iter(content) {
            if let Some(name) = cap.get(1) {
                let name_str = name.as_str();
                if !is_common_type(name_str) && !references.contains(&name_str.to_string()) {
                    references.push(name_str.to_string());
                }
            }
        }

        Ok(references)
    }

    /// Find an entity by name in file content
    fn find_entity_by_name(
        &mut self,
        content: &str,
        file_path: &str,
        name: &str,
    ) -> Result<Option<EntityLocation>> {
        let query = EntityQuery {
            name_pattern: Some(name.to_string()),
            entity_type: None,
            public_only: None,
            exact_match: true,
            file_pattern: None, // Already searching within a specific file
        };

        let entities = self.ast_parser.find_entities(content, file_path, &query)?;
        Ok(entities.into_iter().next())
    }

    /// Find a file that matches the given pattern
    fn find_matching_file(
        &self,
        pattern: &str,
        file_contents: &HashMap<String, String>,
    ) -> Result<Option<String>> {
        let pattern_lower = pattern.to_lowercase();

        // First try exact match
        if file_contents.contains_key(pattern) {
            return Ok(Some(pattern.to_string()));
        }

        // Then try case-insensitive exact match or suffix match
        for file_path in file_contents.keys() {
            let path_lower = file_path.to_lowercase();
            if path_lower == pattern_lower {
                return Ok(Some(file_path.clone()));
            }
            // Match if file path ends with the pattern (e.g., "auth.rs" matches "src/auth.rs")
            if path_lower.ends_with(&pattern_lower) {
                // Ensure it's at a path boundary
                let prefix_len = path_lower.len() - pattern_lower.len();
                if prefix_len == 0
                    || path_lower.as_bytes()[prefix_len - 1] == b'/'
                    || path_lower.as_bytes()[prefix_len - 1] == b'\\'
                {
                    return Ok(Some(file_path.clone()));
                }
            }
        }

        // Finally try substring match (file path contains the pattern)
        for file_path in file_contents.keys() {
            if file_path.to_lowercase().contains(&pattern_lower) {
                return Ok(Some(file_path.clone()));
            }
        }

        Ok(None)
    }

    /// Find the target entity across all files (legacy method for backward compat)
    fn find_target_entity(
        &mut self,
        query: &EntityQuery,
        file_contents: &HashMap<String, String>,
    ) -> Result<Option<EntityLocation>> {
        for (file_path, content) in file_contents {
            // Skip files that don't match the file pattern (if specified)
            if !query.matches_file(file_path) {
                continue;
            }
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
            InclusionReason::ChangedFile => "Changed in diff".to_string(),
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

/// Check if an identifier is a common language keyword (should be filtered from references)
fn is_common_keyword(name: &str) -> bool {
    matches!(
        name,
        // Control flow
        "if" | "else" | "elif" | "for" | "while" | "do" | "switch" | "case" | "break" | "continue" | "return" |
        // Declarations
        "def" | "fn" | "func" | "function" | "class" | "struct" | "enum" | "interface" | "type" | "trait" |
        "let" | "var" | "const" | "static" | "mut" | "pub" | "private" | "public" | "protected" |
        // Other keywords
        "import" | "from" | "as" | "in" | "is" | "not" | "and" | "or" | "try" | "catch" | "except" |
        "finally" | "throw" | "raise" | "with" | "async" | "await" | "yield" |
        // Common builtins
        "print" | "println" | "printf" | "fmt" | "len" | "range" | "enumerate" | "zip" | "map" | "filter" |
        "str" | "int" | "float" | "bool" | "list" | "dict" | "set" | "tuple" |
        "true" | "false" | "True" | "False" | "null" | "None" | "nil" | "undefined" |
        "self" | "this" | "super" | "Self"
    )
}

/// Check if a type name is a common built-in type (should be filtered from references)
fn is_common_type(name: &str) -> bool {
    matches!(
        name,
        "String" | "str" | "Int" | "Integer" | "Float" | "Double" | "Bool" | "Boolean" |
        "List" | "Array" | "Vec" | "Dict" | "Map" | "HashMap" | "Set" | "HashSet" |
        "Option" | "Result" | "Some" | "None" | "Ok" | "Err" |
        "Any" | "Object" | "Void" | "Unit" | "Never" |
        "Promise" | "Future" | "Task" | "Tuple"
    )
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

        let reason = computer.format_inclusion_reason(&InclusionReason::ChangedFile, 0);
        assert_eq!(reason, "Changed in diff");

        let reason = computer.format_inclusion_reason(&InclusionReason::DirectDependency, 1);
        assert_eq!(reason, "Direct dependency of target");

        let reason = computer.format_inclusion_reason(&InclusionReason::TransitiveDependency, 3);
        assert!(reason.contains("distance: 3"));
    }

    #[test]
    fn test_covering_set_for_changed_files() {
        let computer = CoveringSetComputer::new().unwrap();
        let graph = DependencyGraph::new();

        let changed = vec!["src/lib.rs".to_string(), "src/main.rs".to_string()];
        let result = computer
            .compute_covering_set_for_files(
                &changed,
                &graph,
                None,
                &CoveringSetOptions::default(),
            )
            .unwrap();

        assert!(result.target_entity.is_none());
        assert_eq!(result.files.len(), 2);
        assert!(result
            .files
            .iter()
            .all(|f| f.reason == InclusionReason::ChangedFile));
    }

    #[test]
    fn test_entity_level_options() {
        let opts = CoveringSetOptions::entity_level();
        assert_eq!(opts.granularity, CoveringSetGranularity::Entity);
        assert!(opts.include_dependencies);
        assert_eq!(opts.max_depth, Some(3));

        let focused = CoveringSetOptions::entity_focused();
        assert_eq!(focused.granularity, CoveringSetGranularity::Entity);
        assert_eq!(focused.max_depth, Some(2));
    }

    #[test]
    fn test_granularity_default() {
        assert_eq!(CoveringSetGranularity::default(), CoveringSetGranularity::File);

        let opts = CoveringSetOptions::default();
        assert_eq!(opts.granularity, CoveringSetGranularity::File);
    }

    #[test]
    fn test_entity_level_covering_set() {
        let mut computer = CoveringSetComputer::new().unwrap();
        let graph = DependencyGraph::new();

        // Create test file contents with a function that calls another
        let mut file_contents = HashMap::new();
        file_contents.insert(
            "src/lib.py".to_string(),
            r#"
def main():
    result = helper_func(42)
    return result

def helper_func(x):
    return x * 2
"#.to_string(),
        );

        // Use file:entity format
        let query = EntityQuery::for_file_entity("src/lib.py", "main");

        let result = computer
            .compute_covering_set(
                &query,
                &file_contents,
                &graph,
                &CoveringSetOptions::entity_level(),
            )
            .unwrap();

        // Should find the target entity
        assert!(result.target_entity.is_some());
        let target = result.target_entity.as_ref().unwrap();
        assert_eq!(target.entity_name, "main");

        // In entity-level mode, files should be empty
        assert!(result.files.is_empty());

        // Should have at least the target entity
        assert!(!result.entities.is_empty());
        assert!(result.entities.iter().any(|e| e.name == "main"));

        // Statistics should reflect entity-level results
        assert!(result.statistics.entities_selected > 0);
    }

    #[test]
    fn test_file_only_covering_set() {
        let mut computer = CoveringSetComputer::new().unwrap();
        let graph = DependencyGraph::new();

        let mut file_contents = HashMap::new();
        file_contents.insert(
            "src/lib.py".to_string(),
            "def main(): pass".to_string(),
        );

        // Use file-only format (no entity)
        let query = EntityQuery::for_file("src/lib.py");

        let result = computer
            .compute_covering_set(
                &query,
                &file_contents,
                &graph,
                &CoveringSetOptions::default(),
            )
            .unwrap();

        // Should find the file but no specific entity
        assert!(result.target_entity.is_none());
        assert!(!result.files.is_empty());
        assert_eq!(result.files[0].path, "src/lib.py");
        assert_eq!(result.files[0].reason, InclusionReason::TargetFile);
    }

    #[test]
    fn test_file_pattern_requires_file() {
        let mut computer = CoveringSetComputer::new().unwrap();
        let graph = DependencyGraph::new();
        let file_contents = HashMap::new();

        // Query with no file pattern should error
        let query = EntityQuery::by_name("main");

        let result = computer.compute_covering_set(
            &query,
            &file_contents,
            &graph,
            &CoveringSetOptions::default(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_is_common_keyword() {
        assert!(super::is_common_keyword("if"));
        assert!(super::is_common_keyword("def"));
        assert!(super::is_common_keyword("print"));
        assert!(super::is_common_keyword("self"));
        assert!(!super::is_common_keyword("my_function"));
        assert!(!super::is_common_keyword("CustomClass"));
    }

    #[test]
    fn test_is_common_type() {
        assert!(super::is_common_type("String"));
        assert!(super::is_common_type("Option"));
        assert!(super::is_common_type("Vec"));
        assert!(!super::is_common_type("MyCustomType"));
        assert!(!super::is_common_type("UserService"));
    }
}

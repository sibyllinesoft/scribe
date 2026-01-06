//! Type definitions for covering set computation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ast::ast_parser::EntityLocation;

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

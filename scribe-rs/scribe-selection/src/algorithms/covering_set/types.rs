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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covering_set_granularity_default() {
        let granularity = CoveringSetGranularity::default();
        assert_eq!(granularity, CoveringSetGranularity::File);
    }

    #[test]
    fn test_covering_set_granularity_variants() {
        let file = CoveringSetGranularity::File;
        let entity = CoveringSetGranularity::Entity;

        assert_eq!(file, CoveringSetGranularity::File);
        assert_eq!(entity, CoveringSetGranularity::Entity);
        assert_ne!(file, entity);
    }

    #[test]
    fn test_covering_set_granularity_copy() {
        let granularity = CoveringSetGranularity::Entity;
        let copied = granularity;
        assert_eq!(granularity, copied);
    }

    #[test]
    fn test_covering_set_options_default() {
        let options = CoveringSetOptions::default();
        assert!(options.include_dependencies);
        assert!(!options.include_dependents);
        assert!(options.max_depth.is_none());
        assert!(options.max_files.is_none());
        assert!(options.min_importance.is_none());
        assert_eq!(options.granularity, CoveringSetGranularity::File);
    }

    #[test]
    fn test_covering_set_options_for_understanding() {
        let options = CoveringSetOptions::for_understanding();
        assert!(options.include_dependencies);
        assert!(!options.include_dependents);
        assert!(options.max_depth.is_none());
        assert_eq!(options.max_files, Some(100));
        assert_eq!(options.min_importance, Some(0.3));
        assert_eq!(options.granularity, CoveringSetGranularity::File);
    }

    #[test]
    fn test_covering_set_options_for_impact_analysis() {
        let options = CoveringSetOptions::for_impact_analysis();
        assert!(options.include_dependencies);
        assert!(options.include_dependents);
        assert_eq!(options.max_depth, Some(2));
        assert_eq!(options.max_files, Some(50));
        assert_eq!(options.min_importance, Some(0.4));
        assert_eq!(options.granularity, CoveringSetGranularity::File);
    }

    #[test]
    fn test_covering_set_options_minimal() {
        let options = CoveringSetOptions::minimal();
        assert!(options.include_dependencies);
        assert!(!options.include_dependents);
        assert_eq!(options.max_depth, Some(1));
        assert_eq!(options.max_files, Some(20));
        assert_eq!(options.min_importance, Some(0.5));
        assert_eq!(options.granularity, CoveringSetGranularity::File);
    }

    #[test]
    fn test_covering_set_options_entity_level() {
        let options = CoveringSetOptions::entity_level();
        assert!(options.include_dependencies);
        assert!(!options.include_dependents);
        assert_eq!(options.max_depth, Some(3));
        assert_eq!(options.max_files, Some(50));
        assert!(options.min_importance.is_none());
        assert_eq!(options.granularity, CoveringSetGranularity::Entity);
    }

    #[test]
    fn test_covering_set_options_entity_focused() {
        let options = CoveringSetOptions::entity_focused();
        assert!(options.include_dependencies);
        assert!(!options.include_dependents);
        assert_eq!(options.max_depth, Some(2));
        assert_eq!(options.max_files, Some(30));
        assert!(options.min_importance.is_none());
        assert_eq!(options.granularity, CoveringSetGranularity::Entity);
    }

    #[test]
    fn test_covering_set_options_clone() {
        let options = CoveringSetOptions::for_understanding();
        let cloned = options.clone();
        assert_eq!(options.include_dependencies, cloned.include_dependencies);
        assert_eq!(options.max_files, cloned.max_files);
    }

    #[test]
    fn test_covering_set_options_serialize() {
        let options = CoveringSetOptions::default();
        let json = serde_json::to_string(&options).unwrap();
        let deserialized: CoveringSetOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(options.include_dependencies, deserialized.include_dependencies);
    }

    #[test]
    fn test_inclusion_reason_variants() {
        let reasons = [
            InclusionReason::TargetFile,
            InclusionReason::ChangedFile,
            InclusionReason::DirectDependency,
            InclusionReason::TransitiveDependency,
            InclusionReason::DirectDependent,
            InclusionReason::TransitiveDependent,
        ];

        // All variants should be distinct
        for (i, r1) in reasons.iter().enumerate() {
            for (j, r2) in reasons.iter().enumerate() {
                if i == j {
                    assert_eq!(r1, r2);
                } else {
                    assert_ne!(r1, r2);
                }
            }
        }
    }

    #[test]
    fn test_inclusion_reason_clone() {
        let reason = InclusionReason::DirectDependency;
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
    }

    #[test]
    fn test_inclusion_reason_serialize() {
        let reason = InclusionReason::TransitiveDependency;
        let json = serde_json::to_string(&reason).unwrap();
        let deserialized: InclusionReason = serde_json::from_str(&json).unwrap();
        assert_eq!(reason, deserialized);
    }

    #[test]
    fn test_line_range_equality() {
        let range1 = LineRange { start_line: 1, end_line: 10 };
        let range2 = LineRange { start_line: 1, end_line: 10 };
        let range3 = LineRange { start_line: 5, end_line: 15 };

        assert_eq!(range1, range2);
        assert_ne!(range1, range3);
    }

    #[test]
    fn test_line_range_clone() {
        let range = LineRange { start_line: 100, end_line: 200 };
        let cloned = range.clone();
        assert_eq!(range.start_line, cloned.start_line);
        assert_eq!(range.end_line, cloned.end_line);
    }

    #[test]
    fn test_line_range_serialize() {
        let range = LineRange { start_line: 42, end_line: 84 };
        let json = serde_json::to_string(&range).unwrap();
        assert!(json.contains("42"));
        assert!(json.contains("84"));
        let deserialized: LineRange = serde_json::from_str(&json).unwrap();
        assert_eq!(range, deserialized);
    }

    #[test]
    fn test_covering_set_file_creation() {
        let file = CoveringSetFile {
            path: "src/main.rs".to_string(),
            reason: InclusionReason::TargetFile,
            distance: 0,
            importance: Some(0.95),
            line_ranges: vec![LineRange { start_line: 1, end_line: 100 }],
        };

        assert_eq!(file.path, "src/main.rs");
        assert_eq!(file.reason, InclusionReason::TargetFile);
        assert_eq!(file.distance, 0);
        assert_eq!(file.importance, Some(0.95));
        assert_eq!(file.line_ranges.len(), 1);
    }

    #[test]
    fn test_covering_set_file_clone() {
        let file = CoveringSetFile {
            path: "lib.rs".to_string(),
            reason: InclusionReason::DirectDependency,
            distance: 1,
            importance: None,
            line_ranges: vec![],
        };

        let cloned = file.clone();
        assert_eq!(file.path, cloned.path);
        assert_eq!(file.distance, cloned.distance);
    }

    #[test]
    fn test_covering_set_entity_creation() {
        let entity = CoveringSetEntity {
            file_path: "src/utils.rs".to_string(),
            name: "helper_function".to_string(),
            entity_type: "function".to_string(),
            content: "fn helper_function() {}".to_string(),
            start_line: 10,
            end_line: 15,
            reason: InclusionReason::DirectDependency,
            distance: 1,
            references: vec!["other_func".to_string()],
        };

        assert_eq!(entity.name, "helper_function");
        assert_eq!(entity.entity_type, "function");
        assert_eq!(entity.start_line, 10);
        assert_eq!(entity.end_line, 15);
        assert_eq!(entity.references.len(), 1);
    }

    #[test]
    fn test_covering_set_entity_clone() {
        let entity = CoveringSetEntity {
            file_path: "test.rs".to_string(),
            name: "test_fn".to_string(),
            entity_type: "function".to_string(),
            content: "fn test_fn() {}".to_string(),
            start_line: 1,
            end_line: 1,
            reason: InclusionReason::TargetFile,
            distance: 0,
            references: vec![],
        };

        let cloned = entity.clone();
        assert_eq!(entity.name, cloned.name);
        assert_eq!(entity.file_path, cloned.file_path);
    }

    #[test]
    fn test_covering_set_statistics_creation() {
        let stats = CoveringSetStatistics {
            files_examined: 100,
            files_selected: 15,
            files_excluded: 85,
            entities_selected: 50,
            max_depth_reached: 3,
            limits_reached: true,
        };

        assert_eq!(stats.files_examined, 100);
        assert_eq!(stats.files_selected, 15);
        assert_eq!(stats.files_excluded, 85);
        assert_eq!(stats.entities_selected, 50);
        assert_eq!(stats.max_depth_reached, 3);
        assert!(stats.limits_reached);
    }

    #[test]
    fn test_covering_set_statistics_clone() {
        let stats = CoveringSetStatistics {
            files_examined: 50,
            files_selected: 10,
            files_excluded: 40,
            entities_selected: 0,
            max_depth_reached: 2,
            limits_reached: false,
        };

        let cloned = stats.clone();
        assert_eq!(stats.files_examined, cloned.files_examined);
        assert_eq!(stats.limits_reached, cloned.limits_reached);
    }

    #[test]
    fn test_covering_set_result_creation() {
        let result = CoveringSetResult {
            target_entity: None,
            files: vec![
                CoveringSetFile {
                    path: "main.rs".to_string(),
                    reason: InclusionReason::TargetFile,
                    distance: 0,
                    importance: Some(1.0),
                    line_ranges: vec![],
                }
            ],
            entities: vec![],
            statistics: CoveringSetStatistics {
                files_examined: 10,
                files_selected: 1,
                files_excluded: 9,
                entities_selected: 0,
                max_depth_reached: 0,
                limits_reached: false,
            },
            inclusion_reasons: HashMap::new(),
        };

        assert!(result.target_entity.is_none());
        assert_eq!(result.files.len(), 1);
        assert!(result.entities.is_empty());
        assert_eq!(result.statistics.files_selected, 1);
    }

    #[test]
    fn test_covering_set_result_clone() {
        let result = CoveringSetResult {
            target_entity: None,
            files: vec![],
            entities: vec![],
            statistics: CoveringSetStatistics {
                files_examined: 0,
                files_selected: 0,
                files_excluded: 0,
                entities_selected: 0,
                max_depth_reached: 0,
                limits_reached: false,
            },
            inclusion_reasons: HashMap::new(),
        };

        let cloned = result.clone();
        assert_eq!(result.files.len(), cloned.files.len());
    }

    #[test]
    fn test_covering_set_result_with_inclusion_reasons() {
        let mut reasons = HashMap::new();
        reasons.insert("main.rs".to_string(), "Target file".to_string());
        reasons.insert("lib.rs".to_string(), "Direct dependency".to_string());

        let result = CoveringSetResult {
            target_entity: None,
            files: vec![],
            entities: vec![],
            statistics: CoveringSetStatistics {
                files_examined: 2,
                files_selected: 2,
                files_excluded: 0,
                entities_selected: 0,
                max_depth_reached: 1,
                limits_reached: false,
            },
            inclusion_reasons: reasons,
        };

        assert_eq!(result.inclusion_reasons.len(), 2);
        assert!(result.inclusion_reasons.contains_key("main.rs"));
        assert!(result.inclusion_reasons.contains_key("lib.rs"));
    }

    #[test]
    fn test_covering_set_granularity_debug() {
        let file = CoveringSetGranularity::File;
        let entity = CoveringSetGranularity::Entity;
        let file_debug = format!("{:?}", file);
        let entity_debug = format!("{:?}", entity);
        assert!(file_debug.contains("File"));
        assert!(entity_debug.contains("Entity"));
    }

    #[test]
    fn test_inclusion_reason_debug() {
        let reason = InclusionReason::TransitiveDependency;
        let debug = format!("{:?}", reason);
        assert!(debug.contains("TransitiveDependency"));
    }
}

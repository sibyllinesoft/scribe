//! Type definitions for context positioning optimization.

use serde::{Deserialize, Serialize};

use crate::io::streaming::FileMetadata;

/// Configuration for context positioning optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPositioningConfig {
    /// Enable context positioning optimization
    pub enable_positioning: bool,

    /// Percentage of context for HEAD positioning (query-relevant, high centrality)
    pub head_percentage: f64,

    /// Percentage of context for TAIL positioning (core functionality)
    pub tail_percentage: f64,

    /// Weight for centrality in positioning decisions
    pub centrality_weight: f64,

    /// Weight for file relatedness in grouping decisions
    pub relatedness_weight: f64,

    /// Weight for query relevance in HEAD positioning
    pub query_relevance_weight: f64,

    /// Auto-exclude test files from selection (focuses on code and docs only)
    pub auto_exclude_tests: bool,
}

impl Default for ContextPositioningConfig {
    fn default() -> Self {
        Self {
            enable_positioning: true,
            head_percentage: 0.20,
            tail_percentage: 0.20,
            centrality_weight: 0.4,
            relatedness_weight: 0.3,
            query_relevance_weight: 0.3,
            auto_exclude_tests: false,
        }
    }
}

/// Centrality scores for files in the codebase
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CentralityScores {
    /// Betweenness centrality: files connecting different parts
    pub betweenness: f64,

    /// PageRank centrality: heavily referenced files
    pub pagerank: f64,

    /// Degree centrality: files with many connections
    pub degree: f64,

    /// Combined centrality score
    pub combined: f64,
}

/// File with centrality and positioning metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWithCentrality {
    pub metadata: FileMetadata,
    pub centrality: CentralityScores,
    pub query_relevance: f64,
    pub relatedness_group: String,
}

/// Three-tier context positioning structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPositioning {
    /// HEAD: Query-specific high centrality files (first ~20%)
    pub head_files: Vec<FileWithCentrality>,

    /// MIDDLE: Low centrality supporting files (~60%)
    pub middle_files: Vec<FileWithCentrality>,

    /// TAIL: Core functionality, high centrality (~20%)
    pub tail_files: Vec<FileWithCentrality>,
}

/// Result of context positioning with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionedSelection {
    pub positioning: ContextPositioning,
    pub total_tokens: usize,
    pub positioning_reasoning: String,
}

/// Context positioning optimizer
pub struct ContextPositioner {
    pub(crate) config: ContextPositioningConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_positioning_config_default() {
        let config = ContextPositioningConfig::default();
        assert!(config.enable_positioning);
        assert!((config.head_percentage - 0.20).abs() < 0.001);
        assert!((config.tail_percentage - 0.20).abs() < 0.001);
        assert!((config.centrality_weight - 0.4).abs() < 0.001);
        assert!((config.relatedness_weight - 0.3).abs() < 0.001);
        assert!((config.query_relevance_weight - 0.3).abs() < 0.001);
        assert!(!config.auto_exclude_tests);
    }

    #[test]
    fn test_context_positioning_config_custom() {
        let config = ContextPositioningConfig {
            enable_positioning: false,
            head_percentage: 0.30,
            tail_percentage: 0.10,
            centrality_weight: 0.5,
            relatedness_weight: 0.25,
            query_relevance_weight: 0.25,
            auto_exclude_tests: true,
        };

        assert!(!config.enable_positioning);
        assert!((config.head_percentage - 0.30).abs() < 0.001);
        assert!((config.tail_percentage - 0.10).abs() < 0.001);
        assert!(config.auto_exclude_tests);
    }

    #[test]
    fn test_context_positioning_config_clone() {
        let config = ContextPositioningConfig::default();
        let cloned = config.clone();
        assert_eq!(config.enable_positioning, cloned.enable_positioning);
        assert!((config.head_percentage - cloned.head_percentage).abs() < 0.001);
    }

    #[test]
    fn test_context_positioning_config_serialize() {
        let config = ContextPositioningConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("enable_positioning"));
        assert!(json.contains("head_percentage"));
        let deserialized: ContextPositioningConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.enable_positioning, deserialized.enable_positioning);
    }

    #[test]
    fn test_context_positioning_config_debug() {
        let config = ContextPositioningConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ContextPositioningConfig"));
        assert!(debug.contains("enable_positioning"));
    }

    #[test]
    fn test_centrality_scores_default() {
        let scores = CentralityScores::default();
        assert_eq!(scores.betweenness, 0.0);
        assert_eq!(scores.pagerank, 0.0);
        assert_eq!(scores.degree, 0.0);
        assert_eq!(scores.combined, 0.0);
    }

    #[test]
    fn test_centrality_scores_custom() {
        let scores = CentralityScores {
            betweenness: 0.5,
            pagerank: 0.8,
            degree: 0.6,
            combined: 0.7,
        };

        assert!((scores.betweenness - 0.5).abs() < 0.001);
        assert!((scores.pagerank - 0.8).abs() < 0.001);
        assert!((scores.degree - 0.6).abs() < 0.001);
        assert!((scores.combined - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_centrality_scores_clone() {
        let scores = CentralityScores {
            betweenness: 0.3,
            pagerank: 0.5,
            degree: 0.4,
            combined: 0.45,
        };

        let cloned = scores.clone();
        assert!((scores.betweenness - cloned.betweenness).abs() < 0.001);
        assert!((scores.pagerank - cloned.pagerank).abs() < 0.001);
    }

    #[test]
    fn test_centrality_scores_serialize() {
        let scores = CentralityScores {
            betweenness: 0.25,
            pagerank: 0.75,
            degree: 0.5,
            combined: 0.6,
        };

        let json = serde_json::to_string(&scores).unwrap();
        assert!(json.contains("betweenness"));
        assert!(json.contains("pagerank"));
        let deserialized: CentralityScores = serde_json::from_str(&json).unwrap();
        assert!((scores.betweenness - deserialized.betweenness).abs() < 0.001);
    }

    #[test]
    fn test_centrality_scores_debug() {
        let scores = CentralityScores::default();
        let debug = format!("{:?}", scores);
        assert!(debug.contains("CentralityScores"));
        assert!(debug.contains("betweenness"));
    }

    #[test]
    fn test_context_positioning_creation() {
        let positioning = ContextPositioning {
            head_files: vec![],
            middle_files: vec![],
            tail_files: vec![],
        };

        assert!(positioning.head_files.is_empty());
        assert!(positioning.middle_files.is_empty());
        assert!(positioning.tail_files.is_empty());
    }

    #[test]
    fn test_context_positioning_clone() {
        let positioning = ContextPositioning {
            head_files: vec![],
            middle_files: vec![],
            tail_files: vec![],
        };

        let cloned = positioning.clone();
        assert_eq!(positioning.head_files.len(), cloned.head_files.len());
    }

    #[test]
    fn test_context_positioning_serialize() {
        let positioning = ContextPositioning {
            head_files: vec![],
            middle_files: vec![],
            tail_files: vec![],
        };

        let json = serde_json::to_string(&positioning).unwrap();
        assert!(json.contains("head_files"));
        assert!(json.contains("middle_files"));
        assert!(json.contains("tail_files"));
    }

    #[test]
    fn test_context_positioning_debug() {
        let positioning = ContextPositioning {
            head_files: vec![],
            middle_files: vec![],
            tail_files: vec![],
        };

        let debug = format!("{:?}", positioning);
        assert!(debug.contains("ContextPositioning"));
    }

    #[test]
    fn test_positioned_selection_creation() {
        let selection = PositionedSelection {
            positioning: ContextPositioning {
                head_files: vec![],
                middle_files: vec![],
                tail_files: vec![],
            },
            total_tokens: 5000,
            positioning_reasoning: "Test reasoning".to_string(),
        };

        assert_eq!(selection.total_tokens, 5000);
        assert_eq!(selection.positioning_reasoning, "Test reasoning");
    }

    #[test]
    fn test_positioned_selection_clone() {
        let selection = PositionedSelection {
            positioning: ContextPositioning {
                head_files: vec![],
                middle_files: vec![],
                tail_files: vec![],
            },
            total_tokens: 10000,
            positioning_reasoning: "Cloned selection".to_string(),
        };

        let cloned = selection.clone();
        assert_eq!(selection.total_tokens, cloned.total_tokens);
        assert_eq!(
            selection.positioning_reasoning,
            cloned.positioning_reasoning
        );
    }

    #[test]
    fn test_positioned_selection_serialize() {
        let selection = PositionedSelection {
            positioning: ContextPositioning {
                head_files: vec![],
                middle_files: vec![],
                tail_files: vec![],
            },
            total_tokens: 7500,
            positioning_reasoning: "Serialization test".to_string(),
        };

        let json = serde_json::to_string(&selection).unwrap();
        assert!(json.contains("total_tokens"));
        assert!(json.contains("7500"));
        assert!(json.contains("positioning_reasoning"));
    }

    #[test]
    fn test_positioned_selection_debug() {
        let selection = PositionedSelection {
            positioning: ContextPositioning {
                head_files: vec![],
                middle_files: vec![],
                tail_files: vec![],
            },
            total_tokens: 0,
            positioning_reasoning: String::new(),
        };

        let debug = format!("{:?}", selection);
        assert!(debug.contains("PositionedSelection"));
    }

    #[test]
    fn test_context_positioner_creation() {
        let config = ContextPositioningConfig::default();
        let positioner = ContextPositioner {
            config: config.clone(),
        };
        assert_eq!(
            positioner.config.enable_positioning,
            config.enable_positioning
        );
    }
}

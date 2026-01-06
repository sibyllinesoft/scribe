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

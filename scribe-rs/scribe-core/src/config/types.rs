//! Type definitions for output and feature configuration.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Output format options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum OutputFormat {
    Json,
    JsonLines,
    Csv,
    Table,
    Summary,
}

/// Output format configuration
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub struct OutputConfig {
    /// Output format
    pub format: OutputFormat,

    /// Whether to include file content in output
    pub include_content: bool,

    /// Whether to include detailed scores breakdown
    pub include_score_breakdown: bool,

    /// Whether to include repository statistics
    pub include_repo_stats: bool,

    /// Whether to sort results by score
    pub sort_by_score: bool,

    /// Pretty print JSON output
    pub pretty_json: bool,

    /// Custom output fields to include
    pub custom_fields: Vec<String>,

    /// Optional default output path
    pub file_path: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            include_content: false,
            include_score_breakdown: true,
            include_repo_stats: true,
            sort_by_score: true,
            pretty_json: true,
            custom_fields: vec![],
            file_path: None,
        }
    }
}

impl OutputConfig {
    /// Validates output configuration settings.
    ///
    /// Currently no validation constraints - all combinations are valid.
    pub(crate) fn validate(&self) -> Result<()> {
        // Currently no validation needed
        Ok(())
    }
}

/// Feature flags for experimental features
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub struct FeatureFlags {
    /// Enable PageRank centrality computation
    pub centrality_enabled: bool,

    /// Enable entrypoint detection
    pub entrypoint_detection: bool,

    /// Enable examples/usage analysis
    pub examples_analysis: bool,

    /// Enable semantic analysis (if available)
    pub semantic_analysis: bool,

    /// Enable machine learning features
    pub ml_features: bool,

    /// Enable experimental scoring algorithms
    pub experimental_scoring: bool,

    /// Enable scaling optimizations for large repositories
    pub scaling_enabled: bool,

    /// Automatically exclude test files from selection
    pub auto_exclude_tests: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            centrality_enabled: false,
            entrypoint_detection: false,
            examples_analysis: false,
            semantic_analysis: false,
            ml_features: false,
            experimental_scoring: false,
            scaling_enabled: false,
            auto_exclude_tests: false,
        }
    }
}

impl FeatureFlags {
    /// Validates feature flags configuration.
    ///
    /// Currently no validation constraints - all combinations are valid.
    pub(crate) fn validate(&self) -> Result<()> {
        // Currently no validation needed
        Ok(())
    }

    /// Check if any V2 features are enabled
    pub fn has_v2_features(&self) -> bool {
        self.centrality_enabled || self.entrypoint_detection || self.examples_analysis
    }

    /// Get list of enabled feature names
    pub fn enabled_features(&self) -> Vec<&'static str> {
        let mut features = Vec::new();

        if self.centrality_enabled {
            features.push("centrality");
        }
        if self.entrypoint_detection {
            features.push("entrypoint_detection");
        }
        if self.examples_analysis {
            features.push("examples_analysis");
        }
        if self.semantic_analysis {
            features.push("semantic_analysis");
        }
        if self.ml_features {
            features.push("ml_features");
        }
        if self.experimental_scoring {
            features.push("experimental_scoring");
        }
        if self.scaling_enabled {
            features.push("scaling");
        }

        features
    }
}

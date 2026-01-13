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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_variants() {
        assert_eq!(OutputFormat::Json, OutputFormat::Json);
        assert_eq!(OutputFormat::JsonLines, OutputFormat::JsonLines);
        assert_eq!(OutputFormat::Csv, OutputFormat::Csv);
        assert_eq!(OutputFormat::Table, OutputFormat::Table);
        assert_eq!(OutputFormat::Summary, OutputFormat::Summary);

        assert_ne!(OutputFormat::Json, OutputFormat::Csv);
        assert_ne!(OutputFormat::Table, OutputFormat::Summary);
    }

    #[test]
    fn test_output_format_clone() {
        let format = OutputFormat::Json;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_output_format_debug() {
        let json_debug = format!("{:?}", OutputFormat::Json);
        assert_eq!(json_debug, "Json");

        let csv_debug = format!("{:?}", OutputFormat::Csv);
        assert_eq!(csv_debug, "Csv");
    }

    #[test]
    fn test_output_format_serialize() {
        let json = serde_json::to_string(&OutputFormat::Json).unwrap();
        assert_eq!(json, "\"Json\"");

        let table = serde_json::to_string(&OutputFormat::Table).unwrap();
        assert_eq!(table, "\"Table\"");

        let deserialized: OutputFormat = serde_json::from_str("\"Csv\"").unwrap();
        assert_eq!(deserialized, OutputFormat::Csv);
    }

    #[test]
    fn test_output_config_default() {
        let config = OutputConfig::default();

        assert_eq!(config.format, OutputFormat::Json);
        assert!(!config.include_content);
        assert!(config.include_score_breakdown);
        assert!(config.include_repo_stats);
        assert!(config.sort_by_score);
        assert!(config.pretty_json);
        assert!(config.custom_fields.is_empty());
        assert!(config.file_path.is_none());
    }

    #[test]
    fn test_output_config_custom() {
        let config = OutputConfig {
            format: OutputFormat::Csv,
            include_content: true,
            include_score_breakdown: false,
            include_repo_stats: false,
            sort_by_score: false,
            pretty_json: false,
            custom_fields: vec!["field1".to_string(), "field2".to_string()],
            file_path: Some("/tmp/output.csv".to_string()),
        };

        assert_eq!(config.format, OutputFormat::Csv);
        assert!(config.include_content);
        assert!(!config.include_score_breakdown);
        assert!(!config.include_repo_stats);
        assert!(!config.sort_by_score);
        assert!(!config.pretty_json);
        assert_eq!(config.custom_fields.len(), 2);
        assert_eq!(config.file_path, Some("/tmp/output.csv".to_string()));
    }

    #[test]
    fn test_output_config_clone() {
        let config = OutputConfig::default();
        let cloned = config.clone();

        assert_eq!(config.format, cloned.format);
        assert_eq!(config.include_content, cloned.include_content);
        assert_eq!(config.pretty_json, cloned.pretty_json);
    }

    #[test]
    fn test_output_config_debug() {
        let config = OutputConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("OutputConfig"));
        assert!(debug_str.contains("format"));
        assert!(debug_str.contains("Json"));
    }

    #[test]
    fn test_output_config_serialize() {
        let config = OutputConfig::default();
        let json = serde_json::to_string(&config).unwrap();

        assert!(json.contains("format"));
        assert!(json.contains("Json"));
        assert!(json.contains("include_content"));

        let deserialized: OutputConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.format, config.format);
    }

    #[test]
    fn test_output_config_validate() {
        let config = OutputConfig::default();
        assert!(config.validate().is_ok());

        let custom_config = OutputConfig {
            format: OutputFormat::Table,
            include_content: true,
            include_score_breakdown: true,
            include_repo_stats: true,
            sort_by_score: true,
            pretty_json: false,
            custom_fields: vec!["a".to_string(), "b".to_string()],
            file_path: Some("/path".to_string()),
        };
        assert!(custom_config.validate().is_ok());
    }

    #[test]
    fn test_feature_flags_default() {
        let flags = FeatureFlags::default();

        assert!(!flags.centrality_enabled);
        assert!(!flags.entrypoint_detection);
        assert!(!flags.examples_analysis);
        assert!(!flags.semantic_analysis);
        assert!(!flags.ml_features);
        assert!(!flags.experimental_scoring);
        assert!(!flags.scaling_enabled);
        assert!(!flags.auto_exclude_tests);
    }

    #[test]
    fn test_feature_flags_custom() {
        let flags = FeatureFlags {
            centrality_enabled: true,
            entrypoint_detection: true,
            examples_analysis: true,
            semantic_analysis: false,
            ml_features: false,
            experimental_scoring: false,
            scaling_enabled: true,
            auto_exclude_tests: true,
        };

        assert!(flags.centrality_enabled);
        assert!(flags.entrypoint_detection);
        assert!(flags.examples_analysis);
        assert!(!flags.semantic_analysis);
        assert!(!flags.ml_features);
        assert!(!flags.experimental_scoring);
        assert!(flags.scaling_enabled);
        assert!(flags.auto_exclude_tests);
    }

    #[test]
    fn test_feature_flags_clone() {
        let flags = FeatureFlags {
            centrality_enabled: true,
            ..FeatureFlags::default()
        };

        let cloned = flags.clone();
        assert_eq!(flags.centrality_enabled, cloned.centrality_enabled);
        assert_eq!(flags.ml_features, cloned.ml_features);
    }

    #[test]
    fn test_feature_flags_debug() {
        let flags = FeatureFlags::default();
        let debug_str = format!("{:?}", flags);

        assert!(debug_str.contains("FeatureFlags"));
        assert!(debug_str.contains("centrality_enabled"));
    }

    #[test]
    fn test_feature_flags_serialize() {
        let flags = FeatureFlags::default();
        let json = serde_json::to_string(&flags).unwrap();

        assert!(json.contains("centrality_enabled"));
        assert!(json.contains("false"));

        let deserialized: FeatureFlags = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.centrality_enabled, flags.centrality_enabled);
    }

    #[test]
    fn test_feature_flags_validate() {
        let flags = FeatureFlags::default();
        assert!(flags.validate().is_ok());

        let all_enabled = FeatureFlags {
            centrality_enabled: true,
            entrypoint_detection: true,
            examples_analysis: true,
            semantic_analysis: true,
            ml_features: true,
            experimental_scoring: true,
            scaling_enabled: true,
            auto_exclude_tests: true,
        };
        assert!(all_enabled.validate().is_ok());
    }

    #[test]
    fn test_feature_flags_has_v2_features_none() {
        let flags = FeatureFlags::default();
        assert!(!flags.has_v2_features());
    }

    #[test]
    fn test_feature_flags_has_v2_features_centrality() {
        let flags = FeatureFlags {
            centrality_enabled: true,
            ..FeatureFlags::default()
        };
        assert!(flags.has_v2_features());
    }

    #[test]
    fn test_feature_flags_has_v2_features_entrypoint() {
        let flags = FeatureFlags {
            entrypoint_detection: true,
            ..FeatureFlags::default()
        };
        assert!(flags.has_v2_features());
    }

    #[test]
    fn test_feature_flags_has_v2_features_examples() {
        let flags = FeatureFlags {
            examples_analysis: true,
            ..FeatureFlags::default()
        };
        assert!(flags.has_v2_features());
    }

    #[test]
    fn test_feature_flags_has_v2_features_other() {
        let flags = FeatureFlags {
            semantic_analysis: true,
            ml_features: true,
            scaling_enabled: true,
            ..FeatureFlags::default()
        };
        // These don't count as V2 features
        assert!(!flags.has_v2_features());
    }

    #[test]
    fn test_feature_flags_enabled_features_none() {
        let flags = FeatureFlags::default();
        let features = flags.enabled_features();
        assert!(features.is_empty());
    }

    #[test]
    fn test_feature_flags_enabled_features_some() {
        let flags = FeatureFlags {
            centrality_enabled: true,
            entrypoint_detection: true,
            examples_analysis: false,
            semantic_analysis: false,
            ml_features: true,
            experimental_scoring: false,
            scaling_enabled: false,
            auto_exclude_tests: false,
        };

        let features = flags.enabled_features();
        assert_eq!(features.len(), 3);
        assert!(features.contains(&"centrality"));
        assert!(features.contains(&"entrypoint_detection"));
        assert!(features.contains(&"ml_features"));
        assert!(!features.contains(&"examples_analysis"));
    }

    #[test]
    fn test_feature_flags_enabled_features_all() {
        let flags = FeatureFlags {
            centrality_enabled: true,
            entrypoint_detection: true,
            examples_analysis: true,
            semantic_analysis: true,
            ml_features: true,
            experimental_scoring: true,
            scaling_enabled: true,
            auto_exclude_tests: true,
        };

        let features = flags.enabled_features();
        assert_eq!(features.len(), 7); // auto_exclude_tests is not included
        assert!(features.contains(&"centrality"));
        assert!(features.contains(&"scaling"));
    }

    #[test]
    fn test_output_format_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(OutputFormat::Json);
        set.insert(OutputFormat::Csv);
        set.insert(OutputFormat::Json); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&OutputFormat::Json));
        assert!(set.contains(&OutputFormat::Csv));
    }
}

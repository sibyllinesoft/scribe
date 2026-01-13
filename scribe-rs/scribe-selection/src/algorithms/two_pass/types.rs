//! Type definitions for two-pass selection system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for the two-pass selection system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoPassConfig {
    /// Percentage of budget allocated to speculative pass (0.0-1.0)
    pub speculation_ratio: f64,
    /// Minimum confidence threshold for speculative selections
    pub speculation_threshold: f64,
    /// Maximum iterations for rule-based refinement
    pub max_iterations: usize,
    /// Enable coverage gap analysis
    pub enable_gap_analysis: bool,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            speculation_ratio: 0.75,    // 75% speculation, 25% rules
            speculation_threshold: 0.5, // Lower threshold for better test coverage
            max_iterations: 3,
            enable_gap_analysis: true,
        }
    }
}

/// Result of two-pass selection process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoPassResult {
    /// Files selected during speculative pass
    pub speculative_files: Vec<String>,
    /// Files added during rule-based pass
    pub rule_based_files: Vec<String>,
    /// Coverage gaps identified
    pub coverage_gaps: Vec<CoverageGap>,
    /// Total selection score
    pub selection_score: f64,
    /// Budget utilization
    pub budget_utilization: f64,
    /// Execution metrics
    pub metrics: SelectionMetrics,
}

/// Represents a coverage gap in the selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageGap {
    /// Type of gap (dependency, interface, implementation, etc.)
    pub gap_type: String,
    /// Severity of the gap (0.0-1.0)
    pub severity: f64,
    /// Files that could address this gap
    pub candidate_files: Vec<String>,
    /// Reason for the gap
    pub reason: String,
}

/// Metrics collected during selection process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionMetrics {
    /// Time spent in speculative pass (ms)
    pub speculation_time_ms: u64,
    /// Time spent in rule-based pass (ms)
    pub rule_based_time_ms: u64,
    /// Number of rules evaluated
    pub rules_evaluated: usize,
    /// Number of coverage gaps found
    pub gaps_found: usize,
    /// Files considered during process
    pub files_considered: usize,
}

/// Selection rule for rule-based pass
#[derive(Debug, Clone)]
pub struct SelectionRule {
    /// Rule name
    pub name: String,
    /// Priority weight (0.0-1.0)
    pub weight: f64,
    /// Rule evaluation function
    pub evaluator: fn(&SelectionContext, &str) -> f64,
    /// Rule description
    pub description: String,
}

/// Context passed to rule evaluators
#[derive(Debug)]
pub struct SelectionContext<'a> {
    /// Files already selected
    pub selected_files: &'a HashSet<String>,
    /// Available files with metadata
    pub available_files: &'a HashMap<String, FileInfo>,
    /// Dependency graph
    pub dependencies: &'a HashMap<String, Vec<String>>,
    /// Interface definitions
    pub interfaces: &'a HashMap<String, Vec<String>>,
    /// Current budget remaining
    pub remaining_budget: usize,
    /// Reverse dependency lookup: file -> files that depend on it
    pub dependents_map: &'a HashMap<String, Vec<String>>,
    /// Pre-computed count of selected source files (O(1) optimization)
    pub selected_source_count: usize,
}

/// File information for selection decisions
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// File path
    pub path: String,
    /// Estimated token count
    pub token_count: usize,
    /// File type (source, test, config, etc.)
    pub file_type: String,
    /// Importance score (0.0-1.0)
    pub importance: f64,
    /// Dependencies of this file
    pub dependencies: Vec<String>,
    /// Files that depend on this file
    pub dependents: Vec<String>,
    /// Interfaces exposed by this file
    pub exposed_interfaces: Vec<String>,
    /// Interfaces consumed by this file
    pub consumed_interfaces: Vec<String>,
}

/// Main two-pass selection engine
pub struct TwoPassSelector {
    pub(crate) config: TwoPassConfig,
    pub(crate) rules: Vec<SelectionRule>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_pass_config_default() {
        let config = TwoPassConfig::default();
        assert!((config.speculation_ratio - 0.75).abs() < 0.001);
        assert!((config.speculation_threshold - 0.5).abs() < 0.001);
        assert_eq!(config.max_iterations, 3);
        assert!(config.enable_gap_analysis);
    }

    #[test]
    fn test_two_pass_config_custom() {
        let config = TwoPassConfig {
            speculation_ratio: 0.6,
            speculation_threshold: 0.7,
            max_iterations: 5,
            enable_gap_analysis: false,
        };

        assert!((config.speculation_ratio - 0.6).abs() < 0.001);
        assert!((config.speculation_threshold - 0.7).abs() < 0.001);
        assert_eq!(config.max_iterations, 5);
        assert!(!config.enable_gap_analysis);
    }

    #[test]
    fn test_two_pass_config_clone() {
        let config = TwoPassConfig::default();
        let cloned = config.clone();
        assert!((config.speculation_ratio - cloned.speculation_ratio).abs() < 0.001);
        assert_eq!(config.max_iterations, cloned.max_iterations);
    }

    #[test]
    fn test_two_pass_config_serialize() {
        let config = TwoPassConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("speculation_ratio"));
        assert!(json.contains("0.75"));

        let deserialized: TwoPassConfig = serde_json::from_str(&json).unwrap();
        assert!((config.speculation_ratio - deserialized.speculation_ratio).abs() < 0.001);
    }

    #[test]
    fn test_two_pass_config_debug() {
        let config = TwoPassConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("TwoPassConfig"));
        assert!(debug.contains("speculation_ratio"));
    }

    #[test]
    fn test_two_pass_result_creation() {
        let result = TwoPassResult {
            speculative_files: vec!["src/lib.rs".to_string()],
            rule_based_files: vec!["src/utils.rs".to_string()],
            coverage_gaps: vec![],
            selection_score: 0.85,
            budget_utilization: 0.9,
            metrics: SelectionMetrics {
                speculation_time_ms: 100,
                rule_based_time_ms: 50,
                rules_evaluated: 10,
                gaps_found: 0,
                files_considered: 20,
            },
        };

        assert_eq!(result.speculative_files.len(), 1);
        assert_eq!(result.rule_based_files.len(), 1);
        assert!((result.selection_score - 0.85).abs() < 0.001);
        assert!((result.budget_utilization - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_two_pass_result_clone() {
        let result = TwoPassResult {
            speculative_files: vec!["a.rs".to_string()],
            rule_based_files: vec!["b.rs".to_string()],
            coverage_gaps: vec![],
            selection_score: 0.5,
            budget_utilization: 0.7,
            metrics: SelectionMetrics {
                speculation_time_ms: 10,
                rule_based_time_ms: 5,
                rules_evaluated: 3,
                gaps_found: 0,
                files_considered: 10,
            },
        };

        let cloned = result.clone();
        assert_eq!(result.speculative_files, cloned.speculative_files);
        assert_eq!(result.selection_score, cloned.selection_score);
    }

    #[test]
    fn test_two_pass_result_serialize() {
        let result = TwoPassResult {
            speculative_files: vec!["test.rs".to_string()],
            rule_based_files: vec![],
            coverage_gaps: vec![],
            selection_score: 0.75,
            budget_utilization: 0.6,
            metrics: SelectionMetrics {
                speculation_time_ms: 200,
                rule_based_time_ms: 100,
                rules_evaluated: 5,
                gaps_found: 1,
                files_considered: 15,
            },
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("speculative_files"));
        assert!(json.contains("test.rs"));

        let deserialized: TwoPassResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.speculative_files, deserialized.speculative_files);
    }

    #[test]
    fn test_two_pass_result_debug() {
        let result = TwoPassResult {
            speculative_files: vec![],
            rule_based_files: vec![],
            coverage_gaps: vec![],
            selection_score: 0.0,
            budget_utilization: 0.0,
            metrics: SelectionMetrics {
                speculation_time_ms: 0,
                rule_based_time_ms: 0,
                rules_evaluated: 0,
                gaps_found: 0,
                files_considered: 0,
            },
        };

        let debug = format!("{:?}", result);
        assert!(debug.contains("TwoPassResult"));
    }

    #[test]
    fn test_coverage_gap_creation() {
        let gap = CoverageGap {
            gap_type: "dependency".to_string(),
            severity: 0.8,
            candidate_files: vec!["dep.rs".to_string(), "util.rs".to_string()],
            reason: "Missing dependency coverage".to_string(),
        };

        assert_eq!(gap.gap_type, "dependency");
        assert!((gap.severity - 0.8).abs() < 0.001);
        assert_eq!(gap.candidate_files.len(), 2);
        assert!(gap.reason.contains("Missing"));
    }

    #[test]
    fn test_coverage_gap_clone() {
        let gap = CoverageGap {
            gap_type: "interface".to_string(),
            severity: 0.5,
            candidate_files: vec!["interface.rs".to_string()],
            reason: "Interface not covered".to_string(),
        };

        let cloned = gap.clone();
        assert_eq!(gap.gap_type, cloned.gap_type);
        assert_eq!(gap.severity, cloned.severity);
    }

    #[test]
    fn test_coverage_gap_serialize() {
        let gap = CoverageGap {
            gap_type: "implementation".to_string(),
            severity: 0.3,
            candidate_files: vec!["impl.rs".to_string()],
            reason: "Implementation gap".to_string(),
        };

        let json = serde_json::to_string(&gap).unwrap();
        assert!(json.contains("implementation"));
        assert!(json.contains("severity"));

        let deserialized: CoverageGap = serde_json::from_str(&json).unwrap();
        assert_eq!(gap.gap_type, deserialized.gap_type);
    }

    #[test]
    fn test_coverage_gap_debug() {
        let gap = CoverageGap {
            gap_type: "test".to_string(),
            severity: 1.0,
            candidate_files: vec![],
            reason: "Debug".to_string(),
        };

        let debug = format!("{:?}", gap);
        assert!(debug.contains("CoverageGap"));
    }

    #[test]
    fn test_selection_metrics_creation() {
        let metrics = SelectionMetrics {
            speculation_time_ms: 500,
            rule_based_time_ms: 250,
            rules_evaluated: 15,
            gaps_found: 3,
            files_considered: 100,
        };

        assert_eq!(metrics.speculation_time_ms, 500);
        assert_eq!(metrics.rule_based_time_ms, 250);
        assert_eq!(metrics.rules_evaluated, 15);
        assert_eq!(metrics.gaps_found, 3);
        assert_eq!(metrics.files_considered, 100);
    }

    #[test]
    fn test_selection_metrics_clone() {
        let metrics = SelectionMetrics {
            speculation_time_ms: 100,
            rule_based_time_ms: 50,
            rules_evaluated: 5,
            gaps_found: 1,
            files_considered: 25,
        };

        let cloned = metrics.clone();
        assert_eq!(metrics.speculation_time_ms, cloned.speculation_time_ms);
        assert_eq!(metrics.files_considered, cloned.files_considered);
    }

    #[test]
    fn test_selection_metrics_serialize() {
        let metrics = SelectionMetrics {
            speculation_time_ms: 1000,
            rule_based_time_ms: 500,
            rules_evaluated: 20,
            gaps_found: 5,
            files_considered: 200,
        };

        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("speculation_time_ms"));
        assert!(json.contains("1000"));

        let deserialized: SelectionMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics.speculation_time_ms, deserialized.speculation_time_ms);
    }

    #[test]
    fn test_selection_metrics_debug() {
        let metrics = SelectionMetrics {
            speculation_time_ms: 0,
            rule_based_time_ms: 0,
            rules_evaluated: 0,
            gaps_found: 0,
            files_considered: 0,
        };

        let debug = format!("{:?}", metrics);
        assert!(debug.contains("SelectionMetrics"));
    }

    #[test]
    fn test_file_info_creation() {
        let info = FileInfo {
            path: "src/lib.rs".to_string(),
            token_count: 500,
            file_type: "source".to_string(),
            importance: 0.9,
            dependencies: vec!["std".to_string()],
            dependents: vec!["main.rs".to_string()],
            exposed_interfaces: vec!["public_fn".to_string()],
            consumed_interfaces: vec!["trait_impl".to_string()],
        };

        assert_eq!(info.path, "src/lib.rs");
        assert_eq!(info.token_count, 500);
        assert_eq!(info.file_type, "source");
        assert!((info.importance - 0.9).abs() < 0.001);
        assert_eq!(info.dependencies.len(), 1);
        assert_eq!(info.dependents.len(), 1);
    }

    #[test]
    fn test_file_info_clone() {
        let info = FileInfo {
            path: "test.rs".to_string(),
            token_count: 100,
            file_type: "test".to_string(),
            importance: 0.5,
            dependencies: vec![],
            dependents: vec![],
            exposed_interfaces: vec![],
            consumed_interfaces: vec![],
        };

        let cloned = info.clone();
        assert_eq!(info.path, cloned.path);
        assert_eq!(info.token_count, cloned.token_count);
    }

    #[test]
    fn test_file_info_debug() {
        let info = FileInfo {
            path: "debug.rs".to_string(),
            token_count: 10,
            file_type: "source".to_string(),
            importance: 0.1,
            dependencies: vec![],
            dependents: vec![],
            exposed_interfaces: vec![],
            consumed_interfaces: vec![],
        };

        let debug = format!("{:?}", info);
        assert!(debug.contains("FileInfo"));
        assert!(debug.contains("debug.rs"));
    }

    #[test]
    fn test_selection_rule_creation() {
        fn dummy_evaluator(_: &SelectionContext, _: &str) -> f64 {
            0.5
        }

        let rule = SelectionRule {
            name: "test_rule".to_string(),
            weight: 0.8,
            evaluator: dummy_evaluator,
            description: "A test rule".to_string(),
        };

        assert_eq!(rule.name, "test_rule");
        assert!((rule.weight - 0.8).abs() < 0.001);
        assert_eq!(rule.description, "A test rule");
    }

    #[test]
    fn test_selection_rule_debug() {
        fn dummy_evaluator(_: &SelectionContext, _: &str) -> f64 {
            0.0
        }

        let rule = SelectionRule {
            name: "debug_rule".to_string(),
            weight: 1.0,
            evaluator: dummy_evaluator,
            description: "Debug".to_string(),
        };

        let debug = format!("{:?}", rule);
        assert!(debug.contains("SelectionRule"));
        assert!(debug.contains("debug_rule"));
    }

    #[test]
    fn test_two_pass_selector_creation() {
        let selector = TwoPassSelector {
            config: TwoPassConfig::default(),
            rules: vec![],
        };

        assert!((selector.config.speculation_ratio - 0.75).abs() < 0.001);
        assert!(selector.rules.is_empty());
    }
}

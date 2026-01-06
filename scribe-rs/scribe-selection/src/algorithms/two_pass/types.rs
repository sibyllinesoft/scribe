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

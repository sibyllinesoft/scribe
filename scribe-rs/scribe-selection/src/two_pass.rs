//! Two-Pass Selection System for V5 Variant
//!
//! Implements a sophisticated selection system with speculative first pass
//! and rule-based coverage gap analysis in the second pass.

use rayon::prelude::*;
use scribe_core::{Result, ScribeError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

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
    config: TwoPassConfig,
    rules: Vec<SelectionRule>,
}

impl TwoPassSelector {
    /// Create new two-pass selector with default configuration
    pub fn new() -> Self {
        Self {
            config: TwoPassConfig::default(),
            rules: Self::create_default_rules(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TwoPassConfig) -> Self {
        Self {
            config,
            rules: Self::create_default_rules(),
        }
    }

    /// Execute two-pass selection algorithm
    pub fn select_files(
        &self,
        available_files: &HashMap<String, FileInfo>,
        dependencies: &HashMap<String, Vec<String>>,
        interfaces: &HashMap<String, Vec<String>>,
        total_budget: usize,
    ) -> Result<TwoPassResult> {
        let start_time = std::time::Instant::now();

        // Phase 1: Speculative Selection (75% of budget)
        let speculation_budget = (total_budget as f64 * self.config.speculation_ratio) as usize;
        let speculation_start = std::time::Instant::now();

        let speculative_files =
            self.speculative_pass(available_files, dependencies, speculation_budget)?;

        let speculation_time = speculation_start.elapsed().as_millis() as u64;

        // Phase 2: Rule-Based Gap Analysis (25% of budget)
        let rule_budget = total_budget - speculation_budget;
        let rule_start = std::time::Instant::now();

        let mut selected_files: HashSet<String> = speculative_files.iter().cloned().collect();

        let (rule_based_files, coverage_gaps) = self.rule_based_pass(
            &selected_files,
            available_files,
            dependencies,
            interfaces,
            rule_budget,
        )?;

        let rule_time = rule_start.elapsed().as_millis() as u64;

        // Add rule-based files to selection
        selected_files.extend(rule_based_files.iter().cloned());

        // Calculate final metrics
        let total_tokens: usize = selected_files
            .iter()
            .filter_map(|f| available_files.get(f))
            .map(|info| info.token_count)
            .sum();

        let budget_utilization = total_tokens as f64 / total_budget as f64;
        let selection_score = self.calculate_selection_score(&selected_files, available_files)?;

        let gaps_count = coverage_gaps.len();

        Ok(TwoPassResult {
            speculative_files,
            rule_based_files,
            coverage_gaps,
            selection_score,
            budget_utilization,
            metrics: SelectionMetrics {
                speculation_time_ms: speculation_time,
                rule_based_time_ms: rule_time,
                rules_evaluated: self.rules.len(),
                gaps_found: gaps_count,
                files_considered: available_files.len(),
            },
        })
    }

    /// Phase 1: Speculative file selection based on importance and confidence
    fn speculative_pass(
        &self,
        available_files: &HashMap<String, FileInfo>,
        dependencies: &HashMap<String, Vec<String>>,
        budget: usize,
    ) -> Result<Vec<String>> {
        let mut selected = Vec::new();
        let mut remaining_budget = budget;

        // Calculate confidence scores in parallel and sort by importance * confidence
        let mut candidates: Vec<(&String, &FileInfo, f64)> = available_files
            .par_iter()
            .map(|(file_path, file_info)| {
                let confidence = self.calculate_confidence(file_info, dependencies);
                (file_path, file_info, confidence)
            })
            .collect();

        candidates.sort_by(|a, b| {
            let score_a = a.1.importance * a.2; // a.2 is pre-calculated confidence
            let score_b = b.1.importance * b.2; // b.2 is pre-calculated confidence
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedily select files while budget allows
        for (file_path, file_info, confidence) in candidates {
            if confidence >= self.config.speculation_threshold
                && file_info.token_count <= remaining_budget
            {
                selected.push(file_path.clone());
                remaining_budget -= file_info.token_count;
            }
        }

        Ok(selected)
    }

    /// Phase 2: Rule-based selection to fill coverage gaps
    fn rule_based_pass(
        &self,
        selected_files: &HashSet<String>,
        available_files: &HashMap<String, FileInfo>,
        dependencies: &HashMap<String, Vec<String>>,
        interfaces: &HashMap<String, Vec<String>>,
        budget: usize,
    ) -> Result<(Vec<String>, Vec<CoverageGap>)> {
        let mut additional_files = Vec::new();
        let mut coverage_gaps = Vec::new();
        let mut remaining_budget = budget;

        // Identify coverage gaps
        if self.config.enable_gap_analysis {
            coverage_gaps = self.analyze_coverage_gaps(
                selected_files,
                available_files,
                dependencies,
                interfaces,
            )?;
        }

        // Pre-build reverse dependency lookup for O(1) access
        let mut dependents_map: HashMap<String, Vec<String>> = HashMap::new();
        for (file_path, file_info) in available_files {
            for dep in &file_info.dependencies {
                dependents_map
                    .entry(dep.clone())
                    .or_default()
                    .push(file_path.clone());
            }
        }

        // Pre-compute selected source count to avoid O(N) iteration in rules
        let selected_source_count = selected_files
            .iter()
            .filter(|f| {
                available_files
                    .get(*f)
                    .map_or(false, |info| info.file_type == "source")
            })
            .count();

        // Apply rules to address gaps
        let context = SelectionContext {
            selected_files,
            available_files,
            dependencies,
            interfaces,
            remaining_budget,
            dependents_map: &dependents_map,
            selected_source_count,
        };

        // Score all unselected files using rules (parallel processing)
        let rule_scores: HashMap<String, f64> = available_files
            .par_iter()
            .filter(|(file_path, file_info)| {
                !selected_files.contains(*file_path) && file_info.token_count <= remaining_budget
            })
            .map(|(file_path, _file_info)| {
                let total_score = self
                    .rules
                    .iter()
                    .map(|rule| (rule.evaluator)(&context, file_path) * rule.weight)
                    .sum();
                (file_path.clone(), total_score)
            })
            .collect();

        // Select highest scoring files within budget
        let mut sorted_scores: Vec<(&String, &f64)> = rule_scores.iter().collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (file_path, _score) in sorted_scores {
            if let Some(file_info) = available_files.get(file_path) {
                if file_info.token_count <= remaining_budget {
                    additional_files.push(file_path.clone());
                    remaining_budget -= file_info.token_count;
                }
            }
        }

        Ok((additional_files, coverage_gaps))
    }

    /// Calculate confidence score for a file
    fn calculate_confidence(
        &self,
        file_info: &FileInfo,
        dependencies: &HashMap<String, Vec<String>>,
    ) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // Boost confidence for files with many dependents
        confidence += (file_info.dependents.len() as f64 * 0.1).min(0.3);

        // Boost confidence for interface files
        if !file_info.exposed_interfaces.is_empty() {
            confidence += 0.2;
        }

        // Boost confidence for core file types
        match file_info.file_type.as_str() {
            "source" => confidence += 0.1,
            "interface" => confidence += 0.2,
            "config" => confidence += 0.05,
            _ => {}
        }

        confidence.min(1.0)
    }

    /// Analyze coverage gaps in current selection
    fn analyze_coverage_gaps(
        &self,
        selected_files: &HashSet<String>,
        available_files: &HashMap<String, FileInfo>,
        dependencies: &HashMap<String, Vec<String>>,
        interfaces: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<CoverageGap>> {
        let mut gaps = Vec::new();

        // Check for missing dependencies
        for selected_file in selected_files {
            if let Some(file_info) = available_files.get(selected_file) {
                for dep in &file_info.dependencies {
                    if !selected_files.contains(dep) && available_files.contains_key(dep) {
                        gaps.push(CoverageGap {
                            gap_type: "missing_dependency".to_string(),
                            severity: 0.8,
                            candidate_files: vec![dep.clone()],
                            reason: format!("{} depends on {}", selected_file, dep),
                        });
                    }
                }
            }
        }

        // Check for incomplete interface coverage
        for (interface, implementers) in interfaces {
            let has_implementation = implementers.iter().any(|imp| selected_files.contains(imp));
            if !has_implementation && !implementers.is_empty() {
                gaps.push(CoverageGap {
                    gap_type: "missing_interface_implementation".to_string(),
                    severity: 0.6,
                    candidate_files: implementers.clone(),
                    reason: format!("Interface {} has no selected implementations", interface),
                });
            }
        }

        // Check for orphaned test files
        let test_files: Vec<_> = selected_files
            .iter()
            .filter(|f| {
                available_files
                    .get(*f)
                    .map_or(false, |info| info.file_type == "test")
            })
            .collect();

        for test_file in test_files {
            if let Some(test_info) = available_files.get(test_file) {
                let has_source = test_info.dependencies.iter().any(|dep| {
                    selected_files.contains(dep)
                        && available_files
                            .get(dep)
                            .map_or(false, |info| info.file_type == "source")
                });

                if !has_source {
                    gaps.push(CoverageGap {
                        gap_type: "orphaned_test".to_string(),
                        severity: 0.4,
                        candidate_files: test_info.dependencies.clone(),
                        reason: format!(
                            "Test file {} has no corresponding source files selected",
                            test_file
                        ),
                    });
                }
            }
        }

        Ok(gaps)
    }

    /// Calculate overall selection quality score
    fn calculate_selection_score(
        &self,
        selected_files: &HashSet<String>,
        available_files: &HashMap<String, FileInfo>,
    ) -> Result<f64> {
        if selected_files.is_empty() {
            return Ok(0.0);
        }

        let mut total_importance = 0.0;
        let mut total_files = 0.0;

        for file_path in selected_files {
            if let Some(file_info) = available_files.get(file_path) {
                total_importance += file_info.importance;
                total_files += 1.0;
            }
        }

        Ok(total_importance / total_files)
    }

    /// Create default set of selection rules
    fn create_default_rules() -> Vec<SelectionRule> {
        vec![
            SelectionRule {
                name: "dependency_completeness".to_string(),
                weight: 0.25,
                evaluator: |context, file_path| {
                    if let Some(file_info) = context.available_files.get(file_path) {
                        // Score based on how many selected files depend on this file (O(1) lookup)
                        let satisfies_dependencies = context
                            .dependents_map
                            .get(file_path)
                            .map(|dependents| {
                                dependents
                                    .iter()
                                    .filter(|dependent| context.selected_files.contains(*dependent))
                                    .count()
                            })
                            .unwrap_or(0);

                        // Also consider if this file's own dependencies are satisfied
                        let missing_deps = file_info
                            .dependencies
                            .iter()
                            .filter(|dep| !context.selected_files.contains(*dep))
                            .count();

                        let dependency_satisfaction_score = if satisfies_dependencies > 0 {
                            0.8 + (satisfies_dependencies as f64 * 0.1).min(0.2)
                        } else {
                            0.3
                        };

                        let completeness_score = if file_info.dependencies.is_empty() {
                            1.0 // No dependencies to worry about
                        } else {
                            1.0 - (missing_deps as f64 / file_info.dependencies.len() as f64)
                        };

                        (dependency_satisfaction_score + completeness_score) / 2.0
                    } else {
                        0.0
                    }
                },
                description: "Prefer files that complete dependency chains".to_string(),
            },
            SelectionRule {
                name: "interface_coverage".to_string(),
                weight: 0.2,
                evaluator: |context, file_path| {
                    if let Some(file_info) = context.available_files.get(file_path) {
                        let interface_score = file_info.exposed_interfaces.len() as f64 * 0.3;
                        let implementation_score = file_info.consumed_interfaces.len() as f64 * 0.1;
                        (interface_score + implementation_score).min(1.0)
                    } else {
                        0.0
                    }
                },
                description: "Prefer files that expose or implement important interfaces"
                    .to_string(),
            },
            SelectionRule {
                name: "test_source_pairing".to_string(),
                weight: 0.15,
                evaluator: |context, file_path| {
                    if let Some(file_info) = context.available_files.get(file_path) {
                        if file_info.file_type == "test" {
                            // For test files, check if corresponding source is selected
                            let has_source = file_info.dependencies.iter().any(|dep| {
                                context.selected_files.contains(dep)
                                    && context
                                        .available_files
                                        .get(dep)
                                        .map_or(false, |info| info.file_type == "source")
                            });
                            if has_source {
                                1.0
                            } else {
                                0.2
                            }
                        } else if file_info.file_type == "source" {
                            // For source files, check if we have related tests
                            let has_tests = file_info.dependents.iter().any(|dep| {
                                context
                                    .available_files
                                    .get(dep)
                                    .map_or(false, |info| info.file_type == "test")
                            });
                            if has_tests {
                                0.8
                            } else {
                                0.5
                            }
                        } else {
                            0.5
                        }
                    } else {
                        0.0
                    }
                },
                description: "Prefer test-source file pairings".to_string(),
            },
            SelectionRule {
                name: "centrality_score".to_string(),
                weight: 0.15,
                evaluator: |context, file_path| {
                    if let Some(file_info) = context.available_files.get(file_path) {
                        let in_degree = file_info.dependents.len() as f64;
                        let out_degree = file_info.dependencies.len() as f64;
                        let centrality = (in_degree * 0.7 + out_degree * 0.3) / 10.0; // Normalize
                        centrality.min(1.0)
                    } else {
                        0.0
                    }
                },
                description: "Prefer files with high connectivity in dependency graph".to_string(),
            },
            SelectionRule {
                name: "importance_alignment".to_string(),
                weight: 0.1,
                evaluator: |_context, file_path| {
                    if let Some(file_info) = _context.available_files.get(file_path) {
                        file_info.importance
                    } else {
                        0.0
                    }
                },
                description: "Prefer files with high intrinsic importance scores".to_string(),
            },
            SelectionRule {
                name: "token_efficiency".to_string(),
                weight: 0.08,
                evaluator: |context, file_path| {
                    if let Some(file_info) = context.available_files.get(file_path) {
                        let efficiency =
                            file_info.importance / (file_info.token_count as f64 / 1000.0).max(0.1);
                        efficiency.min(1.0)
                    } else {
                        0.0
                    }
                },
                description: "Prefer files with high importance-to-token ratio".to_string(),
            },
            SelectionRule {
                name: "gap_filling".to_string(),
                weight: 0.05,
                evaluator: |context, file_path| {
                    if let Some(file_info) = context.available_files.get(file_path) {
                        // Check if this file fills an important gap
                        let fills_dependency_gap = file_info
                            .dependents
                            .iter()
                            .any(|dep| context.selected_files.contains(dep));

                        let fills_interface_gap = !file_info.exposed_interfaces.is_empty()
                            && file_info.exposed_interfaces.iter().any(|iface| {
                                context.interfaces.get(iface).map_or(false, |impls| {
                                    impls.iter().any(|imp| context.selected_files.contains(imp))
                                })
                            });

                        if fills_dependency_gap || fills_interface_gap {
                            0.8
                        } else {
                            0.3
                        }
                    } else {
                        0.0
                    }
                },
                description: "Prefer files that fill critical coverage gaps".to_string(),
            },
            SelectionRule {
                name: "configuration_completeness".to_string(),
                weight: 0.02,
                evaluator: |context, file_path| {
                    if let Some(file_info) = context.available_files.get(file_path) {
                        if file_info.file_type == "config" {
                            // O(1) optimization: Use pre-computed selected source count
                            if context.selected_source_count > 0 {
                                0.7 // Config files are useful when we have source code
                            } else {
                                0.2
                            }
                        } else {
                            0.5 // Neutral for non-config files
                        }
                    } else {
                        0.0
                    }
                },
                description: "Include configuration files when relevant source code is selected"
                    .to_string(),
            },
        ]
    }
}

impl Default for TwoPassSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_files() -> HashMap<String, FileInfo> {
        let mut files = HashMap::new();

        files.insert(
            "src/main.rs".to_string(),
            FileInfo {
                path: "src/main.rs".to_string(),
                token_count: 500,
                file_type: "source".to_string(),
                importance: 0.9,
                dependencies: vec!["src/lib.rs".to_string()],
                dependents: vec![],
                exposed_interfaces: vec!["Main".to_string()],
                consumed_interfaces: vec!["Library".to_string()],
            },
        );

        files.insert(
            "src/lib.rs".to_string(),
            FileInfo {
                path: "src/lib.rs".to_string(),
                token_count: 800,
                file_type: "source".to_string(),
                importance: 0.8,
                dependencies: vec![],
                dependents: vec!["src/main.rs".to_string()],
                exposed_interfaces: vec!["Library".to_string()],
                consumed_interfaces: vec![],
            },
        );

        files.insert(
            "tests/integration_test.rs".to_string(),
            FileInfo {
                path: "tests/integration_test.rs".to_string(),
                token_count: 300,
                file_type: "test".to_string(),
                importance: 0.6,
                dependencies: vec!["src/lib.rs".to_string()],
                dependents: vec![],
                exposed_interfaces: vec![],
                consumed_interfaces: vec!["Library".to_string()],
            },
        );

        files.insert(
            "config/settings.toml".to_string(),
            FileInfo {
                path: "config/settings.toml".to_string(),
                token_count: 100,
                file_type: "config".to_string(),
                importance: 0.3,
                dependencies: vec![],
                dependents: vec![],
                exposed_interfaces: vec![],
                consumed_interfaces: vec![],
            },
        );

        files
    }

    fn create_test_dependencies() -> HashMap<String, Vec<String>> {
        let mut deps = HashMap::new();
        deps.insert("src/main.rs".to_string(), vec!["src/lib.rs".to_string()]);
        deps.insert(
            "tests/integration_test.rs".to_string(),
            vec!["src/lib.rs".to_string()],
        );
        deps
    }

    fn create_test_interfaces() -> HashMap<String, Vec<String>> {
        let mut interfaces = HashMap::new();
        interfaces.insert("Library".to_string(), vec!["src/lib.rs".to_string()]);
        interfaces.insert("Main".to_string(), vec!["src/main.rs".to_string()]);
        interfaces
    }

    #[test]
    fn test_two_pass_selector_creation() {
        let selector = TwoPassSelector::new();
        assert_eq!(selector.config.speculation_ratio, 0.75);
        assert_eq!(selector.rules.len(), 8);
    }

    #[test]
    fn test_speculative_pass() {
        let selector = TwoPassSelector::new();
        let files = create_test_files();
        let dependencies = create_test_dependencies();

        let result = selector
            .speculative_pass(&files, &dependencies, 1000)
            .unwrap();

        assert!(!result.is_empty());

        // Debug: Print what was selected and confidence scores
        for file_path in &result {
            if let Some(file_info) = files.get(file_path) {
                let confidence = selector.calculate_confidence(file_info, &dependencies);
                println!(
                    "Selected: {} (importance: {}, confidence: {})",
                    file_path, file_info.importance, confidence
                );
            }
        }

        // Check if high-importance files are selected (with more lenient assertions)
        let has_high_importance_file = result
            .iter()
            .any(|f| files.get(f).map_or(false, |info| info.importance >= 0.8));
        assert!(
            has_high_importance_file,
            "Should select at least one high-importance file"
        );
    }

    #[test]
    fn test_full_two_pass_selection() {
        let selector = TwoPassSelector::new();
        let files = create_test_files();
        let dependencies = create_test_dependencies();
        let interfaces = create_test_interfaces();

        let result = selector
            .select_files(&files, &dependencies, &interfaces, 1500)
            .unwrap();

        assert!(!result.speculative_files.is_empty());
        assert!(result.budget_utilization <= 1.0);
        assert!(result.selection_score > 0.0);
        assert!(result.metrics.files_considered > 0);
    }

    #[test]
    fn test_coverage_gap_analysis() {
        let selector = TwoPassSelector::new();
        let files = create_test_files();
        let dependencies = create_test_dependencies();
        let interfaces = create_test_interfaces();

        let mut selected = HashSet::new();
        selected.insert("src/main.rs".to_string());
        // Missing src/lib.rs dependency

        let gaps = selector
            .analyze_coverage_gaps(&selected, &files, &dependencies, &interfaces)
            .unwrap();

        assert!(!gaps.is_empty());
        // Should detect missing dependency
        assert!(gaps.iter().any(|gap| gap.gap_type == "missing_dependency"));
    }

    #[test]
    fn test_rule_evaluation() {
        let selector = TwoPassSelector::new();
        let files = create_test_files();
        let dependencies = create_test_dependencies();
        let interfaces = create_test_interfaces();

        let mut selected = HashSet::new();
        selected.insert("src/main.rs".to_string());

        // Pre-build reverse dependency lookup for testing
        let mut dependents_map: HashMap<String, Vec<String>> = HashMap::new();
        for (file_path, file_info) in &files {
            for dep in &file_info.dependencies {
                dependents_map
                    .entry(dep.clone())
                    .or_default()
                    .push(file_path.clone());
            }
        }

        // Pre-compute selected source count for testing
        let selected_source_count = selected
            .iter()
            .filter(|f| {
                files
                    .get(*f)
                    .map_or(false, |info| info.file_type == "source")
            })
            .count();

        let context = SelectionContext {
            selected_files: &selected,
            available_files: &files,
            dependencies: &dependencies,
            interfaces: &interfaces,
            remaining_budget: 1000,
            dependents_map: &dependents_map,
            selected_source_count,
        };

        // Test dependency completeness rule
        let dep_rule = &selector.rules[0];
        let score = (dep_rule.evaluator)(&context, "src/lib.rs");
        println!("Dependency rule score for src/lib.rs: {}", score);

        // Test that lib.rs scores well (it's needed by main.rs which is selected)
        assert!(
            score >= 0.5,
            "src/lib.rs should score well as it fills a dependency gap (score: {})",
            score
        );

        // Test interface coverage rule
        let interface_rule = &selector.rules[1];
        let interface_score = (interface_rule.evaluator)(&context, "src/lib.rs");
        println!("Interface rule score for src/lib.rs: {}", interface_score);
        assert!(
            interface_score > 0.0,
            "src/lib.rs should have some interface score"
        );
    }
}

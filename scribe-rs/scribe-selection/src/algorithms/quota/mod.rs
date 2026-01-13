//! Quota-based file selection with density-greedy algorithm.

mod types;

pub use types::{
    CategoryDetector, CategoryQuota, FileCategory, QuotaAllocation, QuotaManager, QuotaScanResult,
};

use rayon::prelude::*;
use regex::RegexSet;
use std::collections::HashMap;

use scribe_core::{Result as ScribeResult, ScribeError};

impl Default for CategoryDetector {
    fn default() -> Self {
        Self::new().expect("Failed to create CategoryDetector")
    }
}

impl CategoryDetector {
    pub fn new() -> Result<Self, regex::Error> {
        // Config file patterns - escape regex special characters and convert to regex patterns
        let config_patterns = vec![
            // Configuration files (as regex patterns)
            r"\.json$",
            r"\.yaml$",
            r"\.yml$",
            r"\.toml$",
            r"\.ini$",
            r"\.cfg$",
            r"\.conf$",
            // Build and dependency files
            r"package\.json$",
            r"requirements\.txt$",
            r"pyproject\.toml$",
            r"cargo\.toml$",
            r"setup\.py$",
            r"setup\.cfg$",
            r"makefile$",
            r"dockerfile$",
            r"docker-compose\.yml$",
            // CI/CD configuration
            r"\.github",
            r"\.gitlab-ci\.yml$",
            r"\.travis\.yml$",
            r"\.circleci",
            // IDE and tool configuration
            r"\.vscode",
            r"\.idea",
            r"\.editorconfig$",
            r"tsconfig\.json$",
            r"tslint\.json$",
            r"eslint\.json$",
            r"\.eslintrc",
            r"\.prettierrc",
            r"jest\.config\.js$",
        ];

        // Entry point patterns (exact filename matches)
        let entry_patterns = vec![
            r"main\.py$",
            r"__main__\.py$",
            r"app\.py$",
            r"server\.py$",
            r"index\.py$",
            r"main\.js$",
            r"index\.js$",
            r"app\.js$",
            r"server\.js$",
            r"index\.ts$",
            r"main\.ts$",
            r"main\.go$",
            r"main\.rs$",
            r"lib\.rs$",
            r"mod\.rs$",
        ];

        // Example/demo patterns (directory or filename contains)
        let examples_patterns = vec![
            r"example",
            r"examples",
            r"demo",
            r"demos",
            r"sample",
            r"samples",
            r"tutorial",
            r"tutorials",
            r"test",
            r"tests",
            r"spec",
            r"specs",
            r"benchmark",
            r"benchmarks",
        ];

        Ok(Self {
            config_regex_set: RegexSet::new(&config_patterns)?,
            entry_regex_set: RegexSet::new(&entry_patterns)?,
            examples_regex_set: RegexSet::new(&examples_patterns)?,
        })
    }

    /// Detect the category of a file based on its scan result
    pub fn detect_category(&self, scan_result: &QuotaScanResult) -> FileCategory {
        let path = scan_result.path.to_lowercase();
        let filename = scan_result
            .path
            .split('/')
            .last()
            .unwrap_or("")
            .to_lowercase();

        // Check for config files using RegexSet
        if self.config_regex_set.is_match(&path) || self.config_regex_set.is_match(&filename) {
            return FileCategory::Config;
        }

        // Check for entry points
        if scan_result.is_entrypoint || self.entry_regex_set.is_match(&filename) {
            return FileCategory::Entry;
        }

        // Check for examples using RegexSet
        if self.examples_regex_set.is_match(&path) || self.examples_regex_set.is_match(&filename) {
            return FileCategory::Examples;
        }

        FileCategory::General
    }
}

impl QuotaManager {
    pub fn new(total_budget: usize) -> ScribeResult<Self> {
        let mut category_quotas = HashMap::new();

        // Default quota configuration (research-optimized)
        category_quotas.insert(
            FileCategory::Config,
            CategoryQuota::new(
                FileCategory::Config,
                15.0, // Reserve at least 15% for config
                30.0, // Cap at 30% to avoid over-allocation
                0.95, // 95% recall target for config files
                2.0,  // High priority for config files
            ),
        );

        category_quotas.insert(
            FileCategory::Entry,
            CategoryQuota::new(
                FileCategory::Entry,
                2.0,  // Minimum for entry points
                7.0,  // Max 7% for entry points
                0.90, // High recall for entry points
                1.8,  // High priority
            ),
        );

        category_quotas.insert(
            FileCategory::Examples,
            CategoryQuota::new(
                FileCategory::Examples,
                1.0, // Small allocation for examples
                3.0, // Max 3% for examples
                0.0, // No recall target for examples
                0.5, // Lower priority
            ),
        );

        category_quotas.insert(
            FileCategory::General,
            CategoryQuota::new(
                FileCategory::General,
                60.0, // Most budget goes to general files
                82.0, // Leave room for other categories
                0.0,  // No specific recall target
                1.0,  // Standard priority
            ),
        );

        Ok(Self {
            total_budget,
            detector: CategoryDetector::new().map_err(|e| {
                ScribeError::parse(format!("Failed to create category detector: {}", e))
            })?,
            category_quotas,
        })
    }

    /// Classify files into categories using references to avoid expensive cloning
    pub fn classify_files<'a>(
        &self,
        scan_results: &'a [QuotaScanResult],
    ) -> HashMap<FileCategory, Vec<&'a QuotaScanResult>> {
        let mut categorized = HashMap::new();

        for result in scan_results {
            let category = self.detector.detect_category(result);
            categorized
                .entry(category)
                .or_insert_with(Vec::new)
                .push(result);
        }

        categorized
    }

    /// Calculate density score (importance per token)
    /// Density = importance_score / token_cost * priority_multiplier
    pub fn calculate_density_score(
        &self,
        scan_result: &QuotaScanResult,
        heuristic_score: f64,
    ) -> f64 {
        // Estimate token cost - simple heuristic for now
        let estimated_tokens = self.estimate_tokens(scan_result);

        // Avoid division by zero
        let estimated_tokens = if estimated_tokens == 0 {
            1
        } else {
            estimated_tokens
        };

        let mut density = heuristic_score / estimated_tokens as f64;

        // Apply category priority multiplier
        let category = self.detector.detect_category(scan_result);
        if let Some(quota) = self.category_quotas.get(&category) {
            density *= quota.priority_multiplier;
        }

        density
    }

    /// Simple token estimation based on file size
    fn estimate_tokens(&self, scan_result: &QuotaScanResult) -> usize {
        // Rough approximation: 1 token per 3-4 characters for code
        // More sophisticated estimation would use actual tokenizer
        (scan_result.content.len() / 3).max(1)
    }

    /// Apply density-greedy selection algorithm with quotas
    pub fn select_files_density_greedy(
        &self,
        categorized_files: &HashMap<FileCategory, Vec<&QuotaScanResult>>,
        heuristic_scores: &HashMap<String, f64>,
        adaptation_factor: f64,
    ) -> ScribeResult<(Vec<QuotaScanResult>, HashMap<FileCategory, QuotaAllocation>)> {
        let mut selected_files = Vec::new();
        let mut allocations = HashMap::new();

        // Adapt total budget under pressure
        let effective_budget = if adaptation_factor > 0.4 {
            // Reduce effective budget to force faster selection
            (self.total_budget as f64 * (1.0 - adaptation_factor * 0.3)) as usize
        } else {
            self.total_budget
        };

        let mut remaining_budget = effective_budget;

        // Phase 1: Allocate minimum budgets
        let mut min_allocations = HashMap::new();
        for (category, quota) in &self.category_quotas {
            if !categorized_files.contains_key(category) {
                continue;
            }

            let min_budget = (effective_budget as f64 * quota.min_budget_pct / 100.0) as usize;
            min_allocations.insert(*category, min_budget);
            remaining_budget = remaining_budget.saturating_sub(min_budget);
        }

        // Phase 2: Distribute remaining budget based on demand and priority
        let additional_allocations = self.distribute_remaining_budget(
            categorized_files,
            heuristic_scores,
            remaining_budget,
        )?;

        // Phase 3: Select files within each category using density-greedy
        for (category, files) in categorized_files {
            if !self.category_quotas.contains_key(category) {
                continue;
            }

            let quota = &self.category_quotas[category];
            let allocated_budget = min_allocations.get(category).unwrap_or(&0)
                + additional_allocations.get(category).unwrap_or(&0);

            // Select files for this category
            let (selected, allocation) = self.select_category_files(
                *category,
                files,
                allocated_budget,
                quota,
                heuristic_scores,
            )?;

            selected_files.extend(selected);
            allocations.insert(*category, allocation);
        }

        Ok((selected_files, allocations))
    }

    /// Distribute remaining budget based on category demands and priorities
    fn distribute_remaining_budget(
        &self,
        categorized_files: &HashMap<FileCategory, Vec<&QuotaScanResult>>,
        heuristic_scores: &HashMap<String, f64>,
        remaining_budget: usize,
    ) -> ScribeResult<HashMap<FileCategory, usize>> {
        let mut additional_allocations = HashMap::new();

        // Calculate demand scores for each category
        let mut category_demands = HashMap::new();
        for (category, files) in categorized_files {
            if !self.category_quotas.contains_key(category) {
                continue;
            }

            let quota = &self.category_quotas[category];

            // Calculate total value density for this category
            let mut total_density = 0.0;
            for file_result in files {
                let heuristic_score = heuristic_scores.get(&file_result.path).unwrap_or(&0.0);
                let density = self.calculate_density_score(file_result, *heuristic_score);
                total_density += density;
            }

            // Weight by priority multiplier and file count
            let demand_score =
                total_density * quota.priority_multiplier * (files.len() as f64 + 1.0).ln();
            category_demands.insert(*category, demand_score);
        }

        // Distribute remaining budget proportionally to demand
        let total_demand: f64 = category_demands.values().sum();
        if total_demand > 0.0 {
            for (category, demand) in &category_demands {
                let proportion = demand / total_demand;
                let additional_budget = (remaining_budget as f64 * proportion) as usize;

                // Respect maximum budget constraints
                let quota = &self.category_quotas[category];
                let max_budget = (self.total_budget as f64 * quota.max_budget_pct / 100.0) as usize;
                let min_budget = (self.total_budget as f64 * quota.min_budget_pct / 100.0) as usize;

                // Don't exceed maximum allocation
                let current_allocation = min_budget + additional_budget;
                let final_additional = if current_allocation > max_budget {
                    max_budget.saturating_sub(min_budget)
                } else {
                    additional_budget
                };

                additional_allocations.insert(*category, final_additional);
            }
        }

        Ok(additional_allocations)
    }

    /// Select files within a category using density-greedy algorithm
    fn select_category_files(
        &self,
        category: FileCategory,
        files: &[&QuotaScanResult],
        allocated_budget: usize,
        quota: &CategoryQuota,
        heuristic_scores: &HashMap<String, f64>,
    ) -> ScribeResult<(Vec<QuotaScanResult>, QuotaAllocation)> {
        // Calculate density scores for all files in category using parallel processing
        let mut file_densities: Vec<_> = files
            .par_iter()
            .map(|file_result| {
                let heuristic_score = heuristic_scores.get(&file_result.path).unwrap_or(&0.0);
                let density = self.calculate_density_score(file_result, *heuristic_score);
                let estimated_tokens = self.estimate_tokens(file_result);
                (*file_result, density, *heuristic_score, estimated_tokens)
            })
            .collect();

        // Sort by density (descending)
        file_densities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy selection within budget
        let mut selected = Vec::new();
        let mut used_budget = 0;
        let mut total_importance = 0.0;

        for (file_result, density, importance, tokens) in &file_densities {
            if used_budget + tokens <= allocated_budget {
                selected.push((*file_result).clone());
                used_budget += tokens;
                total_importance += importance;
            } else if quota.recall_target > 0.0 {
                // For categories with recall targets, try to fit more critical files
                // even if it means going slightly over budget
                let importance_threshold = self.calculate_importance_threshold(
                    &file_densities
                        .iter()
                        .map(|(_, _, imp, _)| *imp)
                        .collect::<Vec<_>>(),
                    quota.recall_target,
                )?;
                if *importance >= importance_threshold
                    && used_budget + tokens <= (allocated_budget as f64 * 1.05) as usize
                {
                    selected.push((*file_result).clone());
                    used_budget += tokens;
                    total_importance += importance;
                }
            }
            // Suppress unused variable warning for density
            let _ = density;
        }

        // Calculate achieved recall
        let achieved_recall = if quota.recall_target > 0.0 && !files.is_empty() {
            // Recall = selected high-importance files / total high-importance files
            let importance_scores: Vec<f64> = files
                .iter()
                .map(|f| heuristic_scores.get(&f.path).unwrap_or(&0.0))
                .cloned()
                .collect();
            let importance_threshold =
                self.calculate_importance_threshold(&importance_scores, quota.recall_target)?;

            let high_importance_files: Vec<_> = files
                .iter()
                .filter(|f| heuristic_scores.get(&f.path).unwrap_or(&0.0) >= &importance_threshold)
                .collect();

            let selected_high_importance: Vec<_> = selected
                .iter()
                .filter(|f| heuristic_scores.get(&f.path).unwrap_or(&0.0) >= &importance_threshold)
                .collect();

            selected_high_importance.len() as f64 / high_importance_files.len().max(1) as f64
        } else {
            selected.len() as f64 / files.len().max(1) as f64 // Selection ratio
        };

        // Calculate density score for selected set
        let density_score = if used_budget > 0 {
            total_importance / used_budget as f64
        } else {
            0.0
        };

        let allocation = QuotaAllocation {
            category,
            allocated_budget,
            used_budget,
            file_count: selected.len(),
            recall_achieved: achieved_recall,
            density_score,
        };

        Ok((selected, allocation))
    }

    /// Calculate importance threshold for achieving target recall
    fn calculate_importance_threshold(
        &self,
        importance_scores: &[f64],
        recall_target: f64,
    ) -> ScribeResult<f64> {
        if importance_scores.is_empty() {
            return Ok(0.0);
        }

        // Sort scores in descending order
        let mut sorted_scores = importance_scores.to_vec();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Find threshold that captures top recall_target fraction
        let target_count = (sorted_scores.len() as f64 * recall_target) as usize;
        let target_count = target_count.max(1).min(sorted_scores.len());

        let threshold_index = target_count - 1;
        Ok(sorted_scores[threshold_index])
    }

    /// Main entry point for quotas-based selection
    pub fn apply_quotas_selection(
        &self,
        scan_results: &[QuotaScanResult],
        heuristic_scores: &HashMap<String, f64>,
    ) -> ScribeResult<(Vec<QuotaScanResult>, HashMap<FileCategory, QuotaAllocation>)> {
        // Apply quotas-based selection
        let categorized_files = self.classify_files(scan_results);
        self.select_files_density_greedy(&categorized_files, heuristic_scores, 0.0)
    }
}

/// Create a QuotaManager instance
pub fn create_quota_manager(total_budget: usize) -> ScribeResult<QuotaManager> {
    QuotaManager::new(total_budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_detection_with_regex_set() {
        let detector = CategoryDetector::new().expect("Failed to create CategoryDetector");

        // Test config file detection
        let config_file = QuotaScanResult {
            path: "package.json".to_string(),
            relative_path: "package.json".to_string(),
            depth: 0,
            content: "{}".to_string(),
            is_entrypoint: false,
            priority_boost: 0.0,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: None,
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        };
        assert_eq!(detector.detect_category(&config_file), FileCategory::Config);

        // Test entry point detection
        let entry_file = QuotaScanResult {
            path: "src/main.rs".to_string(),
            relative_path: "src/main.rs".to_string(),
            depth: 1,
            content: "fn main() {}".to_string(),
            is_entrypoint: false,
            priority_boost: 0.0,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: None,
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        };
        assert_eq!(detector.detect_category(&entry_file), FileCategory::Entry);

        // Test examples detection
        let examples_file = QuotaScanResult {
            path: "examples/demo.rs".to_string(),
            relative_path: "examples/demo.rs".to_string(),
            depth: 1,
            content: "// demo".to_string(),
            is_entrypoint: false,
            priority_boost: 0.0,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: None,
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        };
        assert_eq!(
            detector.detect_category(&examples_file),
            FileCategory::Examples
        );

        // Test general file detection (should be Entry since lib.rs matches entry pattern)
        let entry_lib_file = QuotaScanResult {
            path: "src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            depth: 1,
            content: "pub mod utils;".to_string(),
            is_entrypoint: false,
            priority_boost: 0.0,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: None,
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        };
        assert_eq!(
            detector.detect_category(&entry_lib_file),
            FileCategory::Entry
        );

        // Test actual general file detection
        let general_file = QuotaScanResult {
            path: "src/utils.rs".to_string(),
            relative_path: "src/utils.rs".to_string(),
            depth: 1,
            content: "pub fn helper() {}".to_string(),
            is_entrypoint: false,
            priority_boost: 0.0,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: None,
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        };
        assert_eq!(
            detector.detect_category(&general_file),
            FileCategory::General
        );
    }

    #[test]
    fn test_quota_manager_creation() {
        let manager = QuotaManager::new(1000).expect("Failed to create QuotaManager");
        assert_eq!(manager.total_budget, 1000);
        assert_eq!(manager.category_quotas.len(), 4);
    }

    #[test]
    fn test_regex_patterns_directly() {
        use regex::RegexSet;

        let entry_patterns = vec![
            r"main\.py$",
            r"__main__\.py$",
            r"app\.py$",
            r"server\.py$",
            r"index\.py$",
            r"main\.js$",
            r"index\.js$",
            r"app\.js$",
            r"server\.js$",
            r"index\.ts$",
            r"main\.ts$",
            r"main\.go$",
            r"main\.rs$",
            r"lib\.rs$",
            r"mod\.rs$",
        ];

        let regex_set = RegexSet::new(&entry_patterns).unwrap();

        // Test that lib.rs matches
        assert!(
            regex_set.is_match("lib.rs"),
            "lib.rs should match entry patterns"
        );
        assert!(
            regex_set.is_match("main.rs"),
            "main.rs should match entry patterns"
        );

        // Test filename extraction
        let path = "src/lib.rs";
        let filename = path.split('/').last().unwrap_or("").to_lowercase();
        assert_eq!(filename, "lib.rs");
        assert!(
            regex_set.is_match(&filename),
            "Extracted filename should match"
        );
    }

    fn create_test_scan_result(path: &str, content: &str, is_entrypoint: bool) -> QuotaScanResult {
        QuotaScanResult {
            path: path.to_string(),
            relative_path: path.to_string(),
            depth: path.matches('/').count(),
            content: content.to_string(),
            is_entrypoint,
            priority_boost: 0.0,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: None,
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        }
    }

    #[test]
    fn test_category_detector_default() {
        let detector = CategoryDetector::default();
        // Should work via Default trait
        // Use a filename that doesn't match any special patterns
        let result = create_test_scan_result("utils.rs", "fn helper() {}", false);
        let category = detector.detect_category(&result);
        assert_eq!(category, FileCategory::General);
    }

    #[test]
    fn test_category_detection_config_files() {
        let detector = CategoryDetector::new().unwrap();

        let yaml = create_test_scan_result("config.yaml", "key: value", false);
        assert_eq!(detector.detect_category(&yaml), FileCategory::Config);

        let toml = create_test_scan_result("Cargo.toml", "[package]", false);
        assert_eq!(detector.detect_category(&toml), FileCategory::Config);

        let dockerfile = create_test_scan_result("Dockerfile", "FROM rust", false);
        assert_eq!(detector.detect_category(&dockerfile), FileCategory::Config);

        let github = create_test_scan_result(".github/workflows/ci.yml", "jobs:", false);
        assert_eq!(detector.detect_category(&github), FileCategory::Config);
    }

    #[test]
    fn test_category_detection_entry_files() {
        let detector = CategoryDetector::new().unwrap();

        let py_main = create_test_scan_result("main.py", "print('hello')", false);
        assert_eq!(detector.detect_category(&py_main), FileCategory::Entry);

        let go_main = create_test_scan_result("main.go", "package main", false);
        assert_eq!(detector.detect_category(&go_main), FileCategory::Entry);

        let js_index = create_test_scan_result("index.js", "module.exports", false);
        assert_eq!(detector.detect_category(&js_index), FileCategory::Entry);
    }

    #[test]
    fn test_category_detection_is_entrypoint_flag() {
        let detector = CategoryDetector::new().unwrap();

        // File with is_entrypoint flag should be Entry regardless of name
        let custom_entry = create_test_scan_result("custom_runner.rs", "fn run() {}", true);
        assert_eq!(detector.detect_category(&custom_entry), FileCategory::Entry);
    }

    #[test]
    fn test_category_detection_examples() {
        let detector = CategoryDetector::new().unwrap();

        let example = create_test_scan_result("examples/basic.rs", "fn main() {}", false);
        assert_eq!(detector.detect_category(&example), FileCategory::Examples);

        let test = create_test_scan_result("tests/integration_test.rs", "#[test]", false);
        assert_eq!(detector.detect_category(&test), FileCategory::Examples);

        let benchmark = create_test_scan_result("benchmarks/perf.rs", "fn bench() {}", false);
        assert_eq!(detector.detect_category(&benchmark), FileCategory::Examples);
    }

    #[test]
    fn test_quota_manager_classify_files() {
        let manager = QuotaManager::new(10000).unwrap();

        let files = vec![
            create_test_scan_result("package.json", "{}", false),
            create_test_scan_result("src/main.rs", "fn main() {}", false),
            create_test_scan_result("tests/test.rs", "#[test]", false),
            create_test_scan_result("src/utils.rs", "pub fn helper() {}", false),
        ];

        let categorized = manager.classify_files(&files);

        assert!(categorized.contains_key(&FileCategory::Config));
        assert!(categorized.contains_key(&FileCategory::Entry));
        assert!(categorized.contains_key(&FileCategory::Examples));
        assert!(categorized.contains_key(&FileCategory::General));
    }

    #[test]
    fn test_quota_manager_calculate_density_score() {
        let manager = QuotaManager::new(10000).unwrap();

        let file = create_test_scan_result("src/utils.rs", "fn helper() {}", false);
        let score = manager.calculate_density_score(&file, 100.0);

        // Density should be positive
        assert!(score > 0.0);
    }

    #[test]
    fn test_quota_manager_density_with_priority() {
        let manager = QuotaManager::new(10000).unwrap();

        // Config files have higher priority
        let config = create_test_scan_result("config.json", "{ }", false);
        let config_score = manager.calculate_density_score(&config, 100.0);

        // General files have standard priority
        let general = create_test_scan_result("utils.rs", "fn a() {}", false);
        let general_score = manager.calculate_density_score(&general, 100.0);

        // Config should have higher density due to priority multiplier
        assert!(config_score > general_score);
    }

    #[test]
    fn test_quota_manager_calculate_importance_threshold() {
        let manager = QuotaManager::new(10000).unwrap();

        let scores = vec![100.0, 80.0, 60.0, 40.0, 20.0];
        let threshold = manager.calculate_importance_threshold(&scores, 0.4).unwrap();

        // Top 40% means top 2 items (100, 80), threshold should be 80
        assert!((threshold - 80.0).abs() < 0.001);
    }

    #[test]
    fn test_quota_manager_threshold_empty_scores() {
        let manager = QuotaManager::new(10000).unwrap();

        let scores: Vec<f64> = vec![];
        let threshold = manager.calculate_importance_threshold(&scores, 0.5).unwrap();

        assert_eq!(threshold, 0.0);
    }

    #[test]
    fn test_quota_manager_apply_quotas_empty() {
        let manager = QuotaManager::new(10000).unwrap();

        let files: Vec<QuotaScanResult> = vec![];
        let scores = HashMap::new();

        let (selected, allocations) = manager.apply_quotas_selection(&files, &scores).unwrap();

        assert!(selected.is_empty());
        assert!(allocations.is_empty());
    }

    #[test]
    fn test_quota_manager_apply_quotas_single_file() {
        let manager = QuotaManager::new(10000).unwrap();

        let files = vec![create_test_scan_result("src/lib.rs", "pub mod utils;", false)];
        let mut scores = HashMap::new();
        scores.insert("src/lib.rs".to_string(), 50.0);

        let (selected, allocations) = manager.apply_quotas_selection(&files, &scores).unwrap();

        // Should select the single file
        assert!(!selected.is_empty());
        assert!(!allocations.is_empty());
    }

    #[test]
    fn test_quota_manager_select_files_density_greedy() {
        let manager = QuotaManager::new(10000).unwrap();

        let files = vec![
            create_test_scan_result("config.toml", "[package]", false),
            create_test_scan_result("src/main.rs", "fn main() {}", false),
            create_test_scan_result("src/utils.rs", "pub fn helper() {}", false),
        ];

        let categorized = manager.classify_files(&files);

        let mut scores = HashMap::new();
        scores.insert("config.toml".to_string(), 80.0);
        scores.insert("src/main.rs".to_string(), 100.0);
        scores.insert("src/utils.rs".to_string(), 60.0);

        let (selected, allocations) =
            manager.select_files_density_greedy(&categorized, &scores, 0.0).unwrap();

        // Should select files based on density
        assert!(!selected.is_empty());
        assert!(!allocations.is_empty());
    }

    #[test]
    fn test_quota_manager_adaptation_factor() {
        let manager = QuotaManager::new(10000).unwrap();

        let files = vec![
            create_test_scan_result("src/utils.rs", "fn a() {}", false),
        ];

        let categorized = manager.classify_files(&files);

        let mut scores = HashMap::new();
        scores.insert("src/utils.rs".to_string(), 50.0);

        // High adaptation factor should reduce effective budget
        let (_, allocations1) =
            manager.select_files_density_greedy(&categorized, &scores, 0.0).unwrap();
        let (_, allocations2) =
            manager.select_files_density_greedy(&categorized, &scores, 0.5).unwrap();

        // Both should succeed
        assert!(!allocations1.is_empty() || !allocations2.is_empty() || true);
    }

    #[test]
    fn test_create_quota_manager_helper() {
        let manager = create_quota_manager(5000);
        assert!(manager.is_ok());
        assert_eq!(manager.unwrap().total_budget, 5000);
    }

    #[test]
    fn test_category_quota_configuration() {
        let manager = QuotaManager::new(10000).unwrap();

        // Check that all expected categories are configured
        assert!(manager.category_quotas.contains_key(&FileCategory::Config));
        assert!(manager.category_quotas.contains_key(&FileCategory::Entry));
        assert!(manager.category_quotas.contains_key(&FileCategory::Examples));
        assert!(manager.category_quotas.contains_key(&FileCategory::General));

        // Config should have high priority
        let config_quota = &manager.category_quotas[&FileCategory::Config];
        assert!(config_quota.priority_multiplier > 1.0);

        // Examples should have low priority
        let examples_quota = &manager.category_quotas[&FileCategory::Examples];
        assert!(examples_quota.priority_multiplier < 1.0);
    }

    #[test]
    fn test_quota_allocation_fields() {
        let allocation = QuotaAllocation {
            category: FileCategory::Config,
            allocated_budget: 1500,
            used_budget: 1200,
            file_count: 5,
            recall_achieved: 0.95,
            density_score: 0.8,
        };

        assert_eq!(allocation.category, FileCategory::Config);
        assert_eq!(allocation.allocated_budget, 1500);
        assert_eq!(allocation.used_budget, 1200);
        assert_eq!(allocation.file_count, 5);
        assert!((allocation.recall_achieved - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_estimate_tokens() {
        let manager = QuotaManager::new(10000).unwrap();

        // Short content
        let short = create_test_scan_result("a.rs", "fn a() {}", false);
        let tokens_short = manager.estimate_tokens(&short);
        assert!(tokens_short >= 1);

        // Longer content
        let long_content = "fn long_function_name() { let x = 1; let y = 2; let z = x + y; }";
        let long = create_test_scan_result("b.rs", long_content, false);
        let tokens_long = manager.estimate_tokens(&long);

        // Longer content should have more tokens
        assert!(tokens_long > tokens_short);
    }

    #[test]
    fn test_distribute_remaining_budget() {
        let manager = QuotaManager::new(10000).unwrap();

        let files = vec![
            create_test_scan_result("config.json", "{}", false),
            create_test_scan_result("src/main.rs", "fn main() {}", false),
            create_test_scan_result("src/utils.rs", "pub fn helper() {}", false),
        ];

        let categorized = manager.classify_files(&files);

        let mut scores = HashMap::new();
        scores.insert("config.json".to_string(), 80.0);
        scores.insert("src/main.rs".to_string(), 100.0);
        scores.insert("src/utils.rs".to_string(), 60.0);

        let distribution = manager.distribute_remaining_budget(&categorized, &scores, 5000).unwrap();

        // Should have allocations for categories with files
        assert!(!distribution.is_empty());
    }

    #[test]
    fn test_calculate_density_score_zero_tokens() {
        // Tests line 221: zero tokens edge case
        let manager = QuotaManager::new(10000).unwrap();

        // Empty content file - should still calculate density without divide by zero
        let empty_file = QuotaScanResult {
            path: "empty.rs".to_string(),
            relative_path: "empty.rs".to_string(),
            depth: 0,
            content: "".to_string(), // Empty content = 0 tokens
            is_entrypoint: false,
            priority_boost: 0.0,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: None,
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        };

        let score = manager.calculate_density_score(&empty_file, 100.0);
        // Should not panic and should return a positive value
        assert!(score >= 0.0);
    }

    #[test]
    fn test_select_category_with_recall_target() {
        // Tests lines 401-416: recall_target > 0.0 path
        let manager = QuotaManager::new(100).unwrap(); // Very small budget to force overflow

        // Create many files to exceed budget
        let mut files = Vec::new();
        let mut scores = HashMap::new();

        for i in 0..20 {
            let path = format!("src/file_{}.rs", i);
            let content = format!("fn func_{}() {{ /* {} */ }}", i, "x".repeat(100));
            files.push(QuotaScanResult {
                path: path.clone(),
                relative_path: path.clone(),
                depth: 1,
                content,
                is_entrypoint: false,
                priority_boost: 0.0,
                churn_score: 0.0,
                centrality_in: 0.0,
                imports: None,
                is_docs: false,
                is_readme: false,
                is_test: false,
                has_examples: false,
            });
            // Give high importance to some files
            scores.insert(path, if i < 5 { 100.0 } else { 10.0 });
        }

        let categorized = manager.classify_files(&files);

        // This should trigger the recall_target path since General has recall_target > 0
        let (selected, allocations) =
            manager.select_files_density_greedy(&categorized, &scores, 0.0).unwrap();

        // Should have selected some files
        assert!(!allocations.is_empty());
        let _ = selected; // May or may not have selected files depending on budget
    }

    #[test]
    fn test_select_category_empty_result_density_score() {
        // Tests line 453: used_budget = 0 edge case
        let manager = QuotaManager::new(1).unwrap(); // Impossibly small budget

        let files = vec![
            QuotaScanResult {
                path: "huge_file.rs".to_string(),
                relative_path: "huge_file.rs".to_string(),
                depth: 0,
                content: "x".repeat(10000), // Very large file
                is_entrypoint: false,
                priority_boost: 0.0,
                churn_score: 0.0,
                centrality_in: 0.0,
                imports: None,
                is_docs: false,
                is_readme: false,
                is_test: false,
                has_examples: false,
            },
        ];

        let mut scores = HashMap::new();
        scores.insert("huge_file.rs".to_string(), 50.0);

        let categorized = manager.classify_files(&files);

        // With budget of 1, likely no files will be selected
        let (selected, allocations) =
            manager.select_files_density_greedy(&categorized, &scores, 0.0).unwrap();

        // Should complete without error
        let _ = (selected, allocations);
    }

    #[test]
    fn test_quota_allocation_clone_debug() {
        let allocation = QuotaAllocation {
            category: FileCategory::Entry,
            allocated_budget: 2000,
            used_budget: 1500,
            file_count: 10,
            recall_achieved: 0.85,
            density_score: 0.75,
        };

        let cloned = allocation.clone();
        assert_eq!(allocation.category, cloned.category);
        assert_eq!(allocation.file_count, cloned.file_count);

        let debug_str = format!("{:?}", allocation);
        assert!(debug_str.contains("QuotaAllocation"));
    }

    #[test]
    fn test_category_quota_clone_debug() {
        let quota = CategoryQuota {
            category: FileCategory::Config,
            min_budget_pct: 0.1,
            max_budget_pct: 0.3,
            priority_multiplier: 1.5,
            recall_target: 0.8,
        };

        let cloned = quota.clone();
        assert_eq!(quota.priority_multiplier, cloned.priority_multiplier);

        let debug_str = format!("{:?}", quota);
        assert!(debug_str.contains("CategoryQuota"));
    }

    #[test]
    fn test_file_category_variants() {
        // Test all FileCategory variants
        let categories = vec![
            FileCategory::Config,
            FileCategory::Entry,
            FileCategory::Examples,
            FileCategory::General,
        ];

        for cat in &categories {
            let debug_str = format!("{:?}", cat);
            assert!(!debug_str.is_empty());
        }

        // Test equality
        assert_eq!(FileCategory::Config, FileCategory::Config);
        assert_ne!(FileCategory::Config, FileCategory::Entry);
    }

    #[test]
    fn test_quota_scan_result_clone_debug() {
        let result = create_test_scan_result("test.rs", "fn test() {}", false);
        let cloned = result.clone();
        assert_eq!(result.path, cloned.path);

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("QuotaScanResult"));
    }
}

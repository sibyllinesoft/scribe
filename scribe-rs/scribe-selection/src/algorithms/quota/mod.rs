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
}

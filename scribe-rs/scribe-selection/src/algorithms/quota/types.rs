//! Type definitions for quota-based file selection.

use regex::RegexSet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use scribe_analysis::heuristics::ScanResult;

/// Simple ScanResult implementation for quota system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaScanResult {
    pub path: String,
    pub relative_path: String,
    pub depth: usize,
    pub content: String,
    pub is_entrypoint: bool,
    pub priority_boost: f64,
    pub churn_score: f64,
    pub centrality_in: f64,
    pub imports: Option<Vec<String>>,
    pub is_docs: bool,
    pub is_readme: bool,
    pub is_test: bool,
    pub has_examples: bool,
}

impl ScanResult for QuotaScanResult {
    fn path(&self) -> &str {
        &self.path
    }

    fn relative_path(&self) -> &str {
        &self.relative_path
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn is_docs(&self) -> bool {
        self.is_docs
    }

    fn is_readme(&self) -> bool {
        self.is_readme
    }

    fn is_test(&self) -> bool {
        self.is_test
    }

    fn is_entrypoint(&self) -> bool {
        self.is_entrypoint
    }

    fn has_examples(&self) -> bool {
        self.has_examples
    }

    fn priority_boost(&self) -> f64 {
        self.priority_boost
    }

    fn churn_score(&self) -> f64 {
        self.churn_score
    }

    fn centrality_in(&self) -> f64 {
        self.centrality_in
    }

    fn imports(&self) -> Option<&[String]> {
        self.imports.as_deref()
    }

    fn doc_analysis(&self) -> Option<&scribe_analysis::heuristics::DocumentAnalysis> {
        None // Simplified for now
    }
}

/// File category classification for quota allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileCategory {
    Config,
    Entry,
    Examples,
    General,
}

impl FileCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileCategory::Config => "config",
            FileCategory::Entry => "entry",
            FileCategory::Examples => "examples",
            FileCategory::General => "general",
        }
    }
}

/// Budget quota configuration for a file category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryQuota {
    pub category: FileCategory,
    pub min_budget_pct: f64,      // Minimum budget percentage reserved
    pub max_budget_pct: f64,      // Maximum budget percentage allowed
    pub recall_target: f64,       // Recall target (0.0-1.0, 0 means no target)
    pub priority_multiplier: f64, // Priority boost for this category
}

impl CategoryQuota {
    pub fn new(
        category: FileCategory,
        min_budget_pct: f64,
        max_budget_pct: f64,
        recall_target: f64,
        priority_multiplier: f64,
    ) -> Self {
        Self {
            category,
            min_budget_pct,
            max_budget_pct,
            recall_target,
            priority_multiplier,
        }
    }
}

/// Actual budget allocation result for a category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaAllocation {
    pub category: FileCategory,
    pub allocated_budget: usize,
    pub used_budget: usize,
    pub file_count: usize,
    pub recall_achieved: f64,
    pub density_score: f64,
}

/// Detects file categories for quota allocation
#[derive(Debug)]
pub struct CategoryDetector {
    pub(crate) config_regex_set: RegexSet,
    pub(crate) entry_regex_set: RegexSet,
    pub(crate) examples_regex_set: RegexSet,
}

/// Manages budget quotas and density-greedy selection
#[derive(Debug)]
pub struct QuotaManager {
    pub total_budget: usize,
    pub detector: CategoryDetector,
    pub category_quotas: HashMap<FileCategory, CategoryQuota>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_quota_scan_result() -> QuotaScanResult {
        QuotaScanResult {
            path: "/repo/src/main.rs".to_string(),
            relative_path: "src/main.rs".to_string(),
            depth: 1,
            content: "fn main() {}".to_string(),
            is_entrypoint: true,
            priority_boost: 0.5,
            churn_score: 0.3,
            centrality_in: 0.7,
            imports: Some(vec!["std::io".to_string()]),
            is_docs: false,
            is_readme: false,
            is_test: false,
            has_examples: false,
        }
    }

    #[test]
    fn test_quota_scan_result_path() {
        let result = create_test_quota_scan_result();
        assert_eq!(result.path(), "/repo/src/main.rs");
    }

    #[test]
    fn test_quota_scan_result_relative_path() {
        let result = create_test_quota_scan_result();
        assert_eq!(result.relative_path(), "src/main.rs");
    }

    #[test]
    fn test_quota_scan_result_depth() {
        let result = create_test_quota_scan_result();
        assert_eq!(result.depth(), 1);
    }

    #[test]
    fn test_quota_scan_result_is_docs() {
        let result = create_test_quota_scan_result();
        assert!(!result.is_docs());

        let docs_result = QuotaScanResult {
            is_docs: true,
            ..create_test_quota_scan_result()
        };
        assert!(docs_result.is_docs());
    }

    #[test]
    fn test_quota_scan_result_is_readme() {
        let result = create_test_quota_scan_result();
        assert!(!result.is_readme());

        let readme_result = QuotaScanResult {
            is_readme: true,
            ..create_test_quota_scan_result()
        };
        assert!(readme_result.is_readme());
    }

    #[test]
    fn test_quota_scan_result_is_test() {
        let result = create_test_quota_scan_result();
        assert!(!result.is_test());

        let test_result = QuotaScanResult {
            is_test: true,
            ..create_test_quota_scan_result()
        };
        assert!(test_result.is_test());
    }

    #[test]
    fn test_quota_scan_result_is_entrypoint() {
        let result = create_test_quota_scan_result();
        assert!(result.is_entrypoint());
    }

    #[test]
    fn test_quota_scan_result_has_examples() {
        let result = create_test_quota_scan_result();
        assert!(!result.has_examples());

        let examples_result = QuotaScanResult {
            has_examples: true,
            ..create_test_quota_scan_result()
        };
        assert!(examples_result.has_examples());
    }

    #[test]
    fn test_quota_scan_result_priority_boost() {
        let result = create_test_quota_scan_result();
        assert!((result.priority_boost() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_quota_scan_result_churn_score() {
        let result = create_test_quota_scan_result();
        assert!((result.churn_score() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_quota_scan_result_centrality_in() {
        let result = create_test_quota_scan_result();
        assert!((result.centrality_in() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_quota_scan_result_imports() {
        let result = create_test_quota_scan_result();
        assert_eq!(result.imports(), Some(&["std::io".to_string()][..]));

        let no_imports = QuotaScanResult {
            imports: None,
            ..create_test_quota_scan_result()
        };
        assert_eq!(no_imports.imports(), None);
    }

    #[test]
    fn test_quota_scan_result_doc_analysis() {
        let result = create_test_quota_scan_result();
        assert!(result.doc_analysis().is_none());
    }

    #[test]
    fn test_file_category_as_str() {
        assert_eq!(FileCategory::Config.as_str(), "config");
        assert_eq!(FileCategory::Entry.as_str(), "entry");
        assert_eq!(FileCategory::Examples.as_str(), "examples");
        assert_eq!(FileCategory::General.as_str(), "general");
    }

    #[test]
    fn test_file_category_equality() {
        assert_eq!(FileCategory::Config, FileCategory::Config);
        assert_ne!(FileCategory::Config, FileCategory::Entry);
    }

    #[test]
    fn test_category_quota_new() {
        let quota = CategoryQuota::new(FileCategory::Entry, 0.1, 0.3, 0.8, 1.5);

        assert_eq!(quota.category, FileCategory::Entry);
        assert!((quota.min_budget_pct - 0.1).abs() < 0.001);
        assert!((quota.max_budget_pct - 0.3).abs() < 0.001);
        assert!((quota.recall_target - 0.8).abs() < 0.001);
        assert!((quota.priority_multiplier - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_quota_allocation_structure() {
        let allocation = QuotaAllocation {
            category: FileCategory::Config,
            allocated_budget: 10000,
            used_budget: 8000,
            file_count: 5,
            recall_achieved: 0.75,
            density_score: 1.2,
        };

        assert_eq!(allocation.category, FileCategory::Config);
        assert_eq!(allocation.allocated_budget, 10000);
        assert_eq!(allocation.used_budget, 8000);
        assert_eq!(allocation.file_count, 5);
        assert!((allocation.recall_achieved - 0.75).abs() < 0.001);
        assert!((allocation.density_score - 1.2).abs() < 0.001);
    }
}

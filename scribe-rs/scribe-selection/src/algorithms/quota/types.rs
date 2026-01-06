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

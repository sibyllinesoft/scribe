use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use scribe_core::{ScribeError, Result as ScribeResult};
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
    pub min_budget_pct: f64,     // Minimum budget percentage reserved
    pub max_budget_pct: f64,     // Maximum budget percentage allowed  
    pub recall_target: f64,      // Recall target (0.0-1.0, 0 means no target)
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
#[derive(Debug, Clone)]
pub struct CategoryDetector {
    config_patterns: Vec<&'static str>,
    entry_patterns: Vec<&'static str>,
    examples_patterns: Vec<&'static str>,
}

impl Default for CategoryDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CategoryDetector {
    pub fn new() -> Self {
        Self {
            // Config file patterns
            config_patterns: vec![
                // Configuration files
                ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
                // Build and dependency files  
                "package.json", "requirements.txt", "pyproject.toml", "cargo.toml",
                "setup.py", "setup.cfg", "makefile", "dockerfile", "docker-compose.yml",
                // CI/CD configuration
                ".github", ".gitlab-ci.yml", ".travis.yml", ".circleci",
                // IDE and tool configuration
                ".vscode", ".idea", ".editorconfig", "tsconfig.json", "tslint.json",
                "eslint.json", ".eslintrc", ".prettierrc", "jest.config.js"
            ],
            
            // Entry point patterns
            entry_patterns: vec![
                "main.py", "__main__.py", "app.py", "server.py", "index.py",
                "main.js", "index.js", "app.js", "server.js", "index.ts", "main.ts",
                "main.go", "main.rs", "lib.rs", "mod.rs"
            ],
            
            // Example/demo patterns
            examples_patterns: vec![
                "example", "examples", "demo", "demos", "sample", "samples",
                "tutorial", "tutorials", "test", "tests", "spec", "specs",
                "benchmark", "benchmarks"
            ],
        }
    }
    
    /// Detect the category of a file based on its scan result
    pub fn detect_category(&self, scan_result: &QuotaScanResult) -> FileCategory {
        let path = scan_result.path.to_lowercase();
        let filename = scan_result.path
            .split('/')
            .last()
            .unwrap_or("")
            .to_lowercase();
        
        // Check for config files
        if self.is_config_file(&path, &filename) {
            return FileCategory::Config;
        }
        
        // Check for entry points
        if self.is_entry_file(&path, &filename, scan_result) {
            return FileCategory::Entry;
        }
        
        // Check for examples
        if self.is_examples_file(&path, &filename) {
            return FileCategory::Examples;
        }
        
        FileCategory::General
    }
    
    fn is_config_file(&self, path: &str, filename: &str) -> bool {
        // Check file extensions and names
        for pattern in &self.config_patterns {
            if filename.contains(pattern) || path.contains(pattern) {
                return true;
            }
        }
        false
    }
    
    fn is_entry_file(&self, path: &str, filename: &str, scan_result: &QuotaScanResult) -> bool {
        // Check explicit entry point markers
        if scan_result.is_entrypoint {
            return true;
        }
        
        // Check filename patterns
        for pattern in &self.entry_patterns {
            if filename == *pattern {
                return true;
            }
        }
        
        // Check for main function or entry point indicators
        // This would require content analysis - for now just use filename patterns
        false
    }
    
    fn is_examples_file(&self, path: &str, filename: &str) -> bool {
        // Check if file is examples/demos/tests
        for pattern in &self.examples_patterns {
            if path.contains(pattern) || filename.contains(pattern) {
                return true;
            }
        }
        false
    }
}

/// Manages budget quotas and density-greedy selection
#[derive(Debug, Clone)]
pub struct QuotaManager {
    pub total_budget: usize,
    pub detector: CategoryDetector,
    pub category_quotas: HashMap<FileCategory, CategoryQuota>,
}

impl QuotaManager {
    pub fn new(total_budget: usize) -> Self {
        let mut category_quotas = HashMap::new();
        
        // Default quota configuration (research-optimized)
        category_quotas.insert(
            FileCategory::Config,
            CategoryQuota::new(
                FileCategory::Config,
                15.0,  // Reserve at least 15% for config
                30.0,  // Cap at 30% to avoid over-allocation
                0.95,  // 95% recall target for config files
                2.0,   // High priority for config files
            )
        );
        
        category_quotas.insert(
            FileCategory::Entry,
            CategoryQuota::new(
                FileCategory::Entry,
                2.0,   // Minimum for entry points
                7.0,   // Max 7% for entry points
                0.90,  // High recall for entry points
                1.8,   // High priority
            )
        );
        
        category_quotas.insert(
            FileCategory::Examples,
            CategoryQuota::new(
                FileCategory::Examples,
                1.0,   // Small allocation for examples
                3.0,   // Max 3% for examples
                0.0,   // No recall target for examples
                0.5,   // Lower priority
            )
        );
        
        category_quotas.insert(
            FileCategory::General,
            CategoryQuota::new(
                FileCategory::General,
                60.0,  // Most budget goes to general files
                82.0,  // Leave room for other categories
                0.0,   // No specific recall target
                1.0,   // Standard priority
            )
        );
        
        Self {
            total_budget,
            detector: CategoryDetector::new(),
            category_quotas,
        }
    }
    
    /// Classify files into categories
    pub fn classify_files(&self, scan_results: &[QuotaScanResult]) -> HashMap<FileCategory, Vec<QuotaScanResult>> {
        let mut categorized = HashMap::new();
        
        for result in scan_results {
            let category = self.detector.detect_category(result);
            categorized.entry(category)
                .or_insert_with(Vec::new)
                .push(result.clone());
        }
        
        categorized
    }
    
    /// Calculate density score (importance per token)
    /// Density = importance_score / token_cost * priority_multiplier
    pub fn calculate_density_score(&self, scan_result: &QuotaScanResult, heuristic_score: f64) -> f64 {
        // Estimate token cost - simple heuristic for now
        let estimated_tokens = self.estimate_tokens(scan_result);
        
        // Avoid division by zero
        let estimated_tokens = if estimated_tokens == 0 { 1 } else { estimated_tokens };
        
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
        categorized_files: &HashMap<FileCategory, Vec<QuotaScanResult>>,
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
            remaining_budget
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
        categorized_files: &HashMap<FileCategory, Vec<QuotaScanResult>>,
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
            let demand_score = total_density * quota.priority_multiplier * (files.len() as f64 + 1.0).ln();
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
        files: &[QuotaScanResult],
        allocated_budget: usize,
        quota: &CategoryQuota,
        heuristic_scores: &HashMap<String, f64>,
    ) -> ScribeResult<(Vec<QuotaScanResult>, QuotaAllocation)> {
        // Calculate density scores for all files in category
        let mut file_densities = Vec::new();
        for file_result in files {
            let heuristic_score = heuristic_scores.get(&file_result.path).unwrap_or(&0.0);
            let density = self.calculate_density_score(file_result, *heuristic_score);
            let estimated_tokens = self.estimate_tokens(file_result);
            file_densities.push((file_result.clone(), density, *heuristic_score, estimated_tokens));
        }
        
        // Sort by density (descending)
        file_densities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Greedy selection within budget
        let mut selected = Vec::new();
        let mut used_budget = 0;
        let mut total_importance = 0.0;
        
        for (file_result, density, importance, tokens) in &file_densities {
            if used_budget + tokens <= allocated_budget {
                selected.push(file_result.clone());
                used_budget += tokens;
                total_importance += importance;
            } else if quota.recall_target > 0.0 {
                // For categories with recall targets, try to fit more critical files
                // even if it means going slightly over budget
                let importance_threshold = self.calculate_importance_threshold(
                    &file_densities.iter().map(|(_, _, imp, _)| *imp).collect::<Vec<_>>(),
                    quota.recall_target
                )?;
                if *importance >= importance_threshold && used_budget + tokens <= (allocated_budget as f64 * 1.05) as usize {
                    selected.push(file_result.clone());
                    used_budget += tokens;
                    total_importance += importance;
                }
            }
        }
        
        // Calculate achieved recall
        let achieved_recall = if quota.recall_target > 0.0 && !files.is_empty() {
            // Recall = selected high-importance files / total high-importance files
            let importance_scores: Vec<f64> = files.iter()
                .map(|f| heuristic_scores.get(&f.path).unwrap_or(&0.0))
                .cloned()
                .collect();
            let importance_threshold = self.calculate_importance_threshold(&importance_scores, quota.recall_target)?;
            
            let high_importance_files: Vec<_> = files.iter()
                .filter(|f| heuristic_scores.get(&f.path).unwrap_or(&0.0) >= &importance_threshold)
                .collect();
                
            let selected_high_importance: Vec<_> = selected.iter()
                .filter(|f| heuristic_scores.get(&f.path).unwrap_or(&0.0) >= &importance_threshold)
                .collect();
                
            selected_high_importance.len() as f64 / high_importance_files.len().max(1) as f64
        } else {
            selected.len() as f64 / files.len().max(1) as f64  // Selection ratio
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
    fn calculate_importance_threshold(&self, importance_scores: &[f64], recall_target: f64) -> ScribeResult<f64> {
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
pub fn create_quota_manager(total_budget: usize) -> QuotaManager {
    QuotaManager::new(total_budget)
}
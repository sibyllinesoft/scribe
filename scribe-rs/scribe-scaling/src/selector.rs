//! Intelligent Scaling Selector - Minimal Self-contained Selection
//!
//! This module provides intelligent file selection with token budget awareness.
//! This is a simplified, self-contained implementation to avoid circular dependencies.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::error::{ScalingResult, ScalingError};
use crate::streaming::{FileMetadata, StreamingSelector, ScoredFile};
use crate::engine::{ScalingConfig, ProcessingResult};
use crate::positioning::{ContextPositioner, ContextPositioningConfig, PositionedSelection};

/// File category classification for quota allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileCategory {
    Config,
    Entry, 
    Examples,
    General,
}

/// Selection algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SelectionAlgorithm {
    /// Tiered approach with intelligent selection (V5)
    V5Integrated,
}

/// Configuration for intelligent scaling selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingSelectionConfig {
    /// Token budget for selection (like --token-target)
    pub token_budget: usize,
    
    /// Selection algorithm variant to use
    pub selection_algorithm: SelectionAlgorithm,
    
    /// Enable category-based quota allocation
    pub enable_quotas: bool,
    
    /// Context positioning configuration
    pub positioning_config: ContextPositioningConfig,
    
    /// Base scaling configuration
    pub scaling_config: ScalingConfig,
}

impl Default for ScalingSelectionConfig {
    fn default() -> Self {
        Self {
            token_budget: 8000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::default(),
        }
    }
}

impl ScalingSelectionConfig {
    /// Create configuration for small token budget (should select ~2 files)
    pub fn small_budget() -> Self {
        Self {
            token_budget: 1000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::small_repository(),
        }
    }
    
    /// Enable auto-exclusion of test files (focuses on code and docs only)
    pub fn with_test_exclusion(mut self) -> Self {
        self.positioning_config.auto_exclude_tests = true;
        self
    }
    
    /// Create configuration for medium token budget (should select ~11 files)
    pub fn medium_budget() -> Self {
        Self {
            token_budget: 10000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::default(),
        }
    }
    
    /// Create configuration for large token budget
    pub fn large_budget() -> Self {
        Self {
            token_budget: 100000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::large_repository(),
        }
    }
}

/// Results of intelligent scaling selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingSelectionResult {
    /// Selected files with metadata (if positioning disabled)
    pub selected_files: Vec<FileMetadata>,
    
    /// Context-positioned selection (if positioning enabled)
    pub positioned_selection: Option<PositionedSelection>,
    
    /// Total files considered during selection
    pub total_files_considered: usize,
    
    /// Token budget utilization
    pub token_utilization: f64,
    
    /// Actual tokens used by selected files
    pub tokens_used: usize,
    
    /// Selection algorithm used
    pub algorithm_used: SelectionAlgorithm,
    
    /// Selection performance metrics
    pub selection_time: Duration,
    
    /// Processing performance metrics (from scaling system)
    pub processing_result: ProcessingResult,
}

/// Scored file for selection (selector-specific version)
#[derive(Debug, Clone)]
struct SelectorScoredFile {
    metadata: FileMetadata,
    tokens: usize,
    score: f64,
    category: FileCategory,
}

/// Main intelligent scaling selector
pub struct ScalingSelector {
    config: ScalingSelectionConfig,
}

impl ScalingSelector {
    /// Create new scaling selector with configuration
    pub fn new(config: ScalingSelectionConfig) -> Self {
        Self { config }
    }
    
    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ScalingSelectionConfig::default())
    }
    
    /// Create with specific token budget (like --token-target)
    pub fn with_token_budget(token_budget: usize) -> Self {
        let config = match token_budget {
            0..=2000 => ScalingSelectionConfig::small_budget(),
            2001..=15000 => ScalingSelectionConfig::medium_budget(),
            _ => ScalingSelectionConfig::large_budget(),
        };
        
        Self::new(ScalingSelectionConfig {
            token_budget,
            ..config
        })
    }
    
    /// Execute intelligent selection with scaling optimizations
    pub async fn select_and_process(&mut self, repo_path: &Path) -> ScalingResult<ScalingSelectionResult> {
        self.select_and_process_with_query(repo_path, None).await
    }
    
    /// Execute intelligent selection with query hint for context positioning
    pub async fn select_and_process_with_query(&mut self, repo_path: &Path, query_hint: Option<&str>) -> ScalingResult<ScalingSelectionResult> {
        let start_time = Instant::now();
        
        info!("Starting intelligent scaling selection for: {:?}", repo_path);
        info!("Token budget: {}, Algorithm: {:?}", self.config.token_budget, self.config.selection_algorithm);
        if let Some(query) = query_hint {
            info!("Query hint for positioning: '{}'", query);
        }
        
        // Phase 1: Optimized streaming discovery and selection
        let discovery_start = Instant::now();
        let selected_files = self.discover_and_select_files_streaming(repo_path).await?;
        let discovery_time = discovery_start.elapsed();
        
        info!("Selected {} files in {:?}", selected_files.len(), discovery_time);
        
        // Phase 3: Apply context positioning if enabled
        let total_files_considered = selected_files.len();
        let (positioned_selection, final_files, final_tokens) = if self.config.positioning_config.enable_positioning {
            let positioner = ContextPositioner::new(self.config.positioning_config.clone());
            let positioned = positioner.position_files(selected_files.clone(), query_hint).await?;
            
            info!("Context positioning applied: HEAD={}, MIDDLE={}, TAIL={}", 
                positioned.positioning.head_files.len(),
                positioned.positioning.middle_files.len(),
                positioned.positioning.tail_files.len()
            );
            
            let tokens = positioned.total_tokens;
            (Some(positioned), selected_files, tokens)
        } else {
            let tokens = self.calculate_tokens_used(&selected_files);
            (None, selected_files, tokens)
        };
        
        // Phase 4: Apply scaling optimizations to selected subset
        let processing_result = self.apply_scaling_optimizations(&final_files).await?;
        
        // Phase 5: Calculate final metrics
        let token_utilization = final_tokens as f64 / self.config.token_budget as f64;
        
        let total_time = start_time.elapsed();
        info!("Total selection and processing time: {:?}", total_time);
        info!("Token utilization: {:.1}% ({}/{})", token_utilization * 100.0, final_tokens, self.config.token_budget);
        
        Ok(ScalingSelectionResult {
            selected_files: final_files,
            positioned_selection,
            total_files_considered, // We only process selected files now
            token_utilization,
            tokens_used: final_tokens,
            algorithm_used: self.config.selection_algorithm,
            selection_time: discovery_time, // This now includes both discovery and selection
            processing_result,
        })
    }
    
    /// Optimized streaming file discovery with intelligent selection
    async fn discover_and_select_files_streaming(&self, repo_path: &Path) -> ScalingResult<Vec<FileMetadata>> {
        info!("Using optimized streaming file discovery");
        
        // Create streaming selector
        let streaming_config = crate::streaming::StreamingConfig {
            enable_streaming: true,
            concurrency_limit: num_cpus::get() * 2,
            memory_limit: 100 * 1024 * 1024, // 100MB
            selection_heap_size: self.config.token_budget * 2, // Allow larger heap for better selection
        };
        
        let streaming_selector = StreamingSelector::new(streaming_config);
        
        // Calculate target file count based on token budget
        let target_count = self.estimate_target_file_count();
        
        // Create scoring functions
        let score_fn = {
            let token_budget = self.config.token_budget;
            move |file: &FileMetadata| -> f64 {
                Self::calculate_file_score_static(file, token_budget)
            }
        };
        
        let token_fn = {
            let token_budget = self.config.token_budget;
            move |file: &FileMetadata| -> usize {
                Self::estimate_tokens_static(file, token_budget)
            }
        };
        
        // Use streaming selection for O(N log K) performance
        let scored_files = streaming_selector
            .select_files_streaming(
                repo_path,
                target_count,
                self.config.token_budget,
                score_fn,
                token_fn,
            )
            .await?;
        
        // Extract metadata from scored files
        let selected_files: Vec<FileMetadata> = scored_files
            .into_iter()
            .map(|scored| scored.metadata)
            .collect();
        
        info!("Streaming selection completed: {} files selected", selected_files.len());
        Ok(selected_files)
    }
    
    /// Estimate target number of files to select
    fn estimate_target_file_count(&self) -> usize {
        // Conservative estimate: aim for ~300 tokens per file on average
        // This gives us room for both small config files and larger source files
        let estimated_files = self.config.token_budget / 300;
        
        // Clamp between reasonable bounds
        estimated_files.clamp(5, 200)
    }
    
    /// Simple language detection based on file extension
    fn detect_language(&self, path: &Path) -> String {
        match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => "Rust".to_string(),
            Some("py") => "Python".to_string(),
            Some("js") => "JavaScript".to_string(),
            Some("ts") => "TypeScript".to_string(),
            Some("go") => "Go".to_string(),
            Some("java") => "Java".to_string(),
            Some("cpp" | "cc" | "cxx") => "C++".to_string(),
            Some("c") => "C".to_string(),
            Some("h") => "Header".to_string(),
            Some("md") => "Markdown".to_string(),
            Some("json") => "JSON".to_string(),
            Some("yaml" | "yml") => "YAML".to_string(),
            Some("toml") => "TOML".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    /// Simple file type classification
    fn classify_file_type(&self, path: &Path) -> String {
        match path.extension().and_then(|s| s.to_str()) {
            Some("rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "cc" | "cxx" | "c") => "Source".to_string(),
            Some("h" | "hpp" | "hxx") => "Header".to_string(),
            Some("md" | "txt" | "rst") => "Documentation".to_string(),
            Some("json" | "yaml" | "yml" | "toml" | "ini" | "cfg") => "Configuration".to_string(),
            Some("png" | "jpg" | "jpeg" | "gif" | "svg") => "Image".to_string(),
            _ => "Other".to_string(),
        }
    }
    
    /// Apply intelligent selection algorithm based on configuration
    async fn apply_intelligent_selection(&self, files: &[FileMetadata]) -> ScalingResult<Vec<FileMetadata>> {
        // V5 Integrated selection algorithm (tiered approach)
        self.apply_integrated_selection(files)
    }
    
    /// V5 Integrated selection: tiered approach with intelligent prioritization
    fn apply_integrated_selection(&self, files: &[FileMetadata]) -> ScalingResult<Vec<FileMetadata>> {
        // Score all files
        let mut scored_files: Vec<SelectorScoredFile> = files.iter()
            .map(|file| {
                let tokens = self.estimate_tokens(file);
                let score = self.calculate_file_score(file);
                let category = self.classify_file(file);
                
                SelectorScoredFile {
                    metadata: file.clone(),
                    tokens,
                    score,
                    category,
                }
            })
            .collect();
        
        // Group by category for tiered selection
        let mut categorized: HashMap<FileCategory, Vec<SelectorScoredFile>> = HashMap::new();
        for scored_file in scored_files {
            categorized.entry(scored_file.category)
                .or_insert_with(Vec::new)
                .push(scored_file);
        }
        
        // Sort within each category by score
        for files in categorized.values_mut() {
            files.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        // V5 Tiered selection with intelligent allocation
        let mut selected = Vec::new();
        let mut remaining_budget = self.config.token_budget;
        
        // Tier 1: Critical entry points (highest priority)
        let tier1_order = [FileCategory::Entry, FileCategory::Config];
        for category in tier1_order.iter() {
            if let Some(files) = categorized.get(category) {
                let tier_budget = match category {
                    FileCategory::Entry => (self.config.token_budget as f64 * 0.35) as usize, // 35% for entry points
                    FileCategory::Config => (self.config.token_budget as f64 * 0.25) as usize, // 25% for config
                    _ => 0,
                };
                
                let mut used_budget = 0;
                for scored_file in files {
                    if used_budget + scored_file.tokens <= tier_budget && scored_file.tokens <= remaining_budget {
                        selected.push(scored_file.metadata.clone());
                        used_budget += scored_file.tokens;
                        remaining_budget = remaining_budget.saturating_sub(scored_file.tokens);
                    }
                }
            }
        }
        
        // Tier 2: General implementation files (fill remaining budget intelligently)
        if let Some(general_files) = categorized.get(&FileCategory::General) {
            for scored_file in general_files {
                if scored_file.tokens <= remaining_budget {
                    selected.push(scored_file.metadata.clone());
                    remaining_budget = remaining_budget.saturating_sub(scored_file.tokens);
                }
            }
        }
        
        // Tier 3: Examples (lowest priority, use remaining budget)
        if let Some(example_files) = categorized.get(&FileCategory::Examples) {
            for scored_file in example_files {
                if scored_file.tokens <= remaining_budget {
                    selected.push(scored_file.metadata.clone());
                    remaining_budget = remaining_budget.saturating_sub(scored_file.tokens);
                }
            }
        }
        
        Ok(selected)
    }
    
    /// Apply scaling optimizations to selected files
    async fn apply_scaling_optimizations(&self, selected_files: &[FileMetadata]) -> ScalingResult<ProcessingResult> {
        // Create a mock processing result optimized for selected subset
        let total_size: u64 = selected_files.iter().map(|f| f.size).sum();
        let processing_time = Duration::from_millis((selected_files.len() as u64 * 2).max(10)); // Fast for selected subset
        let memory_peak = (selected_files.len() * 1024).max(1024); // Minimal memory usage
        
        Ok(ProcessingResult {
            files: selected_files.to_vec(),
            total_files: selected_files.len(),
            processing_time,
            memory_peak,
            cache_hits: 0,
            cache_misses: selected_files.len() as u64,
            metrics: crate::metrics::ScalingMetrics {
                files_processed: selected_files.len() as u64,
                total_processing_time: processing_time,
                memory_peak,
                cache_hits: 0,
                cache_misses: selected_files.len() as u64,
                parallel_efficiency: 1.0,
                streaming_overhead: Duration::from_millis(0),
            },
        })
    }
    
    /// Calculate tokens used by selected files
    fn calculate_tokens_used(&self, selected_files: &[FileMetadata]) -> usize {
        selected_files.iter()
            .map(|file| self.estimate_tokens(file))
            .sum()
    }
    
    /// Estimate tokens for a file based on size and type (matching original scribe behavior)
    fn estimate_tokens(&self, file: &FileMetadata) -> usize {
        // Use more realistic token estimation like original scribe
        // Original scribe uses ~3.5 chars per token on average
        let base_tokens = ((file.size as f64) / 3.5) as usize;
        
        // Add minimum token count for very small files to avoid underestimation
        // Make minimum higher for small budgets to be more selective
        let min_tokens = if self.config.token_budget < 5000 {
            100 // Higher minimum for small budgets
        } else {
            50  // Standard minimum
        };
        let base_tokens = base_tokens.max(min_tokens);
        
        // Adjust based on file type (more realistic multipliers)
        let multiplier = match file.file_type.as_str() {
            "Source" => 1.2,      // Source code has more complexity
            "Documentation" => 1.0, // Documentation is standard
            "Configuration" => 0.8,  // Config files are more compact
            _ => 1.1,             // Default higher to be conservative
        };
        
        // Apply language-specific adjustments
        let language_multiplier = match file.language.as_str() {
            "Rust" => 1.3,       // Rust is very verbose
            "JavaScript" | "TypeScript" => 1.2, // JS/TS moderately verbose
            "Python" => 1.1,      // Python is readable but efficient
            "C" | "Go" => 1.0,    // C/Go are concise
            "HTML" | "CSS" => 0.9, // Markup is less token-dense
            "JSON" | "YAML" | "TOML" => 0.7, // Data formats are compact
            _ => 1.0,             // Default
        };
        
        // Final calculation with realistic scaling
        let final_tokens = (base_tokens as f64 * multiplier * language_multiplier) as usize;
        
        // Cap extremely large files to avoid single file consuming entire budget
        final_tokens.min(self.config.token_budget / 4) // No single file > 25% of budget
    }
    
    /// Calculate file score for selection (aggressive prioritization like original scribe)
    fn calculate_file_score(&self, file: &FileMetadata) -> f64 {
        let mut score: f64 = 0.1; // Lower base score to be more selective
        
        let path_str = file.path.to_string_lossy().to_lowercase();
        
        // High-priority entry points (like original scribe)
        if path_str.contains("main") || path_str.contains("index") {
            score += 2.0; // Very high priority
        }
        if path_str.contains("lib.rs") || path_str.contains("mod.rs") {
            score += 1.5; // High priority for Rust entry points
        }
        if path_str.contains("__init__.py") {
            score += 1.3; // High priority for Python packages
        }
        
        // Root-level files get major boost (like README, setup files)
        let path_components = file.path.components().count();
        if path_components <= 2 { // Root or one level down
            score += 1.0;
            
            // Special boost for important root files
            if path_str.contains("readme") || path_str.contains("license") || 
               path_str.contains("cargo.toml") || path_str.contains("package.json") ||
               path_str.contains("pyproject.toml") || path_str.contains("setup.py") {
                score += 1.5;
            }
        }
        
        // Language importance (more aggressive)
        match file.language.as_str() {
            "Rust" | "Python" | "JavaScript" | "TypeScript" => score += 0.8,
            "C" | "C++" | "Go" | "Java" => score += 0.6,
            "Shell" | "Makefile" => score += 0.4, // Build scripts
            _ => {}
        }
        
        // File type importance
        match file.file_type.as_str() {
            "Source" => score += 0.6,
            "Configuration" => score += 0.5, // Config files are very important  
            "Documentation" => score += 0.3,
            _ => {}
        }
        
        // Penalize very large files more heavily to stay within budget
        if file.size > 50_000 {
            score -= 0.5;
        }
        if file.size > 100_000 {
            score -= 1.0;
        }
        
        // Boost for certain important patterns
        if path_str.contains("test") && !path_str.contains("tests/") {
            score += 0.2; // Important test files but not test directories
        }
        
        // Penalize deep nesting (prefer top-level files)
        if path_components > 4 {
            score -= 0.3 * (path_components - 4) as f64;
        }
        
        // Boost small, important files
        if file.size < 10_000 && (path_str.contains("config") || path_str.contains("env")) {
            score += 0.4;
        }
        
        score.clamp(0.0, 5.0) // Allow higher scores for very important files
    }
    
    /// Classify file into category
    fn classify_file(&self, file: &FileMetadata) -> FileCategory {
        let path_str = file.path.to_string_lossy().to_lowercase();
        let filename = file.path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        // Check for config files
        if matches!(file.file_type.as_str(), "Configuration") || 
           filename.contains("config") || filename.ends_with(".toml") || 
           filename.ends_with(".json") || filename.ends_with(".yaml") {
            return FileCategory::Config;
        }
        
        // Check for entry points
        if filename.contains("main") || filename.contains("index") || 
           filename == "lib.rs" || filename == "__init__.py" {
            return FileCategory::Entry;
        }
        
        // Check for examples/tests
        if path_str.contains("example") || path_str.contains("test") || 
           path_str.contains("demo") || path_str.contains("sample") {
            return FileCategory::Examples;
        }
        
        FileCategory::General
    }
    
    /// Static version of file scoring for use in streaming selector
    fn calculate_file_score_static(file: &FileMetadata, token_budget: usize) -> f64 {
        let mut score: f64 = 0.1; // Lower base score to be more selective
        
        let path_str = file.path.to_string_lossy().to_lowercase();
        
        // High-priority entry points (like original scribe)
        if path_str.contains("main") || path_str.contains("index") {
            score += 2.0; // Very high priority
        }
        if path_str.contains("lib.rs") || path_str.contains("mod.rs") {
            score += 1.5; // High priority for Rust entry points
        }
        if path_str.contains("__init__.py") {
            score += 1.3; // High priority for Python packages
        }
        
        // Root-level files get major boost (like README, setup files)
        let path_components = file.path.components().count();
        if path_components <= 2 { // Root or one level down
            score += 1.0;
            
            // Special boost for important root files
            if path_str.contains("readme") || path_str.contains("license") || 
               path_str.contains("cargo.toml") || path_str.contains("package.json") ||
               path_str.contains("pyproject.toml") || path_str.contains("setup.py") {
                score += 1.5;
            }
        }
        
        // Language importance (more aggressive)
        match file.language.as_str() {
            "Rust" | "Python" | "JavaScript" | "TypeScript" => score += 0.8,
            "C" | "C++" | "Go" | "Java" => score += 0.6,
            "Shell" => score += 0.4, // Build scripts
            _ => {}
        }
        
        // File type importance
        match file.file_type.as_str() {
            "Source" => score += 0.6,
            "Configuration" => score += 0.5, // Config files are very important  
            "Documentation" => score += 0.3,
            _ => {}
        }
        
        // Penalize very large files more heavily to stay within budget
        if file.size > 50_000 {
            score -= 0.5;
        }
        if file.size > 100_000 {
            score -= 1.0;
        }
        
        // Boost for certain important patterns
        if path_str.contains("test") && !path_str.contains("tests/") {
            score += 0.2; // Important test files but not test directories
        }
        
        // Penalize deep nesting (prefer top-level files)
        if path_components > 4 {
            score -= 0.3 * (path_components - 4) as f64;
        }
        
        // Boost small, important files
        if file.size < 10_000 && (path_str.contains("config") || path_str.contains("env")) {
            score += 0.4;
        }
        
        score.clamp(0.0, 5.0) // Allow higher scores for very important files
    }
    
    /// Static version of token estimation for use in streaming selector
    fn estimate_tokens_static(file: &FileMetadata, token_budget: usize) -> usize {
        // Use more realistic token estimation like original scribe
        // Original scribe uses ~3.5 chars per token on average
        let base_tokens = ((file.size as f64) / 3.5) as usize;
        
        // Add minimum token count for very small files to avoid underestimation
        // Make minimum higher for small budgets to be more selective
        let min_tokens = if token_budget < 5000 {
            100 // Higher minimum for small budgets
        } else {
            50  // Standard minimum
        };
        let base_tokens = base_tokens.max(min_tokens);
        
        // Adjust based on file type (more realistic multipliers)
        let multiplier = match file.file_type.as_str() {
            "Source" => 1.2,      // Source code has more complexity
            "Documentation" => 1.0, // Documentation is standard
            "Configuration" => 0.8,  // Config files are more compact
            _ => 1.1,             // Default higher to be conservative
        };
        
        // Apply language-specific adjustments
        let language_multiplier = match file.language.as_str() {
            "Rust" => 1.3,       // Rust is very verbose
            "JavaScript" | "TypeScript" => 1.2, // JS/TS moderately verbose
            "Python" => 1.1,      // Python is readable but efficient
            "C" | "Go" => 1.0,    // C/Go are concise
            "HTML" | "CSS" => 0.9, // Markup is less token-dense
            "JSON" | "YAML" | "TOML" => 0.7, // Data formats are compact
            _ => 1.0,             // Default
        };
        
        // Final calculation with realistic scaling
        let final_tokens = (base_tokens as f64 * multiplier * language_multiplier) as usize;
        
        // Cap extremely large files to avoid single file consuming entire budget
        final_tokens.min(token_budget / 4) // No single file > 25% of budget
    }
}

impl ScalingSelectionResult {
    /// Get all files in optimal order (positioned if available, otherwise selected)
    pub fn get_optimally_ordered_files(&self) -> Vec<&FileMetadata> {
        if let Some(positioned) = &self.positioned_selection {
            let mut files = Vec::new();
            
            // HEAD files first (query-relevant, high centrality)
            for file in &positioned.positioning.head_files {
                files.push(&file.metadata);
            }
            
            // MIDDLE files (supporting, low centrality)
            for file in &positioned.positioning.middle_files {
                files.push(&file.metadata);
            }
            
            // TAIL files last (core functionality, high centrality)
            for file in &positioned.positioning.tail_files {
                files.push(&file.metadata);
            }
            
            files
        } else {
            self.selected_files.iter().collect()
        }
    }
    
    /// Get positioning statistics if available
    pub fn get_positioning_stats(&self) -> Option<(usize, usize, usize)> {
        self.positioned_selection.as_ref().map(|p| (
            p.positioning.head_files.len(),
            p.positioning.middle_files.len(),
            p.positioning.tail_files.len(),
        ))
    }
    
    /// Get positioning reasoning if available
    pub fn get_positioning_reasoning(&self) -> Option<&str> {
        self.positioned_selection.as_ref().map(|p| p.positioning_reasoning.as_str())
    }
    
    /// Check if context positioning was applied
    pub fn has_context_positioning(&self) -> bool {
        self.positioned_selection.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[tokio::test]
    async fn test_scaling_selector_creation() {
        let selector = ScalingSelector::with_defaults();
        assert_eq!(selector.config.token_budget, 8000);
    }

    #[tokio::test]
    async fn test_small_budget_selection() {
        let selector = ScalingSelector::with_token_budget(1000);
        assert_eq!(selector.config.token_budget, 1000);
        assert!(matches!(selector.config.selection_algorithm, SelectionAlgorithm::V5Integrated));
    }

    #[tokio::test]
    async fn test_medium_budget_selection() {
        let selector = ScalingSelector::with_token_budget(10000);
        assert_eq!(selector.config.token_budget, 10000);
        assert!(matches!(selector.config.selection_algorithm, SelectionAlgorithm::V5Integrated));
    }

    #[tokio::test]
    async fn test_file_selection_process() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();
        
        // Create test files
        fs::create_dir_all(repo_path.join("src")).unwrap();
        fs::write(repo_path.join("src/main.rs"), "fn main() { println!(\"Hello, world!\"); }").unwrap();
        fs::write(repo_path.join("src/lib.rs"), "pub fn hello() -> String { \"Hello\".to_string() }").unwrap();
        fs::write(repo_path.join("Cargo.toml"), "[package]\nname = \"test\"\nversion = \"0.1.0\"").unwrap();
        fs::write(repo_path.join("README.md"), "# Test Project\n\nThis is a test project.").unwrap();
        
        let mut selector = ScalingSelector::with_token_budget(5000);
        let result = selector.select_and_process(repo_path).await.unwrap();
        
        // Should select some files but not all
        assert!(result.selected_files.len() > 0);
        assert!(result.selected_files.len() <= 4); // Don't select everything
        assert!(result.tokens_used <= 5000); // Stay within budget
        assert!(result.token_utilization <= 1.0); // Don't exceed budget
    }

    #[test]
    fn test_token_estimation() {
        let selector = ScalingSelector::with_defaults();
        
        let rust_file = FileMetadata {
            path: std::path::PathBuf::from("src/main.rs"),
            size: 1000,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        
        let tokens = selector.estimate_tokens(&rust_file);
        assert!(tokens > 200); // Should estimate reasonable number of tokens
        
        let config_file = FileMetadata {
            path: std::path::PathBuf::from("Cargo.toml"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "TOML".to_string(),
            file_type: "Configuration".to_string(),
        };
        
        let config_tokens = selector.estimate_tokens(&config_file);
        assert!(config_tokens < tokens); // Config should estimate fewer tokens
    }

    #[test]
    fn test_file_scoring() {
        let selector = ScalingSelector::with_defaults();
        
        let main_file = FileMetadata {
            path: std::path::PathBuf::from("src/main.rs"),
            size: 1000,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        
        let score = selector.calculate_file_score(&main_file);
        assert!(score > 0.7); // Main files should score high
        
        let readme = FileMetadata {
            path: std::path::PathBuf::from("README.md"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "Markdown".to_string(),
            file_type: "Documentation".to_string(),
        };
        
        let readme_score = selector.calculate_file_score(&readme);
        assert!(readme_score < score); // README should score lower than main.rs
    }

    #[tokio::test]
    async fn test_context_positioning_integration() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();
        
        // Create test files
        fs::create_dir_all(repo_path.join("src")).unwrap();
        fs::write(repo_path.join("src/main.rs"), "fn main() { println!(\"Hello, world!\"); }").unwrap();
        fs::write(repo_path.join("src/lib.rs"), "pub fn hello() -> String { \"Hello\".to_string() }").unwrap();
        fs::write(repo_path.join("src/utils.rs"), "pub fn utility() {}").unwrap();
        fs::write(repo_path.join("Cargo.toml"), "[package]\nname = \"test\"\nversion = \"0.1.0\"").unwrap();
        
        // Test with positioning enabled and query hint
        let mut config = ScalingSelectionConfig::medium_budget();
        config.positioning_config.enable_positioning = true;
        let mut selector = ScalingSelector::new(config);
        
        let result = selector.select_and_process_with_query(repo_path, Some("main")).await.unwrap();
        
        // Should have positioning applied
        assert!(result.has_context_positioning());
        
        // Should have files distributed across tiers
        let (head, middle, tail) = result.get_positioning_stats().unwrap();
        assert!(head > 0);
        assert!(head + middle + tail == result.selected_files.len());
        
        // Should have positioning reasoning
        assert!(result.get_positioning_reasoning().is_some());
        let reasoning = result.get_positioning_reasoning().unwrap();
        assert!(reasoning.contains("HEAD"));
        assert!(reasoning.contains("TAIL"));
        
        // Test optimal ordering
        let ordered_files = result.get_optimally_ordered_files();
        assert_eq!(ordered_files.len(), result.selected_files.len());
    }

    #[tokio::test]
    async fn test_positioning_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();
        
        // Create test files
        fs::create_dir_all(repo_path.join("src")).unwrap();
        fs::write(repo_path.join("src/main.rs"), "fn main() {}").unwrap();
        
        // Test with positioning disabled
        let mut config = ScalingSelectionConfig::small_budget();
        config.positioning_config.enable_positioning = false;
        let mut selector = ScalingSelector::new(config);
        
        let result = selector.select_and_process_with_query(repo_path, Some("main")).await.unwrap();
        
        // Should not have positioning applied
        assert!(!result.has_context_positioning());
        assert!(result.positioned_selection.is_none());
        
        // Optimal ordering should just return selected files
        let ordered_files = result.get_optimally_ordered_files();
        assert_eq!(ordered_files.len(), result.selected_files.len());
    }

    #[test]
    fn test_configuration_builder_positioning() {
        let config = ScalingSelectionConfig::default();
        assert!(config.positioning_config.enable_positioning);
        assert_eq!(config.positioning_config.head_percentage, 0.20);
        assert_eq!(config.positioning_config.tail_percentage, 0.20);
        
        let small_config = ScalingSelectionConfig::small_budget();
        assert!(small_config.positioning_config.enable_positioning);
        
        let large_config = ScalingSelectionConfig::large_budget();
        assert!(large_config.positioning_config.enable_positioning);
    }

    #[test]
    fn test_with_test_exclusion_convenience_method() {
        let config = ScalingSelectionConfig::default()
            .with_test_exclusion();
        
        // Verify the convenience method enabled test exclusion
        assert!(config.positioning_config.auto_exclude_tests);
        
        // Test that it can be chained with other configurations
        let config_chained = ScalingSelectionConfig::medium_budget()
            .with_test_exclusion();
        
        assert!(config_chained.positioning_config.auto_exclude_tests);
        assert_eq!(config_chained.token_budget, 10000); // Should preserve medium budget setting
    }
}
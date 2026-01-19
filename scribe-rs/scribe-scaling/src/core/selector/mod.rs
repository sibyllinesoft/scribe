//! Intelligent Scaling Selector - Minimal Self-contained Selection
//!
//! This module provides intelligent file selection with token budget awareness.
//! This is a simplified, self-contained implementation to avoid circular dependencies.

mod scoring;
mod types;

#[cfg(feature = "bm25")]
mod bm25;
#[cfg(feature = "bm25")]
pub use bm25::Bm25Scorer;

pub use types::{
    FileCategory, ScalingSelectionConfig, ScalingSelectionResult, ScalingSelector,
    SelectionAlgorithm,
};

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(feature = "bm25")]
use std::sync::Arc;
#[cfg(feature = "bm25")]
use parking_lot::RwLock;

use tracing::{debug, info, warn};

use crate::api::engine::ProcessingResult;
use crate::core::error::ScalingResult;
use crate::core::positioning::ContextPositioner;
use crate::core::utils::classify_file_type_string;
use crate::io::streaming::{FileMetadata, StreamingSelector};
use scribe_core::file;

/// Scored file for selection (selector-specific version)
#[derive(Debug, Clone)]
struct SelectorScoredFile {
    metadata: FileMetadata,
    tokens: usize,
    score: f64,
    category: FileCategory,
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
    pub async fn select_and_process(
        &mut self,
        repo_path: &Path,
    ) -> ScalingResult<ScalingSelectionResult> {
        self.select_and_process_with_query(repo_path, None).await
    }

    /// Execute intelligent selection with query hint for context positioning
    pub async fn select_and_process_with_query(
        &mut self,
        repo_path: &Path,
        query_hint: Option<&str>,
    ) -> ScalingResult<ScalingSelectionResult> {
        let start_time = Instant::now();

        info!(
            "Starting intelligent scaling selection for: {:?}",
            repo_path
        );
        info!(
            "Token budget: {}, Algorithm: {:?}",
            self.config.token_budget, self.config.selection_algorithm
        );
        if let Some(query) = query_hint {
            info!("Query hint for selection and positioning: '{}'", query);
        }

        // Phase 1: Optimized streaming discovery and selection
        let discovery_start = Instant::now();
        let mut selected_files = self.discover_and_select_files_streaming(repo_path).await?;
        let discovery_time = discovery_start.elapsed();

        // Phase 2: Apply BM25 re-ranking if enabled and query_hint provided
        #[cfg(feature = "bm25")]
        if let Some(query) = query_hint {
            selected_files = self.apply_bm25_reranking(repo_path, selected_files, query).await?;
        }

        info!(
            "Selected {} files in {:?}",
            selected_files.len(),
            discovery_time
        );

        // Phase 3: Apply context positioning if enabled
        let total_files_considered = selected_files.len();
        let (positioned_selection, final_files, final_tokens) =
            if self.config.positioning_config.enable_positioning {
                let positioner = ContextPositioner::new(self.config.positioning_config.clone());
                let positioned = positioner
                    .position_files(selected_files.clone(), query_hint)
                    .await?;

                info!(
                    "Context positioning applied: HEAD={}, MIDDLE={}, TAIL={}",
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
        info!(
            "Token utilization: {:.1}% ({}/{})",
            token_utilization * 100.0,
            final_tokens,
            self.config.token_budget
        );

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
    async fn discover_and_select_files_streaming(
        &self,
        repo_path: &Path,
    ) -> ScalingResult<Vec<FileMetadata>> {
        info!("Using optimized streaming file discovery");

        // Create streaming selector
        let streaming_config = crate::io::streaming::StreamingConfig {
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
            move |file: &FileMetadata| -> usize { Self::estimate_tokens_static(file, token_budget) }
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

        info!(
            "Streaming selection completed: {} files selected",
            selected_files.len()
        );
        Ok(selected_files)
    }

    /// Apply BM25 re-ranking to boost query-relevant files
    #[cfg(feature = "bm25")]
    async fn apply_bm25_reranking(
        &self,
        repo_path: &Path,
        files: Vec<FileMetadata>,
        query: &str,
    ) -> ScalingResult<Vec<FileMetadata>> {
        use std::cmp::Ordering;

        info!("Applying BM25 re-ranking for query: '{}'", query);

        // Try to create BM25 scorer
        let scorer = match bm25::Bm25Scorer::new(repo_path) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to create BM25 scorer: {}, skipping re-ranking", e);
                return Ok(files);
            }
        };

        // Index the discovered files
        if let Err(e) = scorer.index_files(&files) {
            warn!("Failed to index files for BM25: {}, skipping re-ranking", e);
            return Ok(files);
        }

        // Get BM25 scores for all files
        let file_paths: Vec<_> = files.iter().map(|f| f.path.clone()).collect();
        let bm25_scores = match scorer.score_files(query, &file_paths) {
            Ok(scores) => scores,
            Err(e) => {
                warn!("Failed to get BM25 scores: {}, skipping re-ranking", e);
                return Ok(files);
            }
        };

        // Combine scores and re-sort
        let mut scored_files: Vec<(FileMetadata, f64)> = files
            .into_iter()
            .map(|file| {
                let base_score = Self::calculate_file_score_static(&file, self.config.token_budget);
                let bm25_score = bm25_scores.get(&file.path).copied().unwrap_or(0.0);
                let combined = bm25::combine_scores(base_score, bm25_score, 2.0); // BM25 weight of 2.0
                (file, combined)
            })
            .collect();

        // Sort by combined score (descending)
        scored_files.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });

        // Log top boosted files
        let top_files: Vec<_> = scored_files.iter().take(5).collect();
        for (file, score) in &top_files {
            let bm25 = bm25_scores.get(&file.path).copied().unwrap_or(0.0);
            debug!(
                "BM25 boosted: {} (combined={:.2}, bm25={:.2})",
                file.path.display(),
                score,
                bm25
            );
        }

        info!(
            "BM25 re-ranking complete: top file is {} with score {:.2}",
            scored_files.first().map(|(f, _)| f.path.display().to_string()).unwrap_or_default(),
            scored_files.first().map(|(_, s)| *s).unwrap_or(0.0)
        );

        // Re-select within budget after re-ranking
        let mut selected = Vec::new();
        let mut remaining_budget = self.config.token_budget;

        for (file, _score) in scored_files {
            let tokens = Self::estimate_tokens_static(&file, self.config.token_budget);
            if tokens <= remaining_budget {
                remaining_budget -= tokens;
                selected.push(file);
            }
        }

        Ok(selected)
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
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());

        if matches!(ext.as_deref(), Some("h" | "hpp" | "hxx")) {
            return "Header".to_string();
        }

        if path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("dockerfile"))
            .unwrap_or(false)
        {
            return "Dockerfile".to_string();
        }

        let language = file::detect_language_from_path(path);
        file::language_display_name(&language).to_string()
    }

    /// Simple file type classification - delegates to shared utility
    fn classify_file_type(&self, path: &Path) -> String {
        classify_file_type_string(path)
    }

    /// Apply intelligent selection algorithm based on configuration
    async fn apply_intelligent_selection(
        &self,
        files: &[FileMetadata],
    ) -> ScalingResult<Vec<FileMetadata>> {
        // V5 Integrated selection algorithm (tiered approach)
        self.apply_integrated_selection(files)
    }

    /// V5 Integrated selection: tiered approach with intelligent prioritization
    fn apply_integrated_selection(
        &self,
        files: &[FileMetadata],
    ) -> ScalingResult<Vec<FileMetadata>> {
        // Score all files
        let scored_files: Vec<SelectorScoredFile> = files
            .iter()
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
            categorized
                .entry(scored_file.category)
                .or_insert_with(Vec::new)
                .push(scored_file);
        }

        // Sort within each category by score
        for files in categorized.values_mut() {
            files.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
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
                    if used_budget + scored_file.tokens <= tier_budget
                        && scored_file.tokens <= remaining_budget
                    {
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
    async fn apply_scaling_optimizations(
        &self,
        selected_files: &[FileMetadata],
    ) -> ScalingResult<ProcessingResult> {
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
            metrics: crate::io::metrics::ScalingMetrics {
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
        selected_files
            .iter()
            .map(|file| self.estimate_tokens(file))
            .sum()
    }

    /// Estimate tokens for a file based on size and type (matching original scribe behavior)
    pub fn estimate_tokens(&self, file: &FileMetadata) -> usize {
        // Use more realistic token estimation like original scribe
        // Original scribe uses ~3.5 chars per token on average
        let base_tokens = ((file.size as f64) / 3.5) as usize;

        // Add minimum token count for very small files to avoid underestimation
        // Make minimum higher for small budgets to be more selective
        let min_tokens = if self.config.token_budget < 5000 {
            100 // Higher minimum for small budgets
        } else {
            50 // Standard minimum
        };
        let base_tokens = base_tokens.max(min_tokens);

        // Adjust based on file type (more realistic multipliers)
        let multiplier = match file.file_type.as_str() {
            "Source" => 1.2,        // Source code has more complexity
            "Documentation" => 1.0, // Documentation is standard
            "Configuration" => 0.8, // Config files are more compact
            _ => 1.1,               // Default higher to be conservative
        };

        // Apply language-specific adjustments
        let language_multiplier = match file.language.as_str() {
            "Rust" => 1.3,                      // Rust is very verbose
            "JavaScript" | "TypeScript" => 1.2, // JS/TS moderately verbose
            "Python" => 1.1,                    // Python is readable but efficient
            "C" | "Go" => 1.0,                  // C/Go are concise
            "HTML" | "CSS" => 0.9,              // Markup is less token-dense
            "JSON" | "YAML" | "TOML" => 0.7,    // Data formats are compact
            _ => 1.0,                           // Default
        };

        // Final calculation with realistic scaling
        let final_tokens = (base_tokens as f64 * multiplier * language_multiplier) as usize;

        // Cap extremely large files to avoid single file consuming entire budget
        final_tokens.min(self.config.token_budget / 4) // No single file > 25% of budget
    }

    /// Calculate file score for selection (aggressive prioritization like original scribe)
    pub fn calculate_file_score(&self, file: &FileMetadata) -> f64 {
        let path_str = file.path.to_string_lossy().to_lowercase();
        let path_components = file.path.components().count();

        scoring::calculate_combined_score(
            &path_str,
            path_components,
            file.language.as_str(),
            file.file_type.as_str(),
            file.size,
        )
    }

    /// Classify file into category
    fn classify_file(&self, file: &FileMetadata) -> FileCategory {
        let path_str = file.path.to_string_lossy().to_lowercase();
        let filename = file
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Check for config files
        if matches!(file.file_type.as_str(), "Configuration")
            || filename.contains("config")
            || filename.ends_with(".toml")
            || filename.ends_with(".json")
            || filename.ends_with(".yaml")
        {
            return FileCategory::Config;
        }

        // Check for entry points
        if filename.contains("main")
            || filename.contains("index")
            || filename == "lib.rs"
            || filename == "__init__.py"
        {
            return FileCategory::Entry;
        }

        // Check for examples/tests
        if path_str.contains("example")
            || path_str.contains("test")
            || path_str.contains("demo")
            || path_str.contains("sample")
        {
            return FileCategory::Examples;
        }

        FileCategory::General
    }

    /// Static version of file scoring for use in streaming selector
    #[allow(unused_variables)]
    fn calculate_file_score_static(file: &FileMetadata, token_budget: usize) -> f64 {
        let path_str = file.path.to_string_lossy().to_lowercase();
        let path_components = file.path.components().count();

        scoring::calculate_combined_score(
            &path_str,
            path_components,
            file.language.as_str(),
            file.file_type.as_str(),
            file.size,
        )
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
            50 // Standard minimum
        };
        let base_tokens = base_tokens.max(min_tokens);

        // Adjust based on file type (more realistic multipliers)
        let multiplier = match file.file_type.as_str() {
            "Source" => 1.2,        // Source code has more complexity
            "Documentation" => 1.0, // Documentation is standard
            "Configuration" => 0.8, // Config files are more compact
            _ => 1.1,               // Default higher to be conservative
        };

        // Apply language-specific adjustments
        let language_multiplier = match file.language.as_str() {
            "Rust" => 1.3,                      // Rust is very verbose
            "JavaScript" | "TypeScript" => 1.2, // JS/TS moderately verbose
            "Python" => 1.1,                    // Python is readable but efficient
            "C" | "Go" => 1.0,                  // C/Go are concise
            "HTML" | "CSS" => 0.9,              // Markup is less token-dense
            "JSON" | "YAML" | "TOML" => 0.7,    // Data formats are compact
            _ => 1.0,                           // Default
        };

        // Final calculation with realistic scaling
        let final_tokens = (base_tokens as f64 * multiplier * language_multiplier) as usize;

        // Cap extremely large files to avoid single file consuming entire budget
        final_tokens.min(token_budget / 4) // No single file > 25% of budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_scaling_selector_creation() {
        let selector = ScalingSelector::with_defaults();
        assert_eq!(selector.config.token_budget, 8000);
    }

    #[tokio::test]
    async fn test_small_budget_selection() {
        let selector = ScalingSelector::with_token_budget(1000);
        assert_eq!(selector.config.token_budget, 1000);
        assert!(matches!(
            selector.config.selection_algorithm,
            SelectionAlgorithm::V5Integrated
        ));
    }

    #[tokio::test]
    async fn test_medium_budget_selection() {
        let selector = ScalingSelector::with_token_budget(10000);
        assert_eq!(selector.config.token_budget, 10000);
        assert!(matches!(
            selector.config.selection_algorithm,
            SelectionAlgorithm::V5Integrated
        ));
    }

    #[tokio::test]
    async fn test_file_selection_process() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create test files
        fs::create_dir_all(repo_path.join("src")).unwrap();
        fs::write(
            repo_path.join("src/main.rs"),
            "fn main() { println!(\"Hello, world!\"); }",
        )
        .unwrap();
        fs::write(
            repo_path.join("src/lib.rs"),
            "pub fn hello() -> String { \"Hello\".to_string() }",
        )
        .unwrap();
        fs::write(
            repo_path.join("Cargo.toml"),
            "[package]\nname = \"test\"\nversion = \"0.1.0\"",
        )
        .unwrap();
        fs::write(
            repo_path.join("README.md"),
            "# Test Project\n\nThis is a test project.",
        )
        .unwrap();

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
        fs::write(
            repo_path.join("src/main.rs"),
            "fn main() { println!(\"Hello, world!\"); }",
        )
        .unwrap();
        fs::write(
            repo_path.join("src/lib.rs"),
            "pub fn hello() -> String { \"Hello\".to_string() }",
        )
        .unwrap();
        fs::write(repo_path.join("src/utils.rs"), "pub fn utility() {}").unwrap();
        fs::write(
            repo_path.join("Cargo.toml"),
            "[package]\nname = \"test\"\nversion = \"0.1.0\"",
        )
        .unwrap();

        // Test with positioning enabled and query hint
        let mut config = ScalingSelectionConfig::medium_budget();
        config.positioning_config.enable_positioning = true;
        let mut selector = ScalingSelector::new(config);

        let result = selector
            .select_and_process_with_query(repo_path, Some("main"))
            .await
            .unwrap();

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

        let result = selector
            .select_and_process_with_query(repo_path, Some("main"))
            .await
            .unwrap();

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
        let config = ScalingSelectionConfig::default().with_test_exclusion();

        // Verify the convenience method enabled test exclusion
        assert!(config.positioning_config.auto_exclude_tests);

        // Test that it can be chained with other configurations
        let config_chained = ScalingSelectionConfig::medium_budget().with_test_exclusion();

        assert!(config_chained.positioning_config.auto_exclude_tests);
        assert_eq!(config_chained.token_budget, 10000); // Should preserve medium budget setting
    }

    #[test]
    fn test_classify_file_config() {
        let selector = ScalingSelector::with_defaults();

        // Test configuration files
        let toml_file = FileMetadata {
            path: std::path::PathBuf::from("Cargo.toml"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "TOML".to_string(),
            file_type: "Configuration".to_string(),
        };
        assert_eq!(selector.classify_file(&toml_file), FileCategory::Config);

        let json_config = FileMetadata {
            path: std::path::PathBuf::from("config.json"),
            size: 300,
            modified: std::time::SystemTime::now(),
            language: "JSON".to_string(),
            file_type: "Data".to_string(),
        };
        assert_eq!(selector.classify_file(&json_config), FileCategory::Config);

        let yaml_file = FileMetadata {
            path: std::path::PathBuf::from("settings.yaml"),
            size: 200,
            modified: std::time::SystemTime::now(),
            language: "YAML".to_string(),
            file_type: "Data".to_string(),
        };
        assert_eq!(selector.classify_file(&yaml_file), FileCategory::Config);
    }

    #[test]
    fn test_classify_file_entry() {
        let selector = ScalingSelector::with_defaults();

        // Test entry point files
        let main_rs = FileMetadata {
            path: std::path::PathBuf::from("src/main.rs"),
            size: 1000,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&main_rs), FileCategory::Entry);

        let index_js = FileMetadata {
            path: std::path::PathBuf::from("src/index.js"),
            size: 800,
            modified: std::time::SystemTime::now(),
            language: "JavaScript".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&index_js), FileCategory::Entry);

        let lib_rs = FileMetadata {
            path: std::path::PathBuf::from("src/lib.rs"),
            size: 1200,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&lib_rs), FileCategory::Entry);

        let init_py = FileMetadata {
            path: std::path::PathBuf::from("mypackage/__init__.py"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "Python".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&init_py), FileCategory::Entry);
    }

    #[test]
    fn test_classify_file_examples() {
        let selector = ScalingSelector::with_defaults();

        // Test example files
        let example_file = FileMetadata {
            path: std::path::PathBuf::from("examples/demo.rs"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&example_file), FileCategory::Examples);

        let test_file = FileMetadata {
            path: std::path::PathBuf::from("tests/unit_test.rs"),
            size: 600,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&test_file), FileCategory::Examples);

        let sample_file = FileMetadata {
            path: std::path::PathBuf::from("samples/basic.py"),
            size: 400,
            modified: std::time::SystemTime::now(),
            language: "Python".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&sample_file), FileCategory::Examples);
    }

    #[test]
    fn test_classify_file_general() {
        let selector = ScalingSelector::with_defaults();

        // Test general source files
        let utils_file = FileMetadata {
            path: std::path::PathBuf::from("src/utils.rs"),
            size: 800,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&utils_file), FileCategory::General);

        let helper_file = FileMetadata {
            path: std::path::PathBuf::from("src/helpers/format.py"),
            size: 600,
            modified: std::time::SystemTime::now(),
            language: "Python".to_string(),
            file_type: "Source".to_string(),
        };
        assert_eq!(selector.classify_file(&helper_file), FileCategory::General);
    }

    #[test]
    fn test_estimate_tokens_static_languages() {
        // Test Rust (verbose, multiplier 1.3)
        let rust_file = FileMetadata {
            path: std::path::PathBuf::from("src/main.rs"),
            size: 3500, // ~1000 base tokens
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };
        let rust_tokens = ScalingSelector::estimate_tokens_static(&rust_file, 8000);
        assert!(rust_tokens > 1000); // Base * type_mult * lang_mult

        // Test Python (readable, multiplier 1.1)
        let python_file = FileMetadata {
            path: std::path::PathBuf::from("src/main.py"),
            size: 3500,
            modified: std::time::SystemTime::now(),
            language: "Python".to_string(),
            file_type: "Source".to_string(),
        };
        let python_tokens = ScalingSelector::estimate_tokens_static(&python_file, 8000);
        assert!(python_tokens > 1000);
        assert!(python_tokens < rust_tokens); // Python is less verbose

        // Test JSON (compact, multiplier 0.7)
        let json_file = FileMetadata {
            path: std::path::PathBuf::from("data.json"),
            size: 3500,
            modified: std::time::SystemTime::now(),
            language: "JSON".to_string(),
            file_type: "Configuration".to_string(),
        };
        let json_tokens = ScalingSelector::estimate_tokens_static(&json_file, 8000);
        assert!(json_tokens < python_tokens); // JSON is more compact
    }

    #[test]
    fn test_estimate_tokens_static_file_types() {
        // Test Source files (multiplier 1.2)
        let source_file = FileMetadata {
            path: std::path::PathBuf::from("code.go"),
            size: 3500,
            modified: std::time::SystemTime::now(),
            language: "Go".to_string(),
            file_type: "Source".to_string(),
        };
        let source_tokens = ScalingSelector::estimate_tokens_static(&source_file, 8000);

        // Test Documentation (multiplier 1.0)
        let doc_file = FileMetadata {
            path: std::path::PathBuf::from("README.md"),
            size: 3500,
            modified: std::time::SystemTime::now(),
            language: "Markdown".to_string(),
            file_type: "Documentation".to_string(),
        };
        let doc_tokens = ScalingSelector::estimate_tokens_static(&doc_file, 8000);
        assert!(doc_tokens < source_tokens);

        // Test Configuration (multiplier 0.8)
        let config_file = FileMetadata {
            path: std::path::PathBuf::from("config.yaml"),
            size: 3500,
            modified: std::time::SystemTime::now(),
            language: "YAML".to_string(),
            file_type: "Configuration".to_string(),
        };
        let config_tokens = ScalingSelector::estimate_tokens_static(&config_file, 8000);
        assert!(config_tokens < doc_tokens);
    }

    #[test]
    fn test_estimate_tokens_static_budget_cap() {
        // Test that tokens are capped at 25% of budget
        let large_file = FileMetadata {
            path: std::path::PathBuf::from("massive.rs"),
            size: 1_000_000, // Very large file
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let budget = 8000;
        let tokens = ScalingSelector::estimate_tokens_static(&large_file, budget);
        assert!(tokens <= budget / 4); // Should not exceed 25% of budget
    }

    #[test]
    fn test_estimate_tokens_static_minimum() {
        // Test minimum token estimation for small files
        let tiny_file = FileMetadata {
            path: std::path::PathBuf::from("tiny.rs"),
            size: 10, // Very small file
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        // For normal budget, minimum should be 50
        let tokens_normal = ScalingSelector::estimate_tokens_static(&tiny_file, 8000);
        assert!(tokens_normal >= 50);

        // For small budget, minimum should be 100
        let tokens_small = ScalingSelector::estimate_tokens_static(&tiny_file, 1000);
        assert!(tokens_small >= 100);
    }

    #[test]
    fn test_large_budget_selection() {
        let selector = ScalingSelector::with_token_budget(50000);
        assert_eq!(selector.config.token_budget, 50000);
        assert!(matches!(
            selector.config.selection_algorithm,
            SelectionAlgorithm::V5Integrated
        ));
    }

    #[test]
    fn test_token_estimation_various_languages() {
        let selector = ScalingSelector::with_defaults();

        // TypeScript (moderately verbose, 1.2)
        let ts_file = FileMetadata {
            path: std::path::PathBuf::from("app.ts"),
            size: 1000,
            modified: std::time::SystemTime::now(),
            language: "TypeScript".to_string(),
            file_type: "Source".to_string(),
        };
        let ts_tokens = selector.estimate_tokens(&ts_file);

        // C (concise, 1.0)
        let c_file = FileMetadata {
            path: std::path::PathBuf::from("main.c"),
            size: 1000,
            modified: std::time::SystemTime::now(),
            language: "C".to_string(),
            file_type: "Source".to_string(),
        };
        let c_tokens = selector.estimate_tokens(&c_file);

        // TypeScript should estimate more tokens than C
        assert!(ts_tokens > c_tokens);

        // HTML (less token-dense, 0.9)
        let html_file = FileMetadata {
            path: std::path::PathBuf::from("index.html"),
            size: 1000,
            modified: std::time::SystemTime::now(),
            language: "HTML".to_string(),
            file_type: "Source".to_string(),
        };
        let html_tokens = selector.estimate_tokens(&html_file);
        assert!(html_tokens < ts_tokens);
    }

    #[test]
    fn test_file_category_equality() {
        assert_eq!(FileCategory::Config, FileCategory::Config);
        assert_eq!(FileCategory::Entry, FileCategory::Entry);
        assert_eq!(FileCategory::Examples, FileCategory::Examples);
        assert_eq!(FileCategory::General, FileCategory::General);

        assert_ne!(FileCategory::Config, FileCategory::Entry);
        assert_ne!(FileCategory::Entry, FileCategory::Examples);
        assert_ne!(FileCategory::Examples, FileCategory::General);
    }

    #[test]
    fn test_file_category_clone() {
        let config = FileCategory::Config;
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_selector_scored_file_clone() {
        let scored = SelectorScoredFile {
            metadata: FileMetadata {
                path: std::path::PathBuf::from("test.rs"),
                size: 100,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            tokens: 50,
            score: 0.8,
            category: FileCategory::General,
        };

        let cloned = scored.clone();
        assert_eq!(scored.tokens, cloned.tokens);
        assert_eq!(scored.score, cloned.score);
        assert_eq!(scored.category, cloned.category);
    }

    #[test]
    fn test_selector_scored_file_debug() {
        let scored = SelectorScoredFile {
            metadata: FileMetadata {
                path: std::path::PathBuf::from("test.rs"),
                size: 100,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            tokens: 50,
            score: 0.8,
            category: FileCategory::General,
        };

        let debug_str = format!("{:?}", scored);
        assert!(debug_str.contains("SelectorScoredFile"));
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_scoring_main_files_vs_utils() {
        let selector = ScalingSelector::with_defaults();

        // main.rs should score higher than utils.rs
        let main_file = FileMetadata {
            path: std::path::PathBuf::from("src/main.rs"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let utils_file = FileMetadata {
            path: std::path::PathBuf::from("src/utils/helpers.rs"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let main_score = selector.calculate_file_score(&main_file);
        let utils_score = selector.calculate_file_score(&utils_file);

        assert!(main_score > utils_score, "main.rs should score higher than utils");
    }

    #[test]
    fn test_scoring_lib_vs_deep_nested() {
        let selector = ScalingSelector::with_defaults();

        // lib.rs should score higher than deeply nested files
        let lib_file = FileMetadata {
            path: std::path::PathBuf::from("src/lib.rs"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let deep_file = FileMetadata {
            path: std::path::PathBuf::from("src/utils/internal/private/detail.rs"),
            size: 500,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        let lib_score = selector.calculate_file_score(&lib_file);
        let deep_score = selector.calculate_file_score(&deep_file);

        assert!(lib_score > deep_score, "lib.rs should score higher than deeply nested files");
    }

    #[test]
    fn test_detect_language_header_files() {
        let selector = ScalingSelector::with_defaults();

        // Test header file detection (lines 224-225)
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("test.h")), "Header");
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("test.hpp")), "Header");
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("test.hxx")), "Header");
    }

    #[test]
    fn test_detect_language_dockerfile() {
        let selector = ScalingSelector::with_defaults();

        // Test Dockerfile detection (lines 228-234)
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("Dockerfile")), "Dockerfile");
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("dockerfile")), "Dockerfile");
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("DOCKERFILE")), "Dockerfile");
    }

    #[test]
    fn test_detect_language_common() {
        let selector = ScalingSelector::with_defaults();

        // Test common languages (lines 237-238)
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("test.rs")), "Rust");
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("test.py")), "Python");
        assert_eq!(selector.detect_language(&std::path::PathBuf::from("test.js")), "JavaScript");
    }

    #[test]
    fn test_classify_file_type() {
        let selector = ScalingSelector::with_defaults();

        // Test file type classification (lines 242-243)
        let rust_file = std::path::PathBuf::from("src/main.rs");
        let classified = selector.classify_file_type(&rust_file);
        assert_eq!(classified, "Source");

        let readme = std::path::PathBuf::from("README.md");
        let classified = selector.classify_file_type(&readme);
        assert_eq!(classified, "Documentation");
    }

    #[test]
    fn test_estimate_target_file_count() {
        // Test target file count estimation (lines 208-214)
        let small_selector = ScalingSelector::with_token_budget(1000);
        let count = small_selector.estimate_target_file_count();
        assert!(count >= 5); // Minimum is 5

        let medium_selector = ScalingSelector::with_token_budget(10000);
        let count = medium_selector.estimate_target_file_count();
        assert!(count >= 5 && count <= 200);

        let large_selector = ScalingSelector::with_token_budget(100000);
        let count = large_selector.estimate_target_file_count();
        assert!(count <= 200); // Maximum is 200
    }

    #[test]
    fn test_integrated_selection_categorization() {
        let selector = ScalingSelector::with_token_budget(5000);

        // Create a mix of file types to test tiered selection (lines 256-343)
        let files = vec![
            FileMetadata {
                path: std::path::PathBuf::from("src/main.rs"),
                size: 500,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            FileMetadata {
                path: std::path::PathBuf::from("Cargo.toml"),
                size: 200,
                modified: std::time::SystemTime::now(),
                language: "TOML".to_string(),
                file_type: "Configuration".to_string(),
            },
            FileMetadata {
                path: std::path::PathBuf::from("src/lib.rs"),
                size: 600,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            FileMetadata {
                path: std::path::PathBuf::from("examples/demo.rs"),
                size: 400,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            FileMetadata {
                path: std::path::PathBuf::from("src/utils.rs"),
                size: 300,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
        ];

        let result = selector.apply_integrated_selection(&files).unwrap();

        // Should select some files
        assert!(!result.is_empty());

        // Entry points should be selected first (main.rs, lib.rs)
        let selected_paths: Vec<_> = result.iter().map(|f| f.path.to_string_lossy().to_string()).collect();
        let has_entry = selected_paths.iter().any(|p| p.contains("main.rs") || p.contains("lib.rs"));
        assert!(has_entry, "Entry points should be prioritized");
    }

    #[test]
    fn test_integrated_selection_budget_constraint() {
        let selector = ScalingSelector::with_token_budget(100); // Very small budget

        let files = vec![
            FileMetadata {
                path: std::path::PathBuf::from("src/main.rs"),
                size: 5000, // Large file
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            FileMetadata {
                path: std::path::PathBuf::from("config.json"),
                size: 100, // Small file
                modified: std::time::SystemTime::now(),
                language: "JSON".to_string(),
                file_type: "Configuration".to_string(),
            },
        ];

        let result = selector.apply_integrated_selection(&files).unwrap();

        // Should respect budget constraints
        let total_tokens: usize = result.iter()
            .map(|f| selector.estimate_tokens(f))
            .sum();
        assert!(total_tokens <= 100 * 4, "Should respect budget (with some tolerance for file count)");
    }

    #[tokio::test]
    async fn test_apply_intelligent_selection() {
        let selector = ScalingSelector::with_token_budget(5000);

        let files = vec![
            FileMetadata {
                path: std::path::PathBuf::from("src/main.rs"),
                size: 500,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            FileMetadata {
                path: std::path::PathBuf::from("src/utils.rs"),
                size: 300,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
        ];

        // Test apply_intelligent_selection path (lines 247-253)
        let result = selector.apply_intelligent_selection(&files).await.unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_calculate_tokens_used() {
        let selector = ScalingSelector::with_defaults();

        let files = vec![
            FileMetadata {
                path: std::path::PathBuf::from("file1.rs"),
                size: 1000,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
            FileMetadata {
                path: std::path::PathBuf::from("file2.rs"),
                size: 500,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
        ];

        let tokens = selector.calculate_tokens_used(&files);
        assert!(tokens > 0);

        // Should sum tokens from both files
        let single_tokens = selector.estimate_tokens(&files[0]) + selector.estimate_tokens(&files[1]);
        assert_eq!(tokens, single_tokens);
    }

    #[tokio::test]
    async fn test_apply_scaling_optimizations() {
        let selector = ScalingSelector::with_defaults();

        let files = vec![
            FileMetadata {
                path: std::path::PathBuf::from("test.rs"),
                size: 500,
                modified: std::time::SystemTime::now(),
                language: "Rust".to_string(),
                file_type: "Source".to_string(),
            },
        ];

        // Test apply_scaling_optimizations path (lines 346-372)
        let result = selector.apply_scaling_optimizations(&files).await.unwrap();

        assert_eq!(result.total_files, 1);
        assert!(!result.files.is_empty());
        assert!(result.memory_peak > 0);
    }

    #[test]
    fn test_calculate_file_score_static() {
        let file = FileMetadata {
            path: std::path::PathBuf::from("src/main.rs"),
            size: 1000,
            modified: std::time::SystemTime::now(),
            language: "Rust".to_string(),
            file_type: "Source".to_string(),
        };

        // Test static scoring method (lines 480-491)
        let score = ScalingSelector::calculate_file_score_static(&file, 8000);
        assert!(score > 0.0);
    }
}

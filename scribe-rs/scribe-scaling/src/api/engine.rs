//! Main scaling engine that integrates all optimization components.
//!
//! This module provides the primary interface for the scaling system,
//! coordinating streaming, caching, parallel processing, adaptive configuration,
//! and signature extraction for optimal performance at scale.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::api::adaptive::AdaptiveConfig;
use crate::api::caching::{compute_config_hash, compute_repository_hash, CacheConfig, ProcessingCache};
use crate::core::error::{ScalingError, ScalingResult};
use crate::core::signatures::SignatureConfig;
use crate::io::memory::MemoryConfig;
use crate::io::metrics::{BenchmarkResult, ScalingMetrics};
use crate::io::parallel::ParallelConfig;
use crate::io::streaming::{FileMetadata, StreamingConfig};
use crate::core::utils::classify_file_type_string;
use scribe_core::file;

/// Complete scaling configuration combining all subsystems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub streaming: StreamingConfig,
    pub caching: CacheConfig,
    pub parallel: ParallelConfig,
    pub adaptive: AdaptiveConfig,
    pub signatures: SignatureConfig,
    pub memory: MemoryConfig,

    /// Token budget for intelligent selection (0 = unlimited)
    pub token_budget: Option<usize>,

    /// Enable intelligent file selection before processing
    pub enable_intelligent_selection: bool,

    /// Selection algorithm to use when intelligent selection is enabled
    pub selection_algorithm: Option<String>,

    /// Enable context positioning optimization (HEAD/MIDDLE/TAIL)
    pub enable_context_positioning: bool,

    /// Query for context positioning (affects HEAD section)
    pub positioning_query: Option<String>,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            streaming: StreamingConfig::default(),
            caching: CacheConfig::default(),
            parallel: ParallelConfig::default(),
            adaptive: AdaptiveConfig::default(),
            signatures: SignatureConfig::default(),
            memory: MemoryConfig::default(),
            token_budget: None,                  // Unlimited by default
            enable_intelligent_selection: false, // Off by default for backward compatibility
            selection_algorithm: None,           // Will use V5Integrated when enabled
            enable_context_positioning: false,   // Off by default for backward compatibility
            positioning_query: None,             // No query by default
        }
    }
}

impl ScalingConfig {
    /// Create configuration optimized for small repositories
    pub fn small_repository() -> Self {
        Self {
            streaming: StreamingConfig {
                enable_streaming: false,
                concurrency_limit: 2,
                memory_limit: 50 * 1024 * 1024, // 50MB
                selection_heap_size: 1000,
            },
            parallel: ParallelConfig {
                max_concurrent_tasks: 2,
                async_worker_count: 1,
                cpu_worker_count: 1,
                task_timeout: Duration::from_secs(10),
                enable_work_stealing: false,
            },
            token_budget: Some(8000), // Reasonable default for small repos
            enable_intelligent_selection: true,
            selection_algorithm: Some("v2_quotas".to_string()),
            ..Default::default()
        }
    }

    /// Create configuration optimized for large repositories
    pub fn large_repository() -> Self {
        Self {
            streaming: StreamingConfig {
                enable_streaming: true,
                concurrency_limit: 8,
                memory_limit: 500 * 1024 * 1024, // 500MB
                selection_heap_size: 10000,
            },
            parallel: ParallelConfig {
                max_concurrent_tasks: 16,
                async_worker_count: 8,
                cpu_worker_count: 8,
                task_timeout: Duration::from_secs(60),
                enable_work_stealing: true,
            },
            token_budget: Some(15000), // Larger budget for complex repositories
            enable_intelligent_selection: true,
            selection_algorithm: Some("v5_integrated".to_string()),
            ..Default::default()
        }
    }

    /// Create configuration with specific token budget
    pub fn with_token_budget(token_budget: usize) -> Self {
        Self {
            token_budget: Some(token_budget),
            enable_intelligent_selection: true,
            selection_algorithm: Some(
                match token_budget {
                    0..=2000 => "v2_quotas",
                    2001..=15000 => "v5_integrated",
                    _ => "v5_integrated",
                }
                .to_string(),
            ),
            ..Self::default()
        }
    }
}

/// Repository processing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// List of processed files with metadata
    pub files: Vec<FileMetadata>,

    /// Total number of files processed
    pub total_files: usize,

    /// Processing time
    pub processing_time: Duration,

    /// Peak memory usage in bytes
    pub memory_peak: usize,

    /// Cache hit count
    pub cache_hits: u64,

    /// Cache miss count
    pub cache_misses: u64,

    /// Additional metrics
    pub metrics: ScalingMetrics,
}

/// Main scaling engine
pub struct ScalingEngine {
    config: ScalingConfig,
    started_at: Option<Instant>,
    cache: Option<ProcessingCache>,
}

impl ScalingEngine {
    /// Create a new scaling engine with the given configuration
    pub async fn new(config: ScalingConfig) -> ScalingResult<Self> {
        info!(
            "Initializing scaling engine with configuration: {:?}",
            config
        );

        let cache = if config.caching.memory_cache_size > 0 {
            Some(ProcessingCache::new(config.caching.clone()))
        } else {
            None
        };

        Ok(Self {
            config,
            started_at: None,
            cache,
        })
    }

    /// Create a scaling engine with default configuration
    pub async fn with_defaults() -> ScalingResult<Self> {
        Self::new(ScalingConfig::default()).await
    }

    /// Create a scaling engine with the given configuration
    pub fn with_config(config: ScalingConfig) -> Self {
        let cache = if config.caching.memory_cache_size > 0 {
            Some(ProcessingCache::new(config.caching.clone()))
        } else {
            None
        };

        Self {
            config,
            started_at: None,
            cache,
        }
    }

    /// Process a repository with scaling optimizations
    pub async fn process_repository(&mut self, path: &Path) -> ScalingResult<ProcessingResult> {
        let start_time = Instant::now();
        self.started_at = Some(start_time);

        info!("Processing repository: {:?}", path);

        if !path.exists() {
            return Err(ScalingError::path("Repository path does not exist", path));
        }

        if !path.is_dir() {
            return Err(ScalingError::path(
                "Repository path is not a directory",
                path,
            ));
        }

        let config_hash = compute_config_hash(&self.config);
        let mut repo_hash_for_cache = None;

        if let Some(cache) = self.cache.as_mut() {
            match compute_repository_hash(path) {
                Ok(repo_hash) => {
                    if let Some(mut cached) = cache.get(repo_hash, &config_hash) {
                        if std::env::var("SCRIBE_DEBUG").is_ok() {
                            eprintln!("✅ Using cached scaling result for {}", path.display());
                        }

                        cached.processing_time = Duration::from_millis(0);
                        cached.cache_hits = cached.total_files as u64;
                        cached.cache_misses = 0;
                        cached.metrics.cache_hits = cached.cache_hits;
                        cached.metrics.cache_misses = cached.cache_misses;

                        return Ok(cached);
                    }

                    repo_hash_for_cache = Some(repo_hash);
                }
                Err(err) => {
                    if std::env::var("SCRIBE_DEBUG").is_ok() {
                        eprintln!("⚠️  Failed to compute repository hash: {}", err);
                    }
                }
            }
        }

        // Check if intelligent selection is enabled
        if self.config.enable_intelligent_selection && self.config.token_budget.is_some() {
            let mut result = self.process_with_intelligent_selection(path).await?;

            result.cache_hits = 0;
            result.cache_misses = result.total_files as u64;
            result.metrics.cache_hits = result.cache_hits;
            result.metrics.cache_misses = result.cache_misses;

            if let (Some(repo_hash), Some(cache)) = (repo_hash_for_cache, self.cache.as_mut()) {
                cache.insert(repo_hash, &config_hash, result.clone());
                cache.flush();
            }

            return Ok(result);
        }

        // Use optimized streaming file discovery
        info!("Using optimized streaming file discovery for basic processing");

        // Create a streaming selector even for basic processing to avoid memory issues
        let streaming_config = crate::io::streaming::StreamingConfig {
            enable_streaming: true,
            concurrency_limit: self.config.parallel.max_concurrent_tasks,
            memory_limit: self.config.streaming.memory_limit,
            selection_heap_size: 50000, // Large heap for full discovery
        };

        let streaming_selector = crate::io::streaming::StreamingSelector::new(streaming_config);

        // For basic processing, we want all files (within reason), so use very high limits
        let target_count = 50000; // Reasonable limit to prevent memory explosion
        let token_budget = 1_000_000; // Very high budget to include most files

        // Simple scoring that doesn't filter much
        let score_fn = |_file: &FileMetadata| -> f64 { 1.0 };
        let token_fn = |file: &FileMetadata| -> usize { (file.size / 4) as usize };

        let scored_files = streaming_selector
            .select_files_streaming(path, target_count, token_budget, score_fn, token_fn)
            .await?;

        let files: Vec<FileMetadata> = scored_files
            .into_iter()
            .map(|scored| scored.metadata)
            .collect();

        let total_size: u64 = files.iter().map(|f| f.size).sum();

        let processing_time = start_time.elapsed();

        info!("Processed {} files in {:?}", files.len(), processing_time);

        let mut result = ProcessingResult {
            total_files: files.len(),
            processing_time,
            memory_peak: estimate_memory_usage(files.len()),
            cache_hits: 0,
            cache_misses: files.len() as u64,
            metrics: ScalingMetrics {
                files_processed: files.len() as u64,
                total_processing_time: processing_time,
                memory_peak: estimate_memory_usage(files.len()),
                cache_hits: 0,
                cache_misses: files.len() as u64,
                parallel_efficiency: 1.0,
                streaming_overhead: Duration::from_millis(0),
            },
            files,
        };

        if let (Some(repo_hash), Some(cache)) = (repo_hash_for_cache, self.cache.as_mut()) {
            cache.insert(repo_hash, &config_hash, result.clone());
            cache.flush();
        }

        Ok(result)
    }

    /// Run performance benchmarks
    pub async fn benchmark(
        &mut self,
        path: &Path,
        iterations: usize,
    ) -> ScalingResult<Vec<BenchmarkResult>> {
        let mut results = Vec::with_capacity(iterations);

        for i in 0..iterations {
            info!("Running benchmark iteration {}/{}", i + 1, iterations);

            let start = Instant::now();
            let result = self.process_repository(path).await?;
            let duration = start.elapsed();

            let benchmark_result = BenchmarkResult::new(
                format!("iteration_{}", i + 1),
                duration,
                result.memory_peak,
                result.total_files as f64 / duration.as_secs_f64(),
                1.0, // 100% success rate
            );

            results.push(benchmark_result);
        }

        Ok(results)
    }

    /// Get current configuration
    pub fn config(&self) -> &ScalingConfig {
        &self.config
    }

    /// Process repository with intelligent selection enabled
    async fn process_with_intelligent_selection(
        &self,
        path: &Path,
    ) -> ScalingResult<ProcessingResult> {
        info!("Processing repository with intelligent selection enabled");

        // Create ScalingSelector with the configured token budget
        let token_budget = self.config.token_budget.unwrap_or(8000);
        let mut selector = crate::core::selector::ScalingSelector::with_token_budget(token_budget);

        // Execute intelligent selection and processing
        let selection_result = selector.select_and_process(path).await?;

        info!(
            "Intelligent selection completed: {} files selected, {:.1}% token utilization",
            selection_result.selected_files.len(),
            selection_result.token_utilization * 100.0
        );

        // Return the processing result from the selector
        Ok(selection_result.processing_result)
    }

    /// Check if engine is ready for processing
    pub fn is_ready(&self) -> bool {
        true // Always ready in this simple implementation
    }
}

/// Simple language detection based on file extension
fn detect_language(path: &Path) -> String {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    if matches!(extension.as_deref(), Some("h" | "hpp" | "hxx")) {
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
fn classify_file_type(path: &Path) -> String {
    classify_file_type_string(path)
}

/// Estimate memory usage based on file count
fn estimate_memory_usage(file_count: usize) -> usize {
    // Rough estimate: ~1KB per file metadata
    file_count * 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_scaling_engine_creation() {
        let engine = ScalingEngine::with_defaults().await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_repository_processing() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create test files
        fs::write(repo_path.join("main.rs"), "fn main() {}").unwrap();
        fs::write(repo_path.join("lib.rs"), "pub fn test() {}").unwrap();

        let mut engine = ScalingEngine::with_defaults().await.unwrap();
        let result = engine.process_repository(repo_path).await.unwrap();

        assert!(result.total_files >= 2);
        assert!(result.processing_time.as_nanos() > 0);
        assert!(result.memory_peak > 0);
    }

    #[tokio::test]
    async fn test_configuration_presets() {
        let small_config = ScalingConfig::small_repository();
        assert!(!small_config.streaming.enable_streaming);
        assert_eq!(small_config.parallel.max_concurrent_tasks, 2);

        let large_config = ScalingConfig::large_repository();
        assert!(large_config.streaming.enable_streaming);
        assert!(large_config.parallel.max_concurrent_tasks >= 16);
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mut engine = ScalingEngine::with_defaults().await.unwrap();
        let non_existent_path = Path::new("/non/existent/path");

        let result = engine.process_repository(non_existent_path).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_benchmarking() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create test file
        fs::write(repo_path.join("test.rs"), "fn test() {}").unwrap();

        let mut engine = ScalingEngine::with_defaults().await.unwrap();
        let results = engine.benchmark(repo_path, 3).await.unwrap();

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.duration.as_nanos() > 0);
            assert!(result.throughput > 0.0);
            assert_eq!(result.success_rate, 1.0);
        }
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(detect_language(Path::new("test.rs")), "Rust");
        assert_eq!(detect_language(Path::new("test.py")), "Python");
        assert_eq!(detect_language(Path::new("test.unknown")), "Unknown");
    }

    #[test]
    fn test_file_type_classification() {
        assert_eq!(classify_file_type(Path::new("main.rs")), "Source");
        assert_eq!(classify_file_type(Path::new("README.md")), "Documentation");
        assert_eq!(
            classify_file_type(Path::new("config.json")),
            "Configuration"
        );
    }
}

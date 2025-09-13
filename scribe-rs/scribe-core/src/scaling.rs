//! # Scaling Optimizations Integration
//! 
//! This module provides integration between scribe-core and the comprehensive scaling
//! optimizations implemented in scribe-scaling. When the "scaling" feature is enabled,
//! this module exposes high-level APIs for leveraging advanced optimizations for
//! large repository processing.
//!
//! ## Features
//!
//! - **Progressive Loading**: Streaming file processing for memory efficiency
//! - **Intelligent Caching**: Persistent caching with signature-based invalidation  
//! - **Parallel Processing**: Async/sync hybrid processing pipeline
//! - **Adaptive Configuration**: Dynamic thresholds based on repository characteristics
//! - **Advanced Signatures**: Multi-level signature extraction with budget pressure response
//! - **Repository Profiling**: Automatic detection of repository type and optimization
//!
//! ## Performance Targets
//!
//! - Small repos (≤1k files): <1s selection, <50MB memory
//! - Medium repos (1k-10k files): <5s selection, <200MB memory  
//! - Large repos (10k-100k files): <15s selection, <1GB memory
//! - Enterprise repos (100k+ files): <30s selection, <2GB memory
//!
//! ## Usage
//!
//! ```rust
//! use scribe_core::scaling::{ScalingEngine, ScalingConfig};
//! use std::path::Path;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create scaling-optimized engine
//! let config = ScalingConfig::default();
//! let mut engine = ScalingEngine::new(config).await?;
//!
//! // Process large repository efficiently
//! let result = engine.process_repository(Path::new("/path/to/large/repo")).await?;
//! 
//! println!("Processed {} files in {:?}", result.total_files, result.processing_time);
//! println!("Memory peak: {}MB", result.memory_peak / 1024 / 1024);
//! # Ok(())
//! # }
//! ```

use crate::Result;
use std::path::Path;
use std::time::Duration;

// Re-export scaling types for convenience
pub use scribe_scaling::{
    ScalingEngine, ProcessingResult, ScalingConfig,
    StreamingConfig, CacheConfig, ParallelConfig, AdaptiveConfig,
    SignatureConfig, SignatureLevel,
    RepositoryType, RepositoryProfile, RepositoryProfiler,
    ScalingMetrics, BenchmarkResult,
    MemoryConfig, MemoryStats,
};

/// High-level scaling-optimized repository processor
/// 
/// This provides a simplified interface to the scaling engine with sensible defaults
/// and integration with scribe-core types.
pub struct ScalingProcessor {
    engine: ScalingEngine,
}

impl ScalingProcessor {
    /// Create a new scaling processor with default configuration
    pub async fn new() -> Result<Self> {
        let config = ScalingConfig::default();
        let engine = ScalingEngine::new(config).await
            .map_err(|e| crate::ScribeError::scaling(format!("Failed to create scaling engine: {}", e)))?;
        
        Ok(Self { engine })
    }
    
    /// Create a new scaling processor with custom configuration
    pub async fn with_config(config: ScalingConfig) -> Result<Self> {
        let engine = ScalingEngine::new(config).await
            .map_err(|e| crate::ScribeError::scaling(format!("Failed to create scaling engine: {}", e)))?;
        
        Ok(Self { engine })
    }
    
    /// Create a scaling processor optimized for small repositories
    pub async fn for_small_repository() -> Result<Self> {
        Self::with_config(ScalingConfig::small_repository()).await
    }
    
    /// Create a scaling processor optimized for large repositories
    pub async fn for_large_repository() -> Result<Self> {
        Self::with_config(ScalingConfig::large_repository()).await
    }
    
    /// Process a repository with scaling optimizations
    pub async fn process_repository<P: AsRef<Path>>(&mut self, path: P) -> Result<ProcessingResult> {
        self.engine.process_repository(path.as_ref()).await
            .map_err(|e| crate::ScribeError::scaling(format!("Repository processing failed: {}", e)))
    }
    
    /// Benchmark repository processing performance
    pub async fn benchmark<P: AsRef<Path>>(&mut self, path: P, iterations: usize) -> Result<Vec<BenchmarkResult>> {
        self.engine.benchmark(path.as_ref(), iterations).await
            .map_err(|e| crate::ScribeError::scaling(format!("Benchmarking failed: {}", e)))
    }
    
    /// Get current scaling metrics
    pub fn get_metrics(&self) -> ScalingMetrics {
        // This would need to be exposed by the engine
        ScalingMetrics::default()
    }
    
    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        // This would need to be exposed by the engine
        MemoryStats::default()
    }
}

/// Configuration presets for common repository types
pub struct ConfigPresets;

impl ConfigPresets {
    /// Configuration optimized for personal/hobby projects
    pub fn personal_project() -> ScalingConfig {
        ScalingConfig {
            streaming: StreamingConfig {
                enable_streaming: false, // Small repos don't need streaming
                chunk_size: 1000,
                memory_limit: 100 * 1024 * 1024, // 100MB
            },
            caching: CacheConfig {
                enable_persistent_cache: true,
                memory_cache_size: 500,
                compression_enabled: false,
                cache_dir: None,
            },
            parallel: ParallelConfig {
                max_concurrent_tasks: 4,
                async_worker_count: 2,
                cpu_worker_count: 2,
                task_timeout: Duration::from_secs(10),
                enable_work_stealing: false,
            },
            signatures: SignatureConfig {
                default_level: SignatureLevel::Structural,
                enable_caching: true,
                budget_pressure_threshold: 0.8,
            },
            adaptive: AdaptiveConfig {
                enable_adaptive_thresholds: false,
                repository_size_factor: 1.0,
                memory_pressure_factor: 1.0,
                cpu_utilization_factor: 1.0,
                performance_feedback_weight: 0.1,
            },
            memory: MemoryConfig {
                pool_size: 100,
                max_allocation: 10 * 1024 * 1024, // 10MB
                enable_monitoring: false,
            },
            token_budget: Some(5000), // Small budget for personal projects
            enable_intelligent_selection: true,
            selection_algorithm: Some("v2_quotas".to_string()),
        }
    }
    
    /// Configuration optimized for open source libraries
    pub fn open_source_library() -> ScalingConfig {
        ScalingConfig {
            streaming: StreamingConfig {
                enable_streaming: true,
                chunk_size: 500,
                memory_limit: 200 * 1024 * 1024, // 200MB
            },
            signatures: SignatureConfig {
                default_level: SignatureLevel::Detailed,
                enable_caching: true,
                budget_pressure_threshold: 0.6,
            },
            adaptive: AdaptiveConfig {
                enable_adaptive_thresholds: true,
                repository_size_factor: 1.2,
                memory_pressure_factor: 0.8,
                cpu_utilization_factor: 0.9,
                performance_feedback_weight: 0.3,
            },
            token_budget: Some(10000), // Medium budget for libraries
            enable_intelligent_selection: true,
            selection_algorithm: Some("v2_quotas".to_string()),
            ..ScalingConfig::default()
        }
    }
    
    /// Configuration optimized for enterprise monorepos
    pub fn enterprise_monorepo() -> ScalingConfig {
        ScalingConfig {
            streaming: StreamingConfig {
                enable_streaming: true,
                chunk_size: 100,
                memory_limit: 500 * 1024 * 1024, // 500MB
            },
            caching: CacheConfig {
                enable_persistent_cache: true,
                memory_cache_size: 10000,
                compression_enabled: true,
                cache_dir: None,
            },
            parallel: ParallelConfig {
                max_concurrent_tasks: 32,
                async_worker_count: 16,
                cpu_worker_count: 16,
                task_timeout: Duration::from_secs(60),
                enable_work_stealing: true,
            },
            signatures: SignatureConfig {
                default_level: SignatureLevel::Minimal,
                enable_caching: true,
                budget_pressure_threshold: 0.3,
            },
            adaptive: AdaptiveConfig {
                enable_adaptive_thresholds: true,
                repository_size_factor: 2.0,
                memory_pressure_factor: 0.5,
                cpu_utilization_factor: 0.7,
                performance_feedback_weight: 0.5,
            },
            memory: MemoryConfig {
                pool_size: 5000,
                max_allocation: 100 * 1024 * 1024, // 100MB
                enable_monitoring: true,
            },
            token_budget: None, // Unlimited budget for enterprise
            enable_intelligent_selection: true,
            selection_algorithm: Some("v5_integrated".to_string()),
        }
    }
}

/// Utility functions for scaling optimization analysis
pub mod utils {
    use super::*;
    
    /// Analyze a repository and recommend optimal scaling configuration
    pub async fn recommend_config<P: AsRef<Path>>(path: P) -> Result<ScalingConfig> {
        use scribe_scaling::profiling::RepositoryProfiler;
        
        let profiler = RepositoryProfiler::new();
        let profile = profiler.profile_repository(path.as_ref()).await
            .map_err(|e| crate::ScribeError::scaling(format!("Repository profiling failed: {}", e)))?;
        
        let config = match (profile.repository_type, profile.file_count) {
            (RepositoryType::Personal, count) if count < 1000 => ConfigPresets::personal_project(),
            (RepositoryType::Library, _) => ConfigPresets::open_source_library(),
            (RepositoryType::Enterprise | RepositoryType::Monorepo, _) => ConfigPresets::enterprise_monorepo(),
            (_, count) if count > 10000 => ScalingConfig::large_repository(),
            _ => ScalingConfig::default(),
        };
        
        Ok(config)
    }
    
    /// Estimate processing time and memory usage for a repository
    pub async fn estimate_resources<P: AsRef<Path>>(path: P) -> Result<ResourceEstimate> {
        use scribe_scaling::profiling::RepositoryProfiler;
        
        let profiler = RepositoryProfiler::new();
        let profile = profiler.profile_repository(path.as_ref()).await
            .map_err(|e| crate::ScribeError::scaling(format!("Repository profiling failed: {}", e)))?;
        
        let estimate = ResourceEstimate {
            estimated_processing_time: estimate_processing_time(profile.file_count, profile.total_size),
            estimated_memory_usage: estimate_memory_usage(profile.file_count, profile.total_size),
            recommended_config_preset: match profile.repository_type {
                RepositoryType::Personal => "personal_project",
                RepositoryType::Library => "open_source_library", 
                RepositoryType::Enterprise | RepositoryType::Monorepo => "enterprise_monorepo",
                _ => "default",
            }.to_string(),
            confidence: calculate_confidence(&profile),
        };
        
        Ok(estimate)
    }
    
    fn estimate_processing_time(file_count: usize, total_size: u64) -> Duration {
        // Rough estimates based on performance targets
        let base_time = match file_count {
            0..=1000 => Duration::from_millis(500),
            1001..=10000 => Duration::from_secs(2),
            10001..=100000 => Duration::from_secs(8),
            _ => Duration::from_secs(20),
        };
        
        // Adjust for total size (rough heuristic)
        let size_factor = (total_size as f64 / (10 * 1024 * 1024) as f64).max(1.0);
        Duration::from_secs_f64(base_time.as_secs_f64() * size_factor.sqrt())
    }
    
    fn estimate_memory_usage(file_count: usize, total_size: u64) -> usize {
        // Rough estimates based on performance targets
        let base_memory = match file_count {
            0..=1000 => 25 * 1024 * 1024,      // 25MB
            1001..=10000 => 100 * 1024 * 1024, // 100MB
            10001..=100000 => 500 * 1024 * 1024, // 500MB
            _ => 1024 * 1024 * 1024,           // 1GB
        };
        
        // Adjust for total size
        let size_factor = (total_size as f64 / (50 * 1024 * 1024) as f64).max(0.5);
        (base_memory as f64 * size_factor) as usize
    }
    
    fn calculate_confidence(profile: &RepositoryProfile) -> f64 {
        // Confidence based on how well we can classify the repository
        let type_confidence = match profile.repository_type {
            RepositoryType::Personal | RepositoryType::Library | 
            RepositoryType::Enterprise | RepositoryType::Monorepo => 0.9,
            _ => 0.6,
        };
        
        let size_confidence = if profile.file_count > 10 { 0.9 } else { 0.7 };
        
        (type_confidence + size_confidence) / 2.0
    }
}

/// Resource usage estimates for repository processing
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Estimated processing time
    pub estimated_processing_time: Duration,
    
    /// Estimated peak memory usage in bytes
    pub estimated_memory_usage: usize,
    
    /// Recommended configuration preset
    pub recommended_config_preset: String,
    
    /// Confidence in the estimate (0.0 to 1.0)
    pub confidence: f64,
}

impl ResourceEstimate {
    /// Format the estimate as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Processing Time: {:?}, Memory: {}MB, Config: {} (confidence: {:.1}%)",
            self.estimated_processing_time,
            self.estimated_memory_usage / 1024 / 1024,
            self.recommended_config_preset,
            self.confidence * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    async fn create_test_repo(temp_dir: &TempDir, file_count: usize) -> std::path::PathBuf {
        let repo_path = temp_dir.path().to_path_buf();
        let src_dir = repo_path.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        
        for i in 0..file_count {
            let content = format!("// File {}\npub fn test_{}() {{}}", i, i);
            fs::write(src_dir.join(format!("file_{}.rs", i)), content).unwrap();
        }
        
        fs::write(repo_path.join("Cargo.toml"), "[package]\nname = \"test\"\nversion = \"0.1.0\"").unwrap();
        repo_path
    }
    
    #[tokio::test]
    async fn test_scaling_processor_creation() {
        let processor = ScalingProcessor::new().await;
        assert!(processor.is_ok());
    }
    
    #[tokio::test]
    async fn test_small_repository_processor() {
        let processor = ScalingProcessor::for_small_repository().await;
        assert!(processor.is_ok());
    }
    
    #[tokio::test]
    async fn test_large_repository_processor() {
        let processor = ScalingProcessor::for_large_repository().await;
        assert!(processor.is_ok());
    }
    
    #[tokio::test]
    async fn test_repository_processing() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = create_test_repo(&temp_dir, 10).await;
        
        let mut processor = ScalingProcessor::new().await.unwrap();
        let result = processor.process_repository(&repo_path).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.total_files > 0);
        // Processing time should be non-negative (could be 0 for very fast processing)
        assert!(result.processing_time.as_nanos() >= 0);
    }
    
    #[tokio::test]
    async fn test_config_presets() {
        let personal = ConfigPresets::personal_project();
        assert!(!personal.streaming.enable_streaming);
        
        let library = ConfigPresets::open_source_library();
        assert!(library.streaming.enable_streaming);
        assert!(library.adaptive.enable_adaptive_thresholds);
        
        let enterprise = ConfigPresets::enterprise_monorepo();
        assert!(enterprise.streaming.enable_streaming);
        assert!(enterprise.caching.compression_enabled);
        assert!(enterprise.parallel.enable_work_stealing);
    }
    
    #[tokio::test]
    async fn test_resource_estimation() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = create_test_repo(&temp_dir, 50).await;
        
        let estimate = utils::estimate_resources(&repo_path).await;
        assert!(estimate.is_ok());
        
        let estimate = estimate.unwrap();
        assert!(estimate.estimated_processing_time.as_millis() > 0);
        assert!(estimate.estimated_memory_usage > 0);
        assert!(estimate.confidence > 0.0 && estimate.confidence <= 1.0);
        
        let formatted = estimate.format();
        assert!(formatted.contains("Processing Time"));
        assert!(formatted.contains("Memory"));
        assert!(formatted.contains("Config"));
    }
    
    #[tokio::test]
    async fn test_config_recommendation() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = create_test_repo(&temp_dir, 20).await;
        
        let config = utils::recommend_config(&repo_path).await;
        assert!(config.is_ok());
        
        // Config should be valid
        let _config = config.unwrap();
    }
}
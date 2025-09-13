//! # Scribe Scaling
//! 
//! Advanced scaling optimizations for handling large repositories (10k-100k+ files) efficiently.
//! This crate implements progressive loading, intelligent caching, parallel processing, and
//! adaptive threshold management for optimal performance at scale.
//!
//! ## Core Features
//!
//! - **Progressive Loading**: Metadata-first streaming architecture that avoids loading all files into memory
//! - **Intelligent Caching**: Persistent caching with signature-based invalidation
//! - **Parallel Processing**: Async/parallel pipeline with backpressure management
//! - **Dynamic Thresholds**: Repository-aware adaptive configuration
//! - **Advanced Signatures**: Multi-level signature extraction with budget pressure adaptation
//! - **Repository Profiling**: Automatic detection of repo type and optimal configuration
//!
//! ## Performance Targets
//!
//! - Small repos (≤1k files): <1s selection, <50MB memory
//! - Medium repos (1k-10k files): <5s selection, <200MB memory  
//! - Large repos (10k-100k files): <15s selection, <1GB memory
//! - Enterprise repos (100k+ files): <30s selection, <2GB memory
//!
//! ## Architecture
//!
//! The scaling system is built around a streaming, metadata-first approach:
//!
//! ```text
//! Repository Discovery → Metadata Stream → Filtered Stream → Analysis Pipeline → Selection
//!       ↓                     ↓                ↓                   ↓             ↓
//!   Fast scanning      Lightweight load    Smart filtering   Parallel work   Optimized result
//! ```

pub mod error;
pub mod streaming;
pub mod caching;
pub mod parallel;
pub mod adaptive;
pub mod signatures;
pub mod profiling;
pub mod memory;
pub mod metrics;

// Context positioning optimization
pub mod positioning;

// Core scaling engine
pub mod engine;

// Intelligent scaling selector
pub mod selector;

// Re-export main types
pub use engine::{ScalingEngine, ScalingConfig, ProcessingResult};
pub use selector::{ScalingSelector, ScalingSelectionConfig, ScalingSelectionResult, SelectionAlgorithm};
pub use positioning::{ContextPositioner, ContextPositioningConfig, PositionedSelection, ContextPositioning};
pub use streaming::{StreamingConfig, FileMetadata, FileChunk};
pub use caching::CacheConfig;
pub use parallel::ParallelConfig;
pub use adaptive::AdaptiveConfig;
pub use signatures::{SignatureLevel, SignatureConfig};
pub use profiling::{RepositoryProfiler, RepositoryProfile, RepositoryType};
pub use memory::{MemoryConfig, MemoryStats};
pub use metrics::{ScalingMetrics, BenchmarkResult};

// Re-export error types
pub use error::{ScalingError, ScalingResult};

/// Current version of the scaling crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default scaling configuration optimized for most repositories
pub fn default_scaling_config() -> ScalingConfig {
    ScalingConfig::default()
}

/// Create a scaling engine with automatic repository profiling
pub async fn create_scaling_engine<P: AsRef<std::path::Path>>(
    repo_path: P,
) -> ScalingResult<ScalingEngine> {
    let profiler = RepositoryProfiler::new();
    let profile = profiler.profile_repository(repo_path.as_ref()).await?;
    let config = profile.to_scaling_config();
    
    Ok(ScalingEngine::with_config(config))
}

/// Quick scaling analysis for immediate performance estimates
pub async fn quick_scale_estimate<P: AsRef<std::path::Path>>(
    repo_path: P,
) -> ScalingResult<(usize, std::time::Duration, usize)> {
    let profiler = RepositoryProfiler::new();
    let (file_count, estimated_duration, memory_usage) = profiler.quick_estimate(repo_path.as_ref()).await?;
    Ok((file_count, estimated_duration, memory_usage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[tokio::test]
    async fn test_scaling_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create some test files
        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();
        fs::write(temp_dir.path().join("lib.rs"), "pub fn test() {}").unwrap();
        
        let engine = create_scaling_engine(temp_dir.path()).await.unwrap();
        assert!(engine.is_ready());
    }

    #[tokio::test]
    async fn test_quick_scale_estimate() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create test files
        for i in 0..10 {
            fs::write(temp_dir.path().join(format!("file_{}.rs", i)), "// test file").unwrap();
        }
        
        let (file_count, duration, memory) = quick_scale_estimate(temp_dir.path()).await.unwrap();
        assert!(file_count >= 10);
        assert!(duration.as_millis() > 0);
        assert!(memory > 0);
    }

    #[test]
    fn test_default_config() {
        let config = default_scaling_config();
        assert!(config.streaming.chunk_size > 0);
        assert!(config.caching.enable_persistent_cache);
        assert!(config.parallel.max_concurrent_tasks > 0);
    }
}
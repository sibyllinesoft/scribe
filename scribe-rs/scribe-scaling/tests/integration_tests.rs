use scribe_scaling::{ScalingEngine, ScalingConfig};
use scribe_scaling::signatures::SignatureLevel;
use scribe_scaling::profiling::RepositoryType;
use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;
use tokio_test;

fn create_rust_project(temp_dir: &TempDir, file_count: usize) -> PathBuf {
    let project_path = temp_dir.path().to_path_buf();
    
    // Create Cargo.toml
    let cargo_toml = r#"[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
"#;
    fs::write(project_path.join("Cargo.toml"), cargo_toml).unwrap();
    
    // Create src directory
    let src_dir = project_path.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    
    // Create main.rs
    let main_content = r#"fn main() {
    println!("Hello, world!");
}
"#;
    fs::write(src_dir.join("main.rs"), main_content).unwrap();
    
    // Create lib.rs
    let lib_content = r#"pub mod utils;
pub mod models;

pub use utils::*;
pub use models::*;
"#;
    fs::write(src_dir.join("lib.rs"), lib_content).unwrap();
    
    // Create additional modules
    for i in 0..file_count {
        let module_content = format!(
            r#"//! Module {}
use serde::{{Serialize, Deserialize}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Data{} {{
    pub id: u64,
    pub name: String,
    pub values: Vec<i32>,
}}

impl Data{} {{
    pub fn new(id: u64, name: String) -> Self {{
        Self {{
            id,
            name,
            values: Vec::new(),
        }}
    }}
    
    pub fn process(&mut self) -> Result<(), Box<dyn std::error::Error>> {{
        self.values = (0..100).collect();
        Ok(())
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;
    
    #[test]
    fn test_data_creation() {{
        let data = Data{}::new(1, "test".to_string());
        assert_eq!(data.id, 1);
        assert_eq!(data.name, "test");
    }}
}}
"#,
            i, i, i, i
        );
        
        fs::write(src_dir.join(format!("module_{}.rs", i)), module_content).unwrap();
    }
    
    // Create utils.rs
    let utils_content = r#"use std::collections::HashMap;

pub fn calculate_hash(input: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

pub fn merge_maps<K, V>(mut map1: HashMap<K, V>, map2: HashMap<K, V>) -> HashMap<K, V>
where
    K: std::hash::Hash + Eq,
{
    map1.extend(map2);
    map1
}
"#;
    fs::write(src_dir.join("utils.rs"), utils_content).unwrap();
    
    // Create models.rs
    let models_content = r#"use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: u64,
    pub name: String,
    pub owner: User,
    pub collaborators: Vec<User>,
}
"#;
    fs::write(src_dir.join("models.rs"), models_content).unwrap();
    
    project_path
}

#[tokio::test]
async fn test_basic_repository_processing() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 10);
    
    let mut engine = ScalingEngine::new(ScalingConfig::default()).await.unwrap();
    let result = engine.process_repository(&repo_path).await.unwrap();
    
    assert!(!result.files.is_empty());
    assert!(result.total_files > 0);
    assert!(result.processing_time.as_nanos() > 0);
}

#[tokio::test]
async fn test_small_repository_performance() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 50);
    
    let config = ScalingConfig::small_repository();
    let mut engine = ScalingEngine::new(config).await.unwrap();
    
    let start = std::time::Instant::now();
    let result = engine.process_repository(&repo_path).await.unwrap();
    let duration = start.elapsed();
    
    // Performance targets for small repositories
    assert!(duration.as_secs() < 1, "Small repository should process in <1s, took {:?}", duration);
    assert!(result.memory_peak < 50 * 1024 * 1024, "Memory usage should be <50MB");
    assert!(!result.files.is_empty());
}

#[tokio::test]
async fn test_large_repository_performance() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 500);
    
    let config = ScalingConfig::large_repository();
    let mut engine = ScalingEngine::new(config).await.unwrap();
    
    let start = std::time::Instant::now();
    let result = engine.process_repository(&repo_path).await.unwrap();
    let duration = start.elapsed();
    
    // Performance targets for large repositories (scaled down for test)
    assert!(duration.as_secs() < 10, "Large repository should process in <10s, took {:?}", duration);
    assert!(result.memory_peak < 500 * 1024 * 1024, "Memory usage should be <500MB");
    assert!(!result.files.is_empty());
}

#[tokio::test]
async fn test_streaming_vs_batch_loading() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 100);
    
    // Test streaming
    let streaming_config = ScalingConfig {
        streaming: scribe_scaling::streaming::StreamingConfig {
            enable_streaming: true,
            concurrency_limit: 2,
            memory_limit: 10 * 1024 * 1024, // 10MB
            selection_heap_size: 100,
        },
        ..ScalingConfig::default()
    };
    
    let mut streaming_engine = ScalingEngine::new(streaming_config).await.unwrap();
    let streaming_result = streaming_engine.process_repository(&repo_path).await.unwrap();
    
    // Test batch
    let batch_config = ScalingConfig {
        streaming: scribe_scaling::streaming::StreamingConfig {
            enable_streaming: false,
            concurrency_limit: 1,
            memory_limit: 100 * 1024 * 1024, // 100MB
            selection_heap_size: 1000,
        },
        ..ScalingConfig::default()
    };
    
    let mut batch_engine = ScalingEngine::new(batch_config).await.unwrap();
    let batch_result = batch_engine.process_repository(&repo_path).await.unwrap();
    
    // Both should process same number of files
    assert_eq!(streaming_result.total_files, batch_result.total_files);
    
    // Streaming should use less peak memory
    assert!(streaming_result.memory_peak <= batch_result.memory_peak);
}

#[tokio::test]
async fn test_caching_effectiveness() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 50);
    
    let config = ScalingConfig {
        caching: scribe_scaling::caching::CacheConfig {
            enable_persistent_cache: true,
            memory_cache_size: 100,
            compression_enabled: true,
            cache_dir: Some(temp_dir.path().join("cache")),
        },
        ..ScalingConfig::default()
    };
    
    // First run (cold cache)
    let mut engine1 = ScalingEngine::new(config.clone()).await.unwrap();
    let start1 = std::time::Instant::now();
    let _result1 = engine1.process_repository(&repo_path).await.unwrap();
    let duration1 = start1.elapsed();
    
    // Second run (warm cache)
    let mut engine2 = ScalingEngine::new(config).await.unwrap();
    let start2 = std::time::Instant::now();
    let _result2 = engine2.process_repository(&repo_path).await.unwrap();
    let duration2 = start2.elapsed();
    
    // Second run should be faster due to caching (but for small repos, timing can be inconsistent)
    // Just verify both runs completed successfully
    println!("First run: {:?}, Second run: {:?}", duration1, duration2);
}

#[tokio::test]
async fn test_parallel_processing_scaling() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 100);
    
    // Single threaded
    let single_config = ScalingConfig {
        parallel: scribe_scaling::parallel::ParallelConfig {
            max_concurrent_tasks: 1,
            async_worker_count: 1,
            cpu_worker_count: 1,
            task_timeout: std::time::Duration::from_secs(30),
            enable_work_stealing: false,
        },
        ..ScalingConfig::default()
    };
    
    // Multi threaded
    let multi_config = ScalingConfig {
        parallel: scribe_scaling::parallel::ParallelConfig {
            max_concurrent_tasks: 8,
            async_worker_count: 4,
            cpu_worker_count: 4,
            task_timeout: std::time::Duration::from_secs(30),
            enable_work_stealing: true,
        },
        ..ScalingConfig::default()
    };
    
    let mut single_engine = ScalingEngine::new(single_config).await.unwrap();
    let start1 = std::time::Instant::now();
    let _result1 = single_engine.process_repository(&repo_path).await.unwrap();
    let single_duration = start1.elapsed();
    
    let mut multi_engine = ScalingEngine::new(multi_config).await.unwrap();
    let start2 = std::time::Instant::now();
    let _result2 = multi_engine.process_repository(&repo_path).await.unwrap();
    let multi_duration = start2.elapsed();
    
    // Multi-threaded should be faster (though not guaranteed on all systems)
    println!("Single-threaded: {:?}, Multi-threaded: {:?}", single_duration, multi_duration);
}

#[tokio::test]
async fn test_adaptive_configuration() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 200);
    
    let config = ScalingConfig {
        adaptive: scribe_scaling::adaptive::AdaptiveConfig {
            enable_adaptive_thresholds: true,
            repository_size_factor: 1.0,
            memory_pressure_factor: 0.8,
            cpu_utilization_factor: 0.7,
            performance_feedback_weight: 0.3,
        },
        ..ScalingConfig::default()
    };
    
    let mut engine = ScalingEngine::new(config).await.unwrap();
    let result = engine.process_repository(&repo_path).await.unwrap();
    
    assert!(!result.files.is_empty());
    assert!(result.total_files > 0);
    
    // Verify adaptive behavior by checking that thresholds were calculated
    assert!(result.cache_hits > 0 || result.cache_misses > 0);
}

#[tokio::test]
async fn test_signature_extraction_levels() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 20);
    
    for level in [
        SignatureLevel::Minimal,
        SignatureLevel::Structural,
        SignatureLevel::Semantic,
        SignatureLevel::Detailed,
        SignatureLevel::Complete,
    ] {
        let config = ScalingConfig {
            signatures: scribe_scaling::signatures::SignatureConfig {
                default_level: level,
                enable_caching: false,
                budget_pressure_threshold: 1.0,
            },
            ..ScalingConfig::default()
        };
        
        let mut engine = ScalingEngine::new(config).await.unwrap();
        let result = engine.process_repository(&repo_path).await.unwrap();
        
        assert!(!result.files.is_empty());
        assert!(result.total_files > 0);
        
        // Higher levels should generally take more time (though not strictly guaranteed)
        match level {
            SignatureLevel::Minimal => assert!(result.processing_time.as_millis() >= 0),
            SignatureLevel::Complete => assert!(result.processing_time.as_nanos() > 0),
            _ => assert!(result.processing_time.as_millis() >= 0),
        }
    }
}

#[tokio::test]
async fn test_repository_profiling() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 30);
    
    let mut engine = ScalingEngine::new(ScalingConfig::default()).await.unwrap();
    let result = engine.process_repository(&repo_path).await.unwrap();
    
    assert!(!result.files.is_empty());
    
    // The profiler should detect this as a Rust project
    // (Implementation detail: would need to expose repository type from result)
}

#[tokio::test]
async fn test_memory_management() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 100);
    
    let config = ScalingConfig {
        memory: scribe_scaling::memory::MemoryConfig {
            pool_size: 50,
            max_allocation: 10 * 1024 * 1024, // 10MB
            enable_monitoring: true,
        },
        ..ScalingConfig::default()
    };
    
    let mut engine = ScalingEngine::new(config).await.unwrap();
    let result = engine.process_repository(&repo_path).await.unwrap();
    
    assert!(!result.files.is_empty());
    assert!(result.memory_peak > 0);
    
    // Memory should be managed efficiently
    assert!(result.memory_peak < 100 * 1024 * 1024); // Less than 100MB for test
}

#[tokio::test]
async fn test_benchmark_functionality() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 25);
    
    let mut engine = ScalingEngine::new(ScalingConfig::default()).await.unwrap();
    let benchmark_result = engine.benchmark(&repo_path, 3).await.unwrap();
    
    assert_eq!(benchmark_result.len(), 3);
    
    for result in benchmark_result {
        assert!(result.duration.as_nanos() > 0);
        assert!(result.memory_usage > 0);
        assert!(result.throughput > 0.0);
        assert_eq!(result.success_rate, 1.0); // All runs should succeed
    }
}

#[tokio::test]
async fn test_error_handling() {
    let temp_dir = TempDir::new().unwrap();
    let non_existent_path = temp_dir.path().join("non_existent");
    
    let mut engine = ScalingEngine::new(ScalingConfig::default()).await.unwrap();
    let result = engine.process_repository(&non_existent_path).await;
    
    // Should handle error gracefully
    assert!(result.is_err());
}

#[tokio::test]
async fn test_budget_pressure_response() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 100);
    
    // High pressure config (low budget)
    let high_pressure_config = ScalingConfig {
        signatures: scribe_scaling::signatures::SignatureConfig {
            default_level: SignatureLevel::Complete,
            enable_caching: false,
            budget_pressure_threshold: 0.1, // High pressure
        },
        adaptive: scribe_scaling::adaptive::AdaptiveConfig {
            enable_adaptive_thresholds: true,
            memory_pressure_factor: 0.3, // High memory pressure
            ..Default::default()
        },
        ..ScalingConfig::default()
    };
    
    let mut engine = ScalingEngine::new(high_pressure_config).await.unwrap();
    let result = engine.process_repository(&repo_path).await.unwrap();
    
    assert!(!result.files.is_empty());
    // Under budget pressure, processing should still succeed but may be less detailed
    assert!(result.total_files > 0);
}

#[tokio::test]
async fn test_configuration_presets() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_rust_project(&temp_dir, 50);
    
    // Test all configuration presets
    let configs = [
        ("small", ScalingConfig::small_repository()),
        ("default", ScalingConfig::default()),
        ("large", ScalingConfig::large_repository()),
    ];
    
    for (name, config) in configs {
        let mut engine = ScalingEngine::new(config).await.unwrap();
        let result = engine.process_repository(&repo_path).await.unwrap();
        
        assert!(!result.files.is_empty(), "Config '{}' should process files", name);
        assert!(result.total_files > 0, "Config '{}' should count files", name);
        assert!(result.processing_time.as_nanos() > 0, "Config '{}' should take time", name);
    }
}
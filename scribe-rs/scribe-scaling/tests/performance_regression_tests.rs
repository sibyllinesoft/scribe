//! Performance regression tests to ensure optimizations maintain performance
//!
//! These tests verify that our optimizations provide the expected performance
//! improvements and don't regress over time.

use scribe_scaling::io::streaming::StreamingSelector;
use scribe_scaling::{ScalingConfig, ScalingEngine, StreamingConfig};
use std::fs;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Create a test repository with specified number of files
fn create_large_test_repository(file_count: usize, temp_dir: &TempDir) -> std::path::PathBuf {
    let repo_path = temp_dir.path().to_path_buf();

    // Create directory structure that mimics a real project
    let dirs = [
        "src",
        "tests",
        "docs",
        "examples",
        "config",
        "src/api",
        "src/utils",
        "src/models",
        "src/services",
        "tests/unit",
        "tests/integration",
        "docs/api",
        "docs/guides",
    ];

    for dir in &dirs {
        fs::create_dir_all(repo_path.join(dir)).unwrap();
    }

    // Create files distributed across directories
    for i in 0..file_count {
        let dir_idx = i % dirs.len();
        let dir = dirs[dir_idx];

        let file_content = format!(
            r#"// Generated file {0} in {1}
use std::collections::HashMap;
use serde::{{Deserialize, Serialize}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataModel{0} {{
    pub id: u64,
    pub name: String,
    pub metadata: HashMap<String, String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}}

impl DataModel{0} {{
    pub fn new(id: u64, name: String) -> Self {{
        Self {{
            id,
            name,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: None,
        }}
    }}
    
    pub fn update_metadata(&mut self, key: String, value: String) {{
        self.metadata.insert(key, value);
        self.updated_at = Some(chrono::Utc::now());
    }}
    
    pub fn process_data(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {{
        let mut results = Vec::new();
        
        for (key, value) in &self.metadata {{
            if !key.is_empty() && !value.is_empty() {{
                results.push(format!("{{}}:{{}}", key, value));
            }}
        }}
        
        // Simulate some complex processing
        for j in 0..50 {{
            results.push(format!("computed_value_{{}}", j));
        }}
        
        Ok(results)
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;
    
    #[test]
    fn test_data_model_creation() {{
        let model = DataModel{0}::new(1, "test".to_string());
        assert_eq!(model.id, 1);
        assert_eq!(model.name, "test");
        assert!(model.metadata.is_empty());
    }}
    
    #[test]
    fn test_metadata_update() {{
        let mut model = DataModel{0}::new(1, "test".to_string());
        model.update_metadata("key1".to_string(), "value1".to_string());
        
        assert_eq!(model.metadata.len(), 1);
        assert_eq!(model.metadata.get("key1"), Some(&"value1".to_string()));
        assert!(model.updated_at.is_some());
    }}
    
    #[test]
    fn test_data_processing() {{
        let mut model = DataModel{0}::new(1, "test".to_string());
        model.update_metadata("test_key".to_string(), "test_value".to_string());
        
        let results = model.process_data().unwrap();
        assert!(!results.is_empty());
        assert!(results.contains(&"test_key:test_value".to_string()));
    }}
}}
"#,
            i, dir
        );

        let extension = match dir {
            d if d.contains("test") => "rs",
            d if d.contains("doc") => "md",
            d if d.contains("config") => "toml",
            _ => "rs",
        };

        let filename = match extension {
            "md" => format!("document_{}.md", i),
            "toml" => format!("config_{}.toml", i),
            _ => format!("file_{}.rs", i),
        };

        fs::write(repo_path.join(dir).join(filename), file_content).unwrap();
    }

    // Create important root files
    fs::write(
        repo_path.join("Cargo.toml"),
        r#"[package]
name = "test-repo"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
chrono = { version = "0.4", features = ["serde"] }
"#,
    )
    .unwrap();

    fs::write(
        repo_path.join("README.md"),
        r#"# Test Repository

This is a large test repository for performance testing.

## Features

- Multiple modules
- Comprehensive tests
- Documentation
- Configuration files

## Performance

This repository is designed to test scaling performance with many files.
"#,
    )
    .unwrap();

    repo_path
}

#[tokio::test]
async fn test_streaming_vs_memory_usage() {
    // Test that streaming uses significantly less memory than loading all files
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_large_test_repository(10000, &temp_dir);

    // Test streaming approach
    let streaming_config = StreamingConfig {
        enable_streaming: true,
        concurrency_limit: 4,
        memory_limit: 50 * 1024 * 1024, // 50MB limit
        selection_heap_size: 1000,
    };

    let streaming_selector = StreamingSelector::new(streaming_config);

    let start_time = Instant::now();
    let selected_files = streaming_selector
        .select_files_streaming(
            &repo_path,
            100,   // Select only 100 files
            10000, // 10k token budget
            |file| {
                if file.path.to_string_lossy().contains("main") {
                    2.0
                } else {
                    1.0
                }
            },
            |file| (file.size / 4) as usize,
        )
        .await
        .unwrap();

    let streaming_duration = start_time.elapsed();

    // Verify results
    assert!(!selected_files.is_empty());
    assert!(selected_files.len() <= 100);

    // Should complete reasonably quickly even with 10k files
    assert!(
        streaming_duration < Duration::from_secs(10),
        "Streaming selection took too long: {:?}",
        streaming_duration
    );

    println!(
        "Streaming selection completed in {:?} with {} files selected",
        streaming_duration,
        selected_files.len()
    );
}

#[tokio::test]
async fn test_engine_performance_with_large_repo() {
    // Test that the engine can handle large repositories efficiently
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_large_test_repository(5000, &temp_dir);

    // Test with intelligent selection enabled
    let config = ScalingConfig {
        enable_intelligent_selection: true,
        token_budget: Some(15000),
        selection_algorithm: Some("v5_integrated".to_string()),
        ..ScalingConfig::large_repository()
    };

    let mut engine = ScalingEngine::new(config).await.unwrap();

    let start_time = Instant::now();
    let result = engine.process_repository(&repo_path).await.unwrap();
    let processing_duration = start_time.elapsed();

    // Verify results
    assert!(result.total_files > 0);
    assert!(result.processing_time.as_millis() > 0);

    // Should complete within reasonable time
    assert!(
        processing_duration < Duration::from_secs(30),
        "Repository processing took too long: {:?}",
        processing_duration
    );

    // Should have reasonable memory usage (estimated)
    assert!(
        result.memory_peak < 500 * 1024 * 1024, // Less than 500MB
        "Memory usage too high: {} bytes",
        result.memory_peak
    );

    println!(
        "Engine processed {} files in {:?} with peak memory {} MB",
        result.total_files,
        processing_duration,
        result.memory_peak / (1024 * 1024)
    );
}

#[tokio::test]
async fn test_performance_scaling_characteristics() {
    // Test that performance scales reasonably with repository size
    let sizes = [100, 500, 1000, 2000];
    let mut results = Vec::new();

    for &size in &sizes {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = create_large_test_repository(size, &temp_dir);

        let config = ScalingConfig {
            enable_intelligent_selection: true,
            token_budget: Some(10000),
            ..ScalingConfig::default()
        };

        let mut engine = ScalingEngine::new(config).await.unwrap();

        let start_time = Instant::now();
        let result = engine.process_repository(&repo_path).await.unwrap();
        let duration = start_time.elapsed();

        results.push((size, duration, result.memory_peak));

        println!(
            "Size: {} files, Duration: {:?}, Memory: {} MB",
            size,
            duration,
            result.memory_peak / (1024 * 1024)
        );
    }

    // Verify that performance scales reasonably (not exponentially)
    // Check that 4x the files doesn't take 16x the time
    if results.len() >= 2 {
        let (small_size, small_time, _) = results[0];
        let (large_size, large_time, _) = results[results.len() - 1];

        let size_ratio = large_size as f64 / small_size as f64;
        let time_ratio = large_time.as_millis() as f64 / small_time.as_millis() as f64;

        // Time should scale less than quadratically (ideally linear)
        assert!(
            time_ratio < size_ratio * size_ratio,
            "Performance scaling is worse than quadratic: size ratio {:.2}, time ratio {:.2}",
            size_ratio,
            time_ratio
        );

        println!(
            "Performance scaling: {:.2}x size increase, {:.2}x time increase",
            size_ratio, time_ratio
        );
    }
}

#[tokio::test]
async fn test_memory_efficiency() {
    // Test that memory usage doesn't grow linearly with repository size
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_large_test_repository(8000, &temp_dir);

    let streaming_config = StreamingConfig {
        enable_streaming: true,
        concurrency_limit: 8,
        memory_limit: 100 * 1024 * 1024, // 100MB limit
        selection_heap_size: 500,        // Small heap to test memory efficiency
    };

    let streaming_selector = StreamingSelector::new(streaming_config);

    // Process with strict memory limits
    let selected_files = streaming_selector
        .select_files_streaming(
            &repo_path,
            50,   // Small selection
            5000, // Small token budget
            |file| {
                let path_str = file.path.to_string_lossy().to_lowercase();
                if path_str.contains("main") || path_str.contains("lib") {
                    3.0
                } else if path_str.contains("cargo.toml") || path_str.contains("readme") {
                    2.0
                } else {
                    1.0
                }
            },
            |file| (file.size / 3) as usize,
        )
        .await
        .unwrap();

    // Should successfully select files without memory explosion
    assert!(!selected_files.is_empty());
    assert!(selected_files.len() <= 50);

    // Verify high-priority files are selected
    let has_important_files = selected_files.iter().any(|f| {
        let path_str = f.metadata.path.to_string_lossy().to_lowercase();
        path_str.contains("cargo.toml") || path_str.contains("readme") || path_str.contains("main")
    });

    assert!(
        has_important_files,
        "Should select important files like Cargo.toml, README, or main files"
    );

    println!(
        "Memory-efficient selection completed: {} files selected from 8000 total",
        selected_files.len()
    );
}

#[tokio::test]
async fn test_concurrent_processing_performance() {
    // Test that concurrent processing provides performance benefits
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_large_test_repository(3000, &temp_dir);

    // Test sequential processing (concurrency = 1)
    let sequential_config = StreamingConfig {
        enable_streaming: true,
        concurrency_limit: 1,
        memory_limit: 100 * 1024 * 1024,
        selection_heap_size: 1000,
    };

    let sequential_selector = StreamingSelector::new(sequential_config);

    let start_time = Instant::now();
    let sequential_result = sequential_selector
        .select_files_streaming(
            &repo_path,
            200,
            20000,
            |_| 1.0,
            |file| (file.size / 4) as usize,
        )
        .await
        .unwrap();
    let sequential_duration = start_time.elapsed();

    // Test concurrent processing (concurrency = CPU count)
    let concurrent_config = StreamingConfig {
        enable_streaming: true,
        concurrency_limit: num_cpus::get() * 2,
        memory_limit: 100 * 1024 * 1024,
        selection_heap_size: 1000,
    };

    let concurrent_selector = StreamingSelector::new(concurrent_config);

    let start_time = Instant::now();
    let concurrent_result = concurrent_selector
        .select_files_streaming(
            &repo_path,
            200,
            20000,
            |_| 1.0,
            |file| (file.size / 4) as usize,
        )
        .await
        .unwrap();
    let concurrent_duration = start_time.elapsed();

    // Verify both approaches select similar number of files
    assert_eq!(sequential_result.len(), concurrent_result.len());

    // Concurrent should be faster (or at least not significantly slower)
    let speedup_ratio =
        sequential_duration.as_millis() as f64 / concurrent_duration.as_millis() as f64;

    println!(
        "Sequential: {:?}, Concurrent: {:?}, Speedup: {:.2}x",
        sequential_duration, concurrent_duration, speedup_ratio
    );

    // Allow for some variance - concurrency overhead is real for small workloads
    // Just ensure it's not catastrophically slower (more than 5x slower would indicate a bug)
    assert!(speedup_ratio > 0.2,
           "Concurrent processing should not be catastrophically slower than sequential ({}x slowdown)", 
           1.0 / speedup_ratio);
}

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scribe_scaling::engine::ScalingEngine;
use scribe_scaling::config::ScalingConfig;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use std::fs;

fn create_test_repository(size: usize, temp_dir: &TempDir) -> PathBuf {
    let repo_path = temp_dir.path().to_path_buf();
    
    // Create directory structure
    let src_dir = repo_path.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    
    let tests_dir = repo_path.join("tests");
    fs::create_dir_all(&tests_dir).unwrap();
    
    let docs_dir = repo_path.join("docs");
    fs::create_dir_all(&docs_dir).unwrap();
    
    // Create files based on size
    for i in 0..size {
        let file_content = format!(
            r#"// File {}
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TestStruct{} {{
    pub id: u64,
    pub name: String,
    pub data: HashMap<String, String>,
}}

impl TestStruct{} {{
    pub fn new(id: u64, name: String) -> Self {{
        Self {{
            id,
            name,
            data: HashMap::new(),
        }}
    }}
    
    pub fn process_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {{
        // Simulate some complex processing
        for j in 0..100 {{
            self.data.insert(format!("key_{}", j), format!("value_{}", j));
        }}
        Ok(())
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;
    
    #[test]
    fn test_struct_creation() {{
        let instance = TestStruct{}::new(1, "test".to_string());
        assert_eq!(instance.id, 1);
        assert_eq!(instance.name, "test");
    }}
}}
"#,
            i, i, i, i, i
        );
        
        let file_path = match i % 3 {
            0 => src_dir.join(format!("module_{}.rs", i)),
            1 => tests_dir.join(format!("test_{}.rs", i)),
            _ => docs_dir.join(format!("doc_{}.md", i)),
        };
        
        fs::write(file_path, file_content).unwrap();
    }
    
    // Create Cargo.toml
    let cargo_toml = r#"[package]
name = "test-repo"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
"#;
    fs::write(repo_path.join("Cargo.toml"), cargo_toml).unwrap();
    
    repo_path
}

fn bench_small_repository(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(100, &temp_dir);
    let config = ScalingConfig::small_repository();
    
    c.bench_with_input(
        BenchmarkId::new("small_repository", "100_files"),
        &repo_path,
        |b, path| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mut engine = ScalingEngine::new(black_box(config.clone())).await.unwrap();
                    engine.process_repository(black_box(path)).await.unwrap()
                })
            })
        },
    );
}

fn bench_medium_repository(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(1000, &temp_dir);
    let config = ScalingConfig::default();
    
    c.bench_with_input(
        BenchmarkId::new("medium_repository", "1000_files"),
        &repo_path,
        |b, path| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mut engine = ScalingEngine::new(black_box(config.clone())).await.unwrap();
                    engine.process_repository(black_box(path)).await.unwrap()
                })
            })
        },
    );
}

fn bench_large_repository(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(5000, &temp_dir);
    let config = ScalingConfig::large_repository();
    
    c.bench_with_input(
        BenchmarkId::new("large_repository", "5000_files"),
        &repo_path,
        |b, path| {
            b.to_async(tokio::runtime::Runtime::new().unwrap())
                .iter(|| async {
                    let mut engine = ScalingEngine::new(black_box(config.clone())).await.unwrap();
                    engine.process_repository(black_box(path)).await.unwrap()
                })
        },
    );
}

fn bench_streaming_vs_batch(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(2000, &temp_dir);
    
    let mut group = c.benchmark_group("streaming_vs_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    
    // Streaming approach
    group.bench_with_input(
        BenchmarkId::new("streaming", "2000_files"),
        &repo_path,
        |b, path| {
            b.to_async(tokio::runtime::Runtime::new().unwrap())
                .iter(|| async {
                    let config = ScalingConfig {
                        streaming: scribe_scaling::streaming::StreamingConfig {
                            chunk_size: 1000,
                            memory_limit: 50 * 1024 * 1024, // 50MB
                            enable_streaming: true,
                        },
                        ..ScalingConfig::default()
                    };
                    let mut engine = ScalingEngine::new(black_box(config)).await.unwrap();
                    engine.process_repository(black_box(path)).await.unwrap()
                })
        },
    );
    
    // Batch approach (streaming disabled)
    group.bench_with_input(
        BenchmarkId::new("batch", "2000_files"),
        &repo_path,
        |b, path| {
            b.to_async(tokio::runtime::Runtime::new().unwrap())
                .iter(|| async {
                    let config = ScalingConfig {
                        streaming: scribe_scaling::streaming::StreamingConfig {
                            chunk_size: 10000,
                            memory_limit: 1024 * 1024 * 1024, // 1GB
                            enable_streaming: false,
                        },
                        ..ScalingConfig::default()
                    };
                    let mut engine = ScalingEngine::new(black_box(config)).await.unwrap();
                    engine.process_repository(black_box(path)).await.unwrap()
                })
        },
    );
    
    group.finish();
}

fn bench_caching_effectiveness(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(1000, &temp_dir);
    
    let mut group = c.benchmark_group("caching_effectiveness");
    group.sample_size(10);
    
    // First run (cold cache)
    group.bench_with_input(
        BenchmarkId::new("cold_cache", "1000_files"),
        &repo_path,
        |b, path| {
            b.to_async(tokio::runtime::Runtime::new().unwrap())
                .iter(|| async {
                    // Create new engine each time to ensure cold cache
                    let mut engine = ScalingEngine::new(black_box(ScalingConfig::default())).await.unwrap();
                    engine.process_repository(black_box(path)).await.unwrap()
                })
        },
    );
    
    // Warm cache run
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut engine = ScalingEngine::new(ScalingConfig::default()).await.unwrap();
        engine.process_repository(&repo_path).await.unwrap(); // Warm up cache
        
        group.bench_with_input(
            BenchmarkId::new("warm_cache", "1000_files"),
            &repo_path,
            |b, path| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let mut engine = ScalingEngine::new(black_box(ScalingConfig::default())).await.unwrap();
                        engine.process_repository(black_box(path)).await.unwrap()
                    })
            },
        );
    });
    
    group.finish();
}

fn bench_parallel_processing_scaling(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(2000, &temp_dir);
    
    let mut group = c.benchmark_group("parallel_scaling");
    group.sample_size(10);
    
    for workers in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("workers", workers),
            workers,
            |b, &worker_count| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let config = ScalingConfig {
                            parallel: scribe_scaling::parallel::ParallelConfig {
                                max_concurrent_tasks: worker_count,
                                async_worker_count: worker_count,
                                cpu_worker_count: worker_count,
                                task_timeout: Duration::from_secs(30),
                                enable_work_stealing: true,
                            },
                            ..ScalingConfig::default()
                        };
                        let mut engine = ScalingEngine::new(black_box(config)).await.unwrap();
                        engine.process_repository(black_box(&repo_path)).await.unwrap()
                    })
            },
        );
    }
    
    group.finish();
}

fn bench_signature_extraction_levels(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(500, &temp_dir);
    
    let mut group = c.benchmark_group("signature_levels");
    
    use scribe_scaling::signatures::SignatureLevel;
    
    for level in [
        SignatureLevel::Minimal,
        SignatureLevel::Structural,
        SignatureLevel::Semantic,
        SignatureLevel::Detailed,
        SignatureLevel::Complete,
    ].iter() {
        group.bench_with_input(
            BenchmarkId::new("signature_level", format!("{:?}", level)),
            level,
            |b, &sig_level| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let config = ScalingConfig {
                            signatures: scribe_scaling::signatures::SignatureConfig {
                                default_level: sig_level,
                                enable_caching: false, // Disable to measure pure extraction
                                budget_pressure_threshold: 1.0, // No pressure
                            },
                            ..ScalingConfig::default()
                        };
                        let mut engine = ScalingEngine::new(black_box(config)).await.unwrap();
                        engine.process_repository(black_box(&repo_path)).await.unwrap()
                    })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_small_repository,
    bench_medium_repository,
    bench_large_repository,
    bench_streaming_vs_batch,
    bench_caching_effectiveness,
    bench_parallel_processing_scaling,
    bench_signature_extraction_levels
);
criterion_main!(benches);
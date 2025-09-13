//! Main benchmarks for the Scribe library
//!
//! These benchmarks test the performance of the complete Scribe library
//! with all features enabled.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scribe::prelude::*;
use std::path::Path;
use tempfile::TempDir;
use std::fs;
use std::collections::HashMap;

/// Create a test repository with various file types
fn create_test_repo(size: usize) -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create directory structure
    fs::create_dir_all(base_path.join("src")).unwrap();
    fs::create_dir_all(base_path.join("tests")).unwrap();
    fs::create_dir_all(base_path.join("docs")).unwrap();
    fs::create_dir_all(base_path.join("examples")).unwrap();
    fs::create_dir_all(base_path.join("target/debug")).unwrap(); // Should be excluded
    
    // Create files of different types and sizes
    for i in 0..size {
        // Rust source files
        let rust_content = format!(
            "// File {}\n\
             use std::collections::HashMap;\n\
             \n\
             pub fn function_{}() -> Result<HashMap<String, i32>, Box<dyn std::error::Error>> {{\n\
             \tlet mut map = HashMap::new();\n\
             \tmap.insert(\"key_{}\".to_string(), {});\n\
             \tOk(map)\n\
             }}\n\
             \n\
             #[cfg(test)]\n\
             mod tests {{\n\
             \tuse super::*;\n\
             \t\n\
             \t#[test]\n\
             \tfn test_function_{}() {{\n\
             \t\tlet result = function_{}().unwrap();\n\
             \t\tassert!(result.contains_key(\"key_{}\"));\n\
             \t}}\n\
             }}",
            i, i, i, i, i, i, i
        );
        fs::write(base_path.join(format!("src/module_{}.rs", i)), rust_content).unwrap();
        
        // Test files
        if i % 3 == 0 {
            let test_content = format!(
                "use super::*;\n\
                 \n\
                 #[test]\n\
                 fn integration_test_{}() {{\n\
                 \tlet result = some_function();\n\
                 \tassert_eq!(result, {});\n\
                 }}",
                i, i
            );
            fs::write(base_path.join(format!("tests/integration_{}.rs", i)), test_content).unwrap();
        }
        
        // Documentation files
        if i % 5 == 0 {
            let doc_content = format!(
                "# Module {}\n\
                 \n\
                 This module provides functionality for handling data structures.\n\
                 \n\
                 ## Usage\n\
                 \n\
                 ```rust\n\
                 use crate::module_{}::function_{};\n\
                 let result = function_{}().unwrap();\n\
                 ```\n\
                 \n\
                 ## Examples\n\
                 \n\
                 See the examples directory for more detailed usage.",
                i, i, i, i
            );
            fs::write(base_path.join(format!("docs/module_{}.md", i)), doc_content).unwrap();
        }
    }
    
    // Create main files
    fs::write(base_path.join("src/lib.rs"), "pub mod common; // Main library file").unwrap();
    fs::write(base_path.join("src/main.rs"), "fn main() { println!(\"Hello, world!\"); }").unwrap();
    fs::write(base_path.join("README.md"), "# Test Repository\n\nThis is a test repository for benchmarking.").unwrap();
    fs::write(base_path.join("Cargo.toml"), "[package]\nname = \"test\"\nversion = \"0.1.0\"").unwrap();
    
    // Create some build artifacts (should be filtered out)
    fs::write(base_path.join("target/debug/test"), "binary file").unwrap();
    
    temp_dir
}

/// Benchmark full repository analysis
fn bench_full_analysis(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("full_analysis");
    group.throughput(Throughput::Elements(1));
    
    for &size in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("analyze_repository", size),
            &size,
            |b, &size| {
                let temp_repo = create_test_repo(size);
                let config = Config::default();
                
                b.to_async(&rt).iter(|| async {
                    let analysis = analyze_repository(black_box(temp_repo.path()), black_box(&config)).await;
                    black_box(analysis)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark file scanning only
#[cfg(feature = "scanner")]
fn bench_scanning(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("scanning");
    group.throughput(Throughput::Elements(1));
    
    for &size in &[10, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("scan_repository", size),
            &size,
            |b, &size| {
                let temp_repo = create_test_repo(size);
                
                b.to_async(&rt).iter(|| async {
                    let result = scan_repository(
                        black_box(temp_repo.path()),
                        black_box(Some(&["**/*.rs", "**/*.md"])),
                        black_box(Some(&["**/target/**"]))
                    ).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark pattern matching
#[cfg(feature = "patterns")]
fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");
    
    // Create test paths
    let test_paths = vec![
        "src/lib.rs",
        "src/main.rs", 
        "tests/integration.rs",
        "docs/README.md",
        "target/debug/main",
        "node_modules/package/index.js",
        "examples/basic.rs",
        ".git/config",
        "Cargo.toml",
        "assets/image.png",
    ];
    
    group.bench_function("source_code_preset", |b| {
        let mut matcher = scribe::patterns::presets::source_code().unwrap();
        b.iter(|| {
            for path in &test_paths {
                black_box(matcher.should_process(black_box(path)).unwrap());
            }
        });
    });
    
    group.bench_function("quick_matcher", |b| {
        let mut matcher = scribe::patterns::QuickMatcher::new(
            &["**/*.rs", "**/*.py", "**/*.js"],
            &["**/target/**", "**/node_modules/**"]
        ).unwrap();
        
        b.iter(|| {
            for path in &test_paths {
                black_box(matcher.matches(black_box(path)).unwrap());
            }
        });
    });
    
    group.finish();
}

/// Benchmark heuristic scoring
#[cfg(feature = "analysis")]
fn bench_heuristic_scoring(c: &mut Criterion) {
    let config = Config::default();
    let heuristic_system = scribe::analysis::HeuristicSystem::new(config).unwrap();
    
    // Create test file infos
    let test_files = create_test_file_infos();
    
    let mut group = c.benchmark_group("heuristic_scoring");
    group.throughput(Throughput::Elements(test_files.len() as u64));
    
    group.bench_function("score_files", |b| {
        b.iter(|| {
            for file in &test_files {
                black_box(heuristic_system.score_file(black_box(file)).unwrap());
            }
        });
    });
    
    group.finish();
}

/// Create test FileInfo objects for benchmarking
fn create_test_file_infos() -> Vec<FileInfo> {
    use scribe::core::{Language, FileType, GitStatus};
    use std::path::PathBuf;
    
    vec![
        FileInfo {
            path: PathBuf::from("src/lib.rs"),
            relative_path: PathBuf::from("src/lib.rs"),
            language: Language::Rust,
            file_type: FileType::Source,
            size_bytes: Some(2048),
            line_count: Some(80),
            git_status: Some(GitStatus::Tracked),
            last_modified: Some(std::time::SystemTime::now()),
            is_binary: false,
            encoding: Some("utf-8".to_string()),
        },
        FileInfo {
            path: PathBuf::from("README.md"),
            relative_path: PathBuf::from("README.md"),
            language: Language::Markdown,
            file_type: FileType::Documentation,
            size_bytes: Some(1024),
            line_count: Some(40),
            git_status: Some(GitStatus::Tracked),
            last_modified: Some(std::time::SystemTime::now()),
            is_binary: false,
            encoding: Some("utf-8".to_string()),
        },
        FileInfo {
            path: PathBuf::from("tests/integration.rs"),
            relative_path: PathBuf::from("tests/integration.rs"),
            language: Language::Rust,
            file_type: FileType::Test,
            size_bytes: Some(1536),
            line_count: Some(60),
            git_status: Some(GitStatus::Tracked),
            last_modified: Some(std::time::SystemTime::now()),
            is_binary: false,
            encoding: Some("utf-8".to_string()),
        },
        FileInfo {
            path: PathBuf::from("Cargo.toml"),
            relative_path: PathBuf::from("Cargo.toml"),
            language: Language::Toml,
            file_type: FileType::Configuration,
            size_bytes: Some(512),
            line_count: Some(20),
            git_status: Some(GitStatus::Tracked),
            last_modified: Some(std::time::SystemTime::now()),
            is_binary: false,
            encoding: Some("utf-8".to_string()),
        },
    ]
}

/// Benchmark PageRank centrality computation
#[cfg(feature = "graph")]
fn bench_pagerank(c: &mut Criterion) {
    use scribe::graph::PageRankAnalysis;
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("pagerank");
    
    // This would need proper ScanResult implementations for benchmarking
    // For now, we'll create a simple benchmark structure
    
    group.bench_function("pagerank_analysis_creation", |b| {
        b.iter(|| {
            black_box(PageRankAnalysis::for_code_analysis().unwrap())
        });
    });
    
    group.finish();
}

// Benchmark core utilities
fn bench_core_utilities(c: &mut Criterion) {
    use scribe::core::utils::*;
    
    let mut group = c.benchmark_group("core_utilities");
    
    group.bench_function("path_normalization", |b| {
        let test_paths = vec![
            "src/lib.rs",
            "./src/../src/main.rs",
            "src//subdir//file.rs",
            "../parent/src/lib.rs",
        ];
        
        b.iter(|| {
            for path in &test_paths {
                black_box(normalize_path(black_box(path)));
            }
        });
    });
    
    group.bench_function("string_truncation", |b| {
        let test_strings = vec![
            "Short string",
            "This is a longer string that will definitely need truncation when processed",
            "Another example of a very long string with multiple words and various characters",
        ];
        
        b.iter(|| {
            for s in &test_strings {
                black_box(truncate(black_box(s), black_box(30)));
            }
        });
    });
    
    group.bench_function("hash_generation", |b| {
        let test_data = vec![
            "small",
            "medium length string with some content",
            &"large ".repeat(100),
        ];
        
        b.iter(|| {
            for data in &test_data {
                black_box(generate_hash(black_box(data)));
            }
        });
    });
    
    group.finish();
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_full_analysis,
    bench_core_utilities,
);

// Conditionally add feature-specific benchmarks
#[cfg(feature = "scanner")]
criterion_group!(
    scanner_benches,
    bench_scanning,
);

#[cfg(feature = "patterns")]
criterion_group!(
    pattern_benches,
    bench_pattern_matching,
);

#[cfg(feature = "analysis")]
criterion_group!(
    analysis_benches,
    bench_heuristic_scoring,
);

#[cfg(feature = "graph")]
criterion_group!(
    graph_benches,
    bench_pagerank,
);

// Create main function with all available benchmarks
criterion_main!(
    benches,
    #[cfg(feature = "scanner")]
    scanner_benches,
    #[cfg(feature = "patterns")]
    pattern_benches,
    #[cfg(feature = "analysis")]
    analysis_benches,
    #[cfg(feature = "graph")]
    graph_benches,
);
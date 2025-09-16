//! # Heuristics Performance Benchmarks
//!
//! Comprehensive benchmarks for the heuristic scoring system to validate
//! performance against requirements and compare with Python implementation.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use scribe_analysis::heuristics::*;
use std::time::Duration;

/// Mock scan result for benchmarking
#[derive(Debug, Clone)]
struct BenchScanResult {
    path: String,
    relative_path: String,
    depth: usize,
    is_docs: bool,
    is_readme: bool,
    is_test: bool,
    is_entrypoint: bool,
    has_examples: bool,
    priority_boost: f64,
    churn_score: f64,
    centrality_in: f64,
    imports: Option<Vec<String>>,
    doc_analysis: Option<DocumentAnalysis>,
}

impl BenchScanResult {
    fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            relative_path: path.to_string(),
            depth: path.matches('/').count(),
            is_docs: path.contains("doc") || path.ends_with(".md"),
            is_readme: path.to_lowercase().contains("readme"),
            is_test: path.contains("test") || path.contains("spec"),
            is_entrypoint: path.contains("main") || path.contains("index"),
            has_examples: path.contains("example") || path.contains("demo"),
            priority_boost: if path.contains("README") { 0.2 } else { 0.0 },
            churn_score: 0.5,
            centrality_in: 0.3,
            imports: Some(vec![
                "std::collections::HashMap".to_string(),
                "serde::Serialize".to_string(),
                "tokio::io".to_string(),
            ]),
            doc_analysis: Some(DocumentAnalysis::new()),
        }
    }

    fn create_realistic_dataset(size: usize) -> Vec<Self> {
        let mut dataset = Vec::with_capacity(size);

        // Create a realistic mix of files
        for i in 0..size {
            let file_type = i % 10;
            let path = match file_type {
                0 => format!("README.md"),
                1 => format!("src/main.rs"),
                2 => format!("src/lib/mod{}.rs", i),
                3 => format!("src/utils/helper{}.rs", i),
                4 => format!("tests/test_{}.rs", i),
                5 => format!("examples/example_{}.rs", i),
                6 => format!("docs/guide_{}.md", i),
                7 => format!("src/components/component_{}.rs", i),
                8 => format!("src/deep/nested/path/file_{}.rs", i),
                _ => format!("src/misc/file_{}.rs", i),
            };

            dataset.push(BenchScanResult::new(&path));
        }

        dataset
    }
}

impl ScanResult for BenchScanResult {
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
    fn doc_analysis(&self) -> Option<&DocumentAnalysis> {
        self.doc_analysis.as_ref()
    }
}

/// Benchmark individual file scoring performance
fn bench_single_file_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_file_scoring");
    group.measurement_time(Duration::from_secs(10));

    let files = BenchScanResult::create_realistic_dataset(1000);
    let test_file = &files[0];

    // V1 scoring
    group.bench_function("v1_scoring", |b| {
        let mut scorer = HeuristicScorer::new(HeuristicWeights::default());
        b.iter(|| black_box(scorer.score_file(test_file, &files).unwrap()));
    });

    // V2 scoring (with centrality)
    group.bench_function("v2_scoring", |b| {
        let mut scorer = HeuristicScorer::new(HeuristicWeights::with_v2_features());
        b.iter(|| black_box(scorer.score_file(test_file, &files).unwrap()));
    });

    group.finish();
}

/// Benchmark batch scoring performance with different dataset sizes
fn bench_batch_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scoring");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for size in [100, 500, 1000, 2000, 5000].iter() {
        let files = BenchScanResult::create_realistic_dataset(*size);

        group.bench_with_input(BenchmarkId::new("v1_batch", size), size, |b, _| {
            b.iter_batched(
                || HeuristicScorer::new(HeuristicWeights::default()),
                |mut scorer| black_box(scorer.score_all_files(&files).unwrap()),
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("v2_batch", size), size, |b, _| {
            b.iter_batched(
                || HeuristicScorer::new(HeuristicWeights::with_v2_features()),
                |mut scorer| black_box(scorer.score_all_files(&files).unwrap()),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark template detection performance
fn bench_template_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_detection");

    let test_files = vec![
        "template.hbs",
        "component.vue",
        "layout.html",
        "script.js",
        "stylesheet.css",
        "document.md",
        "config.json",
        "src/main.rs",
    ];

    group.bench_function("template_detection", |b| {
        let detector = TemplateDetector::new();
        b.iter(|| {
            for file_path in &test_files {
                black_box(detector.get_score_boost(file_path).unwrap());
            }
        });
    });

    group.bench_function("is_template_check", |b| {
        b.iter(|| {
            for file_path in &test_files {
                black_box(is_template_file(file_path).unwrap());
            }
        });
    });

    group.finish();
}

/// Benchmark import graph construction and analysis
fn bench_import_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("import_analysis");
    group.measurement_time(Duration::from_secs(15));

    for size in [100, 500, 1000].iter() {
        let files = BenchScanResult::create_realistic_dataset(*size);

        group.bench_with_input(
            BenchmarkId::new("graph_construction", size),
            size,
            |b, _| {
                b.iter_batched(
                    || ImportGraphBuilder::new(),
                    |mut builder| black_box(builder.build_graph(&files).unwrap()),
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pagerank_calculation", size),
            size,
            |b, _| {
                let mut builder = ImportGraphBuilder::new();
                let graph = builder.build_graph(&files).unwrap();
                b.iter(|| {
                    let mut graph_copy = graph.clone();
                    black_box(graph_copy.get_pagerank_scores().unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complete heuristic system end-to-end
fn bench_heuristic_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("heuristic_system");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(5);

    for size in [500, 1000, 2000].iter() {
        let files = BenchScanResult::create_realistic_dataset(*size);

        group.bench_with_input(BenchmarkId::new("end_to_end_v1", size), size, |b, _| {
            b.iter_batched(
                || HeuristicSystem::new().unwrap(),
                |mut system| black_box(system.score_all_files(&files).unwrap()),
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("end_to_end_v2", size), size, |b, _| {
            b.iter_batched(
                || HeuristicSystem::with_v2_features().unwrap(),
                |mut system| black_box(system.score_all_files(&files).unwrap()),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark memory efficiency and allocation patterns
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    let files = BenchScanResult::create_realistic_dataset(1000);

    group.bench_function("scorer_creation", |b| {
        b.iter(|| {
            black_box(HeuristicScorer::new(HeuristicWeights::default()));
        });
    });

    group.bench_function("import_graph_creation", |b| {
        b.iter(|| {
            black_box(ImportGraph::new());
        });
    });

    group.bench_function("template_detector_creation", |b| {
        b.iter(|| {
            black_box(TemplateDetector::new());
        });
    });

    group.bench_function("document_analysis", |b| {
        b.iter(|| {
            let mut doc = DocumentAnalysis::new();
            doc.heading_count = 5;
            doc.link_count = 10;
            doc.code_block_count = 3;
            black_box(doc.structure_score());
        });
    });

    group.finish();
}

/// Benchmark performance targets validation
fn bench_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets");
    group.measurement_time(Duration::from_secs(10));

    // Target: Process 1000 files in under 100ms (10 files/ms)
    let files = BenchScanResult::create_realistic_dataset(1000);
    let mut system = HeuristicSystem::new().unwrap();

    group.bench_function("target_1000_files_100ms", |b| {
        b.iter(|| {
            black_box(system.score_all_files(&files).unwrap());
        });
    });

    // Target: Single file scoring under 1ms
    let test_file = &files[0];
    let mut scorer = HeuristicScorer::new(HeuristicWeights::default());

    group.bench_function("target_single_file_1ms", |b| {
        b.iter(|| {
            black_box(scorer.score_file(test_file, &files).unwrap());
        });
    });

    group.finish();
}

/// Benchmark import matching heuristics
fn bench_import_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("import_matching");

    let test_cases = vec![
        ("src/utils", "src/utils.rs"),
        ("./lib", "src/lib.js"),
        ("../components/Button", "src/components/Button.tsx"),
        ("@/helpers/api", "src/helpers/api.ts"),
        ("std::collections::HashMap", "std/collections/hash_map.rs"),
        ("completely_different", "src/utils.rs"),
    ];

    group.bench_function("import_file_matching", |b| {
        b.iter(|| {
            for (import, file) in &test_cases {
                black_box(import_matches_file(import, file));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_file_scoring,
    bench_batch_scoring,
    bench_template_detection,
    bench_import_analysis,
    bench_heuristic_system,
    bench_memory_efficiency,
    bench_performance_targets,
    bench_import_matching
);
criterion_main!(benches);

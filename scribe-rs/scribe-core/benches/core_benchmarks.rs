use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scribe_core::*;
use std::path::PathBuf;

fn benchmark_language_detection(c: &mut Criterion) {
    let extensions = vec!["rs", "py", "js", "ts", "go", "java", "cpp"];

    c.bench_function("language_detection", |b| {
        b.iter(|| {
            for ext in &extensions {
                black_box(Language::from_extension(ext));
            }
        })
    });
}

fn benchmark_path_normalization(c: &mut Criterion) {
    let paths = vec![
        "src/lib.rs",
        "src\\windows\\path.rs",
        "deeply/nested/directory/structure/file.rs",
        "C:\\Windows\\System32\\file.exe",
    ];

    c.bench_function("path_normalization", |b| {
        b.iter(|| {
            for path in &paths {
                black_box(normalize_path(path));
            }
        })
    });
}

fn benchmark_string_truncation(c: &mut Criterion) {
    let strings = vec![
        "short",
        "this is a medium length string that might need truncation",
        "this is a very long string that definitely needs to be truncated because it exceeds the maximum allowed length for display purposes in most contexts",
    ];

    c.bench_function("string_truncation", |b| {
        b.iter(|| {
            for s in &strings {
                black_box(truncate(s, 50));
            }
        })
    });
}

fn benchmark_score_computation(c: &mut Criterion) {
    let weights = HeuristicWeights::default();
    let mut scores = ScoreComponents::zero();
    scores.doc_score = 0.8;
    scores.readme_score = 0.6;
    scores.import_score = 0.4;
    scores.path_score = 0.9;
    scores.test_link_score = 0.2;
    scores.churn_score = 0.7;

    c.bench_function("score_computation", |b| {
        b.iter(|| {
            let mut s = scores.clone();
            s.compute_final_score(black_box(&weights));
            black_box(s.final_score);
        })
    });
}

fn benchmark_config_validation(c: &mut Criterion) {
    let config = Config::default();

    c.bench_function("config_validation", |b| {
        b.iter(|| {
            black_box(config.validate().unwrap());
        })
    });
}

fn benchmark_binary_detection(c: &mut Criterion) {
    let test_strings = vec![
        "Hello, world! This is normal text.",
        "fn main() {\n    println!(\"Hello, world!\");\n}",
        "Some text with null\x00bytes in it",
        "Mostly\x01non\x02printable\x03characters\x04here\x05",
    ];

    c.bench_function("binary_detection", |b| {
        b.iter(|| {
            for s in &test_strings {
                black_box(is_likely_binary(s));
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_language_detection,
    benchmark_path_normalization,
    benchmark_string_truncation,
    benchmark_score_computation,
    benchmark_config_validation,
    benchmark_binary_detection
);
criterion_main!(benches);

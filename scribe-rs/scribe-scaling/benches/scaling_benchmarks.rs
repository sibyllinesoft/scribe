use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scribe_scaling::engine::{ScalingConfig, ScalingEngine};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn create_test_repository(file_count: usize, temp_dir: &TempDir) -> PathBuf {
    let repo_path = temp_dir.path().to_path_buf();
    fs::create_dir_all(repo_path.join("src")).unwrap();

    for idx in 0..file_count {
        let content =
            format!("// bench stub file {idx}\nfn main() {{ println!(\"hello {idx}\"); }}\n");
        let target = repo_path.join("src").join(format!("file_{idx}.rs"));
        fs::write(target, content).unwrap();
    }

    fs::write(
        repo_path.join("Cargo.toml"),
        "[package]\nname = \"bench-repo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();

    repo_path
}

fn run_engine_once(repo_path: &Path, config: ScalingConfig) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut engine = ScalingEngine::new(config).await.unwrap();
        engine.process_repository(repo_path).await.unwrap();
    });
}

fn bench_repository_profiles(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(250, &temp_dir);

    let mut group = c.benchmark_group("scaling_profiles");

    let scenarios = [
        ("small", ScalingConfig::small_repository()),
        ("default", ScalingConfig::default()),
        ("large", ScalingConfig::large_repository()),
    ];

    for (label, config) in scenarios {
        group.bench_with_input(
            BenchmarkId::new("process_repository", label),
            &repo_path,
            |b, path| b.iter(|| run_engine_once(path, config.clone())),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_repository_profiles);
criterion_main!(benches);

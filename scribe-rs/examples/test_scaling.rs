// This example demonstrates the scaling functionality that is available via the --scaling CLI flag
// To test this programmatically: cargo run --example test_scaling
// To test via CLI with scaling: ./target/release/scribe --scaling .

use scribe_scaling::{ScalingConfig, ScalingEngine};
use std::path::Path;
use std::time::Instant;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Scribe Scaling Performance on Repository");
    println!("==================================================");

    let repo_path = ".."; // Test on the scribe repository

    // Test different configurations
    let configs = vec![
        ("Default Config", ScalingConfig::default()),
        ("Small Repository", ScalingConfig::small_repository()),
        ("Large Repository", ScalingConfig::large_repository()),
    ];

    for (name, config) in configs {
        println!("\n📊 Testing with {}", name);
        println!("{}", "-".repeat(50));

        let start_time = Instant::now();

        // Create scaling engine
        let mut engine = ScalingEngine::new(config).await?;

        // Process repository
        println!("⚡ Processing repository...");
        let process_start = Instant::now();
        let result = engine.process_repository(Path::new(repo_path)).await?;
        let process_duration = process_start.elapsed();

        println!("   Files Processed: {}", result.metrics.files_processed);
        println!("   Processing Time: {:?}", process_duration);
        println!(
            "   Memory Peak: {:.2} MB",
            result.memory_peak as f64 / 1024.0 / 1024.0
        );
        println!("   Cache Hits: {}", result.cache_hits);
        println!("   Cache Misses: {}", result.cache_misses);

        let total_duration = start_time.elapsed();
        println!("   Total Time: {:?}", total_duration);

        // Performance check against targets
        let memory_mb = result.memory_peak as f64 / 1024.0 / 1024.0;
        let time_secs = total_duration.as_secs_f64();

        println!("📈 Performance Analysis:");
        if result.metrics.files_processed <= 1000 {
            // Small repo targets: <1s, <50MB
            let time_ok = time_secs < 1.0;
            let memory_ok = memory_mb < 50.0;
            println!(
                "   Target: <1s, <50MB | Actual: {:.2}s, {:.2}MB | Status: {} {}",
                time_secs,
                memory_mb,
                if time_ok { "⏱️ ✅" } else { "⏱️ ❌" },
                if memory_ok { "💾 ✅" } else { "💾 ❌" }
            );
        } else if result.metrics.files_processed <= 10000 {
            // Medium repo targets: <5s, <200MB
            let time_ok = time_secs < 5.0;
            let memory_ok = memory_mb < 200.0;
            println!(
                "   Target: <5s, <200MB | Actual: {:.2}s, {:.2}MB | Status: {} {}",
                time_secs,
                memory_mb,
                if time_ok { "⏱️ ✅" } else { "⏱️ ❌" },
                if memory_ok { "💾 ✅" } else { "💾 ❌" }
            );
        }
    }

    println!("\n🎯 Running Comprehensive Benchmark Suite");
    println!("=========================================");

    // Run full benchmark suite
    let mut engine = ScalingEngine::new(ScalingConfig::default()).await?;
    let benchmark_start = Instant::now();
    let benchmarks = engine.benchmark(Path::new(repo_path), 3).await?;
    let benchmark_duration = benchmark_start.elapsed();

    println!("Benchmark runs: {}", benchmarks.len());
    println!("Benchmark time: {:?}", benchmark_duration);

    if !benchmarks.is_empty() {
        let avg_time: f64 = benchmarks
            .iter()
            .map(|b| b.duration.as_secs_f64())
            .sum::<f64>()
            / benchmarks.len() as f64;
        let avg_memory: f64 = benchmarks
            .iter()
            .map(|b| b.memory_usage as f64)
            .sum::<f64>()
            / benchmarks.len() as f64
            / 1024.0
            / 1024.0;

        println!("📊 Benchmark Results (avg of {} runs):", benchmarks.len());
        println!("   Average Time: {:.3}s", avg_time);
        println!("   Average Memory: {:.2}MB", avg_memory);
        println!(
            "   Consistency: {:.2}% (std dev)",
            benchmarks
                .iter()
                .map(|b| b.duration.as_secs_f64())
                .fold(0.0, |acc, x| acc + (x - avg_time).powi(2))
                / benchmarks.len() as f64
        );
    }

    println!("\n✅ Scaling Performance Test Complete!");
    println!("\n💡 CLI Usage:");
    println!("To use scaling optimizations in the Scribe CLI:");
    println!("  ./target/release/scribe --scaling [repository_path]");
    println!("  ./target/release/scribe --scaling --verbose . # For detailed output");
    println!("\nScaling is now always compiled in but only activated with the --scaling flag!");

    Ok(())
}

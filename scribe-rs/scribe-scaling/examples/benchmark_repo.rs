use scribe_scaling::{ScalingConfig, ScalingEngine};
use std::path::PathBuf;
use std::time::Instant;

/// Get performance rating based on time and memory
fn get_performance_rating(time_secs: f64, memory_mb: f64) -> &'static str {
    if time_secs < 1.0 && memory_mb < 50.0 {
        "🏆 Excellent"
    } else if time_secs < 5.0 && memory_mb < 200.0 {
        "✅ Good"
    } else if time_secs < 15.0 && memory_mb < 1000.0 {
        "⚠️ Fair"
    } else {
        "❌ Needs Improvement"
    }
}

/// Get overall assessment based on metrics
fn get_overall_assessment(avg_time: f64, avg_memory: f64, consistency: f64) -> &'static str {
    if avg_time < 1.0 && avg_memory < 50.0 && consistency > 90.0 {
        "   🌟 EXCELLENT - Ready for production use"
    } else if avg_time < 5.0 && avg_memory < 200.0 && consistency > 80.0 {
        "   ✅ GOOD - Suitable for medium repositories"
    } else if avg_time < 15.0 && avg_memory < 1000.0 && consistency > 70.0 {
        "   ⚠️ FAIR - May need optimization for large repositories"
    } else {
        "   ❌ POOR - Requires significant optimization"
    }
}

/// Print performance check based on repository size
fn print_performance_check(total_files: usize, time_secs: f64, memory_mb: f64) {
    let (time_target, memory_target) = if total_files <= 1000 {
        (1.0, 50.0)
    } else if total_files <= 10000 {
        (5.0, 200.0)
    } else {
        return; // No specific target for very large repos
    };

    let time_ok = time_secs < time_target;
    let memory_ok = memory_mb < memory_target;

    println!(
        "   Target: <{}s, <{}MB | Actual: {:.2}s, {:.2}MB | Status: {} {}",
        time_target as u32,
        memory_target as u32,
        time_secs,
        memory_mb,
        if time_ok { "⏱️ ✅" } else { "⏱️ ❌" },
        if memory_ok { "💾 ✅" } else { "💾 ❌" }
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Scribe Scaling Performance on Repository");
    println!("==================================================");

    // Use the parent directory as test repository (scribe Python codebase)
    let repo_path = PathBuf::from("../../");

    if !repo_path.exists() {
        eprintln!("❌ Repository path not found: {}", repo_path.display());
        return Ok(());
    }

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
        let result = engine.process_repository(&repo_path).await?;
        let process_duration = process_start.elapsed();

        println!("   Files Processed: {}", result.total_files);
        println!("   Processing Time: {:?}", process_duration);
        println!(
            "   Memory Used: {:.2} MB",
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
        print_performance_check(result.total_files, time_secs, memory_mb);
        println!(
            "   Performance Rating: {}",
            get_performance_rating(time_secs, memory_mb)
        );
    }

    println!("\n🎯 Running Comprehensive Benchmark Suite");
    println!("=========================================");

    // Run full benchmark suite
    let mut engine = ScalingEngine::new(ScalingConfig::default()).await?;
    let benchmark_start = Instant::now();
    let benchmarks = engine.benchmark(&repo_path, 3).await?;
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

        let times: Vec<f64> = benchmarks
            .iter()
            .map(|b| b.duration.as_secs_f64())
            .collect();
        let variance: f64 =
            times.iter().map(|x| (x - avg_time).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        let consistency = if avg_time > 0.0 {
            100.0 * (1.0 - (std_dev / avg_time))
        } else {
            100.0
        };

        println!("📊 Benchmark Results (avg of {} runs):", benchmarks.len());
        println!("   Average Time: {:.3}s", avg_time);
        println!("   Average Memory: {:.2}MB", avg_memory);
        println!("   Consistency: {:.1}% (higher is better)", consistency);
        println!("   Standard Deviation: {:.3}s", std_dev);

        // Overall assessment
        println!("\n🏆 Overall Performance Assessment:");
        println!(
            "{}",
            get_overall_assessment(avg_time, avg_memory, consistency)
        );
    }

    println!("\n✅ Scaling Performance Test Complete!");
    Ok(())
}

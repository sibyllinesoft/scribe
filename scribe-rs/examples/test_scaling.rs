use std::time::Instant;
use scribe_core::scaling::{ScalingEngine, ScalingConfig};
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
        println!("-".repeat(50));
        
        let start_time = Instant::now();
        
        // Create scaling engine
        let mut engine = ScalingEngine::new(config).await?;
        
        // Profile the repository first
        println!("🔍 Profiling repository...");
        let profile_start = Instant::now();
        let profile = engine.profile_repository(repo_path).await?;
        let profile_duration = profile_start.elapsed();
        
        println!("   Repository Type: {:?}", profile.repo_type);
        println!("   Size Category: {:?}", profile.size_category);
        println!("   File Count Estimate: {}", profile.estimated_file_count);
        println!("   Profiling Time: {:?}", profile_duration);
        
        // Process repository
        println!("⚡ Processing repository...");
        let process_start = Instant::now();
        let result = engine.process_repository(repo_path).await?;
        let process_duration = process_start.elapsed();
        
        println!("   Files Processed: {}", result.files_processed);
        println!("   Processing Time: {:?}", process_duration);
        
        // Get metrics
        let metrics = engine.get_metrics();
        println!("   Memory Used: {:.2} MB", metrics.memory_used_bytes as f64 / 1024.0 / 1024.0);
        println!("   Cache Hits: {}", metrics.cache_hits);
        println!("   Cache Misses: {}", metrics.cache_misses);
        
        let total_duration = start_time.elapsed();
        println!("   Total Time: {:?}", total_duration);
        
        // Performance check against targets
        let memory_mb = metrics.memory_used_bytes as f64 / 1024.0 / 1024.0;
        let time_secs = total_duration.as_secs_f64();
        
        println!("📈 Performance Analysis:");
        if result.files_processed <= 1000 {
            // Small repo targets: <1s, <50MB
            let time_ok = time_secs < 1.0;
            let memory_ok = memory_mb < 50.0;
            println!("   Target: <1s, <50MB | Actual: {:.2}s, {:.2}MB | Status: {} {}", 
                time_secs, memory_mb,
                if time_ok { "⏱️ ✅" } else { "⏱️ ❌" },
                if memory_ok { "💾 ✅" } else { "💾 ❌" }
            );
        } else if result.files_processed <= 10000 {
            // Medium repo targets: <5s, <200MB
            let time_ok = time_secs < 5.0;
            let memory_ok = memory_mb < 200.0;
            println!("   Target: <5s, <200MB | Actual: {:.2}s, {:.2}MB | Status: {} {}", 
                time_secs, memory_mb,
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
    let benchmarks = engine.benchmark(repo_path, 3).await?;
    let benchmark_duration = benchmark_start.elapsed();
    
    println!("Benchmark runs: {}", benchmarks.len());
    println!("Benchmark time: {:?}", benchmark_duration);
    
    if !benchmarks.is_empty() {
        let avg_time: f64 = benchmarks.iter().map(|b| b.duration.as_secs_f64()).sum::<f64>() / benchmarks.len() as f64;
        let avg_memory: f64 = benchmarks.iter().map(|b| b.memory_used as f64).sum::<f64>() / benchmarks.len() as f64 / 1024.0 / 1024.0;
        
        println!("📊 Benchmark Results (avg of {} runs):", benchmarks.len());
        println!("   Average Time: {:.3}s", avg_time);
        println!("   Average Memory: {:.2}MB", avg_memory);
        println!("   Consistency: {:.2}% (std dev)", 
            benchmarks.iter().map(|b| b.duration.as_secs_f64()).fold(0.0, |acc, x| acc + (x - avg_time).powi(2)) / benchmarks.len() as f64
        );
    }
    
    println!("\n✅ Scaling Performance Test Complete!");
    Ok(())
}
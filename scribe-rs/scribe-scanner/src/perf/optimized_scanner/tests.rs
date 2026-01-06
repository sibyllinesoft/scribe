//! Tests for optimized scanner module.

use super::*;
use tempfile::TempDir;
use tokio::fs;

async fn create_test_repo() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    // Create test files
    fs::write(root.join("main.rs"), "fn main() { println!(\"Hello\"); }").await.unwrap();
    fs::write(root.join("lib.rs"), "pub fn hello() -> String { \"world\".to_string() }").await.unwrap();

    // Create subdirectories
    fs::create_dir(root.join("src")).await.unwrap();
    fs::write(root.join("src/module.rs"), "pub mod submodule;").await.unwrap();
    fs::write(root.join("src/utils.rs"), "pub fn utility_function() {}").await.unwrap();

    fs::create_dir(root.join("tests")).await.unwrap();
    fs::write(root.join("tests/integration.rs"), "#[test] fn test_something() {}").await.unwrap();

    // Create files that should be filtered
    fs::write(root.join("image.png"), &[0u8; 1024]).await.unwrap(); // Binary file
    fs::write(root.join("large.txt"), &vec![b'x'; 100_000]).await.unwrap(); // Large file

    // Create node_modules (should be filtered)
    fs::create_dir(root.join("node_modules")).await.unwrap();
    fs::write(root.join("node_modules/package.js"), "module.exports = {};").await.unwrap();

    temp_dir
}

#[tokio::test]
async fn test_optimized_scanner_creation() {
    let temp_dir = create_test_repo().await;
    let config = OptimizedScanConfig::default();

    let scanner = OptimizedScanner::new(temp_dir.path(), config).await;
    assert!(scanner.is_ok());
}

#[tokio::test]
async fn test_optimized_scan() {
    let temp_dir = create_test_repo().await;
    let config = OptimizedScanConfig {
        enable_git: false, // Disable git for test simplicity
        enable_incremental: false, // Disable incremental for test
        enable_monitoring: true,
        ..Default::default()
    };

    let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();
    let result = scanner.scan().await.unwrap();

    // Should find Rust files but filter out binary and large files
    assert!(result.stats.files_processed >= 4); // At least main.rs, lib.rs, module.rs, utils.rs, integration.rs
    assert!(result.stats.files_filtered > 0); // Should filter out image.png, large.txt, etc.
    assert!(result.performance.files_per_second > 0.0);
    assert!(result.performance.filter_effectiveness > 0.0);
    assert!(result.duration.as_secs_f64() > 0.0);

    println!("Scan completed:");
    println!("- Files discovered: {}", result.stats.files_discovered);
    println!("- Files processed: {}", result.stats.files_processed);
    println!("- Files filtered: {}", result.stats.files_filtered);
    println!("- Throughput: {:.1} files/sec", result.performance.files_per_second);
    println!("- Filter effectiveness: {:.1}%", result.performance.filter_effectiveness * 100.0);
}

#[tokio::test]
async fn test_performance_monitoring() {
    let temp_dir = create_test_repo().await;
    let config = OptimizedScanConfig {
        enable_monitoring: true,
        enable_git: false,
        enable_incremental: false,
        ..Default::default()
    };

    let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();

    // Reset metrics before scan
    scanner.reset_metrics();

    let result = scanner.scan().await.unwrap();

    // Verify metrics were collected
    let monitor_snapshot = PERF_MONITOR.get_current_snapshot();
    assert!(monitor_snapshot.files_per_second >= 0.0);

    // Verify result contains performance data
    assert!(result.performance.files_per_second > 0.0);
    assert!(result.performance.time_breakdown.processing_time_ms > 0.0);
}

#[tokio::test]
async fn test_file_filtering() {
    let temp_dir = create_test_repo().await;
    let config = OptimizedScanConfig {
        include_extensions: vec!["rs".to_string()], // Only Rust files
        exclude_directories: vec!["node_modules".to_string()],
        max_file_size: 10_000, // Exclude large files
        enable_git: false,
        enable_incremental: false,
        ..Default::default()
    };

    let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();
    let result = scanner.scan().await.unwrap();

    // Should only find .rs files
    let files = result.files.to_full_file_infos().unwrap();
    for file in &files {
        assert!(file.path.extension().unwrap_or_default() == "rs");
    }

    // Should have filtered out non-.rs files and large files
    assert!(result.stats.files_filtered > 0);
    assert!(result.performance.filter_effectiveness > 0.0);
}

#[tokio::test]
async fn test_performance_report_generation() {
    let temp_dir = create_test_repo().await;
    let config = OptimizedScanConfig {
        enable_monitoring: true,
        enable_git: false,
        enable_incremental: false,
        ..Default::default()
    };

    let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();
    let result = scanner.scan().await.unwrap();

    let report = result.generate_performance_report();

    assert!(report.contains("Optimized Scan Performance Report"));
    assert!(report.contains("Duration:"));
    assert!(report.contains("Files Processed:"));
    assert!(report.contains("Throughput:"));
    assert!(report.contains("Pre-filtering:"));
    assert!(report.contains("Time Breakdown"));

    println!("Performance Report:\n{}", report);
}

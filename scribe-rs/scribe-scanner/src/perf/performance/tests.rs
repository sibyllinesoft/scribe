//! Tests for performance monitoring module.

use super::*;
use std::thread;

#[test]
fn test_performance_monitor_creation() {
    let monitor = PerformanceMonitor::new();
    let snapshot = monitor.get_current_snapshot();

    assert_eq!(snapshot.files_per_second, 0.0);
    assert_eq!(snapshot.cache_hit_rate, 0.0);
}

#[test]
fn test_real_time_metrics() {
    let monitor = PerformanceMonitor::new();

    // Record some operations
    monitor.record_file_processed(1024, Duration::from_millis(10));
    monitor.record_file_cached();
    monitor.record_cache_miss();

    let snapshot = monitor.get_current_snapshot();
    assert!(snapshot.files_per_second > 0.0);
}

#[test]
fn test_perf_timer() {
    let _timer = PerfTimer::start("test_operation");
    thread::sleep(Duration::from_millis(1));
    // Timer will auto-record on drop
}

#[test]
fn test_perf_timer_macro() {
    let _timer = perf_timer!("macro_test");
    thread::sleep(Duration::from_millis(1));

    let _timer2 = perf_timer!("macro_test_with_bytes", 1024);
    thread::sleep(Duration::from_millis(1));
}

#[test]
fn test_operation_profile() {
    let mut profile = OperationProfile::new("test_op");

    profile.record(Duration::from_millis(10), 1024, true);
    profile.record(Duration::from_millis(20), 2048, false);

    assert_eq!(profile.call_count, 2);
    assert_eq!(profile.success_count, 1);
    assert_eq!(profile.error_count, 1);
    assert_eq!(profile.bytes_processed, 3072);
}

#[test]
fn test_error_recording() {
    let monitor = PerformanceMonitor::new();

    monitor.record_error(ErrorType::Io);
    monitor.record_error(ErrorType::Git);
    monitor.record_error(ErrorType::Parsing);

    assert_eq!(monitor.real_time.io_errors.load(Ordering::Relaxed), 1);
    assert_eq!(monitor.real_time.git_errors.load(Ordering::Relaxed), 1);
    assert_eq!(monitor.real_time.parsing_errors.load(Ordering::Relaxed), 1);
}

#[test]
fn test_memory_tracking() {
    let monitor = PerformanceMonitor::new();

    monitor.update_memory_usage(1024 * 1024); // 1MB
    monitor.update_memory_usage(2048 * 1024); // 2MB
    monitor.update_memory_usage(512 * 1024); // 512KB

    assert_eq!(
        monitor
            .real_time
            .current_memory_bytes
            .load(Ordering::Relaxed),
        512 * 1024
    );
    assert_eq!(
        monitor.real_time.peak_memory_bytes.load(Ordering::Relaxed),
        2048 * 1024
    );
}

#[test]
fn test_performance_report() {
    let monitor = PerformanceMonitor::new();

    // Record some data
    monitor.record_file_processed(1024, Duration::from_millis(10));
    monitor.record_file_cached();
    monitor.update_memory_usage(1024 * 1024);

    let report = monitor.generate_report();

    assert!(report.timestamp > 0);
    assert!(report.current.files_per_second >= 0.0);
}

#[test]
fn test_bottleneck_identification() {
    let monitor = PerformanceMonitor::new();

    // Simulate high memory usage
    monitor.update_memory_usage(2000 * 1024 * 1024); // 2GB

    let bottlenecks = monitor.identify_bottlenecks();
    assert!(bottlenecks.iter().any(|b| b.contains("memory")));
}

#[test]
fn test_metrics_reset() {
    let monitor = PerformanceMonitor::new();

    // Record some data
    monitor.record_file_processed(1024, Duration::from_millis(10));
    monitor.record_cache_miss();

    // Verify data exists
    assert!(monitor.real_time.files_processed.load(Ordering::Relaxed) > 0);

    // Reset and verify
    monitor.reset_metrics();
    assert_eq!(monitor.real_time.files_processed.load(Ordering::Relaxed), 0);
    assert_eq!(monitor.real_time.cache_misses.load(Ordering::Relaxed), 0);
}

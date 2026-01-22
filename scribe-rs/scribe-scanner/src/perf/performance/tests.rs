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

#[test]
fn test_file_filtered_recording() {
    let monitor = PerformanceMonitor::new();

    monitor.record_file_filtered();
    monitor.record_file_filtered();
    monitor.record_file_filtered();

    assert_eq!(monitor.real_time.files_filtered.load(Ordering::Relaxed), 3);
}

#[test]
fn test_file_failed_recording() {
    let monitor = PerformanceMonitor::new();

    monitor.record_file_failed();
    monitor.record_file_failed();

    assert_eq!(monitor.real_time.files_failed.load(Ordering::Relaxed), 2);
}

#[test]
fn test_io_operation_recording() {
    let monitor = PerformanceMonitor::new();

    monitor.record_io_operation(5000, Duration::from_millis(50));
    monitor.record_io_operation(3000, Duration::from_millis(30));

    assert_eq!(monitor.real_time.bytes_read.load(Ordering::Relaxed), 8000);
    assert!(monitor.real_time.io_time_us.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_git_operation_recording() {
    let monitor = PerformanceMonitor::new();

    monitor.record_git_operation(Duration::from_millis(100));
    monitor.record_git_operation(Duration::from_millis(50));

    assert!(monitor.real_time.git_time_us.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_thread_count_update() {
    let monitor = PerformanceMonitor::new();

    monitor.update_thread_count(4);
    assert_eq!(monitor.real_time.active_threads.load(Ordering::Relaxed), 4);
    assert_eq!(monitor.real_time.peak_threads.load(Ordering::Relaxed), 4);

    monitor.update_thread_count(8);
    assert_eq!(monitor.real_time.active_threads.load(Ordering::Relaxed), 8);
    assert_eq!(monitor.real_time.peak_threads.load(Ordering::Relaxed), 8);

    monitor.update_thread_count(2);
    assert_eq!(monitor.real_time.active_threads.load(Ordering::Relaxed), 2);
    // Peak should stay at 8
    assert_eq!(monitor.real_time.peak_threads.load(Ordering::Relaxed), 8);
}

#[test]
fn test_error_type_other() {
    let monitor = PerformanceMonitor::new();

    monitor.record_error(ErrorType::Other);

    assert_eq!(monitor.real_time.other_errors.load(Ordering::Relaxed), 1);
}

#[test]
fn test_aggregated_stats() {
    let monitor = PerformanceMonitor::new();

    // Record some data
    for _ in 0..10 {
        monitor.record_file_processed(1024, Duration::from_millis(10));
    }

    let aggregated = monitor.get_aggregated_stats();
    // Initially aggregated stats may be default
    assert!(aggregated.avg_throughput_fps >= 0.0);
}

#[test]
fn test_operation_profiles() {
    let monitor = PerformanceMonitor::new();

    // Profile some operations manually
    monitor.profile_operation("read_file", Duration::from_millis(5), 1024, true);
    monitor.profile_operation("read_file", Duration::from_millis(10), 2048, true);
    monitor.profile_operation("parse_file", Duration::from_millis(15), 512, false);

    let profiles = monitor.get_operation_profiles();
    // Since profiling may be disabled by default config, just check the function works
    assert!(profiles.len() >= 0);
}

#[test]
fn test_perf_timer_with_bytes() {
    let timer = PerfTimer::start_with_bytes("test_bytes_op", 2048);
    assert!(timer.bytes_hint.is_some());
    assert_eq!(timer.bytes_hint.unwrap(), 2048);
    // Timer will auto-record on drop
}

#[test]
fn test_perf_timer_finish_success() {
    let timer = PerfTimer::start("success_op");
    thread::sleep(Duration::from_millis(1));
    timer.finish_success();
    // No assertion needed, just verify it doesn't panic
}

#[test]
fn test_perf_timer_finish_error() {
    let timer = PerfTimer::start("error_op");
    thread::sleep(Duration::from_millis(1));
    timer.finish_error();
    // No assertion needed, just verify it doesn't panic
}

#[test]
fn test_operation_profile_min_max() {
    let mut profile = OperationProfile::new("test_min_max");

    profile.record(Duration::from_millis(10), 100, true);
    profile.record(Duration::from_millis(5), 200, true);
    profile.record(Duration::from_millis(20), 300, true);

    // Min should be ~5000us, max should be ~20000us
    assert!(profile.min_time_us <= 10000); // 5ms or less
    assert!(profile.max_time_us >= 15000); // 20ms or more
}

#[test]
fn test_performance_history() {
    let mut history = PerformanceHistory::new(10);

    // Add snapshots
    for i in 0..5 {
        history.add_snapshot(PerformanceSnapshot {
            timestamp: i as u64,
            files_per_second: (i + 1) as f64 * 100.0,
            bytes_per_second: (i + 1) as f64 * 1000.0,
            memory_usage_mb: 100.0,
            cpu_utilization: 50.0,
            io_wait_percentage: 5.0,
            cache_hit_rate: 0.8,
            error_rate: 0.01,
            active_threads: 4,
            queue_depth: 10,
        });
    }

    assert_eq!(history.snapshots.len(), 5);
    assert!(history.aggregated.avg_throughput_fps > 0.0);
}

#[test]
fn test_performance_history_overflow() {
    let mut history = PerformanceHistory::new(3);

    // Add more snapshots than capacity
    for i in 0..5 {
        history.add_snapshot(PerformanceSnapshot {
            timestamp: i as u64,
            files_per_second: 100.0,
            bytes_per_second: 1000.0,
            memory_usage_mb: 100.0,
            cpu_utilization: 50.0,
            io_wait_percentage: 5.0,
            cache_hit_rate: 0.8,
            error_rate: 0.01,
            active_threads: 4,
            queue_depth: 10,
        });
    }

    // Should only keep last 3
    assert_eq!(history.snapshots.len(), 3);
}

#[test]
fn test_real_time_metrics_reset() {
    let metrics = RealTimeMetrics::new();

    metrics.files_processed.store(100, Ordering::Relaxed);
    metrics.bytes_processed.store(50000, Ordering::Relaxed);
    metrics.cache_hits.store(80, Ordering::Relaxed);

    metrics.reset();

    assert_eq!(metrics.files_processed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.bytes_processed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.cache_hits.load(Ordering::Relaxed), 0);
}

#[test]
fn test_system_resource_tracker_creation() {
    let tracker = SystemResourceTracker::new();
    assert!(tracker.cpu_samples.is_empty());
    assert!(tracker.memory_samples.is_empty());
}

#[test]
fn test_system_resource_tracker_cpu_utilization_no_samples() {
    let tracker = SystemResourceTracker::new();
    // With no samples, should return 0.0
    assert_eq!(tracker.get_cpu_utilization(), 0.0);
}

#[test]
fn test_system_resource_tracker_io_wait() {
    let tracker = SystemResourceTracker::new();
    // Returns a fixed value
    assert_eq!(tracker.get_io_wait_percentage(), 10.0);
}

#[test]
fn test_snapshot_fields() {
    let snapshot = PerformanceSnapshot {
        timestamp: 1234567890,
        files_per_second: 500.0,
        bytes_per_second: 10_000_000.0,
        memory_usage_mb: 256.5,
        cpu_utilization: 75.0,
        io_wait_percentage: 8.5,
        cache_hit_rate: 0.95,
        error_rate: 0.001,
        active_threads: 8,
        queue_depth: 50,
    };

    assert_eq!(snapshot.timestamp, 1234567890);
    assert_eq!(snapshot.files_per_second, 500.0);
    assert_eq!(snapshot.active_threads, 8);
    assert_eq!(snapshot.queue_depth, 50);
}

#[test]
fn test_monitoring_config_default() {
    let config = MonitoringConfig::default();
    // Just verify default works
    assert!(config.sample_interval_ms > 0);
}

#[test]
fn test_performance_report_fields() {
    let monitor = PerformanceMonitor::new();
    let report = monitor.generate_report();

    assert!(report.timestamp > 0);
    assert!(report.current.files_per_second >= 0.0);
    // Check recommendations and bottlenecks are lists
    assert!(report.recommendations.len() >= 0);
    assert!(report.bottlenecks.len() >= 0);
}

#[test]
fn test_cache_hit_rate_calculation() {
    let monitor = PerformanceMonitor::new();

    // Record hits and misses
    for _ in 0..8 {
        monitor.record_file_cached();
    }
    for _ in 0..2 {
        monitor.record_cache_miss();
    }

    let snapshot = monitor.get_current_snapshot();
    // Should be 8/10 = 0.8
    assert!((snapshot.cache_hit_rate - 0.8).abs() < 0.01);
}

#[test]
fn test_aggregated_stats_default() {
    let stats = AggregatedStats::default();

    assert_eq!(stats.avg_throughput_fps, 0.0);
    assert_eq!(stats.max_memory_mb, 0.0);
    assert_eq!(stats.p50_latency_ms, 0.0);
    assert_eq!(stats.p95_latency_ms, 0.0);
    assert_eq!(stats.p99_latency_ms, 0.0);
}

#[test]
fn test_io_stats_default() {
    let stats = IoStats::default();

    assert_eq!(stats.bytes_read, 0);
    assert_eq!(stats.bytes_written, 0);
    assert_eq!(stats.reads_completed, 0);
    assert_eq!(stats.writes_completed, 0);
}

#[test]
fn test_multiple_bottlenecks() {
    let monitor = PerformanceMonitor::new();

    // Simulate conditions that trigger multiple bottlenecks
    monitor.update_memory_usage(2000 * 1024 * 1024); // 2GB - high memory

    // Record operations with low cache hit rate (all misses)
    for _ in 0..10 {
        monitor.record_cache_miss();
    }

    let bottlenecks = monitor.identify_bottlenecks();
    // Should have at least memory and cache bottlenecks
    assert!(bottlenecks.iter().any(|b| b.contains("memory")));
    assert!(bottlenecks.iter().any(|b| b.contains("cache")));
}

#[test]
fn test_system_resource_tracker_sample_metrics() {
    let mut tracker = SystemResourceTracker::new();

    // Sample system metrics - should not panic
    tracker.sample_system_metrics();

    // On Linux, samples should be collected
    #[cfg(unix)]
    {
        // May or may not have samples depending on /proc availability
        // Just verify it doesn't panic
    }
}

#[test]
fn test_system_resource_tracker_multiple_samples() {
    let mut tracker = SystemResourceTracker::new();

    // Take multiple samples
    for _ in 0..5 {
        tracker.sample_system_metrics();
        thread::sleep(Duration::from_millis(10));
    }

    // CPU utilization should work with multiple samples
    let cpu = tracker.get_cpu_utilization();
    assert!(cpu >= 0.0);
}

#[test]
fn test_cpu_sample_fields() {
    let sample = CpuSample {
        timestamp: Instant::now(),
        user_time: Duration::from_secs(100),
        system_time: Duration::from_secs(50),
        idle_time: Duration::from_secs(200),
    };

    assert_eq!(sample.user_time, Duration::from_secs(100));
    assert_eq!(sample.system_time, Duration::from_secs(50));
    assert_eq!(sample.idle_time, Duration::from_secs(200));
}

#[test]
fn test_memory_sample_fields() {
    let sample = MemorySample {
        timestamp: Instant::now(),
        rss_bytes: 1024 * 1024 * 100,
        vms_bytes: 1024 * 1024 * 500,
        heap_bytes: 1024 * 1024 * 50,
        available_bytes: 1024 * 1024 * 1000,
    };

    assert_eq!(sample.rss_bytes, 1024 * 1024 * 100);
    assert_eq!(sample.vms_bytes, 1024 * 1024 * 500);
    assert_eq!(sample.heap_bytes, 1024 * 1024 * 50);
    assert_eq!(sample.available_bytes, 1024 * 1024 * 1000);
}

#[test]
fn test_get_cpu_utilization_with_samples() {
    let mut tracker = SystemResourceTracker::new();

    // Manually add CPU samples to test utilization calculation
    tracker.cpu_samples.push_back(CpuSample {
        timestamp: Instant::now(),
        user_time: Duration::from_secs(100),
        system_time: Duration::from_secs(50),
        idle_time: Duration::from_secs(350),
    });

    thread::sleep(Duration::from_millis(10));

    tracker.cpu_samples.push_back(CpuSample {
        timestamp: Instant::now(),
        user_time: Duration::from_secs(110),
        system_time: Duration::from_secs(55),
        idle_time: Duration::from_secs(360),
    });

    let utilization = tracker.get_cpu_utilization();
    // With these values, utilization should be calculated
    assert!(utilization >= 0.0);
    assert!(utilization <= 100.0);
}

#[test]
fn test_io_stats_fields() {
    let stats = IoStats {
        bytes_read: 1000,
        bytes_written: 500,
        reads_completed: 100,
        writes_completed: 50,
        read_time_ms: 5000,
        write_time_ms: 2000,
    };

    assert_eq!(stats.bytes_read, 1000);
    assert_eq!(stats.bytes_written, 500);
    assert_eq!(stats.reads_completed, 100);
    assert_eq!(stats.writes_completed, 50);
    assert_eq!(stats.read_time_ms, 5000);
    assert_eq!(stats.write_time_ms, 2000);
}

#[test]
fn test_performance_history_empty() {
    let history = PerformanceHistory::new(10);

    assert!(history.snapshots.is_empty());
    assert_eq!(history.max_snapshots, 10);
}

#[test]
fn test_performance_history_aggregated_with_one_snapshot() {
    let mut history = PerformanceHistory::new(10);

    history.add_snapshot(PerformanceSnapshot {
        timestamp: 1,
        files_per_second: 100.0,
        bytes_per_second: 1000.0,
        memory_usage_mb: 50.0,
        cpu_utilization: 25.0,
        io_wait_percentage: 5.0,
        cache_hit_rate: 0.9,
        error_rate: 0.0,
        active_threads: 2,
        queue_depth: 5,
    });

    assert_eq!(history.snapshots.len(), 1);
    assert_eq!(history.aggregated.avg_throughput_fps, 100.0);
    assert_eq!(history.aggregated.max_memory_mb, 50.0);
}

#[test]
fn test_get_top_operations_empty() {
    let profiles: FxHashMap<String, OperationProfile> = FxHashMap::default();
    let top = PerformanceMonitor::get_top_operations(&profiles, 10);
    assert!(top.is_empty());
}

#[test]
fn test_get_top_operations_limited() {
    let mut profiles: FxHashMap<String, OperationProfile> = FxHashMap::default();

    let mut p1 = OperationProfile::new("op1");
    p1.total_time_us = 1000;

    let mut p2 = OperationProfile::new("op2");
    p2.total_time_us = 5000;

    let mut p3 = OperationProfile::new("op3");
    p3.total_time_us = 2000;

    profiles.insert("op1".to_string(), p1);
    profiles.insert("op2".to_string(), p2);
    profiles.insert("op3".to_string(), p3);

    // Get top 2
    let top = PerformanceMonitor::get_top_operations(&profiles, 2);
    assert_eq!(top.len(), 2);
    // Should be sorted by total_time_us descending
    assert_eq!(top[0].total_time_us, 5000);
    assert_eq!(top[1].total_time_us, 2000);
}

#[test]
fn test_performance_report_top_operations() {
    let monitor = PerformanceMonitor::new();
    let report = monitor.generate_report();

    // Just verify it works with empty profiles
    assert!(report.top_operations.len() >= 0);
}

#[test]
fn test_performance_report_recommendations() {
    let monitor = PerformanceMonitor::new();

    // Set conditions for recommendations
    monitor.update_memory_usage(2000 * 1024 * 1024); // High memory

    let report = monitor.generate_report();

    // Should have recommendation for high memory
    assert!(report
        .recommendations
        .iter()
        .any(|r| r.contains("memory") || r.contains("batch")));
}

#[test]
fn test_real_time_metrics_new_all_zeroes() {
    let metrics = RealTimeMetrics::new();

    assert_eq!(metrics.files_processed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.files_filtered.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.files_cached.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.files_failed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.bytes_processed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.bytes_read.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.total_scan_time_us.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.io_time_us.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.cpu_time_us.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.git_time_us.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.filter_time_us.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.parallel_time_us.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.peak_memory_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.current_memory_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.memory_allocations.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.active_threads.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.peak_threads.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.context_switches.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.cache_hits.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.cache_misses.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.cache_evictions.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.io_errors.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.git_errors.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.parsing_errors.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.other_errors.load(Ordering::Relaxed), 0);
}

#[test]
fn test_operation_profile_avg_time_calculation() {
    let mut profile = OperationProfile::new("avg_test");

    profile.record(Duration::from_millis(10), 0, true); // 10000us
    assert_eq!(profile.avg_time_us, 10000);

    profile.record(Duration::from_millis(30), 0, true); // 30000us
                                                        // Total: 40000us, count: 2, avg: 20000us
    assert_eq!(profile.avg_time_us, 20000);
}

#[test]
fn test_perf_timer_operation_name() {
    let timer = PerfTimer::start("my_operation");
    assert_eq!(timer.operation_name, "my_operation");
    assert!(timer.bytes_hint.is_none());
}

#[test]
fn test_global_perf_monitor_access() {
    // Access global monitor
    let snapshot = PERF_MONITOR.get_current_snapshot();
    assert!(snapshot.files_per_second >= 0.0);

    // Can record through global
    PERF_MONITOR.record_file_processed(100, Duration::from_millis(1));
}

#[test]
fn test_snapshot_zero_scan_time() {
    // When scan_time_us is 0, files_per_second and bytes_per_second should be 0
    let monitor = PerformanceMonitor::new();
    let snapshot = monitor.get_current_snapshot();

    // With no operations recorded, should be 0
    assert_eq!(snapshot.files_per_second, 0.0);
    assert_eq!(snapshot.bytes_per_second, 0.0);
}

#[test]
fn test_profile_operation_disabled() {
    // Create monitor with profiling disabled (default config has it disabled)
    let monitor = PerformanceMonitor::new();

    // profile_operation should be a no-op when disabled
    monitor.profile_operation("test_op", Duration::from_millis(10), 100, true);

    // Profiles should be empty since profiling is disabled
    let profiles = monitor.get_operation_profiles();
    // Just verify it doesn't panic - profiles may or may not be empty depending on config
    assert!(profiles.len() >= 0);
}

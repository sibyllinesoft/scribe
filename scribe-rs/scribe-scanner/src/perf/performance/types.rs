//! Type definitions for performance monitoring.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::time::{Duration, Instant};

/// Real-time performance metrics (atomic counters)
#[derive(Debug)]
pub struct RealTimeMetrics {
    // File processing counters
    pub files_processed: AtomicU64,
    pub files_filtered: AtomicU64,
    pub files_cached: AtomicU64,
    pub files_failed: AtomicU64,
    pub bytes_processed: AtomicU64,
    pub bytes_read: AtomicU64,

    // Timing counters (microseconds)
    pub total_scan_time_us: AtomicU64,
    pub io_time_us: AtomicU64,
    pub cpu_time_us: AtomicU64,
    pub git_time_us: AtomicU64,
    pub filter_time_us: AtomicU64,
    pub parallel_time_us: AtomicU64,

    // Memory tracking
    pub peak_memory_bytes: AtomicU64,
    pub current_memory_bytes: AtomicU64,
    pub memory_allocations: AtomicU64,

    // Concurrency metrics
    pub active_threads: AtomicUsize,
    pub peak_threads: AtomicUsize,
    pub context_switches: AtomicU64,

    // Cache metrics
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_evictions: AtomicU64,

    // Error counters
    pub io_errors: AtomicU64,
    pub git_errors: AtomicU64,
    pub parsing_errors: AtomicU64,
    pub other_errors: AtomicU64,
}

/// Historical performance data
#[derive(Debug)]
pub struct PerformanceHistory {
    /// Time-series performance snapshots
    pub snapshots: VecDeque<PerformanceSnapshot>,
    /// Maximum snapshots to keep
    pub max_snapshots: usize,
    /// Aggregated statistics
    pub aggregated: AggregatedStats,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: u64,
    pub files_per_second: f64,
    pub bytes_per_second: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
    pub io_wait_percentage: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub active_threads: usize,
    pub queue_depth: usize,
}

/// Aggregated performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedStats {
    pub avg_throughput_fps: f64, // files per second
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub max_memory_mb: f64,
    pub avg_memory_mb: f64,
    pub total_files_processed: u64,
    pub total_bytes_processed: u64,
    pub total_runtime_seconds: f64,
    pub cache_efficiency: f64,
    pub error_percentage: f64,
}

/// System resource tracking
#[derive(Debug)]
pub struct SystemResourceTracker {
    /// CPU usage samples
    pub cpu_samples: VecDeque<CpuSample>,
    /// Memory usage samples
    pub memory_samples: VecDeque<MemorySample>,
    /// I/O statistics
    pub io_stats: IoStats,
    /// Last sample time
    pub last_sample_time: Instant,
}

/// CPU usage sample
#[derive(Debug, Clone)]
pub struct CpuSample {
    pub timestamp: Instant,
    pub user_time: Duration,
    pub system_time: Duration,
    pub idle_time: Duration,
}

/// Memory usage sample
#[derive(Debug, Clone)]
pub struct MemorySample {
    pub timestamp: Instant,
    pub rss_bytes: u64,       // Resident Set Size
    pub vms_bytes: u64,       // Virtual Memory Size
    pub heap_bytes: u64,      // Heap usage
    pub available_bytes: u64, // Available system memory
}

/// I/O statistics
#[derive(Debug, Default)]
pub struct IoStats {
    pub reads_completed: u64,
    pub writes_completed: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_time_ms: u64,
    pub write_time_ms: u64,
}

/// Operation-specific performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    pub operation_name: String,
    pub call_count: u64,
    pub total_time_us: u64,
    pub min_time_us: u64,
    pub max_time_us: u64,
    pub avg_time_us: u64,
    pub p95_time_us: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub bytes_processed: u64,
    pub last_updated: u64,
}

/// Configuration for performance monitoring
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Sample interval for system metrics (milliseconds)
    pub sample_interval_ms: u64,
    /// Maximum history snapshots to keep
    pub max_history_snapshots: usize,
    /// Performance report interval (seconds)
    pub report_interval_secs: u64,
    /// Enable memory tracking
    pub track_memory: bool,
    /// Enable I/O tracking
    pub track_io: bool,
    /// Enable per-operation profiling
    pub profile_operations: bool,
}

/// Error types for performance monitoring
#[derive(Debug, Clone, Copy)]
pub enum ErrorType {
    Io,
    Git,
    Parsing,
    Other,
}

/// Complete performance report
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub current: PerformanceSnapshot,
    pub aggregated: AggregatedStats,
    pub top_operations: Vec<OperationProfile>,
    pub bottlenecks: Vec<String>,
    pub recommendations: Vec<String>,
    pub timestamp: u64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            sample_interval_ms: 1000,
            max_history_snapshots: 3600, // 1 hour at 1s intervals
            report_interval_secs: 30,
            track_memory: true,
            track_io: true,
            profile_operations: true,
        }
    }
}

impl Default for AggregatedStats {
    fn default() -> Self {
        Self {
            avg_throughput_fps: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            max_memory_mb: 0.0,
            avg_memory_mb: 0.0,
            total_files_processed: 0,
            total_bytes_processed: 0,
            total_runtime_seconds: 0.0,
            cache_efficiency: 0.0,
            error_percentage: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_snapshot_creation() {
        let snapshot = PerformanceSnapshot {
            timestamp: 1234567890,
            files_per_second: 100.0,
            bytes_per_second: 1048576.0,
            memory_usage_mb: 512.0,
            cpu_utilization: 0.75,
            io_wait_percentage: 0.10,
            cache_hit_rate: 0.85,
            error_rate: 0.01,
            active_threads: 8,
            queue_depth: 10,
        };

        assert_eq!(snapshot.timestamp, 1234567890);
        assert!((snapshot.files_per_second - 100.0).abs() < 0.001);
        assert!((snapshot.bytes_per_second - 1048576.0).abs() < 0.001);
        assert_eq!(snapshot.active_threads, 8);
    }

    #[test]
    fn test_performance_snapshot_clone() {
        let snapshot = PerformanceSnapshot {
            timestamp: 100,
            files_per_second: 50.0,
            bytes_per_second: 1000.0,
            memory_usage_mb: 256.0,
            cpu_utilization: 0.5,
            io_wait_percentage: 0.05,
            cache_hit_rate: 0.9,
            error_rate: 0.0,
            active_threads: 4,
            queue_depth: 5,
        };

        let cloned = snapshot.clone();
        assert_eq!(snapshot.timestamp, cloned.timestamp);
        assert_eq!(snapshot.active_threads, cloned.active_threads);
    }

    #[test]
    fn test_performance_snapshot_serialize() {
        let snapshot = PerformanceSnapshot {
            timestamp: 999,
            files_per_second: 200.0,
            bytes_per_second: 2048.0,
            memory_usage_mb: 128.0,
            cpu_utilization: 0.8,
            io_wait_percentage: 0.15,
            cache_hit_rate: 0.7,
            error_rate: 0.02,
            active_threads: 16,
            queue_depth: 20,
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("timestamp"));
        assert!(json.contains("999"));

        let deserialized: PerformanceSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snapshot.timestamp, deserialized.timestamp);
    }

    #[test]
    fn test_performance_snapshot_debug() {
        let snapshot = PerformanceSnapshot {
            timestamp: 0,
            files_per_second: 0.0,
            bytes_per_second: 0.0,
            memory_usage_mb: 0.0,
            cpu_utilization: 0.0,
            io_wait_percentage: 0.0,
            cache_hit_rate: 0.0,
            error_rate: 0.0,
            active_threads: 0,
            queue_depth: 0,
        };

        let debug = format!("{:?}", snapshot);
        assert!(debug.contains("PerformanceSnapshot"));
    }

    #[test]
    fn test_aggregated_stats_default() {
        let stats = AggregatedStats::default();
        assert_eq!(stats.avg_throughput_fps, 0.0);
        assert_eq!(stats.p50_latency_ms, 0.0);
        assert_eq!(stats.p95_latency_ms, 0.0);
        assert_eq!(stats.p99_latency_ms, 0.0);
        assert_eq!(stats.total_files_processed, 0);
        assert_eq!(stats.total_bytes_processed, 0);
    }

    #[test]
    fn test_aggregated_stats_custom() {
        let stats = AggregatedStats {
            avg_throughput_fps: 150.0,
            p50_latency_ms: 5.0,
            p95_latency_ms: 15.0,
            p99_latency_ms: 25.0,
            max_memory_mb: 1024.0,
            avg_memory_mb: 512.0,
            total_files_processed: 10000,
            total_bytes_processed: 104857600,
            total_runtime_seconds: 60.0,
            cache_efficiency: 0.95,
            error_percentage: 0.001,
        };

        assert!((stats.avg_throughput_fps - 150.0).abs() < 0.001);
        assert_eq!(stats.total_files_processed, 10000);
    }

    #[test]
    fn test_aggregated_stats_clone() {
        let stats = AggregatedStats::default();
        let cloned = stats.clone();
        assert_eq!(stats.total_files_processed, cloned.total_files_processed);
    }

    #[test]
    fn test_aggregated_stats_serialize() {
        let stats = AggregatedStats {
            avg_throughput_fps: 100.0,
            p50_latency_ms: 10.0,
            p95_latency_ms: 20.0,
            p99_latency_ms: 30.0,
            max_memory_mb: 2048.0,
            avg_memory_mb: 1024.0,
            total_files_processed: 5000,
            total_bytes_processed: 50000000,
            total_runtime_seconds: 30.0,
            cache_efficiency: 0.8,
            error_percentage: 0.005,
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("avg_throughput_fps"));
        assert!(json.contains("total_files_processed"));

        let deserialized: AggregatedStats = serde_json::from_str(&json).unwrap();
        assert_eq!(
            stats.total_files_processed,
            deserialized.total_files_processed
        );
    }

    #[test]
    fn test_aggregated_stats_debug() {
        let stats = AggregatedStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("AggregatedStats"));
    }

    #[test]
    fn test_io_stats_default() {
        let stats = IoStats::default();
        assert_eq!(stats.reads_completed, 0);
        assert_eq!(stats.writes_completed, 0);
        assert_eq!(stats.bytes_read, 0);
        assert_eq!(stats.bytes_written, 0);
        assert_eq!(stats.read_time_ms, 0);
        assert_eq!(stats.write_time_ms, 0);
    }

    #[test]
    fn test_io_stats_custom() {
        let stats = IoStats {
            reads_completed: 1000,
            writes_completed: 500,
            bytes_read: 10485760,
            bytes_written: 5242880,
            read_time_ms: 500,
            write_time_ms: 250,
        };

        assert_eq!(stats.reads_completed, 1000);
        assert_eq!(stats.writes_completed, 500);
        assert_eq!(stats.bytes_read, 10485760);
    }

    #[test]
    fn test_io_stats_debug() {
        let stats = IoStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("IoStats"));
    }

    #[test]
    fn test_operation_profile_creation() {
        let profile = OperationProfile {
            operation_name: "file_scan".to_string(),
            call_count: 1000,
            total_time_us: 500000,
            min_time_us: 100,
            max_time_us: 5000,
            avg_time_us: 500,
            p95_time_us: 2000,
            success_count: 990,
            error_count: 10,
            bytes_processed: 104857600,
            last_updated: 1234567890,
        };

        assert_eq!(profile.operation_name, "file_scan");
        assert_eq!(profile.call_count, 1000);
        assert_eq!(profile.success_count, 990);
        assert_eq!(profile.error_count, 10);
    }

    #[test]
    fn test_operation_profile_clone() {
        let profile = OperationProfile {
            operation_name: "test_op".to_string(),
            call_count: 100,
            total_time_us: 50000,
            min_time_us: 50,
            max_time_us: 1000,
            avg_time_us: 500,
            p95_time_us: 800,
            success_count: 100,
            error_count: 0,
            bytes_processed: 1024000,
            last_updated: 100,
        };

        let cloned = profile.clone();
        assert_eq!(profile.operation_name, cloned.operation_name);
        assert_eq!(profile.call_count, cloned.call_count);
    }

    #[test]
    fn test_operation_profile_serialize() {
        let profile = OperationProfile {
            operation_name: "serialize_test".to_string(),
            call_count: 50,
            total_time_us: 25000,
            min_time_us: 100,
            max_time_us: 2000,
            avg_time_us: 500,
            p95_time_us: 1500,
            success_count: 48,
            error_count: 2,
            bytes_processed: 512000,
            last_updated: 999,
        };

        let json = serde_json::to_string(&profile).unwrap();
        assert!(json.contains("serialize_test"));
        assert!(json.contains("call_count"));

        let deserialized: OperationProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile.operation_name, deserialized.operation_name);
    }

    #[test]
    fn test_operation_profile_debug() {
        let profile = OperationProfile {
            operation_name: "debug".to_string(),
            call_count: 1,
            total_time_us: 100,
            min_time_us: 100,
            max_time_us: 100,
            avg_time_us: 100,
            p95_time_us: 100,
            success_count: 1,
            error_count: 0,
            bytes_processed: 100,
            last_updated: 1,
        };

        let debug = format!("{:?}", profile);
        assert!(debug.contains("OperationProfile"));
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert!(config.enable_profiling);
        assert_eq!(config.sample_interval_ms, 1000);
        assert_eq!(config.max_history_snapshots, 3600);
        assert_eq!(config.report_interval_secs, 30);
        assert!(config.track_memory);
        assert!(config.track_io);
        assert!(config.profile_operations);
    }

    #[test]
    fn test_monitoring_config_custom() {
        let config = MonitoringConfig {
            enable_profiling: false,
            sample_interval_ms: 500,
            max_history_snapshots: 1800,
            report_interval_secs: 60,
            track_memory: false,
            track_io: false,
            profile_operations: false,
        };

        assert!(!config.enable_profiling);
        assert_eq!(config.sample_interval_ms, 500);
        assert!(!config.track_memory);
    }

    #[test]
    fn test_monitoring_config_clone() {
        let config = MonitoringConfig::default();
        let cloned = config.clone();
        assert_eq!(config.sample_interval_ms, cloned.sample_interval_ms);
    }

    #[test]
    fn test_monitoring_config_debug() {
        let config = MonitoringConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("MonitoringConfig"));
    }

    #[test]
    fn test_error_type_variants() {
        let io = ErrorType::Io;
        let git = ErrorType::Git;
        let parsing = ErrorType::Parsing;
        let other = ErrorType::Other;

        let io_debug = format!("{:?}", io);
        let git_debug = format!("{:?}", git);
        let parsing_debug = format!("{:?}", parsing);
        let other_debug = format!("{:?}", other);

        assert!(io_debug.contains("Io"));
        assert!(git_debug.contains("Git"));
        assert!(parsing_debug.contains("Parsing"));
        assert!(other_debug.contains("Other"));
    }

    #[test]
    fn test_error_type_clone() {
        let err = ErrorType::Io;
        let cloned = err.clone();
        assert!(matches!(cloned, ErrorType::Io));
    }

    #[test]
    fn test_error_type_copy() {
        let err = ErrorType::Git;
        let copied = err;
        assert!(matches!(err, ErrorType::Git));
        assert!(matches!(copied, ErrorType::Git));
    }

    #[test]
    fn test_performance_report_creation() {
        let report = PerformanceReport {
            current: PerformanceSnapshot {
                timestamp: 100,
                files_per_second: 50.0,
                bytes_per_second: 5000.0,
                memory_usage_mb: 256.0,
                cpu_utilization: 0.6,
                io_wait_percentage: 0.1,
                cache_hit_rate: 0.8,
                error_rate: 0.01,
                active_threads: 4,
                queue_depth: 10,
            },
            aggregated: AggregatedStats::default(),
            top_operations: vec![],
            bottlenecks: vec!["IO bound".to_string()],
            recommendations: vec!["Increase parallelism".to_string()],
            timestamp: 1234567890,
        };

        assert_eq!(report.timestamp, 1234567890);
        assert_eq!(report.bottlenecks.len(), 1);
        assert_eq!(report.recommendations.len(), 1);
    }

    #[test]
    fn test_performance_report_serialize() {
        let report = PerformanceReport {
            current: PerformanceSnapshot {
                timestamp: 0,
                files_per_second: 0.0,
                bytes_per_second: 0.0,
                memory_usage_mb: 0.0,
                cpu_utilization: 0.0,
                io_wait_percentage: 0.0,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                active_threads: 0,
                queue_depth: 0,
            },
            aggregated: AggregatedStats::default(),
            top_operations: vec![],
            bottlenecks: vec![],
            recommendations: vec![],
            timestamp: 999,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("timestamp"));
        assert!(json.contains("current"));
        assert!(json.contains("aggregated"));
    }

    #[test]
    fn test_performance_report_debug() {
        let report = PerformanceReport {
            current: PerformanceSnapshot {
                timestamp: 0,
                files_per_second: 0.0,
                bytes_per_second: 0.0,
                memory_usage_mb: 0.0,
                cpu_utilization: 0.0,
                io_wait_percentage: 0.0,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                active_threads: 0,
                queue_depth: 0,
            },
            aggregated: AggregatedStats::default(),
            top_operations: vec![],
            bottlenecks: vec![],
            recommendations: vec![],
            timestamp: 0,
        };

        let debug = format!("{:?}", report);
        assert!(debug.contains("PerformanceReport"));
    }

    #[test]
    fn test_cpu_sample_creation() {
        let sample = CpuSample {
            timestamp: Instant::now(),
            user_time: Duration::from_millis(100),
            system_time: Duration::from_millis(50),
            idle_time: Duration::from_millis(850),
        };

        assert_eq!(sample.user_time, Duration::from_millis(100));
        assert_eq!(sample.system_time, Duration::from_millis(50));
    }

    #[test]
    fn test_cpu_sample_clone() {
        let sample = CpuSample {
            timestamp: Instant::now(),
            user_time: Duration::from_secs(1),
            system_time: Duration::from_secs(1),
            idle_time: Duration::from_secs(8),
        };

        let cloned = sample.clone();
        assert_eq!(sample.user_time, cloned.user_time);
    }

    #[test]
    fn test_cpu_sample_debug() {
        let sample = CpuSample {
            timestamp: Instant::now(),
            user_time: Duration::from_micros(100),
            system_time: Duration::from_micros(50),
            idle_time: Duration::from_micros(850),
        };

        let debug = format!("{:?}", sample);
        assert!(debug.contains("CpuSample"));
    }

    #[test]
    fn test_memory_sample_creation() {
        let sample = MemorySample {
            timestamp: Instant::now(),
            rss_bytes: 1073741824,
            vms_bytes: 2147483648,
            heap_bytes: 536870912,
            available_bytes: 8589934592,
        };

        assert_eq!(sample.rss_bytes, 1073741824);
        assert_eq!(sample.vms_bytes, 2147483648);
    }

    #[test]
    fn test_memory_sample_clone() {
        let sample = MemorySample {
            timestamp: Instant::now(),
            rss_bytes: 100,
            vms_bytes: 200,
            heap_bytes: 50,
            available_bytes: 1000,
        };

        let cloned = sample.clone();
        assert_eq!(sample.rss_bytes, cloned.rss_bytes);
    }

    #[test]
    fn test_memory_sample_debug() {
        let sample = MemorySample {
            timestamp: Instant::now(),
            rss_bytes: 0,
            vms_bytes: 0,
            heap_bytes: 0,
            available_bytes: 0,
        };

        let debug = format!("{:?}", sample);
        assert!(debug.contains("MemorySample"));
    }
}

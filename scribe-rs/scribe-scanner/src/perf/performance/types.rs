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

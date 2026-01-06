//! Performance instrumentation and monitoring for the scanning system.
//!
//! This module provides comprehensive performance monitoring, profiling, and
//! metrics collection to track scanning performance, identify bottlenecks,
//! and guide optimization efforts.

mod types;
pub use types::*;

use fxhash::FxHashMap;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Global performance monitor instance
pub static PERF_MONITOR: Lazy<PerformanceMonitor> = Lazy::new(PerformanceMonitor::new);

/// Comprehensive performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Real-time metrics
    pub(crate) real_time: Arc<RealTimeMetrics>,
    /// Historical performance data
    history: Arc<RwLock<PerformanceHistory>>,
    /// System resource tracking
    system_tracker: Arc<Mutex<SystemResourceTracker>>,
    /// Performance profiles for different operations
    profiles: Arc<RwLock<FxHashMap<String, OperationProfile>>>,
    /// Configuration
    config: MonitoringConfig,
}

/// Performance timing guard for automatic measurement
#[derive(Debug)]
pub struct PerfTimer {
    start_time: Instant,
    operation_name: String,
    bytes_hint: Option<u64>,
}

/// Macro for easy performance timing
#[macro_export]
macro_rules! perf_timer {
    ($operation:expr) => {
        $crate::perf::performance::PerfTimer::start($operation)
    };
    ($operation:expr, $bytes:expr) => {
        $crate::perf::performance::PerfTimer::start_with_bytes($operation, $bytes)
    };
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            real_time: Arc::new(RealTimeMetrics::new()),
            history: Arc::new(RwLock::new(PerformanceHistory::new(3600))),
            system_tracker: Arc::new(Mutex::new(SystemResourceTracker::new())),
            profiles: Arc::new(RwLock::new(FxHashMap::default())),
            config: MonitoringConfig::default(),
        }
    }

    /// Start performance monitoring
    pub fn start_monitoring(&self) {
        if !self.config.enable_profiling {
            return;
        }

        let real_time = Arc::clone(&self.real_time);
        let history = Arc::clone(&self.history);
        let system_tracker = Arc::clone(&self.system_tracker);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(config.sample_interval_ms));

            loop {
                interval.tick().await;
                let mut tracker = system_tracker.lock();
                tracker.sample_system_metrics();
                drop(tracker);
                let snapshot = Self::create_snapshot(&real_time, &system_tracker);
                let mut hist = history.write();
                hist.add_snapshot(snapshot);
            }
        });

        log::info!("Performance monitoring started");
    }

    /// Record file processing
    pub fn record_file_processed(&self, bytes: u64, duration: Duration) {
        self.real_time.files_processed.fetch_add(1, Ordering::Relaxed);
        self.real_time.bytes_processed.fetch_add(bytes, Ordering::Relaxed);
        self.real_time.total_scan_time_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record file filtered
    pub fn record_file_filtered(&self) {
        self.real_time.files_filtered.fetch_add(1, Ordering::Relaxed);
    }

    /// Record file cached
    pub fn record_file_cached(&self) {
        self.real_time.files_cached.fetch_add(1, Ordering::Relaxed);
        self.real_time.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record file failed
    pub fn record_file_failed(&self) {
        self.real_time.files_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record I/O operation
    pub fn record_io_operation(&self, bytes: u64, duration: Duration) {
        self.real_time.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        self.real_time.io_time_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record git operation
    pub fn record_git_operation(&self, duration: Duration) {
        self.real_time.git_time_us.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.real_time.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record error
    pub fn record_error(&self, error_type: ErrorType) {
        match error_type {
            ErrorType::Io => self.real_time.io_errors.fetch_add(1, Ordering::Relaxed),
            ErrorType::Git => self.real_time.git_errors.fetch_add(1, Ordering::Relaxed),
            ErrorType::Parsing => self.real_time.parsing_errors.fetch_add(1, Ordering::Relaxed),
            ErrorType::Other => self.real_time.other_errors.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, bytes: u64) {
        self.real_time.current_memory_bytes.store(bytes, Ordering::Relaxed);
        self.real_time.peak_memory_bytes.fetch_max(bytes, Ordering::Relaxed);
    }

    /// Update thread count
    pub fn update_thread_count(&self, count: usize) {
        self.real_time.active_threads.store(count, Ordering::Relaxed);
        let mut peak = self.real_time.peak_threads.load(Ordering::Relaxed);
        while peak < count {
            match self.real_time.peak_threads.compare_exchange_weak(
                peak, count, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    /// Profile an operation
    pub fn profile_operation(&self, name: &str, duration: Duration, bytes: u64, success: bool) {
        if !self.config.profile_operations {
            return;
        }
        let mut profiles = self.profiles.write();
        let profile = profiles.entry(name.to_string()).or_insert_with(|| OperationProfile::new(name));
        profile.record(duration, bytes, success);
    }

    /// Get current performance snapshot
    pub fn get_current_snapshot(&self) -> PerformanceSnapshot {
        Self::create_snapshot(&self.real_time, &self.system_tracker)
    }

    /// Get aggregated performance statistics
    pub fn get_aggregated_stats(&self) -> AggregatedStats {
        let history = self.history.read();
        history.aggregated.clone()
    }

    /// Get operation profiles
    pub fn get_operation_profiles(&self) -> FxHashMap<String, OperationProfile> {
        self.profiles.read().clone()
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let snapshot = self.get_current_snapshot();
        let aggregated = self.get_aggregated_stats();
        let profiles = self.get_operation_profiles();

        PerformanceReport {
            current: snapshot,
            aggregated,
            top_operations: Self::get_top_operations(&profiles, 10),
            bottlenecks: self.identify_bottlenecks(),
            recommendations: self.generate_recommendations(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    /// Reset all metrics
    pub fn reset_metrics(&self) {
        self.real_time.reset();
        let mut history = self.history.write();
        history.snapshots.clear();
        history.aggregated = AggregatedStats::default();
        let mut profiles = self.profiles.write();
        profiles.clear();
        log::info!("Performance metrics reset");
    }

    fn create_snapshot(
        real_time: &RealTimeMetrics,
        system_tracker: &Arc<Mutex<SystemResourceTracker>>,
    ) -> PerformanceSnapshot {
        let tracker = system_tracker.lock();
        let files_processed = real_time.files_processed.load(Ordering::Relaxed);
        let bytes_processed = real_time.bytes_processed.load(Ordering::Relaxed);
        let scan_time_us = real_time.total_scan_time_us.load(Ordering::Relaxed);
        let cache_hits = real_time.cache_hits.load(Ordering::Relaxed);
        let cache_misses = real_time.cache_misses.load(Ordering::Relaxed);

        let files_per_second = if scan_time_us > 0 {
            files_processed as f64 / (scan_time_us as f64 / 1_000_000.0)
        } else { 0.0 };

        let bytes_per_second = if scan_time_us > 0 {
            bytes_processed as f64 / (scan_time_us as f64 / 1_000_000.0)
        } else { 0.0 };

        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else { 0.0 };

        PerformanceSnapshot {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            files_per_second,
            bytes_per_second,
            memory_usage_mb: real_time.current_memory_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            cpu_utilization: tracker.get_cpu_utilization(),
            io_wait_percentage: tracker.get_io_wait_percentage(),
            cache_hit_rate,
            error_rate: 0.0,
            active_threads: real_time.active_threads.load(Ordering::Relaxed),
            queue_depth: 0,
        }
    }

    /// Identify performance bottlenecks
    pub fn identify_bottlenecks(&self) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        let snapshot = self.get_current_snapshot();

        if snapshot.cpu_utilization > 80.0 { bottlenecks.push("High CPU utilization".to_string()); }
        if snapshot.io_wait_percentage > 20.0 { bottlenecks.push("High I/O wait time".to_string()); }
        if snapshot.cache_hit_rate < 0.5 { bottlenecks.push("Low cache hit rate".to_string()); }
        if snapshot.memory_usage_mb > 1000.0 { bottlenecks.push("High memory usage".to_string()); }

        bottlenecks
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        for bottleneck in &self.identify_bottlenecks() {
            match bottleneck.as_str() {
                "High CPU utilization" => recommendations.push("Consider reducing parallelism or optimizing CPU-intensive operations".to_string()),
                "High I/O wait time" => recommendations.push("Consider using faster storage or implementing better I/O batching".to_string()),
                "Low cache hit rate" => recommendations.push("Increase cache size or improve cache warming strategies".to_string()),
                "High memory usage" => recommendations.push("Consider reducing batch sizes or implementing memory streaming".to_string()),
                _ => {}
            }
        }
        recommendations
    }

    fn get_top_operations(profiles: &FxHashMap<String, OperationProfile>, limit: usize) -> Vec<OperationProfile> {
        let mut ops: Vec<_> = profiles.values().cloned().collect();
        ops.sort_by(|a, b| b.total_time_us.cmp(&a.total_time_us));
        ops.into_iter().take(limit).collect()
    }
}

impl RealTimeMetrics {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, AtomicUsize};
        Self {
            files_processed: AtomicU64::new(0),
            files_filtered: AtomicU64::new(0),
            files_cached: AtomicU64::new(0),
            files_failed: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            total_scan_time_us: AtomicU64::new(0),
            io_time_us: AtomicU64::new(0),
            cpu_time_us: AtomicU64::new(0),
            git_time_us: AtomicU64::new(0),
            filter_time_us: AtomicU64::new(0),
            parallel_time_us: AtomicU64::new(0),
            peak_memory_bytes: AtomicU64::new(0),
            current_memory_bytes: AtomicU64::new(0),
            memory_allocations: AtomicU64::new(0),
            active_threads: AtomicUsize::new(0),
            peak_threads: AtomicUsize::new(0),
            context_switches: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_evictions: AtomicU64::new(0),
            io_errors: AtomicU64::new(0),
            git_errors: AtomicU64::new(0),
            parsing_errors: AtomicU64::new(0),
            other_errors: AtomicU64::new(0),
        }
    }

    pub fn reset(&self) {
        self.files_processed.store(0, Ordering::Relaxed);
        self.files_filtered.store(0, Ordering::Relaxed);
        self.files_cached.store(0, Ordering::Relaxed);
        self.files_failed.store(0, Ordering::Relaxed);
        self.bytes_processed.store(0, Ordering::Relaxed);
        self.bytes_read.store(0, Ordering::Relaxed);
        self.total_scan_time_us.store(0, Ordering::Relaxed);
        self.io_time_us.store(0, Ordering::Relaxed);
        self.cpu_time_us.store(0, Ordering::Relaxed);
        self.git_time_us.store(0, Ordering::Relaxed);
        self.filter_time_us.store(0, Ordering::Relaxed);
        self.parallel_time_us.store(0, Ordering::Relaxed);
        self.peak_memory_bytes.store(0, Ordering::Relaxed);
        self.current_memory_bytes.store(0, Ordering::Relaxed);
        self.memory_allocations.store(0, Ordering::Relaxed);
        self.active_threads.store(0, Ordering::Relaxed);
        self.peak_threads.store(0, Ordering::Relaxed);
        self.context_switches.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.cache_evictions.store(0, Ordering::Relaxed);
        self.io_errors.store(0, Ordering::Relaxed);
        self.git_errors.store(0, Ordering::Relaxed);
        self.parsing_errors.store(0, Ordering::Relaxed);
        self.other_errors.store(0, Ordering::Relaxed);
    }
}

impl PerformanceHistory {
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(max_snapshots),
            max_snapshots,
            aggregated: AggregatedStats::default(),
        }
    }

    pub fn add_snapshot(&mut self, snapshot: PerformanceSnapshot) {
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
        self.update_aggregated_stats();
    }

    fn update_aggregated_stats(&mut self) {
        if self.snapshots.is_empty() { return; }

        let mut throughputs: Vec<f64> = self.snapshots.iter().map(|s| s.files_per_second).collect();
        throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.aggregated.avg_throughput_fps = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        if !throughputs.is_empty() {
            self.aggregated.p50_latency_ms = throughputs[throughputs.len() / 2];
            self.aggregated.p95_latency_ms = throughputs[(throughputs.len() * 95) / 100];
            self.aggregated.p99_latency_ms = throughputs[(throughputs.len() * 99) / 100];
        }

        self.aggregated.max_memory_mb = self.snapshots.iter().map(|s| s.memory_usage_mb).fold(0.0, f64::max);
        self.aggregated.avg_memory_mb = self.snapshots.iter().map(|s| s.memory_usage_mb).sum::<f64>() / self.snapshots.len() as f64;
    }
}

impl SystemResourceTracker {
    pub fn new() -> Self {
        Self {
            cpu_samples: VecDeque::with_capacity(60),
            memory_samples: VecDeque::with_capacity(60),
            io_stats: IoStats::default(),
            last_sample_time: Instant::now(),
        }
    }

    pub fn sample_system_metrics(&mut self) {
        let now = Instant::now();
        if let Some(cpu_sample) = self.sample_cpu() {
            self.cpu_samples.push_back(cpu_sample);
            if self.cpu_samples.len() > 60 { self.cpu_samples.pop_front(); }
        }
        if let Some(memory_sample) = self.sample_memory() {
            self.memory_samples.push_back(memory_sample);
            if self.memory_samples.len() > 60 { self.memory_samples.pop_front(); }
        }
        self.last_sample_time = now;
    }

    fn sample_cpu(&self) -> Option<CpuSample> {
        #[cfg(unix)]
        { self.sample_cpu_unix() }
        #[cfg(not(unix))]
        { None }
    }

    #[cfg(unix)]
    fn sample_cpu_unix(&self) -> Option<CpuSample> {
        use std::fs;
        if let Ok(contents) = fs::read_to_string("/proc/stat") {
            let line = contents.lines().next()?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 && parts[0] == "cpu" {
                let user: u64 = parts[1].parse().ok()?;
                let system: u64 = parts[3].parse().ok()?;
                let idle: u64 = parts[4].parse().ok()?;
                return Some(CpuSample {
                    timestamp: Instant::now(),
                    user_time: Duration::from_secs(user / 100),
                    system_time: Duration::from_secs(system / 100),
                    idle_time: Duration::from_secs(idle / 100),
                });
            }
        }
        None
    }

    fn sample_memory(&self) -> Option<MemorySample> {
        #[cfg(unix)]
        { self.sample_memory_unix() }
        #[cfg(not(unix))]
        { None }
    }

    #[cfg(unix)]
    fn sample_memory_unix(&self) -> Option<MemorySample> {
        use std::fs;
        if let Ok(contents) = fs::read_to_string("/proc/self/status") {
            let mut rss_kb = 0u64;
            let mut vms_kb = 0u64;
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        rss_kb = value.parse().unwrap_or(0);
                    }
                } else if line.starts_with("VmSize:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        vms_kb = value.parse().unwrap_or(0);
                    }
                }
            }
            return Some(MemorySample {
                timestamp: Instant::now(),
                rss_bytes: rss_kb * 1024,
                vms_bytes: vms_kb * 1024,
                heap_bytes: 0,
                available_bytes: 0,
            });
        }
        None
    }

    pub fn get_cpu_utilization(&self) -> f64 {
        if self.cpu_samples.len() < 2 { return 0.0; }
        let recent = &self.cpu_samples[self.cpu_samples.len() - 1];
        let previous = &self.cpu_samples[self.cpu_samples.len() - 2];
        let total_time = recent.user_time + recent.system_time + recent.idle_time;
        let prev_total_time = previous.user_time + previous.system_time + previous.idle_time;
        let delta_total = total_time.saturating_sub(prev_total_time);
        let delta_idle = recent.idle_time.saturating_sub(previous.idle_time);
        if delta_total.as_secs_f64() > 0.0 {
            100.0 * (1.0 - delta_idle.as_secs_f64() / delta_total.as_secs_f64())
        } else { 0.0 }
    }

    pub fn get_io_wait_percentage(&self) -> f64 { 10.0 }
}

impl OperationProfile {
    pub fn new(name: &str) -> Self {
        Self {
            operation_name: name.to_string(),
            call_count: 0,
            total_time_us: 0,
            min_time_us: u64::MAX,
            max_time_us: 0,
            avg_time_us: 0,
            p95_time_us: 0,
            success_count: 0,
            error_count: 0,
            bytes_processed: 0,
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    pub fn record(&mut self, duration: Duration, bytes: u64, success: bool) {
        let time_us = duration.as_micros() as u64;
        self.call_count += 1;
        self.total_time_us += time_us;
        self.min_time_us = self.min_time_us.min(time_us);
        self.max_time_us = self.max_time_us.max(time_us);
        self.avg_time_us = self.total_time_us / self.call_count;
        self.bytes_processed += bytes;
        if success { self.success_count += 1; } else { self.error_count += 1; }
        self.last_updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }
}

impl PerfTimer {
    pub fn start(operation_name: &str) -> Self {
        Self { start_time: Instant::now(), operation_name: operation_name.to_string(), bytes_hint: None }
    }

    pub fn start_with_bytes(operation_name: &str, bytes: u64) -> Self {
        Self { start_time: Instant::now(), operation_name: operation_name.to_string(), bytes_hint: Some(bytes) }
    }

    pub fn finish_success(self) {
        let duration = self.start_time.elapsed();
        let bytes = self.bytes_hint.unwrap_or(0);
        PERF_MONITOR.profile_operation(&self.operation_name, duration, bytes, true);
    }

    pub fn finish_error(self) {
        let duration = self.start_time.elapsed();
        let bytes = self.bytes_hint.unwrap_or(0);
        PERF_MONITOR.profile_operation(&self.operation_name, duration, bytes, false);
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let bytes = self.bytes_hint.unwrap_or(0);
        PERF_MONITOR.profile_operation(&self.operation_name, duration, bytes, true);
    }
}

#[cfg(test)]
mod tests;

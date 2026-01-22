//! Performance metrics and benchmarking for scaling optimizations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Scaling performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScalingMetrics {
    pub files_processed: u64,
    pub total_processing_time: Duration,
    pub memory_peak: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub parallel_efficiency: f64,
    pub streaming_overhead: Duration,
}

impl ScalingMetrics {
    pub fn throughput(&self) -> f64 {
        if self.total_processing_time.as_secs_f64() > 0.0 {
            self.files_processed as f64 / self.total_processing_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Performance tracker for scaling operations
pub struct PerformanceTracker {
    start_time: Instant,
    metrics: ScalingMetrics,
    checkpoints: HashMap<String, Instant>,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: ScalingMetrics::default(),
            checkpoints: HashMap::new(),
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.insert(name.to_string(), Instant::now());
    }

    pub fn record_files_processed(&mut self, count: u64) {
        self.metrics.files_processed += count;
    }

    pub fn record_memory_peak(&mut self, memory: usize) {
        self.metrics.memory_peak = self.metrics.memory_peak.max(memory);
    }

    pub fn record_cache_hit(&mut self) {
        self.metrics.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.metrics.cache_misses += 1;
    }

    pub fn finish(mut self) -> ScalingMetrics {
        self.metrics.total_processing_time = self.start_time.elapsed();
        self.metrics
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark result for performance testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub duration: Duration,
    pub memory_usage: usize,
    pub throughput: f64,
    pub success_rate: f64,
}

impl BenchmarkResult {
    pub fn new(
        test_name: String,
        duration: Duration,
        memory_usage: usize,
        throughput: f64,
        success_rate: f64,
    ) -> Self {
        Self {
            test_name,
            duration,
            memory_usage,
            throughput,
            success_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_metrics_default() {
        let metrics = ScalingMetrics::default();
        assert_eq!(metrics.files_processed, 0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
        assert_eq!(metrics.memory_peak, 0);
    }

    #[test]
    fn test_scaling_metrics_throughput() {
        let metrics = ScalingMetrics {
            files_processed: 100,
            total_processing_time: Duration::from_secs(10),
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
            parallel_efficiency: 0.0,
            streaming_overhead: Duration::default(),
        };

        assert!((metrics.throughput() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_scaling_metrics_throughput_zero_time() {
        let metrics = ScalingMetrics {
            files_processed: 100,
            total_processing_time: Duration::ZERO,
            ..Default::default()
        };

        assert_eq!(metrics.throughput(), 0.0);
    }

    #[test]
    fn test_scaling_metrics_cache_hit_ratio() {
        let metrics = ScalingMetrics {
            files_processed: 0,
            total_processing_time: Duration::default(),
            memory_peak: 0,
            cache_hits: 80,
            cache_misses: 20,
            parallel_efficiency: 0.0,
            streaming_overhead: Duration::default(),
        };

        assert!((metrics.cache_hit_ratio() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_scaling_metrics_cache_hit_ratio_zero() {
        let metrics = ScalingMetrics::default();
        assert_eq!(metrics.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_scaling_metrics_serialize() {
        let metrics = ScalingMetrics {
            files_processed: 50,
            total_processing_time: Duration::from_millis(500),
            memory_peak: 1024,
            cache_hits: 30,
            cache_misses: 10,
            parallel_efficiency: 0.85,
            streaming_overhead: Duration::from_millis(10),
        };

        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: ScalingMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics.files_processed, deserialized.files_processed);
        assert_eq!(metrics.cache_hits, deserialized.cache_hits);
    }

    #[test]
    fn test_performance_tracker_new() {
        let tracker = PerformanceTracker::new();
        let metrics = tracker.finish();
        assert_eq!(metrics.files_processed, 0);
    }

    #[test]
    fn test_performance_tracker_default() {
        let tracker = PerformanceTracker::default();
        let metrics = tracker.finish();
        assert_eq!(metrics.files_processed, 0);
    }

    #[test]
    fn test_performance_tracker_record_files() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_files_processed(10);
        tracker.record_files_processed(5);
        let metrics = tracker.finish();
        assert_eq!(metrics.files_processed, 15);
    }

    #[test]
    fn test_performance_tracker_record_memory() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_memory_peak(1000);
        tracker.record_memory_peak(500); // Should not update
        tracker.record_memory_peak(2000); // Should update
        let metrics = tracker.finish();
        assert_eq!(metrics.memory_peak, 2000);
    }

    #[test]
    fn test_performance_tracker_record_cache() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_cache_hit();
        tracker.record_cache_hit();
        tracker.record_cache_miss();
        let metrics = tracker.finish();
        assert_eq!(metrics.cache_hits, 2);
        assert_eq!(metrics.cache_misses, 1);
    }

    #[test]
    fn test_performance_tracker_checkpoint() {
        let mut tracker = PerformanceTracker::new();
        tracker.checkpoint("start");
        std::thread::sleep(Duration::from_millis(1));
        tracker.checkpoint("end");
        // Checkpoints are stored but not directly exposed
        let metrics = tracker.finish();
        assert!(metrics.total_processing_time > Duration::ZERO);
    }

    #[test]
    fn test_benchmark_result_new() {
        let result = BenchmarkResult::new(
            "test_benchmark".to_string(),
            Duration::from_secs(5),
            1024,
            100.0,
            0.95,
        );

        assert_eq!(result.test_name, "test_benchmark");
        assert_eq!(result.duration, Duration::from_secs(5));
        assert_eq!(result.memory_usage, 1024);
        assert!((result.throughput - 100.0).abs() < 0.001);
        assert!((result.success_rate - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_result_serialize() {
        let result = BenchmarkResult::new(
            "serialize_test".to_string(),
            Duration::from_millis(100),
            512,
            50.0,
            1.0,
        );

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.test_name, deserialized.test_name);
        assert_eq!(result.memory_usage, deserialized.memory_usage);
    }

    #[test]
    fn test_benchmark_result_clone() {
        let result = BenchmarkResult::new(
            "clone_test".to_string(),
            Duration::from_secs(1),
            256,
            25.0,
            0.99,
        );

        let cloned = result.clone();
        assert_eq!(result.test_name, cloned.test_name);
        assert_eq!(result.throughput, cloned.throughput);
    }
}

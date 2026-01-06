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

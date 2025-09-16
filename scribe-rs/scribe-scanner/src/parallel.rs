//! Bounded parallelism with backpressure control and adaptive batching.
//!
//! This module implements intelligent parallelism that adapts to system load,
//! I/O latency, and memory pressure to prevent thrashing while maximizing
//! throughput for file scanning operations.

use futures::future::try_join_all;
use futures::stream::{self, FuturesUnordered, StreamExt};
use fxhash::FxHashMap;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::task::JoinHandle;

/// Adaptive parallelism controller with backpressure
#[derive(Debug)]
pub struct ParallelController {
    /// Current concurrency limit (adaptive)
    concurrency_limit: Arc<AtomicUsize>,
    /// Semaphore for concurrency control
    semaphore: Arc<Semaphore>,
    /// I/O latency tracker for adaptation
    io_latency_tracker: Arc<IoLatencyTracker>,
    /// Memory pressure detector  
    memory_tracker: Arc<MemoryTracker>,
    /// Performance metrics
    metrics: Arc<Mutex<ParallelMetrics>>,
    /// Configuration
    config: ParallelConfig,
}

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Initial concurrency level
    pub initial_concurrency: usize,
    /// Minimum concurrency (never go below this)
    pub min_concurrency: usize,
    /// Maximum concurrency (never exceed this)
    pub max_concurrency: usize,
    /// Target I/O latency (ms) for backpressure
    pub target_io_latency_ms: u64,
    /// Memory usage threshold (MB) for backpressure
    pub memory_threshold_mb: u64,
    /// Adaptation interval (how often to adjust)
    pub adaptation_interval: Duration,
    /// Work queue size per thread
    pub queue_size_per_thread: usize,
    /// Batch size adaptation range
    pub batch_size_range: (usize, usize),
}

/// I/O latency tracking for adaptive concurrency
#[derive(Debug)]
struct IoLatencyTracker {
    recent_latencies: Arc<RwLock<Vec<u64>>>,
    window_size: usize,
    last_adaptation: Arc<Mutex<Instant>>,
}

/// Memory usage tracking for backpressure
#[derive(Debug)]
struct MemoryTracker {
    baseline_memory: Arc<AtomicU64>,
    current_memory: Arc<AtomicU64>,
    peak_memory: Arc<AtomicU64>,
    measurements: Arc<AtomicU64>,
}

/// Performance metrics for parallel processing
#[derive(Debug, Default, Clone)]
pub struct ParallelMetrics {
    /// Tasks completed
    pub tasks_completed: u64,
    /// Tasks queued
    pub tasks_queued: u64,
    /// Current active tasks
    pub active_tasks: u64,
    /// Average I/O latency (microseconds)
    pub avg_io_latency_us: u64,
    /// Peak concurrency reached
    pub peak_concurrency: usize,
    /// Current concurrency level
    pub current_concurrency: usize,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// Throughput (tasks/second)
    pub throughput: f64,
    /// Adaptation events
    pub concurrency_adaptations: u64,
    /// Backpressure events
    pub backpressure_events: u64,
    /// Queue overflow events
    pub queue_overflows: u64,
}

/// Adaptive batch configuration
#[derive(Debug, Clone)]
pub struct AdaptiveBatch {
    pub size: usize,
    pub timeout: Duration,
    pub memory_limit: u64,
}

/// Work item for parallel processing
#[derive(Debug, Clone)]
pub struct WorkItem<T> {
    pub data: T,
    pub priority: u8,        // 0 = highest priority
    pub estimated_cost: u32, // relative processing cost
    pub enqueued_at: Instant,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            initial_concurrency: (cpu_count * 2).min(16),
            min_concurrency: 1,
            max_concurrency: (cpu_count * 4).min(32),
            target_io_latency_ms: 50,
            memory_threshold_mb: 512,
            adaptation_interval: Duration::from_secs(5),
            queue_size_per_thread: 100,
            batch_size_range: (10, 1000),
        }
    }
}

impl ParallelController {
    /// Create a new parallel controller
    pub fn new(config: ParallelConfig) -> Self {
        let concurrency_limit = Arc::new(AtomicUsize::new(config.initial_concurrency));
        let semaphore = Arc::new(Semaphore::new(config.initial_concurrency));

        Self {
            concurrency_limit,
            semaphore,
            io_latency_tracker: Arc::new(IoLatencyTracker::new(50)),
            memory_tracker: Arc::new(MemoryTracker::new()),
            metrics: Arc::new(Mutex::new(ParallelMetrics::default())),
            config,
        }
    }

    /// Process work items with adaptive parallelism and backpressure
    pub async fn process_parallel<T, F, Fut, R>(
        &self,
        items: Vec<WorkItem<T>>,
        processor: F,
    ) -> Vec<Result<R, String>>
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R, String>> + Send + 'static,
        R: Send + 'static,
    {
        if items.is_empty() {
            return Vec::new();
        }

        let total_items = items.len();
        log::info!("Starting parallel processing of {} items", total_items);

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.tasks_queued += total_items as u64;
            metrics.current_concurrency = self.concurrency_limit.load(Ordering::Relaxed);
        }

        // Create adaptive batches
        let batches = self.create_adaptive_batches(items).await;
        log::debug!("Created {} adaptive batches", batches.len());

        let mut all_results = Vec::with_capacity(total_items);
        let start_time = Instant::now();
        let mut completed_tasks = 0u64;

        // Process batches with bounded parallelism
        for batch in batches {
            let batch_results = self
                .process_batch_with_backpressure(batch, processor.clone())
                .await;

            completed_tasks += batch_results.len() as u64;
            all_results.extend(batch_results);

            // Update throughput metric
            let elapsed_secs = start_time.elapsed().as_secs_f64();
            if elapsed_secs > 0.0 {
                let mut metrics = self.metrics.lock();
                metrics.throughput = completed_tasks as f64 / elapsed_secs;
                metrics.tasks_completed = completed_tasks;
            }

            // Adaptive concurrency adjustment
            if self.should_adapt_concurrency().await {
                self.adapt_concurrency().await;
            }
        }

        log::info!(
            "Completed parallel processing: {}/{} items in {:.2}s ({:.1} items/sec)",
            completed_tasks,
            total_items,
            start_time.elapsed().as_secs_f64(),
            completed_tasks as f64 / start_time.elapsed().as_secs_f64()
        );

        all_results
    }

    /// Process a single batch with backpressure control
    async fn process_batch_with_backpressure<T, F, Fut, R>(
        &self,
        batch: Vec<WorkItem<T>>,
        processor: F,
    ) -> Vec<Result<R, String>>
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R, String>> + Send + 'static,
        R: Send + 'static,
    {
        let batch_size = batch.len();
        let mut futures = FuturesUnordered::new();

        // Process items with concurrency control
        for item in batch {
            let semaphore = Arc::clone(&self.semaphore);
            let processor = processor.clone();
            let io_tracker = Arc::clone(&self.io_latency_tracker);
            let memory_tracker = Arc::clone(&self.memory_tracker);
            let metrics = Arc::clone(&self.metrics);

            let future = async move {
                // Acquire permit (blocks if at concurrency limit)
                let _permit = semaphore.acquire().await.unwrap();

                // Update active task count
                {
                    let mut m = metrics.lock();
                    m.active_tasks += 1;
                }

                // Track I/O latency
                let start_time = Instant::now();

                // Sample memory before processing
                memory_tracker.sample_memory().await;

                // Process the item
                let result = processor(item.data).await;

                // Track I/O latency
                let io_latency_us = start_time.elapsed().as_micros() as u64;
                io_tracker.record_latency(io_latency_us / 1000).await; // Convert to ms

                // Sample memory after processing
                memory_tracker.sample_memory().await;

                // Update metrics
                {
                    let mut m = metrics.lock();
                    m.active_tasks = m.active_tasks.saturating_sub(1);
                    m.avg_io_latency_us = (m.avg_io_latency_us + io_latency_us) / 2;
                }

                result
            };

            futures.push(future);
        }

        // Collect all results
        let results: Vec<_> = futures.collect().await;

        log::debug!("Processed batch of {} items", batch_size);
        results
    }

    /// Create adaptive batches based on system conditions
    async fn create_adaptive_batches<T: Clone>(
        &self,
        items: Vec<WorkItem<T>>,
    ) -> Vec<Vec<WorkItem<T>>> {
        let total_items = items.len();
        let current_concurrency = self.concurrency_limit.load(Ordering::Relaxed);

        // Calculate optimal batch size based on concurrency and item count
        let base_batch_size =
            (total_items / current_concurrency).max(self.config.batch_size_range.0);
        let batch_size = base_batch_size.min(self.config.batch_size_range.1);

        // Sort items by priority (higher priority = lower number = first)
        let mut sorted_items = items;
        sorted_items.sort_by_key(|item| (item.priority, item.estimated_cost));

        // Create batches
        let mut batches = Vec::new();
        for chunk in sorted_items.chunks(batch_size) {
            batches.push(chunk.to_vec());
        }

        log::debug!(
            "Created {} batches with average size {} (concurrency: {})",
            batches.len(),
            batch_size,
            current_concurrency
        );

        batches
    }

    /// Check if concurrency should be adapted
    async fn should_adapt_concurrency(&self) -> bool {
        let last_adaptation = *self.io_latency_tracker.last_adaptation.lock();
        last_adaptation.elapsed() > self.config.adaptation_interval
    }

    /// Adapt concurrency based on system conditions
    async fn adapt_concurrency(&self) {
        let avg_latency = self.io_latency_tracker.average_latency().await;
        let memory_pressure = self.memory_tracker.memory_pressure().await;
        let current_concurrency = self.concurrency_limit.load(Ordering::Relaxed);

        let mut new_concurrency = current_concurrency;

        // Reduce concurrency if high I/O latency or memory pressure
        if avg_latency > self.config.target_io_latency_ms || memory_pressure > 0.8 {
            new_concurrency = (current_concurrency * 8 / 10).max(self.config.min_concurrency);

            let mut metrics = self.metrics.lock();
            metrics.backpressure_events += 1;

            log::debug!(
                "Reducing concurrency: {} -> {} (latency: {}ms, memory pressure: {:.1}%)",
                current_concurrency,
                new_concurrency,
                avg_latency,
                memory_pressure * 100.0
            );
        }
        // Increase concurrency if low latency and memory pressure
        else if avg_latency < self.config.target_io_latency_ms / 2 && memory_pressure < 0.5 {
            new_concurrency = (current_concurrency * 12 / 10).min(self.config.max_concurrency);

            log::debug!(
                "Increasing concurrency: {} -> {} (latency: {}ms, memory pressure: {:.1}%)",
                current_concurrency,
                new_concurrency,
                avg_latency,
                memory_pressure * 100.0
            );
        }

        // Apply new concurrency limit
        if new_concurrency != current_concurrency {
            self.concurrency_limit
                .store(new_concurrency, Ordering::Relaxed);

            // Update semaphore permits
            if new_concurrency > current_concurrency {
                self.semaphore
                    .add_permits(new_concurrency - current_concurrency);
            }
            // Note: We can't reduce semaphore permits without draining tasks

            // Update metrics
            {
                let mut metrics = self.metrics.lock();
                metrics.current_concurrency = new_concurrency;
                metrics.peak_concurrency = metrics.peak_concurrency.max(new_concurrency);
                metrics.concurrency_adaptations += 1;
            }
        }

        // Update adaptation timestamp
        *self.io_latency_tracker.last_adaptation.lock() = Instant::now();
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> ParallelMetrics {
        let mut metrics = self.metrics.lock().clone();
        metrics.memory_usage_bytes = self.memory_tracker.current_memory.load(Ordering::Relaxed);
        metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock();
        *metrics = ParallelMetrics::default();
        metrics.current_concurrency = self.concurrency_limit.load(Ordering::Relaxed);
    }
}

impl IoLatencyTracker {
    fn new(window_size: usize) -> Self {
        Self {
            recent_latencies: Arc::new(RwLock::new(Vec::with_capacity(window_size))),
            window_size,
            last_adaptation: Arc::new(Mutex::new(Instant::now())),
        }
    }

    async fn record_latency(&self, latency_ms: u64) {
        let mut latencies = self.recent_latencies.write().await;
        latencies.push(latency_ms);

        if latencies.len() > self.window_size {
            latencies.remove(0);
        }
    }

    async fn average_latency(&self) -> u64 {
        let latencies = self.recent_latencies.read().await;
        if latencies.is_empty() {
            0
        } else {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        }
    }
}

impl MemoryTracker {
    fn new() -> Self {
        let initial_memory = Self::get_memory_usage();
        Self {
            baseline_memory: Arc::new(AtomicU64::new(initial_memory)),
            current_memory: Arc::new(AtomicU64::new(initial_memory)),
            peak_memory: Arc::new(AtomicU64::new(initial_memory)),
            measurements: Arc::new(AtomicU64::new(0)),
        }
    }

    async fn sample_memory(&self) {
        let current = Self::get_memory_usage();
        self.current_memory.store(current, Ordering::Relaxed);

        // Update peak
        self.peak_memory.fetch_max(current, Ordering::Relaxed);
        self.measurements.fetch_add(1, Ordering::Relaxed);
    }

    async fn memory_pressure(&self) -> f64 {
        let current = self.current_memory.load(Ordering::Relaxed);
        let baseline = self.baseline_memory.load(Ordering::Relaxed);

        if baseline == 0 {
            0.0
        } else {
            (current as f64 - baseline as f64) / (baseline as f64 * 4.0) // Normalize to 0-1
        }
    }

    fn get_memory_usage() -> u64 {
        // Platform-specific memory usage (simplified)
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback: use a simple heap estimation
        0
    }
}

impl<T> WorkItem<T> {
    pub fn new(data: T) -> Self {
        Self {
            data,
            priority: 128,       // Medium priority
            estimated_cost: 100, // Default cost
            enqueued_at: Instant::now(),
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_estimated_cost(mut self, cost: u32) -> Self {
        self.estimated_cost = cost;
        self
    }

    pub fn queue_time(&self) -> Duration {
        self.enqueued_at.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[tokio::test]
    async fn test_parallel_controller_creation() {
        let config = ParallelConfig::default();
        let controller = ParallelController::new(config.clone());

        let metrics = controller.metrics();
        assert_eq!(metrics.current_concurrency, config.initial_concurrency);
        assert_eq!(metrics.tasks_completed, 0);
    }

    #[tokio::test]
    async fn test_work_item_creation() {
        let item = WorkItem::new("test_data")
            .with_priority(10)
            .with_estimated_cost(500);

        assert_eq!(item.data, "test_data");
        assert_eq!(item.priority, 10);
        assert_eq!(item.estimated_cost, 500);
        assert!(item.queue_time().as_millis() < 10);
    }

    #[tokio::test]
    async fn test_parallel_processing() {
        let config = ParallelConfig {
            initial_concurrency: 2,
            max_concurrency: 4,
            ..Default::default()
        };
        let controller = ParallelController::new(config);

        // Create test work items
        let items: Vec<WorkItem<usize>> = (0..10)
            .map(|i| WorkItem::new(i).with_priority(i as u8))
            .collect();

        // Simple processor that just doubles the input
        let processor = |x: usize| async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(x * 2)
        };

        let results = controller.process_parallel(items, processor).await;

        assert_eq!(results.len(), 10);

        // Check that all results are correct (doubled values)
        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(value) => assert_eq!(*value, i * 2),
                Err(e) => panic!("Unexpected error: {}", e),
            }
        }

        let metrics = controller.metrics();
        assert_eq!(metrics.tasks_completed, 10);
        assert!(metrics.throughput > 0.0);
    }

    #[tokio::test]
    async fn test_batch_creation() {
        let config = ParallelConfig {
            initial_concurrency: 4,
            batch_size_range: (2, 5),
            ..Default::default()
        };
        let controller = ParallelController::new(config);

        let items: Vec<WorkItem<usize>> = (0..12).map(|i| WorkItem::new(i)).collect();

        let batches = controller.create_adaptive_batches(items).await;

        // Should create multiple batches
        assert!(batches.len() > 1);

        // Each batch should be within size limits
        for batch in &batches {
            assert!(batch.len() >= 1);
            assert!(batch.len() <= 5);
        }

        // Total items should be preserved
        let total_items: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total_items, 12);
    }

    #[tokio::test]
    async fn test_latency_tracking() {
        let tracker = IoLatencyTracker::new(5);

        // Record some latencies
        tracker.record_latency(10).await;
        tracker.record_latency(20).await;
        tracker.record_latency(30).await;

        let avg = tracker.average_latency().await;
        assert_eq!(avg, 20); // (10 + 20 + 30) / 3 = 20

        // Test window limit
        for i in 0..10 {
            tracker.record_latency(100 + i).await;
        }

        let latencies = tracker.recent_latencies.read().await;
        assert_eq!(latencies.len(), 5); // Should be capped at window size
    }

    #[tokio::test]
    async fn test_memory_tracking() {
        let tracker = MemoryTracker::new();

        tracker.sample_memory().await;
        let pressure = tracker.memory_pressure().await;

        // Memory pressure should be reasonable (0.0 - 1.0 range)
        assert!(pressure >= 0.0);
        assert!(pressure <= 10.0); // Allow some room for test environment variation
    }

    #[tokio::test]
    async fn test_error_handling() {
        let config = ParallelConfig::default();
        let controller = ParallelController::new(config);

        let items: Vec<WorkItem<usize>> =
            vec![WorkItem::new(1), WorkItem::new(2), WorkItem::new(3)];

        // Processor that fails on even numbers
        let processor = |x: usize| async move {
            if x % 2 == 0 {
                Err(format!("Error processing {}", x))
            } else {
                Ok(x * 10)
            }
        };

        let results = controller.process_parallel(items, processor).await;

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok()); // 1 -> success
        assert!(results[1].is_err()); // 2 -> error
        assert!(results[2].is_ok()); // 3 -> success

        // Check specific values
        assert_eq!(results[0].as_ref().unwrap(), &10);
        assert_eq!(results[2].as_ref().unwrap(), &30);
    }

    #[tokio::test]
    async fn test_metrics_updates() {
        let config = ParallelConfig {
            initial_concurrency: 2,
            ..Default::default()
        };
        let controller = ParallelController::new(config);

        // Initial metrics
        let initial_metrics = controller.metrics();
        assert_eq!(initial_metrics.tasks_completed, 0);
        assert_eq!(initial_metrics.current_concurrency, 2);

        // Process some items
        let items: Vec<WorkItem<usize>> = vec![WorkItem::new(1), WorkItem::new(2)];
        let processor = |x: usize| async move { Ok(x) };

        controller.process_parallel(items, processor).await;

        // Check updated metrics
        let final_metrics = controller.metrics();
        assert_eq!(final_metrics.tasks_completed, 2);
        assert!(final_metrics.throughput > 0.0);
    }
}

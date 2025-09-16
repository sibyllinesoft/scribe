//! Parallel processing configuration for scaling operations.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: usize,

    /// Number of async workers
    pub async_worker_count: usize,

    /// Number of CPU-bound workers
    pub cpu_worker_count: usize,

    /// Timeout for individual tasks
    pub task_timeout: Duration,

    /// Enable work-stealing between workers
    pub enable_work_stealing: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: num_cpus::get() * 2,
            async_worker_count: num_cpus::get(),
            cpu_worker_count: num_cpus::get(),
            task_timeout: Duration::from_secs(30),
            enable_work_stealing: true,
        }
    }
}

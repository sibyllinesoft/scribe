//! Performance optimization modules.

// Note: compact_data.rs, incremental.rs, optimized_scanner.rs are not compiled
// They have unresolved dependencies and were not declared in original lib.rs
pub mod parallel;
pub mod performance;

pub use parallel::{ParallelConfig, ParallelController, ParallelMetrics, WorkItem};
pub use performance::{
    ErrorType, PerfTimer, PerformanceMonitor, PerformanceReport, PerformanceSnapshot, PERF_MONITOR,
};

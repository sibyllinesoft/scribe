//! Performance optimization modules.

pub mod parallel;
pub mod performance;

pub use parallel::{ParallelConfig, ParallelController, ParallelMetrics, WorkItem};
pub use performance::{
    ErrorType, PerfTimer, PerformanceMonitor, PerformanceReport, PerformanceSnapshot, PERF_MONITOR,
};

//! Adaptive configuration system that adjusts thresholds based on repository characteristics.

use serde::{Deserialize, Serialize};

/// Configuration for adaptive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Whether to enable adaptive thresholds
    pub enable_adaptive_thresholds: bool,

    /// Factor for repository size adjustment
    pub repository_size_factor: f64,

    /// Factor for memory pressure adjustment
    pub memory_pressure_factor: f64,

    /// Factor for CPU utilization adjustment
    pub cpu_utilization_factor: f64,

    /// Weight for performance feedback
    pub performance_feedback_weight: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_thresholds: true,
            repository_size_factor: 1.0,
            memory_pressure_factor: 0.8,
            cpu_utilization_factor: 0.9,
            performance_feedback_weight: 0.2,
        }
    }
}

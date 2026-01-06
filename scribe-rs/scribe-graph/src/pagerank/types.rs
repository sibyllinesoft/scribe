//! Type definitions for PageRank computation.

use scribe_core::{error::ScribeError, Result};
use serde::{Deserialize, Serialize};

/// PageRank algorithm configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageRankConfig {
    /// Damping factor (probability of following edges vs random jump)
    pub damping_factor: f64,

    /// Maximum number of iterations before stopping
    pub max_iterations: usize,

    /// Convergence threshold (L1 norm difference between iterations)
    pub epsilon: f64,

    /// Whether to use parallel computation
    pub use_parallel: bool,

    /// Minimum score threshold for results (filter noise)
    pub min_score_threshold: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,      // Research standard for web graphs
            max_iterations: 50,        // Sufficient for most dependency graphs
            epsilon: 1e-6,             // High precision convergence
            use_parallel: true,        // Enable parallel computation by default
            min_score_threshold: 1e-8, // Filter very low scores
        }
    }
}

impl PageRankConfig {
    /// Create configuration optimized for code dependency analysis
    pub fn for_code_analysis() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 30,
            epsilon: 1e-5,
            use_parallel: true,
            min_score_threshold: 1e-6,
        }
    }

    /// Create configuration for large codebases (>10k files)
    pub fn for_large_codebases() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 20,
            epsilon: 1e-4,
            use_parallel: true,
            min_score_threshold: 1e-5,
        }
    }

    /// Create configuration for high-precision research analysis
    pub fn for_research() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            epsilon: 1e-8,
            use_parallel: true,
            min_score_threshold: 0.0,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.damping_factor < 0.0 || self.damping_factor >= 1.0 {
            return Err(ScribeError::invalid_operation(
                "Damping factor must be in range [0, 1)".to_string(),
                "pagerank_config_validation".to_string(),
            ));
        }

        if self.max_iterations == 0 {
            return Err(ScribeError::invalid_operation(
                "Max iterations must be greater than 0".to_string(),
                "pagerank_config_validation".to_string(),
            ));
        }

        if self.epsilon <= 0.0 {
            return Err(ScribeError::invalid_operation(
                "Epsilon must be positive".to_string(),
                "pagerank_config_validation".to_string(),
            ));
        }

        Ok(())
    }
}

/// Performance metrics for PageRank computation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total computation time in milliseconds
    pub total_time_ms: u64,

    /// Time per iteration in milliseconds
    pub avg_iteration_time_ms: f64,

    /// Peak memory usage in MB (estimated)
    pub peak_memory_mb: f64,

    /// Number of nodes processed
    pub nodes_processed: usize,

    /// Convergence rate (improvement per iteration)
    pub convergence_rate: f64,

    /// Whether parallel computation was used
    pub used_parallel: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time_ms: 0,
            avg_iteration_time_ms: 0.0,
            peak_memory_mb: 0.0,
            nodes_processed: 0,
            convergence_rate: 0.0,
            used_parallel: false,
        }
    }
}

/// Statistical information about PageRank scores
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min_score: f64,
    pub max_score: f64,
    pub total_nodes: usize,
}

impl Default for ScoreStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min_score: 0.0,
            max_score: 0.0,
            total_nodes: 0,
        }
    }
}

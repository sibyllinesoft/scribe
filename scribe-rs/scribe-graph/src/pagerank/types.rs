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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_config_default() {
        let config = PageRankConfig::default();
        assert_eq!(config.damping_factor, 0.85);
        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.epsilon, 1e-6);
        assert!(config.use_parallel);
        assert_eq!(config.min_score_threshold, 1e-8);
    }

    #[test]
    fn test_pagerank_config_for_code_analysis() {
        let config = PageRankConfig::for_code_analysis();
        assert_eq!(config.damping_factor, 0.85);
        assert_eq!(config.max_iterations, 30);
        assert_eq!(config.epsilon, 1e-5);
        assert!(config.use_parallel);
        assert_eq!(config.min_score_threshold, 1e-6);
    }

    #[test]
    fn test_pagerank_config_for_large_codebases() {
        let config = PageRankConfig::for_large_codebases();
        assert_eq!(config.damping_factor, 0.85);
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.epsilon, 1e-4);
        assert!(config.use_parallel);
        assert_eq!(config.min_score_threshold, 1e-5);
    }

    #[test]
    fn test_pagerank_config_for_research() {
        let config = PageRankConfig::for_research();
        assert_eq!(config.damping_factor, 0.85);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.epsilon, 1e-8);
        assert!(config.use_parallel);
        assert_eq!(config.min_score_threshold, 0.0);
    }

    #[test]
    fn test_pagerank_config_validate_valid() {
        let config = PageRankConfig::default();
        assert!(config.validate().is_ok());

        let config = PageRankConfig::for_code_analysis();
        assert!(config.validate().is_ok());

        let config = PageRankConfig::for_large_codebases();
        assert!(config.validate().is_ok());

        let config = PageRankConfig::for_research();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pagerank_config_validate_invalid_damping() {
        let config = PageRankConfig {
            damping_factor: -0.1,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = PageRankConfig {
            damping_factor: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = PageRankConfig {
            damping_factor: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pagerank_config_validate_zero_iterations() {
        let config = PageRankConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pagerank_config_validate_invalid_epsilon() {
        let config = PageRankConfig {
            epsilon: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = PageRankConfig {
            epsilon: -1e-6,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pagerank_config_clone() {
        let config = PageRankConfig::default();
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_pagerank_config_partial_eq() {
        let config1 = PageRankConfig::default();
        let config2 = PageRankConfig::default();
        let config3 = PageRankConfig::for_code_analysis();

        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_pagerank_config_serialize() {
        let config = PageRankConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PageRankConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_pagerank_config_debug() {
        let config = PageRankConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("PageRankConfig"));
        assert!(debug.contains("damping_factor"));
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.total_time_ms, 0);
        assert_eq!(metrics.avg_iteration_time_ms, 0.0);
        assert_eq!(metrics.peak_memory_mb, 0.0);
        assert_eq!(metrics.nodes_processed, 0);
        assert_eq!(metrics.convergence_rate, 0.0);
        assert!(!metrics.used_parallel);
    }

    #[test]
    fn test_performance_metrics_clone() {
        let metrics = PerformanceMetrics {
            total_time_ms: 100,
            avg_iteration_time_ms: 5.0,
            peak_memory_mb: 50.0,
            nodes_processed: 1000,
            convergence_rate: 0.95,
            used_parallel: true,
        };
        let cloned = metrics.clone();
        assert_eq!(metrics, cloned);
    }

    #[test]
    fn test_performance_metrics_partial_eq() {
        let metrics1 = PerformanceMetrics {
            total_time_ms: 100,
            avg_iteration_time_ms: 5.0,
            peak_memory_mb: 50.0,
            nodes_processed: 1000,
            convergence_rate: 0.95,
            used_parallel: true,
        };
        let metrics2 = PerformanceMetrics {
            total_time_ms: 100,
            avg_iteration_time_ms: 5.0,
            peak_memory_mb: 50.0,
            nodes_processed: 1000,
            convergence_rate: 0.95,
            used_parallel: true,
        };
        let metrics3 = PerformanceMetrics::default();

        assert_eq!(metrics1, metrics2);
        assert_ne!(metrics1, metrics3);
    }

    #[test]
    fn test_performance_metrics_serialize() {
        let metrics = PerformanceMetrics {
            total_time_ms: 500,
            avg_iteration_time_ms: 10.0,
            peak_memory_mb: 25.0,
            nodes_processed: 500,
            convergence_rate: 0.8,
            used_parallel: false,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics, deserialized);
    }

    #[test]
    fn test_performance_metrics_debug() {
        let metrics = PerformanceMetrics::default();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("PerformanceMetrics"));
        assert!(debug.contains("total_time_ms"));
    }

    #[test]
    fn test_score_statistics_default() {
        let stats = ScoreStatistics::default();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.median, 0.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.min_score, 0.0);
        assert_eq!(stats.max_score, 0.0);
        assert_eq!(stats.total_nodes, 0);
    }

    #[test]
    fn test_score_statistics_clone() {
        let stats = ScoreStatistics {
            mean: 0.5,
            median: 0.45,
            std_dev: 0.1,
            min_score: 0.1,
            max_score: 0.9,
            total_nodes: 100,
        };
        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[test]
    fn test_score_statistics_partial_eq() {
        let stats1 = ScoreStatistics {
            mean: 0.5,
            median: 0.5,
            std_dev: 0.2,
            min_score: 0.0,
            max_score: 1.0,
            total_nodes: 50,
        };
        let stats2 = ScoreStatistics {
            mean: 0.5,
            median: 0.5,
            std_dev: 0.2,
            min_score: 0.0,
            max_score: 1.0,
            total_nodes: 50,
        };
        let stats3 = ScoreStatistics::default();

        assert_eq!(stats1, stats2);
        assert_ne!(stats1, stats3);
    }

    #[test]
    fn test_score_statistics_serialize() {
        let stats = ScoreStatistics {
            mean: 0.3,
            median: 0.25,
            std_dev: 0.15,
            min_score: 0.05,
            max_score: 0.75,
            total_nodes: 200,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: ScoreStatistics = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_score_statistics_debug() {
        let stats = ScoreStatistics::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ScoreStatistics"));
        assert!(debug.contains("mean"));
        assert!(debug.contains("median"));
    }

    #[test]
    fn test_pagerank_config_boundary_damping() {
        // Edge case: damping factor at 0.0 is valid
        let config = PageRankConfig {
            damping_factor: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // Edge case: damping factor very close to 1.0 is valid
        let config = PageRankConfig {
            damping_factor: 0.9999,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pagerank_config_min_iterations() {
        // Minimum valid iterations is 1
        let config = PageRankConfig {
            max_iterations: 1,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pagerank_config_small_epsilon() {
        // Very small but positive epsilon should be valid
        let config = PageRankConfig {
            epsilon: 1e-15,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}

//! Public API and high-level engine components.

pub mod adaptive;
pub mod caching;
pub mod engine;
pub mod profiling;

pub use adaptive::AdaptiveConfig;
pub use caching::{compute_config_hash, compute_repository_hash, CacheConfig, ProcessingCache};
pub use engine::{ProcessingResult, ScalingConfig, ScalingEngine};
pub use profiling::{RepositoryProfile, RepositoryProfiler, RepositoryType};

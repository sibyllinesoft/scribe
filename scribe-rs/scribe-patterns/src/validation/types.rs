//! Type definitions for pattern validation.

use thiserror::Error;

/// Validation errors for patterns and configurations
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Invalid glob pattern '{pattern}': {reason}")]
    InvalidGlobPattern { pattern: String, reason: String },

    #[error("Invalid gitignore pattern '{pattern}': {reason}")]
    InvalidGitignorePattern { pattern: String, reason: String },

    #[error("Pattern too complex: {reason}")]
    PatternTooComplex { reason: String },

    #[error("Conflicting patterns detected: {conflict}")]
    ConflictingPatterns { conflict: String },

    #[error("Invalid path '{path}': {reason}")]
    InvalidPath { path: String, reason: String },

    #[error("Pattern limit exceeded: maximum {max} patterns allowed, got {actual}")]
    PatternLimitExceeded { max: usize, actual: usize },

    #[error("Empty pattern not allowed")]
    EmptyPattern,

    #[error("Regex compilation failed for pattern '{pattern}': {source}")]
    RegexError {
        pattern: String,
        #[source]
        source: regex::Error,
    },

    #[error("IO error while validating path '{path}': {source}")]
    IoError {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Result type for validation operations
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Configuration for pattern validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum number of patterns allowed
    pub max_patterns: usize,
    /// Maximum pattern length
    pub max_pattern_length: usize,
    /// Maximum nesting depth for glob patterns
    pub max_glob_depth: usize,
    /// Whether to allow empty patterns
    pub allow_empty_patterns: bool,
    /// Whether to validate paths exist
    pub validate_path_existence: bool,
    /// Whether to check for pattern conflicts
    pub check_conflicts: bool,
    /// Maximum time to spend on validation (milliseconds)
    pub max_validation_time_ms: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_patterns: 1000,
            max_pattern_length: 2048,
            max_glob_depth: 20,
            allow_empty_patterns: false,
            validate_path_existence: false,
            check_conflicts: true,
            max_validation_time_ms: 5000,
        }
    }
}

/// Performance risk assessment for patterns
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceRisk {
    pub level: PerformanceRiskLevel,
    pub score: u32,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Performance risk levels
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceRiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl PerformanceRiskLevel {
    /// Check if the risk level requires attention
    pub fn needs_attention(&self) -> bool {
        matches!(
            self,
            PerformanceRiskLevel::High | PerformanceRiskLevel::Critical
        )
    }

    /// Check if the risk level should prevent pattern usage
    pub fn should_reject(&self) -> bool {
        matches!(self, PerformanceRiskLevel::Critical)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_invalid_glob_pattern() {
        let error = ValidationError::InvalidGlobPattern {
            pattern: "***/foo".to_string(),
            reason: "too many wildcards".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Invalid glob pattern"));
        assert!(msg.contains("***/foo"));
        assert!(msg.contains("too many wildcards"));
    }

    #[test]
    fn test_validation_error_invalid_gitignore_pattern() {
        let error = ValidationError::InvalidGitignorePattern {
            pattern: "![".to_string(),
            reason: "unclosed bracket".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Invalid gitignore pattern"));
        assert!(msg.contains("!["));
    }

    #[test]
    fn test_validation_error_pattern_too_complex() {
        let error = ValidationError::PatternTooComplex {
            reason: "excessive nesting".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Pattern too complex"));
        assert!(msg.contains("excessive nesting"));
    }

    #[test]
    fn test_validation_error_conflicting_patterns() {
        let error = ValidationError::ConflictingPatterns {
            conflict: "*.rs conflicts with !src/*.rs".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Conflicting patterns"));
    }

    #[test]
    fn test_validation_error_invalid_path() {
        let error = ValidationError::InvalidPath {
            path: "/nonexistent".to_string(),
            reason: "path does not exist".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Invalid path"));
        assert!(msg.contains("/nonexistent"));
    }

    #[test]
    fn test_validation_error_pattern_limit_exceeded() {
        let error = ValidationError::PatternLimitExceeded {
            max: 100,
            actual: 150,
        };
        let msg = error.to_string();
        assert!(msg.contains("Pattern limit exceeded"));
        assert!(msg.contains("100"));
        assert!(msg.contains("150"));
    }

    #[test]
    fn test_validation_error_empty_pattern() {
        let error = ValidationError::EmptyPattern;
        let msg = error.to_string();
        assert!(msg.contains("Empty pattern"));
    }

    #[test]
    fn test_validation_error_regex_error() {
        let regex_err = regex::Regex::new("[invalid").unwrap_err();
        let error = ValidationError::RegexError {
            pattern: "[invalid".to_string(),
            source: regex_err,
        };
        let msg = error.to_string();
        assert!(msg.contains("Regex compilation failed"));
        assert!(msg.contains("[invalid"));
    }

    #[test]
    fn test_validation_error_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let error = ValidationError::IoError {
            path: "/some/path".to_string(),
            source: io_err,
        };
        let msg = error.to_string();
        assert!(msg.contains("IO error"));
        assert!(msg.contains("/some/path"));
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();

        assert_eq!(config.max_patterns, 1000);
        assert_eq!(config.max_pattern_length, 2048);
        assert_eq!(config.max_glob_depth, 20);
        assert!(!config.allow_empty_patterns);
        assert!(!config.validate_path_existence);
        assert!(config.check_conflicts);
        assert_eq!(config.max_validation_time_ms, 5000);
    }

    #[test]
    fn test_validation_config_custom() {
        let config = ValidationConfig {
            max_patterns: 500,
            max_pattern_length: 1024,
            max_glob_depth: 10,
            allow_empty_patterns: true,
            validate_path_existence: true,
            check_conflicts: false,
            max_validation_time_ms: 10000,
        };

        assert_eq!(config.max_patterns, 500);
        assert_eq!(config.max_pattern_length, 1024);
        assert_eq!(config.max_glob_depth, 10);
        assert!(config.allow_empty_patterns);
        assert!(config.validate_path_existence);
        assert!(!config.check_conflicts);
        assert_eq!(config.max_validation_time_ms, 10000);
    }

    #[test]
    fn test_validation_config_clone() {
        let config = ValidationConfig::default();
        let cloned = config.clone();

        assert_eq!(config.max_patterns, cloned.max_patterns);
        assert_eq!(config.max_pattern_length, cloned.max_pattern_length);
        assert_eq!(config.check_conflicts, cloned.check_conflicts);
    }

    #[test]
    fn test_validation_config_debug() {
        let config = ValidationConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("ValidationConfig"));
        assert!(debug_str.contains("max_patterns"));
        assert!(debug_str.contains("1000"));
    }

    #[test]
    fn test_performance_risk_creation() {
        let risk = PerformanceRisk {
            level: PerformanceRiskLevel::Low,
            score: 10,
            issues: vec![],
            recommendations: vec![],
        };

        assert_eq!(risk.level, PerformanceRiskLevel::Low);
        assert_eq!(risk.score, 10);
        assert!(risk.issues.is_empty());
        assert!(risk.recommendations.is_empty());
    }

    #[test]
    fn test_performance_risk_with_issues() {
        let risk = PerformanceRisk {
            level: PerformanceRiskLevel::High,
            score: 75,
            issues: vec![
                "Excessive wildcard usage".to_string(),
                "Deep nesting detected".to_string(),
            ],
            recommendations: vec![
                "Simplify patterns".to_string(),
                "Reduce nesting depth".to_string(),
            ],
        };

        assert_eq!(risk.level, PerformanceRiskLevel::High);
        assert_eq!(risk.score, 75);
        assert_eq!(risk.issues.len(), 2);
        assert_eq!(risk.recommendations.len(), 2);
        assert!(risk.issues[0].contains("wildcard"));
    }

    #[test]
    fn test_performance_risk_clone() {
        let risk = PerformanceRisk {
            level: PerformanceRiskLevel::Medium,
            score: 50,
            issues: vec!["issue1".to_string()],
            recommendations: vec!["rec1".to_string()],
        };

        let cloned = risk.clone();
        assert_eq!(risk.level, cloned.level);
        assert_eq!(risk.score, cloned.score);
        assert_eq!(risk.issues, cloned.issues);
    }

    #[test]
    fn test_performance_risk_level_equality() {
        assert_eq!(PerformanceRiskLevel::Low, PerformanceRiskLevel::Low);
        assert_eq!(PerformanceRiskLevel::Medium, PerformanceRiskLevel::Medium);
        assert_eq!(PerformanceRiskLevel::High, PerformanceRiskLevel::High);
        assert_eq!(
            PerformanceRiskLevel::Critical,
            PerformanceRiskLevel::Critical
        );

        assert_ne!(PerformanceRiskLevel::Low, PerformanceRiskLevel::High);
        assert_ne!(PerformanceRiskLevel::Medium, PerformanceRiskLevel::Critical);
    }

    #[test]
    fn test_performance_risk_level_needs_attention() {
        assert!(!PerformanceRiskLevel::Low.needs_attention());
        assert!(!PerformanceRiskLevel::Medium.needs_attention());
        assert!(PerformanceRiskLevel::High.needs_attention());
        assert!(PerformanceRiskLevel::Critical.needs_attention());
    }

    #[test]
    fn test_performance_risk_level_should_reject() {
        assert!(!PerformanceRiskLevel::Low.should_reject());
        assert!(!PerformanceRiskLevel::Medium.should_reject());
        assert!(!PerformanceRiskLevel::High.should_reject());
        assert!(PerformanceRiskLevel::Critical.should_reject());
    }

    #[test]
    fn test_performance_risk_level_clone() {
        let level = PerformanceRiskLevel::High;
        let cloned = level.clone();
        assert_eq!(level, cloned);
    }

    #[test]
    fn test_performance_risk_level_debug() {
        let debug_low = format!("{:?}", PerformanceRiskLevel::Low);
        assert_eq!(debug_low, "Low");

        let debug_medium = format!("{:?}", PerformanceRiskLevel::Medium);
        assert_eq!(debug_medium, "Medium");

        let debug_high = format!("{:?}", PerformanceRiskLevel::High);
        assert_eq!(debug_high, "High");

        let debug_critical = format!("{:?}", PerformanceRiskLevel::Critical);
        assert_eq!(debug_critical, "Critical");
    }

    #[test]
    fn test_performance_risk_equality() {
        let risk1 = PerformanceRisk {
            level: PerformanceRiskLevel::Low,
            score: 10,
            issues: vec![],
            recommendations: vec![],
        };

        let risk2 = PerformanceRisk {
            level: PerformanceRiskLevel::Low,
            score: 10,
            issues: vec![],
            recommendations: vec![],
        };

        assert_eq!(risk1, risk2);
    }

    #[test]
    fn test_performance_risk_inequality() {
        let risk1 = PerformanceRisk {
            level: PerformanceRiskLevel::Low,
            score: 10,
            issues: vec![],
            recommendations: vec![],
        };

        let risk2 = PerformanceRisk {
            level: PerformanceRiskLevel::High,
            score: 75,
            issues: vec![],
            recommendations: vec![],
        };

        assert_ne!(risk1, risk2);
    }

    #[test]
    fn test_validation_error_debug() {
        let error = ValidationError::EmptyPattern;
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("EmptyPattern"));
    }
}

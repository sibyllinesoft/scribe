use regex::Regex;
use std::collections::HashSet;
use std::path::Path;
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

/// Pattern validator with comprehensive validation rules
pub struct PatternValidator {
    config: ValidationConfig,
    glob_regex: Regex,
    dangerous_patterns: HashSet<String>,
}

impl PatternValidator {
    /// Create a new pattern validator
    pub fn new(config: ValidationConfig) -> ValidationResult<Self> {
        // Regex to detect potentially dangerous glob patterns
        let glob_regex = Regex::new(r"[\*\?\[\]{}]").map_err(|e| ValidationError::RegexError {
            pattern: r"[\*\?\[\]{}]".to_string(),
            source: e,
        })?;

        // Known dangerous or problematic patterns
        let mut dangerous_patterns = HashSet::new();
        dangerous_patterns.insert("**/*/**/*/**/*/**/*/**".to_string()); // Too many recursive wildcards
        dangerous_patterns.insert("*".repeat(100)); // Too many consecutive wildcards
        dangerous_patterns.insert("?".repeat(100)); // Too many consecutive single wildcards

        Ok(Self {
            config,
            glob_regex,
            dangerous_patterns,
        })
    }

    /// Create a validator with default configuration
    pub fn default() -> ValidationResult<Self> {
        Self::new(ValidationConfig::default())
    }

    /// Validate a single glob pattern
    pub fn validate_glob_pattern(&self, pattern: &str) -> ValidationResult<()> {
        // Check for empty pattern
        if pattern.is_empty() && !self.config.allow_empty_patterns {
            return Err(ValidationError::EmptyPattern);
        }

        // Check pattern length
        if pattern.len() > self.config.max_pattern_length {
            return Err(ValidationError::InvalidGlobPattern {
                pattern: pattern.to_string(),
                reason: format!(
                    "Pattern too long: {} characters (max: {})",
                    pattern.len(),
                    self.config.max_pattern_length
                ),
            });
        }

        // Check for dangerous patterns
        if self.dangerous_patterns.contains(pattern) {
            return Err(ValidationError::PatternTooComplex {
                reason: "Pattern is known to cause performance issues".to_string(),
            });
        }

        // Validate glob syntax
        self.validate_glob_syntax(pattern)?;

        // Check nesting depth
        self.validate_glob_depth(pattern)?;

        // Check for problematic character sequences
        self.validate_glob_sequences(pattern)?;

        Ok(())
    }

    /// Validate glob pattern syntax
    fn validate_glob_syntax(&self, pattern: &str) -> ValidationResult<()> {
        let mut bracket_depth = 0;
        let mut brace_depth = 0;
        let mut chars = pattern.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '[' => {
                    bracket_depth += 1;
                    if bracket_depth > 1 {
                        return Err(ValidationError::InvalidGlobPattern {
                            pattern: pattern.to_string(),
                            reason: "Nested character classes not allowed".to_string(),
                        });
                    }
                    // Check for empty character class
                    if chars.peek() == Some(&']') {
                        return Err(ValidationError::InvalidGlobPattern {
                            pattern: pattern.to_string(),
                            reason: "Empty character class []".to_string(),
                        });
                    }
                }
                ']' => {
                    if bracket_depth == 0 {
                        return Err(ValidationError::InvalidGlobPattern {
                            pattern: pattern.to_string(),
                            reason: "Unmatched closing bracket ']'".to_string(),
                        });
                    }
                    bracket_depth -= 1;
                }
                '{' => {
                    brace_depth += 1;
                    if brace_depth > 3 {
                        return Err(ValidationError::InvalidGlobPattern {
                            pattern: pattern.to_string(),
                            reason: "Too many nested braces (max 3)".to_string(),
                        });
                    }
                }
                '}' => {
                    if brace_depth == 0 {
                        return Err(ValidationError::InvalidGlobPattern {
                            pattern: pattern.to_string(),
                            reason: "Unmatched closing brace '}'".to_string(),
                        });
                    }
                    brace_depth -= 1;
                }
                '\\' => {
                    // Check for valid escape sequences
                    if let Some(next_ch) = chars.next() {
                        if !matches!(
                            next_ch,
                            '*' | '?' | '[' | ']' | '{' | '}' | '\\' | '/' | '!' | '-' | '^'
                        ) {
                            return Err(ValidationError::InvalidGlobPattern {
                                pattern: pattern.to_string(),
                                reason: format!("Invalid escape sequence '\\{}'", next_ch),
                            });
                        }
                    } else {
                        return Err(ValidationError::InvalidGlobPattern {
                            pattern: pattern.to_string(),
                            reason: "Trailing backslash".to_string(),
                        });
                    }
                }
                _ => {}
            }
        }

        // Check for unclosed brackets or braces
        if bracket_depth > 0 {
            return Err(ValidationError::InvalidGlobPattern {
                pattern: pattern.to_string(),
                reason: "Unclosed character class '['".to_string(),
            });
        }

        if brace_depth > 0 {
            return Err(ValidationError::InvalidGlobPattern {
                pattern: pattern.to_string(),
                reason: "Unclosed brace group '{'".to_string(),
            });
        }

        Ok(())
    }

    /// Validate glob pattern depth
    fn validate_glob_depth(&self, pattern: &str) -> ValidationResult<()> {
        let depth = pattern.matches("**/").count() + pattern.matches("/**/").count();
        if depth > self.config.max_glob_depth {
            return Err(ValidationError::PatternTooComplex {
                reason: format!(
                    "Pattern depth {} exceeds maximum {}",
                    depth, self.config.max_glob_depth
                ),
            });
        }
        Ok(())
    }

    /// Validate problematic character sequences
    fn validate_glob_sequences(&self, pattern: &str) -> ValidationResult<()> {
        // Check for too many consecutive wildcards
        if pattern.contains("****") {
            return Err(ValidationError::InvalidGlobPattern {
                pattern: pattern.to_string(),
                reason: "Too many consecutive wildcards".to_string(),
            });
        }

        // Check for too many consecutive question marks
        if pattern.contains("????") {
            return Err(ValidationError::InvalidGlobPattern {
                pattern: pattern.to_string(),
                reason: "Too many consecutive single-character wildcards".to_string(),
            });
        }

        // Check for problematic combinations
        if pattern.contains("**/**/**/**") {
            return Err(ValidationError::PatternTooComplex {
                reason: "Too many recursive directory wildcards".to_string(),
            });
        }

        Ok(())
    }

    /// Validate a gitignore pattern
    pub fn validate_gitignore_pattern(&self, pattern: &str) -> ValidationResult<()> {
        // Skip comments and empty lines (these are always valid in gitignore)
        let trimmed = pattern.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            return Ok(());
        }

        // Check for empty pattern (only for non-gitignore patterns)
        if trimmed.is_empty() && !self.config.allow_empty_patterns {
            return Err(ValidationError::EmptyPattern);
        }

        // Check pattern length
        if pattern.len() > self.config.max_pattern_length {
            return Err(ValidationError::InvalidGitignorePattern {
                pattern: pattern.to_string(),
                reason: format!(
                    "Pattern too long: {} characters (max: {})",
                    pattern.len(),
                    self.config.max_pattern_length
                ),
            });
        }

        // Validate gitignore-specific syntax
        self.validate_gitignore_syntax(trimmed)?;

        Ok(())
    }

    /// Validate gitignore pattern syntax
    fn validate_gitignore_syntax(&self, pattern: &str) -> ValidationResult<()> {
        // Remove leading negation for validation
        let pattern = if pattern.starts_with('!') {
            &pattern[1..]
        } else {
            pattern
        };

        // Remove trailing slash for validation
        let pattern = pattern.trim_end_matches('/');

        // Validate as glob pattern
        self.validate_glob_pattern(pattern)?;

        // Additional gitignore-specific validations
        if pattern.contains("**/**/**/**") {
            return Err(ValidationError::InvalidGitignorePattern {
                pattern: pattern.to_string(),
                reason: "Too many recursive directory patterns".to_string(),
            });
        }

        Ok(())
    }

    /// Validate a collection of patterns
    pub fn validate_patterns<I, S>(&self, patterns: I) -> ValidationResult<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let patterns: Vec<_> = patterns.into_iter().collect();

        // Check pattern count limit
        if patterns.len() > self.config.max_patterns {
            return Err(ValidationError::PatternLimitExceeded {
                max: self.config.max_patterns,
                actual: patterns.len(),
            });
        }

        // Validate each pattern
        for pattern in &patterns {
            self.validate_glob_pattern(pattern.as_ref())?;
        }

        // Check for conflicts if enabled
        if self.config.check_conflicts {
            self.check_pattern_conflicts(&patterns)?;
        }

        Ok(())
    }

    /// Check for conflicting patterns
    fn check_pattern_conflicts<S: AsRef<str>>(&self, patterns: &[S]) -> ValidationResult<()> {
        let mut seen_patterns = HashSet::new();
        let mut include_patterns = HashSet::new();
        let mut exclude_patterns = HashSet::new();

        for pattern in patterns {
            let pattern_str = pattern.as_ref();

            // Check for duplicate patterns
            if !seen_patterns.insert(pattern_str.to_string()) {
                return Err(ValidationError::ConflictingPatterns {
                    conflict: format!("Duplicate pattern: '{}'", pattern_str),
                });
            }

            // Categorize patterns (simplified logic)
            if pattern_str.starts_with('!') {
                exclude_patterns.insert(&pattern_str[1..]);
            } else {
                include_patterns.insert(pattern_str);
            }
        }

        // Check for include/exclude conflicts
        for include in &include_patterns {
            if exclude_patterns.contains(include) {
                return Err(ValidationError::ConflictingPatterns {
                    conflict: format!("Pattern '{}' is both included and excluded", include),
                });
            }
        }

        Ok(())
    }

    /// Validate a file path
    pub fn validate_path<P: AsRef<Path>>(&self, path: P) -> ValidationResult<()> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy();

        // Check for invalid characters (platform-specific)
        #[cfg(windows)]
        {
            let invalid_chars = ['<', '>', ':', '"', '|', '?', '*'];
            if path_str.chars().any(|c| invalid_chars.contains(&c)) {
                return Err(ValidationError::InvalidPath {
                    path: path_str.to_string(),
                    reason: "Contains invalid characters for Windows".to_string(),
                });
            }
        }

        // Check path length (platform-specific limits)
        #[cfg(windows)]
        const MAX_PATH_LEN: usize = 260;
        #[cfg(not(windows))]
        const MAX_PATH_LEN: usize = 4096;

        if path_str.len() > MAX_PATH_LEN {
            return Err(ValidationError::InvalidPath {
                path: path_str.to_string(),
                reason: format!(
                    "Path too long: {} characters (max: {})",
                    path_str.len(),
                    MAX_PATH_LEN
                ),
            });
        }

        // Check if path exists (if configured)
        if self.config.validate_path_existence && !path.exists() {
            return Err(ValidationError::InvalidPath {
                path: path_str.to_string(),
                reason: "Path does not exist".to_string(),
            });
        }

        Ok(())
    }

    /// Validate pattern performance characteristics
    pub fn validate_pattern_performance(&self, pattern: &str) -> ValidationResult<PerformanceRisk> {
        let mut risk_score = 0;
        let mut issues = Vec::new();

        // Count wildcards
        let wildcard_count = pattern.matches('*').count();
        let single_wildcard_count = pattern.matches('?').count();

        if wildcard_count > 10 {
            risk_score += 3;
            issues.push("High number of wildcards may impact performance".to_string());
        }

        if single_wildcard_count > 20 {
            risk_score += 2;
            issues.push("High number of single-char wildcards may impact performance".to_string());
        }

        // Check for recursive patterns
        let recursive_count = pattern.matches("**/").count();
        if recursive_count > 3 {
            risk_score += 4;
            issues.push(
                "Multiple recursive patterns may cause exponential matching time".to_string(),
            );
        }

        // Check for alternations
        let alternation_count = pattern.matches('{').count();
        if alternation_count > 5 {
            risk_score += 2;
            issues.push("Many alternations may increase compilation time".to_string());
        }

        // Check for character classes
        let char_class_count = pattern.matches('[').count();
        if char_class_count > 10 {
            risk_score += 1;
            issues.push("Many character classes may slow down matching".to_string());
        }

        let risk_level = match risk_score {
            0..=2 => PerformanceRiskLevel::Low,
            3..=5 => PerformanceRiskLevel::Medium,
            6..=8 => PerformanceRiskLevel::High,
            _ => PerformanceRiskLevel::Critical,
        };

        let recommendations = self.generate_performance_recommendations(risk_score, &issues);

        Ok(PerformanceRisk {
            level: risk_level,
            score: risk_score,
            issues,
            recommendations,
        })
    }

    /// Generate performance recommendations
    fn generate_performance_recommendations(
        &self,
        risk_score: u32,
        issues: &[String],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if risk_score > 5 {
            recommendations
                .push("Consider simplifying the pattern to improve performance".to_string());
        }

        if issues.iter().any(|i| i.contains("recursive")) {
            recommendations
                .push("Limit recursive patterns (**/) to essential cases only".to_string());
        }

        if issues.iter().any(|i| i.contains("wildcards")) {
            recommendations.push(
                "Use specific patterns instead of multiple wildcards where possible".to_string(),
            );
        }

        if issues.iter().any(|i| i.contains("alternations")) {
            recommendations.push(
                "Consider splitting complex alternations into multiple simpler patterns"
                    .to_string(),
            );
        }

        recommendations
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

/// Sanitize a pattern to make it safe for use
pub fn sanitize_pattern(pattern: &str) -> String {
    let mut sanitized = String::with_capacity(pattern.len());
    let mut consecutive_wildcards = 0;
    let mut chars = pattern.chars();

    while let Some(ch) = chars.next() {
        match ch {
            '*' => {
                consecutive_wildcards += 1;
                if consecutive_wildcards <= 2 {
                    sanitized.push(ch);
                }
            }
            '?' => {
                consecutive_wildcards = 0;
                sanitized.push(ch);
            }
            '\\' => {
                // Preserve escape sequences
                sanitized.push(ch);
                if let Some(next_ch) = chars.next() {
                    sanitized.push(next_ch);
                }
                consecutive_wildcards = 0;
            }
            _ => {
                consecutive_wildcards = 0;
                sanitized.push(ch);
            }
        }
    }

    // Limit overall length
    if sanitized.len() > 1024 {
        sanitized.truncate(1024);
    }

    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_validator() -> PatternValidator {
        PatternValidator::default().unwrap()
    }

    #[test]
    fn test_valid_glob_patterns() {
        let validator = create_validator();

        let valid_patterns = [
            "*.rs",
            "src/**/*.rs",
            "test/[a-z]*.py",
            "{*.js,*.ts}",
            "file?.txt",
            "src/**/lib.rs",
        ];

        for pattern in &valid_patterns {
            assert!(
                validator.validate_glob_pattern(pattern).is_ok(),
                "Pattern should be valid: {}",
                pattern
            );
        }
    }

    #[test]
    fn test_invalid_glob_patterns() {
        let validator = create_validator();

        let invalid_patterns = [
            "[",    // Unclosed bracket
            "}",    // Unmatched brace
            "\\",   // Trailing backslash
            "[]",   // Empty character class
            "****", // Too many wildcards
            "????", // Too many single wildcards
        ];

        for pattern in &invalid_patterns {
            assert!(
                validator.validate_glob_pattern(pattern).is_err(),
                "Pattern should be invalid: {}",
                pattern
            );
        }
    }

    #[test]
    fn test_valid_gitignore_patterns() {
        let validator = create_validator();

        let valid_patterns = [
            "*.log",
            "!important.log",
            "temp/",
            "/absolute/path",
            "# This is a comment",
            "",
            " ",
        ];

        for pattern in &valid_patterns {
            assert!(
                validator.validate_gitignore_pattern(pattern).is_ok(),
                "Gitignore pattern should be valid: {}",
                pattern
            );
        }
    }

    #[test]
    fn test_pattern_conflicts() {
        let validator = create_validator();

        let conflicting_patterns = [
            "*.rs", "!*.rs", // Conflicts with first pattern
        ];

        assert!(validator
            .validate_patterns(conflicting_patterns.iter())
            .is_err());
    }

    #[test]
    fn test_duplicate_patterns() {
        let validator = create_validator();

        let duplicate_patterns = [
            "*.rs", "*.py", "*.rs", // Duplicate
        ];

        assert!(validator
            .validate_patterns(duplicate_patterns.iter())
            .is_err());
    }

    #[test]
    fn test_pattern_limits() {
        let config = ValidationConfig {
            max_patterns: 2,
            ..Default::default()
        };
        let validator = PatternValidator::new(config).unwrap();

        let too_many_patterns = ["*.rs", "*.py", "*.js"];
        assert!(validator
            .validate_patterns(too_many_patterns.iter())
            .is_err());
    }

    #[test]
    fn test_empty_patterns() {
        let config = ValidationConfig {
            allow_empty_patterns: false,
            ..Default::default()
        };
        let validator = PatternValidator::new(config).unwrap();

        assert!(validator.validate_glob_pattern("").is_err());

        let config = ValidationConfig {
            allow_empty_patterns: true,
            ..Default::default()
        };
        let validator = PatternValidator::new(config).unwrap();

        assert!(validator.validate_glob_pattern("").is_ok());
    }

    #[test]
    fn test_performance_validation() {
        let validator = create_validator();

        // Low risk pattern
        let low_risk = validator.validate_pattern_performance("*.rs").unwrap();
        assert_eq!(low_risk.level, PerformanceRiskLevel::Low);

        // High risk pattern with many wildcards
        let high_risk = validator
            .validate_pattern_performance("**/**/**/**/**/**/*****.rs")
            .unwrap();
        assert!(matches!(
            high_risk.level,
            PerformanceRiskLevel::High | PerformanceRiskLevel::Critical
        ));
        assert!(high_risk.level.needs_attention());
    }

    #[test]
    fn test_path_validation() {
        let validator = create_validator();

        // Valid paths
        assert!(validator.validate_path("src/main.rs").is_ok());
        assert!(validator.validate_path("./relative/path").is_ok());

        // Test with current directory (should exist)
        assert!(validator.validate_path(".").is_ok());
    }

    #[test]
    fn test_pattern_sanitization() {
        assert_eq!(sanitize_pattern("****"), "**");
        assert_eq!(sanitize_pattern("a****b"), "a**b");
        assert_eq!(sanitize_pattern("normal.rs"), "normal.rs");

        // Test length limiting
        let long_pattern = "a".repeat(2000);
        let sanitized = sanitize_pattern(&long_pattern);
        assert!(sanitized.len() <= 1024);
    }

    #[test]
    fn test_escape_sequences() {
        let validator = create_validator();

        // Valid escape sequences
        assert!(validator.validate_glob_pattern(r"\*literal\*").is_ok());
        assert!(validator.validate_glob_pattern(r"file\?.txt").is_ok());
        assert!(validator.validate_glob_pattern(r"\[not a class\]").is_ok());

        // Invalid escape sequence
        assert!(validator.validate_glob_pattern(r"\z").is_err());
    }

    #[test]
    fn test_nested_patterns() {
        let validator = create_validator();

        // Valid nesting
        assert!(validator.validate_glob_pattern("src/{lib,main}.rs").is_ok());
        assert!(validator.validate_glob_pattern("test/[a-z]*.py").is_ok());

        // Too much nesting
        assert!(validator.validate_glob_pattern("{{{{{{{{").is_err());
        assert!(validator.validate_glob_pattern("[[[[[[").is_err());
    }
}

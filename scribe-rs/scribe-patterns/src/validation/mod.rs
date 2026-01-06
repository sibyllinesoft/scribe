//! Pattern validation with comprehensive rules.

mod types;

pub use types::{
    PerformanceRisk, PerformanceRiskLevel, ValidationConfig, ValidationError, ValidationResult,
};

use regex::Regex;
use std::collections::HashSet;
use std::path::Path;

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
        dangerous_patterns.insert("**/*/**/*/**/*/**/*/**".to_string());
        dangerous_patterns.insert("*".repeat(100));
        dangerous_patterns.insert("?".repeat(100));

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
        if pattern.is_empty() && !self.config.allow_empty_patterns {
            return Err(ValidationError::EmptyPattern);
        }

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

        if self.dangerous_patterns.contains(pattern) {
            return Err(ValidationError::PatternTooComplex {
                reason: "Pattern is known to cause performance issues".to_string(),
            });
        }

        self.validate_glob_syntax(pattern)?;
        self.validate_glob_depth(pattern)?;
        self.validate_glob_sequences(pattern)?;

        Ok(())
    }

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

    fn validate_glob_sequences(&self, pattern: &str) -> ValidationResult<()> {
        if pattern.contains("****") {
            return Err(ValidationError::InvalidGlobPattern {
                pattern: pattern.to_string(),
                reason: "Too many consecutive wildcards".to_string(),
            });
        }

        if pattern.contains("????") {
            return Err(ValidationError::InvalidGlobPattern {
                pattern: pattern.to_string(),
                reason: "Too many consecutive single-character wildcards".to_string(),
            });
        }

        if pattern.contains("**/**/**/**") {
            return Err(ValidationError::PatternTooComplex {
                reason: "Too many recursive directory wildcards".to_string(),
            });
        }

        Ok(())
    }

    /// Validate a gitignore pattern
    pub fn validate_gitignore_pattern(&self, pattern: &str) -> ValidationResult<()> {
        let trimmed = pattern.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            return Ok(());
        }

        if trimmed.is_empty() && !self.config.allow_empty_patterns {
            return Err(ValidationError::EmptyPattern);
        }

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

        self.validate_gitignore_syntax(trimmed)?;

        Ok(())
    }

    fn validate_gitignore_syntax(&self, pattern: &str) -> ValidationResult<()> {
        let pattern = if pattern.starts_with('!') {
            &pattern[1..]
        } else {
            pattern
        };

        let pattern = pattern.trim_end_matches('/');

        self.validate_glob_pattern(pattern)?;

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

        if patterns.len() > self.config.max_patterns {
            return Err(ValidationError::PatternLimitExceeded {
                max: self.config.max_patterns,
                actual: patterns.len(),
            });
        }

        for pattern in &patterns {
            self.validate_glob_pattern(pattern.as_ref())?;
        }

        if self.config.check_conflicts {
            self.check_pattern_conflicts(&patterns)?;
        }

        Ok(())
    }

    fn check_pattern_conflicts<S: AsRef<str>>(&self, patterns: &[S]) -> ValidationResult<()> {
        let mut seen_patterns = HashSet::new();
        let mut include_patterns = HashSet::new();
        let mut exclude_patterns = HashSet::new();

        for pattern in patterns {
            let pattern_str = pattern.as_ref();

            if !seen_patterns.insert(pattern_str.to_string()) {
                return Err(ValidationError::ConflictingPatterns {
                    conflict: format!("Duplicate pattern: '{}'", pattern_str),
                });
            }

            if pattern_str.starts_with('!') {
                exclude_patterns.insert(&pattern_str[1..]);
            } else {
                include_patterns.insert(pattern_str);
            }
        }

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

        let recursive_count = pattern.matches("**/").count();
        if recursive_count > 3 {
            risk_score += 4;
            issues.push(
                "Multiple recursive patterns may cause exponential matching time".to_string(),
            );
        }

        let alternation_count = pattern.matches('{').count();
        if alternation_count > 5 {
            risk_score += 2;
            issues.push("Many alternations may increase compilation time".to_string());
        }

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
            "foo/bar/baz.txt",
            "**/lib.rs",
            "src/**",
        ];

        for pattern in &valid_patterns {
            assert!(
                validator.validate_glob_pattern(pattern).is_ok(),
                "Pattern '{}' should be valid",
                pattern
            );
        }
    }

    #[test]
    fn test_invalid_glob_patterns() {
        let validator = create_validator();

        // Unmatched brackets
        assert!(validator.validate_glob_pattern("[a-z").is_err());
        assert!(validator.validate_glob_pattern("test]").is_err());

        // Unmatched braces
        assert!(validator.validate_glob_pattern("{*.rs").is_err());
        assert!(validator.validate_glob_pattern("*.ts}").is_err());

        // Empty character class
        assert!(validator.validate_glob_pattern("[]").is_err());

        // Too many consecutive wildcards
        assert!(validator.validate_glob_pattern("****").is_err());
        assert!(validator.validate_glob_pattern("????").is_err());
    }

    #[test]
    fn test_pattern_length_limit() {
        let validator = create_validator();

        let long_pattern = "*".repeat(3000);
        assert!(validator.validate_glob_pattern(&long_pattern).is_err());
    }

    #[test]
    fn test_gitignore_patterns() {
        let validator = create_validator();

        // Valid gitignore patterns
        assert!(validator.validate_gitignore_pattern("# comment").is_ok());
        assert!(validator.validate_gitignore_pattern("*.log").is_ok());
        assert!(validator.validate_gitignore_pattern("!important.log").is_ok());
        assert!(validator.validate_gitignore_pattern("build/").is_ok());

        // Invalid gitignore patterns
        assert!(validator
            .validate_gitignore_pattern("**/**/**/**/*")
            .is_err());
    }

    #[test]
    fn test_pattern_conflicts() {
        let validator = create_validator();

        // Duplicate patterns
        let patterns = vec!["*.rs", "*.rs"];
        assert!(validator.validate_patterns(patterns).is_err());

        // Include/exclude conflict
        let patterns = vec!["*.rs", "!*.rs"];
        assert!(validator.validate_patterns(patterns).is_err());
    }

    #[test]
    fn test_path_validation() {
        let validator = create_validator();

        // Valid paths
        assert!(validator.validate_path("src/main.rs").is_ok());
        assert!(validator.validate_path("./relative/path").is_ok());

        // Path too long
        let long_path = "a".repeat(5000);
        assert!(validator.validate_path(&long_path).is_err());
    }

    #[test]
    fn test_performance_risk_assessment() {
        let validator = create_validator();

        // Low risk pattern
        let risk = validator
            .validate_pattern_performance("*.rs")
            .unwrap();
        assert_eq!(risk.level, PerformanceRiskLevel::Low);

        // High risk pattern
        let risk = validator
            .validate_pattern_performance("**/**/**/**/*.{rs,py,js,ts,go}")
            .unwrap();
        assert!(risk.level.needs_attention());
    }

    #[test]
    fn test_sanitize_pattern() {
        // Test wildcard collapsing
        assert_eq!(sanitize_pattern("*****"), "**");
        assert_eq!(sanitize_pattern("a***b"), "a**b");

        // Test length limiting
        let long_pattern = "a".repeat(2000);
        let sanitized = sanitize_pattern(&long_pattern);
        assert!(sanitized.len() <= 1024);

        // Test escape preservation
        assert_eq!(sanitize_pattern("\\*\\?"), "\\*\\?");
    }

    #[test]
    fn test_empty_pattern() {
        let validator = create_validator();
        assert!(validator.validate_glob_pattern("").is_err());

        // Allow empty patterns with custom config
        let config = ValidationConfig {
            allow_empty_patterns: true,
            ..Default::default()
        };
        let validator = PatternValidator::new(config).unwrap();
        assert!(validator.validate_glob_pattern("").is_ok());
    }

    #[test]
    fn test_pattern_limit() {
        let validator = create_validator();

        // Create more patterns than allowed
        let patterns: Vec<String> = (0..1500).map(|i| format!("file{}.rs", i)).collect();
        assert!(validator.validate_patterns(patterns).is_err());
    }
}

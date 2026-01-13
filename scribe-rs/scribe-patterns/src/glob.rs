//! High-performance glob pattern matching implementation.
//!
//! This module provides efficient glob pattern matching using the `globset` crate
//! with caching, compilation optimization, and comprehensive pattern support.

use crate::utils::normalize_path;
use globset::{Glob, GlobBuilder, GlobSet, GlobSetBuilder};
use scribe_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// High-performance glob pattern matcher with compilation caching
#[derive(Debug)]
pub struct GlobMatcher {
    patterns: Vec<GlobPattern>,
    compiled_set: Option<GlobSet>,
    options: GlobOptions,
    cache: HashMap<String, bool>,
    cache_hits: u64,
    cache_misses: u64,
}

/// Individual glob pattern with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobPattern {
    pub pattern: String,
    pub case_sensitive: bool,
    pub literal_separator: bool,
    pub backslash_escape: bool,
    pub require_literal_separator: bool,
    pub require_literal_leading_dot: bool,
}

/// Configuration options for glob matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobOptions {
    pub case_sensitive: bool,
    pub literal_separator: bool,
    pub backslash_escape: bool,
    pub require_literal_separator: bool,
    pub require_literal_leading_dot: bool,
    pub cache_enabled: bool,
    pub cache_size_limit: usize,
}

/// Result of a glob match operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobMatchResult {
    pub matched: bool,
    pub pattern_index: Option<usize>,
    pub pattern: Option<String>,
    pub match_method: MatchMethod,
}

/// Method used for pattern matching
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchMethod {
    Cached,
    Compiled,
    Individual,
    Literal,
}

impl Default for GlobOptions {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            literal_separator: false,
            backslash_escape: false,
            require_literal_separator: false,
            require_literal_leading_dot: false,
            cache_enabled: true,
            cache_size_limit: 1000,
        }
    }
}

impl GlobPattern {
    /// Create a new glob pattern with default options
    pub fn new(pattern: &str) -> Result<Self> {
        Self::with_options(pattern, &GlobOptions::default())
    }

    /// Create a new glob pattern with specific options
    pub fn with_options(pattern: &str, options: &GlobOptions) -> Result<Self> {
        // Validate the pattern by trying to compile it
        let _glob = Glob::new(pattern)?;

        Ok(Self {
            pattern: pattern.to_string(),
            case_sensitive: options.case_sensitive,
            literal_separator: options.literal_separator,
            backslash_escape: options.backslash_escape,
            require_literal_separator: options.require_literal_separator,
            require_literal_leading_dot: options.require_literal_leading_dot,
        })
    }

    /// Check if this pattern matches a path
    pub fn matches<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let normalized_path = normalize_path(path);
        let path_str = normalized_path.to_string_lossy();

        let mut glob_builder = globset::GlobBuilder::new(&self.pattern);
        glob_builder.case_insensitive(!self.case_sensitive);
        glob_builder.literal_separator(self.literal_separator);
        glob_builder.backslash_escape(self.backslash_escape);

        let glob = glob_builder.build()?;
        let matcher = glob.compile_matcher();
        Ok(matcher.is_match(path_str.as_ref()))
    }

    /// Check if this is a literal (non-glob) pattern
    pub fn is_literal(&self) -> bool {
        !self.pattern.contains('*')
            && !self.pattern.contains('?')
            && !self.pattern.contains('[')
            && !self.pattern.contains('{')
    }

    /// Get the pattern string
    pub fn as_str(&self) -> &str {
        &self.pattern
    }
}

impl GlobMatcher {
    /// Create a new glob matcher with default options
    pub fn new() -> Self {
        Self::with_options(GlobOptions::default())
    }

    /// Create a new glob matcher with specific options
    pub fn with_options(options: GlobOptions) -> Self {
        Self {
            patterns: Vec::new(),
            compiled_set: None,
            options,
            cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Add a glob pattern to the matcher
    pub fn add_pattern(&mut self, pattern: &str) -> Result<()> {
        let glob_pattern = GlobPattern::with_options(pattern, &self.options)?;
        self.patterns.push(glob_pattern);

        // Invalidate compiled set - will be rebuilt on next match
        self.compiled_set = None;

        Ok(())
    }

    /// Add multiple glob patterns
    pub fn add_patterns<I, S>(&mut self, patterns: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for pattern in patterns {
            self.add_pattern(pattern.as_ref())?;
        }
        Ok(())
    }

    /// Add patterns from comma-separated string
    pub fn add_patterns_csv(&mut self, csv: &str) -> Result<()> {
        let patterns = crate::utils::parse_csv_patterns(csv);
        for pattern in patterns {
            self.add_pattern(&pattern)?;
        }
        Ok(())
    }

    /// Remove all patterns
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.compiled_set = None;
        self.cache.clear();
    }

    /// Check if any pattern matches the given path
    pub fn matches<P: AsRef<Path>>(&mut self, path: P) -> Result<bool> {
        let result = self.match_with_details(path)?;
        Ok(result.matched)
    }

    /// Get detailed match information
    pub fn match_with_details<P: AsRef<Path>>(&mut self, path: P) -> Result<GlobMatchResult> {
        let normalized_path = normalize_path(path);
        let path_str = normalized_path.to_string_lossy().to_string();

        // Check cache first if enabled
        if self.options.cache_enabled {
            if let Some(&cached_result) = self.cache.get(&path_str) {
                self.cache_hits += 1;
                return Ok(GlobMatchResult {
                    matched: cached_result,
                    pattern_index: None, // Cache doesn't store pattern index
                    pattern: None,
                    match_method: MatchMethod::Cached,
                });
            }
            self.cache_misses += 1;
        }

        if self.patterns.is_empty() {
            return Ok(GlobMatchResult {
                matched: false,
                pattern_index: None,
                pattern: None,
                match_method: MatchMethod::Individual,
            });
        }

        // Use compiled set for performance when we have multiple patterns
        let result = if self.patterns.len() > 1 {
            self.match_with_compiled_set(&normalized_path)?
        } else {
            self.match_with_individual_pattern(&normalized_path)?
        };

        // Cache the result if caching is enabled
        if self.options.cache_enabled {
            if self.cache.len() >= self.options.cache_size_limit {
                // Simple cache eviction - remove half the entries
                let keys_to_remove: Vec<String> = self
                    .cache
                    .keys()
                    .take(self.cache.len() / 2)
                    .cloned()
                    .collect();
                for key in keys_to_remove {
                    self.cache.remove(&key);
                }
            }
            self.cache.insert(path_str, result.matched);
        }

        Ok(result)
    }

    /// Match using compiled glob set (efficient for multiple patterns)
    fn match_with_compiled_set(&mut self, path: &Path) -> Result<GlobMatchResult> {
        if self.compiled_set.is_none() {
            self.compiled_set = Some(self.compile_patterns()?);
        }

        let compiled_set = self.compiled_set.as_ref().unwrap();
        let path_str = path.to_string_lossy();

        let matches: Vec<usize> = compiled_set.matches(path_str.as_ref());

        if matches.is_empty() {
            Ok(GlobMatchResult {
                matched: false,
                pattern_index: None,
                pattern: None,
                match_method: MatchMethod::Compiled,
            })
        } else {
            let pattern_index = matches[0];
            let pattern = self.patterns.get(pattern_index).map(|p| p.pattern.clone());

            Ok(GlobMatchResult {
                matched: true,
                pattern_index: Some(pattern_index),
                pattern,
                match_method: MatchMethod::Compiled,
            })
        }
    }

    /// Match using individual pattern (used for single patterns or fallback)
    fn match_with_individual_pattern(&self, path: &Path) -> Result<GlobMatchResult> {
        for (index, pattern) in self.patterns.iter().enumerate() {
            if pattern.matches(path)? {
                return Ok(GlobMatchResult {
                    matched: true,
                    pattern_index: Some(index),
                    pattern: Some(pattern.pattern.clone()),
                    match_method: if pattern.is_literal() {
                        MatchMethod::Literal
                    } else {
                        MatchMethod::Individual
                    },
                });
            }
        }

        Ok(GlobMatchResult {
            matched: false,
            pattern_index: None,
            pattern: None,
            match_method: MatchMethod::Individual,
        })
    }

    /// Compile all patterns into a GlobSet for efficient batch matching
    fn compile_patterns(&self) -> Result<GlobSet> {
        let mut builder = GlobSetBuilder::new();

        for pattern in &self.patterns {
            let mut glob_builder = GlobBuilder::new(&pattern.pattern);
            glob_builder.case_insensitive(!pattern.case_sensitive);
            glob_builder.literal_separator(pattern.literal_separator);
            glob_builder.backslash_escape(pattern.backslash_escape);

            let glob = glob_builder.build()?;
            builder.add(glob);
        }

        Ok(builder.build()?)
    }

    /// Get the number of patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Get all patterns
    pub fn patterns(&self) -> &[GlobPattern] {
        &self.patterns
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (u64, u64, usize) {
        (self.cache_hits, self.cache_misses, self.cache.len())
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Check if patterns are compiled
    pub fn is_compiled(&self) -> bool {
        self.compiled_set.is_some()
    }

    /// Force recompilation of patterns
    pub fn recompile(&mut self) -> Result<()> {
        if !self.patterns.is_empty() {
            self.compiled_set = Some(self.compile_patterns()?);
        }
        Ok(())
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Optimize patterns for better performance
    pub fn optimize(&mut self) {
        // Sort patterns by complexity (literal patterns first)
        self.patterns.sort_by_key(|p| !p.is_literal());

        // Invalidate compiled set to force recompilation with new order
        self.compiled_set = None;
    }

    /// Test all patterns against a path and return all matches
    pub fn match_all<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<usize>> {
        if self.compiled_set.is_none() && self.patterns.len() > 1 {
            self.compiled_set = Some(self.compile_patterns()?);
        }

        if let Some(ref compiled_set) = self.compiled_set {
            let path_str = path.as_ref().to_string_lossy();
            Ok(compiled_set.matches(path_str.as_ref()))
        } else {
            // Fallback to individual matching
            let mut matches = Vec::new();
            for (index, pattern) in self.patterns.iter().enumerate() {
                if pattern.matches(&path)? {
                    matches.push(index);
                }
            }
            Ok(matches)
        }
    }

    /// Check if matcher contains any patterns
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Enable or disable caching
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.options.cache_enabled = enabled;
        if !enabled {
            self.clear_cache();
        }
    }

    /// Set cache size limit
    pub fn set_cache_size_limit(&mut self, limit: usize) {
        self.options.cache_size_limit = limit;

        // Trim cache if it exceeds new limit
        if self.cache.len() > limit {
            let keys_to_remove: Vec<String> = self.cache.keys().skip(limit).cloned().collect();
            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
    }
}

impl Default for GlobMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common glob operations
impl GlobMatcher {
    /// Create a matcher for specific file extensions
    pub fn for_extensions(extensions: &[&str]) -> Result<Self> {
        let mut matcher = Self::new();
        for ext in extensions {
            let pattern = crate::utils::extension_to_glob(ext);
            matcher.add_pattern(&pattern)?;
        }
        Ok(matcher)
    }

    /// Create a matcher for files in specific directories
    pub fn for_directories(directories: &[&str]) -> Result<Self> {
        let mut matcher = Self::new();
        for dir in directories {
            let pattern = format!("{}/**/*", dir.trim_end_matches('/'));
            matcher.add_pattern(&pattern)?;
        }
        Ok(matcher)
    }

    /// Create a case-insensitive matcher
    pub fn case_insensitive() -> Self {
        Self::with_options(GlobOptions {
            case_sensitive: false,
            ..Default::default()
        })
    }
}

// Note: From<globset::Error> for ScribeError is already implemented in scribe-core

#[cfg(test)]
mod tests {
    use super::*;
    // use std::path::PathBuf; // Not used in these tests

    #[test]
    fn test_glob_pattern_creation() {
        let pattern = GlobPattern::new("**/*.rs").unwrap();
        assert_eq!(pattern.pattern, "**/*.rs");
        assert!(pattern.case_sensitive);

        assert!(pattern.matches("src/lib.rs").unwrap());
        assert!(pattern.matches("tests/integration/test.rs").unwrap());
        assert!(!pattern.matches("src/lib.py").unwrap());
    }

    #[test]
    fn test_glob_pattern_literal_detection() {
        let literal = GlobPattern::new("src/lib.rs").unwrap();
        assert!(literal.is_literal());

        let glob = GlobPattern::new("src/**/*.rs").unwrap();
        assert!(!glob.is_literal());

        let question_mark = GlobPattern::new("src/lib?.rs").unwrap();
        assert!(!question_mark.is_literal());

        let bracket = GlobPattern::new("src/lib[123].rs").unwrap();
        assert!(!bracket.is_literal());

        let brace = GlobPattern::new("src/lib.{rs,py}").unwrap();
        assert!(!brace.is_literal());
    }

    #[test]
    fn test_case_insensitive_matching() {
        let options = GlobOptions {
            case_sensitive: false,
            ..Default::default()
        };

        let pattern = GlobPattern::with_options("**/*.RS", &options).unwrap();
        assert!(pattern.matches("src/lib.rs").unwrap());
        assert!(pattern.matches("src/LIB.RS").unwrap());
        assert!(pattern.matches("src/Lib.Rs").unwrap());
    }

    #[test]
    fn test_glob_matcher_single_pattern() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap();

        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("tests/test.rs").unwrap());
        assert!(!matcher.matches("src/lib.py").unwrap());
    }

    #[test]
    fn test_glob_matcher_multiple_patterns() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap();
        matcher.add_pattern("**/*.py").unwrap();
        matcher.add_pattern("**/*.js").unwrap();

        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src/main.py").unwrap());
        assert!(matcher.matches("src/app.js").unwrap());
        assert!(!matcher.matches("src/data.json").unwrap());
    }

    #[test]
    fn test_glob_matcher_csv_patterns() {
        let mut matcher = GlobMatcher::new();
        matcher
            .add_patterns_csv("**/*.rs, **/*.py , **/*.js")
            .unwrap();

        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src/main.py").unwrap());
        assert!(matcher.matches("src/app.js").unwrap());
        assert!(!matcher.matches("src/data.json").unwrap());
        assert_eq!(matcher.pattern_count(), 3);
    }

    #[test]
    fn test_glob_matcher_detailed_results() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap();
        matcher.add_pattern("**/*.py").unwrap();

        let result = matcher.match_with_details("src/lib.rs").unwrap();
        assert!(result.matched);
        assert_eq!(result.pattern_index, Some(0));
        assert_eq!(result.pattern, Some("**/*.rs".to_string()));

        let result = matcher.match_with_details("src/main.py").unwrap();
        assert!(result.matched);
        assert_eq!(result.pattern_index, Some(1));
        assert_eq!(result.pattern, Some("**/*.py".to_string()));

        let result = matcher.match_with_details("src/data.json").unwrap();
        assert!(!result.matched);
        assert_eq!(result.pattern_index, None);
    }

    #[test]
    fn test_glob_matcher_cache() {
        let mut matcher = GlobMatcher::with_options(GlobOptions {
            cache_enabled: true,
            cache_size_limit: 10,
            ..Default::default()
        });

        matcher.add_pattern("**/*.rs").unwrap();

        // First match - cache miss
        assert!(matcher.matches("src/lib.rs").unwrap());
        let (hits, misses, size) = matcher.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
        assert_eq!(size, 1);

        // Second match - cache hit
        assert!(matcher.matches("src/lib.rs").unwrap());
        let (hits, misses, size) = matcher.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(size, 1);

        // Cache hit ratio should be 0.5
        assert_eq!(matcher.cache_hit_ratio(), 0.5);
    }

    #[test]
    fn test_glob_matcher_cache_eviction() {
        let mut matcher = GlobMatcher::with_options(GlobOptions {
            cache_enabled: true,
            cache_size_limit: 2,
            ..Default::default()
        });

        matcher.add_pattern("**/*").unwrap();

        // Fill cache to limit
        matcher.matches("file1.rs").unwrap();
        matcher.matches("file2.py").unwrap();
        assert_eq!(matcher.cache_stats().2, 2);

        // Adding another should trigger eviction
        matcher.matches("file3.js").unwrap();
        assert_eq!(matcher.cache_stats().2, 2); // Should still be at limit
    }

    #[test]
    fn test_glob_matcher_optimization() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap(); // Glob pattern
        matcher.add_pattern("exact/path.py").unwrap(); // Literal pattern
        matcher.add_pattern("src/**/*.js").unwrap(); // Glob pattern

        // Before optimization, order should be as added
        assert_eq!(matcher.patterns()[0].pattern, "**/*.rs");
        assert_eq!(matcher.patterns()[1].pattern, "exact/path.py");
        assert_eq!(matcher.patterns()[2].pattern, "src/**/*.js");

        matcher.optimize();

        // After optimization, literal patterns should come first
        assert_eq!(matcher.patterns()[0].pattern, "exact/path.py");
        assert!(matcher.patterns()[0].is_literal());
    }

    #[test]
    fn test_glob_matcher_match_all() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap();
        matcher.add_pattern("src/**").unwrap();
        matcher.add_pattern("**/*lib*").unwrap();

        let matches = matcher.match_all("src/lib.rs").unwrap();
        assert_eq!(matches.len(), 3); // Should match all patterns
        assert!(matches.contains(&0)); // **/*.rs
        assert!(matches.contains(&1)); // src/**
        assert!(matches.contains(&2)); // **/*lib*

        let matches = matcher.match_all("tests/test.rs").unwrap();
        assert_eq!(matches.len(), 1); // Should only match **/*.rs
        assert!(matches.contains(&0));
    }

    #[test]
    fn test_glob_matcher_convenience_methods() {
        let mut matcher = GlobMatcher::for_extensions(&["rs", "py", "js"]).unwrap();
        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src/main.py").unwrap());
        assert!(matcher.matches("src/app.js").unwrap());
        assert!(!matcher.matches("src/data.json").unwrap());
        assert_eq!(matcher.pattern_count(), 3);

        let mut matcher = GlobMatcher::for_directories(&["src", "tests"]).unwrap();
        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("tests/test.rs").unwrap());
        assert!(!matcher.matches("docs/readme.md").unwrap());
        assert_eq!(matcher.pattern_count(), 2);
    }

    #[test]
    fn test_glob_matcher_case_insensitive() {
        let mut matcher = GlobMatcher::case_insensitive();
        matcher.add_pattern("**/*.RS").unwrap();

        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src/LIB.RS").unwrap());
        assert!(matcher.matches("src/Lib.Rs").unwrap());
    }

    #[test]
    fn test_glob_matcher_empty() {
        let mut matcher = GlobMatcher::new();
        assert!(matcher.is_empty());
        assert!(!matcher.matches("any/path").unwrap());

        matcher.add_pattern("**/*.rs").unwrap();
        assert!(!matcher.is_empty());

        matcher.clear();
        assert!(matcher.is_empty());
        assert!(!matcher.matches("any/path.rs").unwrap());
    }

    #[test]
    fn test_glob_matcher_compilation() {
        let mut matcher = GlobMatcher::new();
        assert!(!matcher.is_compiled());

        matcher.add_pattern("**/*.rs").unwrap();
        matcher.add_pattern("**/*.py").unwrap();

        // Should still not be compiled until first match
        assert!(!matcher.is_compiled());

        // First match should trigger compilation
        matcher.matches("src/lib.rs").unwrap();
        assert!(matcher.is_compiled());

        // Adding pattern should invalidate compilation
        matcher.add_pattern("**/*.js").unwrap();
        assert!(!matcher.is_compiled());

        // Manual recompilation
        matcher.recompile().unwrap();
        assert!(matcher.is_compiled());
    }

    #[test]
    fn test_complex_glob_patterns() {
        let mut matcher = GlobMatcher::new();

        // Brace expansion
        matcher.add_pattern("**/*.{rs,py,js}").unwrap();
        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src/main.py").unwrap());
        assert!(matcher.matches("src/app.js").unwrap());
        assert!(!matcher.matches("src/data.json").unwrap());

        matcher.clear();

        // Character classes
        matcher.add_pattern("test[0-9].rs").unwrap();
        assert!(matcher.matches("test1.rs").unwrap());
        assert!(matcher.matches("test9.rs").unwrap());
        assert!(!matcher.matches("testA.rs").unwrap());

        matcher.clear();

        // Question mark
        matcher.add_pattern("test?.rs").unwrap();
        assert!(matcher.matches("test1.rs").unwrap());
        assert!(matcher.matches("testA.rs").unwrap());
        assert!(!matcher.matches("test12.rs").unwrap());
    }

    #[test]
    fn test_path_normalization_in_matching() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("src/**/*.rs").unwrap();

        // Test various path formats
        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src\\lib.rs").unwrap()); // Windows-style
        assert!(matcher.matches("src/subdir/lib.rs").unwrap());
        assert!(matcher.matches("src\\subdir\\lib.rs").unwrap()); // Windows-style
    }

    #[test]
    fn test_glob_pattern_as_str() {
        let pattern = GlobPattern::new("**/*.rs").unwrap();
        assert_eq!(pattern.as_str(), "**/*.rs");
    }

    #[test]
    fn test_glob_matcher_add_patterns_vec() {
        let mut matcher = GlobMatcher::new();
        let patterns = vec!["**/*.rs", "**/*.py", "**/*.js"];
        matcher.add_patterns(patterns).unwrap();

        assert_eq!(matcher.pattern_count(), 3);
        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src/main.py").unwrap());
    }

    #[test]
    fn test_glob_matcher_default() {
        let matcher = GlobMatcher::default();
        assert!(matcher.is_empty());
        assert!(!matcher.is_compiled());
    }

    #[test]
    fn test_glob_matcher_clear_cache() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap();

        // Add some entries to cache
        matcher.matches("src/lib.rs").unwrap();
        matcher.matches("src/main.rs").unwrap();

        let (hits, misses, size) = matcher.cache_stats();
        assert!(size > 0 || misses > 0);

        // Clear cache and verify
        matcher.clear_cache();
        let (hits, misses, size) = matcher.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
        assert_eq!(size, 0);
    }

    #[test]
    fn test_glob_matcher_cache_hit_ratio_zero() {
        let matcher = GlobMatcher::new();
        // No operations, so ratio should be 0.0
        assert_eq!(matcher.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_glob_matcher_set_cache_enabled() {
        let mut matcher = GlobMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap();

        // Cache should be enabled by default
        matcher.matches("src/lib.rs").unwrap();
        let (_, _, size) = matcher.cache_stats();
        assert!(size > 0);

        // Disable cache - should clear it
        matcher.set_cache_enabled(false);
        let (hits, misses, size) = matcher.cache_stats();
        assert_eq!(size, 0);

        // Matches should not be cached when disabled
        matcher.matches("src/main.rs").unwrap();
        let (_, _, size) = matcher.cache_stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_glob_matcher_set_cache_size_limit() {
        let mut matcher = GlobMatcher::with_options(GlobOptions {
            cache_enabled: true,
            cache_size_limit: 100,
            ..Default::default()
        });
        matcher.add_pattern("**/*").unwrap();

        // Add entries to cache
        for i in 0..10 {
            matcher.matches(format!("file{}.rs", i)).unwrap();
        }

        let (_, _, size) = matcher.cache_stats();
        assert!(size > 0);

        // Set a smaller limit - should trim cache
        matcher.set_cache_size_limit(3);
        let (_, _, size) = matcher.cache_stats();
        assert!(size <= 3);
    }

    #[test]
    fn test_glob_matcher_literal_match_method() {
        let mut matcher = GlobMatcher::new();
        // Add a literal pattern (no glob characters)
        matcher.add_pattern("src/lib.rs").unwrap();

        let result = matcher.match_with_details("src/lib.rs").unwrap();
        assert!(result.matched);
        assert_eq!(result.match_method, MatchMethod::Literal);
    }

    #[test]
    fn test_glob_matcher_match_all_single_pattern() {
        let mut matcher = GlobMatcher::new();
        // Only one pattern - should use fallback
        matcher.add_pattern("**/*.rs").unwrap();

        let matches = matcher.match_all("src/lib.rs").unwrap();
        assert_eq!(matches.len(), 1);
        assert!(matches.contains(&0));
    }

    #[test]
    fn test_glob_options_default() {
        let options = GlobOptions::default();
        assert!(options.case_sensitive);
        assert!(!options.literal_separator);
        assert!(!options.backslash_escape);
        assert!(options.cache_enabled);
        assert_eq!(options.cache_size_limit, 1000);
    }

    #[test]
    fn test_glob_match_result_fields() {
        let result = GlobMatchResult {
            matched: true,
            pattern_index: Some(0),
            pattern: Some("**/*.rs".to_string()),
            match_method: MatchMethod::Compiled,
        };

        assert!(result.matched);
        assert_eq!(result.pattern_index, Some(0));
        assert_eq!(result.pattern, Some("**/*.rs".to_string()));
        assert_eq!(result.match_method, MatchMethod::Compiled);
    }

    #[test]
    fn test_match_method_equality() {
        assert_eq!(MatchMethod::Cached, MatchMethod::Cached);
        assert_eq!(MatchMethod::Compiled, MatchMethod::Compiled);
        assert_eq!(MatchMethod::Individual, MatchMethod::Individual);
        assert_eq!(MatchMethod::Literal, MatchMethod::Literal);
        assert_ne!(MatchMethod::Cached, MatchMethod::Compiled);
    }

    #[test]
    fn test_glob_pattern_clone() {
        let pattern = GlobPattern::new("**/*.rs").unwrap();
        let cloned = pattern.clone();

        assert_eq!(pattern.pattern, cloned.pattern);
        assert_eq!(pattern.case_sensitive, cloned.case_sensitive);
    }

    #[test]
    fn test_glob_options_clone() {
        let options = GlobOptions {
            case_sensitive: false,
            cache_size_limit: 500,
            ..Default::default()
        };
        let cloned = options.clone();

        assert_eq!(options.case_sensitive, cloned.case_sensitive);
        assert_eq!(options.cache_size_limit, cloned.cache_size_limit);
    }

    #[test]
    fn test_empty_patterns_match_result() {
        let mut matcher = GlobMatcher::new();

        // Match with empty patterns should return not matched
        let result = matcher.match_with_details("any/path.rs").unwrap();
        assert!(!result.matched);
        assert!(result.pattern_index.is_none());
        assert!(result.pattern.is_none());
    }

    #[test]
    fn test_recompile_empty_patterns() {
        let mut matcher = GlobMatcher::new();
        // Recompile with no patterns should succeed
        matcher.recompile().unwrap();
        assert!(!matcher.is_compiled());
    }

    #[test]
    fn test_cache_eviction_during_match() {
        // Test that cache eviction actually happens during matching
        // when cache reaches limit (exercises lines 213-216)
        let mut matcher = GlobMatcher::with_options(GlobOptions {
            cache_enabled: true,
            cache_size_limit: 3,
            ..Default::default()
        });

        matcher.add_pattern("**/*").unwrap();

        // Fill the cache to the limit
        matcher.matches("file1.rs").unwrap();
        matcher.matches("file2.py").unwrap();
        matcher.matches("file3.js").unwrap();

        let (_, _, initial_size) = matcher.cache_stats();
        assert_eq!(initial_size, 3);

        // Add one more - this should trigger eviction
        matcher.matches("file4.ts").unwrap();

        // Cache size should be within limit
        let (_, _, final_size) = matcher.cache_stats();
        assert!(final_size <= 3);
    }

    #[test]
    fn test_empty_matcher_match_result() {
        // Test matching on an empty matcher (line 232)
        let mut matcher = GlobMatcher::new();

        // No patterns - should not match with Individual method
        let result = matcher.match_with_details("src/lib.rs").unwrap();
        assert!(!result.matched);
        assert_eq!(result.pattern_index, None);
        assert_eq!(result.pattern, None);
        assert_eq!(result.match_method, MatchMethod::Individual);
    }

    #[test]
    fn test_cached_match_result() {
        // Test that cached results return Cached match method (line 202-205)
        let mut matcher = GlobMatcher::with_options(GlobOptions {
            cache_enabled: true,
            ..Default::default()
        });

        matcher.add_pattern("**/*.rs").unwrap();

        // First match - not cached
        let result1 = matcher.match_with_details("src/lib.rs").unwrap();
        assert!(result1.matched);
        assert_ne!(result1.match_method, MatchMethod::Cached);

        // Second match - should be cached
        let result2 = matcher.match_with_details("src/lib.rs").unwrap();
        assert!(result2.matched);
        assert_eq!(result2.match_method, MatchMethod::Cached);
    }
}

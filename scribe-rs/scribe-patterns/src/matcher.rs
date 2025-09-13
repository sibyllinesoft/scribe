use std::path::{Path, PathBuf};
use std::collections::HashMap;
use anyhow::Result;
use crate::glob::{GlobMatcher, GlobOptions};
use crate::gitignore::GitignoreMatcher;

/// Combined pattern matching result
#[derive(Debug, Clone, PartialEq)]
pub enum MatchResult {
    /// File should be included
    Include,
    /// File should be excluded
    Exclude,
    /// File should be ignored (gitignore)
    Ignore,
    /// No explicit match - use default behavior
    NoMatch,
}

impl MatchResult {
    /// Check if the result indicates the file should be processed
    pub fn should_process(&self) -> bool {
        matches!(self, MatchResult::Include | MatchResult::NoMatch)
    }

    /// Check if the result indicates the file should be skipped
    pub fn should_skip(&self) -> bool {
        matches!(self, MatchResult::Exclude | MatchResult::Ignore)
    }
}

/// Options for combined pattern matching
#[derive(Debug, Clone)]
pub struct MatcherOptions {
    /// Whether to respect gitignore files
    pub respect_gitignore: bool,
    /// Whether pattern matching is case sensitive
    pub case_sensitive: bool,
    /// Whether to match hidden files by default
    pub include_hidden: bool,
    /// Custom gitignore file paths
    pub custom_gitignore_files: Vec<PathBuf>,
    /// Override gitignore patterns (always respected)
    pub override_patterns: Vec<String>,
}

impl Default for MatcherOptions {
    fn default() -> Self {
        Self {
            respect_gitignore: true,
            case_sensitive: true,
            include_hidden: false,
            custom_gitignore_files: Vec::new(),
            override_patterns: Vec::new(),
        }
    }
}

/// Combined pattern matcher that integrates glob and gitignore patterns
#[derive(Debug)]
pub struct PatternMatcher {
    /// Glob patterns for inclusion
    include_matcher: Option<GlobMatcher>,
    /// Glob patterns for exclusion
    exclude_matcher: Option<GlobMatcher>,
    /// Gitignore pattern matcher
    gitignore_matcher: Option<GitignoreMatcher>,
    /// Matcher options
    options: MatcherOptions,
    /// Cached results for performance
    cache: HashMap<PathBuf, MatchResult>,
    /// Cache hit statistics
    cache_hits: u64,
    /// Cache miss statistics
    cache_misses: u64,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new(options: MatcherOptions) -> Self {
        Self {
            include_matcher: None,
            exclude_matcher: None,
            gitignore_matcher: None,
            options,
            cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Create a pattern matcher with include patterns
    pub fn with_includes<I, S>(mut self, patterns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let glob_options = GlobOptions {
            case_sensitive: self.options.case_sensitive,
            ..Default::default()
        };

        let mut matcher = GlobMatcher::with_options(glob_options);
        for pattern in patterns {
            matcher.add_pattern(pattern.as_ref())?;
        }

        if !matcher.is_empty() {
            matcher.recompile()?;
            self.include_matcher = Some(matcher);
        }

        Ok(self)
    }

    /// Create a pattern matcher with exclude patterns
    pub fn with_excludes<I, S>(mut self, patterns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let glob_options = GlobOptions {
            case_sensitive: self.options.case_sensitive,
            ..Default::default()
        };

        let mut matcher = GlobMatcher::with_options(glob_options);
        for pattern in patterns {
            matcher.add_pattern(pattern.as_ref())?;
        }

        if !matcher.is_empty() {
            matcher.recompile()?;
            self.exclude_matcher = Some(matcher);
        }

        Ok(self)
    }

    /// Create a pattern matcher with gitignore support
    pub fn with_gitignore<P: AsRef<Path>>(mut self, base_path: P) -> Result<Self> {
        if self.options.respect_gitignore {
            let mut matcher = if self.options.case_sensitive {
                GitignoreMatcher::new()
            } else {
                GitignoreMatcher::case_insensitive()
            };
            
            // Load standard gitignore files from the directory tree
            let gitignore_files = GitignoreMatcher::discover_gitignore_files(base_path.as_ref())?;
            matcher.add_gitignore_files(gitignore_files)?;
            
            // Load custom gitignore files
            for path in &self.options.custom_gitignore_files {
                if path.exists() {
                    matcher.add_gitignore_file(path)?;
                }
            }

            // Add override patterns as regular patterns (they will take precedence due to order)
            for pattern in &self.options.override_patterns {
                matcher.add_pattern(pattern)?;
            }

            self.gitignore_matcher = Some(matcher);
        }

        Ok(self)
    }

    /// Check if a path matches the patterns
    pub fn is_match<P: AsRef<Path>>(&mut self, path: P) -> Result<MatchResult> {
        let path = path.as_ref();
        let canonical_path = path.to_path_buf();

        // Check cache first
        if let Some(cached_result) = self.cache.get(&canonical_path) {
            self.cache_hits += 1;
            return Ok(cached_result.clone());
        }

        self.cache_misses += 1;
        let result = self.compute_match(path)?;

        // Cache the result
        if self.cache.len() < 10000 { // Prevent unbounded cache growth
            self.cache.insert(canonical_path, result.clone());
        }

        Ok(result)
    }

    /// Compute the match result for a path
    fn compute_match(&mut self, path: &Path) -> Result<MatchResult> {
        // Check if it's a hidden file and we're not including hidden files
        if !self.options.include_hidden {
            if let Some(name) = path.file_name() {
                if let Some(name_str) = name.to_str() {
                    if name_str.starts_with('.') && name_str != ".." && name_str != "." {
                        return Ok(MatchResult::Exclude);
                    }
                }
            }
        }

        // Priority order:
        // 1. Gitignore patterns (if enabled) - can exclude
        // 2. Explicit exclude patterns - can exclude
        // 3. Explicit include patterns - can include
        // 4. Default behavior based on options

        // Check gitignore first (highest priority)
        if let Some(ref mut gitignore_matcher) = self.gitignore_matcher {
            if gitignore_matcher.is_ignored(path)? {
                return Ok(MatchResult::Ignore);
            }
        }

        // Check explicit exclude patterns
        if let Some(ref mut exclude_matcher) = self.exclude_matcher {
            if exclude_matcher.matches(path)? {
                return Ok(MatchResult::Exclude);
            }
        }

        // Check explicit include patterns
        if let Some(ref mut include_matcher) = self.include_matcher {
            if include_matcher.matches(path)? {
                return Ok(MatchResult::Include);
            }
            // If we have include patterns but no match, exclude by default
            return Ok(MatchResult::Exclude);
        }

        // No explicit patterns matched
        Ok(MatchResult::NoMatch)
    }

    /// Check if a path should be processed (not excluded or ignored)
    pub fn should_process<P: AsRef<Path>>(&mut self, path: P) -> Result<bool> {
        Ok(self.is_match(path)?.should_process())
    }

    /// Check if a path should be skipped (excluded or ignored)
    pub fn should_skip<P: AsRef<Path>>(&mut self, path: P) -> Result<bool> {
        Ok(self.is_match(path)?.should_skip())
    }

    /// Clear the internal cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (u64, u64, f64) {
        let total = self.cache_hits + self.cache_misses;
        let hit_rate = if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        };
        (self.cache_hits, self.cache_misses, hit_rate)
    }

    /// Check if the matcher has any patterns
    pub fn is_empty(&self) -> bool {
        self.include_matcher.as_ref().map_or(true, |m| m.is_empty()) &&
        self.exclude_matcher.as_ref().map_or(true, |m| m.is_empty()) &&
        self.gitignore_matcher.as_ref().map_or(true, |m| m.patterns().is_empty())
    }

    /// Get the number of patterns
    pub fn pattern_count(&self) -> usize {
        let include_count = self.include_matcher.as_ref().map_or(0, |m| m.pattern_count());
        let exclude_count = self.exclude_matcher.as_ref().map_or(0, |m| m.pattern_count());
        let gitignore_count = self.gitignore_matcher.as_ref().map_or(0, |m| m.patterns().len());
        include_count + exclude_count + gitignore_count
    }

    /// Compile all patterns for optimal performance
    pub fn compile(&mut self) -> Result<()> {
        if let Some(ref mut matcher) = self.include_matcher {
            matcher.recompile()?;
        }
        if let Some(ref mut matcher) = self.exclude_matcher {
            matcher.recompile()?;
        }
        // Gitignore matcher compiles automatically when patterns are added
        Ok(())
    }
}

/// Builder for creating pattern matchers with a fluent API
#[derive(Debug, Default)]
pub struct PatternMatcherBuilder {
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    options: MatcherOptions,
    base_path: Option<PathBuf>,
}

impl PatternMatcherBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add include patterns
    pub fn includes<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.include_patterns.extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    /// Add a single include pattern
    pub fn include<S: Into<String>>(mut self, pattern: S) -> Self {
        self.include_patterns.push(pattern.into());
        self
    }

    /// Add exclude patterns
    pub fn excludes<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_patterns.extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    /// Add a single exclude pattern
    pub fn exclude<S: Into<String>>(mut self, pattern: S) -> Self {
        self.exclude_patterns.push(pattern.into());
        self
    }

    /// Set whether to respect gitignore files
    pub fn respect_gitignore(mut self, respect: bool) -> Self {
        self.options.respect_gitignore = respect;
        self
    }


    /// Set case sensitivity
    pub fn case_sensitive(mut self, sensitive: bool) -> Self {
        self.options.case_sensitive = sensitive;
        self
    }

    /// Set whether to include hidden files
    pub fn include_hidden(mut self, include: bool) -> Self {
        self.options.include_hidden = include;
        self
    }

    /// Add custom gitignore files
    pub fn custom_gitignore_files<I, P>(mut self, files: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.options.custom_gitignore_files.extend(files.into_iter().map(|p| p.into()));
        self
    }

    /// Add override patterns
    pub fn override_patterns<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options.override_patterns.extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    /// Set the base path for gitignore resolution
    pub fn base_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.base_path = Some(path.into());
        self
    }

    /// Build the pattern matcher
    pub fn build(self) -> Result<PatternMatcher> {
        let mut matcher = PatternMatcher::new(self.options);

        // Add include patterns
        if !self.include_patterns.is_empty() {
            matcher = matcher.with_includes(self.include_patterns)?;
        }

        // Add exclude patterns
        if !self.exclude_patterns.is_empty() {
            matcher = matcher.with_excludes(self.exclude_patterns)?;
        }

        // Set up gitignore if base path is provided
        if let Some(base_path) = self.base_path {
            matcher = matcher.with_gitignore(base_path)?;
        }

        // Compile patterns for optimal performance
        matcher.compile()?;

        Ok(matcher)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_files(dir: &Path) -> Result<()> {
        // Create various test files
        fs::write(dir.join("test.rs"), "// Rust file")?;
        fs::write(dir.join("test.py"), "# Python file")?;
        fs::write(dir.join("README.md"), "# Documentation")?;
        fs::write(dir.join(".hidden"), "hidden file")?;
        
        // Create subdirectory
        let subdir = dir.join("src");
        fs::create_dir(&subdir)?;
        fs::write(subdir.join("main.rs"), "fn main() {}")?;
        fs::write(subdir.join("lib.rs"), "// Library")?;
        
        // Create .gitignore
        fs::write(dir.join(".gitignore"), "*.tmp\ntarget/\n.DS_Store")?;
        
        // Create ignored files
        fs::write(dir.join("test.tmp"), "temporary file")?;
        fs::write(dir.join(".DS_Store"), "system file")?;

        Ok(())
    }

    #[test]
    fn test_basic_matching() -> Result<()> {
        let temp_dir = TempDir::new()?;
        create_test_files(temp_dir.path())?;

        let mut matcher = PatternMatcherBuilder::new()
            .include("*.rs")
            .exclude("**/target/**")
            .base_path(temp_dir.path())
            .build()?;

        // Should match Rust files
        assert!(matcher.should_process("test.rs")?);
        assert!(matcher.should_process("src/main.rs")?);
        
        // Should not match other files
        assert!(!matcher.should_process("test.py")?);
        assert!(!matcher.should_process("README.md")?);

        Ok(())
    }

    #[test]
    fn test_gitignore_integration() -> Result<()> {
        let temp_dir = TempDir::new()?;
        create_test_files(temp_dir.path())?;

        let mut matcher = PatternMatcherBuilder::new()
            .respect_gitignore(true)
            .base_path(temp_dir.path())
            .build()?;

        // Should ignore files matching gitignore
        assert!(matcher.should_skip("test.tmp")?);
        assert!(matcher.should_skip(".DS_Store")?);
        
        // Should not ignore regular files
        assert!(matcher.should_process("test.rs")?);
        assert!(matcher.should_process("README.md")?);

        Ok(())
    }

    #[test]
    fn test_hidden_files() -> Result<()> {
        let temp_dir = TempDir::new()?;
        create_test_files(temp_dir.path())?;

        // Without include_hidden
        let mut matcher = PatternMatcherBuilder::new()
            .include_hidden(false)
            .build()?;
        
        assert!(matcher.should_skip(".hidden")?);

        // With include_hidden
        let mut matcher = PatternMatcherBuilder::new()
            .include_hidden(true)
            .build()?;
        
        assert!(matcher.should_process(".hidden")?);

        Ok(())
    }

    #[test]
    fn test_pattern_priority() -> Result<()> {
        let temp_dir = TempDir::new()?;
        create_test_files(temp_dir.path())?;

        let mut matcher = PatternMatcherBuilder::new()
            .include("*.rs")
            .exclude("**/target/**")
            .respect_gitignore(true)
            .base_path(temp_dir.path())
            .build()?;

        // Gitignore should take priority over include patterns
        fs::write(temp_dir.path().join("ignored.rs"), "// Ignored Rust file")?;
        fs::write(temp_dir.path().join(".gitignore"), "ignored.rs")?;
        
        // Rebuild matcher to pick up new gitignore
        let mut matcher = PatternMatcherBuilder::new()
            .include("*.rs")
            .respect_gitignore(true)
            .base_path(temp_dir.path())
            .build()?;

        assert_eq!(matcher.is_match("ignored.rs")?, MatchResult::Ignore);

        Ok(())
    }

    #[test]
    fn test_cache_functionality() -> Result<()> {
        let mut matcher = PatternMatcherBuilder::new()
            .include("*.rs")
            .build()?;

        // First call should be a cache miss
        let _ = matcher.is_match("test.rs")?;
        let (hits, misses, _) = matcher.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);

        // Second call should be a cache hit
        let _ = matcher.is_match("test.rs")?;
        let (hits, misses, hit_rate) = matcher.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(hit_rate, 0.5);

        // Clear cache
        matcher.clear_cache();
        let (hits, misses, _) = matcher.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);

        Ok(())
    }

    #[test]
    fn test_empty_matcher() -> Result<()> {
        let matcher = PatternMatcherBuilder::new().build()?;
        
        assert!(matcher.is_empty());
        assert_eq!(matcher.pattern_count(), 0);

        Ok(())
    }

    #[test]
    fn test_case_sensitivity() -> Result<()> {
        // Case sensitive
        let mut matcher = PatternMatcherBuilder::new()
            .include("*.RS")
            .case_sensitive(true)
            .build()?;
        
        assert!(!matcher.should_process("test.rs")?);
        assert!(matcher.should_process("test.RS")?);

        // Case insensitive
        let mut matcher = PatternMatcherBuilder::new()
            .include("*.RS")
            .case_sensitive(false)
            .build()?;
        
        assert!(matcher.should_process("test.rs")?);
        assert!(matcher.should_process("test.RS")?);

        Ok(())
    }

    #[test]
    fn test_override_patterns() -> Result<()> {
        let temp_dir = TempDir::new()?;
        create_test_files(temp_dir.path())?;

        let mut matcher = PatternMatcherBuilder::new()
            .respect_gitignore(true)
            .override_patterns(vec!["!*.tmp".to_string()]) // Override gitignore
            .base_path(temp_dir.path())
            .build()?;

        // Should not ignore .tmp files due to override
        assert!(matcher.should_process("test.tmp")?);

        Ok(())
    }
}
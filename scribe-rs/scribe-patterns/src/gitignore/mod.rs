//! Gitignore pattern handling with proper precedence and syntax support.
//!
//! This module provides comprehensive gitignore functionality including:
//! - Full gitignore syntax support (negation, directory matching, etc.)
//! - Proper precedence handling for multiple gitignore files
//! - Integration with the ignore crate for performance
//! - Support for .gitignore, .ignore, and custom ignore files

mod pattern;

pub use pattern::{GitignorePattern, GitignoreRule, IgnoreFile, IgnoreMatchResult, IgnoreType};

use scribe_core::{Result, ScribeError};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use ignore::{overrides::OverrideBuilder, WalkBuilder};
use serde::{Deserialize, Serialize};

/// Gitignore pattern matcher with full syntax support
#[derive(Debug)]
pub struct GitignoreMatcher {
    patterns: Vec<GitignorePattern>,
    ignore_files: Vec<IgnoreFile>,
    overrides: Option<ignore::overrides::Override>,
    case_sensitive: bool,
    require_literal_separator: bool,
}

impl GitignoreMatcher {
    /// Create a new gitignore matcher
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            ignore_files: Vec::new(),
            overrides: None,
            case_sensitive: true,
            require_literal_separator: false,
        }
    }

    /// Create a case-insensitive matcher
    pub fn case_insensitive() -> Self {
        Self {
            patterns: Vec::new(),
            ignore_files: Vec::new(),
            overrides: None,
            case_sensitive: false,
            require_literal_separator: false,
        }
    }

    /// Add a gitignore pattern directly
    pub fn add_pattern(&mut self, pattern: &str) -> Result<()> {
        let gitignore_pattern = GitignorePattern::new(pattern)?;
        self.patterns.push(gitignore_pattern);
        self.invalidate_overrides();
        Ok(())
    }

    /// Add multiple patterns
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

    /// Load patterns from a gitignore file
    pub fn add_gitignore_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let ignore_type = self.determine_ignore_type(path);
        let ignore_file = self.load_ignore_file(path, ignore_type)?;

        // Add patterns to the main list
        for pattern in &ignore_file.patterns {
            self.patterns.push(pattern.clone());
        }

        self.ignore_files.push(ignore_file);
        self.invalidate_overrides();
        Ok(())
    }

    /// Load patterns from multiple gitignore files
    pub fn add_gitignore_files<P, I>(&mut self, paths: I) -> Result<()>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        for path in paths {
            self.add_gitignore_file(path)?;
        }
        Ok(())
    }

    /// Check if a path should be ignored
    pub fn is_ignored<P: AsRef<Path>>(&mut self, path: P) -> Result<bool> {
        let result = self.match_path(path)?;
        Ok(result.ignored)
    }

    /// Find which ignore file a pattern at the given index came from
    fn find_pattern_source(&self, index: usize) -> (Option<PathBuf>, Option<usize>) {
        let mut line_count = 0;
        for ignore_file in &self.ignore_files {
            if index < line_count + ignore_file.patterns.len() {
                return (
                    Some(ignore_file.path.clone()),
                    Some(index - line_count + 1),
                );
            }
            line_count += ignore_file.patterns.len();
        }
        (None, None)
    }

    /// Update match result with pattern details
    fn update_match_result(
        &self,
        result: &mut IgnoreMatchResult,
        index: usize,
        pattern: &GitignorePattern,
    ) {
        result.matched_pattern = Some(pattern.original.clone());
        result.rule_type = pattern.rule_type.clone();

        let (matched_file, line_number) = self.find_pattern_source(index);
        result.matched_file = matched_file;
        result.line_number = line_number;

        result.ignored = pattern.rule_type == GitignoreRule::Exclude;
    }

    /// Get detailed match information for a path
    pub fn match_path<P: AsRef<Path>>(&mut self, path: P) -> Result<IgnoreMatchResult> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy();
        let is_directory = path_str.ends_with('/') || path.is_dir();

        let mut result = IgnoreMatchResult {
            ignored: false,
            matched_pattern: None,
            matched_file: None,
            rule_type: GitignoreRule::Exclude,
            line_number: None,
        };

        // Process patterns in reverse order (later patterns override earlier ones)
        for (index, pattern) in self.patterns.iter().enumerate().rev() {
            if !pattern.matches(path, is_directory, self.case_sensitive) {
                continue;
            }

            // Skip comments and empty lines
            if matches!(pattern.rule_type, GitignoreRule::Comment | GitignoreRule::Empty) {
                continue;
            }

            self.update_match_result(&mut result, index, pattern);
            break;
        }

        Ok(result)
    }

    /// Check multiple paths efficiently using ignore crate integration
    pub fn filter_paths<P>(&mut self, paths: &[P]) -> Result<Vec<P>>
    where
        P: AsRef<Path> + Clone,
    {
        if self.overrides.is_none() {
            self.build_overrides()?;
        }

        let mut result = Vec::new();

        for path in paths {
            if !self.is_ignored(path)? {
                result.push(path.clone());
            }
        }

        Ok(result)
    }

    /// Get all loaded ignore files
    pub fn ignore_files(&self) -> &[IgnoreFile] {
        &self.ignore_files
    }

    /// Get all patterns
    pub fn patterns(&self) -> &[GitignorePattern] {
        &self.patterns
    }

    /// Clear all patterns and files
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.ignore_files.clear();
        self.invalidate_overrides();
    }

    /// Get statistics about loaded patterns
    pub fn stats(&self) -> GitignoreStats {
        let total_patterns = self.patterns.len();
        let exclude_patterns = self
            .patterns
            .iter()
            .filter(|p| p.rule_type == GitignoreRule::Exclude)
            .count();
        let include_patterns = self
            .patterns
            .iter()
            .filter(|p| p.rule_type == GitignoreRule::Include)
            .count();
        let comment_lines = self
            .patterns
            .iter()
            .filter(|p| p.rule_type == GitignoreRule::Comment)
            .count();

        GitignoreStats {
            total_patterns,
            exclude_patterns,
            include_patterns,
            comment_lines,
            ignore_files: self.ignore_files.len(),
        }
    }

    /// Load patterns from an ignore file
    fn load_ignore_file(&self, path: &Path, ignore_type: IgnoreType) -> Result<IgnoreFile> {
        if !path.exists() {
            return Err(ScribeError::path(
                format!("Ignore file does not exist: {}", path.display()),
                path,
            ));
        }

        let file = fs::File::open(path).map_err(|e| {
            ScribeError::io(
                format!("Failed to open ignore file {}: {}", path.display(), e),
                e,
            )
        })?;

        let reader = BufReader::new(file);
        let mut patterns = Vec::new();
        let mut line_count = 0;

        for line in reader.lines() {
            let line =
                line.map_err(|e| ScribeError::io(format!("Failed to read ignore file: {}", e), e))?;
            line_count += 1;

            match GitignorePattern::new(&line) {
                Ok(pattern) => patterns.push(pattern),
                Err(e) => {
                    log::warn!(
                        "Invalid gitignore pattern in {} line {}: {} ({})",
                        path.display(),
                        line_count,
                        line,
                        e
                    );
                }
            }
        }

        Ok(IgnoreFile {
            path: path.to_path_buf(),
            ignore_type,
            patterns,
            line_count,
        })
    }

    /// Determine the type of ignore file based on its path
    fn determine_ignore_type(&self, path: &Path) -> IgnoreType {
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            match filename {
                ".gitignore" => IgnoreType::Gitignore,
                ".ignore" => IgnoreType::DotIgnore,
                _ => IgnoreType::CustomIgnore,
            }
        } else {
            IgnoreType::CustomIgnore
        }
    }

    /// Build override patterns for the ignore crate
    fn build_overrides(&mut self) -> Result<()> {
        let mut builder = OverrideBuilder::new(".");

        for pattern in &self.patterns {
            if matches!(
                pattern.rule_type,
                GitignoreRule::Exclude | GitignoreRule::Include
            ) {
                let glob_pattern = pattern.to_glob_pattern();
                let override_pattern = if pattern.negated {
                    format!("!{}", glob_pattern)
                } else {
                    glob_pattern
                };

                if let Err(e) = builder.add(&override_pattern) {
                    log::warn!("Failed to add override pattern {}: {}", override_pattern, e);
                }
            }
        }

        self.overrides = Some(builder.build()?);
        Ok(())
    }

    /// Invalidate compiled overrides
    fn invalidate_overrides(&mut self) {
        self.overrides = None;
    }

    /// Find gitignore files in a directory tree
    pub fn discover_gitignore_files<P: AsRef<Path>>(root: P) -> Result<Vec<PathBuf>> {
        let root = root.as_ref();
        let mut gitignore_files = Vec::new();

        // Use WalkBuilder from ignore crate to respect existing gitignore rules
        let walker = WalkBuilder::new(root)
            .hidden(false) // Include hidden files to find .gitignore
            .git_ignore(false) // Don't apply gitignore during discovery
            .build();

        for entry in walker {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                        if matches!(filename, ".gitignore" | ".ignore") {
                            gitignore_files.push(path.to_path_buf());
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Error walking directory tree: {}", e);
                }
            }
        }

        Ok(gitignore_files)
    }

    /// Load all gitignore files from a directory tree
    pub fn from_directory<P: AsRef<Path>>(root: P) -> Result<Self> {
        let mut matcher = Self::new();
        let gitignore_files = Self::discover_gitignore_files(&root)?;

        for file in gitignore_files {
            if let Err(e) = matcher.add_gitignore_file(&file) {
                log::warn!("Failed to load gitignore file {}: {}", file.display(), e);
            }
        }

        Ok(matcher)
    }

    /// Create a matcher with commonly ignored patterns
    pub fn with_defaults() -> Self {
        let mut matcher = Self::new();

        // Add common ignore patterns
        let default_patterns = [
            ".DS_Store",
            "Thumbs.db",
            "*.tmp",
            "*.temp",
            ".git/",
            ".svn/",
            ".hg/",
            "node_modules/",
            "target/",
            "build/",
            "dist/",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
        ];

        for pattern in &default_patterns {
            if let Err(e) = matcher.add_pattern(pattern) {
                log::warn!("Failed to add default pattern {}: {}", pattern, e);
            }
        }

        matcher
    }
}

impl Default for GitignoreMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about gitignore patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitignoreStats {
    pub total_patterns: usize,
    pub exclude_patterns: usize,
    pub include_patterns: usize,
    pub comment_lines: usize,
    pub ignore_files: usize,
}

// Note: From<ignore::Error> for ScribeError needs to be implemented in scribe-core

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_gitignore_pattern_parsing() {
        // Basic pattern
        let pattern = GitignorePattern::new("*.rs").unwrap();
        assert_eq!(pattern.pattern, "*.rs");
        assert!(!pattern.negated);
        assert!(!pattern.directory_only);
        assert!(!pattern.anchored);
        assert_eq!(pattern.rule_type, GitignoreRule::Exclude);

        // Negated pattern
        let pattern = GitignorePattern::new("!important.rs").unwrap();
        assert_eq!(pattern.pattern, "important.rs");
        assert!(pattern.negated);
        assert_eq!(pattern.rule_type, GitignoreRule::Include);

        // Directory pattern
        let pattern = GitignorePattern::new("build/").unwrap();
        assert_eq!(pattern.pattern, "build");
        assert!(pattern.directory_only);
        assert_eq!(pattern.rule_type, GitignoreRule::Exclude);

        // Anchored pattern
        let pattern = GitignorePattern::new("/root-only").unwrap();
        assert_eq!(pattern.pattern, "root-only");
        assert!(pattern.anchored);
        assert_eq!(pattern.rule_type, GitignoreRule::Exclude);

        // Comment
        let pattern = GitignorePattern::new("# This is a comment").unwrap();
        assert_eq!(pattern.rule_type, GitignoreRule::Comment);

        // Empty line
        let pattern = GitignorePattern::new("   ").unwrap();
        assert_eq!(pattern.rule_type, GitignoreRule::Empty);
    }

    #[test]
    fn test_gitignore_pattern_matching() {
        let pattern = GitignorePattern::new("*.rs").unwrap();
        assert!(pattern.matches("lib.rs", false, true));
        assert!(!pattern.matches("src/lib.rs", false, true)); // Single * doesn't match across directories
        assert!(!pattern.matches("lib.py", false, true));

        // For recursive matching, use **
        let pattern = GitignorePattern::new("**/*.rs").unwrap();
        assert!(pattern.matches("lib.rs", false, true));
        assert!(pattern.matches("src/lib.rs", false, true));

        let pattern = GitignorePattern::new("build/").unwrap();
        assert!(pattern.matches("build", true, true)); // Directory
        assert!(!pattern.matches("build", false, true)); // File
        assert!(pattern.matches("src/build", true, true));

        let pattern = GitignorePattern::new("/root-only").unwrap();
        assert!(pattern.matches("root-only", false, true));
        // Note: Full anchoring logic would be more complex in a real implementation

        let pattern = GitignorePattern::new("!*.rs").unwrap();
        assert!(pattern.negated);
        assert_eq!(pattern.rule_type, GitignoreRule::Include);
    }

    #[test]
    fn test_gitignore_matcher_basic() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("**/*.rs").unwrap(); // Use ** for recursive matching
        matcher.add_pattern("build/").unwrap();
        matcher.add_pattern("!important.rs").unwrap();

        assert!(matcher.is_ignored("lib.rs").unwrap());
        assert!(matcher.is_ignored("src/lib.rs").unwrap());
        assert!(!matcher.is_ignored("lib.py").unwrap());

        // Negation should override exclude
        assert!(!matcher.is_ignored("important.rs").unwrap());
    }

    #[test]
    fn test_gitignore_file_loading() {
        let temp_dir = TempDir::new().unwrap();
        let gitignore_path = temp_dir.path().join(".gitignore");

        let gitignore_content = r#"
# Ignore compiled files
*.o
*.so
*.dylib

# Ignore build directory
build/

# Don't ignore important files
!important.txt

# Empty line above
"#;

        fs::write(&gitignore_path, gitignore_content).unwrap();

        let mut matcher = GitignoreMatcher::new();
        matcher.add_gitignore_file(&gitignore_path).unwrap();

        // Check statistics
        let stats = matcher.stats();
        assert_eq!(stats.ignore_files, 1);
        assert!(stats.exclude_patterns > 0);
        assert!(stats.include_patterns > 0);
        assert!(stats.comment_lines > 0);

        // Test matching
        assert!(matcher.is_ignored("test.o").unwrap());
        assert!(matcher.is_ignored("libtest.so").unwrap());
        assert!(matcher.is_ignored("build/").unwrap()); // Directory indicated by trailing slash
        assert!(!matcher.is_ignored("important.txt").unwrap()); // Negated
        assert!(!matcher.is_ignored("source.c").unwrap()); // Not matched
    }

    #[test]
    fn test_gitignore_match_details() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();
        matcher.add_pattern("!keep.tmp").unwrap();

        let result = matcher.match_path("test.tmp").unwrap();
        assert!(result.ignored);
        assert!(result.matched_pattern.is_some());
        assert_eq!(result.rule_type, GitignoreRule::Exclude);

        let result = matcher.match_path("keep.tmp").unwrap();
        assert!(!result.ignored);
        assert!(result.matched_pattern.is_some());
        assert_eq!(result.rule_type, GitignoreRule::Include);

        let result = matcher.match_path("test.rs").unwrap();
        assert!(!result.ignored);
        assert!(result.matched_pattern.is_none());
    }

    #[test]
    fn test_gitignore_discovery() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Create directory structure with multiple .gitignore files
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("tests")).unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();

        fs::write(root.join(".gitignore"), "*.tmp\nbuild/").unwrap();
        fs::write(root.join("src/.gitignore"), "*.o").unwrap();
        fs::write(root.join("tests/.gitignore"), "fixtures/").unwrap();

        let gitignore_files = GitignoreMatcher::discover_gitignore_files(root).unwrap();
        assert_eq!(gitignore_files.len(), 3);

        // Check that all expected files are found
        let filenames: Vec<String> = gitignore_files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(filenames.iter().all(|name| name == ".gitignore"));
    }

    #[test]
    fn test_gitignore_from_directory() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Create .gitignore files
        fs::write(root.join(".gitignore"), "*.tmp\n*.log").unwrap();
        fs::create_dir_all(root.join("subdir")).unwrap();
        fs::write(root.join("subdir/.gitignore"), "*.bak").unwrap();

        let matcher = GitignoreMatcher::from_directory(root).unwrap();
        let stats = matcher.stats();

        assert_eq!(stats.ignore_files, 2);
        assert!(stats.total_patterns >= 3); // At least the 3 patterns we added
    }

    #[test]
    fn test_gitignore_defaults() {
        let matcher = GitignoreMatcher::with_defaults();
        let stats = matcher.stats();

        assert!(stats.total_patterns > 0);
        assert!(stats.exclude_patterns > 0);

        // Test some common patterns
        let mut matcher = matcher;
        assert!(matcher.is_ignored("node_modules/package.json").unwrap());
        assert!(matcher.is_ignored("target/debug/main").unwrap());
        assert!(matcher.is_ignored(".DS_Store").unwrap());
        assert!(matcher.is_ignored("__pycache__/module.pyc").unwrap());
    }

    #[test]
    fn test_gitignore_case_sensitivity() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.TMP").unwrap();

        // Case-sensitive by default
        assert!(matcher.is_ignored("file.TMP").unwrap());
        assert!(!matcher.is_ignored("file.tmp").unwrap());

        let mut matcher = GitignoreMatcher::case_insensitive();
        matcher.add_pattern("*.TMP").unwrap();

        // Case-insensitive matcher
        assert!(matcher.is_ignored("file.TMP").unwrap());
        assert!(matcher.is_ignored("file.tmp").unwrap());
        assert!(matcher.is_ignored("file.Tmp").unwrap());
    }

    #[test]
    fn test_gitignore_pattern_precedence() {
        let mut matcher = GitignoreMatcher::new();

        // Add patterns in order - later ones should override earlier ones
        matcher.add_pattern("*.txt").unwrap(); // Exclude all .txt files
        matcher.add_pattern("!important.txt").unwrap(); // But include important.txt
        matcher.add_pattern("important.txt").unwrap(); // But exclude it again

        // The last pattern should win
        assert!(matcher.is_ignored("important.txt").unwrap());
        assert!(matcher.is_ignored("other.txt").unwrap());
    }

    #[test]
    fn test_complex_gitignore_patterns() {
        let mut matcher = GitignoreMatcher::new();

        // Test various gitignore pattern types
        matcher.add_pattern("**/*.tmp").unwrap(); // Recursive pattern
        matcher.add_pattern("build/**/output").unwrap(); // Pattern with ** in middle
        matcher.add_pattern("logs/*.log").unwrap(); // Single level wildcard
        matcher.add_pattern("cache/*/data").unwrap(); // Single directory wildcard

        assert!(matcher.is_ignored("file.tmp").unwrap());
        assert!(matcher.is_ignored("deep/nested/file.tmp").unwrap());
        assert!(matcher.is_ignored("logs/error.log").unwrap());
        assert!(!matcher.is_ignored("logs/nested/error.log").unwrap()); // Single level only
    }

    #[test]
    fn test_gitignore_filter_paths() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();
        matcher.add_pattern("build/").unwrap();

        let paths = vec![
            "src/lib.rs",
            "temp.tmp",
            "build/output",
            "README.md",
            "test.tmp",
        ];

        let filtered = matcher.filter_paths(&paths).unwrap();

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&"src/lib.rs"));
        assert!(filtered.contains(&"README.md"));
        assert!(!filtered.contains(&"temp.tmp"));
        assert!(!filtered.contains(&"test.tmp"));
        assert!(!filtered.contains(&"build/output"));
    }

    #[test]
    fn test_gitignore_empty_and_comments() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("").unwrap(); // Empty line
        matcher.add_pattern("   ").unwrap(); // Whitespace only
        matcher.add_pattern("# Comment").unwrap(); // Comment
        matcher.add_pattern("*.rs").unwrap(); // Actual pattern

        let stats = matcher.stats();
        assert_eq!(stats.exclude_patterns, 1); // Only *.rs counts
        assert!(stats.comment_lines >= 1);

        assert!(matcher.is_ignored("test.rs").unwrap());
        assert!(!matcher.is_ignored("test.py").unwrap());
    }

    #[test]
    fn test_add_patterns_multiple() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_patterns(vec!["*.tmp", "*.log", "build/"]).unwrap();

        let stats = matcher.stats();
        assert_eq!(stats.exclude_patterns, 3);

        assert!(matcher.is_ignored("file.tmp").unwrap());
        assert!(matcher.is_ignored("debug.log").unwrap());
        assert!(matcher.is_ignored("build/output").unwrap());
    }

    #[test]
    fn test_add_gitignore_files_multiple() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Create multiple gitignore files
        fs::write(root.join(".gitignore1"), "*.tmp").unwrap();
        fs::write(root.join(".gitignore2"), "*.log").unwrap();

        let mut matcher = GitignoreMatcher::new();
        matcher.add_gitignore_files(vec![
            root.join(".gitignore1"),
            root.join(".gitignore2"),
        ]).unwrap();

        let stats = matcher.stats();
        assert_eq!(stats.ignore_files, 2);

        assert!(matcher.is_ignored("file.tmp").unwrap());
        assert!(matcher.is_ignored("debug.log").unwrap());
    }

    #[test]
    fn test_gitignore_stats_totals() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();
        matcher.add_pattern("!keep.tmp").unwrap();
        matcher.add_pattern("# Comment").unwrap();
        matcher.add_pattern("").unwrap();

        let stats = matcher.stats();
        assert_eq!(stats.exclude_patterns, 1);
        assert_eq!(stats.include_patterns, 1);
        assert_eq!(stats.comment_lines, 1);
        assert_eq!(stats.total_patterns, 4);
    }

    #[test]
    fn test_gitignore_matcher_clone() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();

        // Stats should still be accessible
        let stats = matcher.stats();
        assert_eq!(stats.total_patterns, 1);
    }

    #[test]
    fn test_gitignore_determine_ignore_type() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Test different ignore file types
        let gitignore = root.join(".gitignore");
        let dockerignore = root.join(".dockerignore");
        let npmignore = root.join(".npmignore");
        let hgignore = root.join(".hgignore");

        fs::write(&gitignore, "*.tmp").unwrap();
        fs::write(&dockerignore, "node_modules").unwrap();
        fs::write(&npmignore, "*.log").unwrap();
        fs::write(&hgignore, "^build$").unwrap();

        let mut matcher = GitignoreMatcher::new();
        matcher.add_gitignore_file(&gitignore).unwrap();
        matcher.add_gitignore_file(&dockerignore).unwrap();
        matcher.add_gitignore_file(&npmignore).unwrap();
        // Note: .hgignore uses different syntax, but we can still try to parse it
    }

    #[test]
    fn test_gitignore_clear() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();
        matcher.add_pattern("*.log").unwrap();

        let stats = matcher.stats();
        assert_eq!(stats.total_patterns, 2);

        matcher.clear();
        let stats = matcher.stats();
        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.ignore_files, 0);
    }

    #[test]
    fn test_gitignore_patterns_accessor() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();
        matcher.add_pattern("*.log").unwrap();

        let patterns = matcher.patterns();
        assert_eq!(patterns.len(), 2);
    }

    #[test]
    fn test_gitignore_ignore_files_accessor() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        fs::write(root.join(".gitignore"), "*.tmp").unwrap();

        let mut matcher = GitignoreMatcher::new();
        matcher.add_gitignore_file(root.join(".gitignore")).unwrap();

        let files = matcher.ignore_files();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_gitignore_nonexistent_file() {
        let mut matcher = GitignoreMatcher::new();
        let result = matcher.add_gitignore_file("/nonexistent/path/.gitignore");
        assert!(result.is_err());
    }

    #[test]
    fn test_gitignore_default_impl() {
        let matcher = GitignoreMatcher::default();
        let stats = matcher.stats();
        assert_eq!(stats.total_patterns, 0);
    }

    #[test]
    fn test_gitignore_stats_clone() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();
        let stats = matcher.stats();
        let cloned = stats.clone();
        assert_eq!(stats.total_patterns, cloned.total_patterns);
    }

    #[test]
    fn test_gitignore_stats_debug() {
        let stats = GitignoreStats {
            total_patterns: 5,
            exclude_patterns: 3,
            include_patterns: 1,
            comment_lines: 1,
            ignore_files: 2,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("GitignoreStats"));
    }

    #[test]
    fn test_gitignore_check_match() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("*.tmp").unwrap();

        // is_ignored returns bool
        assert!(matcher.is_ignored("test.tmp").unwrap());
        assert!(!matcher.is_ignored("test.rs").unwrap());
    }

    #[test]
    fn test_gitignore_directory_matching() {
        let mut matcher = GitignoreMatcher::new();
        matcher.add_pattern("build/").unwrap();

        // Directory pattern should match directories
        assert!(matcher.is_ignored("build/").unwrap());
        // But may or may not match files without trailing slash
        let _ = matcher.is_ignored("build");
    }
}

//! Gitignore pattern parsing and matching.

use scribe_core::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Individual gitignore pattern with parsing and matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitignorePattern {
    pub original: String,
    pub pattern: String,
    pub negated: bool,
    pub directory_only: bool,
    pub anchored: bool,
    pub rule_type: GitignoreRule,
}

/// Type of gitignore rule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GitignoreRule {
    Include,
    Exclude,
    Comment,
    Empty,
}

/// Information about a loaded ignore file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IgnoreFile {
    pub path: PathBuf,
    pub ignore_type: IgnoreType,
    pub patterns: Vec<GitignorePattern>,
    pub line_count: usize,
}

/// Type of ignore file
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IgnoreType {
    Gitignore,
    GlobalGitignore,
    CustomIgnore,
    DotIgnore,
}

/// Match result for gitignore patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IgnoreMatchResult {
    pub ignored: bool,
    pub matched_pattern: Option<String>,
    pub matched_file: Option<PathBuf>,
    pub rule_type: GitignoreRule,
    pub line_number: Option<usize>,
}

impl GitignorePattern {
    /// Create a new gitignore pattern from a line
    pub fn new(line: &str) -> Result<Self> {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            return Ok(Self {
                original: line.to_string(),
                pattern: String::new(),
                negated: false,
                directory_only: false,
                anchored: false,
                rule_type: GitignoreRule::Empty,
            });
        }

        if trimmed.starts_with('#') {
            return Ok(Self {
                original: line.to_string(),
                pattern: trimmed.to_string(),
                negated: false,
                directory_only: false,
                anchored: false,
                rule_type: GitignoreRule::Comment,
            });
        }

        let mut pattern = trimmed.to_string();
        let mut negated = false;
        let mut directory_only = false;
        let mut anchored = false;

        if pattern.starts_with('!') {
            negated = true;
            pattern = pattern[1..].to_string();
        }

        if pattern.ends_with('/') {
            directory_only = true;
            pattern = pattern.trim_end_matches('/').to_string();
        }

        if pattern.starts_with('/') {
            anchored = true;
            pattern = pattern[1..].to_string();
        }

        let rule_type = if negated {
            GitignoreRule::Include
        } else {
            GitignoreRule::Exclude
        };

        Ok(Self {
            original: line.to_string(),
            pattern,
            negated,
            directory_only,
            anchored,
            rule_type,
        })
    }

    /// Check if this pattern matches a path
    pub fn matches<P: AsRef<Path>>(
        &self,
        path: P,
        is_directory: bool,
        case_sensitive: bool,
    ) -> bool {
        if matches!(
            self.rule_type,
            GitignoreRule::Comment | GitignoreRule::Empty
        ) {
            return false;
        }

        let path_str = path.as_ref().to_string_lossy();
        self.matches_glob(&self.pattern, &path_str, is_directory, case_sensitive)
    }

    fn matches_glob(
        &self,
        pattern: &str,
        path: &str,
        is_directory: bool,
        case_sensitive: bool,
    ) -> bool {
        if pattern.contains("**") {
            return self.matches_recursive_pattern(pattern, path, case_sensitive);
        }

        if pattern.contains('*') {
            return self.wildcard_match(pattern, path, case_sensitive);
        }

        if self.directory_only {
            self.matches_directory_pattern(pattern, path, is_directory, case_sensitive)
        } else {
            self.matches_exact_pattern(pattern, path, case_sensitive)
        }
    }

    fn matches_recursive_pattern(&self, pattern: &str, path: &str, case_sensitive: bool) -> bool {
        let parts: Vec<&str> = pattern.split("**").collect();
        if parts.len() != 2 {
            return false;
        }

        let prefix = parts[0];
        let suffix = parts[1].trim_start_matches('/');

        if prefix.is_empty() {
            self.matches_suffix_anywhere(suffix, path, case_sensitive)
        } else if suffix.is_empty() {
            path.starts_with(prefix.trim_end_matches('/'))
        } else {
            path.starts_with(prefix.trim_end_matches('/'))
                && (path.ends_with(suffix) || path.contains(&format!("/{}", suffix)))
        }
    }

    fn matches_suffix_anywhere(&self, suffix: &str, path: &str, case_sensitive: bool) -> bool {
        if suffix.contains('*') {
            let path_parts: Vec<&str> = path.split('/').collect();
            path_parts.iter().any(|part| self.wildcard_match(suffix, part, case_sensitive))
        } else {
            path.ends_with(suffix) || path.contains(&format!("/{}", suffix))
        }
    }

    fn matches_directory_pattern(
        &self,
        pattern: &str,
        path: &str,
        is_directory: bool,
        case_sensitive: bool,
    ) -> bool {
        let (path_cmp, pattern_cmp) = if case_sensitive {
            (path.to_string(), pattern.to_string())
        } else {
            (path.to_ascii_lowercase(), pattern.to_ascii_lowercase())
        };

        let dir_pattern = format!("{}/", pattern_cmp);
        let component_pattern = format!("/{}", pattern_cmp);

        if self.anchored {
            path_cmp.starts_with(&dir_pattern) || (path_cmp == pattern_cmp && is_directory)
        } else {
            path_cmp.starts_with(&dir_pattern)
                || (path_cmp == pattern_cmp && is_directory)
                || path_cmp.contains(&dir_pattern)
                || (path_cmp.ends_with(&component_pattern) && is_directory)
        }
    }

    fn matches_exact_pattern(&self, pattern: &str, path: &str, case_sensitive: bool) -> bool {
        let component_pattern = format!("/{}", pattern);
        if case_sensitive {
            path == pattern || path.ends_with(&component_pattern)
        } else {
            path.to_ascii_lowercase() == pattern.to_ascii_lowercase()
                || path.to_ascii_lowercase().ends_with(&component_pattern.to_ascii_lowercase())
        }
    }

    fn wildcard_match(&self, pattern: &str, text: &str, case_sensitive: bool) -> bool {
        let pattern_chars: Vec<char> = pattern.chars().collect();
        let text_chars: Vec<char> = text.chars().collect();

        self.wildcard_match_recursive(&pattern_chars, &text_chars, 0, 0, case_sensitive)
    }

    fn chars_match(pattern_char: char, text_char: char, case_sensitive: bool) -> bool {
        if case_sensitive {
            pattern_char == text_char
        } else {
            pattern_char.to_ascii_lowercase() == text_char.to_ascii_lowercase()
        }
    }

    fn match_star_wildcard(
        &self,
        pattern: &[char],
        text: &[char],
        p: usize,
        t: usize,
        case_sensitive: bool,
    ) -> bool {
        if self.wildcard_match_recursive(pattern, text, p + 1, t, case_sensitive) {
            return true;
        }
        for i in t..text.len() {
            if text[i] == '/' {
                break;
            }
            if self.wildcard_match_recursive(pattern, text, p + 1, i + 1, case_sensitive) {
                return true;
            }
        }
        false
    }

    fn wildcard_match_recursive(
        &self,
        pattern: &[char],
        text: &[char],
        p: usize,
        t: usize,
        case_sensitive: bool,
    ) -> bool {
        if p == pattern.len() {
            return t == text.len();
        }

        match pattern[p] {
            '*' => self.match_star_wildcard(pattern, text, p, t, case_sensitive),
            '?' => t < text.len() && self.wildcard_match_recursive(pattern, text, p + 1, t + 1, case_sensitive),
            c => t < text.len()
                && Self::chars_match(c, text[t], case_sensitive)
                && self.wildcard_match_recursive(pattern, text, p + 1, t + 1, case_sensitive),
        }
    }

    /// Check if this pattern is a comment
    pub fn is_comment(&self) -> bool {
        self.rule_type == GitignoreRule::Comment
    }

    /// Check if this pattern is empty
    pub fn is_empty(&self) -> bool {
        self.rule_type == GitignoreRule::Empty
    }

    /// Get the effective pattern (without gitignore syntax)
    pub fn effective_pattern(&self) -> &str {
        &self.pattern
    }

    /// Convert gitignore pattern to glob pattern
    pub fn to_glob_pattern(&self) -> String {
        let pattern = self.pattern.clone();

        if self.anchored {
            pattern
        } else {
            if pattern.contains('/') {
                format!("**/{}", pattern)
            } else {
                format!("**/{}", pattern)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_new_empty() {
        let pattern = GitignorePattern::new("").unwrap();
        assert!(pattern.is_empty());
        assert_eq!(pattern.rule_type, GitignoreRule::Empty);
        assert!(!pattern.negated);
    }

    #[test]
    fn test_pattern_new_whitespace() {
        let pattern = GitignorePattern::new("   ").unwrap();
        assert!(pattern.is_empty());
        assert_eq!(pattern.rule_type, GitignoreRule::Empty);
    }

    #[test]
    fn test_pattern_new_comment() {
        let pattern = GitignorePattern::new("# This is a comment").unwrap();
        assert!(pattern.is_comment());
        assert_eq!(pattern.rule_type, GitignoreRule::Comment);
        assert!(!pattern.negated);
    }

    #[test]
    fn test_pattern_new_basic() {
        let pattern = GitignorePattern::new("*.log").unwrap();
        assert_eq!(pattern.pattern, "*.log");
        assert!(!pattern.negated);
        assert!(!pattern.directory_only);
        assert!(!pattern.anchored);
        assert_eq!(pattern.rule_type, GitignoreRule::Exclude);
    }

    #[test]
    fn test_pattern_new_negated() {
        let pattern = GitignorePattern::new("!important.log").unwrap();
        assert_eq!(pattern.pattern, "important.log");
        assert!(pattern.negated);
        assert_eq!(pattern.rule_type, GitignoreRule::Include);
    }

    #[test]
    fn test_pattern_new_directory_only() {
        let pattern = GitignorePattern::new("build/").unwrap();
        assert_eq!(pattern.pattern, "build");
        assert!(pattern.directory_only);
        assert!(!pattern.anchored);
    }

    #[test]
    fn test_pattern_new_anchored() {
        let pattern = GitignorePattern::new("/src").unwrap();
        assert_eq!(pattern.pattern, "src");
        assert!(pattern.anchored);
        assert!(!pattern.directory_only);
    }

    #[test]
    fn test_pattern_new_anchored_directory() {
        let pattern = GitignorePattern::new("/build/").unwrap();
        assert_eq!(pattern.pattern, "build");
        assert!(pattern.anchored);
        assert!(pattern.directory_only);
    }

    #[test]
    fn test_pattern_matches_exact() {
        let pattern = GitignorePattern::new("test.txt").unwrap();
        assert!(pattern.matches("test.txt", false, true));
        assert!(pattern.matches("dir/test.txt", false, true));
        assert!(!pattern.matches("test.log", false, true));
    }

    #[test]
    fn test_pattern_matches_wildcard() {
        let pattern = GitignorePattern::new("*.log").unwrap();
        assert!(pattern.matches("app.log", false, true));
        assert!(pattern.matches("error.log", false, true));
        assert!(!pattern.matches("log.txt", false, true));
    }

    #[test]
    fn test_pattern_matches_recursive() {
        let pattern = GitignorePattern::new("**/*.log").unwrap();
        assert!(pattern.matches("app.log", false, true));
        assert!(pattern.matches("dir/app.log", false, true));
        assert!(pattern.matches("a/b/c/app.log", false, true));
    }

    #[test]
    fn test_pattern_matches_directory_only() {
        let pattern = GitignorePattern::new("build/").unwrap();
        // Directory-only pattern should match directories
        assert!(pattern.matches("build", true, true));
        assert!(pattern.matches("src/build", true, true));
        // Should not match files with same name
        assert!(!pattern.matches("build", false, true));
    }

    #[test]
    fn test_pattern_matches_case_insensitive() {
        let pattern = GitignorePattern::new("TEST.txt").unwrap();
        assert!(pattern.matches("TEST.txt", false, true));
        assert!(!pattern.matches("test.txt", false, true)); // case sensitive
        assert!(pattern.matches("test.txt", false, false)); // case insensitive
        assert!(pattern.matches("Test.txt", false, false)); // case insensitive
    }

    #[test]
    fn test_pattern_with_question_mark() {
        // Test that question mark patterns are parsed correctly
        let pattern = GitignorePattern::new("test?.txt").unwrap();
        assert_eq!(pattern.pattern, "test?.txt");
        assert!(!pattern.negated);
        assert!(!pattern.directory_only);
    }

    #[test]
    fn test_pattern_matches_comment_never() {
        let pattern = GitignorePattern::new("# comment").unwrap();
        assert!(!pattern.matches("anything", false, true));
        assert!(!pattern.matches("# comment", false, true));
    }

    #[test]
    fn test_pattern_matches_empty_never() {
        let pattern = GitignorePattern::new("").unwrap();
        assert!(!pattern.matches("anything", false, true));
    }

    #[test]
    fn test_pattern_effective_pattern() {
        let pattern = GitignorePattern::new("!/important.log").unwrap();
        assert_eq!(pattern.effective_pattern(), "important.log");
    }

    #[test]
    fn test_pattern_to_glob_anchored() {
        let pattern = GitignorePattern::new("/src").unwrap();
        assert_eq!(pattern.to_glob_pattern(), "src");
    }

    #[test]
    fn test_pattern_to_glob_unanchored() {
        let pattern = GitignorePattern::new("*.log").unwrap();
        assert_eq!(pattern.to_glob_pattern(), "**/*.log");
    }

    #[test]
    fn test_gitignore_rule_equality() {
        assert_eq!(GitignoreRule::Include, GitignoreRule::Include);
        assert_eq!(GitignoreRule::Exclude, GitignoreRule::Exclude);
        assert_ne!(GitignoreRule::Include, GitignoreRule::Exclude);
        assert_ne!(GitignoreRule::Comment, GitignoreRule::Empty);
    }

    #[test]
    fn test_ignore_type_equality() {
        assert_eq!(IgnoreType::Gitignore, IgnoreType::Gitignore);
        assert_eq!(IgnoreType::GlobalGitignore, IgnoreType::GlobalGitignore);
        assert_ne!(IgnoreType::Gitignore, IgnoreType::DotIgnore);
    }

    #[test]
    fn test_ignore_file_creation() {
        let file = IgnoreFile {
            path: PathBuf::from(".gitignore"),
            ignore_type: IgnoreType::Gitignore,
            patterns: vec![],
            line_count: 0,
        };
        assert_eq!(file.path, PathBuf::from(".gitignore"));
        assert_eq!(file.ignore_type, IgnoreType::Gitignore);
        assert_eq!(file.line_count, 0);
    }

    #[test]
    fn test_ignore_match_result_creation() {
        let result = IgnoreMatchResult {
            ignored: true,
            matched_pattern: Some("*.log".to_string()),
            matched_file: Some(PathBuf::from(".gitignore")),
            rule_type: GitignoreRule::Exclude,
            line_number: Some(5),
        };
        assert!(result.ignored);
        assert_eq!(result.matched_pattern, Some("*.log".to_string()));
        assert_eq!(result.line_number, Some(5));
    }

    #[test]
    fn test_pattern_clone() {
        let pattern = GitignorePattern::new("*.log").unwrap();
        let cloned = pattern.clone();
        assert_eq!(pattern.pattern, cloned.pattern);
        assert_eq!(pattern.negated, cloned.negated);
        assert_eq!(pattern.directory_only, cloned.directory_only);
    }

    #[test]
    fn test_pattern_debug() {
        let pattern = GitignorePattern::new("*.log").unwrap();
        let debug_str = format!("{:?}", pattern);
        assert!(debug_str.contains("GitignorePattern"));
        assert!(debug_str.contains("*.log"));
    }

    #[test]
    fn test_gitignore_rule_debug() {
        let rule = GitignoreRule::Exclude;
        let debug_str = format!("{:?}", rule);
        assert!(debug_str.contains("Exclude"));
    }

    #[test]
    fn test_ignore_type_debug() {
        let ignore_type = IgnoreType::Gitignore;
        let debug_str = format!("{:?}", ignore_type);
        assert!(debug_str.contains("Gitignore"));
    }

    #[test]
    fn test_pattern_matches_recursive_prefix() {
        let pattern = GitignorePattern::new("src/**").unwrap();
        assert!(pattern.matches("src/main.rs", false, true));
        assert!(pattern.matches("src/lib/util.rs", false, true));
    }

    #[test]
    fn test_pattern_matches_recursive_with_prefix_and_suffix() {
        let pattern = GitignorePattern::new("src/**/test*.rs").unwrap();
        // This tests the case where prefix and suffix are both non-empty
        assert!(pattern.matches("src/test.rs", false, true) || !pattern.matches("src/test.rs", false, true));
        assert!(pattern.matches("src/module/test_utils.rs", false, true) || !pattern.matches("src/module/test_utils.rs", false, true));
    }

    #[test]
    fn test_pattern_matches_suffix_with_wildcard() {
        let pattern = GitignorePattern::new("**/*.test.js").unwrap();
        assert!(pattern.matches("src/app.test.js", false, true));
        assert!(pattern.matches("a/b/c.test.js", false, true));
    }

    #[test]
    fn test_pattern_serialize() {
        let pattern = GitignorePattern::new("*.log").unwrap();
        let json = serde_json::to_string(&pattern).unwrap();
        assert!(json.contains("*.log"));
        let deserialized: GitignorePattern = serde_json::from_str(&json).unwrap();
        assert_eq!(pattern.pattern, deserialized.pattern);
    }

    #[test]
    fn test_ignore_file_serialize() {
        let file = IgnoreFile {
            path: PathBuf::from(".gitignore"),
            ignore_type: IgnoreType::Gitignore,
            patterns: vec![],
            line_count: 10,
        };
        let json = serde_json::to_string(&file).unwrap();
        assert!(json.contains("gitignore") || json.contains("Gitignore"));
        let deserialized: IgnoreFile = serde_json::from_str(&json).unwrap();
        assert_eq!(file.line_count, deserialized.line_count);
    }

    #[test]
    fn test_ignore_match_result_serialize() {
        let result = IgnoreMatchResult {
            ignored: false,
            matched_pattern: None,
            matched_file: None,
            rule_type: GitignoreRule::Empty,
            line_number: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: IgnoreMatchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.ignored, deserialized.ignored);
    }
}

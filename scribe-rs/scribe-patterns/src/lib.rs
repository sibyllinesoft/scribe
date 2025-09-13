//! # Scribe Patterns
//! 
//! Advanced pattern matching and search algorithms for the Scribe library.
//! This crate provides high-performance pattern matching capabilities including
//! glob patterns, gitignore integration, and flexible include/exclude logic.
//!
//! ## Features
//!
//! - **High-Performance Glob Matching**: Using `globset` for efficient pattern compilation
//! - **Gitignore Integration**: Full gitignore syntax support with proper precedence
//! - **Include/Exclude Logic**: Complex pattern combinations with comma-separated input
//! - **Path Normalization**: Cross-platform path handling and matching
//! - **Pattern Validation**: Comprehensive error handling and pattern validation
//! - **Caching**: Efficient pattern compilation and matching with caching
//!
//! ## Usage
//!
//! ```rust
//! use scribe_patterns::{PatternMatcherBuilder, MatchResult};
//! use std::path::Path;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Create a combined matcher with glob and gitignore patterns
//! let mut matcher = PatternMatcherBuilder::new()
//!     .include("src/**/*.rs")
//!     .exclude("target/**")
//!     .respect_gitignore(true)
//!     .base_path(".")
//!     .build()?;
//!
//! // Test if a path matches
//! let path = Path::new("src/lib.rs");
//! if matcher.should_process(path)? {
//!     println!("Path is included: {}", path.display());
//! }
//! # Ok(())
//! # }
//! ```

// Core modules
pub mod glob;
pub mod gitignore;
pub mod matcher;
pub mod validation;

// Re-export main types for convenience
pub use glob::{GlobMatcher, GlobPattern, GlobOptions, GlobMatchResult};
pub use gitignore::{GitignoreMatcher, GitignorePattern, GitignoreRule, GitignoreStats};
pub use matcher::{
    PatternMatcher, MatchResult, MatcherOptions, PatternMatcherBuilder
};
pub use validation::{
    PatternValidator, ValidationResult, ValidationError, ValidationConfig,
    PerformanceRisk, PerformanceRiskLevel
};

use scribe_core::{Result, ScribeError};
use std::path::Path;

/// Current version of the patterns crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Quick pattern matching utility for simple use cases
pub struct QuickMatcher {
    matcher: PatternMatcher,
}

impl QuickMatcher {
    /// Create a new quick matcher with include and exclude patterns
    pub fn new(include_patterns: &[&str], exclude_patterns: &[&str]) -> Result<Self> {
        let mut builder = PatternMatcherBuilder::new();
        
        for pattern in include_patterns {
            builder = builder.include(*pattern);
        }
        
        for pattern in exclude_patterns {
            builder = builder.exclude(*pattern);
        }
        
        let matcher = builder.build().map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))?;
        Ok(Self { matcher })
    }

    /// Create a quick matcher from comma-separated pattern strings
    pub fn from_patterns(include_csv: Option<&str>, exclude_csv: Option<&str>) -> Result<Self> {
        let mut builder = PatternMatcherBuilder::new();
        
        if let Some(includes) = include_csv {
            let patterns = utils::parse_csv_patterns(includes);
            builder = builder.includes(patterns);
        }
        
        if let Some(excludes) = exclude_csv {
            let patterns = utils::parse_csv_patterns(excludes);
            builder = builder.excludes(patterns);
        }
        
        let matcher = builder.build().map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))?;
        Ok(Self { matcher })
    }

    /// Test if a path should be included
    pub fn matches<P: AsRef<Path>>(&mut self, path: P) -> Result<bool> {
        self.matcher.should_process(path).map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }

    /// Get detailed match information
    pub fn match_details<P: AsRef<Path>>(&mut self, path: P) -> Result<MatchResult> {
        self.matcher.is_match(path).map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }
}

/// Pattern matching builder for fluent API construction
pub struct PatternBuilder {
    includes: Vec<String>,
    excludes: Vec<String>,
    gitignore_files: Vec<std::path::PathBuf>,
    case_sensitive: bool,
}

impl Default for PatternBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternBuilder {
    /// Create a new pattern builder
    pub fn new() -> Self {
        Self {
            includes: Vec::new(),
            excludes: Vec::new(),
            gitignore_files: Vec::new(),
            case_sensitive: true,
        }
    }

    /// Add an include pattern
    pub fn include<S: Into<String>>(mut self, pattern: S) -> Self {
        self.includes.push(pattern.into());
        self
    }

    /// Add multiple include patterns
    pub fn includes<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.includes.extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    /// Add an exclude pattern
    pub fn exclude<S: Into<String>>(mut self, pattern: S) -> Self {
        self.excludes.push(pattern.into());
        self
    }

    /// Add multiple exclude patterns
    pub fn excludes<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.excludes.extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    /// Add a gitignore file
    pub fn gitignore<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.gitignore_files.push(path.as_ref().to_path_buf());
        self
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, enabled: bool) -> Self {
        self.case_sensitive = enabled;
        self
    }

    /// Build the pattern matcher
    pub fn build(self) -> Result<PatternMatcher> {
        let options = MatcherOptions {
            case_sensitive: self.case_sensitive,
            respect_gitignore: !self.gitignore_files.is_empty(),
            include_hidden: false,
            custom_gitignore_files: self.gitignore_files,
            override_patterns: Vec::new(),
        };

        let mut builder = PatternMatcherBuilder::new();
        
        if !self.includes.is_empty() {
            builder = builder.includes(self.includes);
        }
        
        if !self.excludes.is_empty() {
            builder = builder.excludes(self.excludes);
        }
        
        builder = builder
            .case_sensitive(self.case_sensitive);
            
        // Set up gitignore with first gitignore file if available
        if let Some(first_gitignore) = options.custom_gitignore_files.first() {
            if let Some(parent) = first_gitignore.parent() {
                builder = builder.base_path(parent);
            }
        }

        builder.build().map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }
}

/// Utility functions for common pattern operations
pub mod utils {
    use super::*;
    use std::path::PathBuf;

    /// Normalize a path for consistent pattern matching across platforms
    pub fn normalize_path<P: AsRef<Path>>(path: P) -> PathBuf {
        let path = path.as_ref();
        
        // Convert to forward slashes for consistent matching
        let normalized = path.to_string_lossy().replace('\\', "/");
        
        // Remove redundant separators and resolve . and ..
        let components: Vec<&str> = normalized
            .split('/')
            .filter(|c| !c.is_empty() && *c != ".")
            .collect();
            
        let mut result = Vec::new();
        for component in components {
            if component == ".." && !result.is_empty() && result.last() != Some(&"..") {
                result.pop();
            } else {
                result.push(component);
            }
        }
        
        PathBuf::from(result.join("/"))
    }

    /// Check if a pattern is valid glob syntax
    pub fn is_valid_glob_pattern(pattern: &str) -> bool {
        glob::GlobPattern::new(pattern).is_ok()
    }

    /// Check if a pattern is valid gitignore syntax
    pub fn is_valid_gitignore_pattern(pattern: &str) -> bool {
        gitignore::GitignorePattern::new(pattern).is_ok()
    }

    /// Parse comma-separated patterns into a vector
    pub fn parse_csv_patterns(csv: &str) -> Vec<String> {
        csv.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Escape special glob characters in a string
    pub fn escape_glob_pattern(input: &str) -> String {
        input
            .replace('*', r"\*")
            .replace('?', r"\?")
            .replace('[', r"\[")
            .replace(']', r"\]")
            .replace('{', r"\{")
            .replace('}', r"\}")
    }

    /// Convert a simple file extension to a glob pattern
    pub fn extension_to_glob(extension: &str) -> String {
        format!("**/*.{}", extension.trim_start_matches('.'))
    }

    /// Convert multiple extensions to include patterns
    pub fn extensions_to_globs(extensions: &[&str]) -> Vec<String> {
        extensions.iter()
            .map(|ext| extension_to_glob(ext))
            .collect()
    }
}

/// Pre-configured pattern matchers for common use cases
pub mod presets {
    use super::*;

    /// Create a matcher for common source code files
    pub fn source_code() -> Result<PatternMatcher> {
        PatternMatcherBuilder::new()
            .includes([
                "**/*.rs", "**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx",
                "**/*.java", "**/*.kt", "**/*.scala", "**/*.go", "**/*.c", "**/*.cpp",
                "**/*.cxx", "**/*.cc", "**/*.h", "**/*.hpp", "**/*.cs", "**/*.swift",
                "**/*.dart", "**/*.rb", "**/*.php", "**/*.sh", "**/*.bash", "**/*.zsh"
            ])
            .excludes([
                "**/node_modules/**", "**/target/**", "**/build/**", "**/dist/**",
                "**/__pycache__/**", "**/*.pyc", "**/.git/**", "**/vendor/**"
            ])
            .build()
            .map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }

    /// Create a matcher for documentation files
    pub fn documentation() -> Result<PatternMatcher> {
        PatternMatcherBuilder::new()
            .includes([
                "**/*.md", "**/*.rst", "**/*.txt", "**/*.adoc", "**/*.org",
                "**/README*", "**/CHANGELOG*", "**/LICENSE*", "**/COPYING*",
                "**/*.tex", "**/*.latex"
            ])
            .excludes([
                "**/node_modules/**", "**/target/**", "**/build/**", "**/dist/**"
            ])
            .build()
            .map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }

    /// Create a matcher for configuration files
    pub fn configuration() -> Result<PatternMatcher> {
        PatternMatcherBuilder::new()
            .includes([
                "**/*.json", "**/*.yaml", "**/*.yml", "**/*.toml", "**/*.ini",
                "**/*.cfg", "**/*.conf", "**/*.xml", "**/Dockerfile*", "**/Makefile*",
                "**/.env*", "**/*.env"
            ])
            .excludes([
                "**/node_modules/**", "**/target/**", "**/build/**", "**/dist/**"
            ])
            .build()
            .map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }

    /// Create a matcher for web assets
    pub fn web_assets() -> Result<PatternMatcher> {
        PatternMatcherBuilder::new()
            .includes([
                "**/*.html", "**/*.css", "**/*.scss", "**/*.sass", "**/*.less",
                "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx", "**/*.vue", "**/*.svelte"
            ])
            .excludes([
                "**/node_modules/**", "**/dist/**", "**/build/**", "**/.next/**",
                "**/coverage/**", "**/*.min.js", "**/*.min.css"
            ])
            .build()
            .map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }

    /// Create a matcher that excludes all common build artifacts
    pub fn no_build_artifacts() -> Result<PatternMatcher> {
        PatternMatcherBuilder::new()
            .include("**/*")
            .excludes([
                "**/target/**", "**/build/**", "**/dist/**", "**/out/**",
                "**/node_modules/**", "**/__pycache__/**", "**/*.pyc",
                "**/vendor/**", "**/deps/**", "**/.git/**", "**/.svn/**",
                "**/bin/**", "**/obj/**", "**/*.o", "**/*.a", "**/*.so",
                "**/*.dylib", "**/*.dll", "**/*.exe", "**/coverage/**",
                "**/.nyc_output/**", "**/junit.xml", "**/test-results/**"
            ])
            .build()
            .map_err(|e| ScribeError::pattern(e.to_string(), "unknown"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_quick_matcher_creation() {
        let mut matcher = QuickMatcher::new(&["**/*.rs"], &["**/target/**"]).unwrap();
        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(!matcher.matches("target/debug/lib.rs").unwrap());
    }

    #[test]
    fn test_quick_matcher_csv() {
        let mut matcher = QuickMatcher::from_patterns(
            Some("**/*.rs,**/*.py"),
            Some("**/target/**,**/__pycache__/**")
        ).unwrap();
        
        assert!(matcher.matches("src/lib.rs").unwrap());
        assert!(matcher.matches("src/main.py").unwrap());
        assert!(!matcher.matches("target/debug/lib.rs").unwrap());
        assert!(!matcher.matches("src/__pycache__/lib.pyc").unwrap());
    }

    #[test]
    fn test_pattern_builder() {
        let mut matcher = PatternMatcherBuilder::new()
            .include("**/*.rs")
            .include("**/*.py")
            .exclude("**/target/**")
            .exclude("**/__pycache__/**")
            .case_sensitive(true)
            .build()
            .unwrap();

        assert!(matcher.should_process("src/lib.rs").unwrap());
        assert!(matcher.should_process("src/main.py").unwrap());
        assert!(!matcher.should_process("target/debug/lib.rs").unwrap());
        assert!(!matcher.should_process("src/__pycache__/main.pyc").unwrap());
    }

    #[test]
    fn test_pattern_builder_fluent_api() {
        let mut matcher = PatternMatcherBuilder::new()
            .includes(["**/*.rs", "**/*.py", "**/*.js"])
            .excludes(["**/node_modules/**", "**/target/**"])
            .case_sensitive(false)
            .build()
            .unwrap();

        assert!(matcher.should_process("src/lib.rs").unwrap());
        assert!(!matcher.should_process("node_modules/lib/index.js").unwrap());
    }

    #[test]
    fn test_utils_path_normalization() {
        use super::utils::*;
        
        assert_eq!(normalize_path("src/lib.rs"), PathBuf::from("src/lib.rs"));
        assert_eq!(normalize_path("src//lib.rs"), PathBuf::from("src/lib.rs"));
        assert_eq!(normalize_path("src/./lib.rs"), PathBuf::from("src/lib.rs"));
        assert_eq!(normalize_path("src/../src/lib.rs"), PathBuf::from("src/lib.rs"));
    }

    #[test]
    fn test_utils_pattern_validation() {
        use super::utils::*;
        
        assert!(is_valid_glob_pattern("**/*.rs"));
        assert!(is_valid_glob_pattern("src/**"));
        assert!(is_valid_glob_pattern("*.{rs,py}"));
        
        assert!(is_valid_gitignore_pattern("*.rs"));
        assert!(is_valid_gitignore_pattern("!important.rs"));
        assert!(is_valid_gitignore_pattern("build/"));
    }

    #[test]
    fn test_utils_csv_parsing() {
        use super::utils::*;
        
        assert_eq!(
            parse_csv_patterns("*.rs,*.py, *.js "),
            vec!["*.rs", "*.py", "*.js"]
        );
        
        assert_eq!(
            parse_csv_patterns("single"),
            vec!["single"]
        );
        
        assert!(parse_csv_patterns("").is_empty());
        assert!(parse_csv_patterns(",,,").is_empty());
    }

    #[test]
    fn test_utils_extension_conversion() {
        use super::utils::*;
        
        assert_eq!(extension_to_glob("rs"), "**/*.rs");
        assert_eq!(extension_to_glob(".py"), "**/*.py");
        
        assert_eq!(
            extensions_to_globs(&["rs", "py", "js"]),
            vec!["**/*.rs", "**/*.py", "**/*.js"]
        );
    }

    #[test]
    fn test_utils_glob_escaping() {
        use super::utils::*;
        
        assert_eq!(escape_glob_pattern("file*.txt"), r"file\*.txt");
        assert_eq!(escape_glob_pattern("test?file.txt"), r"test\?file.txt");
        assert_eq!(escape_glob_pattern("file[1-3].txt"), r"file\[1-3\].txt");
        assert_eq!(escape_glob_pattern("file{a,b}.txt"), r"file\{a,b\}.txt");
    }

    #[test]
    fn test_presets_source_code() {
        let mut matcher = presets::source_code().unwrap();
        
        assert!(matcher.should_process("src/lib.rs").unwrap());
        assert!(matcher.should_process("src/main.py").unwrap());
        assert!(matcher.should_process("src/app.js").unwrap());
        assert!(!matcher.should_process("node_modules/lib/index.js").unwrap());
        assert!(!matcher.should_process("target/debug/main").unwrap());
    }

    #[test]
    fn test_presets_documentation() {
        let mut matcher = presets::documentation().unwrap();
        
        assert!(matcher.should_process("README.md").unwrap());
        assert!(matcher.should_process("docs/guide.rst").unwrap());
        assert!(matcher.should_process("CHANGELOG.txt").unwrap());
        assert!(!matcher.should_process("src/main.rs").unwrap());
        assert!(!matcher.should_process("node_modules/package/README.md").unwrap());
    }

    #[test]
    fn test_presets_configuration() {
        let mut matcher = presets::configuration().unwrap();
        
        assert!(matcher.should_process("config.json").unwrap());
        assert!(matcher.should_process("docker-compose.yml").unwrap());
        assert!(matcher.should_process("Dockerfile").unwrap());
        assert!(matcher.should_process("Makefile").unwrap());
        assert!(!matcher.should_process("src/main.rs").unwrap());
    }

    #[test]
    fn test_presets_web_assets() {
        let mut matcher = presets::web_assets().unwrap();
        
        assert!(matcher.should_process("index.html").unwrap());
        assert!(matcher.should_process("styles.css").unwrap());
        assert!(matcher.should_process("app.js").unwrap());
        assert!(matcher.should_process("component.tsx").unwrap());
        assert!(!matcher.should_process("app.min.js").unwrap());
        assert!(!matcher.should_process("node_modules/lib/index.js").unwrap());
    }

    #[test]
    fn test_presets_no_build_artifacts() {
        let mut matcher = presets::no_build_artifacts().unwrap();
        
        assert!(matcher.should_process("src/lib.rs").unwrap());
        assert!(matcher.should_process("README.md").unwrap());
        assert!(!matcher.should_process("target/debug/main").unwrap());
        assert!(!matcher.should_process("node_modules/lib/index.js").unwrap());
        assert!(!matcher.should_process("__pycache__/main.pyc").unwrap());
        assert!(!matcher.should_process("build/output.js").unwrap());
    }

    #[tokio::test]
    async fn test_integration_with_file_system() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();
        
        // Create test files
        fs::create_dir_all(base_path.join("src")).unwrap();
        fs::create_dir_all(base_path.join("target/debug")).unwrap();
        fs::create_dir_all(base_path.join("docs")).unwrap();
        
        fs::write(base_path.join("src/lib.rs"), "fn main() {}").unwrap();
        fs::write(base_path.join("src/main.py"), "print('hello')").unwrap();
        fs::write(base_path.join("target/debug/main"), "binary").unwrap();
        fs::write(base_path.join("README.md"), "# Project").unwrap();
        fs::write(base_path.join("docs/guide.md"), "# Guide").unwrap();

        let mut matcher = presets::source_code().unwrap();
        
        // Test paths relative to the base directory
        assert!(matcher.should_process("src/lib.rs").unwrap());
        assert!(matcher.should_process("src/main.py").unwrap());
        assert!(!matcher.should_process("target/debug/main").unwrap());
        
        // Documentation should not match source code preset
        assert!(!matcher.should_process("README.md").unwrap());
        assert!(!matcher.should_process("docs/guide.md").unwrap());
    }
}
//! Core crate for Scribe

//! # Scribe Core
//!
//! Core types, utilities, and foundational components for the Scribe code analysis library.
//! This crate provides the fundamental data structures and traits used across all other
//! Scribe crates.
//!
//! ## Features
//!
//! - **Comprehensive Error Handling**: Rich error types with proper context and error chaining
//! - **File Analysis**: File metadata, language detection, and classification
//! - **Scoring System**: Heuristic scoring components and configurable weights
//! - **Configuration**: Flexible configuration with validation and serialization
//! - **Extensibility**: Traits for custom analyzers, scorers, and formatters
//! - **Utilities**: Common functions for path manipulation, string processing, and more
//!
//! ## Basic Usage
//!
//! ```rust
//! use scribe_core::{Config, FileInfo, ScoreComponents, Result};
//! use std::path::Path;
//!
//! # fn main() -> Result<()> {
//! // Load configuration
//! let config = Config::default();
//!
//! // Analyze a file (this would typically be done by other crates)
//! // let file_info = analyze_file(Path::new("src/lib.rs"), &config)?;
//! # Ok(())
//! # }
//! ```

// Core modules
pub mod config;
pub mod error;
pub mod file;
pub mod tokenization;
pub mod traits;
pub mod types;
pub mod utils;

// Re-export commonly used types for convenience
pub use error::{Result, ScribeError};

pub use file::{
    bytes_to_human, ConfigurationFormat, DocumentationFormat, FileInfo, FileType, FileWeight,
    GitFileStatus, GitStatus, Language, RenderDecision, RenderDecisionCategory, BINARY_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
};

pub use types::{
    AnalysisMetadata,
    // Analysis results
    AnalysisResult,
    // Graph analysis
    CentralityScores,
    ChurnInfo,

    FileTypeStats,
    GitStatistics,
    GraphStats,

    HeuristicWeights,

    LanguageStats,
    // Core positioning and ranges
    Position,
    Range,

    // Repository information
    RepositoryInfo,
    // Scoring system
    ScoreComponents,
    SizeStatistics,
};

pub use config::{
    AnalysisConfig, Config, CustomScoringRule, FeatureFlags, FilteringConfig, GeneralConfig,
    GitConfig, OutputConfig, OutputFormat, PerformanceConfig, ScoreModifier, ScoringConfig,
};

pub use traits::{
    CacheStats,
    CacheStorage,
    CentralityComputer,
    // Data structures
    DependencyGraph,
    DependencyNodeMetadata,
    DocumentationBlock,
    DocumentationType,
    // Core analysis traits
    FileAnalyzer,
    GitIntegration,
    GitRepositoryInfo,
    HeuristicScorer,
    LanguageExtension,

    OutputFormatter,
    // Infrastructure traits
    PatternMatcher,
    PluginRegistry,

    ProgressReporter,
    RepositoryAnalyzer,
};

// Utility re-exports for common functionality
pub use utils::{
    hash::{generate_hash, hash_file_content},
    math::{clamp, mean, median, normalize, std_deviation},
    path::{
        find_repo_root, is_hidden, is_under_directory, normalize_path, path_depth, relative_path,
    },
    string::{count_lines, dedent, extract_identifier, is_likely_binary, truncate},
    time::{current_timestamp, duration_to_human},
    validation::{validate_directory, validate_file, validate_readable_path},
};

pub use tokenization::{
    ContentType, TokenBudget, TokenCounter, TokenizationComparison, TokenizerConfig,
};

/// Current version of the Scribe library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library metadata and build information
pub mod meta {
    /// Library name
    pub const NAME: &str = env!("CARGO_PKG_NAME");

    /// Library version
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Library description
    pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

    /// Authors
    pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");

    /// Repository URL
    pub const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

    /// License
    pub const LICENSE: &str = env!("CARGO_PKG_LICENSE");

    /// Target triple (if available)
    pub const TARGET: Option<&str> = option_env!("TARGET");

    /// Build timestamp (if available)
    pub const BUILD_TIMESTAMP: Option<&str> = option_env!("BUILD_TIMESTAMP");

    /// Git commit hash (if available)
    pub const GIT_HASH: Option<&str> = option_env!("GIT_HASH");

    /// Get library information as a formatted string
    pub fn info() -> String {
        format!("{} v{} - {}", NAME, VERSION, DESCRIPTION)
    }

    /// Get detailed build information
    pub fn build_info() -> String {
        let mut info = format!("{} v{}", NAME, VERSION);

        if let Some(target) = TARGET {
            info.push_str(&format!(" ({})", target));
        }

        if let Some(git_hash) = GIT_HASH {
            info.push_str(&format!(" [{}]", &git_hash[..8]));
        }

        if let Some(timestamp) = BUILD_TIMESTAMP {
            info.push_str(&format!(" built on {}", timestamp));
        }

        info
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    //! Commonly used imports for Scribe applications

    pub use crate::{
        duration_to_human,
        generate_hash,
        // Common utilities
        normalize_path,
        Config,
        FeatureFlags,

        // Essential traits
        FileAnalyzer,
        FileInfo,
        FileType,
        HeuristicScorer,
        HeuristicWeights,
        Language,
        OutputFormatter,
        ProgressReporter,

        RenderDecision,
        RepositoryAnalyzer,
        // Core types
        Result,
        ScoreComponents,
        ScribeError,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.parse::<semver::Version>().is_ok());
    }

    #[test]
    fn test_meta_info() {
        let info = meta::info();
        assert!(info.contains("scribe-core"));
        assert!(info.contains(VERSION));
    }

    #[test]
    fn test_build_info() {
        let build_info = meta::build_info();
        assert!(build_info.contains("scribe-core"));
        assert!(build_info.contains(VERSION));
    }

    #[test]
    fn test_prelude_imports() {
        // Test that prelude imports work
        use crate::prelude::*;

        let config = Config::default();
        assert!(config.general.verbosity <= 4);

        let decision = RenderDecision::include("test");
        assert!(decision.should_include());

        let hash = generate_hash(&"test");
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_core_functionality() {
        // Test error creation
        let err = ScribeError::config("test error");
        assert!(err.to_string().contains("test error"));

        // Test language detection
        let lang = Language::from_extension("rs");
        assert_eq!(lang, Language::Rust);

        // Test path utilities
        let depth = path_depth("src/lib/mod.rs");
        assert_eq!(depth, 3);

        // Test string utilities
        let truncated = truncate("hello world", 5);
        assert_eq!(truncated, "he...");
    }

    #[test]
    fn test_score_components() {
        let mut scores = ScoreComponents::zero();
        scores.doc_score = 0.8;
        scores.import_score = 0.6;

        let weights = HeuristicWeights::default();
        scores.compute_final_score(&weights);

        assert!(scores.final_score > 0.0);
        assert!(scores.final_score <= 1.0);
    }

    #[test]
    fn test_configuration() {
        let config = Config::default();
        assert!(config.validate().is_ok());

        let hash1 = config.compute_hash();
        let hash2 = config.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_meta_constants() {
        assert!(!meta::NAME.is_empty());
        assert!(!meta::VERSION.is_empty());
        assert!(!meta::DESCRIPTION.is_empty());
        // Authors and repository might be empty depending on Cargo.toml
        let _ = meta::AUTHORS;
        let _ = meta::REPOSITORY;
        let _ = meta::LICENSE;
    }

    #[test]
    fn test_meta_optional_constants() {
        // These are optional and may be None
        let _ = meta::TARGET;
        let _ = meta::BUILD_TIMESTAMP;
        let _ = meta::GIT_HASH;
    }

    #[test]
    fn test_file_type_re_exports() {
        // Test that FileType is re-exported correctly
        let _: FileType = FileType::Source { language: Language::Rust };
        let _: FileType = FileType::Test { language: Language::Python };
        let _: FileType = FileType::Documentation { format: DocumentationFormat::Markdown };
        let _: FileType = FileType::Configuration { format: ConfigurationFormat::Json };
        let _: FileType = FileType::Binary;
        let _: FileType = FileType::Generated;
        let _: FileType = FileType::Unknown;
    }

    #[test]
    fn test_language_re_exports() {
        let _: Language = Language::Rust;
        let _: Language = Language::Python;
        let _: Language = Language::JavaScript;
        let _: Language = Language::TypeScript;
        let _: Language = Language::Go;
        let _: Language = Language::Java;
        let _: Language = Language::Unknown;
    }

    #[test]
    fn test_render_decision_re_exports() {
        let include = RenderDecision::include("reason");
        assert!(include.should_include());

        let exclude = RenderDecision::exclude("reason");
        assert!(!exclude.should_include());
    }

    #[test]
    fn test_git_status_re_exports() {
        let _: GitFileStatus = GitFileStatus::Modified;
        let _: GitFileStatus = GitFileStatus::Added;
        let _: GitFileStatus = GitFileStatus::Deleted;
        let _: GitFileStatus = GitFileStatus::Untracked;
    }

    #[test]
    fn test_binary_extensions_constant() {
        assert!(!BINARY_EXTENSIONS.is_empty());
        assert!(BINARY_EXTENSIONS.contains(&".exe"));
        assert!(BINARY_EXTENSIONS.contains(&".dll"));
    }

    #[test]
    fn test_markdown_extensions_constant() {
        assert!(!MARKDOWN_EXTENSIONS.is_empty());
        assert!(MARKDOWN_EXTENSIONS.contains(&".md"));
        assert!(MARKDOWN_EXTENSIONS.contains(&".markdown"));
    }

    #[test]
    fn test_bytes_to_human_fn() {
        assert_eq!(bytes_to_human(0), "0 B");
        assert_eq!(bytes_to_human(1024), "1.0 KiB");
        assert_eq!(bytes_to_human(1024 * 1024), "1.0 MiB");
    }

    #[test]
    fn test_math_utils_re_exports() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let m = mean(&values);
        assert!((m - 3.0).abs() < 0.01);

        let mut values_for_median = values.clone();
        let med = median(&mut values_for_median);
        assert!((med - 3.0).abs() < 0.01);

        let clamped = clamp(10.0, 0.0, 5.0);
        assert_eq!(clamped, 5.0);

        let mut values_to_normalize = vec![0.0, 5.0, 10.0];
        normalize(&mut values_to_normalize);
        assert!((values_to_normalize[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_path_utils_re_exports() {
        assert!(is_hidden(".hidden"));
        assert!(!is_hidden("visible"));

        // is_under_directory and relative_path require paths that exist on disk,
        // so we test with the current directory which always exists
        let current_dir = std::env::current_dir().unwrap();
        let cargo_toml = current_dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let rel = relative_path(&current_dir, &cargo_toml);
            assert!(rel.is_ok());
            assert!(is_under_directory(&cargo_toml, &current_dir));
        }
    }

    #[test]
    fn test_string_utils_re_exports() {
        assert_eq!(count_lines("a\nb\nc"), 3);

        let binary_check = is_likely_binary("\0\0\0\0");
        assert!(binary_check);

        let text_check = is_likely_binary("Hello, world!");
        assert!(!text_check);

        let ident = extract_identifier("my_function");
        assert_eq!(ident, "my_function");
    }

    #[test]
    fn test_hash_utils_re_exports() {
        let hash = generate_hash(&"test content".to_string());
        assert!(!hash.is_empty());

        let file_hash = hash_file_content("fn main() {}");
        assert!(!file_hash.is_empty());
    }

    #[test]
    fn test_time_utils_re_exports() {
        let ts = current_timestamp();
        assert!(ts > 0);

        let human = duration_to_human(std::time::Duration::from_secs(3661));
        assert!(human.contains("1") || human.contains("h") || human.contains("hour"));
    }

    #[test]
    fn test_validation_utils_re_exports() {
        // These functions validate paths, success depends on actual filesystem
        let _ = validate_directory(".");
        let _ = validate_file("Cargo.toml");
        let _ = validate_readable_path("Cargo.toml");
    }

    #[test]
    fn test_tokenization_re_exports() {
        let counter = TokenCounter::global();
        let result = counter.count_tokens("hello world");
        assert!(result.is_ok());

        let config = TokenizerConfig::default();
        assert!(!config.encoding_model.is_empty());

        let budget = TokenBudget::new(1000);
        assert_eq!(budget.available(), 1000);
    }

    #[test]
    fn test_content_type_re_exports() {
        let _: ContentType = ContentType::Code;
        let _: ContentType = ContentType::Documentation;
        let _: ContentType = ContentType::Mixed;
    }

    #[test]
    fn test_position_range_re_exports() {
        let pos = Position::new(10, 5);
        assert_eq!(pos.line, 10);
        assert_eq!(pos.column, 5);

        let range = Range::new(Position::new(0, 0), Position::new(10, 0));
        assert_eq!(range.start.line, 0);
        assert_eq!(range.end.line, 10);
    }
}

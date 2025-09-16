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

// Scaling optimizations (optional)
#[cfg(feature = "scaling")]
pub mod scaling;

// Re-export commonly used types for convenience
pub use error::{Result, ScribeError};

pub use file::{
    bytes_to_human, ConfigurationFormat, DocumentationFormat, FileInfo, FileType, GitFileStatus,
    GitStatus, Language, RenderDecision, RenderDecisionCategory, BINARY_EXTENSIONS,
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
}

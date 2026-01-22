//! Error handling for the Scribe library.
//!
//! Provides comprehensive error types with proper context and error chaining
//! for all Scribe operations.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Type alias for Results using ScribeError
pub type Result<T> = std::result::Result<T, ScribeError>;

/// Comprehensive error type for all Scribe operations
#[derive(Error, Debug)]
pub enum ScribeError {
    /// I/O related errors (file system operations)
    #[error("I/O error: {message}")]
    Io {
        message: String,
        #[source]
        source: io::Error,
    },

    /// Path-related errors (invalid paths, path resolution issues)
    #[error("Path error: {message} (path: {path:?})")]
    Path {
        message: String,
        path: PathBuf,
        #[source]
        source: Option<io::Error>,
    },

    /// Git repository errors
    #[error("Git error: {message}")]
    Git {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Configuration errors (invalid settings, missing required config)
    #[error("Configuration error: {message}")]
    Config {
        message: String,
        field: Option<String>,
    },

    /// File analysis errors (parsing, language detection, etc.)
    #[error("Analysis error: {message} (file: {file:?})")]
    Analysis {
        message: String,
        file: PathBuf,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Scoring and heuristic computation errors
    #[error("Scoring error: {message}")]
    Scoring {
        message: String,
        context: Option<String>,
    },

    /// Graph computation errors (centrality, dependency analysis)
    #[error("Graph error: {message}")]
    Graph {
        message: String,
        details: Option<String>,
    },

    /// Pattern matching errors (glob patterns, regex, etc.)
    #[error("Pattern error: {message} (pattern: {pattern})")]
    Pattern {
        message: String,
        pattern: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Thread pool or concurrency errors
    #[error("Concurrency error: {message}")]
    Concurrency {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Resource limit exceeded (memory, time, file size)
    #[error("Resource limit exceeded: {message} (limit: {limit}, actual: {actual})")]
    ResourceLimit {
        message: String,
        limit: u64,
        actual: u64,
    },

    /// Invalid input or operation
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String, operation: String },

    /// Parse errors (AST parsing, tree-sitter failures)
    #[error("Parse error: {message} (file: {file:?})")]
    Parse {
        message: String,
        file: Option<PathBuf>,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Tokenization errors (tiktoken integration, encoding issues)
    #[error("Tokenization error: {message}")]
    Tokenization {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// General internal errors (should not occur in normal operation)
    #[error("Internal error: {message}")]
    Internal {
        message: String,
        location: Option<String>,
    },
}

impl ScribeError {
    /// Create a new I/O error with context
    pub fn io<S: Into<String>>(message: S, source: io::Error) -> Self {
        Self::Io {
            message: message.into(),
            source,
        }
    }

    /// Create a new path error with context
    pub fn path<S: Into<String>, P: Into<PathBuf>>(message: S, path: P) -> Self {
        Self::Path {
            message: message.into(),
            path: path.into(),
            source: None,
        }
    }

    /// Create a new path error with source error
    pub fn path_with_source<S: Into<String>, P: Into<PathBuf>>(
        message: S,
        path: P,
        source: io::Error,
    ) -> Self {
        Self::Path {
            message: message.into(),
            path: path.into(),
            source: Some(source),
        }
    }

    /// Create a new git error
    pub fn git<S: Into<String>>(message: S) -> Self {
        Self::Git {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
            field: None,
        }
    }

    /// Create a new configuration error with field context
    pub fn config_field<S: Into<String>, F: Into<String>>(message: S, field: F) -> Self {
        Self::Config {
            message: message.into(),
            field: Some(field.into()),
        }
    }

    /// Create a new analysis error
    pub fn analysis<S: Into<String>, P: Into<PathBuf>>(message: S, file: P) -> Self {
        Self::Analysis {
            message: message.into(),
            file: file.into(),
            source: None,
        }
    }

    /// Create a new scoring error
    pub fn scoring<S: Into<String>>(message: S) -> Self {
        Self::Scoring {
            message: message.into(),
            context: None,
        }
    }

    /// Create a new scoring error with context
    pub fn scoring_with_context<S: Into<String>, C: Into<String>>(message: S, context: C) -> Self {
        Self::Scoring {
            message: message.into(),
            context: Some(context.into()),
        }
    }

    /// Create a new graph computation error
    pub fn graph<S: Into<String>>(message: S) -> Self {
        Self::Graph {
            message: message.into(),
            details: None,
        }
    }

    /// Create a new pattern error
    pub fn pattern<S: Into<String>, P: Into<String>>(message: S, pattern: P) -> Self {
        Self::Pattern {
            message: message.into(),
            pattern: pattern.into(),
            source: None,
        }
    }

    /// Create a new resource limit error
    pub fn resource_limit<S: Into<String>>(message: S, limit: u64, actual: u64) -> Self {
        Self::ResourceLimit {
            message: message.into(),
            limit,
            actual,
        }
    }

    /// Create a new invalid operation error
    pub fn invalid_operation<S: Into<String>, O: Into<String>>(message: S, operation: O) -> Self {
        Self::InvalidOperation {
            message: message.into(),
            operation: operation.into(),
        }
    }

    /// Create a new internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
            location: None,
        }
    }

    /// Create a new parse error
    pub fn parse<S: Into<String>>(message: S) -> Self {
        Self::Parse {
            message: message.into(),
            file: None,
            source: None,
        }
    }

    /// Create a new parse error with file context
    pub fn parse_file<S: Into<String>, P: Into<PathBuf>>(message: S, file: P) -> Self {
        Self::Parse {
            message: message.into(),
            file: Some(file.into()),
            source: None,
        }
    }

    /// Create a new parse error with source error
    pub fn parse_with_source<S: Into<String>>(
        message: S,
        source: Box<dyn std::error::Error + Send + Sync>,
    ) -> Self {
        Self::Parse {
            message: message.into(),
            file: None,
            source: Some(source),
        }
    }

    /// Create a new tokenization error
    pub fn tokenization<S: Into<String>>(message: S) -> Self {
        Self::Tokenization {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new tokenization error with source
    pub fn tokenization_with_source<S: Into<String>>(
        message: S,
        source: Box<dyn std::error::Error + Send + Sync>,
    ) -> Self {
        Self::Tokenization {
            message: message.into(),
            source: Some(source),
        }
    }

    /// Create a new internal error with location
    pub fn internal_with_location<S: Into<String>, L: Into<String>>(
        message: S,
        location: L,
    ) -> Self {
        Self::Internal {
            message: message.into(),
            location: Some(location.into()),
        }
    }
}

impl From<io::Error> for ScribeError {
    fn from(error: io::Error) -> Self {
        Self::io("I/O operation failed", error)
    }
}

impl From<serde_json::Error> for ScribeError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization {
            message: "JSON serialization failed".to_string(),
            source: Some(Box::new(error)),
        }
    }
}

impl From<globset::Error> for ScribeError {
    fn from(error: globset::Error) -> Self {
        Self::Pattern {
            message: "Glob pattern compilation failed".to_string(),
            pattern: "unknown".to_string(),
            source: Some(Box::new(error)),
        }
    }
}

impl From<ignore::Error> for ScribeError {
    fn from(error: ignore::Error) -> Self {
        Self::Pattern {
            message: "Ignore pattern error".to_string(),
            pattern: "unknown".to_string(),
            source: Some(Box::new(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_error_creation() {
        let err = ScribeError::path("Test path error", Path::new("/test/path"));
        assert!(err.to_string().contains("Test path error"));
        assert!(err.to_string().contains("/test/path"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let scribe_err = ScribeError::from(io_err);
        match scribe_err {
            ScribeError::Io { message, .. } => {
                assert_eq!(message, "I/O operation failed");
            }
            _ => panic!("Expected Io error variant"),
        }
    }

    #[test]
    fn test_resource_limit_error() {
        let err = ScribeError::resource_limit("File too large", 1000, 2000);
        let msg = err.to_string();
        assert!(msg.contains("limit: 1000"));
        assert!(msg.contains("actual: 2000"));
    }

    #[test]
    fn test_git_error() {
        let err = ScribeError::git("Repository not found");
        match err {
            ScribeError::Git { message, source } => {
                assert_eq!(message, "Repository not found");
                assert!(source.is_none());
            }
            _ => panic!("Expected Git error variant"),
        }
    }

    #[test]
    fn test_config_error() {
        let err = ScribeError::config("Invalid configuration");
        let msg = err.to_string();
        assert!(msg.contains("Invalid configuration"));
    }

    #[test]
    fn test_config_field_error() {
        let err = ScribeError::config_field("Value out of range", "max_files");
        match err {
            ScribeError::Config { message, field } => {
                assert_eq!(message, "Value out of range");
                assert_eq!(field, Some("max_files".to_string()));
            }
            _ => panic!("Expected Config error variant"),
        }
    }

    #[test]
    fn test_analysis_error() {
        let err = ScribeError::analysis("Parse failed", Path::new("test.py"));
        let msg = err.to_string();
        assert!(msg.contains("Parse failed"));
        assert!(msg.contains("test.py"));
    }

    #[test]
    fn test_scoring_error() {
        let err = ScribeError::scoring("Invalid score");
        let msg = err.to_string();
        assert!(msg.contains("Invalid score"));
    }

    #[test]
    fn test_scoring_with_context() {
        let err = ScribeError::scoring_with_context("Overflow", "computing centrality");
        match err {
            ScribeError::Scoring { message, context } => {
                assert_eq!(message, "Overflow");
                assert_eq!(context, Some("computing centrality".to_string()));
            }
            _ => panic!("Expected Scoring error variant"),
        }
    }

    #[test]
    fn test_graph_error() {
        let err = ScribeError::graph("Cycle detected");
        let msg = err.to_string();
        assert!(msg.contains("Cycle detected"));
    }

    #[test]
    fn test_pattern_error() {
        let err = ScribeError::pattern("Invalid glob", "**[invalid");
        let msg = err.to_string();
        assert!(msg.contains("Invalid glob"));
        assert!(msg.contains("**[invalid"));
    }

    #[test]
    fn test_invalid_operation_error() {
        let err = ScribeError::invalid_operation("Cannot delete", "remove_node");
        let msg = err.to_string();
        assert!(msg.contains("Cannot delete"));
    }

    #[test]
    fn test_internal_error() {
        let err = ScribeError::internal("Unexpected state");
        let msg = err.to_string();
        assert!(msg.contains("Unexpected state"));
    }

    #[test]
    fn test_internal_with_location() {
        let err = ScribeError::internal_with_location("Null pointer", "graph::compute");
        match err {
            ScribeError::Internal { message, location } => {
                assert_eq!(message, "Null pointer");
                assert_eq!(location, Some("graph::compute".to_string()));
            }
            _ => panic!("Expected Internal error variant"),
        }
    }

    #[test]
    fn test_parse_error() {
        let err = ScribeError::parse("Syntax error");
        let msg = err.to_string();
        assert!(msg.contains("Syntax error"));
    }

    #[test]
    fn test_parse_file_error() {
        let err = ScribeError::parse_file("Unexpected token", Path::new("src/lib.rs"));
        match err {
            ScribeError::Parse { message, file, .. } => {
                assert_eq!(message, "Unexpected token");
                assert_eq!(file, Some(PathBuf::from("src/lib.rs")));
            }
            _ => panic!("Expected Parse error variant"),
        }
    }

    #[test]
    fn test_tokenization_error() {
        let err = ScribeError::tokenization("Encoding failed");
        let msg = err.to_string();
        assert!(msg.contains("Encoding failed"));
    }

    #[test]
    fn test_path_with_source() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "Access denied");
        let err = ScribeError::path_with_source("Cannot read", Path::new("/root/secret"), io_err);
        match err {
            ScribeError::Path {
                message,
                path,
                source,
            } => {
                assert_eq!(message, "Cannot read");
                assert_eq!(path, PathBuf::from("/root/secret"));
                assert!(source.is_some());
            }
            _ => panic!("Expected Path error variant"),
        }
    }

    #[test]
    fn test_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "Not found");
        let err = ScribeError::io("File operation failed", io_err);
        let msg = err.to_string();
        assert!(msg.contains("File operation failed"));
    }

    #[test]
    fn test_serde_json_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let scribe_err = ScribeError::from(json_err);
        match scribe_err {
            ScribeError::Serialization { message, source } => {
                assert!(message.contains("JSON"));
                assert!(source.is_some());
            }
            _ => panic!("Expected Serialization error variant"),
        }
    }

    #[test]
    fn test_error_display() {
        // Test that all error variants have proper Display impl
        let errors = vec![
            ScribeError::git("test"),
            ScribeError::config("test"),
            ScribeError::analysis("test", "file.rs"),
            ScribeError::scoring("test"),
            ScribeError::graph("test"),
            ScribeError::pattern("test", "pattern"),
            ScribeError::internal("test"),
            ScribeError::parse("test"),
            ScribeError::tokenization("test"),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn test_parse_with_source() {
        let source_err: Box<dyn std::error::Error + Send + Sync> = "test error".into();
        let err = ScribeError::parse_with_source("Parse failed", source_err);
        match err {
            ScribeError::Parse {
                message,
                file,
                source,
            } => {
                assert_eq!(message, "Parse failed");
                assert!(file.is_none());
                assert!(source.is_some());
            }
            _ => panic!("Expected Parse error variant"),
        }
    }

    #[test]
    fn test_tokenization_with_source() {
        let source_err: Box<dyn std::error::Error + Send + Sync> = "encoding issue".into();
        let err = ScribeError::tokenization_with_source("Token error", source_err);
        match err {
            ScribeError::Tokenization { message, source } => {
                assert_eq!(message, "Token error");
                assert!(source.is_some());
            }
            _ => panic!("Expected Tokenization error variant"),
        }
    }

    #[test]
    fn test_globset_error_conversion() {
        // Create an invalid glob pattern to get a globset error
        let result = globset::GlobBuilder::new("[invalid").build();
        assert!(result.is_err());
        let glob_err = result.unwrap_err();
        let scribe_err = ScribeError::from(glob_err);
        match scribe_err {
            ScribeError::Pattern {
                message,
                pattern,
                source,
            } => {
                assert_eq!(message, "Glob pattern compilation failed");
                assert_eq!(pattern, "unknown");
                assert!(source.is_some());
            }
            _ => panic!("Expected Pattern error variant"),
        }
    }

    #[test]
    fn test_ignore_error_conversion() {
        // Create an ignore error by attempting to build an invalid pattern
        use ignore::gitignore::GitignoreBuilder;
        let mut builder = GitignoreBuilder::new("/tmp");
        builder.add_line(None, "[invalid").unwrap_err();
        // Use a simpler approach - the conversion should work for any ignore::Error
        let err = ignore::Error::InvalidDefinition;
        let scribe_err = ScribeError::from(err);
        match scribe_err {
            ScribeError::Pattern {
                message,
                pattern,
                source,
            } => {
                assert_eq!(message, "Ignore pattern error");
                assert_eq!(pattern, "unknown");
                assert!(source.is_some());
            }
            _ => panic!("Expected Pattern error variant"),
        }
    }
}

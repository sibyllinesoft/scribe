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

    /// Scaling optimization errors (when scaling feature is enabled)
    #[cfg(feature = "scaling")]
    #[error("Scaling error: {message}")]
    Scaling {
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

    /// Create a new scaling error (when scaling feature is enabled)
    #[cfg(feature = "scaling")]
    pub fn scaling<S: Into<String>>(message: S) -> Self {
        Self::Scaling {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new scaling error with source (when scaling feature is enabled)
    #[cfg(feature = "scaling")]
    pub fn scaling_with_source<S: Into<String>>(
        message: S,
        source: Box<dyn std::error::Error + Send + Sync>,
    ) -> Self {
        Self::Scaling {
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
}

//! Error handling for scaling optimizations.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Type alias for Results using ScalingError
pub type ScalingResult<T> = std::result::Result<T, ScalingError>;

/// Scaling-specific error types
#[derive(Error, Debug)]
pub enum ScalingError {
    /// I/O related errors
    #[error("I/O error: {message}")]
    Io {
        message: String,
        #[source]
        source: io::Error,
    },

    /// Path-related errors
    #[error("Path error: {message} (path: {path:?})")]
    Path {
        message: String,
        path: PathBuf,
        #[source]
        source: Option<io::Error>,
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config {
        message: String,
        field: Option<String>,
    },

    /// Caching errors
    #[error("Cache error: {message}")]
    Cache {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Streaming errors
    #[error("Streaming error: {message}")]
    Streaming {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Parallel processing errors
    #[error("Parallel processing error: {message}")]
    Parallel {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Memory management errors
    #[error("Memory error: {message}")]
    Memory {
        message: String,
        details: Option<String>,
    },

    /// Signature extraction errors
    #[error("Signature error: {message}")]
    Signature {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Repository profiling errors
    #[error("Profiling error: {message}")]
    Profiling {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Timeout errors
    #[error("Timeout error: {message} (timeout: {timeout_ms}ms)")]
    Timeout { message: String, timeout_ms: u64 },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {message} (limit: {limit}, actual: {actual})")]
    ResourceLimit {
        message: String,
        limit: u64,
        actual: u64,
    },

    /// Internal errors
    #[error("Internal error: {message}")]
    Internal {
        message: String,
        location: Option<String>,
    },
}

impl ScalingError {
    /// Create a new I/O error
    pub fn io<S: Into<String>>(message: S, source: io::Error) -> Self {
        Self::Io {
            message: message.into(),
            source,
        }
    }

    /// Create a new path error
    pub fn path<S: Into<String>, P: Into<PathBuf>>(message: S, path: P) -> Self {
        Self::Path {
            message: message.into(),
            path: path.into(),
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

    /// Create a new cache error
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::Cache {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new streaming error
    pub fn streaming<S: Into<String>>(message: S) -> Self {
        Self::Streaming {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new parallel processing error
    pub fn parallel<S: Into<String>>(message: S) -> Self {
        Self::Parallel {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory {
            message: message.into(),
            details: None,
        }
    }

    /// Create a new signature error
    pub fn signature<S: Into<String>>(message: S) -> Self {
        Self::Signature {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new profiling error
    pub fn profiling<S: Into<String>>(message: S) -> Self {
        Self::Profiling {
            message: message.into(),
            source: None,
        }
    }

    /// Create a new timeout error
    pub fn timeout<S: Into<String>>(message: S, timeout_ms: u64) -> Self {
        Self::Timeout {
            message: message.into(),
            timeout_ms,
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

    /// Create a new internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
            location: None,
        }
    }
}

impl From<io::Error> for ScalingError {
    fn from(error: io::Error) -> Self {
        Self::Io {
            message: "I/O operation failed".to_string(),
            source: error,
        }
    }
}

impl From<serde_json::Error> for ScalingError {
    fn from(error: serde_json::Error) -> Self {
        Self::Config {
            message: format!("JSON serialization/deserialization failed: {}", error),
            field: None,
        }
    }
}

impl From<bincode::Error> for ScalingError {
    fn from(error: bincode::Error) -> Self {
        Self::Cache {
            message: format!("Binary serialization failed: {}", error),
            source: Some(Box::new(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_error_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err = ScalingError::io("Failed to read file", io_err);

        match &err {
            ScalingError::Io { message, .. } => {
                assert_eq!(message, "Failed to read file");
            }
            _ => panic!("Expected Io error"),
        }

        let err_string = format!("{}", err);
        assert!(err_string.contains("Failed to read file"));
    }

    #[test]
    fn test_scaling_error_path() {
        let err = ScalingError::path("Invalid path", "/some/path");

        match &err {
            ScalingError::Path { message, path, source } => {
                assert_eq!(message, "Invalid path");
                assert_eq!(*path, PathBuf::from("/some/path"));
                assert!(source.is_none());
            }
            _ => panic!("Expected Path error"),
        }

        let err_string = format!("{}", err);
        assert!(err_string.contains("Invalid path"));
        assert!(err_string.contains("/some/path"));
    }

    #[test]
    fn test_scaling_error_config() {
        let err = ScalingError::config("Invalid configuration");

        match &err {
            ScalingError::Config { message, field } => {
                assert_eq!(message, "Invalid configuration");
                assert!(field.is_none());
            }
            _ => panic!("Expected Config error"),
        }

        let err_string = format!("{}", err);
        assert!(err_string.contains("Invalid configuration"));
    }

    #[test]
    fn test_scaling_error_cache() {
        let err = ScalingError::cache("Cache miss");

        match err {
            ScalingError::Cache { message, source } => {
                assert_eq!(message, "Cache miss");
                assert!(source.is_none());
            }
            _ => panic!("Expected Cache error"),
        }
    }

    #[test]
    fn test_scaling_error_streaming() {
        let err = ScalingError::streaming("Stream interrupted");

        match err {
            ScalingError::Streaming { message, source } => {
                assert_eq!(message, "Stream interrupted");
                assert!(source.is_none());
            }
            _ => panic!("Expected Streaming error"),
        }
    }

    #[test]
    fn test_scaling_error_parallel() {
        let err = ScalingError::parallel("Thread panic");

        match err {
            ScalingError::Parallel { message, source } => {
                assert_eq!(message, "Thread panic");
                assert!(source.is_none());
            }
            _ => panic!("Expected Parallel error"),
        }
    }

    #[test]
    fn test_scaling_error_memory() {
        let err = ScalingError::memory("Out of memory");

        match err {
            ScalingError::Memory { message, details } => {
                assert_eq!(message, "Out of memory");
                assert!(details.is_none());
            }
            _ => panic!("Expected Memory error"),
        }
    }

    #[test]
    fn test_scaling_error_signature() {
        let err = ScalingError::signature("Invalid signature");

        match err {
            ScalingError::Signature { message, source } => {
                assert_eq!(message, "Invalid signature");
                assert!(source.is_none());
            }
            _ => panic!("Expected Signature error"),
        }
    }

    #[test]
    fn test_scaling_error_profiling() {
        let err = ScalingError::profiling("Profiling failed");

        match err {
            ScalingError::Profiling { message, source } => {
                assert_eq!(message, "Profiling failed");
                assert!(source.is_none());
            }
            _ => panic!("Expected Profiling error"),
        }
    }

    #[test]
    fn test_scaling_error_timeout() {
        let err = ScalingError::timeout("Operation timed out", 5000);

        match &err {
            ScalingError::Timeout { message, timeout_ms } => {
                assert_eq!(message, "Operation timed out");
                assert_eq!(*timeout_ms, 5000);
            }
            _ => panic!("Expected Timeout error"),
        }

        let err_string = format!("{}", err);
        assert!(err_string.contains("5000ms"));
    }

    #[test]
    fn test_scaling_error_resource_limit() {
        let err = ScalingError::resource_limit("Memory limit exceeded", 1024, 2048);

        match &err {
            ScalingError::ResourceLimit { message, limit, actual } => {
                assert_eq!(message, "Memory limit exceeded");
                assert_eq!(*limit, 1024);
                assert_eq!(*actual, 2048);
            }
            _ => panic!("Expected ResourceLimit error"),
        }

        let err_string = format!("{}", err);
        assert!(err_string.contains("limit: 1024"));
        assert!(err_string.contains("actual: 2048"));
    }

    #[test]
    fn test_scaling_error_internal() {
        let err = ScalingError::internal("Unexpected state");

        match err {
            ScalingError::Internal { message, location } => {
                assert_eq!(message, "Unexpected state");
                assert!(location.is_none());
            }
            _ => panic!("Expected Internal error"),
        }
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let err: ScalingError = io_err.into();

        match err {
            ScalingError::Io { message, .. } => {
                assert!(message.contains("I/O operation failed"));
            }
            _ => panic!("Expected Io error"),
        }
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<i32>("invalid").unwrap_err();
        let err: ScalingError = json_err.into();

        match err {
            ScalingError::Config { message, field } => {
                assert!(message.contains("JSON"));
                assert!(field.is_none());
            }
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_scaling_error_debug() {
        let err = ScalingError::config("test error");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Config"));
        assert!(debug_str.contains("test error"));
    }

    #[test]
    fn test_scaling_error_display_all_variants() {
        // Test display for all error variants
        let errors = vec![
            ScalingError::io("io msg", io::Error::new(io::ErrorKind::Other, "test")),
            ScalingError::path("path msg", "/test"),
            ScalingError::config("config msg"),
            ScalingError::cache("cache msg"),
            ScalingError::streaming("streaming msg"),
            ScalingError::parallel("parallel msg"),
            ScalingError::memory("memory msg"),
            ScalingError::signature("signature msg"),
            ScalingError::profiling("profiling msg"),
            ScalingError::timeout("timeout msg", 100),
            ScalingError::resource_limit("resource msg", 10, 20),
            ScalingError::internal("internal msg"),
        ];

        for err in errors {
            let display = format!("{}", err);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_scaling_result_type_alias() {
        fn test_fn() -> ScalingResult<i32> {
            Ok(42)
        }

        assert_eq!(test_fn().unwrap(), 42);
    }

    #[test]
    fn test_scaling_result_error() {
        fn test_fn() -> ScalingResult<i32> {
            Err(ScalingError::config("test"))
        }

        assert!(test_fn().is_err());
    }

    #[test]
    fn test_error_into_string() {
        // Test that Into<String> works for error constructors
        let err = ScalingError::config(String::from("owned string"));
        match err {
            ScalingError::Config { message, .. } => {
                assert_eq!(message, "owned string");
            }
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_path_error_into_pathbuf() {
        // Test that Into<PathBuf> works for path error constructor
        let err = ScalingError::path("msg", String::from("/string/path"));
        match err {
            ScalingError::Path { path, .. } => {
                assert_eq!(path, PathBuf::from("/string/path"));
            }
            _ => panic!("Expected Path error"),
        }
    }

    #[test]
    fn test_error_message_formatting() {
        let io_err = ScalingError::Io {
            message: "read failed".to_string(),
            source: io::Error::new(io::ErrorKind::NotFound, "not found"),
        };
        assert!(format!("{}", io_err).contains("read failed"));

        let path_err = ScalingError::Path {
            message: "invalid".to_string(),
            path: PathBuf::from("/test"),
            source: None,
        };
        assert!(format!("{}", path_err).contains("/test"));

        let config_err = ScalingError::Config {
            message: "bad config".to_string(),
            field: Some("field_name".to_string()),
        };
        assert!(format!("{}", config_err).contains("bad config"));

        let memory_err = ScalingError::Memory {
            message: "oom".to_string(),
            details: Some("heap exhausted".to_string()),
        };
        assert!(format!("{}", memory_err).contains("oom"));

        let internal_err = ScalingError::Internal {
            message: "bug".to_string(),
            location: Some("module::func".to_string()),
        };
        assert!(format!("{}", internal_err).contains("bug"));
    }
}

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

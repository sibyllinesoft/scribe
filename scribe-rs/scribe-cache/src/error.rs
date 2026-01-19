//! Error types for scribe-cache

use std::path::PathBuf;
use thiserror::Error;

/// Result type for cache operations
pub type CacheResult<T> = Result<T, CacheError>;

/// Errors that can occur during cache operations
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Failed to open cache database: {0}")]
    DatabaseOpen(#[from] redb::DatabaseError),

    #[error("Database transaction error: {0}")]
    Transaction(#[from] redb::TransactionError),

    #[error("Database table error: {0}")]
    Table(#[from] redb::TableError),

    #[error("Database storage error: {0}")]
    Storage(#[from] redb::StorageError),

    #[error("Database commit error: {0}")]
    Commit(#[from] redb::CommitError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cache directory not found")]
    NoCacheDir,

    #[error("Cache version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },

    #[error("File not found in cache: {0}")]
    FileNotFound(PathBuf),

    #[error("Invalid cache state: {0}")]
    InvalidState(String),
}

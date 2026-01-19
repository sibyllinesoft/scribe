//! Error types for the index crate

use thiserror::Error;

/// Errors that can occur during index operations
#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("Query parse error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Index directory not found")]
    NoIndexDir,

    #[error("Schema mismatch - index needs rebuild")]
    SchemaMismatch,

    #[error("Index not initialized")]
    NotInitialized,
}

/// Result type for index operations
pub type IndexResult<T> = Result<T, IndexError>;

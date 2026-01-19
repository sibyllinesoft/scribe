//! Full-text search index for scribe with BM25 ranking
//!
//! This crate provides efficient code search using tantivy for:
//! - Query-hint based file relevance scoring
//! - Identifier and content search
//! - Code-aware tokenization
//!
//! The index is persisted alongside the cache for incremental updates.

pub mod error;
pub mod schema;
pub mod index;
pub mod query;
pub mod tokenizer;

pub use error::{IndexError, IndexResult};
pub use index::CodeIndex;
pub use query::{SearchQuery, SearchResult};
pub use schema::CodeDocument;

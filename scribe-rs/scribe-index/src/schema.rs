//! Schema definition for code document indexing

use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, INDEXED, STORED,
    STRING, TEXT,
};

/// Fields in the code document schema
#[derive(Clone)]
pub struct CodeFields {
    /// File path (stored, not tokenized)
    pub path: Field,
    /// Content hash for cache correlation
    pub content_hash: Field,
    /// Full file content (indexed for search)
    pub content: Field,
    /// Extracted symbols/identifiers (indexed with higher boost)
    pub symbols: Field,
    /// Import statements
    pub imports: Field,
    /// Programming language
    pub language: Field,
}

impl CodeFields {
    /// Create fields from a schema
    pub fn from_schema(schema: &Schema) -> Self {
        Self {
            path: schema.get_field("path").unwrap(),
            content_hash: schema.get_field("content_hash").unwrap(),
            content: schema.get_field("content").unwrap(),
            symbols: schema.get_field("symbols").unwrap(),
            imports: schema.get_field("imports").unwrap(),
            language: schema.get_field("language").unwrap(),
        }
    }
}

/// Create the tantivy schema for code indexing
pub fn create_schema() -> (Schema, CodeFields) {
    let mut schema_builder = Schema::builder();

    // Path: stored for retrieval, indexed as string for exact matching
    let path = schema_builder.add_text_field("path", STRING | STORED);

    // Content hash: fast field for filtering/correlation
    let content_hash = schema_builder.add_u64_field("content_hash", FAST | STORED);

    // Content: full text search with code-aware tokenization
    let content_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("code")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let content = schema_builder.add_text_field("content", content_options);

    // Symbols: indexed with default tokenization, boosted in queries
    let symbols = schema_builder.add_text_field("symbols", TEXT | STORED);

    // Imports: indexed for dependency search
    let imports = schema_builder.add_text_field("imports", TEXT | STORED);

    // Language: stored and indexed as string
    let language = schema_builder.add_text_field("language", STRING | STORED);

    let schema = schema_builder.build();
    let fields = CodeFields {
        path,
        content_hash,
        content,
        symbols,
        imports,
        language,
    };

    (schema, fields)
}

/// A document to be indexed
#[derive(Debug, Clone)]
pub struct CodeDocument {
    pub path: String,
    pub content_hash: u64,
    pub content: String,
    pub symbols: Vec<String>,
    pub imports: Vec<String>,
    pub language: String,
}

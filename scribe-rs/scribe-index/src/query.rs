//! Query types and result structures

use std::path::PathBuf;

/// A search query with options
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The query string (keywords, identifiers)
    pub query: String,
    /// Maximum number of results
    pub limit: usize,
    /// Optional language filter
    pub language: Option<String>,
    /// Optional path prefix filter
    pub path_prefix: Option<String>,
}

impl SearchQuery {
    /// Create a simple query
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 100,
            language: None,
            path_prefix: None,
        }
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Filter by language
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Filter by path prefix
    pub fn with_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.path_prefix = Some(prefix.into());
        self
    }

    /// Build the tantivy query string
    pub fn to_query_string(&self) -> String {
        let mut parts = vec![self.query.clone()];

        if let Some(lang) = &self.language {
            parts.push(format!("language:{}", lang));
        }

        if let Some(prefix) = &self.path_prefix {
            parts.push(format!("path:{}*", prefix));
        }

        parts.join(" ")
    }
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            query: String::new(),
            limit: 100,
            language: None,
            path_prefix: None,
        }
    }
}

/// A search result with relevance information
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// File path
    pub path: PathBuf,
    /// BM25 relevance score
    pub score: f32,
    /// Normalized score (0-1)
    pub normalized_score: f32,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(path: PathBuf, score: f32) -> Self {
        Self {
            path,
            score,
            normalized_score: 0.0,
        }
    }
}

/// Normalize scores to 0-1 range
pub fn normalize_scores(results: &mut [SearchResult]) {
    if results.is_empty() {
        return;
    }

    let max_score = results.iter().map(|r| r.score).fold(0.0f32, f32::max);

    if max_score > 0.0 {
        for result in results.iter_mut() {
            result.normalized_score = result.score / max_score;
        }
    }
}

/// Convert raw search results to SearchResult structs
pub fn to_search_results(raw: Vec<(String, f32)>) -> Vec<SearchResult> {
    let mut results: Vec<_> = raw
        .into_iter()
        .map(|(path, score)| SearchResult::new(PathBuf::from(path), score))
        .collect();

    normalize_scores(&mut results);
    results
}

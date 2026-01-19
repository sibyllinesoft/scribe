//! BM25-based query relevance scoring for file selection.
//!
//! This module provides BM25 text relevance scoring using tantivy
//! to boost files that match the query hint.

#[cfg(feature = "bm25")]
use scribe_index::{CodeDocument, CodeIndex};
#[cfg(feature = "bm25")]
use std::collections::HashMap;
#[cfg(feature = "bm25")]
use std::path::{Path, PathBuf};
#[cfg(feature = "bm25")]
use tracing::{debug, info, warn};

#[cfg(feature = "bm25")]
use crate::io::streaming::FileMetadata;

/// BM25 scorer for query-based file relevance
#[cfg(feature = "bm25")]
pub struct Bm25Scorer {
    index: CodeIndex,
}

#[cfg(feature = "bm25")]
impl Bm25Scorer {
    /// Create a new BM25 scorer for a repository
    pub fn new(repo_path: &Path) -> Result<Self, scribe_index::IndexError> {
        let index = CodeIndex::open_for_repo(repo_path)?;
        Ok(Self { index })
    }

    /// Index a batch of files for BM25 search
    pub fn index_files(&self, files: &[FileMetadata]) -> Result<(), scribe_index::IndexError> {
        let docs: Vec<CodeDocument> = files
            .iter()
            .filter_map(|file| {
                let content = std::fs::read_to_string(&file.path).ok()?;
                let content_hash = scribe_cache::ContentHash::from_content(content.as_bytes());

                // Extract simple symbols (function/class names)
                let symbols = extract_symbols(&content, &file.language);

                Some(CodeDocument {
                    path: file.path.to_string_lossy().to_string(),
                    content_hash: content_hash.as_u64(),
                    content,
                    symbols,
                    imports: vec![],
                    language: file.language.clone(),
                })
            })
            .collect();

        if !docs.is_empty() {
            self.index.index_documents(&docs)?;
            self.index.reload()?;
            info!("Indexed {} files for BM25 search", docs.len());
        }

        Ok(())
    }

    /// Get BM25 relevance scores for files given a query
    pub fn score_files(
        &self,
        query: &str,
        files: &[PathBuf],
    ) -> Result<HashMap<PathBuf, f32>, scribe_index::IndexError> {
        let scored = self.index.score_files(query, files)?;
        Ok(scored.into_iter().collect())
    }

    /// Search for top relevant files
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>, scribe_index::IndexError> {
        self.index.search(query, limit)
    }
}

/// Extract simple symbols from code content
#[cfg(feature = "bm25")]
fn extract_symbols(content: &str, language: &str) -> Vec<String> {
    let mut symbols = Vec::new();

    // Simple regex-based extraction for common patterns
    let patterns = match language.to_lowercase().as_str() {
        "rust" => vec![
            r"fn\s+(\w+)",
            r"struct\s+(\w+)",
            r"enum\s+(\w+)",
            r"trait\s+(\w+)",
            r"impl\s+(\w+)",
        ],
        "python" => vec![
            r"def\s+(\w+)",
            r"class\s+(\w+)",
        ],
        "go" => vec![
            r"func\s+(\w+)",
            r"func\s+\([^)]+\)\s+(\w+)",
            r"type\s+(\w+)\s+struct",
            r"type\s+(\w+)\s+interface",
        ],
        "javascript" | "typescript" => vec![
            r"function\s+(\w+)",
            r"class\s+(\w+)",
            r"const\s+(\w+)\s*=",
            r"let\s+(\w+)\s*=",
        ],
        "java" => vec![
            r"class\s+(\w+)",
            r"interface\s+(\w+)",
            r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(",
        ],
        _ => vec![],
    };

    for pattern in patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            for cap in re.captures_iter(content) {
                if let Some(name) = cap.get(1) {
                    symbols.push(name.as_str().to_string());
                }
            }
        }
    }

    symbols
}

/// Combine BM25 score with base file score
#[cfg(feature = "bm25")]
pub fn combine_scores(base_score: f64, bm25_score: f32, bm25_weight: f64) -> f64 {
    // Normalize BM25 score (typically ranges from 0 to ~30)
    let normalized_bm25 = (bm25_score as f64 / 10.0).min(3.0);

    // Combine scores: base + weighted BM25 boost
    base_score + (normalized_bm25 * bm25_weight)
}

#[cfg(test)]
#[cfg(feature = "bm25")]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_extract_symbols_rust() {
        let content = r#"
fn main() {}
struct Config {}
enum Status { Active, Inactive }
trait Handler {}
impl Config {}
"#;
        let symbols = extract_symbols(content, "rust");
        assert!(symbols.contains(&"main".to_string()));
        assert!(symbols.contains(&"Config".to_string()));
        assert!(symbols.contains(&"Status".to_string()));
        assert!(symbols.contains(&"Handler".to_string()));
    }

    #[test]
    fn test_extract_symbols_go() {
        let content = r#"
func HandleRequest() {}
func (c *Context) JSON() {}
type Router struct {}
type Handler interface {}
"#;
        let symbols = extract_symbols(content, "go");
        assert!(symbols.contains(&"HandleRequest".to_string()));
        assert!(symbols.contains(&"Router".to_string()));
        assert!(symbols.contains(&"Handler".to_string()));
    }

    #[test]
    fn test_combine_scores() {
        let base = 2.0;
        let bm25 = 15.0; // High relevance
        let weight = 1.0;

        let combined = combine_scores(base, bm25, weight);
        assert!(combined > base, "BM25 should boost the score");

        // Zero BM25 shouldn't change score much
        let combined_zero = combine_scores(base, 0.0, weight);
        assert!((combined_zero - base).abs() < 0.01);
    }
}

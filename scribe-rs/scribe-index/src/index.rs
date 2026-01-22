//! Main code index implementation with tantivy

use crate::error::{IndexError, IndexResult};
use crate::schema::{create_schema, CodeDocument, CodeFields};
use crate::tokenizer::register_code_tokenizer;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tantivy::{doc, Directory, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};
use tracing::{debug, info, warn};

/// Default heap size for index writer (50MB)
const DEFAULT_HEAP_SIZE: usize = 50_000_000;

/// Main code index for BM25-based search
pub struct CodeIndex {
    index: Index,
    reader: IndexReader,
    fields: CodeFields,
    index_dir: PathBuf,
}

impl CodeIndex {
    /// Open or create an index at the given path
    pub fn open(index_dir: &Path) -> IndexResult<Self> {
        std::fs::create_dir_all(index_dir)?;

        let (schema, fields) = create_schema();

        // Try to open existing index or create new one
        let index = if index_dir.join("meta.json").exists() {
            match Index::open_in_dir(index_dir) {
                Ok(idx) => {
                    // Verify schema compatibility
                    if idx.schema() != schema {
                        warn!("Schema mismatch, rebuilding index");
                        std::fs::remove_dir_all(index_dir)?;
                        std::fs::create_dir_all(index_dir)?;
                        Index::create_in_dir(index_dir, schema.clone())?
                    } else {
                        idx
                    }
                }
                Err(e) => {
                    warn!("Failed to open index: {}, rebuilding", e);
                    std::fs::remove_dir_all(index_dir)?;
                    std::fs::create_dir_all(index_dir)?;
                    Index::create_in_dir(index_dir, schema.clone())?
                }
            }
        } else {
            Index::create_in_dir(index_dir, schema.clone())?
        };

        // Register custom tokenizer
        register_code_tokenizer(&index);

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        let fields = CodeFields::from_schema(&index.schema());

        Ok(Self {
            index,
            reader,
            fields,
            index_dir: index_dir.to_path_buf(),
        })
    }

    /// Open index from the default cache location for a repository
    pub fn open_for_repo(repo_path: &Path) -> IndexResult<Self> {
        let repo_id = scribe_cache::keys::repo_identifier(repo_path);
        let cache_dir = dirs::cache_dir()
            .ok_or(IndexError::NoIndexDir)?
            .join("scribe")
            .join(&repo_id)
            .join("index");

        info!(
            "Opening index at {} for repo {}",
            cache_dir.display(),
            repo_id
        );
        Self::open(&cache_dir)
    }

    /// Index a batch of documents (parallel)
    pub fn index_documents(&self, documents: &[CodeDocument]) -> IndexResult<()> {
        if documents.is_empty() {
            return Ok(());
        }

        let mut writer: IndexWriter = self.index.writer(DEFAULT_HEAP_SIZE)?;

        // Process documents in parallel to build tantivy docs
        let tantivy_docs: Vec<_> = documents
            .par_iter()
            .map(|doc| {
                let mut tantivy_doc = TantivyDocument::new();
                tantivy_doc.add_text(self.fields.path, &doc.path);
                tantivy_doc.add_u64(self.fields.content_hash, doc.content_hash);
                tantivy_doc.add_text(self.fields.content, &doc.content);
                tantivy_doc.add_text(self.fields.symbols, &doc.symbols.join(" "));
                tantivy_doc.add_text(self.fields.imports, &doc.imports.join(" "));
                tantivy_doc.add_text(self.fields.language, &doc.language);
                (doc.path.clone(), tantivy_doc)
            })
            .collect();

        // Add documents sequentially (tantivy requirement)
        for (path, tantivy_doc) in tantivy_docs {
            // Delete existing document with same path first
            let path_term = tantivy::Term::from_field_text(self.fields.path, &path);
            writer.delete_term(path_term);
            writer.add_document(tantivy_doc)?;
        }

        writer.commit()?;

        debug!("Indexed {} documents", documents.len());
        Ok(())
    }

    /// Remove documents by path
    pub fn remove_documents(&self, paths: &[PathBuf]) -> IndexResult<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let mut writer: IndexWriter = self.index.writer(DEFAULT_HEAP_SIZE)?;

        for path in paths {
            let path_str = path.to_string_lossy();
            let term = tantivy::Term::from_field_text(self.fields.path, &path_str);
            writer.delete_term(term);
        }

        writer.commit()?;

        debug!("Removed {} documents from index", paths.len());
        Ok(())
    }

    /// Search the index with a query string
    /// Returns file paths sorted by BM25 relevance score
    pub fn search(&self, query_str: &str, limit: usize) -> IndexResult<Vec<(String, f32)>> {
        let searcher = self.reader.searcher();

        // Create query parser that searches across all relevant fields
        // Boost symbols field for better identifier matching
        let mut query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.fields.content,
                self.fields.symbols,
                self.fields.imports,
            ],
        );

        // Boost symbols field
        query_parser.set_field_boost(self.fields.symbols, 2.0);

        let query = query_parser.parse_query(query_str)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let results: Vec<_> = top_docs
            .into_iter()
            .filter_map(|(score, doc_address)| {
                let doc: TantivyDocument = searcher.doc(doc_address).ok()?;
                let path = doc.get_first(self.fields.path)?.as_str()?.to_string();
                Some((path, score))
            })
            .collect();

        Ok(results)
    }

    /// Search and return just paths with scores
    pub fn search_paths(&self, query_str: &str, limit: usize) -> IndexResult<Vec<String>> {
        let results = self.search(query_str, limit)?;
        Ok(results.into_iter().map(|(path, _)| path).collect())
    }

    /// Get BM25 scores for specific files given a query
    /// Returns a map of path -> score (files not in results get score 0)
    pub fn score_files(
        &self,
        query_str: &str,
        file_paths: &[PathBuf],
    ) -> IndexResult<Vec<(PathBuf, f32)>> {
        // Search with high limit to get all potential matches
        let all_results = self.search(query_str, file_paths.len().max(1000))?;

        // Create lookup map
        let score_map: std::collections::HashMap<_, _> = all_results.into_iter().collect();

        // Map requested files to their scores
        let scored: Vec<_> = file_paths
            .iter()
            .map(|path| {
                let path_str = path.to_string_lossy().to_string();
                let score = score_map.get(&path_str).copied().unwrap_or(0.0);
                (path.clone(), score)
            })
            .collect();

        Ok(scored)
    }

    /// Get the index directory
    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// Force reload the reader to see latest changes
    pub fn reload(&self) -> IndexResult<()> {
        self.reader.reload()?;
        Ok(())
    }

    /// Get document count
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_index() -> (TempDir, CodeIndex) {
        let temp = TempDir::new().unwrap();
        let index = CodeIndex::open(temp.path()).unwrap();
        (temp, index)
    }

    #[test]
    fn test_create_and_search() {
        let (_temp, index) = create_test_index();

        let docs = vec![
            CodeDocument {
                path: "router.go".to_string(),
                content_hash: 12345,
                content: "func RedirectTrailingSlash() bool".to_string(),
                symbols: vec!["RedirectTrailingSlash".to_string()],
                imports: vec![],
                language: "go".to_string(),
            },
            CodeDocument {
                path: "handler.go".to_string(),
                content_hash: 67890,
                content: "func handleRequest(ctx Context)".to_string(),
                symbols: vec!["handleRequest".to_string()],
                imports: vec![],
                language: "go".to_string(),
            },
        ];

        index.index_documents(&docs).unwrap();
        index.reload().unwrap();

        let results = index.search("RedirectTrailingSlash", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "router.go");
    }

    #[test]
    fn test_camel_case_search() {
        let (_temp, index) = create_test_index();

        let docs = vec![CodeDocument {
            path: "router.go".to_string(),
            content_hash: 12345,
            content: "func RedirectTrailingSlash() bool".to_string(),
            symbols: vec!["RedirectTrailingSlash".to_string()],
            imports: vec![],
            language: "go".to_string(),
        }];

        index.index_documents(&docs).unwrap();
        index.reload().unwrap();

        // Search for individual words should match
        let results = index.search("redirect trailing", 10).unwrap();
        assert!(!results.is_empty());

        let results = index.search("trailing slash", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_score_files() {
        let (_temp, index) = create_test_index();

        let docs = vec![
            CodeDocument {
                path: "relevant.go".to_string(),
                content_hash: 1,
                content: "func parseConfig() Config".to_string(),
                symbols: vec!["parseConfig".to_string(), "Config".to_string()],
                imports: vec![],
                language: "go".to_string(),
            },
            CodeDocument {
                path: "irrelevant.go".to_string(),
                content_hash: 2,
                content: "func handleNetwork()".to_string(),
                symbols: vec!["handleNetwork".to_string()],
                imports: vec![],
                language: "go".to_string(),
            },
        ];

        index.index_documents(&docs).unwrap();
        index.reload().unwrap();

        let files = vec![
            PathBuf::from("relevant.go"),
            PathBuf::from("irrelevant.go"),
            PathBuf::from("nonexistent.go"),
        ];

        let scores = index.score_files("Config parse", &files).unwrap();

        // relevant.go should have highest score
        let relevant_score = scores
            .iter()
            .find(|(p, _)| p.to_str() == Some("relevant.go"))
            .unwrap()
            .1;
        let irrelevant_score = scores
            .iter()
            .find(|(p, _)| p.to_str() == Some("irrelevant.go"))
            .unwrap()
            .1;
        let nonexistent_score = scores
            .iter()
            .find(|(p, _)| p.to_str() == Some("nonexistent.go"))
            .unwrap()
            .1;

        assert!(relevant_score > irrelevant_score);
        assert_eq!(nonexistent_score, 0.0);
    }
}

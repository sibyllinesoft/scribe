//! Integration tests for scribe-cache and scribe-index working together

use scribe_cache::{CachedFileData, ChangedFile, ContentHash, ScribeCache};
use scribe_index::{CodeDocument, CodeIndex};
use std::path::PathBuf;
use tempfile::TempDir;

/// Test the full workflow: detect changes, index, and search
#[test]
fn test_full_workflow() {
    // Create a temp directory for our test repo
    let repo = TempDir::new().unwrap();
    let repo_path = repo.path();

    // Create some test files
    std::fs::write(repo_path.join("router.go"), r#"
package gin

// RedirectTrailingSlash enables automatic redirection
var RedirectTrailingSlash = true

// RedirectFixedPath enables case-insensitive path matching
var RedirectFixedPath = true

func handleRequest(ctx Context) error {
    return nil
}
"#).unwrap();

    std::fs::write(repo_path.join("context.go"), r#"
package gin

type Context struct {
    Request *http.Request
    Writer  ResponseWriter
}

func (c *Context) JSON(code int, obj any) {
    c.Writer.WriteHeader(code)
}
"#).unwrap();

    std::fs::write(repo_path.join("utils.go"), r#"
package gin

func parseQueryParams(query string) map[string]string {
    result := make(map[string]string)
    return result
}
"#).unwrap();

    // Open cache and index
    let cache_dir = TempDir::new().unwrap();
    let index_dir = TempDir::new().unwrap();

    let cache = ScribeCache::open(cache_dir.path()).unwrap();
    let index = CodeIndex::open(index_dir.path()).unwrap();

    // Discover files
    let files: Vec<PathBuf> = std::fs::read_dir(repo_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "go").unwrap_or(false))
        .collect();

    // Check what changed (all files are new)
    let diff = cache.diff_files(&files);
    assert_eq!(diff.new_files.len(), 3);
    assert!(diff.is_up_to_date() == false);

    // Process changed files and build index documents
    let mut docs = Vec::new();
    let mut cache_entries = Vec::new();

    for changed in diff.new_files.iter().chain(diff.changed.iter()) {
        let content = String::from_utf8_lossy(&changed.content).to_string();

        // Extract symbols (simplified - just look for func/var declarations)
        let symbols: Vec<String> = content
            .lines()
            .filter_map(|line| {
                if line.starts_with("func ") || line.starts_with("var ") {
                    line.split_whitespace().nth(1).map(|s| {
                        s.trim_matches(|c: char| !c.is_alphanumeric()).to_string()
                    })
                } else {
                    None
                }
            })
            .collect();

        // Create index document
        docs.push(CodeDocument {
            path: changed.path.to_string_lossy().to_string(),
            content_hash: changed.hash.as_u64(),
            content: content.clone(),
            symbols: symbols.clone(),
            imports: vec![],
            language: "go".to_string(),
        });

        // Create cache entry
        cache_entries.push((
            changed.hash,
            CachedFileData {
                token_count: content.len() / 4, // rough estimate
                symbols,
                imports: vec![],
                language: "go".to_string(),
                size: changed.content.len() as u64,
            },
        ));
    }

    // Index documents
    index.index_documents(&docs).unwrap();
    index.reload().unwrap();

    // Store in cache
    cache.store_file_data_batch(&cache_entries).unwrap();
    cache.update_path_mappings(&diff.new_files.iter().chain(diff.changed.iter()).cloned().collect::<Vec<_>>()).unwrap();

    // Test searching for specific symbols
    let results = index.search("RedirectTrailingSlash", 10).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].0.contains("router.go"));

    // Test searching for function names
    let results = index.search("handleRequest Context", 10).unwrap();
    assert!(!results.is_empty());

    // Test camelCase splitting - search for "redirect trailing"
    let results = index.search("redirect trailing", 10).unwrap();
    assert!(!results.is_empty());

    // Test score_files for ranking
    let scored = index.score_files("Context JSON", &files).unwrap();

    // context.go should score highest for "Context JSON"
    let context_score = scored.iter()
        .find(|(p, _)| p.to_string_lossy().contains("context.go"))
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    let utils_score = scored.iter()
        .find(|(p, _)| p.to_string_lossy().contains("utils.go"))
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    assert!(context_score > utils_score, "context.go should score higher than utils.go for 'Context JSON'");

    // Now check cache - files should be unchanged on second check
    let diff2 = cache.diff_files(&files);
    assert_eq!(diff2.unchanged.len(), 3);
    assert!(diff2.is_up_to_date());

    // Modify a file
    std::fs::write(repo_path.join("router.go"), r#"
package gin

// RedirectTrailingSlash - modified!
var RedirectTrailingSlash = false

func handleRequest(ctx Context) error {
    return nil
}
"#).unwrap();

    // Check changes - should detect modification
    let diff3 = cache.diff_files(&files);
    assert_eq!(diff3.changed.len(), 1);
    assert_eq!(diff3.unchanged.len(), 2);

    // Verify cached data retrieval
    for path in &diff3.unchanged {
        let data = cache.get_file_data_by_path(path);
        assert!(data.is_some(), "Should have cached data for unchanged file");
    }
}

/// Test that the index handles updates correctly
#[test]
fn test_incremental_index_update() {
    let index_dir = TempDir::new().unwrap();
    let index = CodeIndex::open(index_dir.path()).unwrap();

    // Initial document
    let doc1 = CodeDocument {
        path: "main.rs".to_string(),
        content_hash: 123,
        content: "fn main() { println!(\"hello\"); }".to_string(),
        symbols: vec!["main".to_string()],
        imports: vec![],
        language: "rust".to_string(),
    };

    index.index_documents(&[doc1]).unwrap();
    index.reload().unwrap();

    assert_eq!(index.num_docs(), 1);

    // Update the same file
    let doc1_updated = CodeDocument {
        path: "main.rs".to_string(),
        content_hash: 456,
        content: "fn main() { println!(\"goodbye\"); }".to_string(),
        symbols: vec!["main".to_string()],
        imports: vec![],
        language: "rust".to_string(),
    };

    index.index_documents(&[doc1_updated]).unwrap();
    index.reload().unwrap();

    // Should still be 1 doc (updated, not duplicated)
    assert_eq!(index.num_docs(), 1);

    // Search should find the updated content
    let results = index.search("goodbye", 10).unwrap();
    assert!(!results.is_empty());

    // Old content should not be found
    let results = index.search("hello", 10).unwrap();
    assert!(results.is_empty());
}

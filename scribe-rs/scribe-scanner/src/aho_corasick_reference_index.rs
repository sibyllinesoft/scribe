//! High-performance file reference indexing using Aho-Corasick multi-pattern search.
//!
//! This module replaces the quadratic O(files × bytes) string matching with
//! linear O(total_bytes + matches) scanning using a single compiled automaton.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use scribe_core::{Result, ScribeError};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Global stoplist for tokens that are too common to be useful
const STOPLIST: &[&str] = &[
    "README",
    "readme",
    "index",
    "test",
    "tests",
    "spec",
    "specs",
    "main",
    "app",
    "lib",
    "src",
    "build",
    "dist",
    "target",
    "node_modules",
    "package",
    "config",
    "util",
    "utils",
    "helper",
    "helpers",
    "common",
    "base",
    "core",
    "api",
    "client",
    "server",
    "data",
    "model",
    "view",
];

/// Configuration for the reference indexing
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Minimum token length to consider
    pub min_token_length: usize,
    /// Maximum token length to consider  
    pub max_token_length: usize,
    /// Maximum number of patterns to compile into automaton
    pub max_patterns: usize,
    /// Chunk size for streaming reads
    pub chunk_size: usize,
    /// Whether to include file stems (filename without extension)
    pub include_stems: bool,
    /// Whether to include directory components
    pub include_dirs: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            min_token_length: 3,
            max_token_length: 64,
            max_patterns: 10_000,
            chunk_size: 64 * 1024, // 64 KiB
            include_stems: true,
            include_dirs: false,
        }
    }
}

/// A token extracted from a file path for reference matching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileToken {
    pub text: String,
    pub token_type: TokenType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenType {
    Basename,  // file.ext
    Stem,      // file (without extension)
    Directory, // parent directory name
}

/// High-performance file reference index using Aho-Corasick
#[derive(Debug)]
pub struct AhoCorasickReferenceIndex {
    /// Maps tokens to the files that contain them (as IDs)
    token_to_files: HashMap<String, Vec<usize>>,
    /// Maps file IDs to their generated tokens
    file_tokens: Vec<Vec<FileToken>>,
    /// Compiled Aho-Corasick automaton for pattern matching
    automaton: Option<AhoCorasick>,
    /// Patterns used in the automaton (for result mapping)
    patterns: Vec<String>,
    /// Configuration
    config: IndexConfig,
    /// Performance metrics
    pub metrics: IndexMetrics,
}

/// Performance metrics for the indexing process
#[derive(Debug, Default, Clone)]
pub struct IndexMetrics {
    pub total_files: usize,
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub automaton_build_ms: u64,
    pub bytes_scanned: u64,
    pub matches_found: usize,
    pub boundary_checks: usize,
    pub valid_matches: usize,
}

impl AhoCorasickReferenceIndex {
    /// Create a new index from file paths
    pub fn new(file_paths: &[impl AsRef<Path>], config: IndexConfig) -> Result<Self> {
        let start_time = std::time::Instant::now();

        let mut index = Self {
            token_to_files: HashMap::new(),
            file_tokens: Vec::with_capacity(file_paths.len()),
            automaton: None,
            patterns: Vec::new(),
            config,
            metrics: IndexMetrics {
                total_files: file_paths.len(),
                ..Default::default()
            },
        };

        // Step 1: Extract tokens from all file paths
        for (file_id, path) in file_paths.iter().enumerate() {
            let tokens = index.extract_tokens(path.as_ref());

            for token in &tokens {
                index
                    .token_to_files
                    .entry(token.text.clone())
                    .or_insert_with(Vec::new)
                    .push(file_id);
            }

            index.metrics.total_tokens += tokens.len();
            index.file_tokens.push(tokens);
        }

        // Step 2: Build Aho-Corasick automaton
        index.build_automaton()?;

        index.metrics.automaton_build_ms = start_time.elapsed().as_millis() as u64;

        Ok(index)
    }

    /// Extract relevant tokens from a file path
    fn extract_tokens(&self, path: &Path) -> Vec<FileToken> {
        let mut tokens = Vec::new();

        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            // Add basename (full filename)
            if self.is_valid_token(filename) {
                tokens.push(FileToken {
                    text: filename.to_string(),
                    token_type: TokenType::Basename,
                });
            }

            // Add stem (filename without extension) if enabled
            if self.config.include_stems {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if self.is_valid_token(stem) && stem != filename {
                        tokens.push(FileToken {
                            text: stem.to_string(),
                            token_type: TokenType::Stem,
                        });
                    }
                }
            }
        }

        // Add directory components if enabled
        if self.config.include_dirs {
            if let Some(parent) = path.parent() {
                for component in parent.components() {
                    if let Some(dir_name) = component.as_os_str().to_str() {
                        if self.is_valid_token(dir_name) {
                            tokens.push(FileToken {
                                text: dir_name.to_string(),
                                token_type: TokenType::Directory,
                            });
                        }
                    }
                }
            }
        }

        tokens
    }

    /// Check if a token meets the criteria for inclusion
    fn is_valid_token(&self, token: &str) -> bool {
        let len = token.len();

        // Length check
        if len < self.config.min_token_length || len > self.config.max_token_length {
            return false;
        }

        // Stoplist check
        if STOPLIST.contains(&token) {
            return false;
        }

        // All-numeric check
        if token.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }

        // Must contain at least one alphanumeric character
        if !token.chars().any(|c| c.is_alphanumeric()) {
            return false;
        }

        true
    }

    /// Build the Aho-Corasick automaton from collected tokens
    fn build_automaton(&mut self) -> Result<()> {
        // Collect unique patterns, limited by max_patterns
        let mut patterns: Vec<String> = self.token_to_files.keys().cloned().collect();
        patterns.sort_by_key(|p| std::cmp::Reverse(self.token_to_files[p].len())); // Sort by frequency
        patterns.truncate(self.config.max_patterns);

        self.metrics.unique_tokens = patterns.len();

        if patterns.is_empty() {
            return Ok(());
        }

        // Build automaton with optimal settings for our use case
        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::Standard) // Find all matches
            .build(&patterns)
            .map_err(|e| {
                ScribeError::io(
                    format!("Failed to build Aho-Corasick automaton: {}", e),
                    std::io::Error::new(std::io::ErrorKind::Other, e),
                )
            })?;

        self.patterns = patterns;
        self.automaton = Some(automaton);

        Ok(())
    }

    /// Scan a file for references to other files in the index
    pub fn scan_file_for_references<P: AsRef<Path>>(&mut self, file_path: P) -> Result<Vec<usize>> {
        let Some(ref automaton) = self.automaton else {
            return Ok(Vec::new());
        };

        let file = File::open(file_path.as_ref()).map_err(|e| {
            ScribeError::io(
                format!("Failed to open file: {}", file_path.as_ref().display()),
                e,
            )
        })?;

        let mut reader = BufReader::new(file);
        let mut referenced_files = HashSet::new();

        // Stream the file in chunks with overlap for boundary handling
        let max_pattern_len = self.patterns.iter().map(|p| p.len()).max().unwrap_or(0);
        let overlap = max_pattern_len.saturating_sub(1);

        let mut buffer = vec![0u8; self.config.chunk_size + overlap];
        let mut carried_bytes = 0;

        loop {
            // Read chunk, preserving overlap from previous chunk
            let bytes_read = reader
                .read(&mut buffer[carried_bytes..])
                .map_err(|e| ScribeError::io("Failed to read file chunk".to_string(), e))?;

            if bytes_read == 0 {
                break; // EOF
            }

            let total_bytes = carried_bytes + bytes_read;
            let chunk = &buffer[..total_bytes];
            self.metrics.bytes_scanned += bytes_read as u64;

            // Find all matches in this chunk
            for mat in automaton.find_iter(chunk) {
                self.metrics.matches_found += 1;

                // Perform boundary check
                if self.is_boundary_valid(chunk, mat.start(), mat.end()) {
                    self.metrics.boundary_checks += 1;
                    self.metrics.valid_matches += 1;

                    // Map pattern back to file IDs
                    let pattern = &self.patterns[mat.pattern()];
                    if let Some(file_ids) = self.token_to_files.get(pattern) {
                        referenced_files.extend(file_ids.iter().copied());
                    }
                }
            }

            // Prepare overlap for next iteration
            if bytes_read == self.config.chunk_size {
                // Copy overlap bytes to start of buffer
                if overlap > 0 && total_bytes > overlap {
                    buffer.copy_within(total_bytes - overlap.., 0);
                    carried_bytes = overlap;
                } else {
                    carried_bytes = 0;
                }
            } else {
                break; // Last chunk, no more data
            }
        }

        Ok(referenced_files.into_iter().collect())
    }

    /// Check if a match occurs at a valid token boundary
    fn is_boundary_valid(&self, text: &[u8], start: usize, end: usize) -> bool {
        // Check left boundary
        let left_ok = start == 0 || !self.is_identifier_char(text[start - 1]);

        // Check right boundary
        let right_ok = end == text.len() || !self.is_identifier_char(text[end]);

        left_ok && right_ok
    }

    /// Check if a byte is part of an identifier (alphanumeric, underscore, dot, slash, dash)
    fn is_identifier_char(&self, byte: u8) -> bool {
        byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.' || byte == b'/' || byte == b'-'
    }

    /// Get the number of files that reference the given file ID
    pub fn get_reference_count(&self, file_id: usize) -> usize {
        // This would need to be computed by scanning all files
        // For now, return 0 as this is a placeholder for compatibility
        0
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &IndexMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_token_extraction() {
        let config = IndexConfig::default();
        let index = AhoCorasickReferenceIndex {
            token_to_files: HashMap::new(),
            file_tokens: Vec::new(),
            automaton: None,
            patterns: Vec::new(),
            config,
            metrics: IndexMetrics::default(),
        };

        let path = Path::new("src/auth/user.rs");
        let tokens = index.extract_tokens(path);

        assert!(tokens
            .iter()
            .any(|t| t.text == "user.rs" && t.token_type == TokenType::Basename));
        assert!(tokens
            .iter()
            .any(|t| t.text == "user" && t.token_type == TokenType::Stem));
    }

    #[test]
    fn test_stoplist_filtering() {
        let config = IndexConfig::default();
        let index = AhoCorasickReferenceIndex {
            token_to_files: HashMap::new(),
            file_tokens: Vec::new(),
            automaton: None,
            patterns: Vec::new(),
            config,
            metrics: IndexMetrics::default(),
        };

        assert!(!index.is_valid_token("README"));
        assert!(!index.is_valid_token("test"));
        assert!(!index.is_valid_token("123"));
        assert!(index.is_valid_token("auth"));
        assert!(index.is_valid_token("user"));
    }

    #[test]
    fn test_boundary_detection() {
        let mut index = AhoCorasickReferenceIndex {
            token_to_files: HashMap::new(),
            file_tokens: Vec::new(),
            automaton: None,
            patterns: Vec::new(),
            config: IndexConfig::default(),
            metrics: IndexMetrics::default(),
        };

        let text = b"import user from './user.js'";
        assert!(index.is_boundary_valid(text, 7, 11)); // "user" with spaces around
        assert!(!index.is_boundary_valid(text, 8, 11)); // "ser" inside "user"
    }

    #[tokio::test]
    async fn test_file_scanning() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        let file1_path = temp_dir.path().join("auth.rs");
        let file2_path = temp_dir.path().join("user.rs");

        fs::write(&file1_path, "use user;\nmod user_service;")?;
        fs::write(&file2_path, "struct User { name: String }")?;

        // Create index
        let config = IndexConfig::default();
        let paths = vec![file1_path.clone(), file2_path.clone()];
        let mut index = AhoCorasickReferenceIndex::new(&paths, config)?;

        // Scan auth.rs for references
        let references = index.scan_file_for_references(&file1_path)?;

        // Should find reference to user.rs (file ID 1)
        assert!(references.contains(&1));

        Ok(())
    }
}

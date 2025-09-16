//! Multi-Fidelity Demotion System for V4 variant
//!
//! Implements progressive content reduction: FULL → CHUNK → SIGNATURE
//! - Intelligent content reduction when approaching budget limits
//! - Maintains most important information while reducing token usage
//! - Progressive degradation preserves critical functionality
//! - Language-specific semantic chunking and signature extraction using tree-sitter AST parsing

use crate::ast_parser::{AstChunk as AstParserChunk, AstLanguage, AstParser, AstSignature};
use scribe_core::{Result, ScribeError};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Content fidelity levels for demotion system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FidelityMode {
    /// Complete file content
    Full,
    /// Important chunks only
    Chunk,
    /// Type signatures and interfaces only
    Signature,
}

/// Result of applying demotion to a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemotionResult {
    pub original_path: String,
    pub original_tokens: usize,
    pub demoted_tokens: usize,
    pub fidelity_mode: FidelityMode,
    pub content: String,
    pub chunks_kept: usize,
    pub chunks_total: usize,
    pub compression_ratio: f64,
    pub quality_score: f64, // How much important info was preserved
}

/// Information about a code chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_type: String,
    pub content: String,
    pub importance_score: f64,
    pub estimated_tokens: usize,
    pub dependencies: Vec<String>, // Other chunks this depends on
}

/// Splits code into semantic chunks for selective demotion using tree-sitter AST parsing
pub struct CodeChunker {
    language_cache: HashMap<String, Option<AstLanguage>>,
    ast_parser: Rc<RefCell<AstParser>>,
}

impl CodeChunker {
    pub fn new(ast_parser: Rc<RefCell<AstParser>>) -> Self {
        Self {
            language_cache: HashMap::new(),
            ast_parser,
        }
    }

    pub fn detect_language(&mut self, file_path: &str) -> Option<AstLanguage> {
        if let Some(cached) = self.language_cache.get(file_path) {
            return cached.clone();
        }

        let ext = file_path.split('.').last().unwrap_or("");
        let language = AstLanguage::from_extension(ext);

        self.language_cache
            .insert(file_path.to_string(), language.clone());
        language
    }

    pub fn chunk_content(&mut self, content: &str, file_path: &str) -> Result<Vec<ChunkInfo>> {
        let language = match self.detect_language(file_path) {
            Some(lang) => lang,
            None => return Ok(self.chunk_generic(content)),
        };

        // Use AST parser to get semantic chunks
        // Create a temporary file path with the correct extension
        let temp_path = format!(
            "temp.{}",
            match language {
                AstLanguage::Python => "py",
                AstLanguage::JavaScript => "js",
                AstLanguage::TypeScript => "ts",
                AstLanguage::Go => "go",
                AstLanguage::Rust => "rs",
            }
        );
        let ast_chunks = self
            .ast_parser
            .borrow_mut()
            .parse_chunks(content, &temp_path)?;

        let mut chunks = Vec::new();
        for ast_chunk in ast_chunks {
            let chunk = ChunkInfo {
                start_line: ast_chunk.start_line,
                end_line: ast_chunk.end_line,
                chunk_type: ast_chunk.chunk_type,
                content: ast_chunk.content.clone(),
                importance_score: ast_chunk.importance_score,
                estimated_tokens: ast_chunk.content.len() / 4, // Rough estimate
                dependencies: Vec::new(), // Could be enhanced with dependency analysis
            };
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn chunk_generic(&self, content: &str) -> Vec<ChunkInfo> {
        let lines: Vec<&str> = content.split('\n').collect();
        let chunk_size = 20; // Lines per chunk
        let mut chunks = Vec::new();

        for (i, chunk_lines) in lines.chunks(chunk_size).enumerate() {
            let start_line = i * chunk_size + 1;
            let end_line = start_line + chunk_lines.len() - 1;
            let content = chunk_lines.join("\n");

            let chunk = ChunkInfo {
                start_line,
                end_line,
                chunk_type: "generic".to_string(),
                content: content.clone(),
                importance_score: 0.5, // Default score for generic chunks
                estimated_tokens: content.len() / 4,
                dependencies: Vec::new(),
            };
            chunks.push(chunk);
        }

        chunks
    }

    pub fn select_chunks_by_budget(&self, chunks: &[ChunkInfo], token_budget: usize) -> Vec<usize> {
        // Sort chunks by importance score (descending)
        let mut indexed_chunks: Vec<(usize, &ChunkInfo)> = chunks.iter().enumerate().collect();
        indexed_chunks.sort_by(|a, b| {
            b.1.importance_score
                .partial_cmp(&a.1.importance_score)
                .unwrap()
        });

        let mut selected_indices = Vec::new();
        let mut used_tokens = 0;

        for (index, chunk) in indexed_chunks {
            if used_tokens + chunk.estimated_tokens <= token_budget {
                selected_indices.push(index);
                used_tokens += chunk.estimated_tokens;
            }
        }

        // Sort indices to maintain original order
        selected_indices.sort();
        selected_indices
    }
}

impl Default for CodeChunker {
    fn default() -> Self {
        let ast_parser = Rc::new(RefCell::new(
            AstParser::new().expect("Failed to create AstParser"),
        ));
        Self::new(ast_parser)
    }
}

/// Extracts type signatures and interfaces for the highest fidelity reduction using tree-sitter
pub struct SignatureExtractor {
    ast_parser: Rc<RefCell<AstParser>>,
}

impl SignatureExtractor {
    pub fn new(ast_parser: Rc<RefCell<AstParser>>) -> Self {
        Self { ast_parser }
    }

    pub fn extract_signatures(&mut self, content: &str, file_path: &str) -> Result<Vec<String>> {
        let language = AstLanguage::from_extension(file_path.split('.').last().unwrap_or(""));

        let language = match language {
            Some(lang) => lang,
            None => return Ok(vec![self.extract_generic_signatures(content)]),
        };

        // Use AST parser to extract signatures
        // Create a temporary file path with the correct extension
        let temp_path = format!(
            "temp.{}",
            match language {
                AstLanguage::Python => "py",
                AstLanguage::JavaScript => "js",
                AstLanguage::TypeScript => "ts",
                AstLanguage::Go => "go",
                AstLanguage::Rust => "rs",
            }
        );
        let signatures = self
            .ast_parser
            .borrow_mut()
            .extract_signatures(content, &temp_path)?;

        Ok(signatures
            .into_iter()
            .map(|sig| format!("{}:{} // {}", sig.name, sig.signature_type, sig.signature))
            .collect())
    }

    fn extract_generic_signatures(&self, content: &str) -> String {
        // For unknown file types, try to extract function-like patterns
        let lines: Vec<&str> = content.lines().collect();
        let mut signatures = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("#") {
                continue;
            }

            // Look for function-like patterns (very basic heuristic)
            if trimmed.contains("(")
                && trimmed.contains(")")
                && (trimmed.contains("def ")
                    || trimmed.contains("function ")
                    || trimmed.contains("fn ")
                    || trimmed.contains("func "))
            {
                signatures.push(trimmed.to_string());
            }
        }

        signatures.join("\n")
    }
}

impl Default for SignatureExtractor {
    fn default() -> Self {
        let ast_parser = Rc::new(RefCell::new(
            AstParser::new().expect("Failed to create AstParser"),
        ));
        Self::new(ast_parser)
    }
}

/// Main demotion engine that orchestrates the progressive content reduction
pub struct DemotionEngine {
    chunker: CodeChunker,
    signature_extractor: SignatureExtractor,
}

impl DemotionEngine {
    pub fn new() -> Result<Self> {
        let ast_parser = Rc::new(RefCell::new(AstParser::new()?));
        Ok(Self {
            chunker: CodeChunker::new(ast_parser.clone()),
            signature_extractor: SignatureExtractor::new(ast_parser),
        })
    }

    pub fn demote_content(
        &mut self,
        content: &str,
        file_path: &str,
        target_mode: FidelityMode,
        token_budget: Option<usize>,
    ) -> Result<DemotionResult> {
        let original_tokens = content.len() / 4; // Rough estimate

        match target_mode {
            FidelityMode::Full => Ok(DemotionResult {
                original_path: file_path.to_string(),
                original_tokens,
                demoted_tokens: original_tokens,
                fidelity_mode: FidelityMode::Full,
                content: content.to_string(),
                chunks_kept: 1,
                chunks_total: 1,
                compression_ratio: 1.0,
                quality_score: 1.0,
            }),
            FidelityMode::Chunk => {
                self.demote_to_chunks(content, file_path, token_budget, original_tokens)
            }
            FidelityMode::Signature => {
                self.demote_to_signatures(content, file_path, original_tokens)
            }
        }
    }

    fn demote_to_chunks(
        &mut self,
        content: &str,
        file_path: &str,
        token_budget: Option<usize>,
        original_tokens: usize,
    ) -> Result<DemotionResult> {
        let chunks = self.chunker.chunk_content(content, file_path)?;
        let chunks_total = chunks.len();

        let selected_indices = if let Some(budget) = token_budget {
            self.chunker.select_chunks_by_budget(&chunks, budget)
        } else {
            // Keep all chunks if no budget specified
            (0..chunks.len()).collect()
        };

        let chunks_kept = selected_indices.len();
        let selected_chunks: Vec<String> = selected_indices
            .iter()
            .map(|&i| chunks[i].content.clone())
            .collect();

        let demoted_content = if selected_chunks.is_empty() {
            // Fallback: create basic structure summary if no chunks extracted
            let lines: Vec<&str> = content.lines().collect();
            lines
                .iter()
                .filter(|line| !line.trim().is_empty())
                .take(10) // Take more lines for chunks than signatures
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            selected_chunks.join("\n\n// ... [content omitted] ...\n\n")
        };

        let demoted_tokens = if demoted_content.is_empty() {
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!(
                    "CHUNK DEMOTION BUG: Empty demoted content for {}",
                    file_path
                );
            }
            1 // Minimum tokens for empty content
        } else {
            let word_count = demoted_content.split_whitespace().count();
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!(
                    "CHUNK DEMOTION DEBUG: {} has {} chars, {} words -> {} tokens",
                    file_path,
                    demoted_content.len(),
                    word_count,
                    std::cmp::max(1, word_count)
                );
            }
            std::cmp::max(1, word_count)
        };

        let quality_score = if chunks_total > 0 {
            selected_indices
                .iter()
                .map(|&i| chunks[i].importance_score)
                .sum::<f64>()
                / chunks_total as f64
        } else {
            0.0
        };

        Ok(DemotionResult {
            original_path: file_path.to_string(),
            original_tokens,
            demoted_tokens,
            fidelity_mode: FidelityMode::Chunk,
            content: demoted_content,
            chunks_kept,
            chunks_total,
            compression_ratio: demoted_tokens as f64 / original_tokens as f64,
            quality_score,
        })
    }

    fn demote_to_signatures(
        &mut self,
        content: &str,
        file_path: &str,
        original_tokens: usize,
    ) -> Result<DemotionResult> {
        let signatures = self
            .signature_extractor
            .extract_signatures(content, file_path)?;

        // If no signatures extracted, fall back to basic fallback
        let demoted_content = if signatures.is_empty() {
            // Create basic fallback signatures for Rust files
            let lines: Vec<&str> = content.lines().collect();
            let mut fallback_signatures = Vec::new();

            // Extract key lines (function definitions, struct/type definitions, imports)
            for line in lines.iter() {
                let trimmed = line.trim();
                if trimmed.starts_with("pub fn ")
                    || trimmed.starts_with("fn ")
                    || trimmed.starts_with("pub struct ")
                    || trimmed.starts_with("struct ")
                    || trimmed.starts_with("pub enum ")
                    || trimmed.starts_with("enum ")
                    || trimmed.starts_with("pub trait ")
                    || trimmed.starts_with("trait ")
                    || trimmed.starts_with("impl ")
                    || trimmed.starts_with("use ")
                    || trimmed.starts_with("mod ")
                    || trimmed.starts_with("pub mod ")
                {
                    fallback_signatures.push(trimmed.to_string());
                }
            }

            if fallback_signatures.is_empty() {
                // Last resort: take first few non-empty lines
                lines
                    .iter()
                    .filter(|line| !line.trim().is_empty())
                    .take(5)
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                fallback_signatures.join("\n")
            }
        } else {
            signatures.join("\n")
        };

        // Better token estimation based on actual content
        let demoted_tokens = if demoted_content.is_empty() {
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!("DEMOTION BUG: Empty demoted content for {}", file_path);
            }
            1 // Minimum tokens for empty content
        } else {
            let word_count = demoted_content.split_whitespace().count();
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!(
                    "DEMOTION DEBUG: {} has {} chars, {} words -> {} tokens",
                    file_path,
                    demoted_content.len(),
                    word_count,
                    std::cmp::max(1, word_count)
                );
            }
            std::cmp::max(1, word_count)
        };

        Ok(DemotionResult {
            original_path: file_path.to_string(),
            original_tokens,
            demoted_tokens,
            fidelity_mode: FidelityMode::Signature,
            content: demoted_content,
            chunks_kept: signatures.len(),
            chunks_total: signatures.len(), // For signatures, kept = total
            compression_ratio: demoted_tokens as f64 / original_tokens as f64,
            quality_score: 0.8, // Signatures preserve high-level structure
        })
    }
}

impl Default for DemotionEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create DemotionEngine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        let ast_parser = Rc::new(RefCell::new(AstParser::new().unwrap()));
        let mut chunker = CodeChunker::new(ast_parser);

        assert_eq!(
            chunker.detect_language("test.py"),
            Some(AstLanguage::Python)
        );
        assert_eq!(
            chunker.detect_language("test.js"),
            Some(AstLanguage::JavaScript)
        );
        assert_eq!(
            chunker.detect_language("test.ts"),
            Some(AstLanguage::TypeScript)
        );
        assert_eq!(chunker.detect_language("test.go"), Some(AstLanguage::Go));
        assert_eq!(chunker.detect_language("test.rs"), Some(AstLanguage::Rust));
        assert_eq!(chunker.detect_language("test.txt"), None);
    }

    #[test]
    fn test_fidelity_modes() {
        let engine = DemotionEngine::new().unwrap();

        // Test that all fidelity modes are correctly represented
        assert_eq!(FidelityMode::Full as u8, 0);
        assert_ne!(FidelityMode::Chunk, FidelityMode::Signature);
    }

    #[test]
    fn test_chunk_budget_selection() {
        let ast_parser = Rc::new(RefCell::new(AstParser::new().unwrap()));
        let chunker = CodeChunker::new(ast_parser);

        let chunks = vec![
            ChunkInfo {
                start_line: 1,
                end_line: 5,
                chunk_type: "function".to_string(),
                content: "def test(): pass".to_string(),
                importance_score: 0.8,
                estimated_tokens: 10,
                dependencies: vec![],
            },
            ChunkInfo {
                start_line: 6,
                end_line: 10,
                chunk_type: "comment".to_string(),
                content: "# This is a comment".to_string(),
                importance_score: 0.2,
                estimated_tokens: 5,
                dependencies: vec![],
            },
        ];

        let selected = chunker.select_chunks_by_budget(&chunks, 12);
        assert_eq!(selected, vec![0]); // Should select the function with higher importance
    }
}

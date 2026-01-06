//! AST parsing types and data structures

use serde::{Deserialize, Serialize};
use tree_sitter::Language;

/// Supported programming languages for AST parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AstLanguage {
    Python,
    JavaScript,
    TypeScript,
    Go,
    Rust,
}

impl AstLanguage {
    /// Get the tree-sitter language for this language
    pub fn tree_sitter_language(&self) -> Language {
        match self {
            AstLanguage::Python => tree_sitter_python::language(),
            AstLanguage::JavaScript => tree_sitter_javascript::language(),
            AstLanguage::TypeScript => tree_sitter_typescript::language_typescript(),
            AstLanguage::Go => tree_sitter_go::language(),
            AstLanguage::Rust => tree_sitter_rust::language(),
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "py" | "pyi" | "pyw" => Some(AstLanguage::Python),
            "js" | "mjs" | "cjs" => Some(AstLanguage::JavaScript),
            "ts" | "mts" | "cts" => Some(AstLanguage::TypeScript),
            "go" => Some(AstLanguage::Go),
            "rs" => Some(AstLanguage::Rust),
            _ => None,
        }
    }
}

/// Import information extracted from AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstImport {
    /// The module being imported
    pub module: String,
    /// Optional alias for the import
    pub alias: Option<String>,
    /// Specific items being imported (for from-imports)
    pub items: Vec<String>,
    /// Line number where the import appears
    pub line_number: usize,
    /// Whether this is a relative import
    pub is_relative: bool,
}

/// A parsed code chunk with semantic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstChunk {
    /// The text content of this chunk
    pub content: String,
    /// Type of the chunk (function, class, import, etc.)
    pub chunk_type: String,
    /// Start line (1-indexed)
    pub start_line: usize,
    /// End line (1-indexed)
    pub end_line: usize,
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Semantic importance score (0.0-1.0)
    pub importance_score: f64,
    /// Estimated token count
    pub estimated_tokens: usize,
    /// Dependencies (other chunks this depends on)
    pub dependencies: Vec<String>,
    /// Name/identifier of this chunk (if applicable)
    pub name: Option<String>,
    /// Whether this is publicly visible
    pub is_public: bool,
    /// Whether this has documentation
    pub has_documentation: bool,
}

/// Extracted signature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstSignature {
    /// The signature text
    pub signature: String,
    /// Type of signature (function, class, interface, etc.)
    pub signature_type: String,
    /// Name/identifier
    pub name: String,
    /// Parameters (for functions/methods)
    pub parameters: Vec<String>,
    /// Return type (if available)
    pub return_type: Option<String>,
    /// Whether this is public/exported
    pub is_public: bool,
    /// Line number
    pub line: usize,
    /// Associated documentation (docstring or doc comment)
    pub documentation: Option<String>,
}

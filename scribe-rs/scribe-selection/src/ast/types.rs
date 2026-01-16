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
            "js" | "mjs" | "cjs" | "jsx" => Some(AstLanguage::JavaScript),
            "ts" | "mts" | "cts" | "tsx" => Some(AstLanguage::TypeScript),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_language_python() {
        let lang = AstLanguage::Python;
        assert!(matches!(lang, AstLanguage::Python));
    }

    #[test]
    fn test_ast_language_javascript() {
        let lang = AstLanguage::JavaScript;
        assert!(matches!(lang, AstLanguage::JavaScript));
    }

    #[test]
    fn test_ast_language_typescript() {
        let lang = AstLanguage::TypeScript;
        assert!(matches!(lang, AstLanguage::TypeScript));
    }

    #[test]
    fn test_ast_language_go() {
        let lang = AstLanguage::Go;
        assert!(matches!(lang, AstLanguage::Go));
    }

    #[test]
    fn test_ast_language_rust() {
        let lang = AstLanguage::Rust;
        assert!(matches!(lang, AstLanguage::Rust));
    }

    #[test]
    fn test_ast_language_from_extension_python() {
        assert!(matches!(AstLanguage::from_extension("py"), Some(AstLanguage::Python)));
        assert!(matches!(AstLanguage::from_extension("pyi"), Some(AstLanguage::Python)));
        assert!(matches!(AstLanguage::from_extension("pyw"), Some(AstLanguage::Python)));
    }

    #[test]
    fn test_ast_language_from_extension_javascript() {
        assert!(matches!(AstLanguage::from_extension("js"), Some(AstLanguage::JavaScript)));
        assert!(matches!(AstLanguage::from_extension("mjs"), Some(AstLanguage::JavaScript)));
        assert!(matches!(AstLanguage::from_extension("cjs"), Some(AstLanguage::JavaScript)));
    }

    #[test]
    fn test_ast_language_from_extension_typescript() {
        assert!(matches!(AstLanguage::from_extension("ts"), Some(AstLanguage::TypeScript)));
        assert!(matches!(AstLanguage::from_extension("mts"), Some(AstLanguage::TypeScript)));
        assert!(matches!(AstLanguage::from_extension("cts"), Some(AstLanguage::TypeScript)));
    }

    #[test]
    fn test_ast_language_from_extension_go() {
        assert!(matches!(AstLanguage::from_extension("go"), Some(AstLanguage::Go)));
    }

    #[test]
    fn test_ast_language_from_extension_rust() {
        assert!(matches!(AstLanguage::from_extension("rs"), Some(AstLanguage::Rust)));
    }

    #[test]
    fn test_ast_language_from_extension_unknown() {
        assert!(AstLanguage::from_extension("xyz").is_none());
        assert!(AstLanguage::from_extension("c").is_none());
        assert!(AstLanguage::from_extension("cpp").is_none());
    }

    #[test]
    fn test_ast_language_from_extension_case_insensitive() {
        assert!(matches!(AstLanguage::from_extension("PY"), Some(AstLanguage::Python)));
        assert!(matches!(AstLanguage::from_extension("RS"), Some(AstLanguage::Rust)));
        assert!(matches!(AstLanguage::from_extension("Js"), Some(AstLanguage::JavaScript)));
    }

    #[test]
    fn test_ast_language_tree_sitter_language() {
        // Just verify these don't panic - the tree_sitter Language type isn't easily testable
        let _ = AstLanguage::Python.tree_sitter_language();
        let _ = AstLanguage::JavaScript.tree_sitter_language();
        let _ = AstLanguage::TypeScript.tree_sitter_language();
        let _ = AstLanguage::Go.tree_sitter_language();
        let _ = AstLanguage::Rust.tree_sitter_language();
    }

    #[test]
    fn test_ast_language_clone() {
        let lang = AstLanguage::Rust;
        let cloned = lang.clone();
        assert!(matches!(cloned, AstLanguage::Rust));
    }

    #[test]
    fn test_ast_language_copy() {
        let lang = AstLanguage::Python;
        let copied = lang;
        assert!(matches!(lang, AstLanguage::Python));
        assert!(matches!(copied, AstLanguage::Python));
    }

    #[test]
    fn test_ast_language_eq() {
        let lang1 = AstLanguage::Go;
        let lang2 = AstLanguage::Go;
        let lang3 = AstLanguage::Rust;
        assert_eq!(lang1, lang2);
        assert_ne!(lang1, lang3);
    }

    #[test]
    fn test_ast_language_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(AstLanguage::Python);
        set.insert(AstLanguage::Rust);
        set.insert(AstLanguage::Python); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_ast_language_serialize() {
        let lang = AstLanguage::TypeScript;
        let json = serde_json::to_string(&lang).unwrap();
        let deserialized: AstLanguage = serde_json::from_str(&json).unwrap();
        assert_eq!(lang, deserialized);
    }

    #[test]
    fn test_ast_language_debug() {
        let lang = AstLanguage::JavaScript;
        let debug = format!("{:?}", lang);
        assert!(debug.contains("JavaScript"));
    }

    #[test]
    fn test_ast_import_creation() {
        let import = AstImport {
            module: "std::collections".to_string(),
            alias: Some("col".to_string()),
            items: vec!["HashMap".to_string(), "HashSet".to_string()],
            line_number: 5,
            is_relative: false,
        };

        assert_eq!(import.module, "std::collections");
        assert_eq!(import.alias, Some("col".to_string()));
        assert_eq!(import.items.len(), 2);
        assert_eq!(import.line_number, 5);
        assert!(!import.is_relative);
    }

    #[test]
    fn test_ast_import_relative() {
        let import = AstImport {
            module: "..utils".to_string(),
            alias: None,
            items: vec!["helper".to_string()],
            line_number: 10,
            is_relative: true,
        };

        assert!(import.is_relative);
        assert!(import.alias.is_none());
    }

    #[test]
    fn test_ast_import_clone() {
        let import = AstImport {
            module: "module".to_string(),
            alias: None,
            items: vec![],
            line_number: 1,
            is_relative: false,
        };

        let cloned = import.clone();
        assert_eq!(import.module, cloned.module);
    }

    #[test]
    fn test_ast_import_serialize() {
        let import = AstImport {
            module: "test".to_string(),
            alias: Some("t".to_string()),
            items: vec!["a".to_string()],
            line_number: 3,
            is_relative: false,
        };

        let json = serde_json::to_string(&import).unwrap();
        assert!(json.contains("test"));

        let deserialized: AstImport = serde_json::from_str(&json).unwrap();
        assert_eq!(import.module, deserialized.module);
    }

    #[test]
    fn test_ast_import_debug() {
        let import = AstImport {
            module: "debug".to_string(),
            alias: None,
            items: vec![],
            line_number: 1,
            is_relative: false,
        };

        let debug = format!("{:?}", import);
        assert!(debug.contains("AstImport"));
    }

    #[test]
    fn test_ast_chunk_creation() {
        let chunk = AstChunk {
            content: "fn main() {}".to_string(),
            chunk_type: "function".to_string(),
            start_line: 1,
            end_line: 3,
            start_byte: 0,
            end_byte: 12,
            importance_score: 0.8,
            estimated_tokens: 10,
            dependencies: vec!["std".to_string()],
            name: Some("main".to_string()),
            is_public: true,
            has_documentation: false,
        };

        assert_eq!(chunk.content, "fn main() {}");
        assert_eq!(chunk.chunk_type, "function");
        assert!(chunk.is_public);
    }

    #[test]
    fn test_ast_chunk_clone() {
        let chunk = AstChunk {
            content: "code".to_string(),
            chunk_type: "import".to_string(),
            start_line: 1,
            end_line: 1,
            start_byte: 0,
            end_byte: 4,
            importance_score: 0.5,
            estimated_tokens: 2,
            dependencies: vec![],
            name: None,
            is_public: false,
            has_documentation: false,
        };

        let cloned = chunk.clone();
        assert_eq!(chunk.content, cloned.content);
    }

    #[test]
    fn test_ast_chunk_serialize() {
        let chunk = AstChunk {
            content: "test".to_string(),
            chunk_type: "class".to_string(),
            start_line: 5,
            end_line: 10,
            start_byte: 50,
            end_byte: 100,
            importance_score: 0.9,
            estimated_tokens: 25,
            dependencies: vec![],
            name: Some("TestClass".to_string()),
            is_public: true,
            has_documentation: true,
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("TestClass"));

        let deserialized: AstChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk.name, deserialized.name);
    }

    #[test]
    fn test_ast_chunk_debug() {
        let chunk = AstChunk {
            content: "debug".to_string(),
            chunk_type: "expression".to_string(),
            start_line: 1,
            end_line: 1,
            start_byte: 0,
            end_byte: 5,
            importance_score: 0.1,
            estimated_tokens: 1,
            dependencies: vec![],
            name: None,
            is_public: false,
            has_documentation: false,
        };

        let debug = format!("{:?}", chunk);
        assert!(debug.contains("AstChunk"));
    }

    #[test]
    fn test_ast_signature_creation() {
        let sig = AstSignature {
            signature: "fn process(data: &str) -> Result<(), Error>".to_string(),
            signature_type: "function".to_string(),
            name: "process".to_string(),
            parameters: vec!["data: &str".to_string()],
            return_type: Some("Result<(), Error>".to_string()),
            is_public: true,
            line: 42,
            documentation: Some("Processes data".to_string()),
        };

        assert_eq!(sig.name, "process");
        assert_eq!(sig.signature_type, "function");
        assert!(sig.is_public);
        assert!(sig.documentation.is_some());
    }

    #[test]
    fn test_ast_signature_no_documentation() {
        let sig = AstSignature {
            signature: "fn internal()".to_string(),
            signature_type: "function".to_string(),
            name: "internal".to_string(),
            parameters: vec![],
            return_type: None,
            is_public: false,
            line: 10,
            documentation: None,
        };

        assert!(!sig.is_public);
        assert!(sig.documentation.is_none());
    }

    #[test]
    fn test_ast_signature_clone() {
        let sig = AstSignature {
            signature: "test".to_string(),
            signature_type: "method".to_string(),
            name: "test_method".to_string(),
            parameters: vec!["self".to_string()],
            return_type: Some("Self".to_string()),
            is_public: true,
            line: 1,
            documentation: None,
        };

        let cloned = sig.clone();
        assert_eq!(sig.name, cloned.name);
    }

    #[test]
    fn test_ast_signature_serialize() {
        let sig = AstSignature {
            signature: "pub fn new() -> Self".to_string(),
            signature_type: "function".to_string(),
            name: "new".to_string(),
            parameters: vec![],
            return_type: Some("Self".to_string()),
            is_public: true,
            line: 25,
            documentation: Some("Creates new instance".to_string()),
        };

        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("new"));
        assert!(json.contains("documentation"));

        let deserialized: AstSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(sig.name, deserialized.name);
    }

    #[test]
    fn test_ast_signature_debug() {
        let sig = AstSignature {
            signature: "debug".to_string(),
            signature_type: "interface".to_string(),
            name: "Interface".to_string(),
            parameters: vec![],
            return_type: None,
            is_public: true,
            line: 1,
            documentation: None,
        };

        let debug = format!("{:?}", sig);
        assert!(debug.contains("AstSignature"));
    }

    #[test]
    fn test_ast_signature_multiple_params() {
        let sig = AstSignature {
            signature: "fn multi(a: i32, b: String, c: bool) -> i32".to_string(),
            signature_type: "function".to_string(),
            name: "multi".to_string(),
            parameters: vec![
                "a: i32".to_string(),
                "b: String".to_string(),
                "c: bool".to_string(),
            ],
            return_type: Some("i32".to_string()),
            is_public: false,
            line: 50,
            documentation: None,
        };

        assert_eq!(sig.parameters.len(), 3);
    }
}

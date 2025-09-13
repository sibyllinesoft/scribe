//! Simple AST-based import extraction for analysis module
//!
//! This module provides a lightweight AST parser specifically for extracting
//! import statements from source code without the full complexity of the
//! scribe-selection module.

use std::collections::HashMap;
use scribe_core::Result;
use tree_sitter::{Parser, Language, Node, Tree};

/// Simple import information
#[derive(Debug, Clone)]
pub struct SimpleImport {
    /// The module being imported
    pub module: String,
    /// Line number where the import appears
    pub line_number: usize,
}

/// Supported programming languages for import extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportLanguage {
    Python,
    JavaScript,
    TypeScript,
    Go,
    Rust,
}

impl ImportLanguage {
    /// Get the tree-sitter language for this language
    pub fn tree_sitter_language(&self) -> Language {
        match self {
            ImportLanguage::Python => tree_sitter_python::language(),
            ImportLanguage::JavaScript => tree_sitter_javascript::language(),
            ImportLanguage::TypeScript => tree_sitter_typescript::language_typescript(),
            ImportLanguage::Go => tree_sitter_go::language(),
            ImportLanguage::Rust => tree_sitter_rust::language(),
        }
    }
    
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "py" | "pyi" | "pyw" => Some(ImportLanguage::Python),
            "js" | "mjs" | "cjs" => Some(ImportLanguage::JavaScript),
            "ts" | "mts" | "cts" => Some(ImportLanguage::TypeScript),
            "go" => Some(ImportLanguage::Go),
            "rs" => Some(ImportLanguage::Rust),
            _ => None,
        }
    }
}

/// Simple AST parser for import extraction
pub struct SimpleAstParser {
    parsers: HashMap<ImportLanguage, Parser>,
}

impl std::fmt::Debug for SimpleAstParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleAstParser")
            .field("parsers", &format!("[{} parsers]", self.parsers.len()))
            .finish()
    }
}

impl SimpleAstParser {
    /// Create a new simple AST parser
    pub fn new() -> Result<Self> {
        let mut parsers = HashMap::new();
        
        for language in [
            ImportLanguage::Python,
            ImportLanguage::JavaScript,
            ImportLanguage::TypeScript,
            ImportLanguage::Go,
            ImportLanguage::Rust,
        ] {
            let mut parser = Parser::new();
            parser.set_language(language.tree_sitter_language())
                .map_err(|e| scribe_core::ScribeError::parse(format!("Failed to set tree-sitter language: {}", e)))?;
            parsers.insert(language, parser);
        }
        
        Ok(Self { parsers })
    }
    
    /// Extract imports from the given content using tree-sitter
    pub fn extract_imports(&self, content: &str, language: ImportLanguage) -> Result<Vec<SimpleImport>> {
        // Create a fresh parser for this operation to avoid mutable borrow issues
        let mut parser = Parser::new();
        parser.set_language(language.tree_sitter_language()).map_err(|e| 
            scribe_core::ScribeError::parse(format!("Failed to set language: {}", e)))?;
        
        let tree = parser.parse(content, None)
            .ok_or_else(|| scribe_core::ScribeError::parse("Failed to parse content"))?;
        
        let mut imports = Vec::new();
        let root_node = tree.root_node();
        
        // Extract imports based on language
        match language {
            ImportLanguage::Python => {
                self.extract_python_imports(&root_node, content, &mut imports)?;
            }
            ImportLanguage::JavaScript | ImportLanguage::TypeScript => {
                self.extract_js_ts_imports(&root_node, content, &mut imports)?;
            }
            ImportLanguage::Go => {
                self.extract_go_imports(&root_node, content, &mut imports)?;
            }
            ImportLanguage::Rust => {
                self.extract_rust_imports(&root_node, content, &mut imports)?;
            }
        }
        
        Ok(imports)
    }

    /// Extract Python imports from AST
    fn extract_python_imports(&self, node: &Node, content: &str, imports: &mut Vec<SimpleImport>) -> Result<()> {
        if node.kind() == "import_statement" {
            // Handle import statements like "import os" or "import sys as system"
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "dotted_name" || child.kind() == "identifier" {
                        let module = self.node_text(child, content);
                        let line_number = child.start_position().row + 1;
                        
                        imports.push(SimpleImport {
                            module,
                            line_number,
                        });
                    }
                }
            }
        } else if node.kind() == "import_from_statement" {
            if let Some(module_node) = node.child_by_field_name("module_name") {
                let module = self.node_text(module_node, content);
                let line_number = node.start_position().row + 1;
                imports.push(SimpleImport {
                    module,
                    line_number,
                });
            }
        }
        
        // Recursively process children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.extract_python_imports(&child, content, imports)?;
            }
        }
        
        Ok(())
    }

    /// Extract JavaScript/TypeScript imports from AST
    fn extract_js_ts_imports(&self, node: &Node, content: &str, imports: &mut Vec<SimpleImport>) -> Result<()> {
        if node.kind() == "import_statement" {
            // Find the source
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "string" {
                        let mut module = self.node_text(child, content);
                        // Remove quotes
                        module = module.trim_matches('"').trim_matches('\'').to_string();
                        let line_number = node.start_position().row + 1;
                        imports.push(SimpleImport {
                            module,
                            line_number,
                        });
                        break;
                    }
                }
            }
        }
        
        // Recursively process children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.extract_js_ts_imports(&child, content, imports)?;
            }
        }
        
        Ok(())
    }

    /// Extract Go imports from AST
    fn extract_go_imports(&self, node: &Node, content: &str, imports: &mut Vec<SimpleImport>) -> Result<()> {
        if node.kind() == "import_spec" {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "interpreted_string_literal" {
                        let module = self.node_text(child, content);
                        let module = module.trim_matches('"').to_string();
                        let line_number = child.start_position().row + 1;
                        
                        imports.push(SimpleImport {
                            module,
                            line_number,
                        });
                    }
                }
            }
        }
        
        // Recursively process children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.extract_go_imports(&child, content, imports)?;
            }
        }
        
        Ok(())
    }

    /// Extract Rust imports from AST
    fn extract_rust_imports(&self, node: &Node, content: &str, imports: &mut Vec<SimpleImport>) -> Result<()> {
        if node.kind() == "use_declaration" {
            if let Some(use_tree) = node.child_by_field_name("argument") {
                let module = self.node_text(use_tree, content);
                let line_number = node.start_position().row + 1;
                
                imports.push(SimpleImport {
                    module,
                    line_number,
                });
            }
        }
        
        // Recursively process children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.extract_rust_imports(&child, content, imports)?;
            }
        }
        
        Ok(())
    }

    /// Helper to extract text from a node
    fn node_text(&self, node: Node, content: &str) -> String {
        content[node.start_byte()..node.end_byte()].to_string()
    }
}

impl Default for SimpleAstParser {
    fn default() -> Self {
        Self::new().expect("Failed to create SimpleAstParser")
    }
}
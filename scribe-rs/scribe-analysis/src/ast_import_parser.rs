//! Optimized AST-based import extraction for analysis module
//!
//! This module provides a high-performance AST parser specifically for extracting
//! import statements from source code using TreeCursor for efficient traversal
//! and parser reuse for better performance.

use scribe_core::Result;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tree_sitter::{Language, Node, Parser, Tree, TreeCursor};

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

/// Thread-safe parser pool for reusing parsers
static PARSER_POOL: Lazy<Arc<Mutex<HashMap<ImportLanguage, Vec<Parser>>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Node types that can contain imports - for fast filtering
const IMPORT_NODE_TYPES: &[&str] = &[
    "import_statement",
    "import_from_statement", 
    "use_declaration",
    "import_declaration",
    "import_spec",
    "source_file",
    "module",
];

/// Optimized AST parser for import extraction with parser reuse and TreeCursor traversal
pub struct SimpleAstParser {
    // We don't need to store parsers anymore - we use the pool
}

impl std::fmt::Debug for SimpleAstParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleAstParser")
            .field("parsers", &"[reusable pool]")
            .finish()
    }
}

impl SimpleAstParser {
    /// Create a new simple AST parser
    pub fn new() -> Result<Self> {
        // Initialize the parser pool on first creation
        Self::ensure_parser_pool_initialized()?;
        Ok(Self {})
    }

    /// Ensure the parser pool is initialized with all supported languages
    fn ensure_parser_pool_initialized() -> Result<()> {
        let mut pool = PARSER_POOL.lock().unwrap();
        
        for language in [
            ImportLanguage::Python,
            ImportLanguage::JavaScript,
            ImportLanguage::TypeScript,
            ImportLanguage::Go,
            ImportLanguage::Rust,
        ] {
            if !pool.contains_key(&language) {
                let mut parser = Parser::new();
                parser
                    .set_language(language.tree_sitter_language())
                    .map_err(|e| {
                        scribe_core::ScribeError::parse(format!(
                            "Failed to set tree-sitter language: {}",
                            e
                        ))
                    })?;
                pool.insert(language, vec![parser]);
            }
        }
        
        Ok(())
    }

    /// Get a parser from the pool or create a new one
    fn get_parser(&self, language: ImportLanguage) -> Result<Parser> {
        let mut pool = PARSER_POOL.lock().unwrap();
        
        if let Some(parsers) = pool.get_mut(&language) {
            if let Some(parser) = parsers.pop() {
                return Ok(parser);
            }
        }
        
        // Create a new parser if pool is empty
        let mut parser = Parser::new();
        parser
            .set_language(language.tree_sitter_language())
            .map_err(|e| {
                scribe_core::ScribeError::parse(format!(
                    "Failed to set tree-sitter language: {}",
                    e
                ))
            })?;
        Ok(parser)
    }

    /// Return a parser to the pool
    fn return_parser(&self, language: ImportLanguage, parser: Parser) {
        let mut pool = PARSER_POOL.lock().unwrap();
        pool.entry(language).or_insert_with(Vec::new).push(parser);
    }

    /// Extract imports from the given content using optimized tree-sitter traversal
    pub fn extract_imports(
        &self,
        content: &str,
        language: ImportLanguage,
    ) -> Result<Vec<SimpleImport>> {
        // Get parser from pool
        let mut parser = self.get_parser(language)?;
        
        let tree = parser
            .parse(content, None)
            .ok_or_else(|| scribe_core::ScribeError::parse("Failed to parse content"))?;

        let mut imports = Vec::new();
        
        // Use TreeCursor for efficient traversal
        let mut cursor = tree.walk();
        self.extract_imports_with_cursor(&mut cursor, content, language, &mut imports)?;

        // Return parser to pool
        self.return_parser(language, parser);

        Ok(imports)
    }

    /// Extract imports using TreeCursor for optimal performance
    fn extract_imports_with_cursor(
        &self,
        cursor: &mut TreeCursor,
        content: &str,
        language: ImportLanguage,
        imports: &mut Vec<SimpleImport>,
    ) -> Result<()> {
        let node = cursor.node();
        
        // Fast filter: skip nodes that can't contain imports
        if !self.node_can_contain_imports(node.kind()) {
            return Ok(());
        }

        // Process current node if it's an import
        if self.is_import_node(node.kind()) {
            self.extract_import_from_node(node, content, language, imports)?;
        }

        // Traverse children using cursor (much faster than child(i) loops)
        if cursor.goto_first_child() {
            loop {
                self.extract_imports_with_cursor(cursor, content, language, imports)?;
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }

        Ok(())
    }

    /// Check if a node type can contain imports (fast filter)
    fn node_can_contain_imports(&self, kind: &str) -> bool {
        IMPORT_NODE_TYPES.contains(&kind) || 
        kind.contains("import") || 
        kind.contains("use") ||
        kind == "program" ||
        kind == "translation_unit" ||
        kind == "block" ||
        kind == "statement_block"
    }

    /// Check if a node is an import statement
    fn is_import_node(&self, kind: &str) -> bool {
        matches!(kind, 
            "import_statement" | 
            "import_from_statement" | 
            "use_declaration" | 
            "import_declaration" | 
            "import_spec"
        )
    }

    /// Extract import from a specific node (no recursion needed)
    fn extract_import_from_node(
        &self,
        node: Node,
        content: &str,
        language: ImportLanguage,
        imports: &mut Vec<SimpleImport>,
    ) -> Result<()> {
        match language {
            ImportLanguage::Python => {
                self.extract_python_import_node(node, content, imports)?;
            }
            ImportLanguage::JavaScript | ImportLanguage::TypeScript => {
                self.extract_js_ts_import_node(node, content, imports)?;
            }
            ImportLanguage::Go => {
                self.extract_go_import_node(node, content, imports)?;
            }
            ImportLanguage::Rust => {
                self.extract_rust_import_node(node, content, imports)?;
            }
        }
        Ok(())
    }

    /// Extract Python import from a single node (optimized, no recursion)
    fn extract_python_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<SimpleImport>,
    ) -> Result<()> {
        if node.kind() == "import_statement" {
            // Handle import statements like "import os" or "import sys as system"
            let mut cursor = node.walk();
            if cursor.goto_first_child() {
                loop {
                    let child = cursor.node();
                    if child.kind() == "dotted_name" || child.kind() == "identifier" {
                        let module = self.node_text(child, content);
                        let line_number = child.start_position().row + 1;

                        imports.push(SimpleImport {
                            module,
                            line_number,
                        });
                    }
                    if !cursor.goto_next_sibling() {
                        break;
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
        Ok(())
    }

    /// Extract JavaScript/TypeScript import from a single node (optimized, no recursion)
    fn extract_js_ts_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<SimpleImport>,
    ) -> Result<()> {
        if node.kind() == "import_statement" {
            // Find the source
            let mut cursor = node.walk();
            if cursor.goto_first_child() {
                loop {
                    let child = cursor.node();
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
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract Go import from a single node (optimized, no recursion)
    fn extract_go_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<SimpleImport>,
    ) -> Result<()> {
        if node.kind() == "import_spec" {
            let mut cursor = node.walk();
            if cursor.goto_first_child() {
                loop {
                    let child = cursor.node();
                    if child.kind() == "interpreted_string_literal" {
                        let module = self.node_text(child, content);
                        let module = module.trim_matches('"').to_string();
                        let line_number = child.start_position().row + 1;

                        imports.push(SimpleImport {
                            module,
                            line_number,
                        });
                    }
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract Rust import from a single node (optimized, no recursion)
    fn extract_rust_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<SimpleImport>,
    ) -> Result<()> {
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
        Ok(())
    }

    /// Helper to extract text from a node
    fn node_text(&self, node: Node, content: &str) -> String {
        content[node.start_byte()..node.end_byte()].to_string()
    }

    /// Extract imports from multiple files in parallel for maximum performance
    pub fn extract_imports_parallel(
        &self,
        files: &[(String, String, ImportLanguage)], // (path, content, language)
    ) -> Result<Vec<(String, Vec<SimpleImport>)>> {
        // Use rayon for parallel processing
        files
            .par_iter()
            .map(|(path, content, language)| {
                let imports = self.extract_imports(content, *language)?;
                Ok((path.clone(), imports))
            })
            .collect()
    }

    /// Batch extract imports for multiple contents with the same language
    pub fn extract_imports_batch(
        &self,
        contents: &[&str],
        language: ImportLanguage,
    ) -> Result<Vec<Vec<SimpleImport>>> {
        contents
            .par_iter()
            .map(|content| self.extract_imports(content, language))
            .collect()
    }
}

impl Default for SimpleAstParser {
    fn default() -> Self {
        Self::new().expect("Failed to create SimpleAstParser")
    }
}

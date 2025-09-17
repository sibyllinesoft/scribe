//! Tree-sitter based AST parsing for accurate code analysis
//!
//! This module replaces regex-based parsing with proper syntax-aware analysis
//! using tree-sitter parsers for multiple programming languages.

use scribe_core::{Result, ScribeError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tree_sitter::{Language, Node, Parser, Query, QueryCursor, Tree};

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
}

/// Tree-sitter based AST parser and analyzer
pub struct AstParser {
    parsers: HashMap<AstLanguage, Parser>,
}

impl AstParser {
    /// Create a new AST parser with support for all languages
    pub fn new() -> Result<Self> {
        let mut parsers = HashMap::new();

        for language in [
            AstLanguage::Python,
            AstLanguage::JavaScript,
            AstLanguage::TypeScript,
            AstLanguage::Go,
            AstLanguage::Rust,
        ] {
            let mut parser = Parser::new();
            parser
                .set_language(language.tree_sitter_language())
                .map_err(|e| {
                    ScribeError::parse(format!("Failed to set tree-sitter language: {}", e))
                })?;
            parsers.insert(language, parser);
        }

        Ok(Self { parsers })
    }

    /// Parse code into chunks using tree-sitter AST
    pub fn parse_chunks(&mut self, content: &str, file_path: &str) -> Result<Vec<AstChunk>> {
        let language = self.detect_language(file_path)?;
        let parser = self
            .parsers
            .get_mut(&language)
            .ok_or_else(|| ScribeError::parse(format!("No parser for language: {:?}", language)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| ScribeError::parse("Failed to parse source code".to_string()))?;

        let chunks = match language {
            AstLanguage::Python => self.parse_python_chunks(content, &tree)?,
            AstLanguage::JavaScript => self.parse_javascript_chunks(content, &tree)?,
            AstLanguage::TypeScript => self.parse_typescript_chunks(content, &tree)?,
            AstLanguage::Go => self.parse_go_chunks(content, &tree)?,
            AstLanguage::Rust => self.parse_rust_chunks(content, &tree)?,
        };

        Ok(chunks)
    }

    /// Extract signatures using tree-sitter AST
    /// Extract imports from the given content using optimized tree-sitter traversal
    pub fn extract_imports(&self, content: &str, language: AstLanguage) -> Result<Vec<AstImport>> {
        // Create a fresh parser for this operation to avoid mutable borrow issues
        let mut parser = Parser::new();
        parser
            .set_language(language.tree_sitter_language())
            .map_err(|e| ScribeError::parse(format!("Failed to set language: {}", e)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| ScribeError::parse("Failed to parse content"))?;

        let mut imports = Vec::new();

        // Use TreeCursor for efficient traversal
        let mut cursor = tree.walk();
        self.extract_imports_with_cursor(&mut cursor, content, language, &mut imports)?;

        Ok(imports)
    }

    /// Extract imports using TreeCursor for optimal performance
    fn extract_imports_with_cursor(
        &self,
        cursor: &mut tree_sitter::TreeCursor,
        content: &str,
        language: AstLanguage,
        imports: &mut Vec<AstImport>,
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
        matches!(
            kind,
            "import_statement"
                | "import_from_statement"
                | "use_declaration"
                | "import_declaration"
                | "import_spec"
                | "source_file"
                | "module"
                | "program"
                | "translation_unit"
                | "block"
                | "statement_block"
        ) || kind.contains("import")
            || kind.contains("use")
    }

    /// Check if a node is an import statement
    fn is_import_node(&self, kind: &str) -> bool {
        matches!(
            kind,
            "import_statement"
                | "import_from_statement"
                | "use_declaration"
                | "import_declaration"
                | "import_spec"
        )
    }

    /// Extract import from a specific node (no recursion needed)
    fn extract_import_from_node(
        &self,
        node: Node,
        content: &str,
        language: AstLanguage,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        match language {
            AstLanguage::Python => {
                self.extract_python_import_node(node, content, imports)?;
            }
            AstLanguage::JavaScript | AstLanguage::TypeScript => {
                self.extract_js_ts_import_node(node, content, imports)?;
            }
            AstLanguage::Go => {
                self.extract_go_import_node(node, content, imports)?;
            }
            AstLanguage::Rust => {
                self.extract_rust_import_node(node, content, imports)?;
            }
        }
        Ok(())
    }

    pub fn extract_signatures(
        &mut self,
        content: &str,
        file_path: &str,
    ) -> Result<Vec<AstSignature>> {
        let language = self.detect_language(file_path)?;
        let parser = self
            .parsers
            .get_mut(&language)
            .ok_or_else(|| ScribeError::parse(format!("No parser for language: {:?}", language)))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| ScribeError::parse("Failed to parse source code".to_string()))?;

        let signatures = match language {
            AstLanguage::Python => self.extract_python_signatures(content, &tree)?,
            AstLanguage::JavaScript => self.extract_javascript_signatures(content, &tree)?,
            AstLanguage::TypeScript => self.extract_typescript_signatures(content, &tree)?,
            AstLanguage::Go => self.extract_go_signatures(content, &tree)?,
            AstLanguage::Rust => self.extract_rust_signatures(content, &tree)?,
        };

        Ok(signatures)
    }

    /// Detect language from file path
    fn detect_language(&self, file_path: &str) -> Result<AstLanguage> {
        let extension = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        AstLanguage::from_extension(extension)
            .ok_or_else(|| ScribeError::parse(format!("Unsupported file extension: {}", extension)))
    }

    /// Parse Python code chunks using tree-sitter
    fn parse_python_chunks(&self, content: &str, tree: &Tree) -> Result<Vec<AstChunk>> {
        let mut chunks = Vec::new();
        let root_node = tree.root_node();

        // Query for Python constructs
        let query_str = r#"
            (import_statement) @import
            (import_from_statement) @import_from
            (function_definition) @function
            (class_definition) @class
            (assignment 
                left: (identifier) @const_name
                right: (_) @const_value
                (#match? @const_name "^[A-Z_][A-Z0-9_]*$")
            ) @constant
        "#;

        let query = Query::new(AstLanguage::Python.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid Python query: {}", e)))?;

        let mut cursor = QueryCursor::new();
        let captures = cursor.matches(&query, root_node, content.as_bytes());

        for match_ in captures {
            for capture in match_.captures {
                let node = capture.node;
                let chunk_type = &query.capture_names()[capture.index as usize];

                let chunk =
                    self.create_chunk_from_node(content, node, chunk_type, &AstLanguage::Python)?;
                chunks.push(chunk);
            }
        }

        // Sort by start position
        chunks.sort_by_key(|c| c.start_byte);
        Ok(chunks)
    }

    /// Parse JavaScript code chunks using tree-sitter
    fn parse_javascript_chunks(&self, content: &str, tree: &Tree) -> Result<Vec<AstChunk>> {
        let mut chunks = Vec::new();
        let root_node = tree.root_node();

        let query_str = r#"
            (import_statement) @import
            (export_statement) @export
            (function_declaration) @function
            (arrow_function) @arrow_function
            (class_declaration) @class
            (interface_declaration) @interface
            (type_alias_declaration) @type_alias
            (variable_declaration
                declarations: (variable_declarator
                    name: (identifier) @const_name
                    value: (_) @const_value
                ) @const_declarator
                (#match? @const_name "^[A-Z_][A-Z0-9_]*$")
            ) @constant
        "#;

        let query = Query::new(AstLanguage::JavaScript.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid JavaScript query: {}", e)))?;

        let mut cursor = QueryCursor::new();
        let captures = cursor.matches(&query, root_node, content.as_bytes());

        for match_ in captures {
            for capture in match_.captures {
                let node = capture.node;
                let chunk_type = &query.capture_names()[capture.index as usize];

                let chunk = self.create_chunk_from_node(
                    content,
                    node,
                    chunk_type,
                    &AstLanguage::JavaScript,
                )?;
                chunks.push(chunk);
            }
        }

        chunks.sort_by_key(|c| c.start_byte);
        Ok(chunks)
    }

    /// Parse TypeScript code chunks using tree-sitter
    fn parse_typescript_chunks(&self, content: &str, tree: &Tree) -> Result<Vec<AstChunk>> {
        let mut chunks = Vec::new();
        let root_node = tree.root_node();

        let query_str = r#"
            (import_statement) @import
            (export_statement) @export
            (function_declaration) @function
            (arrow_function) @arrow_function
            (class_declaration) @class
            (interface_declaration) @interface
            (type_alias_declaration) @type_alias
            (enum_declaration) @enum
            (module_declaration) @module
            (variable_declaration
                declarations: (variable_declarator
                    name: (identifier) @const_name
                    value: (_) @const_value
                ) @const_declarator
                (#match? @const_name "^[A-Z_][A-Z0-9_]*$")
            ) @constant
        "#;

        let query = Query::new(AstLanguage::TypeScript.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid TypeScript query: {}", e)))?;

        let mut cursor = QueryCursor::new();
        let captures = cursor.matches(&query, root_node, content.as_bytes());

        for match_ in captures {
            for capture in match_.captures {
                let node = capture.node;
                let chunk_type = &query.capture_names()[capture.index as usize];

                let chunk = self.create_chunk_from_node(
                    content,
                    node,
                    chunk_type,
                    &AstLanguage::TypeScript,
                )?;
                chunks.push(chunk);
            }
        }

        chunks.sort_by_key(|c| c.start_byte);
        Ok(chunks)
    }

    /// Parse Go code chunks using tree-sitter
    fn parse_go_chunks(&self, content: &str, tree: &Tree) -> Result<Vec<AstChunk>> {
        let mut chunks = Vec::new();
        let root_node = tree.root_node();

        let query_str = r#"
            (package_clause) @package
            (import_declaration) @import
            (function_declaration) @function
            (method_declaration) @method
            (type_declaration) @type
            (const_declaration) @const
            (var_declaration) @var
        "#;

        let query = Query::new(AstLanguage::Go.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid Go query: {}", e)))?;

        let mut cursor = QueryCursor::new();
        let captures = cursor.matches(&query, root_node, content.as_bytes());

        for match_ in captures {
            for capture in match_.captures {
                let node = capture.node;
                let chunk_type = &query.capture_names()[capture.index as usize];

                let chunk =
                    self.create_chunk_from_node(content, node, chunk_type, &AstLanguage::Go)?;
                chunks.push(chunk);
            }
        }

        chunks.sort_by_key(|c| c.start_byte);
        Ok(chunks)
    }

    /// Parse Rust code chunks using tree-sitter
    fn parse_rust_chunks(&self, content: &str, tree: &Tree) -> Result<Vec<AstChunk>> {
        let mut chunks = Vec::new();
        let root_node = tree.root_node();

        let query_str = r#"
            (use_declaration) @use
            (mod_item) @mod
            (struct_item) @struct
            (enum_item) @enum
            (trait_item) @trait
            (impl_item) @impl
            (function_item) @function
            (const_item) @const
            (static_item) @static
            (type_item) @type_alias
        "#;

        let query = Query::new(AstLanguage::Rust.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid Rust query: {}", e)))?;

        let mut cursor = QueryCursor::new();
        let captures = cursor.matches(&query, root_node, content.as_bytes());

        for match_ in captures {
            for capture in match_.captures {
                let node = capture.node;
                let chunk_type = &query.capture_names()[capture.index as usize];

                let chunk =
                    self.create_chunk_from_node(content, node, chunk_type, &AstLanguage::Rust)?;
                chunks.push(chunk);
            }
        }

        chunks.sort_by_key(|c| c.start_byte);
        Ok(chunks)
    }

    /// Create a chunk from a tree-sitter node
    fn create_chunk_from_node(
        &self,
        content: &str,
        node: Node,
        chunk_type: &str,
        language: &AstLanguage,
    ) -> Result<AstChunk> {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let start_position = node.start_position();
        let end_position = node.end_position();

        let chunk_content = &content[start_byte..end_byte];
        let estimated_tokens = chunk_content.split_whitespace().count();

        // Calculate importance score based on chunk type and language
        let importance_score = self.calculate_importance_score(chunk_type, language, node, content);

        // Extract name if available
        let name = self.extract_name_from_node(node, content);

        // Check if public/exported
        let is_public = self.is_node_public(node, content);

        // Check for documentation
        let has_documentation = self.has_documentation(node, content);

        // Extract dependencies (simplified for now)
        let dependencies = self.extract_dependencies(node, content);

        Ok(AstChunk {
            content: chunk_content.to_string(),
            chunk_type: chunk_type.to_string(),
            start_line: start_position.row + 1,
            end_line: end_position.row + 1,
            start_byte,
            end_byte,
            importance_score,
            estimated_tokens,
            dependencies,
            name,
            is_public,
            has_documentation,
        })
    }

    /// Calculate importance score based on AST analysis
    fn calculate_importance_score(
        &self,
        chunk_type: &str,
        language: &AstLanguage,
        node: Node,
        content: &str,
    ) -> f64 {
        let mut score: f64 = match chunk_type {
            "import" | "import_from" | "use" => 0.9, // Imports are crucial
            "package" => 0.95,                       // Package declarations are essential
            "class" | "struct_item" | "trait_item" => 0.85, // Type definitions
            "interface" | "type_alias" | "enum" => 0.8, // Type definitions
            "function" | "method" => 0.75,           // Functions
            "const" | "constant" | "static" => 0.6,  // Constants
            "export" => 0.7,                         // Exports
            "mod" | "module" => 0.65,                // Modules
            _ => 0.5,                                // Default
        };

        // Boost score for public/exported items
        if self.is_node_public(node, content) {
            score += 0.1;
        }

        // Boost score for documented items
        if self.has_documentation(node, content) {
            score += 0.05;
        }

        // Language-specific adjustments
        match language {
            AstLanguage::Rust => {
                // Rust impl blocks are very important
                if chunk_type == "impl" {
                    score = 0.85;
                }
            }
            AstLanguage::TypeScript => {
                // TypeScript interfaces are crucial
                if chunk_type == "interface" {
                    score = 0.9;
                }
            }
            _ => {}
        }

        score.min(1.0)
    }

    /// Extract name/identifier from a node
    fn extract_name_from_node(&self, node: Node, content: &str) -> Option<String> {
        // Look for name field in node
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "identifier" || child.kind() == "type_identifier" {
                    let name_bytes = &content.as_bytes()[child.start_byte()..child.end_byte()];
                    if let Ok(name) = std::str::from_utf8(name_bytes) {
                        return Some(name.to_string());
                    }
                }
            }
        }
        None
    }

    /// Check if a node represents a public/exported item
    fn is_node_public(&self, node: Node, content: &str) -> bool {
        // Check for pub keyword in Rust
        if let Some(parent) = node.parent() {
            for i in 0..parent.child_count() {
                if let Some(child) = parent.child(i) {
                    if child.kind() == "visibility_modifier" {
                        let vis_bytes = &content.as_bytes()[child.start_byte()..child.end_byte()];
                        if let Ok(vis) = std::str::from_utf8(vis_bytes) {
                            return vis.contains("pub");
                        }
                    }
                }
            }
        }

        // Check for export in JS/TS
        let node_text = &content[node.start_byte()..node.end_byte()];
        node_text.starts_with("export") || node_text.contains("export")
    }

    /// Check if a node has associated documentation
    fn has_documentation(&self, node: Node, content: &str) -> bool {
        // Look for comments before the node
        if let Some(prev_sibling) = node.prev_sibling() {
            if prev_sibling.kind() == "comment" {
                return true;
            }
        }

        // Look for docstrings in Python
        if node.kind() == "function_definition" || node.kind() == "class_definition" {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "expression_statement" {
                        if let Some(grandchild) = child.child(0) {
                            if grandchild.kind() == "string" {
                                let string_content =
                                    &content[grandchild.start_byte()..grandchild.end_byte()];
                                if string_content.starts_with("\"\"\"")
                                    || string_content.starts_with("'''")
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }

        false
    }

    /// Extract dependencies from a node (simplified implementation)
    fn extract_dependencies(&self, node: Node, content: &str) -> Vec<String> {
        let mut dependencies = Vec::new();

        // For import nodes, extract the imported modules
        if node.kind() == "import_statement"
            || node.kind() == "import_from_statement"
            || node.kind() == "use_declaration"
        {
            // This is a simplified implementation
            // In a full implementation, we'd parse the specific import syntax
            let import_text = &content[node.start_byte()..node.end_byte()];

            // Extract quoted strings as module names
            let mut in_quote = false;
            let mut quote_char = '"';
            let mut current_module = String::new();

            for ch in import_text.chars() {
                if ch == '"' || ch == '\'' {
                    if !in_quote {
                        in_quote = true;
                        quote_char = ch;
                    } else if ch == quote_char {
                        in_quote = false;
                        if !current_module.is_empty() {
                            dependencies.push(current_module.clone());
                            current_module.clear();
                        }
                    }
                } else if in_quote {
                    current_module.push(ch);
                }
            }
        }

        dependencies
    }

    /// Extract signatures for Python
    fn extract_python_signatures(&self, content: &str, tree: &Tree) -> Result<Vec<AstSignature>> {
        let mut signatures = Vec::new();
        let root_node = tree.root_node();

        let query_str = r#"
            (function_definition 
                name: (identifier) @func_name
                parameters: (parameters) @func_params
            ) @function
            (class_definition 
                name: (identifier) @class_name
            ) @class
            (import_statement) @import
            (import_from_statement) @import_from
        "#;

        let query = Query::new(AstLanguage::Python.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid Python signature query: {}", e)))?;

        let mut cursor = QueryCursor::new();
        let captures = cursor.matches(&query, root_node, content.as_bytes());

        for match_ in captures {
            let signature = self.extract_signature_from_match(content, &match_, &query)?;
            signatures.push(signature);
        }

        Ok(signatures)
    }

    /// Extract signatures for other languages (similar pattern)
    fn extract_javascript_signatures(
        &self,
        content: &str,
        tree: &Tree,
    ) -> Result<Vec<AstSignature>> {
        let query_str = r#"
            (function_declaration
                name: (identifier) @name
            ) @function

            (arrow_function) @function

            (class_declaration
                name: (identifier) @name
            ) @class

            (import_statement) @import
            (export_statement) @export
        "#;

        let query =
            Query::new(AstLanguage::JavaScript.tree_sitter_language(), query_str).map_err(|e| {
                ScribeError::parse(format!("Invalid JavaScript signature query: {}", e))
            })?;

        let root_node = tree.root_node();
        let mut cursor = tree_sitter::QueryCursor::new();
        let matches = cursor.matches(&query, root_node, content.as_bytes());

        let mut signatures = Vec::new();
        for match_ in matches {
            let signature = self.extract_signature_from_match(content, &match_, &query)?;
            signatures.push(signature);
        }

        Ok(signatures)
    }

    fn extract_typescript_signatures(
        &self,
        content: &str,
        tree: &Tree,
    ) -> Result<Vec<AstSignature>> {
        let query_str = r#"
            (function_declaration
                name: (identifier) @name
            ) @function

            (interface_declaration
                name: (type_identifier) @name
            ) @interface

            (type_alias_declaration
                name: (type_identifier) @name
            ) @type

            (class_declaration
                name: (identifier) @name
            ) @class

            (import_statement) @import
            (export_statement) @export
        "#;

        let query =
            Query::new(AstLanguage::TypeScript.tree_sitter_language(), query_str).map_err(|e| {
                ScribeError::parse(format!("Invalid TypeScript signature query: {}", e))
            })?;

        let root_node = tree.root_node();
        let mut cursor = tree_sitter::QueryCursor::new();
        let matches = cursor.matches(&query, root_node, content.as_bytes());

        let mut signatures = Vec::new();
        for match_ in matches {
            let signature = self.extract_signature_from_match(content, &match_, &query)?;
            signatures.push(signature);
        }

        Ok(signatures)
    }

    fn extract_go_signatures(&self, content: &str, tree: &Tree) -> Result<Vec<AstSignature>> {
        let query_str = r#"
            (function_declaration
                name: (identifier) @name
            ) @function

            (type_declaration
                (type_spec
                    name: (type_identifier) @name
                )
            ) @type

            (import_declaration) @import
            (package_clause) @package
        "#;

        let query = Query::new(AstLanguage::Go.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid Go signature query: {}", e)))?;

        let root_node = tree.root_node();
        let mut cursor = tree_sitter::QueryCursor::new();
        let matches = cursor.matches(&query, root_node, content.as_bytes());

        let mut signatures = Vec::new();
        for match_ in matches {
            let signature = self.extract_signature_from_match(content, &match_, &query)?;
            signatures.push(signature);
        }

        Ok(signatures)
    }

    fn extract_rust_signatures(&self, content: &str, tree: &Tree) -> Result<Vec<AstSignature>> {
        let query_str = r#"
            (function_item
                name: (identifier) @name
            ) @function

            (impl_item
                type: (type_identifier) @type_name
            ) @impl

            (struct_item
                name: (type_identifier) @name
            ) @struct

            (enum_item
                name: (type_identifier) @name
            ) @enum

            (trait_item
                name: (type_identifier) @name
            ) @trait

            (mod_item
                name: (identifier) @name
            ) @module

            (use_declaration) @use
        "#;

        let query = Query::new(AstLanguage::Rust.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid Rust signature query: {}", e)))?;

        let root_node = tree.root_node();
        let mut cursor = tree_sitter::QueryCursor::new();
        let matches = cursor.matches(&query, root_node, content.as_bytes());

        let mut signatures = Vec::new();
        for match_ in matches {
            let signature = self.extract_signature_from_match(content, &match_, &query)?;
            signatures.push(signature);
        }

        Ok(signatures)
    }

    /// Extract signature from a query match
    fn extract_signature_from_match(
        &self,
        content: &str,
        match_: &tree_sitter::QueryMatch,
        query: &Query,
    ) -> Result<AstSignature> {
        let mut signature_text = String::new();
        let mut signature_type = String::new();
        let mut name = String::new();
        let mut line = 0;

        for capture in match_.captures {
            let capture_name = &query.capture_names()[capture.index as usize];
            let node = capture.node;
            let node_text = &content[node.start_byte()..node.end_byte()];

            match capture_name.as_str() {
                "function" | "class" | "import" | "import_from" => {
                    signature_text = node_text.lines().next().unwrap_or("").to_string();
                    signature_type = capture_name.to_string();
                    line = node.start_position().row + 1;
                }
                "func_name" | "class_name" => {
                    name = node_text.to_string();
                }
                _ => {}
            }
        }

        Ok(AstSignature {
            signature: signature_text,
            signature_type,
            name,
            parameters: Vec::new(), // Simplified
            return_type: None,      // Simplified
            is_public: false,       // Simplified
            line,
        })
    }

    /// Extract Python import from a single node (optimized, no recursion)
    fn extract_python_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        // Look for import_statement and import_from_statement nodes
        if node.kind() == "import_statement" {
            // Handle import statements like "import os" or "import sys as system"
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "aliased_import" {
                        // Handle "import sys as system"
                        if let Some(name_node) = child.child_by_field_name("name") {
                            let module = self.node_text(name_node, content);
                            let alias = child
                                .child_by_field_name("alias")
                                .map(|alias_node| self.node_text(alias_node, content));
                            let line_number = name_node.start_position().row + 1;

                            imports.push(AstImport {
                                module,
                                alias,
                                items: vec![],
                                line_number,
                                is_relative: false,
                            });
                        }
                    } else if child.kind() == "dotted_as_name" {
                        // Handle dotted imports with alias like "import package.module as mod"
                        if let Some(name_node) = child.child_by_field_name("name") {
                            let module = self.node_text(name_node, content);
                            let alias = child
                                .child_by_field_name("alias")
                                .map(|alias_node| self.node_text(alias_node, content));
                            let line_number = name_node.start_position().row + 1;

                            imports.push(AstImport {
                                module,
                                alias,
                                items: vec![],
                                line_number,
                                is_relative: false,
                            });
                        }
                    } else if child.kind() == "dotted_name" || child.kind() == "identifier" {
                        // Handle simple "import os"
                        let module = self.node_text(child, content);
                        let line_number = child.start_position().row + 1;

                        imports.push(AstImport {
                            module,
                            alias: None,
                            items: vec![],
                            line_number,
                            is_relative: false,
                        });
                    }
                }
            }
        } else if node.kind() == "import_from_statement" {
            let mut module = String::new();
            let mut items = Vec::new();
            let mut is_relative = false;

            if let Some(module_node) = node.child_by_field_name("module_name") {
                module = self.node_text(module_node, content);
                is_relative = module.starts_with('.');
            }

            // Get imported items
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "import_list" {
                        for j in 0..child.child_count() {
                            if let Some(item) = child.child(j) {
                                if item.kind() == "dotted_name" || item.kind() == "identifier" {
                                    items.push(self.node_text(item, content));
                                }
                            }
                        }
                    }
                }
            }

            let line_number = node.start_position().row + 1;
            imports.push(AstImport {
                module,
                alias: None,
                items,
                line_number,
                is_relative,
            });
        }

        Ok(())
    }

    /// Extract JavaScript/TypeScript import from a single node (optimized, no recursion)
    fn extract_js_ts_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        if node.kind() == "import_statement" {
            let mut module = String::new();
            let items = Vec::new();

            // Find the source
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "string" {
                        module = self.node_text(child, content);
                        // Remove quotes
                        module = module.trim_matches('"').trim_matches('\'').to_string();
                        break;
                    }
                }
            }

            let line_number = node.start_position().row + 1;
            imports.push(AstImport {
                module,
                alias: None,
                items,
                line_number,
                is_relative: false,
            });
        }
        Ok(())
    }

    /// Extract Go import from a single node (optimized, no recursion)
    fn extract_go_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        if node.kind() == "import_spec" {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "interpreted_string_literal" {
                        let module = self.node_text(child, content);
                        let module = module.trim_matches('"').to_string();
                        let line_number = child.start_position().row + 1;

                        imports.push(AstImport {
                            module,
                            alias: None,
                            items: vec![],
                            line_number,
                            is_relative: false,
                        });
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
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        if node.kind() == "use_declaration" {
            if let Some(use_tree) = node.child_by_field_name("argument") {
                let module = self.node_text(use_tree, content);
                let line_number = node.start_position().row + 1;

                imports.push(AstImport {
                    module,
                    alias: None,
                    items: vec![],
                    line_number,
                    is_relative: false,
                });
            }
        }
        Ok(())
    }

    /// Helper to extract text from a node
    fn node_text(&self, node: Node, content: &str) -> String {
        content[node.start_byte()..node.end_byte()].to_string()
    }
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new().expect("Failed to create AstParser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_parser_creation() {
        let parser = AstParser::new();
        assert!(parser.is_ok());
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(AstLanguage::from_extension("py"), Some(AstLanguage::Python));
        assert_eq!(
            AstLanguage::from_extension("js"),
            Some(AstLanguage::JavaScript)
        );
        assert_eq!(
            AstLanguage::from_extension("ts"),
            Some(AstLanguage::TypeScript)
        );
        assert_eq!(AstLanguage::from_extension("go"), Some(AstLanguage::Go));
        assert_eq!(AstLanguage::from_extension("rs"), Some(AstLanguage::Rust));
        assert_eq!(AstLanguage::from_extension("unknown"), None);
    }

    #[test]
    fn test_python_parsing() {
        let mut parser = AstParser::new().unwrap();
        let content = r#"
import os
import sys

def hello_world():
    """A simple function."""
    print("Hello, world!")

class Calculator:
    """A simple calculator."""
    
    def add(self, a, b):
        return a + b
"#;

        let chunks = parser.parse_chunks(content, "test.py").unwrap();
        assert!(!chunks.is_empty());

        // Should find imports, function, and class
        let chunk_types: Vec<&str> = chunks.iter().map(|c| c.chunk_type.as_str()).collect();
        assert!(chunk_types.contains(&"import"));
        assert!(chunk_types.contains(&"function"));
        assert!(chunk_types.contains(&"class"));
    }

    #[test]
    fn test_rust_parsing() {
        let mut parser = AstParser::new().unwrap();
        let content = r#"
use std::collections::HashMap;

pub struct DataProcessor {
    data: HashMap<String, i32>,
}

impl DataProcessor {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}
"#;

        let chunks = parser.parse_chunks(content, "test.rs").unwrap();
        assert!(!chunks.is_empty());

        let chunk_types: Vec<&str> = chunks.iter().map(|c| c.chunk_type.as_str()).collect();
        assert!(chunk_types.contains(&"use"));
        assert!(chunk_types.contains(&"struct"));
        assert!(chunk_types.contains(&"impl"));
    }

    #[test]
    fn test_signature_extraction() {
        let mut parser = AstParser::new().unwrap();
        let content = r#"
def calculate(a: int, b: int) -> int:
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
"#;

        let signatures = parser.extract_signatures(content, "test.py").unwrap();
        assert!(!signatures.is_empty());
    }
}

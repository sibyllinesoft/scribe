//! Tree-sitter based AST parsing for accurate code analysis
//!
//! This module replaces regex-based parsing with proper syntax-aware analysis
//! using tree-sitter parsers for multiple programming languages.

use scribe_core::tokenization::{utils as token_utils, TokenCounter};
use scribe_core::{Result, ScribeError};
use std::collections::HashMap;
use tree_sitter::{Node, Parser, Query, QueryCursor, Tree};

// Re-export types from submodules for API compatibility
pub use super::entity::{EntityLocation, EntityQuery, EntityType};
pub use super::queries::{chunk_query_for_language, signature_query_for_language};
pub use super::types::{AstChunk, AstImport, AstLanguage, AstSignature};

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

        self.parse_language_chunks(content, &tree, language)
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

        self.extract_language_signatures(content, &tree, language)
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

    /// Parse code chunks for a given language using tree-sitter
    fn parse_language_chunks(
        &self,
        content: &str,
        tree: &Tree,
        language: AstLanguage,
    ) -> Result<Vec<AstChunk>> {
        let query_str = chunk_query_for_language(language);
        let query = Query::new(language.tree_sitter_language(), query_str)
            .map_err(|e| ScribeError::parse(format!("Invalid {:?} query: {}", language, e)))?;

        let root_node = tree.root_node();
        let mut cursor = QueryCursor::new();
        let captures = cursor.matches(&query, root_node, content.as_bytes());

        let mut chunks = Vec::new();
        for match_ in captures {
            for capture in match_.captures {
                let node = capture.node;
                let chunk_type = &query.capture_names()[capture.index as usize];
                let chunk = self.create_chunk_from_node(content, node, chunk_type, &language)?;
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

        let raw_chunk = &content[start_byte..end_byte];
        let doc_text = self.extract_documentation_for_node(*language, node, content);
        let combined_chunk = if let Some(ref doc) = doc_text {
            format!("{doc}\n{raw_chunk}")
        } else {
            raw_chunk.to_string()
        };

        let estimated_tokens = TokenCounter::global()
            .count_tokens(&combined_chunk)
            .unwrap_or_else(|_| token_utils::estimate_tokens_legacy(&combined_chunk));

        // Calculate importance score based on chunk type and language
        let importance_score = self.calculate_importance_score(chunk_type, language, node, content);

        // Extract name if available
        let name = self.extract_name_from_node(node, content);

        // Check if public/exported
        let is_public = self.is_node_public(node, content);

        // Check for documentation
        let has_documentation = doc_text.is_some() || self.has_documentation(node, content);

        // Extract dependencies (simplified for now)
        let dependencies = self.extract_dependencies(node, content);

        Ok(AstChunk {
            content: combined_chunk,
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

    /// Get base importance score for a chunk type
    fn base_importance_score(chunk_type: &str) -> f64 {
        match chunk_type {
            "import" | "import_from" | "use" => 0.9,
            "package" => 0.95,
            "class" | "struct_item" | "trait_item" => 0.85,
            "interface" | "type_alias" | "enum" => 0.8,
            "function" | "method" => 0.75,
            "export" => 0.7,
            "mod" | "module" => 0.65,
            "const" | "constant" | "static" => 0.6,
            _ => 0.5,
        }
    }

    /// Apply language-specific importance adjustments
    fn language_importance_adjustment(chunk_type: &str, language: &AstLanguage) -> Option<f64> {
        match language {
            AstLanguage::Rust if chunk_type == "impl" => Some(0.85),
            AstLanguage::TypeScript if chunk_type == "interface" => Some(0.9),
            _ => None,
        }
    }

    /// Calculate importance score based on AST analysis
    fn calculate_importance_score(
        &self,
        chunk_type: &str,
        language: &AstLanguage,
        node: Node,
        content: &str,
    ) -> f64 {
        let mut score = Self::base_importance_score(chunk_type);

        // Apply language-specific override if applicable
        if let Some(lang_score) = Self::language_importance_adjustment(chunk_type, language) {
            score = lang_score;
        }

        // Boost score for public/exported items
        if self.is_node_public(node, content) {
            score += 0.1;
        }

        // Boost score for documented items
        if self.has_documentation(node, content) {
            score += 0.05;
        }

        score.min(1.0)
    }

    /// Extract name/identifier from a node
    fn extract_name_from_node(&self, node: Node, content: &str) -> Option<String> {
        // Look for name field in node (direct children first)
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                // Include field_identifier for Go methods
                if child.kind() == "identifier"
                    || child.kind() == "type_identifier"
                    || child.kind() == "field_identifier"
                {
                    let name_bytes = &content.as_bytes()[child.start_byte()..child.end_byte()];
                    if let Ok(name) = std::str::from_utf8(name_bytes) {
                        return Some(name.to_string());
                    }
                }
                // For lexical/variable declarations, look inside variable_declarator
                if child.kind() == "variable_declarator" {
                    for j in 0..child.child_count() {
                        if let Some(grandchild) = child.child(j) {
                            if grandchild.kind() == "identifier" {
                                let name_bytes = &content.as_bytes()
                                    [grandchild.start_byte()..grandchild.end_byte()];
                                if let Ok(name) = std::str::from_utf8(name_bytes) {
                                    return Some(name.to_string());
                                }
                            }
                        }
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

    /// Extract documentation (doc comment or docstring) associated with a node.
    fn extract_documentation_for_node(
        &self,
        language: AstLanguage,
        node: Node,
        content: &str,
    ) -> Option<String> {
        if language == AstLanguage::Python {
            if let Some(doc) = self.extract_python_docstring(node, content) {
                return Some(doc);
            }
        }

        self.collect_leading_comment_block(content, node.start_byte())
    }

    fn collect_leading_comment_block(&self, content: &str, start_byte: usize) -> Option<String> {
        let prefix = &content[..start_byte];
        let mut doc_lines = Vec::new();
        for line in prefix.lines().rev() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                if doc_lines.is_empty() {
                    continue;
                } else {
                    break;
                }
            }

            let stripped = if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                Some(
                    trimmed
                        .trim_start_matches('/')
                        .trim_start_matches('!')
                        .trim(),
                )
            } else if trimmed.starts_with("//") {
                Some(trimmed.trim_start_matches('/').trim())
            } else if trimmed.starts_with('#') {
                Some(trimmed.trim_start_matches('#').trim())
            } else if trimmed.starts_with("/*") || trimmed.starts_with("*") {
                Some(
                    trimmed
                        .trim_start_matches("/*")
                        .trim_start_matches('*')
                        .trim_end_matches("*/")
                        .trim(),
                )
            } else {
                None
            };

            if let Some(clean) = stripped {
                doc_lines.push(clean.to_string());
            } else if doc_lines.is_empty() {
                // Keep scanning upward until we hit a real comment block
                continue;
            } else {
                break;
            }
        }

        if doc_lines.is_empty() {
            None
        } else {
            doc_lines.reverse();
            Some(doc_lines.join("\n"))
        }
    }

    fn extract_python_docstring(&self, node: Node, content: &str) -> Option<String> {
        if node.kind() == "function_definition" || node.kind() == "class_definition" {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "block" || child.kind() == "suite" || child.kind() == "colon"
                    {
                        continue;
                    }
                    if child.kind() == "expression_statement" {
                        if let Some(grandchild) = child.child(0) {
                            if grandchild.kind() == "string" {
                                let raw = &content[grandchild.start_byte()..grandchild.end_byte()];
                                return Some(Self::strip_string_quotes(raw));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn strip_string_quotes(raw: &str) -> String {
        let mut trimmed = raw.trim().to_string();
        let quotes = [r#""""#, "'''", "\"", "'"];
        for q in quotes {
            if trimmed.starts_with(q) && trimmed.ends_with(q) && trimmed.len() >= q.len() * 2 {
                trimmed = trimmed[q.len()..trimmed.len() - q.len()].to_string();
                break;
            }
        }
        trimmed
    }

    /// Check if a node has a preceding comment
    fn has_preceding_comment(&self, node: Node) -> bool {
        node.prev_sibling()
            .map(|sibling| sibling.kind() == "comment")
            .unwrap_or(false)
    }

    /// Check if an expression statement contains a docstring
    fn is_docstring_expression(&self, child: Node, content: &str) -> bool {
        if child.kind() != "expression_statement" {
            return false;
        }
        let Some(grandchild) = child.child(0) else {
            return false;
        };
        if grandchild.kind() != "string" {
            return false;
        }
        let string_content = &content[grandchild.start_byte()..grandchild.end_byte()];
        string_content.starts_with("\"\"\"") || string_content.starts_with("'''")
    }

    /// Check if a Python node has a docstring
    fn has_python_docstring(&self, node: Node, content: &str) -> bool {
        if node.kind() != "function_definition" && node.kind() != "class_definition" {
            return false;
        }
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if self.is_docstring_expression(child, content) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if a node has associated documentation
    fn has_documentation(&self, node: Node, content: &str) -> bool {
        self.has_preceding_comment(node) || self.has_python_docstring(node, content)
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

    /// Extract signatures for a given language using tree-sitter
    fn extract_language_signatures(
        &self,
        content: &str,
        tree: &Tree,
        language: AstLanguage,
    ) -> Result<Vec<AstSignature>> {
        let query_str = signature_query_for_language(language);
        let query = Query::new(language.tree_sitter_language(), query_str).map_err(|e| {
            ScribeError::parse(format!("Invalid {:?} signature query: {}", language, e))
        })?;

        let root_node = tree.root_node();
        let mut cursor = QueryCursor::new();
        let matches = cursor.matches(&query, root_node, content.as_bytes());

        let mut signatures = Vec::new();
        for match_ in matches {
            let signature =
                self.extract_signature_from_match(content, &match_, &query, language)?;
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
        language: AstLanguage,
    ) -> Result<AstSignature> {
        let mut signature_text = String::new();
        let mut signature_type = String::new();
        let mut name = String::new();
        let mut line = 0;
        let mut primary_node: Option<Node> = None;

        for capture in match_.captures {
            let capture_name = &query.capture_names()[capture.index as usize];
            let node = capture.node;
            let node_text = &content[node.start_byte()..node.end_byte()];

            match capture_name.as_str() {
                // Primary node captures - extract signature
                "function" | "class" | "import" | "import_from" | "interface" | "type_alias"
                | "enum" | "method" | "field" | "struct" | "trait" | "impl" | "module" | "use"
                | "type" | "export" | "package" | "arrow_const" | "arrow_var" => {
                    signature_text = Self::extract_signature_lines(node_text);
                    signature_type = capture_name.to_string();
                    line = node.start_position().row + 1;
                    primary_node = Some(node);
                }
                // Name captures
                "func_name" | "class_name" | "interface_name" | "type_name" | "enum_name"
                | "method_name" | "field_name" | "name" | "arrow_name" => {
                    name = node_text.to_string();
                }
                _ => {}
            }
        }

        let documentation =
            primary_node.and_then(|n| self.extract_documentation_for_node(language, n, content));

        Ok(AstSignature {
            signature: signature_text,
            signature_type,
            name,
            parameters: Vec::new(), // Simplified
            return_type: None,      // Simplified
            is_public: false,       // Simplified
            line,
            documentation,
        })
    }

    /// Extract signature lines from node text.
    /// For functions/methods, capture the full signature up to the opening brace.
    /// For types/interfaces, capture the declaration line.
    fn extract_signature_lines(node_text: &str) -> String {
        // Find the opening brace to determine where the signature ends
        if let Some(brace_pos) = node_text.find('{') {
            // Get everything before the brace, trimmed
            let sig = node_text[..brace_pos].trim();
            // If signature spans multiple lines, normalize whitespace
            if sig.contains('\n') {
                sig.split_whitespace().collect::<Vec<_>>().join(" ")
            } else {
                sig.to_string()
            }
        } else {
            // No brace - just take first line (e.g., type alias, import)
            node_text.lines().next().unwrap_or("").to_string()
        }
    }

    // Delegate to import_extractors module
    fn extract_python_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        super::import_extractors::extract_python_import_node(node, content, imports)
    }

    fn extract_js_ts_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        super::import_extractors::extract_js_ts_import_node(node, content, imports)
    }

    fn extract_go_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        super::import_extractors::extract_go_import_node(node, content, imports)
    }

    fn extract_rust_import_node(
        &self,
        node: Node,
        content: &str,
        imports: &mut Vec<AstImport>,
    ) -> Result<()> {
        super::import_extractors::extract_rust_import_node(node, content, imports)
    }

    fn node_text(&self, node: Node, content: &str) -> String {
        super::import_extractors::node_text(node, content)
    }

    /// Search for entities (functions, classes, etc.) by name within parsed content
    ///
    /// Returns locations of all matching entities across the provided content.
    pub fn find_entities(
        &mut self,
        content: &str,
        file_path: &str,
        query: &EntityQuery,
    ) -> Result<Vec<EntityLocation>> {
        let chunks = self.parse_chunks(content, file_path)?;
        let mut locations = Vec::new();

        for chunk in chunks {
            if self.matches_query(&chunk, query) {
                locations.push(EntityLocation {
                    file_path: file_path.to_string(),
                    entity_type: chunk.chunk_type.clone(),
                    entity_name: chunk.name.clone().unwrap_or_default(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    is_public: chunk.is_public,
                    content: chunk.content.clone(),
                });
            }
        }

        Ok(locations)
    }

    /// Check if a chunk matches the entity query
    fn matches_query(&self, chunk: &AstChunk, query: &EntityQuery) -> bool {
        // Match by entity type if specified
        if let Some(ref entity_type) = query.entity_type {
            if !self.chunk_type_matches(entity_type, &chunk.chunk_type) {
                return false;
            }
        }

        // Match by name if specified
        if let Some(ref name_pattern) = query.name_pattern {
            let chunk_name = chunk.name.as_deref().unwrap_or("");
            if query.exact_match {
                // Case-insensitive exact match (whole name must match)
                if chunk_name.to_lowercase() != name_pattern.to_lowercase() {
                    return false;
                }
            } else {
                // Case-insensitive substring match
                if !chunk_name
                    .to_lowercase()
                    .contains(&name_pattern.to_lowercase())
                {
                    return false;
                }
            }
        }

        // Match by visibility if specified
        if let Some(public_only) = query.public_only {
            if public_only && !chunk.is_public {
                return false;
            }
        }

        true
    }

    /// Check if chunk type matches the requested entity type
    fn chunk_type_matches(&self, requested: &EntityType, chunk_type: &str) -> bool {
        match requested {
            EntityType::Function => matches!(chunk_type, "function" | "method"),
            EntityType::Class => matches!(chunk_type, "class" | "struct_item" | "trait_item"),
            EntityType::Module => matches!(chunk_type, "mod" | "module" | "package"),
            EntityType::Interface => matches!(chunk_type, "interface" | "trait_item"),
            EntityType::Constant => matches!(chunk_type, "const" | "constant" | "static"),
            EntityType::Any => true,
        }
    }
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new().expect("Failed to create AstParser")
    }
}

#[cfg(test)]
#[path = "ast_parser_tests.rs"]
mod tests;

//! Optimized AST-based import extraction for analysis module
//!
//! This module provides a high-performance AST parser specifically for extracting
//! import statements from source code using TreeCursor for efficient traversal
//! and parser reuse for better performance.

use once_cell::sync::Lazy;
use rayon::prelude::*;
use scribe_core::Result;
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
    Elixir,
}

impl ImportLanguage {
    /// Get the tree-sitter language for this language when available.
    pub fn tree_sitter_language(&self) -> Option<Language> {
        match self {
            ImportLanguage::Python => Some(tree_sitter_python::language()),
            ImportLanguage::JavaScript => Some(tree_sitter_javascript::language()),
            ImportLanguage::TypeScript => Some(tree_sitter_typescript::language_typescript()),
            ImportLanguage::Go => Some(tree_sitter_go::language()),
            ImportLanguage::Rust => Some(tree_sitter_rust::language()),
            ImportLanguage::Elixir => None,
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
            "ex" | "exs" => Some(ImportLanguage::Elixir),
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
                let ts_language = language.tree_sitter_language().ok_or_else(|| {
                    scribe_core::ScribeError::parse(
                        "No tree-sitter language available for import parser language",
                    )
                })?;
                parser.set_language(ts_language).map_err(|e| {
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
        let ts_language = language.tree_sitter_language().ok_or_else(|| {
            scribe_core::ScribeError::parse(
                "No tree-sitter language available for import parser language",
            )
        })?;
        parser.set_language(ts_language).map_err(|e| {
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

    /// Extract imports from the given content using optimized traversal
    ///
    /// For TypeScript and JavaScript, uses SWC for better accuracy with:
    /// - Type-only imports
    /// - Re-exports (`export * from`, `export { x } from`)
    /// - TSX/JSX syntax
    ///
    /// For other languages, uses tree-sitter.
    pub fn extract_imports(
        &self,
        content: &str,
        language: ImportLanguage,
    ) -> Result<Vec<SimpleImport>> {
        match language {
            // Use SWC for TypeScript/JavaScript (faster, handles edge cases better)
            ImportLanguage::TypeScript | ImportLanguage::JavaScript => {
                let is_typescript = matches!(language, ImportLanguage::TypeScript);
                Ok(crate::swc_import_extractor::extract_imports(
                    content,
                    is_typescript,
                ))
            }
            // Elixir currently uses a regex-based fallback (no tree-sitter dependency)
            ImportLanguage::Elixir => Ok(self.extract_elixir_imports_regex(content)),
            // Use tree-sitter for other languages
            _ => self.extract_imports_treesitter(content, language),
        }
    }

    /// Extract imports using tree-sitter (for Python, Go, Rust)
    fn extract_imports_treesitter(
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
        IMPORT_NODE_TYPES.contains(&kind)
            || kind.contains("import")
            || kind.contains("use")
            || kind == "program"
            || kind == "translation_unit"
            || kind == "block"
            || kind == "statement_block"
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
            ImportLanguage::Elixir => {
                // Elixir is handled by regex fallback in extract_imports
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

    /// Extract Elixir imports using a lightweight regex-free line parser.
    fn extract_elixir_imports_regex(&self, content: &str) -> Vec<SimpleImport> {
        let mut imports = Vec::new();

        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            let without_comments = trimmed.split('#').next().unwrap_or("").trim();
            if without_comments.is_empty() {
                continue;
            }

            for keyword in ["alias ", "import ", "require ", "use "] {
                if let Some(statement) = without_comments.strip_prefix(keyword) {
                    self.extract_elixir_statement(statement, idx + 1, &mut imports);
                    break;
                }
            }
        }

        imports
    }

    fn extract_elixir_statement(
        &self,
        statement: &str,
        line_number: usize,
        imports: &mut Vec<SimpleImport>,
    ) {
        if let Some((base, remainder)) = statement.split_once('{') {
            let base = Self::normalize_elixir_module(base.trim_end_matches('.'));
            if let Some(end) = remainder.find('}') {
                let grouped = &remainder[..end];
                for module in grouped.split(',') {
                    if let Some(module) = Self::normalize_elixir_module(module) {
                        let module = if let Some(ref base) = base {
                            format!("{}.{}", base, module)
                        } else {
                            module
                        };
                        imports.push(SimpleImport {
                            module,
                            line_number,
                        });
                    }
                }
            }
            return;
        }

        if let Some(module) = Self::normalize_elixir_module(statement) {
            imports.push(SimpleImport {
                module,
                line_number,
            });
        }
    }

    fn normalize_elixir_module(raw: &str) -> Option<String> {
        let mut module = raw.trim();

        if let Some((before_options, _)) = module.split_once(',') {
            module = before_options.trim();
        }

        if module.ends_with(" do") {
            module = module.trim_end_matches(" do").trim_end();
        }

        module = module.trim_matches(|c: char| matches!(c, '"' | '\'' | '(' | ')'));

        if let Some(stripped) = module.strip_prefix("Elixir.") {
            module = stripped;
        }

        let cleaned: String = module
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
            .collect();
        let cleaned = cleaned.trim_end_matches('.').to_string();

        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = SimpleAstParser::new();
        assert!(parser.is_ok());
    }

    #[test]
    fn test_parser_default() {
        let parser = SimpleAstParser::default();
        // Just verify it doesn't panic
        let _ = parser;
    }

    #[test]
    fn test_python_imports() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
"#;

        let imports = parser
            .extract_imports(code, ImportLanguage::Python)
            .unwrap();

        assert!(!imports.is_empty());
        assert!(imports.iter().any(|i| i.module.contains("os")));
        assert!(imports.iter().any(|i| i.module.contains("sys")));
        assert!(imports.iter().any(|i| i.module.contains("pathlib")));
    }

    #[test]
    fn test_javascript_imports() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
import React from 'react';
import { useState, useEffect } from 'react';
const fs = require('fs');
const path = require('path');
"#;

        let imports = parser
            .extract_imports(code, ImportLanguage::JavaScript)
            .unwrap();

        assert!(!imports.is_empty());
        assert!(imports.iter().any(|i| i.module.contains("react")));
    }

    #[test]
    fn test_typescript_imports() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
import { Component } from '@angular/core';
import type { Config } from './config';
import * as utils from './utils';
"#;

        let imports = parser
            .extract_imports(code, ImportLanguage::TypeScript)
            .unwrap();

        assert!(!imports.is_empty());
    }

    #[test]
    fn test_rust_imports() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
use std::collections::HashMap;
use std::path::PathBuf;
use crate::module::SubModule;
use super::parent_module;
"#;

        let imports = parser.extract_imports(code, ImportLanguage::Rust).unwrap();

        assert!(!imports.is_empty());
        assert!(imports.iter().any(|i| i.module.contains("std")));
    }

    #[test]
    fn test_go_imports() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
package main

import (
    "fmt"
    "os"
    "path/filepath"
)
"#;

        let imports = parser.extract_imports(code, ImportLanguage::Go).unwrap();

        assert!(!imports.is_empty());
        assert!(imports.iter().any(|i| i.module.contains("fmt")));
    }

    #[test]
    fn test_elixir_imports() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
alias MyApp.Repo
alias MyApp.{Accounts.User, Accounts.Team}
import Plug.Conn
require Logger
use MyAppWeb, :controller
"#;

        let imports = parser.extract_imports(code, ImportLanguage::Elixir).unwrap();

        assert!(!imports.is_empty());
        assert!(imports.iter().any(|i| i.module == "MyApp.Repo"));
        assert!(imports.iter().any(|i| i.module == "MyApp.Accounts.User"));
        assert!(imports.iter().any(|i| i.module == "MyApp.Accounts.Team"));
        assert!(imports.iter().any(|i| i.module == "Plug.Conn"));
        assert!(imports.iter().any(|i| i.module == "Logger"));
        assert!(imports.iter().any(|i| i.module == "MyAppWeb"));
    }

    #[test]
    fn test_import_language_from_extension_elixir() {
        assert_eq!(ImportLanguage::from_extension("ex"), Some(ImportLanguage::Elixir));
        assert_eq!(
            ImportLanguage::from_extension("exs"),
            Some(ImportLanguage::Elixir)
        );
    }

    #[test]
    fn test_elixir_multiline_grouped_alias_does_not_emit_partial_module() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
alias MyApp.{
  Repo,
  Accounts.User
}
"#;

        let imports = parser.extract_imports(code, ImportLanguage::Elixir).unwrap();
        assert!(!imports.iter().any(|i| i.module == "MyApp"));
        assert!(!imports.iter().any(|i| i.module == "MyApp."));
    }

    #[test]
    fn test_empty_code() {
        let parser = SimpleAstParser::new().unwrap();

        let imports = parser.extract_imports("", ImportLanguage::Python).unwrap();
        assert!(imports.is_empty());
    }

    #[test]
    fn test_no_imports() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
"#;

        let imports = parser
            .extract_imports(code, ImportLanguage::Python)
            .unwrap();
        assert!(imports.is_empty());
    }

    #[test]
    fn test_simple_import_struct() {
        let import = SimpleImport {
            module: "std::collections".to_string(),
            line_number: 5,
        };

        assert_eq!(import.module, "std::collections");
        assert_eq!(import.line_number, 5);
    }

    #[test]
    fn test_simple_import_clone() {
        let import = SimpleImport {
            module: "test::module".to_string(),
            line_number: 10,
        };

        let cloned = import.clone();
        assert_eq!(import.module, cloned.module);
        assert_eq!(import.line_number, cloned.line_number);
    }

    #[test]
    fn test_simple_import_debug() {
        let import = SimpleImport {
            module: "my::module".to_string(),
            line_number: 1,
        };

        let debug_str = format!("{:?}", import);
        assert!(debug_str.contains("SimpleImport"));
        assert!(debug_str.contains("my::module"));
    }

    #[test]
    fn test_import_language_variants() {
        // Test that all variants exist and can be created
        let _python = ImportLanguage::Python;
        let _javascript = ImportLanguage::JavaScript;
        let _typescript = ImportLanguage::TypeScript;
        let _rust = ImportLanguage::Rust;
        let _go = ImportLanguage::Go;
        let _elixir = ImportLanguage::Elixir;
    }

    #[test]
    fn test_import_language_copy() {
        let lang = ImportLanguage::Python;
        let copied = lang; // Copy trait
        assert_eq!(lang, copied);
    }

    #[test]
    fn test_import_language_clone() {
        let lang = ImportLanguage::Rust;
        let cloned = lang.clone();
        assert_eq!(lang, cloned);
    }

    #[test]
    fn test_import_language_debug() {
        let lang = ImportLanguage::TypeScript;
        let debug_str = format!("{:?}", lang);
        assert!(debug_str.contains("TypeScript"));
    }

    #[test]
    fn test_parallel_extraction() {
        let parser = SimpleAstParser::new().unwrap();

        let files = vec![
            (
                "file1.py".to_string(),
                "import os\nimport sys".to_string(),
                ImportLanguage::Python,
            ),
            (
                "file2.py".to_string(),
                "from pathlib import Path".to_string(),
                ImportLanguage::Python,
            ),
            (
                "file3.rs".to_string(),
                "use std::collections::HashMap;".to_string(),
                ImportLanguage::Rust,
            ),
        ];

        let results = parser.extract_imports_parallel(&files).unwrap();

        assert_eq!(results.len(), 3);
        for (path, imports) in &results {
            if path.ends_with(".py") {
                assert!(!imports.is_empty());
            }
        }
    }

    #[test]
    fn test_batch_extraction() {
        let parser = SimpleAstParser::new().unwrap();

        let contents: Vec<&str> = vec![
            "import os",
            "import sys\nimport json",
            "from collections import Counter",
        ];

        let results = parser
            .extract_imports_batch(&contents, ImportLanguage::Python)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
        assert!(!results[2].is_empty());
    }

    #[test]
    fn test_line_numbers_correct() {
        let parser = SimpleAstParser::new().unwrap();
        let code = r#"# Comment
import os
import sys
"#;

        let imports = parser
            .extract_imports(code, ImportLanguage::Python)
            .unwrap();

        // Line numbers should be 1-indexed
        assert!(imports.iter().any(|i| i.line_number >= 2));
    }
}

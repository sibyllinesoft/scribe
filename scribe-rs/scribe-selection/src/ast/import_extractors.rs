//! Language-specific import extraction methods for the AST parser.

use super::types::AstImport;
use scribe_core::Result;
use tree_sitter::Node;

/// Extract import items from an import_list node
pub fn extract_import_items(list_node: Node, content: &str) -> Vec<String> {
    let mut items = Vec::new();
    for j in 0..list_node.child_count() {
        if let Some(item) = list_node.child(j) {
            if item.kind() == "dotted_name" || item.kind() == "identifier" {
                items.push(node_text(item, content));
            }
        }
    }
    items
}

/// Create an AstImport from a node with name field
pub fn create_import_from_named_node(child: Node, content: &str) -> Option<AstImport> {
    let name_node = child.child_by_field_name("name")?;
    let module = node_text(name_node, content);
    let alias = child
        .child_by_field_name("alias")
        .map(|alias_node| node_text(alias_node, content));
    let line_number = name_node.start_position().row + 1;

    Some(AstImport {
        module,
        alias,
        items: vec![],
        line_number,
        is_relative: false,
    })
}

/// Extract Python import from a single node
pub fn extract_python_import_node(
    node: Node,
    content: &str,
    imports: &mut Vec<AstImport>,
) -> Result<()> {
    match node.kind() {
        "import_statement" => {
            extract_python_simple_import(node, content, imports);
        }
        "import_from_statement" => {
            extract_python_from_import(node, content, imports);
        }
        _ => {}
    }
    Ok(())
}

/// Extract simple Python import statement
fn extract_python_simple_import(node: Node, content: &str, imports: &mut Vec<AstImport>) {
    for i in 0..node.child_count() {
        let Some(child) = node.child(i) else { continue };

        match child.kind() {
            "aliased_import" | "dotted_as_name" => {
                if let Some(import) = create_import_from_named_node(child, content) {
                    imports.push(import);
                }
            }
            "dotted_name" | "identifier" => {
                let module = node_text(child, content);
                let line_number = child.start_position().row + 1;
                imports.push(AstImport {
                    module,
                    alias: None,
                    items: vec![],
                    line_number,
                    is_relative: false,
                });
            }
            _ => {}
        }
    }
}

/// Extract Python from-import statement
fn extract_python_from_import(node: Node, content: &str, imports: &mut Vec<AstImport>) {
    let mut module = String::new();
    let mut is_relative = false;

    if let Some(module_node) = node.child_by_field_name("module_name") {
        module = node_text(module_node, content);
        is_relative = module.starts_with('.');
    }

    let mut items = Vec::new();
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if child.kind() == "import_list" {
                items = extract_import_items(child, content);
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
        is_relative,
    });
}

/// Extract JavaScript/TypeScript import from a single node
pub fn extract_js_ts_import_node(
    node: Node,
    content: &str,
    imports: &mut Vec<AstImport>,
) -> Result<()> {
    if node.kind() == "import_statement" {
        let mut module = String::new();
        let items = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "string" {
                    module = node_text(child, content);
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

/// Extract Go import from a single node
pub fn extract_go_import_node(
    node: Node,
    content: &str,
    imports: &mut Vec<AstImport>,
) -> Result<()> {
    if node.kind() == "import_spec" {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "interpreted_string_literal" {
                    let module = node_text(child, content);
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

/// Extract Rust import from a single node
pub fn extract_rust_import_node(
    node: Node,
    content: &str,
    imports: &mut Vec<AstImport>,
) -> Result<()> {
    if node.kind() == "use_declaration" {
        if let Some(use_tree) = node.child_by_field_name("argument") {
            let module = node_text(use_tree, content);
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
pub fn node_text(node: Node, content: &str) -> String {
    content[node.start_byte()..node.end_byte()].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    fn parse_python(content: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_python::language()).unwrap();
        parser.parse(content, None).unwrap()
    }

    fn parse_javascript(content: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(tree_sitter_javascript::language())
            .unwrap();
        parser.parse(content, None).unwrap()
    }

    fn parse_go(content: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_go::language()).unwrap();
        parser.parse(content, None).unwrap()
    }

    fn parse_rust(content: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_rust::language()).unwrap();
        parser.parse(content, None).unwrap()
    }

    #[test]
    fn test_node_text() {
        let content = "hello world";
        let tree = parse_python("hello");
        let root = tree.root_node();
        // The root might have different structure, just test the function works
        assert!(!node_text(root, "hello").is_empty());
    }

    #[test]
    fn test_python_simple_import() {
        let content = "import os";
        let tree = parse_python(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_python_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "os");
        assert!(!imports[0].is_relative);
    }

    #[test]
    fn test_python_import_multiple() {
        let content = "import os, sys";
        let tree = parse_python(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_python_import_node(child, content, &mut imports);
            }
        }

        assert!(imports.len() >= 1);
    }

    #[test]
    fn test_python_from_import() {
        let content = "from collections import OrderedDict";
        let tree = parse_python(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_python_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "collections");
        assert!(!imports[0].is_relative);
    }

    #[test]
    fn test_python_relative_import() {
        let content = "from . import utils";
        let tree = parse_python(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_python_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        // Relative import detection
        assert!(
            imports[0].is_relative
                || imports[0].module.starts_with('.')
                || imports[0].module.is_empty()
        );
    }

    #[test]
    fn test_javascript_import() {
        let content = "import { useState } from 'react';";
        let tree = parse_javascript(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_js_ts_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "react");
    }

    #[test]
    fn test_javascript_default_import() {
        let content = "import React from 'react';";
        let tree = parse_javascript(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_js_ts_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "react");
    }

    #[test]
    fn test_go_import_single() {
        let content = r#"package main

import "fmt"
"#;
        let tree = parse_go(content);
        let root = tree.root_node();

        let mut imports = Vec::new();

        fn traverse_for_imports(node: Node, content: &str, imports: &mut Vec<AstImport>) {
            let _ = extract_go_import_node(node, content, imports);
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    traverse_for_imports(child, content, imports);
                }
            }
        }

        traverse_for_imports(root, content, &mut imports);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "fmt");
    }

    #[test]
    fn test_rust_use_declaration() {
        let content = "use std::collections::HashMap;";
        let tree = parse_rust(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_rust_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert!(imports[0].module.contains("std"));
    }

    #[test]
    fn test_rust_use_with_braces() {
        let content = "use std::io::{Read, Write};";
        let tree = parse_rust(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_rust_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert!(imports[0].module.contains("std::io"));
    }

    #[test]
    fn test_ast_import_structure() {
        let import = AstImport {
            module: "test_module".to_string(),
            alias: Some("tm".to_string()),
            items: vec!["Item1".to_string(), "Item2".to_string()],
            line_number: 5,
            is_relative: false,
        };

        assert_eq!(import.module, "test_module");
        assert_eq!(import.alias, Some("tm".to_string()));
        assert_eq!(import.items.len(), 2);
        assert_eq!(import.line_number, 5);
        assert!(!import.is_relative);
    }

    #[test]
    fn test_ast_import_relative() {
        let import = AstImport {
            module: ".utils".to_string(),
            alias: None,
            items: vec![],
            line_number: 1,
            is_relative: true,
        };

        assert!(import.is_relative);
        assert!(import.module.starts_with('.'));
    }

    #[test]
    fn test_empty_imports_list() {
        let content = "x = 1"; // No imports
        let tree = parse_python(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_python_import_node(child, content, &mut imports);
            }
        }

        assert!(imports.is_empty());
    }

    #[test]
    fn test_python_import_with_alias() {
        let content = "import numpy as np";
        let tree = parse_python(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_python_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        // The module or alias should contain "numpy" or "np"
        assert!(
            imports[0].module.contains("numpy")
                || imports[0]
                    .alias
                    .as_ref()
                    .map_or(false, |a| a.contains("np"))
        );
    }

    #[test]
    fn test_python_from_import_multiple_items() {
        let content = "from typing import List, Dict, Optional";
        let tree = parse_python(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_python_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "typing");
    }

    #[test]
    fn test_javascript_double_quote_import() {
        let content = r#"import React from "react";"#;
        let tree = parse_javascript(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_js_ts_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "react");
    }

    #[test]
    fn test_go_import_multiple() {
        let content = r#"package main

import (
    "fmt"
    "os"
)
"#;
        let tree = parse_go(content);
        let root = tree.root_node();

        let mut imports = Vec::new();

        fn traverse(node: Node, content: &str, imports: &mut Vec<AstImport>) {
            let _ = extract_go_import_node(node, content, imports);
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    traverse(child, content, imports);
                }
            }
        }

        traverse(root, content, &mut imports);

        assert!(imports.len() >= 2);
        assert!(imports.iter().any(|i| i.module == "fmt"));
        assert!(imports.iter().any(|i| i.module == "os"));
    }

    #[test]
    fn test_rust_use_crate() {
        let content = "use crate::utils;";
        let tree = parse_rust(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_rust_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert!(imports[0].module.contains("crate"));
    }

    #[test]
    fn test_rust_use_self() {
        let content = "use self::helper;";
        let tree = parse_rust(content);
        let root = tree.root_node();

        let mut imports = Vec::new();
        for i in 0..root.child_count() {
            if let Some(child) = root.child(i) {
                let _ = extract_rust_import_node(child, content, &mut imports);
            }
        }

        assert_eq!(imports.len(), 1);
        assert!(imports[0].module.contains("self"));
    }

    #[test]
    fn test_ast_import_clone() {
        let import = AstImport {
            module: "test".to_string(),
            alias: None,
            items: vec!["item1".to_string()],
            line_number: 1,
            is_relative: false,
        };

        let cloned = import.clone();
        assert_eq!(import.module, cloned.module);
        assert_eq!(import.items, cloned.items);
    }

    #[test]
    fn test_ast_import_debug() {
        let import = AstImport {
            module: "debug_test".to_string(),
            alias: None,
            items: vec![],
            line_number: 10,
            is_relative: false,
        };

        let debug_str = format!("{:?}", import);
        assert!(debug_str.contains("debug_test"));
    }
}

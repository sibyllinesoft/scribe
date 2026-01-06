//! Tests for AST parser module.

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

#[test]
fn test_entity_query_parse_file_only() {
    // No colon: entire string is a file path
    let query = EntityQuery::parse("src/auth.rs");
    assert!(query.name_pattern.is_none());
    assert_eq!(query.file_pattern, Some("src/auth.rs".to_string()));
}

#[test]
fn test_entity_query_parse_file_entity() {
    // file:entity format
    let query = EntityQuery::parse("src/auth.rs:login");
    assert_eq!(query.name_pattern, Some("login".to_string()));
    assert_eq!(query.file_pattern, Some("src/auth.rs".to_string()));
}

#[test]
fn test_entity_query_parse_windows_file_only() {
    // Windows path with single colon (drive letter) - file only
    let query = EntityQuery::parse(r"C:\project\auth.rs");
    assert!(query.name_pattern.is_none());
    assert_eq!(query.file_pattern, Some(r"C:\project\auth.rs".to_string()));
}

#[test]
fn test_entity_query_parse_windows_file_entity() {
    // Windows path with entity (multiple colons, rightmost is separator)
    let query = EntityQuery::parse(r"C:\project\auth.rs:login");
    assert_eq!(query.name_pattern, Some("login".to_string()));
    assert_eq!(query.file_pattern, Some(r"C:\project\auth.rs".to_string()));
}

#[test]
fn test_entity_query_parse_simple_file_entity() {
    // Simple file:entity without path separators
    let query = EntityQuery::parse("auth:UserService");
    assert_eq!(query.name_pattern, Some("UserService".to_string()));
    assert_eq!(query.file_pattern, Some("auth".to_string()));
}

#[test]
fn test_entity_query_matches_file() {
    let query = EntityQuery::parse("auth.rs:login");
    assert!(query.matches_file("src/auth.rs"));
    assert!(query.matches_file("/home/user/project/auth.rs"));
    assert!(query.matches_file("AUTH.rs")); // case insensitive
    assert!(!query.matches_file("src/user.rs"));
}

#[test]
fn test_entity_query_matches_file_with_pattern() {
    // File pattern should match substring
    let query = EntityQuery::parse("auth:login");
    assert!(query.matches_file("src/auth/module.rs"));
    assert!(query.matches_file("authentication.rs"));
    assert!(!query.matches_file("src/user.rs"));
}

//! # Function and Class Extraction from AST
//!
//! Extracts function definitions, class definitions, and methods from source code
//! using tree-sitter AST parsing for accurate analysis.

use super::ast_language::AstLanguage;
use scribe_core::{Result, ScribeError};
use serde::{Deserialize, Serialize};
use tree_sitter::{Language, Node, Parser, Query, QueryCursor, Tree};

/// Information about a function extracted from source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    /// Line number where function starts
    pub start_line: usize,
    /// Line number where function ends
    pub end_line: usize,
    /// Function parameters
    pub parameters: Vec<String>,
    /// Return type (if available)
    pub return_type: Option<String>,
    /// Documentation/docstring
    pub documentation: Option<String>,
    /// Function visibility (public, private, etc.)
    pub visibility: Option<String>,
    /// Whether this is a method (inside a class)
    pub is_method: bool,
    /// Parent class name (if this is a method)
    pub parent_class: Option<String>,
}

/// Information about a class extracted from source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassInfo {
    /// Class name
    pub name: String,
    /// Line number where class starts
    pub start_line: usize,
    /// Line number where class ends
    pub end_line: usize,
    /// Parent classes/interfaces
    pub parents: Vec<String>,
    /// Documentation/docstring
    pub documentation: Option<String>,
    /// Class visibility
    pub visibility: Option<String>,
    /// Methods in this class
    pub methods: Vec<FunctionInfo>,
}

/// Extracts functions and classes from source code using tree-sitter
pub struct FunctionExtractor {
    language: AstLanguage,
    parser: Parser,
    function_query: Option<Query>,
    class_query: Option<Query>,
}

impl FunctionExtractor {
    /// Create a new function extractor for the given language
    pub fn new(language: AstLanguage) -> Result<Self> {
        let mut parser = Parser::new();

        // Set up tree-sitter language if available
        let (function_query, class_query) =
            if let Some(ts_language) = language.tree_sitter_language() {
                parser
                    .set_language(ts_language)
                    .map_err(|e| ScribeError::Analysis {
                        message: format!("Failed to set tree-sitter language: {}", e),
                        source: None,
                        file: std::path::PathBuf::from("<unknown>"),
                    })?;

                let function_query = Self::create_function_query(language, ts_language)?;
                let class_query = Self::create_class_query(language, ts_language)?;
                (function_query, class_query)
            } else {
                (None, None)
            };

        Ok(Self {
            language,
            parser,
            function_query,
            class_query,
        })
    }

    /// Create tree-sitter query for finding functions
    fn create_function_query(
        language: AstLanguage,
        ts_language: Language,
    ) -> Result<Option<Query>> {
        let query_string = match language {
            AstLanguage::Python => {
                r#"
                (function_definition) @function.definition
            "#
            }
            AstLanguage::JavaScript | AstLanguage::TypeScript => {
                r#"
                (function_declaration) @function.definition
                (method_definition) @function.definition
            "#
            }
            AstLanguage::Rust => {
                r#"
                (function_item) @function.definition
            "#
            }
            AstLanguage::Go => {
                r#"
                (function_declaration) @function.definition
                (method_declaration) @function.definition
            "#
            }
            // Future languages - placeholder queries
            AstLanguage::Java => {
                r#"
                (method_declaration) @function.definition
            "#
            }
            AstLanguage::C | AstLanguage::Cpp => {
                r#"
                (function_definition) @function.definition
            "#
            }
            AstLanguage::Ruby => {
                r#"
                (method) @function.definition
            "#
            }
            AstLanguage::CSharp => {
                r#"
                (method_declaration) @function.definition
            "#
            }
            _ => return Ok(None),
        };

        Query::new(ts_language, query_string)
            .map(Some)
            .map_err(|e| ScribeError::Analysis {
                message: format!("Failed to create function query: {}", e),
                source: None,
                file: std::path::PathBuf::from("<unknown>"),
            })
    }

    /// Create tree-sitter query for finding classes
    fn create_class_query(language: AstLanguage, ts_language: Language) -> Result<Option<Query>> {
        let query_string = match language {
            AstLanguage::Python => {
                r#"
                (class_definition) @class.definition
            "#
            }
            AstLanguage::JavaScript | AstLanguage::TypeScript => {
                r#"
                (class_declaration) @class.definition
            "#
            }
            AstLanguage::Rust => {
                r#"
                (struct_item) @class.definition
            "#
            }
            AstLanguage::Go => {
                r#"
                (type_declaration) @class.definition
            "#
            }
            // Future languages - placeholder queries
            AstLanguage::Java => {
                r#"
                (class_declaration) @class.definition
            "#
            }
            AstLanguage::Cpp => {
                r#"
                (class_specifier) @class.definition
            "#
            }
            AstLanguage::Ruby => {
                r#"
                (class) @class.definition
            "#
            }
            AstLanguage::CSharp => {
                r#"
                (class_declaration) @class.definition
            "#
            }
            _ => return Ok(None),
        };

        Query::new(ts_language, query_string)
            .map(Some)
            .map_err(|e| ScribeError::Analysis {
                message: format!("Failed to create class query: {}", e),
                source: None,
                file: std::path::PathBuf::from("<unknown>"),
            })
    }

    /// Extract all functions from source code
    pub fn extract_functions(&mut self, content: &str) -> Result<Vec<FunctionInfo>> {
        let tree = self
            .parser
            .parse(content, None)
            .ok_or_else(|| ScribeError::Analysis {
                message: "Failed to parse source code".to_string(),
                source: None,
                file: std::path::PathBuf::from("<unknown>"),
            })?;

        let mut functions = Vec::new();

        if let Some(query) = &self.function_query {
            let mut query_cursor = QueryCursor::new();
            let matches = query_cursor.matches(query, tree.root_node(), content.as_bytes());

            for query_match in matches {
                if let Some(function_info) =
                    self.extract_function_from_match(&query_match, content, &tree)?
                {
                    functions.push(function_info);
                }
            }
        }

        Ok(functions)
    }

    /// Extract all classes from source code
    pub fn extract_classes(&mut self, content: &str) -> Result<Vec<ClassInfo>> {
        let tree = self
            .parser
            .parse(content, None)
            .ok_or_else(|| ScribeError::Analysis {
                message: "Failed to parse source code".to_string(),
                source: None,
                file: std::path::PathBuf::from("<unknown>"),
            })?;

        let mut classes = Vec::new();

        if let Some(query) = &self.class_query {
            let mut query_cursor = QueryCursor::new();
            let matches = query_cursor.matches(query, tree.root_node(), content.as_bytes());

            for query_match in matches {
                if let Some(class_info) =
                    self.extract_class_from_match(&query_match, content, &tree)?
                {
                    classes.push(class_info);
                }
            }
        }

        Ok(classes)
    }

    /// Extract function information from a query match
    fn extract_function_from_match(
        &self,
        query_match: &tree_sitter::QueryMatch,
        content: &str,
        tree: &Tree,
    ) -> Result<Option<FunctionInfo>> {
        for capture in query_match.captures {
            let node = capture.node;
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;

            // Extract function name from the AST node structure
            let name = self.extract_function_name(node, content);
            let parameters = self.extract_function_parameters(node, content);

            if let Some(function_name) = name {
                return Ok(Some(FunctionInfo {
                    name: function_name,
                    start_line,
                    end_line,
                    parameters,
                    return_type: None,   // TODO: Extract return type
                    documentation: None, // TODO: Extract documentation
                    visibility: None,    // TODO: Extract visibility
                    is_method: false,    // TODO: Determine if method
                    parent_class: None,  // TODO: Find parent class
                }));
            }
        }
        Ok(None)
    }

    /// Extract class information from a query match
    fn extract_class_from_match(
        &self,
        query_match: &tree_sitter::QueryMatch,
        content: &str,
        tree: &Tree,
    ) -> Result<Option<ClassInfo>> {
        for capture in query_match.captures {
            let node = capture.node;
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;

            // Extract class name from the AST node structure
            let name = self.extract_class_name(node, content);
            let parents = self.extract_class_parents(node, content);

            if let Some(class_name) = name {
                return Ok(Some(ClassInfo {
                    name: class_name,
                    start_line,
                    end_line,
                    parents,
                    documentation: None, // TODO: Extract documentation
                    visibility: None,    // TODO: Extract visibility
                    methods: Vec::new(), // TODO: Extract methods
                }));
            }
        }
        Ok(None)
    }

    /// Extract function name from AST node
    fn extract_function_name(&self, node: Node, content: &str) -> Option<String> {
        // Look for identifier child nodes that represent the function name
        let mut cursor = node.walk();
        cursor.goto_first_child();

        loop {
            let child = cursor.node();
            match child.kind() {
                "identifier" => {
                    if let Ok(name) = child.utf8_text(content.as_bytes()) {
                        return Some(name.to_string());
                    }
                }
                _ => {}
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
        None
    }

    /// Extract function parameters from AST node
    fn extract_function_parameters(&self, node: Node, content: &str) -> Vec<String> {
        let mut parameters = Vec::new();
        let mut cursor = node.walk();
        cursor.goto_first_child();

        loop {
            let child = cursor.node();
            match child.kind() {
                "parameters" | "parameter_list" => {
                    // Extract parameter names from parameter list
                    let mut param_cursor = child.walk();
                    param_cursor.goto_first_child();

                    loop {
                        let param_node = param_cursor.node();
                        if param_node.kind() == "identifier" {
                            if let Ok(param_name) = param_node.utf8_text(content.as_bytes()) {
                                if param_name != "self" {
                                    parameters.push(param_name.to_string());
                                }
                            }
                        }

                        if !param_cursor.goto_next_sibling() {
                            break;
                        }
                    }
                    break;
                }
                _ => {}
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
        parameters
    }

    /// Extract class name from AST node
    fn extract_class_name(&self, node: Node, content: &str) -> Option<String> {
        // Look for identifier child nodes that represent the class name
        let mut cursor = node.walk();
        cursor.goto_first_child();

        loop {
            let child = cursor.node();
            match child.kind() {
                "identifier" | "type_identifier" => {
                    if let Ok(name) = child.utf8_text(content.as_bytes()) {
                        return Some(name.to_string());
                    }
                }
                _ => {}
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
        None
    }

    /// Extract class parent classes from AST node
    fn extract_class_parents(&self, node: Node, content: &str) -> Vec<String> {
        let mut parents = Vec::new();
        let mut cursor = node.walk();
        cursor.goto_first_child();

        loop {
            let child = cursor.node();
            match child.kind() {
                "argument_list" | "superclass" | "inheritance" => {
                    // Extract parent class names
                    let mut parent_cursor = child.walk();
                    parent_cursor.goto_first_child();

                    loop {
                        let parent_node = parent_cursor.node();
                        if parent_node.kind() == "identifier"
                            || parent_node.kind() == "type_identifier"
                        {
                            if let Ok(parent_name) = parent_node.utf8_text(content.as_bytes()) {
                                parents.push(parent_name.to_string());
                            }
                        }

                        if !parent_cursor.goto_next_sibling() {
                            break;
                        }
                    }
                }
                _ => {}
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
        parents
    }

    /// Extract parameter names from parameter list text
    fn extract_parameters(&self, params_text: &str, _node: Node) -> Vec<String> {
        // Simple parameter extraction - can be improved per language
        params_text
            .split(',')
            .filter_map(|param| {
                let param = param.trim();
                if param.is_empty() || param == "self" {
                    None
                } else {
                    // Extract just the parameter name (before type annotations)
                    let name = param.split(':').next().unwrap_or(param).trim();
                    if name.is_empty() {
                        None
                    } else {
                        Some(name.to_string())
                    }
                }
            })
            .collect()
    }

    /// Extract parent class names from inheritance clause
    fn extract_parent_classes(&self, parents_text: &str) -> Vec<String> {
        // Simple parent class extraction - can be improved per language
        parents_text
            .split(',')
            .filter_map(|parent| {
                let parent = parent.trim();
                if parent.is_empty() {
                    None
                } else {
                    Some(parent.to_string())
                }
            })
            .collect()
    }
}

impl AstLanguage {
    /// Get tree-sitter language for supported languages
    pub fn tree_sitter_language(&self) -> Option<tree_sitter::Language> {
        match self {
            AstLanguage::Python => Some(tree_sitter_python::language()),
            AstLanguage::JavaScript => Some(tree_sitter_javascript::language()),
            AstLanguage::TypeScript => Some(tree_sitter_typescript::language_typescript()),
            AstLanguage::Go => Some(tree_sitter_go::language()),
            AstLanguage::Rust => Some(tree_sitter_rust::language()),
            AstLanguage::Html => Some(tree_sitter_html::language()),
            // TODO: Add when dependencies are available
            // AstLanguage::Java => Some(tree_sitter_java::language()),
            // AstLanguage::C => Some(tree_sitter_c::language()),
            // AstLanguage::Cpp => Some(tree_sitter_cpp::language()),
            // AstLanguage::Ruby => Some(tree_sitter_ruby::language()),
            // AstLanguage::CSharp => Some(tree_sitter_c_sharp::language()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_extractor_creation() {
        let extractor = FunctionExtractor::new(AstLanguage::Python);
        assert!(extractor.is_ok());
    }

    #[test]
    fn test_python_function_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();
        let python_code = r#"
def hello_world():
    """A simple function."""
    print("Hello, World!")

def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

class Calculator:
    """A simple calculator."""
    
    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y
"#;

        let functions = extractor.extract_functions(python_code).unwrap();
        assert!(!functions.is_empty());

        // Should find at least the standalone functions
        let function_names: Vec<&String> = functions.iter().map(|f| &f.name).collect();
        assert!(function_names.contains(&&"hello_world".to_string()));
        assert!(function_names.contains(&&"add_numbers".to_string()));
    }

    #[test]
    fn test_python_class_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();
        let python_code = r#"
class Calculator:
    """A simple calculator."""
    pass

class AdvancedCalculator(Calculator):
    """An advanced calculator that inherits from Calculator."""
    pass
"#;

        let classes = extractor.extract_classes(python_code).unwrap();
        assert!(!classes.is_empty());

        let class_names: Vec<&String> = classes.iter().map(|c| &c.name).collect();
        assert!(class_names.contains(&&"Calculator".to_string()));
        assert!(class_names.contains(&&"AdvancedCalculator".to_string()));
    }

    #[test]
    fn test_javascript_function_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::JavaScript).unwrap();
        let js_code = r#"
function greetUser(name) {
    return `Hello, ${name}!`;
}

class UserManager {
    constructor() {
        this.users = [];
    }
    
    addUser(user) {
        this.users.push(user);
    }
}
"#;

        let functions = extractor.extract_functions(js_code).unwrap();
        assert!(!functions.is_empty());
    }
}

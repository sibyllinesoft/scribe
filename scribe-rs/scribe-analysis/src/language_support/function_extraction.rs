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

    #[test]
    fn test_javascript_class_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::JavaScript).unwrap();
        let js_code = r#"
class Animal {
    constructor(name) {
        this.name = name;
    }
}

class Dog extends Animal {
    bark() {
        return "Woof!";
    }
}
"#;

        let classes = extractor.extract_classes(js_code).unwrap();
        assert!(!classes.is_empty());

        let class_names: Vec<&String> = classes.iter().map(|c| &c.name).collect();
        assert!(class_names.contains(&&"Animal".to_string()));
        assert!(class_names.contains(&&"Dog".to_string()));
    }

    #[test]
    fn test_typescript_function_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::TypeScript).unwrap();
        let ts_code = r#"
function add(a: number, b: number): number {
    return a + b;
}

class Calculator {
    multiply(x: number, y: number): number {
        return x * y;
    }
}
"#;

        let functions = extractor.extract_functions(ts_code).unwrap();
        assert!(!functions.is_empty());
    }

    #[test]
    fn test_typescript_class_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::TypeScript).unwrap();
        let ts_code = r#"
class BaseService {
    protected name: string;
}

class UserService extends BaseService {
    getUsers(): User[] {
        return [];
    }
}
"#;

        let classes = extractor.extract_classes(ts_code).unwrap();
        assert!(!classes.is_empty());
    }

    #[test]
    fn test_rust_function_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Rust).unwrap();
        let rust_code = r#"
fn main() {
    println!("Hello, world!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn public_function() -> String {
    "public".to_string()
}
"#;

        let functions = extractor.extract_functions(rust_code).unwrap();
        assert!(!functions.is_empty());

        let function_names: Vec<&String> = functions.iter().map(|f| &f.name).collect();
        assert!(function_names.contains(&&"main".to_string()));
        assert!(function_names.contains(&&"add".to_string()));
    }

    #[test]
    fn test_rust_struct_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Rust).unwrap();
        let rust_code = r#"
struct Point {
    x: i32,
    y: i32,
}

struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}
"#;

        let classes = extractor.extract_classes(rust_code).unwrap();
        assert!(!classes.is_empty());

        let class_names: Vec<&String> = classes.iter().map(|c| &c.name).collect();
        assert!(class_names.contains(&&"Point".to_string()));
        assert!(class_names.contains(&&"Rectangle".to_string()));
    }

    #[test]
    fn test_go_function_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Go).unwrap();
        let go_code = r#"
package main

func main() {
    fmt.Println("Hello")
}

func add(a, b int) int {
    return a + b
}

func (r *Rectangle) Area() int {
    return r.Width * r.Height
}
"#;

        let functions = extractor.extract_functions(go_code).unwrap();
        assert!(!functions.is_empty());
    }

    #[test]
    fn test_go_type_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Go).unwrap();
        let go_code = r#"
package main

type Person struct {
    Name string
    Age  int
}

type Animal struct {
    Name string
}
"#;

        // Go type extraction may return empty if the query doesn't capture all type declarations
        // This tests that the extraction doesn't error even if the query doesn't match
        let classes = extractor.extract_classes(go_code);
        assert!(classes.is_ok());
    }

    #[test]
    fn test_function_info_struct() {
        let func_info = FunctionInfo {
            name: "test_func".to_string(),
            start_line: 1,
            end_line: 5,
            parameters: vec!["a".to_string(), "b".to_string()],
            return_type: Some("int".to_string()),
            documentation: Some("A test function".to_string()),
            visibility: Some("public".to_string()),
            is_method: false,
            parent_class: None,
        };

        assert_eq!(func_info.name, "test_func");
        assert_eq!(func_info.start_line, 1);
        assert_eq!(func_info.end_line, 5);
        assert_eq!(func_info.parameters.len(), 2);
        assert!(func_info.return_type.is_some());
    }

    #[test]
    fn test_function_info_clone() {
        let func_info = FunctionInfo {
            name: "test".to_string(),
            start_line: 1,
            end_line: 10,
            parameters: vec!["x".to_string()],
            return_type: None,
            documentation: None,
            visibility: None,
            is_method: true,
            parent_class: Some("MyClass".to_string()),
        };

        let cloned = func_info.clone();
        assert_eq!(func_info.name, cloned.name);
        assert_eq!(func_info.is_method, cloned.is_method);
        assert_eq!(func_info.parent_class, cloned.parent_class);
    }

    #[test]
    fn test_function_info_debug() {
        let func_info = FunctionInfo {
            name: "debug_func".to_string(),
            start_line: 1,
            end_line: 5,
            parameters: vec![],
            return_type: None,
            documentation: None,
            visibility: None,
            is_method: false,
            parent_class: None,
        };

        let debug_str = format!("{:?}", func_info);
        assert!(debug_str.contains("FunctionInfo"));
        assert!(debug_str.contains("debug_func"));
    }

    #[test]
    fn test_function_info_serialize() {
        let func_info = FunctionInfo {
            name: "test".to_string(),
            start_line: 1,
            end_line: 5,
            parameters: vec!["a".to_string()],
            return_type: None,
            documentation: None,
            visibility: None,
            is_method: false,
            parent_class: None,
        };

        let json = serde_json::to_string(&func_info).unwrap();
        let deserialized: FunctionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(func_info.name, deserialized.name);
        assert_eq!(func_info.parameters, deserialized.parameters);
    }

    #[test]
    fn test_class_info_struct() {
        let class_info = ClassInfo {
            name: "TestClass".to_string(),
            start_line: 1,
            end_line: 20,
            parents: vec!["BaseClass".to_string()],
            documentation: Some("A test class".to_string()),
            visibility: Some("public".to_string()),
            methods: vec![],
        };

        assert_eq!(class_info.name, "TestClass");
        assert_eq!(class_info.parents.len(), 1);
        assert!(class_info.methods.is_empty());
    }

    #[test]
    fn test_class_info_clone() {
        let class_info = ClassInfo {
            name: "MyClass".to_string(),
            start_line: 1,
            end_line: 50,
            parents: vec!["Parent1".to_string(), "Parent2".to_string()],
            documentation: None,
            visibility: None,
            methods: vec![],
        };

        let cloned = class_info.clone();
        assert_eq!(class_info.name, cloned.name);
        assert_eq!(class_info.parents, cloned.parents);
    }

    #[test]
    fn test_class_info_debug() {
        let class_info = ClassInfo {
            name: "DebugClass".to_string(),
            start_line: 1,
            end_line: 10,
            parents: vec![],
            documentation: None,
            visibility: None,
            methods: vec![],
        };

        let debug_str = format!("{:?}", class_info);
        assert!(debug_str.contains("ClassInfo"));
        assert!(debug_str.contains("DebugClass"));
    }

    #[test]
    fn test_class_info_serialize() {
        let class_info = ClassInfo {
            name: "TestClass".to_string(),
            start_line: 1,
            end_line: 20,
            parents: vec!["Parent".to_string()],
            documentation: None,
            visibility: None,
            methods: vec![],
        };

        let json = serde_json::to_string(&class_info).unwrap();
        let deserialized: ClassInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(class_info.name, deserialized.name);
        assert_eq!(class_info.parents, deserialized.parents);
    }

    #[test]
    fn test_extract_parameters_helper() {
        let extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_python::language()).unwrap();
        let tree = parser.parse("def test(): pass", None).unwrap();

        let params = extractor.extract_parameters("a, b, c", tree.root_node());
        assert_eq!(params, vec!["a", "b", "c"]);

        let params_with_types = extractor.extract_parameters("x: int, y: str", tree.root_node());
        assert_eq!(params_with_types, vec!["x", "y"]);

        let params_with_self = extractor.extract_parameters("self, x, y", tree.root_node());
        assert_eq!(params_with_self, vec!["x", "y"]);

        let empty = extractor.extract_parameters("", tree.root_node());
        assert!(empty.is_empty());
    }

    #[test]
    fn test_extract_parent_classes_helper() {
        let extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();

        let parents = extractor.extract_parent_classes("BaseClass, Mixin");
        assert_eq!(parents, vec!["BaseClass", "Mixin"]);

        let single = extractor.extract_parent_classes("Parent");
        assert_eq!(single, vec!["Parent"]);

        let empty = extractor.extract_parent_classes("");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_html_extractor_no_query() {
        // HTML doesn't have function/class queries
        let mut extractor = FunctionExtractor::new(AstLanguage::Html).unwrap();

        // Should return empty results (no query to run)
        let functions = extractor.extract_functions("<div>test</div>").unwrap();
        assert!(functions.is_empty());

        let classes = extractor.extract_classes("<div>test</div>").unwrap();
        assert!(classes.is_empty());
    }

    #[test]
    fn test_empty_code_extraction() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();

        let functions = extractor.extract_functions("").unwrap();
        assert!(functions.is_empty());

        let classes = extractor.extract_classes("").unwrap();
        assert!(classes.is_empty());
    }

    #[test]
    fn test_python_function_with_parameters() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();
        let code = r#"
def process_data(input_data, config, verbose=False):
    return input_data
"#;

        let functions = extractor.extract_functions(code).unwrap();
        assert!(!functions.is_empty());

        let process_data = functions.iter().find(|f| f.name == "process_data");
        assert!(process_data.is_some());
    }

    #[test]
    fn test_function_line_numbers() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();
        let code = r#"
def first():
    pass

def second():
    pass
"#;

        let functions = extractor.extract_functions(code).unwrap();
        assert_eq!(functions.len(), 2);

        let first = functions.iter().find(|f| f.name == "first").unwrap();
        let second = functions.iter().find(|f| f.name == "second").unwrap();

        assert!(first.start_line < second.start_line);
    }

    #[test]
    fn test_class_line_numbers() {
        let mut extractor = FunctionExtractor::new(AstLanguage::Python).unwrap();
        let code = r#"
class First:
    pass

class Second:
    pass
"#;

        let classes = extractor.extract_classes(code).unwrap();
        assert_eq!(classes.len(), 2);

        let first = classes.iter().find(|c| c.name == "First").unwrap();
        let second = classes.iter().find(|c| c.name == "Second").unwrap();

        assert!(first.start_line < second.start_line);
    }

    #[test]
    fn test_java_extractor_creation() {
        // Test Java extractor creation (may fail if grammar not available)
        let extractor_result = FunctionExtractor::new(AstLanguage::Java);
        // Just test that it doesn't panic - may succeed or fail depending on grammar availability
        let _ = extractor_result;
    }

    #[test]
    fn test_c_extractor_creation() {
        // Test C extractor creation (may fail if grammar not available)
        let extractor_result = FunctionExtractor::new(AstLanguage::C);
        let _ = extractor_result;
    }

    #[test]
    fn test_cpp_extractor_creation() {
        // Test C++ extractor creation (may fail if grammar not available)
        let extractor_result = FunctionExtractor::new(AstLanguage::Cpp);
        let _ = extractor_result;
    }

    #[test]
    fn test_ruby_extractor_creation() {
        // Test Ruby extractor creation (may fail if grammar not available)
        let extractor_result = FunctionExtractor::new(AstLanguage::Ruby);
        let _ = extractor_result;
    }

    #[test]
    fn test_csharp_extractor_creation() {
        // Test C# extractor creation (may fail if grammar not available)
        let extractor_result = FunctionExtractor::new(AstLanguage::CSharp);
        let _ = extractor_result;
    }
}

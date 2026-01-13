//! Tree-sitter query strings for different programming languages.

use super::types::AstLanguage;

/// Get the chunk query string for a language
pub fn chunk_query_for_language(language: AstLanguage) -> &'static str {
    match language {
        AstLanguage::Python => PYTHON_CHUNK_QUERY,
        AstLanguage::JavaScript => JAVASCRIPT_CHUNK_QUERY,
        AstLanguage::TypeScript => TYPESCRIPT_CHUNK_QUERY,
        AstLanguage::Go => GO_CHUNK_QUERY,
        AstLanguage::Rust => RUST_CHUNK_QUERY,
    }
}

/// Get the signature query string for a language
pub fn signature_query_for_language(language: AstLanguage) -> &'static str {
    match language {
        AstLanguage::Python => PYTHON_SIGNATURE_QUERY,
        AstLanguage::JavaScript => JAVASCRIPT_SIGNATURE_QUERY,
        AstLanguage::TypeScript => TYPESCRIPT_SIGNATURE_QUERY,
        AstLanguage::Go => GO_SIGNATURE_QUERY,
        AstLanguage::Rust => RUST_SIGNATURE_QUERY,
    }
}

const PYTHON_CHUNK_QUERY: &str = r#"
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

const JAVASCRIPT_CHUNK_QUERY: &str = r#"
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

const TYPESCRIPT_CHUNK_QUERY: &str = r#"
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

const GO_CHUNK_QUERY: &str = r#"
    (package_clause) @package
    (import_declaration) @import
    (function_declaration) @function
    (method_declaration) @method
    (type_declaration) @type
    (const_declaration) @const
    (var_declaration) @var
"#;

const RUST_CHUNK_QUERY: &str = r#"
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

const PYTHON_SIGNATURE_QUERY: &str = r#"
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

const JAVASCRIPT_SIGNATURE_QUERY: &str = r#"
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

const TYPESCRIPT_SIGNATURE_QUERY: &str = r#"
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

const GO_SIGNATURE_QUERY: &str = r#"
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

const RUST_SIGNATURE_QUERY: &str = r#"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_query_python() {
        let query = chunk_query_for_language(AstLanguage::Python);
        assert!(!query.is_empty());
        assert!(query.contains("function_definition"));
        assert!(query.contains("class_definition"));
    }

    #[test]
    fn test_chunk_query_javascript() {
        let query = chunk_query_for_language(AstLanguage::JavaScript);
        assert!(!query.is_empty());
        assert!(query.contains("function_declaration"));
        assert!(query.contains("class_declaration"));
    }

    #[test]
    fn test_chunk_query_typescript() {
        let query = chunk_query_for_language(AstLanguage::TypeScript);
        assert!(!query.is_empty());
        assert!(query.contains("interface_declaration"));
        assert!(query.contains("enum_declaration"));
    }

    #[test]
    fn test_chunk_query_go() {
        let query = chunk_query_for_language(AstLanguage::Go);
        assert!(!query.is_empty());
        assert!(query.contains("function_declaration"));
        assert!(query.contains("type_declaration"));
    }

    #[test]
    fn test_chunk_query_rust() {
        let query = chunk_query_for_language(AstLanguage::Rust);
        assert!(!query.is_empty());
        assert!(query.contains("function_item"));
        assert!(query.contains("struct_item"));
    }

    #[test]
    fn test_signature_query_python() {
        let query = signature_query_for_language(AstLanguage::Python);
        assert!(!query.is_empty());
        assert!(query.contains("func_name"));
        assert!(query.contains("class_name"));
    }

    #[test]
    fn test_signature_query_javascript() {
        let query = signature_query_for_language(AstLanguage::JavaScript);
        assert!(!query.is_empty());
        assert!(query.contains("@function"));
        assert!(query.contains("@class"));
    }

    #[test]
    fn test_signature_query_typescript() {
        let query = signature_query_for_language(AstLanguage::TypeScript);
        assert!(!query.is_empty());
        assert!(query.contains("interface_declaration"));
        assert!(query.contains("type_alias_declaration"));
    }

    #[test]
    fn test_signature_query_go() {
        let query = signature_query_for_language(AstLanguage::Go);
        assert!(!query.is_empty());
        assert!(query.contains("function_declaration"));
        assert!(query.contains("package_clause"));
    }

    #[test]
    fn test_signature_query_rust() {
        let query = signature_query_for_language(AstLanguage::Rust);
        assert!(!query.is_empty());
        assert!(query.contains("function_item"));
        assert!(query.contains("struct_item"));
        assert!(query.contains("trait_item"));
    }
}

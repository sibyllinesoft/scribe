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

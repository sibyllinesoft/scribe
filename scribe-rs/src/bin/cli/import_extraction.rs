//! Import extraction utilities for different programming languages

use scribe_core::Language;
use std::collections::HashSet;

/// Extract imports from file content based on language
pub fn extract_imports_for_diff(content: &str, language: &Language) -> Vec<String> {
    let mut imports = HashSet::new();

    match language {
        Language::Rust => extract_rust_imports(content, &mut imports),
        Language::Python => extract_python_imports(content, &mut imports),
        Language::JavaScript | Language::TypeScript => extract_js_imports(content, &mut imports),
        Language::Go => extract_go_imports(content, &mut imports),
        Language::Elixir => extract_elixir_imports(content, &mut imports),
        _ => {}
    }

    let mut ordered: Vec<String> = imports.into_iter().collect();
    ordered.sort();
    ordered.truncate(64);
    ordered
}

pub fn extract_rust_imports(content: &str, imports: &mut HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("use ") {
            let statement = trimmed
                .trim_start_matches("use ")
                .trim_end_matches(';')
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_end_matches("::");
            if !statement.is_empty() {
                imports.insert(statement.to_string());
            }
        } else if trimmed.starts_with("mod ") {
            let module = trimmed
                .trim_start_matches("mod ")
                .trim_end_matches(';')
                .trim();
            if !module.is_empty() {
                imports.insert(module.to_string());
            }
        }
    }
}

pub fn extract_python_imports(content: &str, imports: &mut HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            for module in trimmed.trim_start_matches("import ").split(',') {
                let module = module.trim().split_whitespace().next().unwrap_or("");
                if !module.is_empty() {
                    imports.insert(module.to_string());
                }
            }
        } else if trimmed.starts_with("from ") && trimmed.contains(" import ") {
            let module = trimmed
                .trim_start_matches("from ")
                .split(" import ")
                .next()
                .unwrap_or("")
                .trim();
            if !module.is_empty() {
                imports.insert(module.to_string());
            }
        }
    }
}

pub fn extract_js_imports(content: &str, imports: &mut HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            if let Some(start) = trimmed.find('"') {
                if let Some(end) = trimmed[start + 1..].find('"') {
                    imports.insert(trimmed[start + 1..start + 1 + end].to_string());
                }
            } else if let Some(start) = trimmed.find('\'') {
                if let Some(end) = trimmed[start + 1..].find('\'') {
                    imports.insert(trimmed[start + 1..start + 1 + end].to_string());
                }
            }
        } else if trimmed.contains("require(") {
            if let Some(start) = trimmed.find("require(") {
                let start = start + "require(".len();
                let slice = &trimmed[start..];
                if let Some(end_idx) = slice.find(')') {
                    let inner = &slice[..end_idx];
                    let inner = inner.trim_matches(&['\'', '"'][..]);
                    if !inner.is_empty() {
                        imports.insert(inner.to_string());
                    }
                }
            }
        }
    }
}

pub fn extract_go_imports(content: &str, imports: &mut HashSet<String>) {
    let mut in_block = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "import (" {
            in_block = true;
            continue;
        }
        if in_block {
            if trimmed == ")" {
                in_block = false;
                continue;
            }
            let import_path = trimmed.trim_matches(&['"', '`'][..]);
            if !import_path.is_empty() {
                imports.insert(import_path.to_string());
            }
        } else if trimmed.starts_with("import ") {
            let import_path = trimmed
                .trim_start_matches("import ")
                .trim_matches(&['"', '`'][..]);
            if !import_path.is_empty() {
                imports.insert(import_path.to_string());
            }
        }
    }
}

/// Extract Elixir imports (`alias`, `import`, `require`, `use`) from content
pub fn extract_elixir_imports(content: &str, imports: &mut HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        let without_comments = trimmed.split('#').next().unwrap_or("").trim();
        if without_comments.is_empty() {
            continue;
        }

        for keyword in ["alias ", "import ", "require ", "use "] {
            if let Some(statement) = without_comments.strip_prefix(keyword) {
                extract_elixir_import_statement(statement, imports);
                break;
            }
        }
    }
}

fn extract_elixir_import_statement(statement: &str, imports: &mut HashSet<String>) {
    if let Some((base, remainder)) = statement.split_once('{') {
        let base = normalize_elixir_module(base.trim_end_matches('.'));
        if let Some(end) = remainder.find('}') {
            let grouped = &remainder[..end];
            for module in grouped.split(',') {
                if let Some(module) = normalize_elixir_module(module) {
                    if let Some(ref base) = base {
                        imports.insert(format!("{}.{}", base, module));
                    } else {
                        imports.insert(module);
                    }
                }
            }
        }
        return;
    }

    if let Some(module) = normalize_elixir_module(statement) {
        imports.insert(module);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_imports_for_diff_elixir() {
        let content = r#"
alias MyApp.Repo
alias MyApp.{Accounts.User, Accounts.Team}
import Plug.Conn
require Logger
use MyAppWeb, :controller
"#;

        let imports = extract_imports_for_diff(content, &Language::Elixir);

        assert!(imports.contains(&"MyApp.Repo".to_string()));
        assert!(imports.contains(&"MyApp.Accounts.User".to_string()));
        assert!(imports.contains(&"MyApp.Accounts.Team".to_string()));
        assert!(imports.contains(&"Plug.Conn".to_string()));
        assert!(imports.contains(&"Logger".to_string()));
        assert!(imports.contains(&"MyAppWeb".to_string()));
    }
}

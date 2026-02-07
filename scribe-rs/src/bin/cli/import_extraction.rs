//! Import extraction utilities for different programming languages

use scribe_core::Language;
use std::collections::HashSet;

#[cfg(feature = "analysis")]
use scribe_analysis::ast_import_parser::{ImportLanguage, SimpleAstParser};

/// Extract imports from file content based on language
pub fn extract_imports_for_diff(content: &str, language: &Language) -> Vec<String> {
    let mut imports = HashSet::new();

    match language {
        Language::Rust => extract_rust_imports(content, &mut imports),
        Language::Python => extract_python_imports(content, &mut imports),
        Language::JavaScript | Language::TypeScript => extract_js_imports(content, &mut imports),
        Language::Go => extract_go_imports(content, &mut imports),
        Language::Elixir => {
            if let Some(ast_imports) = try_extract_elixir_imports_with_ast(content) {
                imports.extend(ast_imports);
            } else {
                extract_elixir_imports_fallback(content, &mut imports);
            }
        }
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

#[cfg(feature = "analysis")]
fn try_extract_elixir_imports_with_ast(content: &str) -> Option<Vec<String>> {
    let parser = SimpleAstParser::new().ok()?;
    let imports = parser
        .extract_imports(content, ImportLanguage::Elixir)
        .ok()?;

    Some(imports.into_iter().map(|import| import.module).collect())
}

#[cfg(not(feature = "analysis"))]
fn try_extract_elixir_imports_with_ast(_content: &str) -> Option<Vec<String>> {
    None
}

pub fn extract_elixir_imports_fallback(content: &str, imports: &mut HashSet<String>) {
    let lines: Vec<&str> = content.lines().collect();
    let mut index = 0;
    let mut in_heredoc: Option<&'static str> = None;

    while index < lines.len() {
        let line = lines[index];
        let trimmed = line.trim_start();

        if let Some(delimiter) = in_heredoc {
            if trimmed.contains(delimiter) {
                in_heredoc = None;
            }
            index += 1;
            continue;
        }

        if let Some(delimiter) = detect_heredoc_start(trimmed) {
            in_heredoc = Some(delimiter);
            index += 1;
            continue;
        }

        if !is_elixir_import_statement_start(trimmed) {
            index += 1;
            continue;
        }

        let mut statement = strip_inline_comment(trimmed).to_string();
        let mut brace_depth = brace_delta(&statement);
        index += 1;

        while index < lines.len() && brace_depth > 0 {
            let next_line = lines[index].trim_start();

            if let Some(delimiter) = in_heredoc {
                if next_line.contains(delimiter) {
                    in_heredoc = None;
                }
                index += 1;
                continue;
            }

            if let Some(delimiter) = detect_heredoc_start(next_line) {
                in_heredoc = Some(delimiter);
                index += 1;
                continue;
            }

            let sanitized = strip_inline_comment(next_line);
            if !sanitized.is_empty() {
                if !statement.is_empty() {
                    statement.push(' ');
                }
                statement.push_str(sanitized);
                brace_depth += brace_delta(sanitized);
            }

            index += 1;
        }

        for module in parse_elixir_import_statement(&statement) {
            imports.insert(module);
        }
    }
}

fn parse_elixir_import_statement(statement: &str) -> Vec<String> {
    let statement = statement.trim();
    let Some((keyword, rest)) = ["alias", "import", "require", "use"]
        .iter()
        .find_map(|keyword| {
            if statement.starts_with(keyword)
                && statement
                    .chars()
                    .nth(keyword.len())
                    .map(|ch| ch.is_whitespace())
                    .unwrap_or(false)
            {
                Some((*keyword, statement[keyword.len()..].trim()))
            } else {
                None
            }
        })
    else {
        return Vec::new();
    };

    let mut modules = Vec::new();
    for segment in split_top_level(rest, ',') {
        let compact_segment = compact_elixir_expression(&segment);

        if compact_segment.is_empty() {
            continue;
        }

        if looks_like_elixir_option(&compact_segment) {
            break;
        }

        if keyword == "use" && compact_segment.starts_with(':') {
            continue;
        }

        if let Some(grouped_modules) = expand_grouped_elixir_alias(&compact_segment) {
            modules.extend(grouped_modules);
            continue;
        }

        if looks_like_elixir_module(&compact_segment) {
            modules.push(compact_segment);
        }
    }

    modules
}

fn split_top_level(input: &str, delimiter: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut braces = 0i32;
    let mut brackets = 0i32;
    let mut parens = 0i32;

    for ch in input.chars() {
        match ch {
            '{' => braces += 1,
            '}' => braces -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            _ => {}
        }

        if ch == delimiter && braces == 0 && brackets == 0 && parens == 0 {
            if !current.trim().is_empty() {
                parts.push(current.trim().to_string());
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }

    parts
}

fn expand_grouped_elixir_alias(module: &str) -> Option<Vec<String>> {
    let group_start = module.find(".{")?;
    if !module.ends_with('}') {
        return None;
    }

    let prefix = &module[..group_start];
    if !looks_like_elixir_module(prefix) {
        return None;
    }

    let inner = &module[group_start + 2..module.len() - 1];
    let grouped = split_top_level(inner, ',')
        .into_iter()
        .map(|item| compact_elixir_expression(&item))
        .filter(|item| looks_like_elixir_module(item))
        .map(|item| format!("{}.{}", prefix, item))
        .collect::<Vec<_>>();

    Some(grouped)
}

fn looks_like_elixir_option(segment: &str) -> bool {
    let Some(colon_pos) = segment.find(':') else {
        return false;
    };

    if colon_pos == 0 {
        return true;
    }

    segment[..colon_pos]
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch == '_' || ch == '?' || ch == '!')
}

fn looks_like_elixir_module(module: &str) -> bool {
    if module.is_empty() || module.starts_with(':') {
        return false;
    }

    module
        .chars()
        .next()
        .map(|ch| ch.is_ascii_uppercase())
        .unwrap_or(false)
        || module.starts_with("__MODULE__")
}

fn compact_elixir_expression(input: &str) -> String {
    input.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn detect_heredoc_start(line: &str) -> Option<&'static str> {
    ["\"\"\"", "'''"]
        .into_iter()
        .find(|delimiter| line.matches(delimiter).count() % 2 == 1)
}

fn is_elixir_import_statement_start(line: &str) -> bool {
    ["alias", "import", "require", "use"].iter().any(|keyword| {
        line.starts_with(keyword)
            && line
                .chars()
                .nth(keyword.len())
                .map(|ch| ch.is_whitespace())
                .unwrap_or(false)
    })
}

fn strip_inline_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or("").trim_end()
}

fn brace_delta(line: &str) -> i32 {
    line.chars().fold(0i32, |acc, ch| match ch {
        '{' => acc + 1,
        '}' => acc - 1,
        _ => acc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_imports_for_diff_elixir() {
        let content = r#"
alias MyApp.Context
import MyApp.Helpers
require Logger
use MyAppWeb, :controller
alias MyApp.{Repo, Accounts.User}, as: Context
"#;

        let imports = extract_imports_for_diff(content, &Language::Elixir);
        assert!(imports.contains(&"MyApp.Context".to_string()));
        assert!(imports.contains(&"MyApp.Helpers".to_string()));
        assert!(imports.contains(&"Logger".to_string()));
        assert!(imports.contains(&"MyAppWeb".to_string()));
        assert!(imports.contains(&"MyApp.Repo".to_string()));
        assert!(imports.contains(&"MyApp.Accounts.User".to_string()));
    }

    #[test]
    fn test_elixir_fallback_ignores_non_code_import_text() {
        let content = r#"
# alias Fake.Comment
text = "import Fake.String"
heredoc = """
require Fake.Heredoc
"""
sigil = ~S"""
use Fake.SigilHeredoc
"""
alias Real.Module
"#;

        let mut imports = HashSet::new();
        extract_elixir_imports_fallback(content, &mut imports);

        assert!(imports.contains("Real.Module"));
        assert!(!imports.contains("Fake.Comment"));
        assert!(!imports.contains("Fake.String"));
        assert!(!imports.contains("Fake.Heredoc"));
        assert!(!imports.contains("Fake.SigilHeredoc"));
    }
}

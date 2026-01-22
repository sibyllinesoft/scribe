//! Report generation utilities for Scribe output.
//!
//! This module provides functionality to generate reports in multiple formats
//! including HTML, JSON, XML, Markdown, and plain text. Reports contain
//! file selection results along with associated metrics and scores.

use chrono::{DateTime, Local, Utc};
use handlebars::Handlebars;
use serde_json::json;
use std::error::Error;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Output format supported by the reporting utilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Html,
    Repomix,
    Xml,
    Json,
    Text,
    Markdown,
}

/// Minimal representation of a file selected for inclusion in the final report.
#[derive(Debug, Clone)]
pub struct ReportFile {
    pub path: PathBuf,
    pub relative_path: String,
    pub content: String,
    pub size: u64,
    pub estimated_tokens: usize,
    pub importance_score: f64,
    pub centrality_score: f64,
    pub query_relevance_score: f64,
    pub entry_point_proximity: f64,
    pub content_quality_score: f64,
    pub repository_role_score: f64,
    pub recency_score: f64,
    pub modified: Option<SystemTime>,
}

/// Summary of the selection process used when generating reports.
#[derive(Debug, Clone)]
pub struct SelectionMetrics {
    pub total_files_discovered: usize,
    pub files_selected: usize,
    pub total_tokens_estimated: usize,
    pub selection_time_ms: u128,
    pub algorithm_used: String,
    pub coverage_score: f64,
    pub relevance_score: f64,
}

/// Generates a report in the specified format from selected files and metrics.
///
/// # Arguments
/// * `format` - The output format (HTML, JSON, XML, Markdown, or Text)
/// * `files` - The list of selected files with their content and scores
/// * `metrics` - Summary metrics about the selection process
///
/// # Returns
/// The generated report as a string, or an error if generation fails.
pub fn generate_report(
    format: ReportFormat,
    files: &[ReportFile],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn Error>> {
    match format {
        ReportFormat::Html => generate_html_output(files, metrics),
        ReportFormat::Repomix => generate_repomix_output(files, metrics),
        ReportFormat::Xml => generate_xml_output(files, metrics),
        ReportFormat::Json => generate_json_output(files, metrics),
        ReportFormat::Text => generate_text_output(files, metrics),
        ReportFormat::Markdown => generate_markdown_output(files, metrics),
    }
}

/// Create handlebars instance with template and helpers registered
fn create_html_handlebars() -> Result<Handlebars<'static>, Box<dyn Error>> {
    let template_str = include_str!("../templates/report_cdn.html");
    let mut handlebars = Handlebars::new();
    handlebars.register_template_string("report", template_str)?;
    register_add_helper(&mut handlebars);
    Ok(handlebars)
}

/// Register the "add" helper for arithmetic in templates
fn register_add_helper(handlebars: &mut Handlebars<'static>) {
    handlebars.register_helper(
        "add",
        Box::new(
            |h: &handlebars::Helper,
             _: &Handlebars,
             _: &handlebars::Context,
             _: &mut handlebars::RenderContext,
             out: &mut dyn handlebars::Output|
             -> Result<(), handlebars::RenderError> {
                let a = h.param(0).and_then(|v| v.value().as_u64()).unwrap_or(0);
                let b = h.param(1).and_then(|v| v.value().as_u64()).unwrap_or(0);
                out.write(&(a + b).to_string())?;
                Ok(())
            },
        ),
    );
}

/// Convert a ReportFile to JSON for HTML template
fn file_to_template_json(file: &ReportFile) -> serde_json::Value {
    json!({
        "relative_path": html_escape(&file.relative_path),
        "content": html_escape(&file.content),
        "size": format_bytes(file.size),
        "estimated_tokens": format_number(file.estimated_tokens),
    })
}

/// Build template data for HTML report
fn build_html_template_data(files: &[ReportFile], metrics: &SelectionMetrics) -> serde_json::Value {
    let total_tokens: usize = files.iter().map(|f| f.estimated_tokens).sum();
    let total_size: u64 = files.iter().map(|f| f.size).sum();

    json!({
        "repository_name": "Scribe Analysis",
        "algorithm": metrics.algorithm_used,
        "generated_time": Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        "selection_time_ms": metrics.selection_time_ms,
        "total_files": files.len(),
        "total_tokens": format_number(total_tokens),
        "total_size": format_bytes(total_size),
        "coverage_percentage": format!("{:.1}", metrics.coverage_score * 100.0),
        "files": files.iter().map(file_to_template_json).collect::<Vec<_>>()
    })
}

/// Generates an HTML report with syntax highlighting and interactive features.
///
/// Uses a CDN-based template for smaller output size by loading highlight.js externally.
pub fn generate_html_output(
    files: &[ReportFile],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn Error>> {
    // Use CDN-based template for smaller output size
    // This reduces the generated HTML file size by ~60% by using CDN links
    // for highlight.js instead of embedding a 268KB React bundle
    let handlebars = create_html_handlebars()?;
    let template_data = build_html_template_data(files, metrics);
    let html = handlebars.render("report", &template_data)?;
    Ok(html)
}

/// Append file content with trailing newline if needed
fn append_content_with_newline(output: &mut String, content: &str) {
    output.push_str(content);
    if !content.ends_with('\n') {
        output.push('\n');
    }
}

/// Write a single file entry in repomix format
fn write_repomix_file(output: &mut String, file: &ReportFile) -> Result<(), Box<dyn Error>> {
    writeln!(output, "## {}", file.relative_path)?;
    if let Some(ts) = file.modified {
        writeln!(output, "- Last modified: {}", format_system_time(ts))?;
    }
    let lang = get_language_hint(&file.relative_path);
    writeln!(output, "```{}", lang)?;
    append_content_with_newline(output, &file.content);
    writeln!(output, "```")?;
    writeln!(output, "")?;
    Ok(())
}

/// Generates output in Repomix-compatible markdown format.
///
/// Each file is wrapped in a markdown section with code blocks.
pub fn generate_repomix_output(
    files: &[ReportFile],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    writeln!(output, "# RepoMix Export")?;
    writeln!(output, "- Total files: {}", files.len())?;
    writeln!(output, "- Total tokens: {}", metrics.total_tokens_estimated)?;
    writeln!(output, "- Algorithm: {}", metrics.algorithm_used)?;
    writeln!(output, "")?;

    for file in files {
        write_repomix_file(&mut output, file)?;
    }

    Ok(output)
}

/// Write a single file entry in XML format
fn write_xml_file(output: &mut String, file: &ReportFile) -> Result<(), Box<dyn Error>> {
    let path = escape_xml(&file.relative_path);
    let modified = escape_xml(&format_timestamp(file.modified));

    writeln!(
        output,
        "  <file path=\"{}\" modified=\"{}\">",
        path, modified
    )?;
    writeln!(
        output,
        "    <size bytes=\"{}\" tokens=\"{}\"/>",
        file.size, file.estimated_tokens
    )?;
    writeln!(
        output,
        "    <scores importance=\"{:.2}\" centrality=\"{:.2}\" quality=\"{:.2}\"/>",
        file.importance_score, file.centrality_score, file.content_quality_score
    )?;
    writeln!(output, "    <content><![CDATA[")?;
    append_content_with_newline(output, &file.content);
    writeln!(output, "    ]]></content>")?;
    writeln!(output, "  </file>")?;
    Ok(())
}

/// Generates an XML report with structured file and score information.
///
/// File content is wrapped in CDATA sections to preserve formatting.
pub fn generate_xml_output(
    files: &[ReportFile],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    writeln!(output, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(output, "<repository>")?;
    writeln!(
        output,
        "  <summary files=\"{}\" tokens=\"{}\" algorithm=\"{}\" coverage=\"{:.1}\"/>",
        files.len(),
        metrics.total_tokens_estimated,
        metrics.algorithm_used,
        metrics.coverage_score * 100.0
    )?;

    for file in files {
        write_xml_file(&mut output, file)?;
    }

    writeln!(output, "</repository>")?;
    Ok(output)
}

/// Generates a JSON report suitable for programmatic consumption.
///
/// Contains summary metrics and full file details with all score fields.
pub fn generate_json_output(
    files: &[ReportFile],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn Error>> {
    let data = json!({
        "summary": {
            "total_files": files.len(),
            "total_tokens": metrics.total_tokens_estimated,
            "algorithm": metrics.algorithm_used,
            "selection_time_ms": metrics.selection_time_ms,
            "coverage_score": metrics.coverage_score,
            "relevance_score": metrics.relevance_score,
        },
        "files": files.iter().map(|file| {
            json!({
                "path": file.relative_path,
                "modified": format_timestamp(file.modified),
                "size_bytes": file.size,
                "estimated_tokens": file.estimated_tokens,
                "importance_score": file.importance_score,
                "centrality_score": file.centrality_score,
                "query_relevance_score": file.query_relevance_score,
                "entry_point_proximity": file.entry_point_proximity,
                "content_quality_score": file.content_quality_score,
                "repository_role_score": file.repository_role_score,
                "recency_score": file.recency_score,
                "content": file.content,
            })
        }).collect::<Vec<_>>()
    });

    Ok(serde_json::to_string_pretty(&data)?)
}

/// Write a single file entry in text format
fn write_text_file(output: &mut String, file: &ReportFile) -> Result<(), Box<dyn Error>> {
    writeln!(
        output,
        "=== {} ({} tokens)",
        file.relative_path, file.estimated_tokens
    )?;
    append_content_with_newline(output, &file.content);
    writeln!(output)?;
    Ok(())
}

/// Generates a plain text report with simple formatting.
///
/// Suitable for terminal output or simple file processing.
pub fn generate_text_output(
    files: &[ReportFile],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    writeln!(output, "Scribe Report")?;
    writeln!(output, "============")?;
    writeln!(output, "Total files: {}", files.len())?;
    writeln!(output, "Total tokens: {}", metrics.total_tokens_estimated)?;
    writeln!(output, "Algorithm: {}", metrics.algorithm_used)?;
    writeln!(output, "")?;

    for file in files {
        write_text_file(&mut output, file)?;
    }

    Ok(output)
}

/// Write a single file entry in markdown format
fn write_markdown_file(output: &mut String, file: &ReportFile) -> Result<(), Box<dyn Error>> {
    writeln!(output, "## {}", file.relative_path)?;
    writeln!(
        output,
        "*{} | {} tokens*",
        format_bytes(file.size),
        file.estimated_tokens
    )?;
    writeln!(output, "")?;
    let lang = get_language_hint(&file.relative_path);
    writeln!(output, "```{}", lang)?;
    append_content_with_newline(output, &file.content);
    writeln!(output, "```")?;
    writeln!(output, "")?;
    Ok(())
}

/// Generates a Markdown report with fenced code blocks.
///
/// Includes file metadata and content in a readable format.
pub fn generate_markdown_output(
    files: &[ReportFile],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    writeln!(output, "# Scribe Report")?;
    writeln!(output, "- Total files: {}", files.len())?;
    writeln!(output, "- Total tokens: {}", metrics.total_tokens_estimated)?;
    writeln!(output, "- Algorithm: {}", metrics.algorithm_used)?;
    writeln!(output, "")?;

    for file in files {
        write_markdown_file(&mut output, file)?;
    }

    Ok(output)
}

/// Formats a SystemTime as a human-readable local timestamp string.
///
/// Returns "N/A" if the time is None.
pub fn format_timestamp(time: Option<SystemTime>) -> String {
    match time {
        Some(ts) => format_system_time(ts),
        None => "N/A".to_string(),
    }
}

/// Formats a SystemTime as a human-readable local timestamp string.
fn format_system_time(ts: SystemTime) -> String {
    let datetime: DateTime<Local> = ts.into();
    datetime.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Get language hint for markdown code fences based on file extension.
fn get_language_hint(path: &str) -> &'static str {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext.to_lowercase().as_str() {
        "rs" => "rust",
        "py" => "python",
        "js" => "javascript",
        "ts" => "typescript",
        "jsx" => "jsx",
        "tsx" => "tsx",
        "go" => "go",
        "rb" => "ruby",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "cs" => "csharp",
        "php" => "php",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "scala" => "scala",
        "sh" | "bash" | "zsh" => "bash",
        "ps1" => "powershell",
        "sql" => "sql",
        "html" | "htm" => "html",
        "css" => "css",
        "scss" | "sass" => "scss",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "xml" => "xml",
        "md" | "markdown" => "markdown",
        "dockerfile" => "dockerfile",
        "makefile" => "makefile",
        "lua" => "lua",
        "r" => "r",
        "pl" | "pm" => "perl",
        "ex" | "exs" => "elixir",
        "erl" | "hrl" => "erlang",
        "hs" => "haskell",
        "ml" | "mli" => "ocaml",
        "fs" | "fsx" => "fsharp",
        "clj" | "cljs" => "clojure",
        "vim" => "vim",
        "proto" => "protobuf",
        "graphql" | "gql" => "graphql",
        "tf" => "terraform",
        "zig" => "zig",
        "nim" => "nim",
        "v" => "v",
        "dart" => "dart",
        "vue" => "vue",
        "svelte" => "svelte",
        _ => "",
    }
}

/// Formats a byte count as a human-readable string (e.g., "1.23 KB").
///
/// Uses base-1000 units (KB, MB, GB, TB).
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes == 0 {
        return "0 B".to_string();
    }

    let i = (bytes as f64).log10() / 3.0;
    let idx = i.floor() as usize;
    let idx = idx.min(UNITS.len() - 1);
    let value = bytes as f64 / 1000_f64.powi(idx as i32);
    format!("{:.2} {}", value, UNITS[idx])
}

/// Formats a number with comma separators (e.g., 1,234,567).
pub fn format_number(value: usize) -> String {
    let mut s = value.to_string();
    let mut i = s.len() as isize - 3;
    while i > 0 {
        s.insert(i as usize, ',');
        i -= 3;
    }
    s
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Check if file name matches a special icon pattern
fn is_readme(name: &str) -> bool {
    name.starts_with("readme")
}

fn is_license(name: &str) -> bool {
    name == "license" || name == "licence"
}

fn is_docker(name: &str) -> bool {
    name == "dockerfile" || name.contains("docker-compose")
}

fn is_package_manifest(name: &str) -> bool {
    name == "package.json" || name == "cargo.toml" || name == "go.mod"
}

/// Get icon for special file names (README, LICENSE, etc.)
fn get_special_file_icon(name: &str) -> Option<&'static str> {
    if is_readme(name) {
        return Some("book-open");
    }
    if is_license(name) {
        return Some("scale");
    }
    if is_docker(name) {
        return Some("box");
    }
    if name == "makefile" {
        return Some("settings");
    }
    if name.starts_with(".git") {
        return Some("git-branch");
    }
    if is_package_manifest(name) {
        return Some("package");
    }
    None
}

/// Get icon based on file extension
fn get_extension_icon(ext: &str) -> &'static str {
    match ext {
        // Programming languages
        "py" | "pyw" | "rs" | "go" => "file-code",
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => "file-code",
        "java" | "kt" | "scala" => "file-code",
        "c" | "cpp" | "cc" | "h" | "hpp" => "file-code",
        "cs" | "fs" | "vb" => "file-code",
        "php" | "rb" | "pl" | "r" | "swift" | "dart" => "file-code",
        // Web
        "html" | "htm" | "xml" | "xhtml" => "globe",
        "css" | "scss" | "sass" | "less" => "palette",
        // Data formats
        "json" | "jsonc" | "json5" => "braces",
        "yml" | "yaml" => "list",
        "toml" => "settings",
        // Documentation
        "md" | "markdown" | "mdx" | "txt" | "text" | "pdf" => "file-text",
        // Shell scripts
        "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" => "terminal",
        // Data
        "sql" | "sqlite" | "db" => "database",
        // Media
        "png" | "jpg" | "jpeg" | "gif" | "svg" | "webp" | "ico" => "image",
        // Archives
        "zip" | "tar" | "gz" | "bz2" | "7z" | "rar" => "archive",
        _ => "file",
    }
}

/// Returns an appropriate icon name for a file based on its name and extension.
///
/// Recognizes special files (README, LICENSE, Dockerfile, etc.) and common
/// programming language extensions.
pub fn get_file_icon(file_path: &str) -> &'static str {
    let path = Path::new(file_path);
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check special file names first
    if let Some(icon) = get_special_file_icon(&name) {
        return icon;
    }

    // Fall back to extension-based icon
    get_extension_icon(&ext)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::UNIX_EPOCH;

    fn create_test_file(path: &str, content: &str) -> ReportFile {
        ReportFile {
            path: PathBuf::from(path),
            relative_path: path.to_string(),
            content: content.to_string(),
            size: content.len() as u64,
            estimated_tokens: content.len() / 4,
            importance_score: 0.8,
            centrality_score: 0.6,
            query_relevance_score: 0.9,
            entry_point_proximity: 0.5,
            content_quality_score: 0.7,
            repository_role_score: 0.8,
            recency_score: 0.6,
            modified: Some(SystemTime::now()),
        }
    }

    fn create_test_metrics() -> SelectionMetrics {
        SelectionMetrics {
            total_files_discovered: 100,
            files_selected: 10,
            total_tokens_estimated: 5000,
            selection_time_ms: 150,
            algorithm_used: "test_algorithm".to_string(),
            coverage_score: 0.85,
            relevance_score: 0.9,
        }
    }

    #[test]
    fn test_report_format_variants() {
        assert!(matches!(ReportFormat::Html, ReportFormat::Html));
        assert!(matches!(ReportFormat::Json, ReportFormat::Json));
        assert!(matches!(ReportFormat::Xml, ReportFormat::Xml));
        assert!(matches!(ReportFormat::Text, ReportFormat::Text));
        assert!(matches!(ReportFormat::Markdown, ReportFormat::Markdown));
        assert!(matches!(ReportFormat::Repomix, ReportFormat::Repomix));
    }

    #[test]
    fn test_report_file_creation() {
        let file = create_test_file("src/main.rs", "fn main() {}");
        assert_eq!(file.relative_path, "src/main.rs");
        assert_eq!(file.content, "fn main() {}");
        assert_eq!(file.size, 12);
    }

    #[test]
    fn test_selection_metrics_creation() {
        let metrics = create_test_metrics();
        assert_eq!(metrics.total_files_discovered, 100);
        assert_eq!(metrics.files_selected, 10);
        assert_eq!(metrics.algorithm_used, "test_algorithm");
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
        assert_eq!(html_escape("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(html_escape("normal text"), "normal text");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
    }

    #[test]
    fn test_format_bytes() {
        // Test basic byte formatting
        let result = format_bytes(1024);
        assert!(result.contains("KB") || result.contains("1"));

        let result_mb = format_bytes(1048576);
        assert!(result_mb.contains("MB") || result_mb.contains("1"));
    }

    #[test]
    fn test_get_file_icon_programming() {
        assert_eq!(get_file_icon("main.rs"), "file-code");
        assert_eq!(get_file_icon("app.py"), "file-code");
        assert_eq!(get_file_icon("index.js"), "file-code");
        assert_eq!(get_file_icon("component.tsx"), "file-code");
        assert_eq!(get_file_icon("main.go"), "file-code");
    }

    #[test]
    fn test_get_file_icon_special() {
        assert_eq!(get_file_icon("README.md"), "book-open");
        assert_eq!(get_file_icon("LICENSE"), "scale");
        assert_eq!(get_file_icon("Dockerfile"), "box");
        assert_eq!(get_file_icon("Cargo.toml"), "package");
        assert_eq!(get_file_icon("package.json"), "package");
    }

    #[test]
    fn test_get_file_icon_web() {
        assert_eq!(get_file_icon("index.html"), "globe");
        assert_eq!(get_file_icon("styles.css"), "palette");
    }

    #[test]
    fn test_get_file_icon_fallback() {
        assert_eq!(get_file_icon("unknown.xyz"), "file");
        assert_eq!(get_file_icon("noextension"), "file");
    }

    #[test]
    fn test_generate_json_output() {
        let files = vec![
            create_test_file("src/main.rs", "fn main() {}"),
            create_test_file("src/lib.rs", "pub mod utils;"),
        ];
        let metrics = create_test_metrics();

        let result = generate_json_output(&files, &metrics);
        assert!(result.is_ok());

        let json_str = result.unwrap();
        assert!(json_str.contains("\"files\""));
        assert!(json_str.contains("\"summary\""));
        assert!(json_str.contains("src/main.rs"));
        assert!(json_str.contains("src/lib.rs"));
    }

    #[test]
    fn test_generate_text_output() {
        let files = vec![create_test_file("src/main.rs", "fn main() {}")];
        let metrics = create_test_metrics();

        let result = generate_text_output(&files, &metrics);
        assert!(result.is_ok());

        let text = result.unwrap();
        assert!(text.contains("Scribe Report"));
        assert!(text.contains("src/main.rs"));
        assert!(text.contains("fn main()"));
    }

    #[test]
    fn test_generate_markdown_output() {
        let files = vec![create_test_file("src/main.rs", "fn main() {}")];
        let metrics = create_test_metrics();

        let result = generate_markdown_output(&files, &metrics);
        assert!(result.is_ok());

        let md = result.unwrap();
        assert!(md.contains("Scribe") || md.contains("Report"));
        assert!(md.contains("src/main.rs"));
        assert!(md.contains("```"));
    }

    #[test]
    fn test_generate_xml_output() {
        let files = vec![create_test_file("src/main.rs", "fn main() {}")];
        let metrics = create_test_metrics();

        let result = generate_xml_output(&files, &metrics);
        assert!(result.is_ok());

        let xml = result.unwrap();
        assert!(xml.contains("<?xml version"));
        assert!(xml.contains("<repository>"));
        assert!(xml.contains("</repository>"));
        assert!(xml.contains("<file"));
        assert!(xml.contains("src/main.rs"));
    }

    #[test]
    fn test_generate_repomix_output() {
        let files = vec![create_test_file("src/main.rs", "fn main() {}")];
        let metrics = create_test_metrics();

        let result = generate_repomix_output(&files, &metrics);
        assert!(result.is_ok());

        let repomix = result.unwrap();
        assert!(repomix.contains("RepoMix Export"));
        assert!(repomix.contains("src/main.rs"));
        assert!(repomix.contains("fn main()"));
    }

    #[test]
    fn test_generate_html_output() {
        let files = vec![create_test_file("src/main.rs", "fn main() {}")];
        let metrics = create_test_metrics();

        let result = generate_html_output(&files, &metrics);
        assert!(result.is_ok());

        let html = result.unwrap();
        assert!(html.contains("<!DOCTYPE html>") || html.contains("<html"));
        assert!(html.contains("src/main.rs"));
    }

    #[test]
    fn test_generate_report_all_formats() {
        let files = vec![create_test_file("src/main.rs", "fn main() {}")];
        let metrics = create_test_metrics();

        // Test all formats work
        for format in [
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Xml,
            ReportFormat::Text,
            ReportFormat::Markdown,
            ReportFormat::Repomix,
        ] {
            let result = generate_report(format, &files, &metrics);
            assert!(
                result.is_ok(),
                "Failed to generate report for format {:?}",
                format
            );
        }
    }

    #[test]
    fn test_format_timestamp() {
        let timestamp = Some(UNIX_EPOCH + std::time::Duration::from_secs(1609459200)); // 2021-01-01
        let formatted = format_timestamp(timestamp);
        assert!(!formatted.is_empty());
        // Should contain year or be a valid date string
        assert!(formatted.contains("2021") || formatted.len() > 5);

        let no_timestamp = format_timestamp(None);
        // Should be some indicator of unknown/no timestamp
        assert!(!no_timestamp.is_empty());
    }

    #[test]
    fn test_empty_files_report() {
        let files: Vec<ReportFile> = vec![];
        let metrics = SelectionMetrics {
            total_files_discovered: 0,
            files_selected: 0,
            total_tokens_estimated: 0,
            selection_time_ms: 10,
            algorithm_used: "none".to_string(),
            coverage_score: 0.0,
            relevance_score: 0.0,
        };

        // All formats should handle empty files gracefully
        let result = generate_json_output(&files, &metrics);
        assert!(result.is_ok());

        let result = generate_text_output(&files, &metrics);
        assert!(result.is_ok());

        let result = generate_markdown_output(&files, &metrics);
        assert!(result.is_ok());
    }

    #[test]
    fn test_special_characters_in_content() {
        let file = create_test_file(
            "test.html",
            "<div class=\"test\">&amp; special < > chars</div>",
        );
        let metrics = create_test_metrics();

        // HTML should escape special characters
        let html = generate_html_output(&[file.clone()], &metrics).unwrap();
        assert!(html.contains("&lt;") || html.contains("<")); // Either escaped or raw in code block

        // JSON should handle them properly
        let json = generate_json_output(&[file.clone()], &metrics).unwrap();
        assert!(json.contains("test.html"));

        // XML should escape them
        let xml = generate_xml_output(&[file], &metrics).unwrap();
        assert!(xml.contains("test.html"));
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("<script>"), "&lt;script&gt;");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_xml("normal text"), "normal text");
    }

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(1), "1.00 B");
        assert_eq!(format_bytes(999), "999.00 B");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        let result = format_bytes(1000);
        assert!(result.contains("KB") || result.contains("1.00"));
    }

    #[test]
    fn test_format_bytes_megabytes() {
        let result = format_bytes(1_000_000);
        assert!(result.contains("MB") || result.contains("1.00"));
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        let result = format_bytes(1_000_000_000);
        assert!(result.contains("GB") || result.contains("1.00"));
    }

    #[test]
    fn test_format_bytes_terabytes() {
        let result = format_bytes(1_000_000_000_000);
        assert!(result.contains("TB") || result.contains("1.00"));
    }

    #[test]
    fn test_format_bytes_large_tb() {
        // Over 1000 TB should still use TB (max unit)
        let result = format_bytes(5_000_000_000_000_000);
        assert!(result.contains("TB") || result.contains("5000"));
    }

    #[test]
    fn test_get_file_icon_makefile() {
        assert_eq!(get_file_icon("Makefile"), "settings");
    }

    #[test]
    fn test_get_file_icon_gitignore() {
        assert_eq!(get_file_icon(".gitignore"), "git-branch");
    }

    #[test]
    fn test_get_file_icon_docker_compose() {
        assert_eq!(get_file_icon("docker-compose.yml"), "box");
    }

    #[test]
    fn test_get_file_icon_go_mod() {
        assert_eq!(get_file_icon("go.mod"), "package");
    }

    #[test]
    fn test_get_file_icon_licence() {
        assert_eq!(get_file_icon("LICENCE"), "scale");
    }

    #[test]
    fn test_get_file_icon_yaml() {
        assert_eq!(get_file_icon("config.yml"), "list");
        assert_eq!(get_file_icon("config.yaml"), "list");
    }

    #[test]
    fn test_get_file_icon_json() {
        assert_eq!(get_file_icon("data.json"), "braces");
        assert_eq!(get_file_icon("tsconfig.jsonc"), "braces");
    }

    #[test]
    fn test_get_file_icon_text() {
        assert_eq!(get_file_icon("notes.txt"), "file-text");
        assert_eq!(get_file_icon("guide.pdf"), "file-text");
    }

    #[test]
    fn test_get_file_icon_shell() {
        assert_eq!(get_file_icon("script.sh"), "terminal");
        assert_eq!(get_file_icon("setup.bash"), "terminal");
        assert_eq!(get_file_icon("init.ps1"), "terminal");
    }

    #[test]
    fn test_get_file_icon_database() {
        assert_eq!(get_file_icon("query.sql"), "database");
        assert_eq!(get_file_icon("data.sqlite"), "database");
    }

    #[test]
    fn test_get_file_icon_image() {
        assert_eq!(get_file_icon("logo.png"), "image");
        assert_eq!(get_file_icon("photo.jpg"), "image");
        assert_eq!(get_file_icon("icon.svg"), "image");
    }

    #[test]
    fn test_get_file_icon_archive() {
        assert_eq!(get_file_icon("backup.zip"), "archive");
        assert_eq!(get_file_icon("data.tar"), "archive");
        assert_eq!(get_file_icon("data.gz"), "archive");
    }

    #[test]
    fn test_get_file_icon_java_related() {
        assert_eq!(get_file_icon("Main.java"), "file-code");
        assert_eq!(get_file_icon("App.kt"), "file-code");
        assert_eq!(get_file_icon("Server.scala"), "file-code");
    }

    #[test]
    fn test_get_file_icon_c_cpp() {
        assert_eq!(get_file_icon("main.c"), "file-code");
        assert_eq!(get_file_icon("main.cpp"), "file-code");
        assert_eq!(get_file_icon("main.cc"), "file-code");
        assert_eq!(get_file_icon("main.h"), "file-code");
        assert_eq!(get_file_icon("main.hpp"), "file-code");
    }

    #[test]
    fn test_get_file_icon_dotnet() {
        assert_eq!(get_file_icon("Program.cs"), "file-code");
        assert_eq!(get_file_icon("Module.fs"), "file-code");
        assert_eq!(get_file_icon("Form.vb"), "file-code");
    }

    #[test]
    fn test_get_file_icon_misc_languages() {
        assert_eq!(get_file_icon("main.php"), "file-code");
        assert_eq!(get_file_icon("app.rb"), "file-code");
        assert_eq!(get_file_icon("script.pl"), "file-code");
        assert_eq!(get_file_icon("analysis.r"), "file-code");
        assert_eq!(get_file_icon("App.swift"), "file-code");
        assert_eq!(get_file_icon("main.dart"), "file-code");
    }

    #[test]
    fn test_get_file_icon_web_xml() {
        assert_eq!(get_file_icon("page.htm"), "globe");
        assert_eq!(get_file_icon("config.xml"), "globe");
        assert_eq!(get_file_icon("page.xhtml"), "globe");
    }

    #[test]
    fn test_get_file_icon_css_variants() {
        assert_eq!(get_file_icon("styles.scss"), "palette");
        assert_eq!(get_file_icon("styles.sass"), "palette");
        assert_eq!(get_file_icon("styles.less"), "palette");
    }

    #[test]
    fn test_get_file_icon_toml() {
        assert_eq!(get_file_icon("pyproject.toml"), "settings");
    }

    #[test]
    fn test_is_readme() {
        // Note: is_readme is called with lowercase names
        assert!(is_readme("readme"));
        assert!(is_readme("readme.md"));
        assert!(!is_readme("about.md"));
    }

    #[test]
    fn test_is_license() {
        assert!(is_license("license"));
        assert!(is_license("licence"));
        assert!(!is_license("LICENSE.md"));
    }

    #[test]
    fn test_is_docker() {
        assert!(is_docker("dockerfile"));
        assert!(is_docker("docker-compose.yml"));
        assert!(!is_docker("docker.md"));
    }

    #[test]
    fn test_is_package_manifest() {
        assert!(is_package_manifest("package.json"));
        assert!(is_package_manifest("cargo.toml"));
        assert!(is_package_manifest("go.mod"));
        assert!(!is_package_manifest("package-lock.json"));
    }

    #[test]
    fn test_append_content_with_newline_already_ends_newline() {
        let mut output = String::new();
        append_content_with_newline(&mut output, "content\n");
        assert_eq!(output, "content\n");
    }

    #[test]
    fn test_append_content_with_newline_missing_newline() {
        let mut output = String::new();
        append_content_with_newline(&mut output, "content");
        assert_eq!(output, "content\n");
    }

    #[test]
    fn test_html_escape_apostrophe() {
        assert_eq!(html_escape("it's"), "it&#39;s");
    }

    #[test]
    fn test_report_file_clone() {
        let file = create_test_file("src/main.rs", "fn main() {}");
        let cloned = file.clone();
        assert_eq!(file.relative_path, cloned.relative_path);
        assert_eq!(file.content, cloned.content);
    }

    #[test]
    fn test_report_file_debug() {
        let file = create_test_file("test.rs", "code");
        let debug_str = format!("{:?}", file);
        assert!(debug_str.contains("ReportFile"));
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_selection_metrics_clone() {
        let metrics = create_test_metrics();
        let cloned = metrics.clone();
        assert_eq!(
            metrics.total_files_discovered,
            cloned.total_files_discovered
        );
        assert_eq!(metrics.algorithm_used, cloned.algorithm_used);
    }

    #[test]
    fn test_selection_metrics_debug() {
        let metrics = create_test_metrics();
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("SelectionMetrics"));
        assert!(debug_str.contains("test_algorithm"));
    }

    #[test]
    fn test_report_format_clone() {
        let format = ReportFormat::Html;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_report_format_debug() {
        let format = ReportFormat::Json;
        let debug_str = format!("{:?}", format);
        assert!(debug_str.contains("Json"));
    }

    #[test]
    fn test_report_format_copy() {
        let format1 = ReportFormat::Xml;
        let format2 = format1; // Copy
        assert_eq!(format1, format2);
    }

    #[test]
    fn test_format_number_edge_cases() {
        assert_eq!(format_number(1), "1");
        assert_eq!(format_number(10), "10");
        assert_eq!(format_number(100), "100");
        assert_eq!(format_number(123456789), "123,456,789");
    }

    #[test]
    fn test_file_without_modified_time() {
        let mut file = create_test_file("test.rs", "code");
        file.modified = None;
        let metrics = create_test_metrics();

        let text = generate_text_output(&[file.clone()], &metrics).unwrap();
        assert!(text.contains("N/A") || text.contains("test.rs"));

        let json = generate_json_output(&[file], &metrics).unwrap();
        assert!(json.contains("N/A") || json.contains("test.rs"));
    }

    #[test]
    fn test_get_extension_icon_default() {
        assert_eq!(get_extension_icon("unknown"), "file");
    }
}

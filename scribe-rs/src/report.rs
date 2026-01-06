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
        "importance_score": format!("{:.2}", file.importance_score),
        "centrality_score": format!("{:.2}", file.centrality_score),
        "query_relevance_score": format!("{:.2}", file.query_relevance_score),
        "entry_point_proximity": format!("{:.2}", file.entry_point_proximity),
        "content_quality_score": format!("{:.2}", file.content_quality_score),
        "repository_role_score": format!("{:.2}", file.repository_role_score),
        "recency_score": format!("{:.2}", file.recency_score),
        "modified": format_timestamp(file.modified),
        "icon": get_file_icon(&file.relative_path)
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
        writeln!(output, "## {}", file.relative_path)?;
        writeln!(
            output,
            "- Last modified: {}",
            format_timestamp(file.modified)
        )?;
        writeln!(output, "```")?;
        output.push_str(&file.content);
        if !file.content.ends_with('\n') {
            output.push('\n');
        }
        writeln!(output, "```")?;
        writeln!(output, "")?;
    }

    Ok(output)
}

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
        output.push_str(&file.content);
        if !file.content.ends_with('\n') {
            output.push('\n');
        }
        writeln!(output, "    ]]></content>")?;
        writeln!(output, "  </file>")?;
    }

    writeln!(output, "</repository>")?;
    Ok(output)
}

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
        writeln!(
            output,
            "--- {} ({} tokens) ---",
            file.relative_path, file.estimated_tokens
        )?;
        writeln!(output, "Last modified: {}", format_timestamp(file.modified))?;
        output.push_str(&file.content);
        if !file.content.ends_with('\n') {
            output.push('\n');
        }
        writeln!(output)?;
    }

    Ok(output)
}

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
        writeln!(output, "## {}", file.relative_path)?;
        writeln!(output, "- Size: {}", format_bytes(file.size))?;
        writeln!(output, "- Tokens: {}", file.estimated_tokens)?;
        writeln!(output, "- Importance: {:.2}", file.importance_score)?;
        writeln!(output, "- Modified: {}", format_timestamp(file.modified))?;
        writeln!(output, "")?;
        writeln!(output, "```")?;
        output.push_str(&file.content);
        if !file.content.ends_with('\n') {
            output.push('\n');
        }
        writeln!(output, "```")?;
        writeln!(output, "")?;
    }

    Ok(output)
}

pub fn format_timestamp(time: Option<SystemTime>) -> String {
    match time {
        Some(ts) => {
            let datetime: DateTime<Local> = ts.into();
            datetime.format("%Y-%m-%d %H:%M:%S").to_string()
        }
        None => "N/A".to_string(),
    }
}

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

/// Get icon for special file names (README, LICENSE, etc.)
fn get_special_file_icon(name: &str) -> Option<&'static str> {
    if name.starts_with("readme") {
        Some("book-open")
    } else if name == "license" || name == "licence" {
        Some("scale")
    } else if name == "dockerfile" || name.contains("docker-compose") {
        Some("box")
    } else if name == "makefile" {
        Some("settings")
    } else if name.starts_with(".git") {
        Some("git-branch")
    } else if name == "package.json" || name == "cargo.toml" || name == "go.mod" {
        Some("package")
    } else {
        None
    }
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

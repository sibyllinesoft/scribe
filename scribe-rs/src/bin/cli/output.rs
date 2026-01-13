//! Output formatting and report generation utilities

use std::path::{Path, PathBuf};

use scribe::{ReportFile, ReportFormat, SelectionMetrics};

/// Determine output file path from CLI args, config, or auto-generate
pub fn determine_output_path(
    cli_output: Option<&String>,
    config_path: Option<&String>,
    repo_dir: &Path,
    report_format: ReportFormat,
) -> PathBuf {
    if let Some(output) = cli_output {
        return PathBuf::from(output);
    }

    if let Some(config_path) = config_path {
        let path = PathBuf::from(config_path);
        return if path.is_absolute() { path } else { repo_dir.join(path) };
    }

    auto_generate_output_path(repo_dir, report_format)
}

/// Auto-generate output filename based on repository name and format
fn auto_generate_output_path(repo_dir: &Path, format: ReportFormat) -> PathBuf {
    let base_name = repo_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("repository");

    let extension = report_format_extension(format);
    PathBuf::from(format!("{}.{}", base_name, extension))
}

/// Get file extension for a report format
pub fn report_format_extension(format: ReportFormat) -> &'static str {
    match format {
        ReportFormat::Html => "html",
        ReportFormat::Repomix => "repomix",
        ReportFormat::Xml => "xml",
        ReportFormat::Json => "json",
        ReportFormat::Text => "txt",
        ReportFormat::Markdown => "md",
    }
}

/// Get human-readable label for a report format
pub fn report_format_label(format: ReportFormat) -> &'static str {
    match format {
        ReportFormat::Html => "HTML",
        ReportFormat::Repomix => "Repomix",
        ReportFormat::Xml => "XML",
        ReportFormat::Json => "JSON",
        ReportFormat::Text => "Text",
        ReportFormat::Markdown => "Markdown",
    }
}

pub fn apply_line_numbers_to_files(files: &mut [ReportFile]) {
    for file in files {
        file.content = add_line_numbers(&file.content);
    }
}

fn add_line_numbers(content: &str) -> String {
    let lines: Vec<&str> = content.split('\n').collect();
    let width = lines.len().max(1).to_string().len().max(3);

    let mut numbered = String::with_capacity(content.len() + lines.len() * (width + 3));
    for (idx, line) in lines.iter().enumerate() {
        let line_no = idx + 1;
        numbered.push_str(&format!("{:width$} | {}", line_no, line, width = width));
        if idx + 1 < lines.len() {
            numbered.push('\n');
        }
    }

    numbered
}

/// Print selection summary to stdout
pub fn print_selection_summary(
    metrics: &SelectionMetrics,
    eligible_file_count: usize,
    token_target: usize,
    unlimited_budget: bool,
) {
    println!("📊 Selection summary");
    println!("  • Files scanned   : {}", metrics.total_files_discovered);
    println!("  • Eligible files  : {}", eligible_file_count);
    println!(
        "  • Files selected  : {} ({} tokens)",
        metrics.files_selected, metrics.total_tokens_estimated
    );
    println!(
        "  • Files excluded  : {}",
        eligible_file_count.saturating_sub(metrics.files_selected)
    );
    println!("  • Coverage        : {:.1}%", metrics.coverage_score * 100.0);
    if unlimited_budget || token_target == 0 {
        println!("  • Token usage     : unlimited");
    } else {
        println!(
            "  • Token usage     : {} / {}",
            metrics.total_tokens_estimated, token_target
        );
    }
}

//! Helper functions for HTTP handlers

use crate::{AnalysisOutput, BundleState, Result, WebReportFile, WebSelectionMetrics, WebServiceError};
use scribe_core::FileInfo;
use scribe_selection::SelectionResult;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::SystemTime;

/// File entry for API responses
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub size: usize,
    pub tokens: usize,
    pub file_type: String,
    pub included: bool,
}

pub fn recompute_bundle_summary(
    bundle_state: &mut BundleState,
    analysis: &AnalysisOutput,
) -> HashMap<String, Vec<FileEntry>> {
    let repo_lookup: HashMap<&str, &FileInfo> = analysis
        .repository_files
        .iter()
        .map(|info| (info.relative_path.as_str(), info))
        .collect();

    let mut seen = HashSet::new();
    let mut normalized = Vec::new();

    for path in bundle_state.included_files.drain(..) {
        if repo_lookup.contains_key(path.as_str()) && seen.insert(path.clone()) {
            normalized.push(path);
        }
    }

    bundle_state.included_files = normalized;

    let mut total_size: u64 = 0;
    let mut total_tokens: usize = 0;

    for path in &bundle_state.included_files {
        if let Some(info) = repo_lookup.get(path.as_str()) {
            total_size += info.size;
            total_tokens = total_tokens.saturating_add(info.token_estimate.unwrap_or(0));
        }
    }

    let included_entries: Vec<FileEntry> = bundle_state
        .included_files
        .iter()
        .filter_map(|path| repo_lookup.get(path.as_str()).copied())
        .map(|info| file_entry_from_fileinfo(info, true))
        .collect();

    let included_set: HashSet<&str> = bundle_state
        .included_files
        .iter()
        .map(|path| path.as_str())
        .collect();

    let mut excluded_paths = Vec::new();
    let mut excluded_entries = Vec::new();

    for info in &analysis.repository_files {
        if !included_set.contains(info.relative_path.as_str()) {
            excluded_paths.push(info.relative_path.clone());
            excluded_entries.push(file_entry_from_fileinfo(info, false));
        }
    }

    bundle_state.total_size = file_size_to_usize(total_size);
    bundle_state.token_estimate = total_tokens;
    bundle_state.excluded_files = HashMap::from([("excluded".to_string(), excluded_paths)]);
    bundle_state.last_updated = chrono::Utc::now();

    let mut categories = HashMap::new();
    categories.insert("included".to_string(), included_entries);
    categories.insert("excluded".to_string(), excluded_entries);

    categories
}

pub fn file_entry_from_fileinfo(file: &FileInfo, included: bool) -> FileEntry {
    FileEntry {
        path: file.relative_path.clone(),
        size: file_size_to_usize(file.size),
        tokens: file.token_estimate.unwrap_or(0),
        file_type: format!("{:?}", file.language),
        included,
    }
}

pub fn file_entry_from_report_file(
    file: &WebReportFile,
    matching_info: Option<&FileInfo>,
) -> FileEntry {
    let file_type = matching_info
        .map(|info| format!("{:?}", info.language))
        .unwrap_or_else(|| "unknown".to_string());

    FileEntry {
        path: file.relative_path.clone(),
        size: file_size_to_usize(file.size),
        tokens: file.estimated_tokens,
        file_type,
        included: true,
    }
}

pub fn file_size_to_usize(value: u64) -> usize {
    if value > usize::MAX as u64 {
        usize::MAX
    } else {
        value as usize
    }
}

pub fn build_selection_result(analysis: &AnalysisOutput, included_files: &[String]) -> SelectionResult {
    let included_set: HashSet<&str> = included_files.iter().map(|path| path.as_str()).collect();

    let mut selected_infos = Vec::new();
    let mut total_tokens = 0usize;

    for info in &analysis.repository_files {
        if included_set.contains(info.relative_path.as_str()) {
            total_tokens = total_tokens.saturating_add(info.token_estimate.unwrap_or(0));
            selected_infos.push(info.clone());
        }
    }

    let budget = if analysis.token_budget == 0 {
        analysis.metrics.total_tokens_estimated
    } else {
        analysis.token_budget
    };

    let unused_tokens = if budget == 0 {
        0
    } else {
        budget.saturating_sub(total_tokens)
    };

    SelectionResult {
        files: selected_infos,
        total_tokens_used: total_tokens,
        budget,
        unused_tokens,
        total_files_considered: analysis.metrics.total_files_discovered,
    }
}

pub fn build_reports_for_selection(
    analysis: &AnalysisOutput,
    bundle_state: &BundleState,
) -> Result<(Vec<WebReportFile>, WebSelectionMetrics)> {
    let mut reports = Vec::new();
    let mut total_tokens = 0usize;

    for path in &bundle_state.included_files {
        if let Some(existing) = analysis
            .selected_files
            .iter()
            .find(|file| &file.relative_path == path)
        {
            total_tokens = total_tokens.saturating_add(existing.estimated_tokens);
            reports.push(existing.clone());
            continue;
        }

        if let Some(info) = analysis
            .repository_files
            .iter()
            .find(|info| &info.relative_path == path)
        {
            let content = match &info.content {
                Some(cached) => cached.clone(),
                None => std::fs::read_to_string(&info.path).unwrap_or_else(|_| {
                    format!("<unable to read file {}>", info.path.to_string_lossy())
                }),
            };

            let estimated_tokens = info
                .token_estimate
                .unwrap_or_else(|| (content.len() / 4).max(1));

            total_tokens = total_tokens.saturating_add(estimated_tokens);

            reports.push(WebReportFile {
                path: info.path.clone(),
                relative_path: info.relative_path.clone(),
                content,
                size: info.size,
                estimated_tokens,
                importance_score: 0.0,
                centrality_score: info.centrality_score.unwrap_or(0.0),
                query_relevance_score: 0.0,
                entry_point_proximity: 0.0,
                content_quality_score: 0.0,
                repository_role_score: 0.0,
                recency_score: 0.0,
                modified: format_modified(info.modified),
            });
        } else {
            return Err(WebServiceError::FileNotFound { path: path.clone() });
        }
    }

    let mut metrics = analysis.metrics.clone();
    metrics.files_selected = reports.len();
    metrics.total_tokens_estimated = total_tokens;

    if metrics.total_files_discovered == 0 {
        metrics.total_files_discovered = analysis.repository_files.len();
    }

    Ok((reports, metrics))
}

pub fn format_file_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

pub fn get_file_icon(extension: &str) -> String {
    match extension.to_lowercase().as_str() {
        "rs" => "🦀",
        "py" => "🐍",
        "js" | "mjs" | "cjs" => "📜",
        "ts" | "tsx" => "📘",
        "go" => "🐹",
        "java" => "☕",
        "c" | "h" => "⚙️",
        "cpp" | "hpp" | "cc" | "cxx" => "⚙️",
        "md" | "markdown" => "📝",
        "json" => "📋",
        "toml" | "yaml" | "yml" => "⚙️",
        "html" | "htm" => "🌐",
        "css" | "scss" | "sass" => "🎨",
        _ => "📄",
    }
    .to_string()
}

pub fn format_modified(time: Option<SystemTime>) -> String {
    time.and_then(|t| {
        t.duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| {
                chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_default()
            })
    })
    .unwrap_or_default()
}

/// Template data structure for Handlebars rendering
#[derive(Serialize, Clone)]
pub struct TemplateData {
    pub repository_name: String,
    pub algorithm: String,
    pub generated_time: String,
    pub selection_time_ms: u64,
    pub total_files: usize,
    pub total_tokens: String,
    pub total_size: String,
    pub coverage_percentage: u32,
    pub files: Vec<TemplateFile>,
}

#[derive(Serialize, Clone)]
pub struct TemplateFile {
    pub relative_path: String,
    pub icon: String,
    pub size: String,
    pub estimated_tokens: String,
    pub importance_score: String,
    pub content: String,
}

pub fn prepare_template_data(
    repo_path: &Path,
    selected_files: &[WebReportFile],
    metrics: &WebSelectionMetrics,
) -> TemplateData {
    let repo_name = repo_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let total_size: u64 = selected_files.iter().map(|file| file.size).sum();
    let total_tokens: usize = selected_files.iter().map(|file| file.estimated_tokens).sum();

    let files: Vec<TemplateFile> = selected_files
        .iter()
        .map(|file| {
            let extension = file
                .relative_path
                .rsplit('.')
                .next()
                .unwrap_or("");

            TemplateFile {
                relative_path: file.relative_path.clone(),
                icon: get_file_icon(extension),
                size: format_file_size(file.size),
                estimated_tokens: format!("{}", file.estimated_tokens),
                importance_score: format!("{:.2}", file.importance_score),
                content: file.content.clone(),
            }
        })
        .collect();

    let coverage_percentage = if metrics.total_files_discovered > 0 {
        ((selected_files.len() as f64 / metrics.total_files_discovered as f64) * 100.0) as u32
    } else {
        0
    };

    TemplateData {
        repository_name: repo_name,
        algorithm: metrics.algorithm_used.clone(),
        generated_time: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        selection_time_ms: metrics.selection_time_ms as u64,
        total_files: selected_files.len(),
        total_tokens: format!("{}", total_tokens),
        total_size: format_file_size(total_size),
        coverage_percentage,
        files,
    }
}

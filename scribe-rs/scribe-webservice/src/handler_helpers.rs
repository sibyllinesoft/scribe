//! Helper functions for HTTP handlers

use crate::{
    AnalysisOutput, BundleState, Result, WebReportFile, WebSelectionMetrics, WebServiceError,
};
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

pub fn build_selection_result(
    analysis: &AnalysisOutput,
    included_files: &[String],
) -> SelectionResult {
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
        t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| {
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
    let total_tokens: usize = selected_files
        .iter()
        .map(|file| file.estimated_tokens)
        .sum();

    let files: Vec<TemplateFile> = selected_files
        .iter()
        .map(|file| {
            let extension = file.relative_path.rsplit('.').next().unwrap_or("");

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
        generated_time: chrono::Utc::now()
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string(),
        selection_time_ms: metrics.selection_time_ms as u64,
        total_files: selected_files.len(),
        total_tokens: format!("{}", total_tokens),
        total_size: format_file_size(total_size),
        coverage_percentage,
        files,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_file_size_to_usize_normal() {
        assert_eq!(file_size_to_usize(1024), 1024);
        assert_eq!(file_size_to_usize(0), 0);
        assert_eq!(file_size_to_usize(1_000_000), 1_000_000);
    }

    #[test]
    fn test_format_file_size_bytes() {
        assert_eq!(format_file_size(0), "0 bytes");
        assert_eq!(format_file_size(512), "512 bytes");
        assert_eq!(format_file_size(1023), "1023 bytes");
    }

    #[test]
    fn test_format_file_size_kb() {
        let result = format_file_size(1024);
        assert!(result.contains("KB"));
        assert!(result.contains("1.00"));

        let result = format_file_size(2048);
        assert!(result.contains("KB"));
        assert!(result.contains("2.00"));
    }

    #[test]
    fn test_format_file_size_mb() {
        let result = format_file_size(1024 * 1024);
        assert!(result.contains("MB"));
        assert!(result.contains("1.00"));

        let result = format_file_size(5 * 1024 * 1024);
        assert!(result.contains("MB"));
        assert!(result.contains("5.00"));
    }

    #[test]
    fn test_format_file_size_gb() {
        let result = format_file_size(1024 * 1024 * 1024);
        assert!(result.contains("GB"));
        assert!(result.contains("1.00"));
    }

    #[test]
    fn test_get_file_icon_rust() {
        assert_eq!(get_file_icon("rs"), "🦀");
    }

    #[test]
    fn test_get_file_icon_python() {
        assert_eq!(get_file_icon("py"), "🐍");
    }

    #[test]
    fn test_get_file_icon_javascript() {
        assert_eq!(get_file_icon("js"), "📜");
        assert_eq!(get_file_icon("mjs"), "📜");
        assert_eq!(get_file_icon("cjs"), "📜");
    }

    #[test]
    fn test_get_file_icon_typescript() {
        assert_eq!(get_file_icon("ts"), "📘");
        assert_eq!(get_file_icon("tsx"), "📘");
    }

    #[test]
    fn test_get_file_icon_go() {
        assert_eq!(get_file_icon("go"), "🐹");
    }

    #[test]
    fn test_get_file_icon_java() {
        assert_eq!(get_file_icon("java"), "☕");
    }

    #[test]
    fn test_get_file_icon_c() {
        assert_eq!(get_file_icon("c"), "⚙️");
        assert_eq!(get_file_icon("h"), "⚙️");
    }

    #[test]
    fn test_get_file_icon_cpp() {
        assert_eq!(get_file_icon("cpp"), "⚙️");
        assert_eq!(get_file_icon("hpp"), "⚙️");
        assert_eq!(get_file_icon("cc"), "⚙️");
        assert_eq!(get_file_icon("cxx"), "⚙️");
    }

    #[test]
    fn test_get_file_icon_markdown() {
        assert_eq!(get_file_icon("md"), "📝");
        assert_eq!(get_file_icon("markdown"), "📝");
    }

    #[test]
    fn test_get_file_icon_json() {
        assert_eq!(get_file_icon("json"), "📋");
    }

    #[test]
    fn test_get_file_icon_config() {
        assert_eq!(get_file_icon("toml"), "⚙️");
        assert_eq!(get_file_icon("yaml"), "⚙️");
        assert_eq!(get_file_icon("yml"), "⚙️");
    }

    #[test]
    fn test_get_file_icon_web() {
        assert_eq!(get_file_icon("html"), "🌐");
        assert_eq!(get_file_icon("htm"), "🌐");
    }

    #[test]
    fn test_get_file_icon_css() {
        assert_eq!(get_file_icon("css"), "🎨");
        assert_eq!(get_file_icon("scss"), "🎨");
        assert_eq!(get_file_icon("sass"), "🎨");
    }

    #[test]
    fn test_get_file_icon_unknown() {
        assert_eq!(get_file_icon("xyz"), "📄");
        assert_eq!(get_file_icon("unknown"), "📄");
    }

    #[test]
    fn test_get_file_icon_case_insensitive() {
        assert_eq!(get_file_icon("RS"), "🦀");
        assert_eq!(get_file_icon("PY"), "🐍");
        assert_eq!(get_file_icon("Js"), "📜");
    }

    #[test]
    fn test_format_modified_none() {
        let result = format_modified(None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_modified_some() {
        use std::time::{Duration, UNIX_EPOCH};
        let time = UNIX_EPOCH + Duration::from_secs(1609459200); // 2021-01-01 00:00:00 UTC
        let result = format_modified(Some(time));
        assert!(result.contains("2021"));
        assert!(result.contains("01-01"));
    }

    #[test]
    fn test_file_entry_clone() {
        let entry = FileEntry {
            path: "src/lib.rs".to_string(),
            size: 1024,
            tokens: 256,
            file_type: "Rust".to_string(),
            included: true,
        };

        let cloned = entry.clone();
        assert_eq!(entry.path, cloned.path);
        assert_eq!(entry.size, cloned.size);
        assert_eq!(entry.tokens, cloned.tokens);
        assert_eq!(entry.included, cloned.included);
    }

    #[test]
    fn test_file_entry_serialize() {
        let entry = FileEntry {
            path: "test.rs".to_string(),
            size: 100,
            tokens: 25,
            file_type: "rust".to_string(),
            included: false,
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("test.rs"));
        assert!(json.contains("100"));
        assert!(json.contains("rust"));
    }

    #[test]
    fn test_template_data_serialize() {
        let data = TemplateData {
            repository_name: "test-repo".to_string(),
            algorithm: "two-pass".to_string(),
            generated_time: "2024-01-01".to_string(),
            selection_time_ms: 100,
            total_files: 5,
            total_tokens: "1000".to_string(),
            total_size: "10 KB".to_string(),
            coverage_percentage: 50,
            files: vec![],
        };

        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("test-repo"));
        assert!(json.contains("two-pass"));
    }

    #[test]
    fn test_template_file_serialize() {
        let file = TemplateFile {
            relative_path: "src/main.rs".to_string(),
            icon: "🦀".to_string(),
            size: "1.00 KB".to_string(),
            estimated_tokens: "250".to_string(),
            importance_score: "0.85".to_string(),
            content: "fn main() {}".to_string(),
        };

        let json = serde_json::to_string(&file).unwrap();
        assert!(json.contains("src/main.rs"));
        assert!(json.contains("fn main"));
    }

    #[test]
    fn test_template_data_clone() {
        let data = TemplateData {
            repository_name: "repo".to_string(),
            algorithm: "test".to_string(),
            generated_time: "now".to_string(),
            selection_time_ms: 50,
            total_files: 10,
            total_tokens: "500".to_string(),
            total_size: "5 KB".to_string(),
            coverage_percentage: 25,
            files: vec![],
        };

        let cloned = data.clone();
        assert_eq!(data.repository_name, cloned.repository_name);
        assert_eq!(data.total_files, cloned.total_files);
    }

    #[test]
    fn test_template_file_clone() {
        let file = TemplateFile {
            relative_path: "test.rs".to_string(),
            icon: "📄".to_string(),
            size: "1 KB".to_string(),
            estimated_tokens: "100".to_string(),
            importance_score: "0.5".to_string(),
            content: "code".to_string(),
        };

        let cloned = file.clone();
        assert_eq!(file.relative_path, cloned.relative_path);
        assert_eq!(file.content, cloned.content);
    }

    #[test]
    fn test_prepare_template_data_empty() {
        let repo_path = PathBuf::from("/path/to/repo");
        let files: Vec<WebReportFile> = vec![];
        let metrics = WebSelectionMetrics {
            total_files_discovered: 0,
            files_selected: 0,
            total_tokens_estimated: 0,
            selection_time_ms: 10,
            algorithm_used: "test".to_string(),
            coverage_score: 0.0,
            relevance_score: 0.0,
        };

        let data = prepare_template_data(&repo_path, &files, &metrics);

        assert_eq!(data.repository_name, "repo");
        assert_eq!(data.total_files, 0);
        assert_eq!(data.coverage_percentage, 0);
    }

    #[test]
    fn test_prepare_template_data_with_files() {
        let repo_path = PathBuf::from("/path/to/my-project");
        let files = vec![WebReportFile {
            path: PathBuf::from("/path/to/my-project/src/main.rs"),
            relative_path: "src/main.rs".to_string(),
            content: "fn main() {}".to_string(),
            size: 1024,
            estimated_tokens: 256,
            importance_score: 0.9,
            centrality_score: 0.8,
            query_relevance_score: 0.0,
            entry_point_proximity: 1.0,
            content_quality_score: 0.7,
            repository_role_score: 0.6,
            recency_score: 0.5,
            modified: "2024-01-01".to_string(),
        }];
        let metrics = WebSelectionMetrics {
            total_files_discovered: 10,
            files_selected: 1,
            total_tokens_estimated: 256,
            selection_time_ms: 50,
            algorithm_used: "two-pass".to_string(),
            coverage_score: 0.1,
            relevance_score: 0.9,
        };

        let data = prepare_template_data(&repo_path, &files, &metrics);

        assert_eq!(data.repository_name, "my-project");
        assert_eq!(data.total_files, 1);
        assert_eq!(data.algorithm, "two-pass");
        assert_eq!(data.files.len(), 1);
        assert_eq!(data.files[0].relative_path, "src/main.rs");
        assert_eq!(data.files[0].icon, "🦀");
        assert_eq!(data.coverage_percentage, 10);
    }

    #[test]
    fn test_build_selection_result_empty() {
        let analysis = AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 0,
                files_selected: 0,
                total_tokens_estimated: 0,
                selection_time_ms: 0,
                algorithm_used: "test".to_string(),
                coverage_score: 0.0,
                relevance_score: 0.0,
            },
            repository_files: vec![],
            token_budget: 1000,
        };

        let result = build_selection_result(&analysis, &[]);

        assert_eq!(result.files.len(), 0);
        assert_eq!(result.total_tokens_used, 0);
        assert_eq!(result.budget, 1000);
    }

    #[test]
    fn test_build_selection_result_zero_budget() {
        let analysis = AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 5,
                files_selected: 0,
                total_tokens_estimated: 500,
                selection_time_ms: 10,
                algorithm_used: "test".to_string(),
                coverage_score: 0.0,
                relevance_score: 0.0,
            },
            repository_files: vec![],
            token_budget: 0,
        };

        let result = build_selection_result(&analysis, &[]);

        // When budget is 0, it should use total_tokens_estimated from metrics
        assert_eq!(result.budget, 500);
        assert_eq!(result.unused_tokens, 500);
    }

    fn create_test_file_info(
        path: &str,
        relative: &str,
        size: u64,
        tokens: Option<usize>,
    ) -> FileInfo {
        use scribe_core::file::FileType;
        use scribe_core::{FileWeight, Language, RenderDecision};

        FileInfo {
            path: PathBuf::from(path),
            relative_path: relative.to_string(),
            size,
            language: Language::Rust,
            content: None,
            token_estimate: tokens,
            modified: None,
            is_binary: false,
            centrality_score: None,
            git_status: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            line_count: None,
            char_count: None,
            weight: FileWeight::default(),
        }
    }

    #[test]
    fn test_file_entry_from_fileinfo() {
        let file_info = create_test_file_info("/test/src/lib.rs", "src/lib.rs", 2048, Some(512));

        let entry = file_entry_from_fileinfo(&file_info, true);

        assert_eq!(entry.path, "src/lib.rs");
        assert_eq!(entry.size, 2048);
        assert_eq!(entry.tokens, 512);
        assert!(entry.included);
        assert!(entry.file_type.contains("Rust"));
    }

    #[test]
    fn test_file_entry_from_fileinfo_excluded() {
        let file_info = create_test_file_info("/test/src/utils.py", "src/utils.py", 1024, None);

        let entry = file_entry_from_fileinfo(&file_info, false);

        assert_eq!(entry.path, "src/utils.py");
        assert_eq!(entry.tokens, 0); // No token estimate
        assert!(!entry.included);
    }

    #[test]
    fn test_file_entry_from_report_file_with_info() {
        let report_file = WebReportFile {
            path: PathBuf::from("/test/main.rs"),
            relative_path: "main.rs".to_string(),
            content: "fn main() {}".to_string(),
            size: 512,
            estimated_tokens: 100,
            importance_score: 0.8,
            centrality_score: 0.5,
            query_relevance_score: 0.0,
            entry_point_proximity: 1.0,
            content_quality_score: 0.7,
            repository_role_score: 0.6,
            recency_score: 0.5,
            modified: "2024-01-01".to_string(),
        };

        let file_info = create_test_file_info("/test/main.rs", "main.rs", 512, Some(100));

        let entry = file_entry_from_report_file(&report_file, Some(&file_info));

        assert_eq!(entry.path, "main.rs");
        assert_eq!(entry.size, 512);
        assert_eq!(entry.tokens, 100);
        assert!(entry.included);
        assert!(entry.file_type.contains("Rust"));
    }

    #[test]
    fn test_file_entry_from_report_file_without_info() {
        let report_file = WebReportFile {
            path: PathBuf::from("/test/script.js"),
            relative_path: "script.js".to_string(),
            content: "console.log('hi');".to_string(),
            size: 256,
            estimated_tokens: 50,
            importance_score: 0.5,
            centrality_score: 0.3,
            query_relevance_score: 0.0,
            entry_point_proximity: 0.0,
            content_quality_score: 0.5,
            repository_role_score: 0.4,
            recency_score: 0.5,
            modified: "2024-01-01".to_string(),
        };

        let entry = file_entry_from_report_file(&report_file, None);

        assert_eq!(entry.path, "script.js");
        assert_eq!(entry.size, 256);
        assert_eq!(entry.tokens, 50);
        assert!(entry.included);
        assert_eq!(entry.file_type, "unknown");
    }

    #[test]
    fn test_file_size_to_usize_overflow() {
        // Test with maximum u64 value
        let result = file_size_to_usize(u64::MAX);
        // On 64-bit systems, this should return usize::MAX
        assert!(result > 0);
    }

    #[test]
    fn test_build_selection_result_with_files() {
        let file_info = create_test_file_info("/test/src/main.rs", "src/main.rs", 1024, Some(256));

        let analysis = AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 10,
                files_selected: 1,
                total_tokens_estimated: 256,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 0.1,
                relevance_score: 0.8,
            },
            repository_files: vec![file_info],
            token_budget: 1000,
        };

        let included_files = vec!["src/main.rs".to_string()];
        let result = build_selection_result(&analysis, &included_files);

        assert_eq!(result.files.len(), 1);
        assert_eq!(result.total_tokens_used, 256);
        assert_eq!(result.budget, 1000);
        assert_eq!(result.unused_tokens, 744);
        assert_eq!(result.total_files_considered, 10);
    }

    #[test]
    fn test_recompute_bundle_summary_empty() {
        let mut bundle_state = BundleState {
            included_files: vec![],
            excluded_files: HashMap::new(),
            total_size: 0,
            token_estimate: 0,
            last_updated: chrono::Utc::now(),
        };

        let analysis = AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 0,
                files_selected: 0,
                total_tokens_estimated: 0,
                selection_time_ms: 0,
                algorithm_used: "test".to_string(),
                coverage_score: 0.0,
                relevance_score: 0.0,
            },
            repository_files: vec![],
            token_budget: 1000,
        };

        let categories = recompute_bundle_summary(&mut bundle_state, &analysis);

        assert!(categories.contains_key("included"));
        assert!(categories.contains_key("excluded"));
        assert!(categories.get("included").unwrap().is_empty());
    }

    #[test]
    fn test_recompute_bundle_summary_with_files() {
        let file_info1 = create_test_file_info("/test/src/main.rs", "src/main.rs", 1024, Some(256));
        let file_info2 = create_test_file_info("/test/src/lib.rs", "src/lib.rs", 2048, Some(512));

        let mut bundle_state = BundleState {
            included_files: vec!["src/main.rs".to_string()],
            excluded_files: HashMap::new(),
            total_size: 0,
            token_estimate: 0,
            last_updated: chrono::Utc::now(),
        };

        let analysis = AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 2,
                files_selected: 1,
                total_tokens_estimated: 256,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 0.5,
                relevance_score: 0.8,
            },
            repository_files: vec![file_info1, file_info2],
            token_budget: 1000,
        };

        let categories = recompute_bundle_summary(&mut bundle_state, &analysis);

        assert_eq!(categories.get("included").unwrap().len(), 1);
        assert_eq!(categories.get("excluded").unwrap().len(), 1);
        assert_eq!(bundle_state.total_size, 1024);
        assert_eq!(bundle_state.token_estimate, 256);
    }

    #[test]
    fn test_recompute_bundle_summary_deduplicates() {
        let file_info = create_test_file_info("/test/src/main.rs", "src/main.rs", 1024, Some(256));

        let mut bundle_state = BundleState {
            // Include same file twice (duplicates)
            included_files: vec!["src/main.rs".to_string(), "src/main.rs".to_string()],
            excluded_files: HashMap::new(),
            total_size: 0,
            token_estimate: 0,
            last_updated: chrono::Utc::now(),
        };

        let analysis = AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 1,
                files_selected: 1,
                total_tokens_estimated: 256,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 1.0,
                relevance_score: 0.8,
            },
            repository_files: vec![file_info],
            token_budget: 1000,
        };

        let categories = recompute_bundle_summary(&mut bundle_state, &analysis);

        // Should deduplicate to 1 file
        assert_eq!(categories.get("included").unwrap().len(), 1);
        assert_eq!(bundle_state.included_files.len(), 1);
    }

    #[test]
    fn test_recompute_bundle_summary_removes_invalid_paths() {
        let file_info = create_test_file_info("/test/src/main.rs", "src/main.rs", 1024, Some(256));

        let mut bundle_state = BundleState {
            // Include a non-existent file path
            included_files: vec!["src/main.rs".to_string(), "nonexistent.rs".to_string()],
            excluded_files: HashMap::new(),
            total_size: 0,
            token_estimate: 0,
            last_updated: chrono::Utc::now(),
        };

        let analysis = AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 1,
                files_selected: 1,
                total_tokens_estimated: 256,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 1.0,
                relevance_score: 0.8,
            },
            repository_files: vec![file_info],
            token_budget: 1000,
        };

        let categories = recompute_bundle_summary(&mut bundle_state, &analysis);

        // Should only include the valid file
        assert_eq!(categories.get("included").unwrap().len(), 1);
        assert_eq!(bundle_state.included_files.len(), 1);
        assert_eq!(bundle_state.included_files[0], "src/main.rs");
    }
}

//! HTTP handlers for the Scribe web service

use crate::{
    AnalysisOutput, ApiResponse, AppState, BundleState, Result, WebReportFile, WebSelectionMetrics,
    WebServiceError,
};
use axum::{
    extract::State,
    response::{Html, IntoResponse, Json},
};
use handlebars::{Context as HbContext, Handlebars, Helper, HelperResult, Output, RenderContext};
use once_cell::sync::Lazy;
use scribe_core::FileInfo;
use scribe_selection::{
    BundleOptions, CodeBundler, ContextExtractor, ContextOptions, SelectionResult,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, error, info};

/// Health check endpoint
pub async fn status() -> Json<ApiResponse<StatusInfo>> {
    let status = StatusInfo {
        service: "scribe-webservice".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        status: "healthy".to_string(),
    };
    Json(ApiResponse::success(status))
}

/// Ping endpoint to keep server alive
pub async fn ping(State(state): State<AppState>) -> Json<ApiResponse<PingResponse>> {
    let now = tokio::time::Instant::now();

    // Update last ping time
    {
        let mut last_ping = state.last_ping.write().await;
        *last_ping = now;
    }

    debug!("Received ping, updated last activity");

    let config = state.config.read().await.clone();
    let response = PingResponse {
        timestamp: chrono::Utc::now(),
        auto_shutdown_enabled: config.auto_shutdown,
        timeout_seconds: config.auto_shutdown_timeout,
    };

    Json(ApiResponse::success(response))
}

/// Manual shutdown endpoint
pub async fn shutdown(State(state): State<AppState>) -> Json<ApiResponse<String>> {
    info!("Manual shutdown requested");

    // Send shutdown signal
    {
        let mut sender_lock = state.shutdown_sender.write().await;
        if let Some(sender) = sender_lock.take() {
            let _ = sender.send(());
            Json(ApiResponse::success("Shutdown initiated".to_string()))
        } else {
            Json(ApiResponse::error(
                "Shutdown already in progress".to_string(),
            ))
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusInfo {
    pub service: String,
    pub version: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PingResponse {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub auto_shutdown_enabled: bool,
    pub timeout_seconds: u64,
}

/// Main index page
pub async fn index() -> Html<String> {
    let html = r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scribe Web Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .nav { margin-bottom: 40px; }
            .nav a { margin-right: 20px; padding: 10px 20px; background: #007acc; color: white; text-decoration: none; border-radius: 5px; }
            .nav a:hover { background: #005a99; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Scribe Web Service</h1>
                <p>Interactive repository analysis and bundle generation</p>
            </div>
            <div class="nav">
                <a href="/editor">Bundle Editor</a>
                <a href="/api/status">API Status</a>
            </div>
            <div class="content">
                <h2>Features</h2>
                <ul>
                    <li>Real-time bundle generation and saving</li>
                    <li>Interactive file selection</li>
                    <li>REST API endpoints</li>
                    <li>Direct download capabilities</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    "#.to_string();
    Html(html)
}

/// Bundle editor interface
pub async fn bundle_editor(State(state): State<AppState>) -> impl IntoResponse {
    let repo_path = {
        let cfg = state.config.read().await;
        cfg.repo_path.clone()
    };
    info!(
        "Starting repository analysis for web editor: {}",
        repo_path.display()
    );

    match get_or_compute_analysis(&state).await {
        Ok(cached) => {
            let _ = update_bundle_state(&state, &cached.analysis).await;

            if let Some(html) = cached
                .rendered_html
                .clone()
                .or_else(|| render_template(&cached.template_data).ok())
            {
                Html(html).into_response()
            } else {
                Html(
                    r#"
        <!DOCTYPE html>
        <html>
        <head><title>Scribe Bundle Editor - Error</title></head>
        <body>
            <h1>Unable to render bundle editor</h1>
            <p>Error: rendering pipeline failed.</p>
        </body>
        </html>
        "#
                    .to_string(),
                )
                .into_response()
            }
        }
        Err(err) => {
            error!("Bundle editor failed: {}", err);
            Html(format!(
                r#"
        <!DOCTYPE html>
        <html>
        <head><title>Scribe Bundle Editor - Error</title></head>
        <body>
            <h1>Unable to generate bundle</h1>
            <p>Error: {}</p>
        </body>
        </html>
        "#,
                err
            ))
            .into_response()
        }
    }
}

/// Scan repository and return file information
pub async fn scan_repository(State(state): State<AppState>) -> impl IntoResponse {
    let repo_path = {
        let cfg = state.config.read().await;
        cfg.repo_path.clone()
    };
    info!("Scanning repository: {}", repo_path.display());
    match get_or_compute_analysis(&state).await {
        Ok(cached) => {
            let categories = update_bundle_state(&state, &cached.analysis).await;
            let bundle_snapshot = state.bundle_state.read().await.clone();
            let scan_result = build_scan_result(
                &cached.analysis,
                &bundle_snapshot,
                cached.rendered_html.clone(),
                categories,
            );

            Json(ApiResponse::success(scan_result))
        }
        Err(err) => {
            error!("Scan failed: {}", err);
            Json(ApiResponse::error(err.to_string()))
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScanResult {
    pub total_files: usize,
    pub selected_files: usize,
    pub excluded_files: usize,
    pub token_estimate: usize,
    pub total_size: usize,
    pub categories: HashMap<String, Vec<FileEntry>>,
    pub rendered_html: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub size: usize,
    pub tokens: usize,
    pub file_type: String,
    pub included: bool,
}

/// List files in the repository
pub async fn list_files(State(state): State<AppState>) -> impl IntoResponse {
    debug!("Listing files for repository");

    match get_or_compute_analysis(&state).await {
        Ok(cached) => {
            let selected_paths: HashSet<&str> = cached
                .analysis
                .selected_files
                .iter()
                .map(|file| file.relative_path.as_str())
                .collect();

            let entries: Vec<FileEntry> = cached
                .analysis
                .repository_files
                .iter()
                .map(|file| {
                    let included = selected_paths.contains(file.relative_path.as_str());
                    file_entry_from_fileinfo(file, included)
                })
                .collect();

            Json(ApiResponse::success(entries))
        }
        Err(err) => {
            error!("File listing failed: {}", err);
            Json(ApiResponse::error(err.to_string()))
        }
    }
}

/// Toggle file inclusion in bundle
pub async fn toggle_file(
    State(state): State<AppState>,
    Json(request): Json<ToggleRequest>,
) -> impl IntoResponse {
    debug!("Toggling file: {}", request.path);

    let cached = match get_or_compute_analysis(&state).await {
        Ok(cached) => cached,
        Err(err) => {
            error!("Toggle failed during analysis: {}", err);
            return Json(ApiResponse::error(err.to_string()));
        }
    };

    let available_paths: HashSet<&str> = cached
        .analysis
        .repository_files
        .iter()
        .map(|info| info.relative_path.as_str())
        .collect();

    if !available_paths.contains(request.path.as_str()) {
        return Json(ApiResponse::error(format!(
            "File {} not found in repository analysis",
            request.path
        )));
    }

    let mut bundle_state = state.bundle_state.write().await;

    if let Some(position) = bundle_state
        .included_files
        .iter()
        .position(|path| path == &request.path)
    {
        bundle_state.included_files.remove(position);
        info!("Removed file from bundle: {}", request.path);
    } else {
        bundle_state.included_files.push(request.path.clone());
        info!("Added file to bundle: {}", request.path);
    }

    let categories = recompute_bundle_summary(&mut bundle_state, &cached.analysis);
    let snapshot = bundle_state.clone();
    drop(bundle_state);

    let scan_result = build_scan_result(&cached.analysis, &snapshot, None, categories);

    Json(ApiResponse::success(scan_result))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToggleRequest {
    pub path: String,
}

/// Toggle directory inclusion in bundle
pub async fn toggle_directory(
    State(state): State<AppState>,
    Json(request): Json<ToggleRequest>,
) -> impl IntoResponse {
    debug!("Toggling directory: {}", request.path);

    let cached = match get_or_compute_analysis(&state).await {
        Ok(cached) => cached,
        Err(err) => {
            error!("Directory toggle failed: {}", err);
            return Json(ApiResponse::error(err.to_string()));
        }
    };

    let mut prefix = request.path.trim().to_string();
    if prefix.is_empty() {
        prefix = ".".to_string();
    }
    if !prefix.ends_with('/') {
        prefix.push('/');
    }

    let mut directory_files: Vec<String> = cached
        .analysis
        .repository_files
        .iter()
        .filter_map(|file| {
            if file.relative_path.starts_with(&prefix) {
                Some(file.relative_path.clone())
            } else {
                None
            }
        })
        .collect();

    if directory_files.is_empty() {
        return Json(ApiResponse::error(format!(
            "No files found under directory {}",
            request.path
        )));
    }

    directory_files.sort();
    directory_files.dedup();

    let mut bundle_state = state.bundle_state.write().await;
    let included_set: HashSet<_> = bundle_state.included_files.iter().cloned().collect();
    let currently_selected = directory_files
        .iter()
        .all(|path| included_set.contains(path));

    if currently_selected {
        bundle_state
            .included_files
            .retain(|path| !path.starts_with(&prefix));
        info!("Removed directory {} from bundle", request.path);
    } else {
        for path in directory_files {
            if !bundle_state.included_files.iter().any(|p| p == &path) {
                bundle_state.included_files.push(path);
            }
        }
        info!("Added directory {} to bundle", request.path);
    }

    let categories = recompute_bundle_summary(&mut bundle_state, &cached.analysis);
    let snapshot = bundle_state.clone();
    drop(bundle_state);

    let scan_result = build_scan_result(&cached.analysis, &snapshot, None, categories);

    Json(ApiResponse::success(scan_result))
}

/// Generate bundle with current selection
pub async fn generate_bundle(
    State(state): State<AppState>,
    Json(request): Json<GenerateBundleRequest>,
) -> impl IntoResponse {
    info!("Generating bundle in {} format", request.format);

    match get_or_compute_analysis(&state).await {
        Ok(cached) => {
            let _ = update_bundle_state(&state, &cached.analysis).await;

            let bundle_snapshot = state.bundle_state.read().await.clone();
            let included_files = bundle_snapshot.included_files.clone();

            if included_files.is_empty() {
                return Json(ApiResponse::error(
                    "No files selected for bundle".to_string(),
                ));
            }

            if request.format.eq_ignore_ascii_case("html") {
                let repo_root = {
                    let cfg = state.config.read().await;
                    cfg.repo_path.clone()
                };

                match build_reports_for_selection(&cached.analysis, &bundle_snapshot) {
                    Ok((reports, metrics)) => {
                        let template_data = prepare_template_data(&repo_root, &reports, &metrics);

                        return match render_template(&template_data) {
                            Ok(content) => {
                                let bundle = GeneratedBundle {
                                    format: request.format.clone(),
                                    filename: "scribe-bundle.html".to_string(),
                                    size: content.len(),
                                    content,
                                };
                                Json(ApiResponse::success(bundle))
                            }
                            Err(err) => {
                                error!("Bundle generation failed: {}", err);
                                Json(ApiResponse::error(
                                    "Unable to render HTML bundle".to_string(),
                                ))
                            }
                        };
                    }
                    Err(err) => {
                        error!("Failed to build HTML reports: {}", err);
                        return Json(ApiResponse::error(err.to_string()));
                    }
                }
            }

            let context_extractor = ContextExtractor::new();
            let selection_result = build_selection_result(&cached.analysis, &included_files);
            let context = match context_extractor
                .extract(&selection_result, &ContextOptions::default())
                .await
            {
                Ok(context) => context,
                Err(err) => {
                    error!("Context extraction failed: {}", err);
                    return Json(ApiResponse::error(err.to_string()));
                }
            };

            let bundler = CodeBundler::new();
            let bundle_options = BundleOptions {
                format: request.format.clone(),
                include_metadata: true,
            };

            match bundler.bundle(&context, &bundle_options).await {
                Ok(code_bundle) => {
                    let filename = format!(
                        "scribe-bundle.{}",
                        match request.format.as_str() {
                            "markdown" => "md",
                            "json" => "json",
                            "plain" => "txt",
                            _ => "txt",
                        }
                    );

                    let content = code_bundle.content;
                    let size = content.len();
                    let bundle = GeneratedBundle {
                        format: request.format.clone(),
                        content,
                        filename,
                        size,
                    };

                    Json(ApiResponse::success(bundle))
                }
                Err(err) => {
                    error!("Bundle generation failed: {}", err);
                    Json(ApiResponse::error(err.to_string()))
                }
            }
        }
        Err(err) => {
            error!("Bundle generation scan failed: {}", err);
            Json(ApiResponse::error(err.to_string()))
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateBundleRequest {
    pub format: String,
    pub options: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeneratedBundle {
    pub format: String,
    pub content: String,
    pub filename: String,
    pub size: usize,
}

/// Save bundle to file system
pub async fn save_bundle(
    State(state): State<AppState>,
    Json(request): Json<SaveBundleRequest>,
) -> impl IntoResponse {
    info!("Saving bundle to: {}", request.path);

    let cached = match get_or_compute_analysis(&state).await {
        Ok(cached) => cached,
        Err(err) => {
            error!("Bundle save failed during analysis: {}", err);
            return Json(ApiResponse::error(err.to_string()));
        }
    };

    let _ = update_bundle_state(&state, &cached.analysis).await;

    let bundle_snapshot = state.bundle_state.read().await.clone();
    let included_files = bundle_snapshot.included_files.clone();

    if included_files.is_empty() {
        return Json(ApiResponse::error(
            "No files selected for bundle".to_string(),
        ));
    }

    let repo_root = {
        let cfg = state.config.read().await;
        cfg.repo_path.clone()
    };

    let content = if request.format.eq_ignore_ascii_case("html") {
        match build_reports_for_selection(&cached.analysis, &bundle_snapshot) {
            Ok((reports, metrics)) => {
                let template_data = prepare_template_data(&repo_root, &reports, &metrics);
                match render_template(&template_data) {
                    Ok(html) => html,
                    Err(err) => {
                        error!("Bundle generation failed: {}", err);
                        return Json(ApiResponse::error(
                            "Unable to render HTML bundle".to_string(),
                        ));
                    }
                }
            }
            Err(err) => {
                error!("Failed to build HTML reports: {}", err);
                return Json(ApiResponse::error(err.to_string()));
            }
        }
    } else {
        let context_extractor = ContextExtractor::new();
        let selection_result = build_selection_result(&cached.analysis, &included_files);
        let context = match context_extractor
            .extract(&selection_result, &ContextOptions::default())
            .await
        {
            Ok(context) => context,
            Err(err) => {
                error!("Context extraction failed: {}", err);
                return Json(ApiResponse::error(err.to_string()));
            }
        };

        let bundler = CodeBundler::new();
        let options = BundleOptions {
            format: request.format.clone(),
            include_metadata: true,
        };

        match bundler.bundle(&context, &options).await {
            Ok(bundle) => bundle.content,
            Err(err) => {
                error!("Bundle generation failed: {}", err);
                return Json(ApiResponse::error(err.to_string()));
            }
        }
    };

    let mut target_path = PathBuf::from(&request.path);
    if target_path.is_relative() {
        target_path = repo_root.join(&target_path);
    }

    if let Some(parent) = target_path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            error!(
                "Failed to create directories for {}: {}",
                target_path.display(),
                err
            );
            return Json(ApiResponse::error(err.to_string()));
        }
    }

    let size = content.len();
    if let Err(err) = fs::write(&target_path, content) {
        error!(
            "Failed to write bundle to {}: {}",
            target_path.display(),
            err
        );
        return Json(ApiResponse::error(err.to_string()));
    }

    {
        let mut bundle_state = state.bundle_state.write().await;
        bundle_state.last_updated = chrono::Utc::now();
    }

    let result = SaveResult {
        path: target_path.to_string_lossy().to_string(),
        size,
        format: request.format.clone(),
    };

    Json(ApiResponse::success(result))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SaveBundleRequest {
    pub path: String,
    pub format: String,
    pub options: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SaveResult {
    pub path: String,
    pub size: usize,
    pub format: String,
}

/// Export bundle (generate and return for download)
pub async fn export_bundle(
    State(state): State<AppState>,
    Json(request): Json<GenerateBundleRequest>,
) -> impl IntoResponse {
    // This is the same as generate_bundle but with different semantic meaning
    generate_bundle(State(state), Json(request)).await
}

/// Get current configuration
pub async fn get_config(
    State(state): State<AppState>,
) -> Json<ApiResponse<crate::WebServiceConfig>> {
    let config = state.config.read().await.clone();
    Json(ApiResponse::success(config))
}

/// Update configuration
pub async fn update_config(
    State(state): State<AppState>,
    Json(new_config): Json<crate::WebServiceConfig>,
) -> impl IntoResponse {
    info!("Updating configuration");

    if !new_config.repo_path.exists() {
        return Json(ApiResponse::error(format!(
            "Repository not found: {}",
            new_config.repo_path.display()
        )));
    }

    {
        let mut config = state.config.write().await;
        *config = new_config.clone();
    }

    {
        let mut bundle_state = state.bundle_state.write().await;
        *bundle_state = BundleState::default();
    }

    {
        let mut cache = ANALYSIS_CACHE.lock().unwrap();
        cache.clear();
    }

    Json(ApiResponse::success(new_config))
}

fn recompute_bundle_summary(
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

async fn update_bundle_state(
    state: &AppState,
    analysis: &AnalysisOutput,
) -> HashMap<String, Vec<FileEntry>> {
    let mut bundle_state = state.bundle_state.write().await;
    if bundle_state.included_files.is_empty() {
        bundle_state.included_files = analysis
            .selected_files
            .iter()
            .map(|file| file.relative_path.clone())
            .collect();
    } else {
        let valid_paths: HashSet<&str> = analysis
            .repository_files
            .iter()
            .map(|file| file.relative_path.as_str())
            .collect();

        bundle_state
            .included_files
            .retain(|path| valid_paths.contains(path.as_str()));

        if bundle_state.included_files.is_empty() {
            bundle_state.included_files = analysis
                .selected_files
                .iter()
                .map(|file| file.relative_path.clone())
                .collect();
        }
    }
    recompute_bundle_summary(&mut bundle_state, analysis)
}

fn file_entry_from_fileinfo(file: &FileInfo, included: bool) -> FileEntry {
    FileEntry {
        path: file.relative_path.clone(),
        size: file_size_to_usize(file.size),
        tokens: file.token_estimate.unwrap_or(0),
        file_type: format!("{:?}", file.language),
        included,
    }
}

fn file_entry_from_report_file(
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

fn file_size_to_usize(value: u64) -> usize {
    if value > usize::MAX as u64 {
        usize::MAX
    } else {
        value as usize
    }
}

fn build_selection_result(analysis: &AnalysisOutput, included_files: &[String]) -> SelectionResult {
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

fn build_scan_result(
    analysis: &AnalysisOutput,
    bundle_state: &BundleState,
    rendered_html: Option<String>,
    mut categories: HashMap<String, Vec<FileEntry>>,
) -> ScanResult {
    categories.entry("included".to_string()).or_default();
    categories.entry("excluded".to_string()).or_default();

    ScanResult {
        total_files: analysis.repository_files.len(),
        selected_files: bundle_state.included_files.len(),
        excluded_files: analysis
            .repository_files
            .len()
            .saturating_sub(bundle_state.included_files.len()),
        token_estimate: bundle_state.token_estimate,
        total_size: bundle_state.total_size,
        categories,
        rendered_html,
    }
}

fn build_reports_for_selection(
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

/// Template data structure for Handlebars rendering
#[derive(Serialize, Clone)]
struct TemplateData {
    repository_name: String,
    algorithm: String,
    generated_time: String,
    selection_time_ms: u64,
    total_files: usize,
    total_tokens: String,
    total_size: String,
    coverage_percentage: u32,
    files: Vec<TemplateFile>,
}

#[derive(Serialize, Clone)]
struct TemplateFile {
    relative_path: String,
    icon: String,
    size: String,
    estimated_tokens: String,
    importance_score: String,
    content: String,
}

fn prepare_template_data(
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
    let formatted_size = format_file_size(total_size);

    let coverage_percentage = (metrics.coverage_score * 100.0).min(100.0) as u32;

    let template_files: Vec<TemplateFile> = selected_files
        .iter()
        .map(|file| {
            let icon = Path::new(&file.relative_path)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(get_file_icon)
                .unwrap_or_else(|| get_file_icon(""));

            TemplateFile {
                relative_path: file.relative_path.clone(),
                icon,
                size: format_file_size(file.size),
                estimated_tokens: file.estimated_tokens.to_string(),
                importance_score: format!("{:.2}", file.importance_score),
                content: file.content.clone(),
            }
        })
        .collect();

    let selection_time_ms = metrics.selection_time_ms as u64;

    TemplateData {
        repository_name: format!("Scribe Analysis - {}", repo_name),
        algorithm: metrics.algorithm_used.clone(),
        generated_time: chrono::Utc::now()
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string(),
        selection_time_ms,
        total_files: metrics.total_files_discovered,
        total_tokens: metrics.total_tokens_estimated.to_string(),
        total_size: formatted_size,
        coverage_percentage,
        files: template_files,
    }
}

fn render_template(
    template_data: &TemplateData,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let template_content = include_str!("../templates/report_bundled.html");

    let mut handlebars = Handlebars::new();
    handlebars.register_helper(
        "add",
        Box::new(
            |helper: &Helper<'_, '_>,
             _hb: &Handlebars<'_>,
             _ctx: &HbContext,
             _rc: &mut RenderContext<'_, '_>,
             out: &mut dyn Output|
             -> HelperResult {
                let lhs = helper
                    .param(0)
                    .and_then(|v| v.value().as_i64())
                    .unwrap_or(0);
                let rhs = helper
                    .param(1)
                    .and_then(|v| v.value().as_i64())
                    .unwrap_or(0);
                let sum = lhs + rhs;
                out.write(sum.to_string().as_ref())?;
                Ok(())
            },
        ),
    );
    handlebars.register_template_string("report", template_content)?;

    let mut rendered = handlebars.render("report", template_data)?;

    // Fix static asset path
    rendered = rendered.replace(
        "src=\"assets/scribe-tree-bundle.js\"",
        "src=\"/static/scribe-tree-bundle.js\"",
    );

    Ok(rendered)
}

const ANALYSIS_CACHE_TTL: Duration = Duration::from_secs(5);

#[derive(Hash, Eq, PartialEq, Clone)]
struct CacheKey {
    repo_path: String,
    token_budget: usize,
    max_file_size: usize,
    auto_exclude_tests: bool,
}

#[derive(Clone)]
struct CachedAnalysis {
    generated_at: Instant,
    analysis: AnalysisOutput,
    scan_result: ScanResult,
    template_data: TemplateData,
    rendered_html: Option<String>,
}

static ANALYSIS_CACHE: Lazy<Mutex<HashMap<CacheKey, CachedAnalysis>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn build_cache_key(config: &crate::WebServiceConfig) -> CacheKey {
    CacheKey {
        repo_path: config.repo_path.to_string_lossy().into_owned(),
        token_budget: config.token_budget,
        max_file_size: config.max_file_size,
        auto_exclude_tests: config.auto_exclude_tests,
    }
}

fn build_render_data(
    analysis: &AnalysisOutput,
    config: &crate::WebServiceConfig,
) -> (ScanResult, TemplateData, Option<String>) {
    let selected_paths: HashSet<&str> = analysis
        .selected_files
        .iter()
        .map(|file| file.relative_path.as_str())
        .collect();

    let included_entries: Vec<FileEntry> = analysis
        .selected_files
        .iter()
        .map(|file| {
            let matching_info = analysis
                .repository_files
                .iter()
                .find(|info| info.relative_path == file.relative_path);
            file_entry_from_report_file(file, matching_info)
        })
        .collect();

    let excluded_entries: Vec<FileEntry> = analysis
        .repository_files
        .iter()
        .filter(|file| !selected_paths.contains(file.relative_path.as_str()))
        .map(|file| file_entry_from_fileinfo(file, false))
        .collect();

    let mut categories = HashMap::new();
    categories.insert("included".to_string(), included_entries.clone());
    categories.insert("excluded".to_string(), excluded_entries.clone());

    let total_size: usize = included_entries.iter().map(|entry| entry.size).sum();

    let template_data = prepare_template_data(
        &config.repo_path,
        &analysis.selected_files,
        &analysis.metrics,
    );

    let rendered_html = render_template(&template_data).ok();

    (
        ScanResult {
            total_files: analysis.repository_files.len(),
            selected_files: included_entries.len(),
            excluded_files: excluded_entries.len(),
            token_estimate: analysis.metrics.total_tokens_estimated,
            total_size,
            categories,
            rendered_html: rendered_html.clone(),
        },
        template_data,
        rendered_html,
    )
}

async fn get_or_compute_analysis(state: &AppState) -> crate::Result<CachedAnalysis> {
    let config_snapshot = { state.config.read().await.clone() };
    let key = build_cache_key(&config_snapshot);

    if let Some(entry) = {
        let cache = ANALYSIS_CACHE.lock().unwrap();
        cache.get(&key).cloned()
    } {
        if entry.generated_at.elapsed() < ANALYSIS_CACHE_TTL {
            return Ok(entry);
        }
    }

    let analysis = state.analysis_provider.analyze(&config_snapshot).await?;
    let (scan_result, template_data, rendered_html) =
        build_render_data(&analysis, &config_snapshot);

    let cached = CachedAnalysis {
        generated_at: Instant::now(),
        analysis: analysis.clone(),
        scan_result: scan_result.clone(),
        template_data: template_data.clone(),
        rendered_html: rendered_html.clone(),
    };

    let mut cache = ANALYSIS_CACHE.lock().unwrap();
    cache.insert(key, cached.clone());

    Ok(cached)
}

fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

fn get_file_icon(extension: &str) -> String {
    match extension.to_lowercase().as_str() {
        "rs" => "🦀",
        "js" | "ts" => "⚡",
        "py" => "🐍",
        "go" => "🔷",
        "java" => "☕",
        "cpp" | "c" | "cc" => "⚙️",
        "html" => "🌐",
        "css" => "🎨",
        "md" => "📝",
        "json" => "📄",
        "yml" | "yaml" => "⚙️",
        "toml" => "📋",
        "sh" | "bash" => "🔧",
        _ => "📄",
    }
    .to_string()
}

fn format_modified(time: Option<SystemTime>) -> String {
    match time {
        Some(ts) => {
            let datetime: chrono::DateTime<chrono::Local> = ts.into();
            datetime.format("%Y-%m-%d %H:%M:%S").to_string()
        }
        None => "N/A".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnalysisProvider, AppState, BundleState, WebServiceConfig};
    use async_trait::async_trait;
    use axum::{routing::get, Router};
    use axum_test::TestServer;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::RwLock;

    struct DummyProvider;

    #[async_trait]
    impl AnalysisProvider for DummyProvider {
        async fn analyze(&self, config: &WebServiceConfig) -> crate::Result<AnalysisOutput> {
            Ok(AnalysisOutput {
                selected_files: Vec::new(),
                selected_file_infos: Vec::new(),
                metrics: WebSelectionMetrics {
                    total_files_discovered: 0,
                    files_selected: 0,
                    total_tokens_estimated: 0,
                    selection_time_ms: 0,
                    algorithm_used: "test".to_string(),
                    coverage_score: 0.0,
                    relevance_score: 0.0,
                },
                repository_files: Vec::new(),
                token_budget: config.token_budget,
            })
        }
    }

    fn create_test_app_state() -> AppState {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 0,
            host: "127.0.0.1".to_string(),
            token_budget: 10000,
            auto_open_browser: false,
            max_file_size: 1024,
            auto_exclude_tests: true,
            auto_shutdown: false,
            auto_shutdown_timeout: 60,
        };

        AppState {
            config: Arc::new(RwLock::new(config)),
            bundle_state: Arc::new(RwLock::new(BundleState::default())),
            last_ping: Arc::new(tokio::sync::RwLock::new(tokio::time::Instant::now())),
            shutdown_sender: Arc::new(tokio::sync::RwLock::new(None)),
            analysis_provider: Arc::new(DummyProvider),
        }
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let response = status().await;
        let status_info = response.0;

        assert!(status_info.success);
        assert!(status_info.data.is_some());

        let data = status_info.data.unwrap();
        assert_eq!(data.service, "scribe-webservice");
        assert_eq!(data.status, "healthy");
        assert!(!data.version.is_empty());
    }

    #[tokio::test]
    async fn test_index_handler() {
        let response = index().await;
        let html = response.0;

        assert!(html.contains("Scribe Web Service"));
        assert!(html.contains("Bundle Editor"));
        assert!(html.contains("<!DOCTYPE html>"));
    }

    #[tokio::test]
    async fn test_bundle_editor() {
        let state = create_test_app_state();
        let response = bundle_editor(axum::extract::State(state)).await;

        // Since we can't easily extract the HTML from IntoResponse in tests,
        // just verify that the handler completes without panicking
        let _ = response.into_response();

        // The integration tests will verify the actual HTML content
    }

    #[tokio::test]
    async fn test_scan_repository() {
        let state = create_test_app_state();
        let response = scan_repository(axum::extract::State(state)).await;

        // Since we can't directly access the Json content easily, let's test the function logic
        // The function should complete without panicking and return a valid response
        let _ = response.into_response();
    }

    #[tokio::test]
    async fn test_list_files() {
        let state = create_test_app_state();
        let response = list_files(axum::extract::State(state)).await;

        let _ = response.into_response();
        // Should return an empty list initially
    }

    #[tokio::test]
    async fn test_toggle_file() {
        let state = create_test_app_state();
        let request = ToggleRequest {
            path: "src/lib.rs".to_string(),
        };

        let response = toggle_file(
            axum::extract::State(state.clone()),
            axum::extract::Json(request),
        )
        .await;

        let _ = response.into_response();

        // Verify the file was added to the bundle state
        let bundle_state = state.bundle_state.read().await;
        assert!(bundle_state
            .included_files
            .contains(&"src/lib.rs".to_string()));
    }

    #[tokio::test]
    async fn test_toggle_file_remove() {
        let state = create_test_app_state();

        // First add a file
        {
            let mut bundle_state = state.bundle_state.write().await;
            bundle_state.included_files.push("src/lib.rs".to_string());
        }

        let request = ToggleRequest {
            path: "src/lib.rs".to_string(),
        };

        let response = toggle_file(
            axum::extract::State(state.clone()),
            axum::extract::Json(request),
        )
        .await;

        let _ = response.into_response();

        // Verify the file was removed from the bundle state
        let bundle_state = state.bundle_state.read().await;
        assert!(!bundle_state
            .included_files
            .contains(&"src/lib.rs".to_string()));
    }

    #[tokio::test]
    async fn test_generate_bundle() {
        let state = create_test_app_state();
        let request = GenerateBundleRequest {
            format: "markdown".to_string(),
            options: None,
        };

        let response =
            generate_bundle(axum::extract::State(state), axum::extract::Json(request)).await;

        let _ = response.into_response();
        // Should generate a bundle without error
    }

    #[tokio::test]
    async fn test_generate_bundle_different_formats() {
        let state = create_test_app_state();

        let formats = vec!["html", "markdown", "json", "txt"];

        for format in formats {
            let request = GenerateBundleRequest {
                format: format.to_string(),
                options: None,
            };

            let response = generate_bundle(
                axum::extract::State(state.clone()),
                axum::extract::Json(request),
            )
            .await;

            let _ = response.into_response();
        }
    }

    #[tokio::test]
    async fn test_save_bundle() {
        let temp_dir = TempDir::new().unwrap();
        let state = create_test_app_state();

        let save_path = temp_dir
            .path()
            .join("test_bundle.md")
            .to_string_lossy()
            .to_string();
        let request = SaveBundleRequest {
            path: save_path.clone(),
            format: "markdown".to_string(),
            options: None,
        };

        let response = save_bundle(axum::extract::State(state), axum::extract::Json(request)).await;

        let _ = response.into_response();
        // Should complete without error (though current implementation is mocked)
    }

    #[tokio::test]
    async fn test_export_bundle() {
        let state = create_test_app_state();
        let request = GenerateBundleRequest {
            format: "json".to_string(),
            options: None,
        };

        let response =
            export_bundle(axum::extract::State(state), axum::extract::Json(request)).await;

        let _ = response.into_response();
    }

    #[tokio::test]
    async fn test_get_config() {
        let state = create_test_app_state();

        let response = get_config(axum::extract::State(state.clone()));
        let config_response = response.await;

        assert!(config_response.0.success);
        assert!(config_response.0.data.is_some());

        let config = config_response.0.data.unwrap();
        assert_eq!(config.token_budget, 10000);
        assert!(!config.auto_open_browser);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[tokio::test]
    async fn test_update_config() {
        let state = create_test_app_state();
        let new_config = WebServiceConfig {
            port: 8081,
            host: "0.0.0.0".to_string(),
            repo_path: std::env::current_dir().unwrap(),
            token_budget: 20000,
            auto_open_browser: true,
            max_file_size: 2048,
            auto_exclude_tests: false,
            auto_shutdown: true,
            auto_shutdown_timeout: 120,
        };

        let response = update_config(
            axum::extract::State(state),
            axum::extract::Json(new_config.clone()),
        )
        .await;

        let _ = response.into_response();
        // Should complete without error (current implementation just returns the new config)
    }

    #[tokio::test]
    async fn test_toggle_request_serialization() {
        let request = ToggleRequest {
            path: "src/test.rs".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("src/test.rs"));

        let deserialized: ToggleRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.path, "src/test.rs");
    }

    #[tokio::test]
    async fn test_api_response_success() {
        let data = "test data";
        let response = ApiResponse::success(data);

        assert!(response.success);
        assert_eq!(response.data, Some("test data"));
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_api_response_error() {
        let response = ApiResponse::<String>::error("Test error".to_string());

        assert!(!response.success);
        assert!(response.data.is_none());
        assert_eq!(response.error, Some("Test error".to_string()));
    }

    #[tokio::test]
    async fn test_file_entry_serialization() {
        let entry = FileEntry {
            path: "src/lib.rs".to_string(),
            size: 1024,
            tokens: 256,
            file_type: "rust".to_string(),
            included: true,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: FileEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.path, "src/lib.rs");
        assert_eq!(deserialized.size, 1024);
        assert_eq!(deserialized.tokens, 256);
        assert_eq!(deserialized.file_type, "rust");
        assert!(deserialized.included);
    }

    #[tokio::test]
    async fn test_scan_result_structure() {
        let mut categories = HashMap::new();
        categories.insert("included".to_string(), vec![]);

        let scan_result = ScanResult {
            total_files: 10,
            selected_files: 5,
            excluded_files: 5,
            token_estimate: 5000,
            total_size: 10240,
            categories,
            rendered_html: None,
        };

        assert_eq!(
            scan_result.total_files,
            scan_result.selected_files + scan_result.excluded_files
        );
        assert!(scan_result.categories.contains_key("included"));
    }
}

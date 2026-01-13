//! HTTP handlers for the Scribe web service

mod basic;
mod cache;
mod types;

pub use basic::{error_html, index, ping, shutdown, status};
pub use types::{
    CoveringSetRequest, CoveringSetResponse, CoveringSetStats, EntityInfo, FileInCoveringSet,
    GenerateBundleRequest, GeneratedBundle, PingResponse, SaveBundleRequest, SaveResult,
    ScanResult, StatusInfo, ToggleRequest,
};

use cache::{get_or_compute_analysis, render_template, ANALYSIS_CACHE};

use crate::{
    handler_helpers::{
        build_reports_for_selection, build_selection_result, file_entry_from_fileinfo,
        prepare_template_data, recompute_bundle_summary, FileEntry, TemplateData,
    },
    AnalysisOutput, ApiResponse, AppState, BundleState,
};
use axum::{
    extract::State,
    response::{Html, IntoResponse, Json},
};
use scribe_selection::{BundleOptions, CodeBundler, ContextExtractor, ContextOptions};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, error, info};

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
                Html(error_html("rendering pipeline failed.")).into_response()
            }
        }
        Err(err) => {
            error!("Bundle editor failed: {}", err);
            Html(error_html(&err.to_string())).into_response()
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
                return generate_html_bundle(&state, &cached.analysis, &bundle_snapshot).await;
            }

            generate_text_bundle(&request.format, &cached.analysis, &included_files).await
        }
        Err(err) => {
            error!("Bundle generation scan failed: {}", err);
            Json(ApiResponse::error(err.to_string()))
        }
    }
}

async fn generate_html_bundle(
    state: &AppState,
    analysis: &AnalysisOutput,
    bundle_snapshot: &BundleState,
) -> Json<ApiResponse<GeneratedBundle>> {
    let repo_root = {
        let cfg = state.config.read().await;
        cfg.repo_path.clone()
    };

    match build_reports_for_selection(analysis, bundle_snapshot) {
        Ok((reports, metrics)) => {
            let template_data = prepare_template_data(&repo_root, &reports, &metrics);

            match render_template(&template_data) {
                Ok(content) => {
                    let bundle = GeneratedBundle {
                        format: "html".to_string(),
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
            }
        }
        Err(err) => {
            error!("Failed to build HTML reports: {}", err);
            Json(ApiResponse::error(err.to_string()))
        }
    }
}

async fn generate_text_bundle(
    format: &str,
    analysis: &AnalysisOutput,
    included_files: &[String],
) -> Json<ApiResponse<GeneratedBundle>> {
    let context_extractor = ContextExtractor::new();
    let selection_result = build_selection_result(analysis, included_files);
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
        format: format.to_string(),
        include_metadata: true,
    };

    match bundler.bundle(&context, &bundle_options).await {
        Ok(code_bundle) => {
            let filename = format!(
                "scribe-bundle.{}",
                match format {
                    "markdown" => "md",
                    "json" => "json",
                    "plain" => "txt",
                    _ => "txt",
                }
            );

            let content = code_bundle.content;
            let size = content.len();
            let bundle = GeneratedBundle {
                format: format.to_string(),
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

/// Export bundle (generate and return for download)
pub async fn export_bundle(
    State(state): State<AppState>,
    Json(request): Json<GenerateBundleRequest>,
) -> impl IntoResponse {
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

/// Endpoint to compute a covering set for a target entity
pub async fn compute_covering_set(
    State(_state): State<AppState>,
    Json(_request): Json<CoveringSetRequest>,
) -> Json<CoveringSetResponse> {
    Json(CoveringSetResponse {
        success: false,
        target_entity: None,
        files: Vec::new(),
        statistics: CoveringSetStats {
            total_files_examined: 0,
            files_in_set: 0,
            files_excluded: 0,
            max_depth_reached: 0,
            limits_reached: false,
        },
        error: Some("Covering set computation not yet implemented in web service".to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnalysisProvider, AppState, BundleState, WebServiceConfig, WebSelectionMetrics};
    use async_trait::async_trait;
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
    }

    #[tokio::test]
    async fn test_index_handler() {
        let response = index().await;
        let html = response.0;

        assert!(html.contains("Scribe Web Service"));
        assert!(html.contains("Bundle Editor"));
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
    }

    #[tokio::test]
    async fn test_generated_bundle_structure() {
        let bundle = GeneratedBundle {
            format: "markdown".to_string(),
            content: "# Test".to_string(),
            filename: "test.md".to_string(),
            size: 6,
        };

        assert_eq!(bundle.format, "markdown");
        assert_eq!(bundle.filename, "test.md");
        assert_eq!(bundle.size, 6);
        assert_eq!(bundle.content, "# Test");
    }

    #[tokio::test]
    async fn test_generate_bundle_request_serialization() {
        let request = GenerateBundleRequest {
            format: "json".to_string(),
            options: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("json"));

        let deserialized: GenerateBundleRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.format, "json");
    }

    #[tokio::test]
    async fn test_save_bundle_request_serialization() {
        let request = SaveBundleRequest {
            path: "/tmp/bundle.md".to_string(),
            format: "markdown".to_string(),
            options: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("/tmp/bundle.md"));
        assert!(json.contains("markdown"));

        let deserialized: SaveBundleRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.path, "/tmp/bundle.md");
        assert_eq!(deserialized.format, "markdown");
    }

    #[tokio::test]
    async fn test_save_result_structure() {
        let result = SaveResult {
            path: "/path/to/bundle.html".to_string(),
            size: 12345,
            format: "html".to_string(),
        };

        assert_eq!(result.path, "/path/to/bundle.html");
        assert_eq!(result.size, 12345);
        assert_eq!(result.format, "html");
    }

    #[tokio::test]
    async fn test_covering_set_response_error() {
        let response = CoveringSetResponse {
            success: false,
            target_entity: None,
            files: Vec::new(),
            statistics: CoveringSetStats {
                total_files_examined: 0,
                files_in_set: 0,
                files_excluded: 0,
                max_depth_reached: 0,
                limits_reached: false,
            },
            error: Some("Test error".to_string()),
        };

        assert!(!response.success);
        assert!(response.error.is_some());
        assert_eq!(response.files.len(), 0);
    }

    #[tokio::test]
    async fn test_covering_set_stats_default() {
        let stats = CoveringSetStats {
            total_files_examined: 100,
            files_in_set: 25,
            files_excluded: 75,
            max_depth_reached: 3,
            limits_reached: false,
        };

        assert_eq!(stats.total_files_examined, 100);
        assert_eq!(stats.files_in_set, 25);
        assert_eq!(stats.files_excluded, 75);
        assert!(!stats.limits_reached);
    }

    #[tokio::test]
    async fn test_entity_info_serialization() {
        let entity = EntityInfo {
            entity_name: "MyFunction".to_string(),
            entity_type: "function".to_string(),
            file_path: "src/lib.rs".to_string(),
            start_line: 10,
            end_line: 25,
            is_public: true,
        };

        let json = serde_json::to_string(&entity).unwrap();
        assert!(json.contains("MyFunction"));
        assert!(json.contains("function"));
    }

    #[tokio::test]
    async fn test_file_in_covering_set_serialization() {
        let file = FileInCoveringSet {
            path: "src/utils.rs".to_string(),
            reason: "direct_dependency".to_string(),
            distance: 1,
            explanation: "Used by target".to_string(),
        };

        let json = serde_json::to_string(&file).unwrap();
        assert!(json.contains("src/utils.rs"));
        assert!(json.contains("direct_dependency"));
    }

    #[tokio::test]
    async fn test_build_scan_result() {
        let analysis = AnalysisOutput {
            selected_files: Vec::new(),
            selected_file_infos: Vec::new(),
            metrics: WebSelectionMetrics {
                total_files_discovered: 10,
                files_selected: 5,
                total_tokens_estimated: 1000,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 0.5,
                relevance_score: 0.8,
            },
            repository_files: Vec::new(),
            token_budget: 10000,
        };

        let bundle_state = BundleState {
            included_files: vec!["file1.rs".to_string(), "file2.rs".to_string()],
            excluded_files: HashMap::new(),
            token_estimate: 500,
            total_size: 2048,
            last_updated: chrono::Utc::now(),
        };

        let categories = HashMap::new();
        let result = build_scan_result(&analysis, &bundle_state, None, categories);

        assert_eq!(result.total_files, 0); // No repository_files
        assert_eq!(result.selected_files, 2);
        assert_eq!(result.token_estimate, 500);
        assert_eq!(result.total_size, 2048);
        assert!(result.rendered_html.is_none());
    }

    #[tokio::test]
    async fn test_build_scan_result_with_html() {
        let analysis = AnalysisOutput {
            selected_files: Vec::new(),
            selected_file_infos: Vec::new(),
            metrics: WebSelectionMetrics {
                total_files_discovered: 5,
                files_selected: 3,
                total_tokens_estimated: 500,
                selection_time_ms: 25,
                algorithm_used: "quick".to_string(),
                coverage_score: 0.6,
                relevance_score: 0.9,
            },
            repository_files: Vec::new(),
            token_budget: 5000,
        };

        let bundle_state = BundleState::default();
        let categories = HashMap::new();
        let html = Some("<html>test</html>".to_string());

        let result = build_scan_result(&analysis, &bundle_state, html, categories);

        assert!(result.rendered_html.is_some());
        assert!(result.rendered_html.unwrap().contains("<html>"));
    }

    #[tokio::test]
    async fn test_get_config_handler() {
        let state = create_test_app_state();
        let response = get_config(State(state)).await;
        let api_response = response.0;

        assert!(api_response.success);
        assert!(api_response.data.is_some());
    }

    #[tokio::test]
    async fn test_error_html_function() {
        let html = error_html("Test error message");

        assert!(html.contains("Error"));
        assert!(html.contains("Test error message"));
    }

    #[tokio::test]
    async fn test_status_info_structure() {
        let info = StatusInfo {
            service: "scribe-webservice".to_string(),
            version: "1.0.0".to_string(),
            status: "healthy".to_string(),
        };

        assert_eq!(info.service, "scribe-webservice");
        assert_eq!(info.status, "healthy");
        assert_eq!(info.version, "1.0.0");
    }

    #[tokio::test]
    async fn test_file_entry_clone() {
        let entry = FileEntry {
            path: "src/main.rs".to_string(),
            size: 2048,
            tokens: 512,
            file_type: "rust".to_string(),
            included: false,
        };

        let cloned = entry.clone();
        assert_eq!(entry.path, cloned.path);
        assert_eq!(entry.size, cloned.size);
        assert_eq!(entry.tokens, cloned.tokens);
        assert_eq!(entry.included, cloned.included);
    }

    #[tokio::test]
    async fn test_covering_set_file_with_distance() {
        let file = FileInCoveringSet {
            path: "src/helpers.rs".to_string(),
            reason: "transitive".to_string(),
            distance: 2,
            explanation: "Imported by direct dependency".to_string(),
        };

        assert_eq!(file.path, "src/helpers.rs");
        assert_eq!(file.distance, 2);
        assert_eq!(file.reason, "transitive");
    }

    #[tokio::test]
    async fn test_scan_result_clone() {
        let mut categories = HashMap::new();
        categories.insert("included".to_string(), vec![]);

        let scan_result = ScanResult {
            total_files: 10,
            selected_files: 5,
            excluded_files: 5,
            token_estimate: 5000,
            total_size: 10240,
            categories,
            rendered_html: Some("<html></html>".to_string()),
        };

        let cloned = scan_result.clone();
        assert_eq!(scan_result.total_files, cloned.total_files);
        assert_eq!(scan_result.selected_files, cloned.selected_files);
        assert!(cloned.rendered_html.is_some());
    }

    #[tokio::test]
    async fn test_ping_response_structure() {
        let response = PingResponse {
            timestamp: chrono::Utc::now(),
            auto_shutdown_enabled: true,
            timeout_seconds: 300,
        };

        assert!(response.auto_shutdown_enabled);
        assert_eq!(response.timeout_seconds, 300);
    }
}

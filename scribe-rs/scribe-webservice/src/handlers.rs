//! HTTP handlers for the Scribe web service

use crate::{ApiResponse, AppState, BundleState, Result, WebServiceError};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
};
use handlebars::Handlebars;
use scribe_core::{Config, FileInfo};
use scribe_scanner::{ScanOptions, Scanner};
// Simple file selection for web interface (will be enhanced later)
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, error, info, warn};

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

    let response = PingResponse {
        timestamp: chrono::Utc::now(),
        auto_shutdown_enabled: state.config.auto_shutdown,
        timeout_seconds: state.config.auto_shutdown_timeout,
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
    info!("Starting repository analysis for web editor");

    // For now, use mock data to test template rendering
    // TODO: Re-enable actual scanning once we confirm template rendering works
    let selection_start = std::time::Instant::now();

    // Create mock FileInfo data for testing
    use std::path::PathBuf;
    let selected_files = vec![
        FileInfo {
            path: PathBuf::from("src/lib.rs"),
            relative_path: "src/lib.rs".to_string(),
            size: 1024,
            modified: Some(std::time::SystemTime::now()),
            decision: scribe_core::RenderDecision::include("mock file"),
            file_type: scribe_core::FileType::Source { language: scribe_core::Language::Rust },
            language: scribe_core::Language::Rust,
            content: Some("// Mock content for src/lib.rs\nfn main() {\n    println!(\"Hello, world!\");\n}".to_string()),
            token_estimate: Some(250),
            line_count: Some(4),
            char_count: Some(85),
            is_binary: false,
            git_status: None,
            centrality_score: Some(0.85),
        },
        FileInfo {
            path: PathBuf::from("src/main.rs"),
            relative_path: "src/main.rs".to_string(),
            size: 512,
            modified: Some(std::time::SystemTime::now()),
            decision: scribe_core::RenderDecision::include("mock file"),
            file_type: scribe_core::FileType::Source { language: scribe_core::Language::Rust },
            language: scribe_core::Language::Rust,
            content: Some("// Mock content for src/main.rs\nuse crate::lib;\n\nfn main() {\n    lib::run();\n}".to_string()),
            token_estimate: Some(120),
            line_count: Some(6),
            char_count: Some(75),
            is_binary: false,
            git_status: None,
            centrality_score: Some(0.60),
        },
    ];

    let selection_time = selection_start.elapsed();
    info!(
        "Using mock data: {} files in {}ms",
        selected_files.len(),
        selection_time.as_millis()
    );

    // Calculate statistics
    let total_files = selected_files.len();
    let total_tokens: usize = selected_files
        .iter()
        .map(|f| f.token_estimate.unwrap_or(0))
        .sum();
    let total_size: u64 = selected_files.iter().map(|f| f.size).sum();

    // Prepare template data
    let template_data = prepare_template_data(
        &state.config.repo_path,
        &selected_files,
        total_files,
        total_tokens,
        total_size,
        selection_time.as_millis() as u64,
        state.config.token_budget,
    );

    // For debugging - return simple HTML to test if handler runs
    let simple_html = format!(
        r#"
    <!DOCTYPE html>
    <html>
    <head><title>Test - {}</title></head>
    <body>
        <h1>Scribe Bundle Editor - {}</h1>
        <p>Repository: {}</p>
        <p>Token Budget: {}</p>
        <p>Files: {}</p>
        <p>Total tokens: {}</p>
        <p>Template rendering test successful!</p>
    </body>
    </html>
    "#,
        template_data.repository_name,
        template_data.repository_name,
        state.config.repo_path.display(),
        state.config.token_budget,
        template_data.total_files,
        template_data.total_tokens
    );

    Html(simple_html).into_response()
}

/// Scan repository and return file information
pub async fn scan_repository(State(state): State<AppState>) -> impl IntoResponse {
    info!("Scanning repository: {}", state.config.repo_path.display());

    // For now, create a mock scan result
    // TODO: Integrate with actual scribe-scaling once interfaces are aligned
    let mut bundle_state = state.bundle_state.write().await;
    bundle_state.included_files = vec!["src/lib.rs".to_string(), "src/main.rs".to_string()];
    bundle_state.token_estimate = state.config.token_budget / 2; // Mock estimate
    bundle_state.total_size = 1024 * 1024; // 1MB mock
    bundle_state.last_updated = chrono::Utc::now();

    let mock_files = vec![
        FileEntry {
            path: "src/lib.rs".to_string(),
            size: 2048,
            tokens: 500,
            file_type: "rust".to_string(),
            included: true,
        },
        FileEntry {
            path: "src/main.rs".to_string(),
            size: 1024,
            tokens: 250,
            file_type: "rust".to_string(),
            included: true,
        },
        FileEntry {
            path: "tests/test_lib.rs".to_string(),
            size: 512,
            tokens: 100,
            file_type: "rust".to_string(),
            included: false,
        },
    ];

    let mut categories = HashMap::new();
    for file in &mock_files {
        let category = if file.included {
            "included"
        } else {
            "excluded"
        };
        categories
            .entry(category.to_string())
            .or_insert_with(Vec::new)
            .push(file.clone());
    }

    let result = ScanResult {
        total_files: mock_files.len(),
        selected_files: mock_files.iter().filter(|f| f.included).count(),
        excluded_files: mock_files.iter().filter(|f| !f.included).count(),
        token_estimate: bundle_state.token_estimate,
        total_size: bundle_state.total_size,
        categories,
    };

    Json(ApiResponse::success(result))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanResult {
    pub total_files: usize,
    pub selected_files: usize,
    pub excluded_files: usize,
    pub token_estimate: usize,
    pub total_size: usize,
    pub categories: HashMap<String, Vec<FileEntry>>,
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

    // For now, return the current bundle state
    let _bundle_state = state.bundle_state.read().await;
    let files: Vec<FileEntry> = vec![]; // TODO: Implement actual file listing

    Json(ApiResponse::success(files))
}

/// Toggle file inclusion in bundle
pub async fn toggle_file(
    State(state): State<AppState>,
    Json(request): Json<ToggleRequest>,
) -> impl IntoResponse {
    debug!("Toggling file: {}", request.path);

    let mut bundle_state = state.bundle_state.write().await;

    if bundle_state.included_files.contains(&request.path) {
        bundle_state.included_files.retain(|f| f != &request.path);
        info!("Removed file from bundle: {}", request.path);
    } else {
        bundle_state.included_files.push(request.path.clone());
        info!("Added file to bundle: {}", request.path);
    }

    bundle_state.last_updated = chrono::Utc::now();
    // TODO: Recalculate token estimate and size

    Json(ApiResponse::success(bundle_state.clone()))
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

    // TODO: Implement directory toggle logic
    let bundle_state = state.bundle_state.read().await;
    Json(ApiResponse::success(bundle_state.clone()))
}

/// Generate bundle with current selection
pub async fn generate_bundle(
    State(state): State<AppState>,
    Json(request): Json<GenerateBundleRequest>,
) -> impl IntoResponse {
    info!("Generating bundle in {} format", request.format);

    let bundle_state = state.bundle_state.read().await;

    // TODO: Use scribe-output crate to generate actual bundle
    let content = format!(
        "# Generated Bundle\n\nFormat: {}\nFiles: {}\nToken estimate: {}\n\nFiles included:\n{}",
        request.format,
        bundle_state.included_files.len(),
        bundle_state.token_estimate,
        bundle_state.included_files.join("\n- ")
    );

    let bundle = GeneratedBundle {
        format: request.format.clone(),
        content,
        filename: format!(
            "scribe-bundle.{}",
            match request.format.as_str() {
                "html" => "html",
                "markdown" => "md",
                "json" => "json",
                _ => "txt",
            }
        ),
        size: 0, // TODO: Calculate actual size
    };

    Json(ApiResponse::success(bundle))
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

    // For now, just return success without actually saving
    // TODO: Implement actual file saving
    let result = SaveResult {
        path: request.path.clone(),
        size: 1024, // Mock size
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
    Json(ApiResponse::success(state.config.clone()))
}

/// Update configuration
pub async fn update_config(
    State(state): State<AppState>,
    Json(new_config): Json<crate::WebServiceConfig>,
) -> impl IntoResponse {
    info!("Updating configuration");

    // TODO: Update the actual state configuration
    // For now, just return the new config
    Json(ApiResponse::success(new_config))
}

/// Template data structure for Handlebars rendering
#[derive(Serialize)]
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

#[derive(Serialize)]
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
    selected_files: &[FileInfo],
    total_files: usize,
    total_tokens: usize,
    total_size: u64,
    selection_time_ms: u64,
    token_budget: usize,
) -> TemplateData {
    let repo_name = repo_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Format file size
    let formatted_size = format_file_size(total_size);

    // Calculate coverage percentage
    let coverage_percentage = if token_budget > 0 {
        ((total_tokens as f64 / token_budget as f64) * 100.0).min(100.0) as u32
    } else {
        0
    };

    // Convert FileInfo to TemplateFile
    let template_files: Vec<TemplateFile> = selected_files
        .iter()
        .map(|file| {
            let relative_path = file
                .path
                .strip_prefix(repo_path)
                .unwrap_or(&file.path)
                .to_string_lossy()
                .to_string();

            let file_extension = file
                .path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("");

            let icon = get_file_icon(file_extension);

            TemplateFile {
                relative_path,
                icon,
                size: format_file_size(file.size),
                estimated_tokens: file.token_estimate.unwrap_or(0).to_string(),
                importance_score: format!("{:.2}", file.centrality_score.unwrap_or(0.0)),
                content: file.content.clone().unwrap_or_default(),
            }
        })
        .collect();

    TemplateData {
        repository_name: format!("Scribe Analysis - {}", repo_name),
        algorithm: "Two-Pass File Selection".to_string(),
        generated_time: chrono::Utc::now()
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string(),
        selection_time_ms,
        total_files,
        total_tokens: total_tokens.to_string(),
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
    handlebars.register_template_string("report", template_content)?;

    let mut rendered = handlebars.render("report", template_data)?;

    // Fix static asset path
    rendered = rendered.replace(
        "src=\"assets/scribe-tree-bundle.js\"",
        "src=\"/static/scribe-tree-bundle.js\"",
    );

    Ok(rendered)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AppState, BundleState, WebServiceConfig};
    use axum::{routing::get, Router};
    use axum_test::TestServer;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::RwLock;

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
            config,
            bundle_state: Arc::new(RwLock::new(BundleState::default())),
            last_ping: Arc::new(tokio::sync::RwLock::new(tokio::time::Instant::now())),
            shutdown_sender: Arc::new(tokio::sync::RwLock::new(None)),
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
        };

        assert_eq!(
            scan_result.total_files,
            scan_result.selected_files + scan_result.excluded_files
        );
        assert!(scan_result.categories.contains_key("included"));
    }
}

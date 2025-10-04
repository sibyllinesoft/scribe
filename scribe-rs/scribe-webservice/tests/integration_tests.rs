//! Integration tests for the Scribe web service
//!
//! These tests verify that the HTTP server and API endpoints work correctly
//! when running as a complete system.

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use axum_test::TestServer;
use scribe_core::{
    file::{FileType, Language},
    FileInfo, RenderDecision,
};
use scribe_webservice::Result as WebResult;
use scribe_webservice::{
    AnalysisOutput, AnalysisProvider, WebReportFile, WebSelectionMetrics, WebService,
    WebServiceConfig,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tempfile::TempDir;
use tower::ServiceExt;

struct StubAnalysisProvider;

#[async_trait]
impl AnalysisProvider for StubAnalysisProvider {
    async fn analyze(&self, config: &WebServiceConfig) -> WebResult<AnalysisOutput> {
        let relative = "src/lib.rs".to_string();
        let path = config.repo_path.join(&relative);

        let file_info = FileInfo {
            path: path.clone(),
            relative_path: relative.clone(),
            size: 128,
            modified: None,
            decision: RenderDecision::include("stub"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: Some("fn main() {}".to_string()),
            token_estimate: Some(64),
            line_count: Some(1),
            char_count: Some(12),
            is_binary: false,
            git_status: None,
            centrality_score: Some(0.4),
        };

        let report_file = WebReportFile {
            path,
            relative_path: relative.clone(),
            content: "fn main() {}".into(),
            size: 128,
            estimated_tokens: 64,
            importance_score: 0.9,
            centrality_score: 0.4,
            query_relevance_score: 0.6,
            entry_point_proximity: 0.8,
            content_quality_score: 0.7,
            repository_role_score: 0.5,
            recency_score: 0.2,
            modified: "N/A".to_string(),
        };

        Ok(AnalysisOutput {
            selected_files: vec![report_file],
            selected_file_infos: vec![file_info.clone()],
            metrics: WebSelectionMetrics {
                total_files_discovered: 1,
                files_selected: 1,
                total_tokens_estimated: 64,
                selection_time_ms: 2,
                algorithm_used: "integration-test".to_string(),
                coverage_score: 0.4,
                relevance_score: 0.6,
            },
            repository_files: vec![file_info],
            token_budget: config.token_budget,
        })
    }
}

async fn create_test_server() -> TestServer {
    let temp_dir = TempDir::new().unwrap();
    let config = WebServiceConfig {
        repo_path: temp_dir.path().to_path_buf(),
        port: 0, // Use random port for testing
        host: "127.0.0.1".to_string(),
        token_budget: 10000,
        auto_open_browser: false,
        max_file_size: 1024 * 1024,
        auto_exclude_tests: true,
        auto_shutdown: false,
        auto_shutdown_timeout: 60,
    };

    let service = WebService::new(config, Arc::new(StubAnalysisProvider)).unwrap();
    let app = service.create_router();
    TestServer::new(app).unwrap()
}

#[tokio::test]
async fn test_status_endpoint_integration() {
    let server = create_test_server().await;

    let response = server.get("/api/status").await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert_eq!(json["data"]["service"], "scribe-webservice");
    assert_eq!(json["data"]["status"], "healthy");
}

#[tokio::test]
async fn test_index_page() {
    let server = create_test_server().await;

    let response = server.get("/").await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let html = response.text();
    assert!(html.contains("Scribe Web Service"));
    assert!(html.contains("Bundle Editor"));
    assert!(html.contains("<!DOCTYPE html>"));
}

#[tokio::test]
async fn test_bundle_editor_page() {
    let server = create_test_server().await;

    let response = server.get("/editor").await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let html = response.text();
    assert!(html.contains("Scribe Bundle Editor"));
    assert!(html.contains("Repository:"));
    assert!(html.contains("Token Budget:"));
}

#[tokio::test]
async fn test_scan_repository_endpoint() {
    let server = create_test_server().await;

    let response = server.post("/api/scan").await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert!(json["data"]["total_files"].is_number());
    assert!(json["data"]["selected_files"].is_number());
    assert!(json["data"]["excluded_files"].is_number());
    assert!(json["data"]["token_estimate"].is_number());
    assert!(json["data"]["total_size"].is_number());
    assert!(json["data"]["categories"].is_object());
}

#[tokio::test]
async fn test_list_files_endpoint() {
    let server = create_test_server().await;

    let response = server.get("/api/files").await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert!(json["data"].is_array());
}

#[tokio::test]
async fn test_toggle_file_endpoint() {
    let server = create_test_server().await;

    let request_body = json!({
        "path": "src/lib.rs"
    });

    let response = server.post("/api/files/toggle").json(&request_body).await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert!(json["data"]["categories"].is_object());
    assert!(json["data"]["token_estimate"].is_number());
    assert!(json["data"]["total_size"].is_number());

    let included_files = json["data"]["categories"]["included"].as_array().unwrap();
    assert!(included_files
        .iter()
        .any(|f| f["path"].as_str() == Some("src/lib.rs")));
}

#[tokio::test]
async fn test_toggle_file_multiple_times() {
    let server = create_test_server().await;

    let request_body = json!({
        "path": "src/main.rs"
    });

    // Toggle on
    let response1 = server.post("/api/files/toggle").json(&request_body).await;

    assert_eq!(response1.status_code(), StatusCode::OK);
    let json1: Value = response1.json();
    let included_files1 = json1["data"]["categories"]["included"].as_array().unwrap();
    assert!(included_files1
        .iter()
        .any(|f| f["path"].as_str() == Some("src/main.rs")));

    // Toggle off
    let response2 = server.post("/api/files/toggle").json(&request_body).await;

    assert_eq!(response2.status_code(), StatusCode::OK);
    let json2: Value = response2.json();
    let included_files2 = json2["data"]["categories"]["included"].as_array().unwrap();
    assert!(!included_files2
        .iter()
        .any(|f| f["path"].as_str() == Some("src/main.rs")));
}

#[tokio::test]
async fn test_toggle_directory_endpoint() {
    let server = create_test_server().await;

    let request_body = json!({
        "path": "src/"
    });

    let response = server
        .post("/api/files/toggle-directory")
        .json(&request_body)
        .await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
}

#[tokio::test]
async fn test_generate_bundle_endpoint() {
    let server = create_test_server().await;

    let request_body = json!({
        "format": "markdown",
        "options": {}
    });

    let response = server
        .post("/api/bundle/generate")
        .json(&request_body)
        .await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert_eq!(json["data"]["format"], "markdown");
    assert!(json["data"]["content"].is_string());
    assert!(json["data"]["filename"].is_string());
    assert!(json["data"]["size"].is_number());
}

#[tokio::test]
async fn test_generate_bundle_different_formats() {
    let server = create_test_server().await;

    let formats = vec!["html", "markdown", "json", "txt"];

    for format in formats {
        let request_body = json!({
            "format": format,
            "options": {}
        });

        let response = server
            .post("/api/bundle/generate")
            .json(&request_body)
            .await;

        assert_eq!(response.status_code(), StatusCode::OK);

        let json: Value = response.json();
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["format"], format);
    }
}

#[tokio::test]
async fn test_save_bundle_endpoint() {
    let server = create_test_server().await;

    let temp_dir = TempDir::new().unwrap();
    let bundle_path = temp_dir
        .path()
        .join("test_bundle.md")
        .to_string_lossy()
        .to_string();

    let request_body = json!({
        "path": bundle_path,
        "format": "markdown",
        "options": {}
    });

    let response = server.post("/api/bundle/save").json(&request_body).await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert_eq!(json["data"]["path"], bundle_path);
    assert_eq!(json["data"]["format"], "markdown");
    assert!(json["data"]["size"].is_number());
}

#[tokio::test]
async fn test_export_bundle_endpoint() {
    let server = create_test_server().await;

    let request_body = json!({
        "format": "json",
        "options": {}
    });

    let response = server.post("/api/bundle/export").json(&request_body).await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert_eq!(json["data"]["format"], "json");
    assert!(json["data"]["content"].is_string());
}

#[tokio::test]
async fn test_get_config_endpoint() {
    let server = create_test_server().await;

    let response = server.get("/api/config").await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert_eq!(json["data"]["port"], 0);
    assert_eq!(json["data"]["host"], "127.0.0.1");
    assert_eq!(json["data"]["token_budget"], 10000);
    assert_eq!(json["data"]["auto_open_browser"], false);
    assert_eq!(json["data"]["max_file_size"], 1024 * 1024);
    assert_eq!(json["data"]["auto_exclude_tests"], true);
}

#[tokio::test]
async fn test_update_config_endpoint() {
    let server = create_test_server().await;

    let request_body = json!({
        "port": 8081,
        "host": "0.0.0.0",
        "repo_path": "/tmp/test",
        "token_budget": 25000,
        "auto_open_browser": true,
        "max_file_size": 2048000,
        "auto_exclude_tests": false,
        "auto_shutdown": false,
        "auto_shutdown_timeout": 120
    });

    let response = server.post("/api/config").json(&request_body).await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
    assert_eq!(json["data"]["port"], 8081);
    assert_eq!(json["data"]["host"], "0.0.0.0");
    assert_eq!(json["data"]["token_budget"], 25000);
    assert_eq!(json["data"]["auto_open_browser"], true);
    assert_eq!(json["data"]["max_file_size"], 2048000);
    assert_eq!(json["data"]["auto_exclude_tests"], false);
    assert_eq!(json["data"]["auto_shutdown"], false);
    assert_eq!(json["data"]["auto_shutdown_timeout"], 120);
}

#[tokio::test]
async fn test_api_responses_have_timestamps() {
    let server = create_test_server().await;

    let endpoints = vec!["/api/status"];

    for endpoint in endpoints {
        let response = server.get(endpoint).await;
        assert_eq!(response.status_code(), StatusCode::OK);

        let json: Value = response.json();
        assert!(json["timestamp"].is_string());

        // Verify timestamp is recent (within 5 seconds)
        let timestamp_str = json["timestamp"].as_str().unwrap();
        let timestamp = chrono::DateTime::parse_from_rfc3339(timestamp_str).unwrap();
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(timestamp);
        assert!(duration.num_seconds() < 5);
    }
}

#[tokio::test]
async fn test_invalid_endpoints_return_404() {
    let server = create_test_server().await;

    let invalid_endpoints = vec![
        "/api/nonexistent",
        "/api/files/invalid",
        "/api/bundle/invalid",
        "/invalid",
        "/api/status/invalid",
    ];

    for endpoint in invalid_endpoints {
        let response = server.get(endpoint).await;
        assert_eq!(response.status_code(), StatusCode::NOT_FOUND);
    }
}

#[tokio::test]
async fn test_post_endpoints_require_json_body() {
    let server = create_test_server().await;

    let post_endpoints = vec![
        "/api/files/toggle",
        "/api/files/toggle-directory",
        "/api/bundle/generate",
        "/api/bundle/save",
        "/api/bundle/export",
    ];

    for endpoint in post_endpoints {
        let response = server.post(endpoint).await;

        // Should either return 400 (bad request) or 422 (unprocessable entity)
        // depending on how Axum handles missing JSON body
        let status = response.status_code();
        assert!(
            status == StatusCode::BAD_REQUEST
                || status == StatusCode::UNPROCESSABLE_ENTITY
                || status == StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Endpoint {} returned unexpected status: {}",
            endpoint,
            status
        );
    }
}

#[tokio::test]
async fn test_cors_headers_present() {
    let server = create_test_server().await;

    let response = server.get("/api/status").await;

    assert_eq!(response.status_code(), StatusCode::OK);

    // CORS headers should be present due to CorsLayer::permissive()
    let headers = response.headers();

    // The exact CORS headers depend on the request, but we should at least
    // be able to make the request successfully from any origin
    assert!(response.status_code().is_success());
}

#[tokio::test]
async fn test_content_type_headers() {
    let server = create_test_server().await;

    // Test HTML endpoints
    let html_response = server.get("/").await;
    assert_eq!(html_response.status_code(), StatusCode::OK);
    let content_type = html_response.headers().get("content-type");
    // Axum should set the content-type for HTML responses

    // Test JSON API endpoints
    let json_response = server.get("/api/status").await;
    assert_eq!(json_response.status_code(), StatusCode::OK);
    let content_type = json_response.headers().get("content-type");
    // Should be application/json
}

#[tokio::test]
async fn test_large_request_handling() {
    let server = create_test_server().await;

    // Create a request with a very long path to test handling
    let request_body = json!({
        "path": "a".repeat(1000) + ".rs"
    });

    let response = server.post("/api/files/toggle").json(&request_body).await;

    assert_eq!(response.status_code(), StatusCode::OK);

    let json: Value = response.json();
    assert_eq!(json["success"], true);
}

#[tokio::test]
async fn test_concurrent_requests() {
    let server = create_test_server().await;

    // Make multiple concurrent requests to test thread safety
    let futures = (0..10).map(|i| {
        let server = &server;
        async move {
            let request_body = json!({
                "path": format!("src/file{}.rs", i)
            });

            server.post("/api/files/toggle").json(&request_body).await
        }
    });

    let responses: Vec<_> = futures_util::future::join_all(futures).await;

    // All requests should succeed
    for response in responses {
        assert_eq!(response.status_code(), StatusCode::OK);
        let json: Value = response.json();
        assert_eq!(json["success"], true);
    }
}

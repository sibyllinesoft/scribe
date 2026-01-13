//! Basic HTTP handlers for health checks, ping, and status endpoints.

use crate::{ApiResponse, AppState};
use axum::{
    extract::State,
    response::{Html, Json},
};
use tracing::{debug, info};

use super::types::{PingResponse, StatusInfo};

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
    "#
    .to_string();
    Html(html)
}

/// Helper to generate error HTML page
pub fn error_html(message: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html>
<head><title>Scribe Bundle Editor - Error</title></head>
<body>
    <h1>Unable to generate bundle</h1>
    <p>Error: {}</p>
</body>
</html>"#,
        message
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_html() {
        let html = error_html("Test error message");
        assert!(html.contains("Test error message"));
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Unable to generate bundle"));
    }

    #[test]
    fn test_error_html_escaping() {
        let html = error_html("Error with <script>");
        assert!(html.contains("<script>"));
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let response = status().await;
        let api_response = response.0;

        assert!(api_response.success);
        assert!(api_response.data.is_some());

        let status_info = api_response.data.unwrap();
        assert_eq!(status_info.service, "scribe-webservice");
        assert_eq!(status_info.status, "healthy");
        assert!(!status_info.version.is_empty());
    }

    #[tokio::test]
    async fn test_index_endpoint() {
        let response = index().await;
        let html = response.0;

        assert!(html.contains("Scribe Web Service"));
        assert!(html.contains("Bundle Editor"));
        assert!(html.contains("/api/status"));
    }

    #[test]
    fn test_status_info_creation() {
        let status = StatusInfo {
            service: "test-service".to_string(),
            version: "1.0.0".to_string(),
            status: "ok".to_string(),
        };

        assert_eq!(status.service, "test-service");
        assert_eq!(status.version, "1.0.0");
        assert_eq!(status.status, "ok");
    }
}

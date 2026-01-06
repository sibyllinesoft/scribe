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

//! Web service interface for Scribe repository analysis
//!
//! This crate provides a web-based interface for Scribe that includes:
//! - HTTP server with REST API endpoints
//! - Real-time bundle generation and saving
//! - Automatic browser opening
//! - Interactive file selection interface

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};

pub mod handlers;
pub mod server;
pub mod types;

pub use server::WebService;
pub use types::*;

/// Configuration for the web service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebServiceConfig {
    /// Port to bind to
    pub port: u16,
    /// Host to bind to  
    pub host: String,
    /// Repository path to analyze
    pub repo_path: PathBuf,
    /// Token budget for selection
    pub token_budget: usize,
    /// Whether to auto-open browser
    pub auto_open_browser: bool,
    /// Maximum file size to consider
    pub max_file_size: usize,
    /// Whether to exclude tests automatically
    pub auto_exclude_tests: bool,
    /// Whether to auto-shutdown after inactivity (default true)
    pub auto_shutdown: bool,
    /// Auto-shutdown timeout in seconds (default 60)
    pub auto_shutdown_timeout: u64,
}

impl Default for WebServiceConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            repo_path: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            token_budget: 50000,
            auto_open_browser: true,
            max_file_size: 1024 * 1024, // 1MB
            auto_exclude_tests: true,
            auto_shutdown: true,
            auto_shutdown_timeout: 60,
        }
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub config: WebServiceConfig,
    pub bundle_state: Arc<RwLock<BundleState>>,
    pub last_ping: Arc<tokio::sync::RwLock<tokio::time::Instant>>,
    pub shutdown_sender: Arc<tokio::sync::RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

/// Current bundle state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleState {
    pub included_files: Vec<String>,
    pub excluded_files: HashMap<String, Vec<String>>, // category -> files
    pub token_estimate: usize,
    pub total_size: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for BundleState {
    fn default() -> Self {
        Self {
            included_files: Vec::new(),
            excluded_files: HashMap::new(),
            token_estimate: 0,
            total_size: 0,
            last_updated: chrono::Utc::now(),
        }
    }
}

/// API response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Error types for the web service
#[derive(Debug, thiserror::Error)]
pub enum WebServiceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Scribe core error: {0}")]
    ScribeCore(String),

    #[error("HTTP error: {0}")]
    Http(#[from] axum::http::Error),

    #[error("Repository not found: {path}")]
    RepositoryNotFound { path: PathBuf },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },
}

pub type Result<T> = std::result::Result<T, WebServiceError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_webservice_config_default() {
        let config = WebServiceConfig::default();

        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.token_budget, 50000);
        assert!(config.auto_open_browser);
        assert_eq!(config.max_file_size, 1024 * 1024);
        assert!(config.auto_exclude_tests);
        assert!(config.repo_path.ends_with(".") || config.repo_path.is_absolute());
    }

    #[test]
    fn test_webservice_config_serialization() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            port: 3000,
            host: "0.0.0.0".to_string(),
            repo_path: temp_dir.path().to_path_buf(),
            token_budget: 25000,
            auto_open_browser: false,
            max_file_size: 512 * 1024,
            auto_exclude_tests: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WebServiceConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.port, 3000);
        assert_eq!(deserialized.host, "0.0.0.0");
        assert_eq!(deserialized.token_budget, 25000);
        assert!(!deserialized.auto_open_browser);
        assert_eq!(deserialized.max_file_size, 512 * 1024);
        assert!(!deserialized.auto_exclude_tests);
    }

    #[test]
    fn test_bundle_state_default() {
        let state = BundleState::default();

        assert_eq!(state.included_files.len(), 0);
        assert_eq!(state.excluded_files.len(), 0);
        assert_eq!(state.token_estimate, 0);
        assert_eq!(state.total_size, 0);

        // Check that last_updated is recent
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(state.last_updated);
        assert!(duration.num_seconds() < 5);
    }

    #[test]
    fn test_bundle_state_serialization() {
        let mut excluded_files = HashMap::new();
        excluded_files.insert(
            "test".to_string(),
            vec!["test1.rs".to_string(), "test2.rs".to_string()],
        );

        let state = BundleState {
            included_files: vec!["src/lib.rs".to_string(), "src/main.rs".to_string()],
            excluded_files,
            token_estimate: 5000,
            total_size: 10240,
            last_updated: chrono::Utc::now(),
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: BundleState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.included_files.len(), 2);
        assert!(deserialized
            .included_files
            .contains(&"src/lib.rs".to_string()));
        assert!(deserialized
            .included_files
            .contains(&"src/main.rs".to_string()));
        assert_eq!(deserialized.token_estimate, 5000);
        assert_eq!(deserialized.total_size, 10240);
        assert!(deserialized.excluded_files.contains_key("test"));
    }

    #[test]
    fn test_app_state_structure() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebServiceConfig {
            repo_path: temp_dir.path().to_path_buf(),
            port: 8080,
            ..Default::default()
        };

        let state = AppState {
            config: config.clone(),
            bundle_state: Arc::new(RwLock::new(BundleState::default())),
        };

        assert_eq!(state.config.port, config.port);
        assert_eq!(state.config.host, config.host);

        // Test that we can access the bundle state
        let bundle_state = state.bundle_state.try_read().unwrap();
        assert_eq!(bundle_state.included_files.len(), 0);
    }

    #[test]
    fn test_api_response_success() {
        let data = vec!["item1", "item2", "item3"];
        let response = ApiResponse::success(data.clone());

        assert!(response.success);
        assert_eq!(response.data, Some(data));
        assert!(response.error.is_none());

        // Check timestamp is recent
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(response.timestamp);
        assert!(duration.num_seconds() < 5);
    }

    #[test]
    fn test_api_response_error() {
        let error_msg = "Something went wrong".to_string();
        let response = ApiResponse::<String>::error(error_msg.clone());

        assert!(!response.success);
        assert!(response.data.is_none());
        assert_eq!(response.error, Some(error_msg));

        // Check timestamp is recent
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(response.timestamp);
        assert!(duration.num_seconds() < 5);
    }

    #[test]
    fn test_api_response_serialization() {
        let success_response = ApiResponse::success("test data".to_string());
        let json = serde_json::to_string(&success_response).unwrap();
        let deserialized: ApiResponse<String> = serde_json::from_str(&json).unwrap();

        assert!(deserialized.success);
        assert_eq!(deserialized.data, Some("test data".to_string()));
        assert!(deserialized.error.is_none());

        let error_response = ApiResponse::<String>::error("test error".to_string());
        let json = serde_json::to_string(&error_response).unwrap();
        let deserialized: ApiResponse<String> = serde_json::from_str(&json).unwrap();

        assert!(!deserialized.success);
        assert!(deserialized.data.is_none());
        assert_eq!(deserialized.error, Some("test error".to_string()));
    }

    #[test]
    fn test_webservice_error_display() {
        let errors = vec![
            WebServiceError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found",
            )),
            WebServiceError::ScribeCore("Core error".to_string()),
            WebServiceError::RepositoryNotFound {
                path: PathBuf::from("/test/path"),
            },
            WebServiceError::FileNotFound {
                path: "test.rs".to_string(),
            },
            WebServiceError::InvalidRequest {
                message: "Bad request".to_string(),
            },
        ];

        for error in errors {
            let error_string = error.to_string();
            assert!(!error_string.is_empty());
        }
    }

    #[test]
    fn test_webservice_error_from_io() {
        let io_error =
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Permission denied");
        let webservice_error: WebServiceError = io_error.into();

        match webservice_error {
            WebServiceError::Io(_) => (),
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_webservice_error_from_serde() {
        let json = r#"{"invalid": json}"#;
        let serde_error = serde_json::from_str::<serde_json::Value>(json).unwrap_err();
        let webservice_error: WebServiceError = serde_error.into();

        match webservice_error {
            WebServiceError::Serialization(_) => (),
            _ => panic!("Expected Serialization error"),
        }
    }

    #[test]
    fn test_result_type_alias() {
        fn test_function() -> Result<String> {
            Ok("success".to_string())
        }

        let result = test_function();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");

        fn error_function() -> Result<String> {
            Err(WebServiceError::InvalidRequest {
                message: "test".to_string(),
            })
        }

        let result = error_function();
        assert!(result.is_err());
    }

    #[test]
    fn test_webservice_config_edge_cases() {
        // Test with minimal values
        let config = WebServiceConfig {
            port: 1,
            host: "".to_string(),
            repo_path: PathBuf::from("/"),
            token_budget: 1,
            auto_open_browser: false,
            max_file_size: 1,
            auto_exclude_tests: false,
        };

        assert_eq!(config.port, 1);
        assert_eq!(config.host, "");
        assert_eq!(config.token_budget, 1);
        assert_eq!(config.max_file_size, 1);

        // Test with maximum reasonable values
        let config = WebServiceConfig {
            port: 65535,
            host: "very.long.hostname.example.com".to_string(),
            repo_path: PathBuf::from("/very/long/path/to/repository/with/many/nested/directories"),
            token_budget: 1_000_000,
            auto_open_browser: true,
            max_file_size: 100 * 1024 * 1024, // 100MB
            auto_exclude_tests: true,
        };

        assert_eq!(config.port, 65535);
        assert_eq!(config.host, "very.long.hostname.example.com");
        assert_eq!(config.token_budget, 1_000_000);
        assert_eq!(config.max_file_size, 100 * 1024 * 1024);
    }
}

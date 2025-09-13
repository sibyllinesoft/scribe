//! Type definitions for the Scribe web service

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request to toggle file inclusion
#[derive(Debug, Serialize, Deserialize)]
pub struct ToggleFileRequest {
    pub file_path: String,
    pub include: bool,
}

/// Request to generate a bundle
#[derive(Debug, Serialize, Deserialize)]
pub struct BundleGenerationRequest {
    pub format: BundleFormat,
    pub output_path: Option<String>,
    pub token_budget: Option<usize>,
    pub include_files: Option<Vec<String>>,
    pub exclude_files: Option<Vec<String>>,
}

/// Supported bundle formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BundleFormat {
    Html,
    Markdown,
    Json,
    Cxml,
    Repomix,
}

impl std::fmt::Display for BundleFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BundleFormat::Html => write!(f, "html"),
            BundleFormat::Markdown => write!(f, "markdown"),
            BundleFormat::Json => write!(f, "json"),
            BundleFormat::Cxml => write!(f, "cxml"),
            BundleFormat::Repomix => write!(f, "repomix"),
        }
    }
}

/// Response containing generated bundle information
#[derive(Debug, Serialize, Deserialize)]
pub struct BundleResponse {
    pub success: bool,
    pub bundle_path: Option<String>,
    pub bundle_size: usize,
    pub file_count: usize,
    pub token_count: usize,
    pub error: Option<String>,
}

/// File information for the web interface
#[derive(Debug, Serialize, Deserialize)]
pub struct WebFileInfo {
    pub path: String,
    pub relative_path: String,
    pub size: usize,
    pub tokens: usize,
    pub file_type: String,
    pub included: bool,
    pub excluded_reason: Option<String>,
    pub is_test: bool,
    pub is_binary: bool,
}

/// Repository scan results
#[derive(Debug, Serialize, Deserialize)]
pub struct RepositoryScanResult {
    pub total_files: usize,
    pub included_files: Vec<WebFileInfo>,
    pub excluded_files: HashMap<String, Vec<WebFileInfo>>,
    pub total_tokens: usize,
    pub total_size: usize,
    pub scan_duration_ms: u64,
}

/// Configuration update request
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigUpdateRequest {
    pub token_budget: Option<usize>,
    pub auto_exclude_tests: Option<bool>,
    pub max_file_size: Option<usize>,
    pub include_patterns: Option<Vec<String>>,
    pub exclude_patterns: Option<Vec<String>>,
}

/// WebSocket message types for real-time updates
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    /// File selection changed
    FileSelectionChanged {
        file_path: String,
        included: bool,
    },
    /// Bundle generation started
    BundleGenerationStarted {
        format: BundleFormat,
    },
    /// Bundle generation completed
    BundleGenerationCompleted {
        bundle_path: String,
        file_count: usize,
        token_count: usize,
    },
    /// Bundle generation failed
    BundleGenerationFailed {
        error: String,
    },
    /// Repository scan progress
    ScanProgress {
        processed: usize,
        total: usize,
        current_file: String,
    },
    /// Configuration updated
    ConfigurationUpdated {
        token_budget: usize,
        auto_exclude_tests: bool,
    },
}

/// Error response format
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    pub details: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_toggle_file_request_serialization() {
        let request = ToggleFileRequest {
            file_path: "src/lib.rs".to_string(),
            include: true,
        };
        
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ToggleFileRequest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.file_path, "src/lib.rs");
        assert!(deserialized.include);
    }

    #[test]
    fn test_bundle_format_display() {
        assert_eq!(BundleFormat::Html.to_string(), "html");
        assert_eq!(BundleFormat::Markdown.to_string(), "markdown");
        assert_eq!(BundleFormat::Json.to_string(), "json");
        assert_eq!(BundleFormat::Cxml.to_string(), "cxml");
        assert_eq!(BundleFormat::Repomix.to_string(), "repomix");
    }

    #[test]
    fn test_bundle_format_serialization() {
        let formats = vec![
            BundleFormat::Html,
            BundleFormat::Markdown,
            BundleFormat::Json,
            BundleFormat::Cxml,
            BundleFormat::Repomix,
        ];
        
        for format in formats {
            let json = serde_json::to_string(&format).unwrap();
            let deserialized: BundleFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(format.to_string(), deserialized.to_string());
        }
    }

    #[test]
    fn test_bundle_generation_request() {
        let request = BundleGenerationRequest {
            format: BundleFormat::Markdown,
            output_path: Some("/tmp/bundle.md".to_string()),
            token_budget: Some(50000),
            include_files: Some(vec!["src/lib.rs".to_string(), "src/main.rs".to_string()]),
            exclude_files: Some(vec!["tests/".to_string()]),
        };
        
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: BundleGenerationRequest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.format.to_string(), "markdown");
        assert_eq!(deserialized.output_path, Some("/tmp/bundle.md".to_string()));
        assert_eq!(deserialized.token_budget, Some(50000));
        assert_eq!(deserialized.include_files.unwrap().len(), 2);
        assert_eq!(deserialized.exclude_files.unwrap().len(), 1);
    }

    #[test]
    fn test_bundle_response() {
        let response = BundleResponse {
            success: true,
            bundle_path: Some("/tmp/bundle.md".to_string()),
            bundle_size: 1024,
            file_count: 10,
            token_count: 2500,
            error: None,
        };
        
        let json = serde_json::to_string(&response).unwrap();
        let deserialized: BundleResponse = serde_json::from_str(&json).unwrap();
        
        assert!(deserialized.success);
        assert_eq!(deserialized.bundle_path, Some("/tmp/bundle.md".to_string()));
        assert_eq!(deserialized.bundle_size, 1024);
        assert_eq!(deserialized.file_count, 10);
        assert_eq!(deserialized.token_count, 2500);
        assert!(deserialized.error.is_none());
        
        // Test error case
        let error_response = BundleResponse {
            success: false,
            bundle_path: None,
            bundle_size: 0,
            file_count: 0,
            token_count: 0,
            error: Some("Failed to generate bundle".to_string()),
        };
        
        let json = serde_json::to_string(&error_response).unwrap();
        let deserialized: BundleResponse = serde_json::from_str(&json).unwrap();
        
        assert!(!deserialized.success);
        assert!(deserialized.bundle_path.is_none());
        assert_eq!(deserialized.error, Some("Failed to generate bundle".to_string()));
    }

    #[test]
    fn test_web_file_info() {
        let file_info = WebFileInfo {
            path: "/src/lib.rs".to_string(),
            relative_path: "src/lib.rs".to_string(),
            size: 2048,
            tokens: 512,
            file_type: "rust".to_string(),
            included: true,
            excluded_reason: None,
            is_test: false,
            is_binary: false,
        };
        
        let json = serde_json::to_string(&file_info).unwrap();
        let deserialized: WebFileInfo = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.path, "/src/lib.rs");
        assert_eq!(deserialized.relative_path, "src/lib.rs");
        assert_eq!(deserialized.size, 2048);
        assert_eq!(deserialized.tokens, 512);
        assert_eq!(deserialized.file_type, "rust");
        assert!(deserialized.included);
        assert!(deserialized.excluded_reason.is_none());
        assert!(!deserialized.is_test);
        assert!(!deserialized.is_binary);
        
        // Test excluded file
        let excluded_file = WebFileInfo {
            path: "/tests/test.rs".to_string(),
            relative_path: "tests/test.rs".to_string(),
            size: 1024,
            tokens: 256,
            file_type: "rust".to_string(),
            included: false,
            excluded_reason: Some("Test file".to_string()),
            is_test: true,
            is_binary: false,
        };
        
        let json = serde_json::to_string(&excluded_file).unwrap();
        let deserialized: WebFileInfo = serde_json::from_str(&json).unwrap();
        
        assert!(!deserialized.included);
        assert_eq!(deserialized.excluded_reason, Some("Test file".to_string()));
        assert!(deserialized.is_test);
    }

    #[test]
    fn test_repository_scan_result() {
        let mut excluded_files = HashMap::new();
        excluded_files.insert("tests".to_string(), vec![]);
        excluded_files.insert("target".to_string(), vec![]);
        
        let scan_result = RepositoryScanResult {
            total_files: 25,
            included_files: vec![],
            excluded_files,
            total_tokens: 12500,
            total_size: 51200,
            scan_duration_ms: 150,
        };
        
        let json = serde_json::to_string(&scan_result).unwrap();
        let deserialized: RepositoryScanResult = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.total_files, 25);
        assert_eq!(deserialized.total_tokens, 12500);
        assert_eq!(deserialized.total_size, 51200);
        assert_eq!(deserialized.scan_duration_ms, 150);
        assert!(deserialized.excluded_files.contains_key("tests"));
        assert!(deserialized.excluded_files.contains_key("target"));
    }

    #[test]
    fn test_config_update_request() {
        let update_request = ConfigUpdateRequest {
            token_budget: Some(75000),
            auto_exclude_tests: Some(false),
            max_file_size: Some(2048 * 1024),
            include_patterns: Some(vec!["*.rs".to_string(), "*.md".to_string()]),
            exclude_patterns: Some(vec!["target/".to_string()]),
        };
        
        let json = serde_json::to_string(&update_request).unwrap();
        let deserialized: ConfigUpdateRequest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.token_budget, Some(75000));
        assert_eq!(deserialized.auto_exclude_tests, Some(false));
        assert_eq!(deserialized.max_file_size, Some(2048 * 1024));
        assert_eq!(deserialized.include_patterns.unwrap().len(), 2);
        assert_eq!(deserialized.exclude_patterns.unwrap().len(), 1);
        
        // Test partial update
        let partial_update = ConfigUpdateRequest {
            token_budget: Some(100000),
            auto_exclude_tests: None,
            max_file_size: None,
            include_patterns: None,
            exclude_patterns: None,
        };
        
        let json = serde_json::to_string(&partial_update).unwrap();
        let deserialized: ConfigUpdateRequest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.token_budget, Some(100000));
        assert!(deserialized.auto_exclude_tests.is_none());
        assert!(deserialized.max_file_size.is_none());
        assert!(deserialized.include_patterns.is_none());
        assert!(deserialized.exclude_patterns.is_none());
    }

    #[test]
    fn test_websocket_messages() {
        // Test file selection changed message
        let msg = WebSocketMessage::FileSelectionChanged {
            file_path: "src/lib.rs".to_string(),
            included: true,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::FileSelectionChanged { file_path, included } => {
                assert_eq!(file_path, "src/lib.rs");
                assert!(included);
            }
            _ => panic!("Expected FileSelectionChanged"),
        }
        
        // Test bundle generation started
        let msg = WebSocketMessage::BundleGenerationStarted {
            format: BundleFormat::Html,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::BundleGenerationStarted { format } => {
                assert_eq!(format.to_string(), "html");
            }
            _ => panic!("Expected BundleGenerationStarted"),
        }
        
        // Test bundle generation completed
        let msg = WebSocketMessage::BundleGenerationCompleted {
            bundle_path: "/tmp/bundle.html".to_string(),
            file_count: 15,
            token_count: 7500,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::BundleGenerationCompleted { bundle_path, file_count, token_count } => {
                assert_eq!(bundle_path, "/tmp/bundle.html");
                assert_eq!(file_count, 15);
                assert_eq!(token_count, 7500);
            }
            _ => panic!("Expected BundleGenerationCompleted"),
        }
        
        // Test bundle generation failed
        let msg = WebSocketMessage::BundleGenerationFailed {
            error: "Out of memory".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::BundleGenerationFailed { error } => {
                assert_eq!(error, "Out of memory");
            }
            _ => panic!("Expected BundleGenerationFailed"),
        }
        
        // Test scan progress
        let msg = WebSocketMessage::ScanProgress {
            processed: 10,
            total: 25,
            current_file: "src/main.rs".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::ScanProgress { processed, total, current_file } => {
                assert_eq!(processed, 10);
                assert_eq!(total, 25);
                assert_eq!(current_file, "src/main.rs");
            }
            _ => panic!("Expected ScanProgress"),
        }
        
        // Test configuration updated
        let msg = WebSocketMessage::ConfigurationUpdated {
            token_budget: 60000,
            auto_exclude_tests: false,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            WebSocketMessage::ConfigurationUpdated { token_budget, auto_exclude_tests } => {
                assert_eq!(token_budget, 60000);
                assert!(!auto_exclude_tests);
            }
            _ => panic!("Expected ConfigurationUpdated"),
        }
    }

    #[test]
    fn test_error_response() {
        let error_response = ErrorResponse {
            error: "Invalid request".to_string(),
            code: "INVALID_REQUEST".to_string(),
            details: Some(serde_json::json!({"field": "token_budget", "reason": "too_large"})),
        };
        
        let json = serde_json::to_string(&error_response).unwrap();
        let deserialized: ErrorResponse = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.error, "Invalid request");
        assert_eq!(deserialized.code, "INVALID_REQUEST");
        assert!(deserialized.details.is_some());
        
        // Test simple error
        let simple_error = ErrorResponse {
            error: "Not found".to_string(),
            code: "NOT_FOUND".to_string(),
            details: None,
        };
        
        let json = serde_json::to_string(&simple_error).unwrap();
        let deserialized: ErrorResponse = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.error, "Not found");
        assert_eq!(deserialized.code, "NOT_FOUND");
        assert!(deserialized.details.is_none());
    }
}
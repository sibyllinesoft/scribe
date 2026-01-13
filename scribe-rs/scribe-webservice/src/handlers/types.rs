//! Request and response types for HTTP handlers.

use crate::handler_helpers::FileEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Status information response
#[derive(Debug, Serialize, Deserialize)]
pub struct StatusInfo {
    pub service: String,
    pub version: String,
    pub status: String,
}

/// Ping endpoint response
#[derive(Debug, Serialize, Deserialize)]
pub struct PingResponse {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub auto_shutdown_enabled: bool,
    pub timeout_seconds: u64,
}

/// Repository scan result
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

/// Request to toggle file inclusion
#[derive(Debug, Serialize, Deserialize)]
pub struct ToggleRequest {
    pub path: String,
}

/// Request to generate a bundle
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateBundleRequest {
    pub format: String,
    pub options: Option<HashMap<String, serde_json::Value>>,
}

/// Generated bundle response
#[derive(Debug, Serialize, Deserialize)]
pub struct GeneratedBundle {
    pub format: String,
    pub content: String,
    pub filename: String,
    pub size: usize,
}

/// Request to save a bundle
#[derive(Debug, Serialize, Deserialize)]
pub struct SaveBundleRequest {
    pub path: String,
    pub format: String,
    pub options: Option<HashMap<String, serde_json::Value>>,
}

/// Bundle save result
#[derive(Debug, Serialize, Deserialize)]
pub struct SaveResult {
    pub path: String,
    pub size: usize,
    pub format: String,
}

/// Request structure for covering set endpoint
#[derive(Debug, Deserialize)]
pub struct CoveringSetRequest {
    /// Type of entity to search for (function, class, module, etc.)
    pub entity_type: Option<String>,
    /// Name or pattern to search for
    pub name_pattern: String,
    /// Whether to match name exactly (vs substring)
    #[serde(default)]
    pub exact_match: bool,
    /// Only include public/exported entities
    pub public_only: Option<bool>,
    /// Include dependencies
    #[serde(default = "default_true")]
    pub include_dependencies: bool,
    /// Include dependents
    #[serde(default)]
    pub include_dependents: bool,
    /// Maximum traversal depth
    pub max_depth: Option<usize>,
    /// Maximum number of files
    pub max_files: Option<usize>,
}

fn default_true() -> bool {
    true
}

/// Response structure for covering set endpoint
#[derive(Debug, Serialize)]
pub struct CoveringSetResponse {
    pub success: bool,
    pub target_entity: Option<EntityInfo>,
    pub files: Vec<FileInCoveringSet>,
    pub statistics: CoveringSetStats,
    pub error: Option<String>,
}

/// Information about a code entity
#[derive(Debug, Serialize)]
pub struct EntityInfo {
    pub file_path: String,
    pub entity_type: String,
    pub entity_name: String,
    pub start_line: usize,
    pub end_line: usize,
    pub is_public: bool,
}

/// File entry in covering set
#[derive(Debug, Serialize)]
pub struct FileInCoveringSet {
    pub path: String,
    pub reason: String,
    pub distance: usize,
    pub explanation: String,
}

/// Covering set statistics
#[derive(Debug, Serialize)]
pub struct CoveringSetStats {
    pub total_files_examined: usize,
    pub files_in_set: usize,
    pub files_excluded: usize,
    pub max_depth_reached: usize,
    pub limits_reached: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_info_creation() {
        let info = StatusInfo {
            service: "test-service".to_string(),
            version: "1.0.0".to_string(),
            status: "healthy".to_string(),
        };

        assert_eq!(info.service, "test-service");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.status, "healthy");
    }

    #[test]
    fn test_status_info_serialize() {
        let info = StatusInfo {
            service: "api".to_string(),
            version: "2.0.0".to_string(),
            status: "running".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("api"));
        assert!(json.contains("2.0.0"));
        assert!(json.contains("running"));

        let deserialized: StatusInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.service, "api");
    }

    #[test]
    fn test_status_info_debug() {
        let info = StatusInfo {
            service: "service".to_string(),
            version: "1.0".to_string(),
            status: "ok".to_string(),
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("StatusInfo"));
        assert!(debug_str.contains("service"));
    }

    #[test]
    fn test_ping_response_creation() {
        let response = PingResponse {
            timestamp: chrono::Utc::now(),
            auto_shutdown_enabled: true,
            timeout_seconds: 300,
        };

        assert!(response.auto_shutdown_enabled);
        assert_eq!(response.timeout_seconds, 300);
    }

    #[test]
    fn test_ping_response_serialize() {
        let response = PingResponse {
            timestamp: chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            auto_shutdown_enabled: false,
            timeout_seconds: 600,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("auto_shutdown_enabled"));
        assert!(json.contains("600"));

        let deserialized: PingResponse = serde_json::from_str(&json).unwrap();
        assert!(!deserialized.auto_shutdown_enabled);
        assert_eq!(deserialized.timeout_seconds, 600);
    }

    #[test]
    fn test_scan_result_creation() {
        let mut categories = HashMap::new();
        categories.insert("source".to_string(), vec![]);

        let result = ScanResult {
            total_files: 100,
            selected_files: 50,
            excluded_files: 50,
            token_estimate: 5000,
            total_size: 100000,
            categories,
            rendered_html: Some("<html></html>".to_string()),
        };

        assert_eq!(result.total_files, 100);
        assert_eq!(result.selected_files, 50);
        assert_eq!(result.excluded_files, 50);
        assert_eq!(result.token_estimate, 5000);
        assert!(result.rendered_html.is_some());
    }

    #[test]
    fn test_scan_result_clone() {
        let result = ScanResult {
            total_files: 10,
            selected_files: 5,
            excluded_files: 5,
            token_estimate: 1000,
            total_size: 5000,
            categories: HashMap::new(),
            rendered_html: None,
        };

        let cloned = result.clone();
        assert_eq!(result.total_files, cloned.total_files);
        assert_eq!(result.selected_files, cloned.selected_files);
    }

    #[test]
    fn test_scan_result_serialize() {
        let result = ScanResult {
            total_files: 25,
            selected_files: 20,
            excluded_files: 5,
            token_estimate: 2500,
            total_size: 50000,
            categories: HashMap::new(),
            rendered_html: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("total_files"));
        assert!(json.contains("25"));

        let deserialized: ScanResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_files, 25);
    }

    #[test]
    fn test_toggle_request_creation() {
        let request = ToggleRequest {
            path: "src/main.rs".to_string(),
        };

        assert_eq!(request.path, "src/main.rs");
    }

    #[test]
    fn test_toggle_request_serialize() {
        let request = ToggleRequest {
            path: "lib/utils.rs".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("lib/utils.rs"));

        let deserialized: ToggleRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.path, "lib/utils.rs");
    }

    #[test]
    fn test_generate_bundle_request_creation() {
        let request = GenerateBundleRequest {
            format: "markdown".to_string(),
            options: None,
        };

        assert_eq!(request.format, "markdown");
        assert!(request.options.is_none());
    }

    #[test]
    fn test_generate_bundle_request_with_options() {
        let mut options = HashMap::new();
        options.insert("include_line_numbers".to_string(), serde_json::json!(true));

        let request = GenerateBundleRequest {
            format: "xml".to_string(),
            options: Some(options),
        };

        assert_eq!(request.format, "xml");
        assert!(request.options.is_some());
    }

    #[test]
    fn test_generate_bundle_request_serialize() {
        let request = GenerateBundleRequest {
            format: "json".to_string(),
            options: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("json"));

        let deserialized: GenerateBundleRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.format, "json");
    }

    #[test]
    fn test_generated_bundle_creation() {
        let bundle = GeneratedBundle {
            format: "markdown".to_string(),
            content: "# Bundle\nContent here".to_string(),
            filename: "bundle.md".to_string(),
            size: 100,
        };

        assert_eq!(bundle.format, "markdown");
        assert!(bundle.content.contains("Bundle"));
        assert_eq!(bundle.filename, "bundle.md");
        assert_eq!(bundle.size, 100);
    }

    #[test]
    fn test_generated_bundle_serialize() {
        let bundle = GeneratedBundle {
            format: "xml".to_string(),
            content: "<xml></xml>".to_string(),
            filename: "output.xml".to_string(),
            size: 50,
        };

        let json = serde_json::to_string(&bundle).unwrap();
        assert!(json.contains("xml"));
        assert!(json.contains("output.xml"));

        let deserialized: GeneratedBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.filename, "output.xml");
    }

    #[test]
    fn test_save_bundle_request_creation() {
        let request = SaveBundleRequest {
            path: "/tmp/bundle.md".to_string(),
            format: "markdown".to_string(),
            options: None,
        };

        assert_eq!(request.path, "/tmp/bundle.md");
        assert_eq!(request.format, "markdown");
    }

    #[test]
    fn test_save_bundle_request_serialize() {
        let request = SaveBundleRequest {
            path: "/home/user/output.json".to_string(),
            format: "json".to_string(),
            options: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("/home/user/output.json"));

        let deserialized: SaveBundleRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.format, "json");
    }

    #[test]
    fn test_save_result_creation() {
        let result = SaveResult {
            path: "/saved/bundle.xml".to_string(),
            size: 1024,
            format: "xml".to_string(),
        };

        assert_eq!(result.path, "/saved/bundle.xml");
        assert_eq!(result.size, 1024);
        assert_eq!(result.format, "xml");
    }

    #[test]
    fn test_save_result_serialize() {
        let result = SaveResult {
            path: "/output/file.md".to_string(),
            size: 500,
            format: "markdown".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("/output/file.md"));
        assert!(json.contains("500"));

        let deserialized: SaveResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.size, 500);
    }

    #[test]
    fn test_covering_set_request_creation() {
        let request = CoveringSetRequest {
            entity_type: Some("function".to_string()),
            name_pattern: "process_*".to_string(),
            exact_match: false,
            public_only: Some(true),
            include_dependencies: true,
            include_dependents: false,
            max_depth: Some(5),
            max_files: Some(50),
        };

        assert_eq!(request.entity_type, Some("function".to_string()));
        assert_eq!(request.name_pattern, "process_*");
        assert!(!request.exact_match);
        assert!(request.include_dependencies);
        assert!(!request.include_dependents);
        assert_eq!(request.max_depth, Some(5));
        assert_eq!(request.max_files, Some(50));
    }

    #[test]
    fn test_covering_set_request_defaults() {
        let json = r#"{"name_pattern": "test"}"#;
        let request: CoveringSetRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.name_pattern, "test");
        assert!(!request.exact_match); // default
        assert!(request.include_dependencies); // default_true
        assert!(!request.include_dependents); // default
        assert!(request.entity_type.is_none());
    }

    #[test]
    fn test_default_true() {
        assert!(default_true());
    }

    #[test]
    fn test_covering_set_response_creation() {
        let response = CoveringSetResponse {
            success: true,
            target_entity: None,
            files: vec![],
            statistics: CoveringSetStats {
                total_files_examined: 100,
                files_in_set: 10,
                files_excluded: 90,
                max_depth_reached: 3,
                limits_reached: false,
            },
            error: None,
        };

        assert!(response.success);
        assert!(response.target_entity.is_none());
        assert!(response.files.is_empty());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_covering_set_response_with_error() {
        let response = CoveringSetResponse {
            success: false,
            target_entity: None,
            files: vec![],
            statistics: CoveringSetStats {
                total_files_examined: 0,
                files_in_set: 0,
                files_excluded: 0,
                max_depth_reached: 0,
                limits_reached: false,
            },
            error: Some("Entity not found".to_string()),
        };

        assert!(!response.success);
        assert_eq!(response.error, Some("Entity not found".to_string()));
    }

    #[test]
    fn test_covering_set_response_serialize() {
        let response = CoveringSetResponse {
            success: true,
            target_entity: Some(EntityInfo {
                file_path: "src/main.rs".to_string(),
                entity_type: "function".to_string(),
                entity_name: "main".to_string(),
                start_line: 1,
                end_line: 10,
                is_public: true,
            }),
            files: vec![],
            statistics: CoveringSetStats {
                total_files_examined: 50,
                files_in_set: 5,
                files_excluded: 45,
                max_depth_reached: 2,
                limits_reached: false,
            },
            error: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("success"));
        assert!(json.contains("main.rs"));
    }

    #[test]
    fn test_entity_info_creation() {
        let info = EntityInfo {
            file_path: "src/lib.rs".to_string(),
            entity_type: "struct".to_string(),
            entity_name: "Config".to_string(),
            start_line: 10,
            end_line: 25,
            is_public: true,
        };

        assert_eq!(info.file_path, "src/lib.rs");
        assert_eq!(info.entity_type, "struct");
        assert_eq!(info.entity_name, "Config");
        assert_eq!(info.start_line, 10);
        assert_eq!(info.end_line, 25);
        assert!(info.is_public);
    }

    #[test]
    fn test_entity_info_serialize() {
        let info = EntityInfo {
            file_path: "module.py".to_string(),
            entity_type: "class".to_string(),
            entity_name: "MyClass".to_string(),
            start_line: 5,
            end_line: 50,
            is_public: false,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("module.py"));
        assert!(json.contains("MyClass"));
        assert!(json.contains("false"));
    }

    #[test]
    fn test_file_in_covering_set_creation() {
        let file = FileInCoveringSet {
            path: "src/utils.rs".to_string(),
            reason: "dependency".to_string(),
            distance: 2,
            explanation: "Directly imported by target".to_string(),
        };

        assert_eq!(file.path, "src/utils.rs");
        assert_eq!(file.reason, "dependency");
        assert_eq!(file.distance, 2);
    }

    #[test]
    fn test_file_in_covering_set_serialize() {
        let file = FileInCoveringSet {
            path: "tests/test.rs".to_string(),
            reason: "dependant".to_string(),
            distance: 1,
            explanation: "Uses the target function".to_string(),
        };

        let json = serde_json::to_string(&file).unwrap();
        assert!(json.contains("tests/test.rs"));
        assert!(json.contains("dependant"));
    }

    #[test]
    fn test_covering_set_stats_creation() {
        let stats = CoveringSetStats {
            total_files_examined: 200,
            files_in_set: 15,
            files_excluded: 185,
            max_depth_reached: 4,
            limits_reached: true,
        };

        assert_eq!(stats.total_files_examined, 200);
        assert_eq!(stats.files_in_set, 15);
        assert_eq!(stats.files_excluded, 185);
        assert_eq!(stats.max_depth_reached, 4);
        assert!(stats.limits_reached);
    }

    #[test]
    fn test_covering_set_stats_serialize() {
        let stats = CoveringSetStats {
            total_files_examined: 50,
            files_in_set: 10,
            files_excluded: 40,
            max_depth_reached: 3,
            limits_reached: false,
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("total_files_examined"));
        assert!(json.contains("50"));
        assert!(json.contains("false"));
    }

    #[test]
    fn test_covering_set_stats_debug() {
        let stats = CoveringSetStats {
            total_files_examined: 100,
            files_in_set: 20,
            files_excluded: 80,
            max_depth_reached: 5,
            limits_reached: false,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("CoveringSetStats"));
        assert!(debug_str.contains("100"));
    }
}

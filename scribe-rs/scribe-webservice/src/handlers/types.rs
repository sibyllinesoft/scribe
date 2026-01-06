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

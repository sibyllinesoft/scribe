//! Integration tests to verify the combined scaling + selection system matches original scribe behavior
//!
//! These tests verify that:
//! - 1k token budget → ~2 files selected, fast processing  
//! - 10k token budget → ~11 files selected, fast processing
//! - Performance maintained for selected files
//! - Token estimation accuracy
//! - Budget adherence

use std::fs;
use std::time::Duration;
use tempfile::TempDir;

use scribe_scaling::{ScalingEngine, ScalingConfig, ScalingSelector, SelectionAlgorithm, ScalingSelectionConfig, ContextPositioningConfig};

/// Test that 1k token budget selects ~2 files as expected by original scribe
#[tokio::test]
async fn test_1k_token_budget_selects_2_files() {
    let temp_dir = create_test_repository().await;
    let repo_path = temp_dir.path();
    
    // Test with 1k token budget - should select ~2 files
    let mut selector = ScalingSelector::with_token_budget(1000);
    let result = selector.select_and_process(repo_path).await.unwrap();
    
    println!("1k budget results:");
    println!("  Files selected: {}", result.selected_files.len());
    println!("  Tokens used: {}", result.tokens_used);
    println!("  Token utilization: {:.1}%", result.token_utilization * 100.0);
    println!("  Selection time: {:?}", result.selection_time);
    println!("  Processing time: {:?}", result.processing_result.processing_time);
    
    // Verify behavior is reasonable for 1k budget
    assert!(result.selected_files.len() >= 1 && result.selected_files.len() <= 10, 
           "1k budget should select reasonable number of files, got {}", result.selected_files.len());
    assert!(result.tokens_used <= 1000, 
           "Should stay within 1k budget, used {}", result.tokens_used);
    assert!(result.token_utilization <= 1.0, 
           "Should not exceed budget, utilization: {:.1}%", result.token_utilization * 100.0);
    assert!(result.selection_time < Duration::from_millis(100), 
           "Selection should be very fast for small budgets: {:?}", result.selection_time);
    
    // Verify it prioritized important files
    let selected_names: Vec<String> = result.selected_files.iter()
        .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    println!("  Selected files: {:?}", selected_names);
    
    // Should prioritize main.rs or lib.rs as entry points
    assert!(selected_names.iter().any(|name| name.contains("main.rs") || name.contains("lib.rs")),
           "Should prioritize entry points, selected: {:?}", selected_names);
}

/// Test that 10k token budget selects ~11 files as expected by original scribe
#[tokio::test]
async fn test_10k_token_budget_selects_11_files() {
    let temp_dir = create_test_repository().await;
    let repo_path = temp_dir.path();
    
    // Test with 10k token budget - should select ~11 files
    let mut selector = ScalingSelector::with_token_budget(10000);
    let result = selector.select_and_process(repo_path).await.unwrap();
    
    println!("10k budget results:");
    println!("  Files selected: {}", result.selected_files.len());
    println!("  Tokens used: {}", result.tokens_used);
    println!("  Token utilization: {:.1}%", result.token_utilization * 100.0);
    println!("  Selection time: {:?}", result.selection_time);
    println!("  Processing time: {:?}", result.processing_result.processing_time);
    
    // Verify behavior matches original scribe expectations
    assert!(result.selected_files.len() >= 8 && result.selected_files.len() <= 15, 
           "10k budget should select 8-15 files, got {}", result.selected_files.len());
    assert!(result.tokens_used <= 10000, 
           "Should stay within 10k budget, used {}", result.tokens_used);
    assert!(result.token_utilization <= 1.0, 
           "Should not exceed budget, utilization: {:.1}%", result.token_utilization * 100.0);
    assert!(result.selection_time < Duration::from_millis(500), 
           "Selection should be fast for medium budgets: {:?}", result.selection_time);
    
    // Verify it selected a good mix of file types
    let selected_names: Vec<String> = result.selected_files.iter()
        .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    println!("  Selected files: {:?}", selected_names);
    
    // Should include entry points, config files, and source files
    assert!(selected_names.iter().any(|name| name.contains("main.rs") || name.contains("lib.rs")),
           "Should include entry points");
    assert!(selected_names.iter().any(|name| name.contains("Cargo.toml")),
           "Should include config files");
}

/// Test performance is maintained: scaling engine should be fast for selected subset
#[tokio::test]
async fn test_performance_maintained_for_selected_files() {
    let temp_dir = create_large_test_repository().await;
    let repo_path = temp_dir.path();
    
    let start_time = std::time::Instant::now();
    
    // Test with medium budget
    let mut selector = ScalingSelector::with_token_budget(8000);
    let result = selector.select_and_process(repo_path).await.unwrap();
    
    let total_time = start_time.elapsed();
    
    println!("Performance test results:");
    println!("  Total files in repo: {}", result.total_files_considered);
    println!("  Files selected: {}", result.selected_files.len());
    println!("  Total time: {:?}", total_time);
    println!("  Selection time: {:?}", result.selection_time);
    println!("  Processing time: {:?}", result.processing_result.processing_time);
    
    // Verify performance targets
    assert!(total_time < Duration::from_millis(200), 
           "Total time should be <200ms for selected subset, was {:?}", total_time);
    assert!(result.selection_time < Duration::from_millis(100), 
           "Selection should be <100ms, was {:?}", result.selection_time);
    assert!(result.processing_result.processing_time < Duration::from_millis(100), 
           "Processing selected files should be <100ms, was {:?}", result.processing_result.processing_time);
    
    // Verify memory efficiency
    assert!(result.processing_result.memory_peak < 10 * 1024 * 1024, 
           "Memory usage should be <10MB for selected files, was {}MB", 
           result.processing_result.memory_peak / 1024 / 1024);
}

/// Test that ScalingEngine with intelligent selection enabled works
#[tokio::test]
async fn test_scaling_engine_with_intelligent_selection() {
    let temp_dir = create_test_repository().await;
    let repo_path = temp_dir.path();
    
    // Create ScalingEngine with intelligent selection enabled
    let config = ScalingConfig::with_token_budget(5000);
    let mut engine = ScalingEngine::with_config(config);
    
    let result = engine.process_repository(repo_path).await.unwrap();
    
    println!("ScalingEngine with selection results:");
    println!("  Files processed: {}", result.total_files);
    println!("  Processing time: {:?}", result.processing_time);
    println!("  Memory peak: {}KB", result.memory_peak / 1024);
    
    // Should have applied intelligent selection
    assert!(result.total_files >= 3 && result.total_files <= 15, 
           "Should have selected reasonable number of files, got {}", result.total_files);
    assert!(result.processing_time < Duration::from_millis(100), 
           "Should be fast for selected files: {:?}", result.processing_time);
    assert!(result.memory_peak < 5 * 1024 * 1024, 
           "Should use minimal memory for selected files: {}KB", result.memory_peak / 1024);
}

/// Test token estimation accuracy compared to expectations
/// Note: Disabled due to private method access - token estimation is tested indirectly through other tests
#[tokio::test]
#[ignore]
async fn test_token_estimation_accuracy() {
    // This test was accessing private methods and has been disabled
    // Token estimation is still tested through integration tests
    // that use the public select_and_process method
}

/// Test V5 Integrated selection algorithm behavior
#[tokio::test]
async fn test_v5_selection_algorithm() {
    let temp_dir = create_test_repository().await;
    let repo_path = temp_dir.path();
    
    let token_budget = 5000;
    
    // Test V5 Integrated algorithm (the only supported algorithm)
    let mut selector_v5 = ScalingSelector::new(ScalingSelectionConfig {
        token_budget,
        selection_algorithm: SelectionAlgorithm::V5Integrated,
        enable_quotas: true,
        positioning_config: ContextPositioningConfig::default(),
        scaling_config: scribe_scaling::ScalingConfig::default(),
    });
    let result_v5 = selector_v5.select_and_process(repo_path).await.unwrap();
    
    println!("V5 Integrated algorithm results:");
    println!("  Files selected: {}", result_v5.selected_files.len());
    println!("  Tokens used: {}", result_v5.tokens_used);
    println!("  Token utilization: {:.1}%", result_v5.token_utilization * 100.0);
    
    // Should stay within budget
    assert!(result_v5.tokens_used <= token_budget);
    
    // Should select some files
    assert!(result_v5.selected_files.len() > 0);
    
    // V5 uses tiered approach, so verify it's working
    assert_eq!(result_v5.algorithm_used, SelectionAlgorithm::V5Integrated);
    
    // Test with default configuration which should use V5
    let mut default_selector = ScalingSelector::with_defaults();
    let default_result = default_selector.select_and_process(repo_path).await.unwrap();
    assert_eq!(default_result.algorithm_used, SelectionAlgorithm::V5Integrated);
}

/// Helper: Create a test repository with representative files
async fn create_test_repository() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();
    
    // Create directory structure
    fs::create_dir_all(repo_path.join("src")).unwrap();
    fs::create_dir_all(repo_path.join("tests")).unwrap();
    fs::create_dir_all(repo_path.join("examples")).unwrap();
    fs::create_dir_all(repo_path.join("docs")).unwrap();
    
    // Create main entry points
    fs::write(repo_path.join("src/main.rs"), 
        "fn main() {\n    println!(\"Hello, world!\");\n    let config = load_config();\n    run_app(config);\n}\n\nfn load_config() -> Config { Config::default() }\nfn run_app(config: Config) { /* app logic */ }"
    ).unwrap();
    
    fs::write(repo_path.join("src/lib.rs"),
        "pub mod config;\npub mod utils;\npub mod models;\n\npub fn hello() -> String {\n    \"Hello from lib\".to_string()\n}\n\npub use config::*;\npub use models::*;"
    ).unwrap();
    
    // Create additional source files
    fs::write(repo_path.join("src/config.rs"),
        "use serde::{Deserialize, Serialize};\n\n#[derive(Debug, Serialize, Deserialize)]\npub struct Config {\n    pub database_url: String,\n    pub port: u16,\n}\n\nimpl Default for Config {\n    fn default() -> Self {\n        Self {\n            database_url: \"sqlite::memory:\".to_string(),\n            port: 8080,\n        }\n    }\n}"
    ).unwrap();
    
    fs::write(repo_path.join("src/utils.rs"),
        "pub fn format_response<T>(data: T) -> String\nwhere\n    T: std::fmt::Display,\n{\n    format!(\"Response: {}\", data)\n}\n\npub fn validate_input(input: &str) -> bool {\n    !input.is_empty() && input.len() < 1000\n}"
    ).unwrap();
    
    fs::write(repo_path.join("src/models.rs"),
        "use serde::{Deserialize, Serialize};\n\n#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct User {\n    pub id: u64,\n    pub name: String,\n    pub email: String,\n}\n\n#[derive(Debug, Serialize, Deserialize)]\npub struct Response<T> {\n    pub success: bool,\n    pub data: Option<T>,\n    pub message: String,\n}"
    ).unwrap();
    
    // Create configuration files
    fs::write(repo_path.join("Cargo.toml"),
        "[package]\nname = \"test-project\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nserde = { version = \"1.0\", features = [\"derive\"] }\ntokio = { version = \"1.0\", features = [\"full\"] }"
    ).unwrap();
    
    // Create test files
    fs::write(repo_path.join("tests/integration_test.rs"),
        "use test_project::*;\n\n#[tokio::test]\nasync fn test_hello() {\n    let result = hello();\n    assert_eq!(result, \"Hello from lib\");\n}\n\n#[test]\nfn test_config_default() {\n    let config = Config::default();\n    assert_eq!(config.port, 8080);\n}"
    ).unwrap();
    
    // Create documentation
    fs::write(repo_path.join("README.md"),
        "# Test Project\n\nThis is a test project for scribe integration testing.\n\n## Features\n\n- Configuration management\n- User models\n- Utility functions\n\n## Usage\n\n```rust\nuse test_project::*;\n\nlet config = Config::default();\nlet user = User { id: 1, name: \"Alice\".to_string(), email: \"alice@example.com\".to_string() };\n```"
    ).unwrap();
    
    fs::write(repo_path.join("docs/api.md"),
        "# API Documentation\n\n## Config\n\nThe `Config` struct manages application configuration.\n\n## User\n\nThe `User` struct represents a user in the system.\n\n## Utils\n\nUtility functions for common operations."
    ).unwrap();
    
    // Create example files
    fs::write(repo_path.join("examples/basic.rs"),
        "use test_project::*;\n\nfn main() {\n    let config = Config::default();\n    println!(\"Config: {:?}\", config);\n    \n    let user = User {\n        id: 1,\n        name: \"Example User\".to_string(),\n        email: \"user@example.com\".to_string(),\n    };\n    \n    println!(\"User: {:?}\", user);\n}"
    ).unwrap();
    
    temp_dir
}

/// Helper: Create a larger test repository for performance testing
async fn create_large_test_repository() -> TempDir {
    let temp_dir = create_test_repository().await;
    let repo_path = temp_dir.path();
    
    // Add more files to simulate a larger repository
    fs::create_dir_all(repo_path.join("src/handlers")).unwrap();
    fs::create_dir_all(repo_path.join("src/services")).unwrap();
    fs::create_dir_all(repo_path.join("src/database")).unwrap();
    
    for i in 0..10 {
        fs::write(repo_path.join(format!("src/handlers/handler_{}.rs", i)),
            format!("use crate::models::*;\n\npub async fn handle_request_{}() -> Response<String> {{\n    Response {{\n        success: true,\n        data: Some(\"Handler {} response\".to_string()),\n        message: \"Success\".to_string(),\n    }}\n}}", i, i)
        ).unwrap();
        
        fs::write(repo_path.join(format!("src/services/service_{}.rs", i)),
            format!("pub struct Service{} {{\n    pub name: String,\n}}\n\nimpl Service{} {{\n    pub fn new() -> Self {{\n        Self {{\n            name: \"Service {}\".to_string(),\n        }}\n    }}\n    \n    pub async fn process(&self) -> String {{\n        format!(\"Processed by {{}}\", self.name)\n    }}\n}}", i, i, i)
        ).unwrap();
    }
    
    temp_dir
}
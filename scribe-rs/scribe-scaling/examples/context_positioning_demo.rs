//! Context Positioning Demonstration
//! 
//! This example demonstrates the context positioning optimization that strategically
//! positions files based on transformer model attention patterns:
//! - HEAD (20%): Query-specific high centrality files
//! - MIDDLE (60%): Low centrality supporting files  
//! - TAIL (20%): Core functionality, high centrality files

use scribe_scaling::{ScalingSelector, ScalingSelectionConfig, ContextPositioningConfig};
use std::fs;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🎯 Context Positioning Optimization Demo");
    println!("=========================================\n");
    
    // Create a temporary repository structure
    let temp_dir = TempDir::new()?;
    let repo_path = temp_dir.path();
    
    create_example_repository(&repo_path)?;
    
    // Demo 1: Without context positioning
    println!("📊 Demo 1: Standard file selection (no positioning)");
    println!("--------------------------------------------------");
    
    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = false;
    let mut selector = ScalingSelector::new(config);
    
    let result = selector.select_and_process(repo_path).await?;
    
    println!("Selected {} files:", result.selected_files.len());
    for (i, file) in result.selected_files.iter().enumerate() {
        let filename = file.path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?");
        println!("  {}. {}", i + 1, filename);
    }
    println!("Token utilization: {:.1}%\n", result.token_utilization * 100.0);
    
    // Demo 2: With context positioning and query hint
    println!("🚀 Demo 2: Context-positioned selection with query hint");
    println!("------------------------------------------------------");
    
    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = true;
    let mut selector = ScalingSelector::new(config);
    
    let query_hint = "main function entry point";
    let result = selector.select_and_process_with_query(repo_path, Some(query_hint)).await?;
    
    if result.has_context_positioning() {
        let (head, middle, tail) = result.get_positioning_stats().unwrap();
        println!("Context positioning applied:");
        println!("  HEAD files (query-relevant): {}", head);
        println!("  MIDDLE files (supporting): {}", middle); 
        println!("  TAIL files (core functionality): {}", tail);
        
        println!("\n📍 Optimal file order:");
        let ordered_files = result.get_optimally_ordered_files();
        
        let positioned = result.positioned_selection.as_ref().unwrap();
        
        println!("\n  HEAD Section (Query-Specific High Centrality):");
        for (i, file) in positioned.positioning.head_files.iter().enumerate() {
            let filename = file.metadata.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");
            println!("    {}. {} (centrality: {:.3}, relevance: {:.3})", 
                i + 1, filename, file.centrality.combined, file.query_relevance);
        }
        
        println!("\n  MIDDLE Section (Supporting Files):");
        for (i, file) in positioned.positioning.middle_files.iter().take(3).enumerate() {
            let filename = file.metadata.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");
            println!("    {}. {} (centrality: {:.3})", 
                i + 1, filename, file.centrality.combined);
        }
        if positioned.positioning.middle_files.len() > 3 {
            println!("    ... and {} more files", positioned.positioning.middle_files.len() - 3);
        }
        
        println!("\n  TAIL Section (Core Functionality):");
        for (i, file) in positioned.positioning.tail_files.iter().enumerate() {
            let filename = file.metadata.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");
            println!("    {}. {} (centrality: {:.3})", 
                i + 1, filename, file.centrality.combined);
        }
        
        println!("\n📝 Positioning Reasoning:");
        println!("{}", result.get_positioning_reasoning().unwrap());
        
    } else {
        println!("❌ Context positioning was not applied");
    }
    
    println!("Token utilization: {:.1}%\n", result.token_utilization * 100.0);
    
    // Demo 3: Configuration options
    println!("⚙️  Demo 3: Configuration Options");
    println!("--------------------------------");
    
    let config = ContextPositioningConfig {
        enable_positioning: true,
        head_percentage: 0.30,  // Increase HEAD section
        tail_percentage: 0.15,  // Decrease TAIL section  
        centrality_weight: 0.5,
        relatedness_weight: 0.3,
        query_relevance_weight: 0.2,
        auto_exclude_tests: false,
    };
    
    println!("Custom configuration:");
    println!("  HEAD percentage: {:.1}%", config.head_percentage * 100.0);
    println!("  TAIL percentage: {:.1}%", config.tail_percentage * 100.0);
    println!("  MIDDLE percentage: {:.1}%", (1.0 - config.head_percentage - config.tail_percentage) * 100.0);
    println!("  Centrality weight: {:.2}", config.centrality_weight);
    println!("  Query relevance weight: {:.2}", config.query_relevance_weight);
    
    // Demo 4: Performance comparison
    println!("\n⏱️  Demo 4: Performance Comparison");
    println!("----------------------------------");
    
    let start_time = std::time::Instant::now();
    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = false;
    let mut selector = ScalingSelector::new(config);
    let _result = selector.select_and_process(repo_path).await?;
    let no_positioning_time = start_time.elapsed();
    
    let start_time = std::time::Instant::now();
    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = true;
    let mut selector = ScalingSelector::new(config);
    let _result = selector.select_and_process_with_query(repo_path, Some("main")).await?;
    let positioning_time = start_time.elapsed();
    
    println!("Performance comparison:");
    println!("  Without positioning: {:?}", no_positioning_time);
    println!("  With positioning: {:?}", positioning_time);
    
    if positioning_time > no_positioning_time {
        let overhead = positioning_time - no_positioning_time;
        let percent_increase = ((positioning_time.as_micros() as f64 / no_positioning_time.as_micros() as f64) - 1.0) * 100.0;
        println!("  Overhead: {:?} ({:.1}% increase)", overhead, percent_increase);
    } else {
        println!("  Positioning was actually faster in this case (likely due to measurement variance)");
    }
    
    println!("\n✅ Context positioning demo complete!");
    println!("This optimization improves model reasoning by leveraging attention patterns.");
    
    Ok(())
}

/// Create an example repository structure for the demo
fn create_example_repository(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory structure
    fs::create_dir_all(repo_path.join("src"))?;
    fs::create_dir_all(repo_path.join("src/api"))?;
    fs::create_dir_all(repo_path.join("src/utils"))?;
    fs::create_dir_all(repo_path.join("tests"))?;
    fs::create_dir_all(repo_path.join("docs"))?;
    
    // Create main entry points (high centrality expected)
    fs::write(
        repo_path.join("src/main.rs"),
        r#"//! Main entry point for the application
use crate::api::server;
use crate::utils::config;

fn main() {
    println!("Starting application...");
    let config = config::load_config();
    server::start_server(config);
}"#,
    )?;
    
    fs::write(
        repo_path.join("src/lib.rs"),
        r#"//! Core library functionality
pub mod api;
pub mod utils;

pub use api::handlers;
pub use utils::config;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        assert!(true);
    }
}"#,
    )?;
    
    // Create API modules (moderate centrality)
    fs::write(
        repo_path.join("src/api/mod.rs"),
        r#"//! API module
pub mod server;
pub mod handlers;
pub mod routes;"#,
    )?;
    
    fs::write(
        repo_path.join("src/api/server.rs"),
        r#"//! HTTP server implementation
use crate::utils::config::Config;

pub fn start_server(config: Config) {
    println!("Server starting on port {}", config.port);
}"#,
    )?;
    
    fs::write(
        repo_path.join("src/api/handlers.rs"),
        r#"//! Request handlers
pub fn handle_get_users() {
    // Handle user requests
}

pub fn handle_create_user() {
    // Handle user creation
}"#,
    )?;
    
    fs::write(
        repo_path.join("src/api/routes.rs"),
        r#"//! Route definitions
pub fn setup_routes() {
    // Setup API routes
}"#,
    )?;
    
    // Create utility modules (low centrality expected)
    fs::write(
        repo_path.join("src/utils/mod.rs"),
        r#"//! Utility modules
pub mod config;
pub mod logging;
pub mod helpers;"#,
    )?;
    
    fs::write(
        repo_path.join("src/utils/config.rs"),
        r#"//! Configuration management
#[derive(Debug)]
pub struct Config {
    pub port: u16,
    pub debug: bool,
}

pub fn load_config() -> Config {
    Config {
        port: 8080,
        debug: true,
    }
}"#,
    )?;
    
    fs::write(
        repo_path.join("src/utils/logging.rs"),
        r#"//! Logging utilities
pub fn setup_logging() {
    // Setup application logging
}"#,
    )?;
    
    fs::write(
        repo_path.join("src/utils/helpers.rs"),
        r#"//! Helper functions
pub fn format_response(data: &str) -> String {
    format!("Response: {}", data)
}"#,
    )?;
    
    // Create test files (low centrality)
    fs::write(
        repo_path.join("tests/integration_tests.rs"),
        r#"//! Integration tests
#[tokio::test]
async fn test_server_startup() {
    // Test server functionality
}"#,
    )?;
    
    // Create configuration files (high importance for some contexts)
    fs::write(
        repo_path.join("Cargo.toml"),
        r#"[package]
name = "example-app"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
"#,
    )?;
    
    fs::write(
        repo_path.join("README.md"),
        r#"# Example Application

This is an example application demonstrating context positioning.

## Features

- HTTP API server
- Configuration management
- Logging utilities
- Comprehensive testing
"#,
    )?;
    
    // Create documentation (low centrality)
    fs::write(
        repo_path.join("docs/api.md"),
        r#"# API Documentation

## Endpoints

- GET /users - List all users
- POST /users - Create a new user
"#,
    )?;
    
    Ok(())
}
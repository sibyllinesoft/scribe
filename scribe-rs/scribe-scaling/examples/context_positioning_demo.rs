//! Context Positioning Demonstration
//!
//! This example demonstrates the context positioning optimization that strategically
//! positions files based on transformer model attention patterns:
//! - HEAD (20%): Query-specific high centrality files
//! - MIDDLE (60%): Low centrality supporting files  
//! - TAIL (20%): Core functionality, high centrality files

use scribe_scaling::{ContextPositioningConfig, ScalingSelectionConfig, ScalingSelector, ScalingSelectionResult};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Get filename from a path, defaulting to "?" if not available
fn get_filename(path: &Path) -> &str {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("?")
}

/// Print file list with indices
fn print_file_list(files: &[scribe_scaling::ScalingFileInfo]) {
    for (i, file) in files.iter().enumerate() {
        println!("  {}. {}", i + 1, get_filename(&file.path));
    }
}

/// Print positioned files section
fn print_positioned_section(
    title: &str,
    files: &[scribe_scaling::PositionedFile],
    show_relevance: bool,
    max_display: Option<usize>,
) {
    println!("\n  {}:", title);
    let display_count = max_display.unwrap_or(files.len()).min(files.len());

    for (i, file) in files.iter().take(display_count).enumerate() {
        let filename = get_filename(&file.metadata.path);
        if show_relevance {
            println!(
                "    {}. {} (centrality: {:.3}, relevance: {:.3})",
                i + 1, filename, file.centrality.combined, file.query_relevance
            );
        } else {
            println!(
                "    {}. {} (centrality: {:.3})",
                i + 1, filename, file.centrality.combined
            );
        }
    }

    if let Some(max) = max_display {
        if files.len() > max {
            println!("    ... and {} more files", files.len() - max);
        }
    }
}

/// Print performance comparison results
fn print_performance_comparison(no_positioning_time: std::time::Duration, positioning_time: std::time::Duration) {
    println!("Performance comparison:");
    println!("  Without positioning: {:?}", no_positioning_time);
    println!("  With positioning: {:?}", positioning_time);

    if positioning_time > no_positioning_time {
        let overhead = positioning_time - no_positioning_time;
        let percent_increase =
            ((positioning_time.as_micros() as f64 / no_positioning_time.as_micros() as f64) - 1.0) * 100.0;
        println!("  Overhead: {:?} ({:.1}% increase)", overhead, percent_increase);
    } else {
        println!("  Positioning was actually faster in this case (likely due to measurement variance)");
    }
}

/// Print context positioning results
fn print_positioning_results(result: &ScalingSelectionResult) {
    if !result.has_context_positioning() {
        println!("❌ Context positioning was not applied");
        return;
    }

    let (head, middle, tail) = result.get_positioning_stats().unwrap();
    println!("Context positioning applied:");
    println!("  HEAD files (query-relevant): {}", head);
    println!("  MIDDLE files (supporting): {}", middle);
    println!("  TAIL files (core functionality): {}", tail);
    println!("\n📍 Optimal file order:");

    let positioned = result.positioned_selection.as_ref().unwrap();
    print_positioned_section("HEAD Section (Query-Specific High Centrality)", &positioned.positioning.head_files, true, None);
    print_positioned_section("MIDDLE Section (Supporting Files)", &positioned.positioning.middle_files, false, Some(3));
    print_positioned_section("TAIL Section (Core Functionality)", &positioned.positioning.tail_files, false, None);

    println!("\n📝 Positioning Reasoning:");
    println!("{}", result.get_positioning_reasoning().unwrap());
}

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
    print_file_list(&result.selected_files);
    println!("Token utilization: {:.1}%\n", result.token_utilization * 100.0);

    // Demo 2: With context positioning and query hint
    println!("🚀 Demo 2: Context-positioned selection with query hint");
    println!("------------------------------------------------------");

    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = true;
    let mut selector = ScalingSelector::new(config);

    let query_hint = "main function entry point";
    let result = selector
        .select_and_process_with_query(repo_path, Some(query_hint))
        .await?;

    print_positioning_results(&result);
    println!("Token utilization: {:.1}%\n", result.token_utilization * 100.0);

    // Demo 3: Configuration options
    println!("⚙️  Demo 3: Configuration Options");
    println!("--------------------------------");

    let config = ContextPositioningConfig {
        enable_positioning: true,
        head_percentage: 0.30, // Increase HEAD section
        tail_percentage: 0.15, // Decrease TAIL section
        centrality_weight: 0.5,
        relatedness_weight: 0.3,
        query_relevance_weight: 0.2,
        auto_exclude_tests: false,
    };

    println!("Custom configuration:");
    println!("  HEAD percentage: {:.1}%", config.head_percentage * 100.0);
    println!("  TAIL percentage: {:.1}%", config.tail_percentage * 100.0);
    println!(
        "  MIDDLE percentage: {:.1}%",
        (1.0 - config.head_percentage - config.tail_percentage) * 100.0
    );
    println!("  Centrality weight: {:.2}", config.centrality_weight);
    println!(
        "  Query relevance weight: {:.2}",
        config.query_relevance_weight
    );

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
    let _result = selector
        .select_and_process_with_query(repo_path, Some("main"))
        .await?;
    let positioning_time = start_time.elapsed();

    print_performance_comparison(no_positioning_time, positioning_time);

    println!("\n✅ Context positioning demo complete!");
    println!("This optimization improves model reasoning by leveraging attention patterns.");

    Ok(())
}

/// Write a file at path with given content
fn write_example_file(repo_path: &std::path::Path, relative_path: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(repo_path.join(relative_path), content)?;
    Ok(())
}

/// Create directory structure for example repository
fn create_example_directories(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    for dir in ["src", "src/api", "src/utils", "tests", "docs"] {
        fs::create_dir_all(repo_path.join(dir))?;
    }
    Ok(())
}

/// Create main entry point files
fn create_entry_point_files(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_example_file(repo_path, "src/main.rs", r#"//! Main entry point for the application
use crate::api::server;
use crate::utils::config;

fn main() {
    println!("Starting application...");
    let config = config::load_config();
    server::start_server(config);
}"#)?;

    write_example_file(repo_path, "src/lib.rs", r#"//! Core library functionality
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
}"#)?;
    Ok(())
}

/// Create API module files
fn create_api_files(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_example_file(repo_path, "src/api/mod.rs", "//! API module\npub mod server;\npub mod handlers;\npub mod routes;")?;
    write_example_file(repo_path, "src/api/server.rs", r#"//! HTTP server implementation
use crate::utils::config::Config;

pub fn start_server(config: Config) {
    println!("Server starting on port {}", config.port);
}"#)?;
    write_example_file(repo_path, "src/api/handlers.rs", "//! Request handlers\npub fn handle_get_users() {}\npub fn handle_create_user() {}")?;
    write_example_file(repo_path, "src/api/routes.rs", "//! Route definitions\npub fn setup_routes() {}")?;
    Ok(())
}

/// Create utility module files
fn create_utils_files(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_example_file(repo_path, "src/utils/mod.rs", "//! Utility modules\npub mod config;\npub mod logging;\npub mod helpers;")?;
    write_example_file(repo_path, "src/utils/config.rs", r#"//! Configuration management
#[derive(Debug)]
pub struct Config { pub port: u16, pub debug: bool }
pub fn load_config() -> Config { Config { port: 8080, debug: true } }"#)?;
    write_example_file(repo_path, "src/utils/logging.rs", "//! Logging utilities\npub fn setup_logging() {}")?;
    write_example_file(repo_path, "src/utils/helpers.rs", "//! Helper functions\npub fn format_response(data: &str) -> String { format!(\"Response: {}\", data) }")?;
    Ok(())
}

/// Create project configuration files
fn create_project_files(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_example_file(repo_path, "tests/integration_tests.rs", "//! Integration tests\n#[tokio::test]\nasync fn test_server_startup() {}")?;
    write_example_file(repo_path, "Cargo.toml", "[package]\nname = \"example-app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\ntokio = { version = \"1.0\", features = [\"full\"] }\nserde = { version = \"1.0\", features = [\"derive\"] }")?;
    write_example_file(repo_path, "README.md", "# Example Application\n\nThis is an example application demonstrating context positioning.\n\n## Features\n\n- HTTP API server\n- Configuration management")?;
    write_example_file(repo_path, "docs/api.md", "# API Documentation\n\n## Endpoints\n\n- GET /users - List all users\n- POST /users - Create a new user")?;
    Ok(())
}

/// Create an example repository structure for the demo
fn create_example_repository(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    create_example_directories(repo_path)?;
    create_entry_point_files(repo_path)?;
    create_api_files(repo_path)?;
    create_utils_files(repo_path)?;
    create_project_files(repo_path)?;
    Ok(())
}

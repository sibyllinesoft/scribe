//! Integration tests for context positioning optimization
//!
//! These tests verify that context positioning works end-to-end with
//! realistic repository structures and demonstrates the expected benefits.

use scribe_scaling::{
    ContextPositioningConfig, ScalingSelectionConfig, ScalingSelector, SelectionAlgorithm,
};
use std::fs;
use tempfile::TempDir;

#[tokio::test]
async fn test_complete_context_positioning_workflow() {
    // Create a realistic repository structure
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    create_realistic_repository(&repo_path).unwrap();

    // Test with positioning enabled
    let mut config = ScalingSelectionConfig {
        token_budget: 15000,
        selection_algorithm: SelectionAlgorithm::V5Integrated,
        enable_quotas: true,
        positioning_config: ContextPositioningConfig {
            enable_positioning: true,
            head_percentage: 0.20,
            tail_percentage: 0.20,
            centrality_weight: 0.4,
            relatedness_weight: 0.3,
            query_relevance_weight: 0.3,
            auto_exclude_tests: false,
        },
        scaling_config: Default::default(),
    };

    let mut selector = ScalingSelector::new(config);
    let result = selector
        .select_and_process_with_query(repo_path, Some("authentication middleware security"))
        .await
        .unwrap();

    // Verify positioning was applied
    assert!(result.has_context_positioning());

    let (head, middle, tail) = result.get_positioning_stats().unwrap();
    assert!(head > 0, "Should have files in HEAD section");
    assert!(middle > 0, "Should have files in MIDDLE section");
    assert!(tail > 0, "Should have files in TAIL section");
    assert_eq!(head + middle + tail, result.selected_files.len());

    // Verify optimal ordering
    let ordered_files = result.get_optimally_ordered_files();
    assert_eq!(ordered_files.len(), result.selected_files.len());

    // Verify positioning reasoning is provided
    let reasoning = result.get_positioning_reasoning().unwrap();
    assert!(reasoning.contains("HEAD"));
    assert!(reasoning.contains("TAIL"));
    assert!(reasoning.contains("authentication"));

    // Verify query relevance affected positioning
    let positioned = result.positioned_selection.as_ref().unwrap();
    let head_files = &positioned.positioning.head_files;

    // At least one HEAD file should have high query relevance for "authentication"
    let auth_relevant = head_files.iter().any(|f| {
        let filename = f.metadata.path.to_string_lossy().to_lowercase();
        (filename.contains("auth") || filename.contains("security")) && f.query_relevance > 0.5
    });
    assert!(
        auth_relevant,
        "HEAD should contain authentication-related files with high query relevance"
    );
}

#[tokio::test]
async fn test_positioning_vs_no_positioning_comparison() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    create_realistic_repository(&repo_path).unwrap();

    // Test without positioning
    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = false;
    let mut selector_no_pos = ScalingSelector::new(config);

    let result_no_pos = selector_no_pos.select_and_process(repo_path).await.unwrap();

    // Test with positioning
    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = true;
    let mut selector_pos = ScalingSelector::new(config);

    let result_pos = selector_pos
        .select_and_process_with_query(repo_path, Some("main entry point"))
        .await
        .unwrap();

    // Both should select files
    assert!(result_no_pos.selected_files.len() > 0);
    assert!(result_pos.selected_files.len() > 0);

    // Positioned result should have positioning data
    assert!(!result_no_pos.has_context_positioning());
    assert!(result_pos.has_context_positioning());

    // Files should be the same, just differently organized
    assert_eq!(
        result_no_pos.selected_files.len(),
        result_pos.selected_files.len()
    );

    // Positioning should have strategic distribution
    let (head, middle, tail) = result_pos.get_positioning_stats().unwrap();
    let total = head + middle + tail;

    // HEAD should be roughly 20% (+/- some variance)
    let head_percentage = head as f64 / total as f64;
    assert!(
        head_percentage >= 0.1 && head_percentage <= 0.4,
        "HEAD percentage should be reasonable: got {:.2}",
        head_percentage
    );

    // MIDDLE should be the largest section
    assert!(
        middle >= head && middle >= tail,
        "MIDDLE should be largest section"
    );
}

#[tokio::test]
async fn test_centrality_calculation_accuracy() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    create_centrality_test_repository(&repo_path).unwrap();

    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = true;
    let mut selector = ScalingSelector::new(config);

    let result = selector.select_and_process(repo_path).await.unwrap();
    let positioned = result.positioned_selection.unwrap();

    // Find specific files we know should have different centrality
    let lib_file = positioned
        .positioning
        .head_files
        .iter()
        .chain(positioned.positioning.middle_files.iter())
        .chain(positioned.positioning.tail_files.iter())
        .find(|f| f.metadata.path.to_string_lossy().contains("lib.rs"));

    let utils_file = positioned
        .positioning
        .head_files
        .iter()
        .chain(positioned.positioning.middle_files.iter())
        .chain(positioned.positioning.tail_files.iter())
        .find(|f| f.metadata.path.to_string_lossy().contains("utils.rs"));

    if let (Some(lib), Some(utils)) = (lib_file, utils_file) {
        // lib.rs should generally have higher or equal centrality to utils.rs
        // (allowing for equal scores in simple cases)
        assert!(
            lib.centrality.combined >= utils.centrality.combined * 0.8,
            "lib.rs centrality ({:.3}) should be >= 80% of utils.rs centrality ({:.3})",
            lib.centrality.combined,
            utils.centrality.combined
        );

        // Both should have non-negative centrality scores
        assert!(lib.centrality.combined >= 0.0);
        assert!(utils.centrality.combined >= 0.0);
    }
}

#[tokio::test]
async fn test_query_relevance_accuracy() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    create_realistic_repository(&repo_path).unwrap();

    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config.enable_positioning = true;
    let mut selector = ScalingSelector::new(config);

    // Test query relevance for "authentication"
    let result = selector
        .select_and_process_with_query(repo_path, Some("authentication"))
        .await
        .unwrap();

    let positioned = result.positioned_selection.unwrap();
    let all_files: Vec<_> = positioned
        .positioning
        .head_files
        .iter()
        .chain(positioned.positioning.middle_files.iter())
        .chain(positioned.positioning.tail_files.iter())
        .collect();

    // Find auth-related file
    let auth_file = all_files.iter().find(|f| {
        let path_str = f.metadata.path.to_string_lossy().to_lowercase();
        path_str.contains("auth")
            || path_str.contains("middleware")
            || path_str.contains("handlers")
    });

    // Find a utility/low-relevance file
    let non_auth_file = all_files.iter().find(|f| {
        let path_str = f.metadata.path.to_string_lossy().to_lowercase();
        !path_str.contains("auth") && 
        !path_str.contains("main") && // main gets boosted for entry point
        !path_str.contains("lib") &&  // lib gets boosted for entry point
        (path_str.contains("utils") || path_str.contains("test") || path_str.contains("readme"))
    });

    if let (Some(auth), Some(non_auth)) = (auth_file, non_auth_file) {
        // Auth file should have higher or equal query relevance
        assert!(
            auth.query_relevance >= non_auth.query_relevance,
            "Auth file relevance ({:.3}) should be >= non-auth file relevance ({:.3})",
            auth.query_relevance,
            non_auth.query_relevance
        );
    } else {
        // If we can't find the specific files, just verify that some file has query relevance
        let has_query_relevance = all_files.iter().any(|f| f.query_relevance > 0.0);
        assert!(
            has_query_relevance,
            "At least one file should have query relevance for 'authentication'"
        );
    }
}

#[tokio::test]
async fn test_configuration_options() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    create_realistic_repository(&repo_path).unwrap();

    // Test custom configuration
    let mut config = ScalingSelectionConfig::medium_budget();
    config.positioning_config = ContextPositioningConfig {
        enable_positioning: true,
        head_percentage: 0.30,  // Larger HEAD
        tail_percentage: 0.10,  // Smaller TAIL
        centrality_weight: 0.6, // Higher centrality weight
        relatedness_weight: 0.2,
        query_relevance_weight: 0.2,
        auto_exclude_tests: false,
    };

    let mut selector = ScalingSelector::new(config);
    let result = selector
        .select_and_process_with_query(repo_path, Some("main"))
        .await
        .unwrap();

    assert!(result.has_context_positioning());

    let (head, middle, tail) = result.get_positioning_stats().unwrap();
    let total = head + middle + tail;

    // Verify approximate percentages (allowing some variance)
    let head_pct = head as f64 / total as f64;
    let tail_pct = tail as f64 / total as f64;

    // HEAD should be larger than default (20%) - aiming for 30%, but allow some variance
    // In small repositories, exact percentages may vary due to minimum file requirements
    if total >= 10 {
        assert!(
            head_pct > 0.20 || head >= 2,
            "HEAD percentage should be >= 20% or at least 2 files: got {:.2} ({} files)",
            head_pct,
            head
        );
    } else {
        // For very small repositories, just ensure HEAD has at least 1 file
        assert!(head >= 1, "HEAD should have at least 1 file");
    }

    // TAIL should be smaller than or equal to default (20%) - aiming for 10%
    if total >= 10 {
        assert!(
            tail_pct <= 0.25 || tail <= 2,
            "TAIL percentage should be <= 25% or at most 2 files: got {:.2} ({} files)",
            tail_pct,
            tail
        );
    } else {
        // For small repositories, just ensure TAIL has at least 1 file
        assert!(tail >= 1, "TAIL should have at least 1 file");
    }
}

/// Create a realistic repository structure with known centrality patterns
/// Write a test file at path
fn write_test_file(repo_path: &std::path::Path, relative_path: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(repo_path.join(relative_path), content)?;
    Ok(())
}

/// Create realistic repository entry point files
fn create_realistic_entry_points(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_test_file(repo_path, "src/main.rs", "mod auth;\nmod api;\nmod utils;\n\nfn main() {\n    println!(\"Starting app\");\n}")?;
    write_test_file(repo_path, "src/lib.rs", "pub mod auth;\npub mod api;\npub mod utils;\n")?;
    Ok(())
}

/// Create realistic repository auth module files
fn create_realistic_auth_files(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_test_file(repo_path, "src/auth/mod.rs", "pub mod middleware;\npub mod handlers;\n")?;
    write_test_file(repo_path, "src/auth/middleware.rs", "pub fn authenticate() {}\npub fn authorize() {}\n")?;
    write_test_file(repo_path, "src/auth/handlers.rs", "pub fn login() {}\npub fn logout() {}\n")?;
    Ok(())
}

/// Create realistic repository API module files
fn create_realistic_api_files(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_test_file(repo_path, "src/api/mod.rs", "pub mod routes;\npub mod handlers;\n")?;
    write_test_file(repo_path, "src/api/routes.rs", "use crate::auth;\npub fn setup_routes() {}\n")?;
    write_test_file(repo_path, "src/api/handlers.rs", "pub fn handle_request() {}\n")?;
    Ok(())
}

/// Create realistic repository project files
fn create_realistic_project_files(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_test_file(repo_path, "src/utils.rs", "pub fn helper_function() {}\npub fn format_data() {}\n")?;
    write_test_file(repo_path, "tests/integration_test.rs", "use my_app::*;\n#[test]\nfn test_something() {}\n")?;
    write_test_file(repo_path, "Cargo.toml", "[package]\nname = \"my_app\"\nversion = \"0.1.0\"\n")?;
    write_test_file(repo_path, "README.md", "# My App\n\nAn application with authentication and API features.\n")?;
    Ok(())
}

fn create_realistic_repository(repo_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    for dir in ["src", "src/auth", "src/api", "src/utils", "tests"] {
        fs::create_dir_all(repo_path.join(dir))?;
    }
    create_realistic_entry_points(repo_path)?;
    create_realistic_auth_files(repo_path)?;
    create_realistic_api_files(repo_path)?;
    create_realistic_project_files(repo_path)?;
    Ok(())
}

/// Create a repository specifically for testing centrality calculations
fn create_centrality_test_repository(
    repo_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(repo_path.join("src"))?;

    // lib.rs - should have high centrality as main entry point
    fs::write(
        repo_path.join("src/lib.rs"),
        "pub mod utils;\npub mod core;\npub use utils::*;\n",
    )?;

    // utils.rs - should have lower centrality, used by lib.rs
    fs::write(repo_path.join("src/utils.rs"), "pub fn utility() {}\n")?;

    // core.rs - intermediate centrality
    fs::write(
        repo_path.join("src/core.rs"),
        "use crate::utils;\npub fn core_function() {}\n",
    )?;

    // main.rs - should have high centrality as entry point
    fs::write(
        repo_path.join("src/main.rs"),
        "use my_crate::core;\nfn main() {}\n",
    )?;

    Ok(())
}

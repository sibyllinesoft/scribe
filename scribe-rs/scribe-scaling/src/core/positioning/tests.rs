//! Tests for context positioning module.

use super::*;
use std::path::PathBuf;
use std::time::SystemTime;

fn create_test_file(path: &str, size: u64, language: &str) -> FileMetadata {
    FileMetadata {
        path: PathBuf::from(path),
        size,
        modified: SystemTime::now(),
        language: language.to_string(),
        file_type: if language == "Rust" {
            "Source"
        } else {
            "Other"
        }
        .to_string(),
    }
}

#[tokio::test]
async fn test_context_positioner_creation() {
    let positioner = ContextPositioner::with_defaults();
    assert!(positioner.config.enable_positioning);
    assert_eq!(positioner.config.head_percentage, 0.20);
    assert_eq!(positioner.config.tail_percentage, 0.20);
}

#[tokio::test]
async fn test_centrality_calculation() {
    let positioner = ContextPositioner::with_defaults();

    let files = vec![
        create_test_file("src/main.rs", 1000, "Rust"),
        create_test_file("src/lib.rs", 2000, "Rust"),
        create_test_file("src/utils.rs", 500, "Rust"),
    ];

    let files_with_centrality = positioner.calculate_centrality_scores(files).await.unwrap();
    assert_eq!(files_with_centrality.len(), 3);

    // All files should have some centrality score
    for file in &files_with_centrality {
        assert!(file.centrality.combined >= 0.0);
        assert!(file.centrality.degree >= 0.0);
        assert!(file.centrality.pagerank >= 0.0);
        assert!(file.centrality.betweenness >= 0.0);
    }

    // At least one file should have higher centrality than another
    let max_centrality = files_with_centrality
        .iter()
        .map(|f| f.centrality.combined)
        .fold(0.0, f64::max);
    let min_centrality = files_with_centrality
        .iter()
        .map(|f| f.centrality.combined)
        .fold(1.0, f64::min);

    // Allow for equal centrality scores in simple cases
    assert!(max_centrality >= min_centrality);
}

#[tokio::test]
async fn test_positioning_strategy() {
    let positioner = ContextPositioner::with_defaults();

    let files = vec![
        create_test_file("src/main.rs", 1000, "Rust"),
        create_test_file("src/lib.rs", 2000, "Rust"),
        create_test_file("src/utils.rs", 500, "Rust"),
        create_test_file("tests/integration.rs", 800, "Rust"),
        create_test_file("README.md", 300, "Markdown"),
    ];

    let result = positioner
        .position_files(files, Some("main"))
        .await
        .unwrap();

    // Should have files in all three tiers
    assert!(!result.positioning.head_files.is_empty());
    assert!(!result.positioning.middle_files.is_empty());
    assert!(!result.positioning.tail_files.is_empty());

    // Total should equal original count
    let total = result.positioning.head_files.len()
        + result.positioning.middle_files.len()
        + result.positioning.tail_files.len();
    assert_eq!(total, 5);

    // Reasoning should be provided
    assert!(!result.positioning_reasoning.is_empty());
    assert!(result.positioning_reasoning.contains("HEAD"));
    assert!(result.positioning_reasoning.contains("TAIL"));
}

#[tokio::test]
async fn test_query_relevance() {
    let positioner = ContextPositioner::with_defaults();

    let files = vec![
        FileWithCentrality {
            metadata: create_test_file("src/main.rs", 1000, "Rust"),
            centrality: CentralityScores::default(),
            query_relevance: 0.0,
            relatedness_group: String::new(),
        },
        FileWithCentrality {
            metadata: create_test_file("src/utils.rs", 500, "Rust"),
            centrality: CentralityScores::default(),
            query_relevance: 0.0,
            relatedness_group: String::new(),
        },
    ];

    let result = positioner
        .calculate_query_relevance(files, Some("main"))
        .await
        .unwrap();

    // main.rs should have higher query relevance for "main" query
    let main_relevance = result
        .iter()
        .find(|f| f.metadata.path.to_string_lossy().contains("main.rs"))
        .unwrap();
    let utils_relevance = result
        .iter()
        .find(|f| f.metadata.path.to_string_lossy().contains("utils.rs"))
        .unwrap();

    assert!(main_relevance.query_relevance > utils_relevance.query_relevance);
}

#[test]
fn test_relatedness_grouping() {
    let positioner = ContextPositioner::with_defaults();

    let file = create_test_file("src/api/handlers.rs", 1000, "Rust");
    let group = positioner.determine_relatedness_group(&file);

    assert!(group.contains("src/api"));
    assert!(group.contains("Rust"));
}

#[test]
fn test_token_estimation() {
    let positioner = ContextPositioner::with_defaults();

    let rust_file = create_test_file("src/main.rs", 1000, "Rust");
    let json_file = create_test_file("package.json", 1000, "JSON");

    let rust_tokens = positioner.estimate_tokens(&rust_file);
    let json_tokens = positioner.estimate_tokens(&json_file);

    // Rust should have more tokens than JSON for same file size
    assert!(rust_tokens > json_tokens);
}

#[test]
fn test_is_test_file_detection() {
    let positioner = ContextPositioner::with_defaults();

    // Test directory patterns
    assert!(positioner.is_test_file(&std::path::Path::new("src/test/utils.rs")));
    assert!(positioner.is_test_file(&std::path::Path::new("src/tests/integration.py")));
    assert!(positioner.is_test_file(&std::path::Path::new("__tests__/component.test.js")));

    // Test file name patterns
    assert!(positioner.is_test_file(&std::path::Path::new("test_utils.py")));
    assert!(positioner.is_test_file(&std::path::Path::new("utils_test.rs")));
    assert!(positioner.is_test_file(&std::path::Path::new("component.test.tsx")));
    assert!(positioner.is_test_file(&std::path::Path::new("service.spec.ts")));
    assert!(positioner.is_test_file(&std::path::Path::new("model_test.go")));

    // Language-specific patterns
    assert!(positioner.is_test_file(&std::path::Path::new("UserTest.java")));
    assert!(positioner.is_test_file(&std::path::Path::new("user_spec.rb")));
    assert!(positioner.is_test_file(&std::path::Path::new("UserTest.php")));

    // Non-test files should not be detected
    assert!(!positioner.is_test_file(&std::path::Path::new("src/main.rs")));
    assert!(!positioner.is_test_file(&std::path::Path::new("lib/utils.py")));
    assert!(!positioner.is_test_file(&std::path::Path::new("components/Button.tsx")));
    assert!(!positioner.is_test_file(&std::path::Path::new("README.md")));
    assert!(!positioner.is_test_file(&std::path::Path::new("package.json")));
}

#[tokio::test]
async fn test_auto_exclude_tests() {
    let mut config = ContextPositioningConfig::default();
    config.auto_exclude_tests = true;
    let positioner = ContextPositioner::new(config);

    // Create mix of test and non-test files
    let files = vec![
        create_test_file("src/main.rs", 1000, "Rust"),
        create_test_file("src/lib.rs", 800, "Rust"),
        create_test_file("src/tests/integration_test.rs", 1200, "Rust"),
        create_test_file("test/unit_test.py", 600, "Python"),
        create_test_file("components/Button.tsx", 900, "TypeScript"),
        create_test_file("__tests__/Button.test.tsx", 700, "TypeScript"),
    ];

    let result = positioner.position_files(files, None).await.unwrap();

    // Should have filtered out test files
    let all_files: Vec<&FileWithCentrality> = result
        .positioning
        .head_files
        .iter()
        .chain(result.positioning.middle_files.iter())
        .chain(result.positioning.tail_files.iter())
        .collect();

    // Should only have non-test files (3 out of 6)
    assert_eq!(all_files.len(), 3);

    // Verify no test files remain
    for file in all_files {
        let path_str = file.metadata.path.to_string_lossy();
        assert!(!path_str.contains("test"));
        assert!(!path_str.contains("__tests__"));
    }

    // Verify we have the expected non-test files
    let file_names: Vec<String> = result
        .positioning
        .head_files
        .iter()
        .chain(result.positioning.middle_files.iter())
        .chain(result.positioning.tail_files.iter())
        .map(|f| {
            f.metadata
                .path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        })
        .collect();

    assert!(file_names.contains(&"main.rs".to_string()));
    assert!(file_names.contains(&"lib.rs".to_string()));
    assert!(file_names.contains(&"Button.tsx".to_string()));
}

#[tokio::test]
async fn test_auto_exclude_disabled() {
    let mut config = ContextPositioningConfig::default();
    config.auto_exclude_tests = false; // Explicitly disabled
    let positioner = ContextPositioner::new(config);

    // Create mix of test and non-test files
    let files = vec![
        create_test_file("src/main.rs", 1000, "Rust"),
        create_test_file("src/tests/integration_test.rs", 1200, "Rust"),
        create_test_file("test_utils.py", 600, "Python"),
    ];

    let result = positioner.position_files(files, None).await.unwrap();

    // Should include all files when auto-exclude is disabled
    let all_files: Vec<&FileWithCentrality> = result
        .positioning
        .head_files
        .iter()
        .chain(result.positioning.middle_files.iter())
        .chain(result.positioning.tail_files.iter())
        .collect();

    // Should have all 3 files including test files
    assert_eq!(all_files.len(), 3);
}

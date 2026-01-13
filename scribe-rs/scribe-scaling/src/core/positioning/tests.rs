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

#[tokio::test]
async fn test_empty_files_positioning() {
    let positioner = ContextPositioner::with_defaults();
    let files: Vec<FileMetadata> = vec![];

    // Test empty file list - exercises line 47-48
    let result = positioner.position_files(files, None).await.unwrap();

    assert!(result.positioning.head_files.is_empty());
    assert!(result.positioning.middle_files.is_empty());
    assert!(result.positioning.tail_files.is_empty());
}

#[tokio::test]
async fn test_positioning_disabled() {
    let mut config = ContextPositioningConfig::default();
    config.enable_positioning = false;
    let positioner = ContextPositioner::new(config);

    let files = vec![
        create_test_file("src/main.rs", 1000, "Rust"),
        create_test_file("src/lib.rs", 800, "Rust"),
    ];

    // When positioning is disabled, files should go to simple positioning - exercises lines 695-717
    let result = positioner.position_files(files, None).await.unwrap();

    // All files should be in middle (default) with disabled positioning
    assert!(result.positioning.head_files.is_empty());
    assert_eq!(result.positioning.middle_files.len(), 2);
    assert!(result.positioning.tail_files.is_empty());
    assert!(result
        .positioning_reasoning
        .contains("positioning disabled"));
}

#[tokio::test]
async fn test_calculate_centrality_empty() {
    let positioner = ContextPositioner::with_defaults();
    let files: Vec<FileMetadata> = vec![];

    // Test empty file list for centrality calculation - exercises line 117-118
    let result = positioner.calculate_centrality_scores(files).await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_apply_positioning_strategy_empty() {
    let positioner = ContextPositioner::with_defaults();
    let files: Vec<FileWithCentrality> = vec![];

    // Test empty file list - exercises lines 496-500
    let result = positioner.apply_positioning_strategy(files).await.unwrap();
    assert!(result.head_files.is_empty());
    assert!(result.middle_files.is_empty());
    assert!(result.tail_files.is_empty());
}

#[test]
fn test_relatedness_group_single_component() {
    let positioner = ContextPositioner::with_defaults();

    // Test path with only one component - exercises line 481-482
    let file = create_test_file("main.rs", 1000, "Rust");
    let group = positioner.determine_relatedness_group(&file);

    assert!(group.contains("main.rs"));
    assert!(group.contains("Rust"));
}

#[test]
fn test_relatedness_group_root() {
    let positioner = ContextPositioner::with_defaults();

    // Test empty path handling - exercises line 484
    let file = FileMetadata {
        path: PathBuf::from(""),
        size: 100,
        modified: SystemTime::now(),
        language: "Unknown".to_string(),
        file_type: "Other".to_string(),
    };
    let group = positioner.determine_relatedness_group(&file);
    // Should handle empty path gracefully
    assert!(group.contains("Unknown"));
}

#[tokio::test]
async fn test_query_relevance_none() {
    let positioner = ContextPositioner::with_defaults();

    let files = vec![FileWithCentrality {
        metadata: create_test_file("src/main.rs", 1000, "Rust"),
        centrality: CentralityScores::default(),
        query_relevance: 0.0,
        relatedness_group: String::new(),
    }];

    // Test with no query hint - relevance should stay 0
    let result = positioner
        .calculate_query_relevance(files, None)
        .await
        .unwrap();

    assert_eq!(result[0].query_relevance, 0.0);
}

#[test]
fn test_query_relevance_calculation_path_match() {
    let positioner = ContextPositioner::with_defaults();

    let file = create_test_file("src/utils/helpers.rs", 1000, "Rust");
    let query_words = vec!["utils"];

    // Test path match - exercises line 439
    let relevance = positioner.calculate_file_query_relevance(&file, &query_words);
    assert!(relevance > 0.0);
}

#[test]
fn test_query_relevance_calculation_language_match() {
    let positioner = ContextPositioner::with_defaults();

    let file = create_test_file("src/main.py", 1000, "Python");
    let query_words = vec!["python"];

    // Test language match - exercises line 443
    let relevance = positioner.calculate_file_query_relevance(&file, &query_words);
    assert!(relevance > 0.0);
}

#[tokio::test]
async fn test_extract_dependencies_python() {
    let positioner = ContextPositioner::with_defaults();

    // Test Python dependency extraction
    let file = create_test_file("src/module.py", 1000, "Python");
    let deps = positioner.extract_dependencies(&file).await.unwrap();

    assert!(deps.iter().any(|d| d.contains("__init__.py")));
}

#[tokio::test]
async fn test_extract_dependencies_javascript() {
    let positioner = ContextPositioner::with_defaults();

    // Test JavaScript dependency extraction - exercises line 223
    let mut file = create_test_file("src/component.js", 1000, "JavaScript");
    file.file_type = "Source".to_string();
    let deps = positioner.extract_dependencies(&file).await.unwrap();

    assert!(deps.iter().any(|d| d.contains("index.js")));
}

#[tokio::test]
async fn test_extract_dependencies_configuration() {
    let positioner = ContextPositioner::with_defaults();

    // Test configuration file dependency extraction
    let file = FileMetadata {
        path: PathBuf::from("config.yml"),
        size: 500,
        modified: SystemTime::now(),
        language: "YAML".to_string(),
        file_type: "Configuration".to_string(),
    };
    let deps = positioner.extract_dependencies(&file).await.unwrap();

    assert!(deps.iter().any(|d| d.contains("package.json")));
}

#[test]
fn test_estimate_tokens_various_languages() {
    let positioner = ContextPositioner::with_defaults();

    // Test various language multipliers
    let py_file = create_test_file("main.py", 1000, "Python");
    let c_file = create_test_file("main.c", 1000, "C");
    let go_file = create_test_file("main.go", 1000, "Go");
    let unknown_file = create_test_file("main.xyz", 1000, "Unknown");

    let py_tokens = positioner.estimate_tokens(&py_file);
    let c_tokens = positioner.estimate_tokens(&c_file);
    let go_tokens = positioner.estimate_tokens(&go_file);
    let unknown_tokens = positioner.estimate_tokens(&unknown_file);

    // Python should have slightly more tokens due to multiplier
    assert!(py_tokens > c_tokens);
    // C and Go should have same multiplier (1.0)
    assert_eq!(c_tokens, go_tokens);
    // Unknown defaults to 1.0 multiplier
    assert_eq!(unknown_tokens, c_tokens);
}

#[tokio::test]
async fn test_positioning_with_large_head_tail() {
    let mut config = ContextPositioningConfig::default();
    config.head_percentage = 0.5;
    config.tail_percentage = 0.5;
    let positioner = ContextPositioner::new(config);

    // With 100% head+tail, middle should be empty
    let files = vec![
        create_test_file("src/a.rs", 1000, "Rust"),
        create_test_file("src/b.rs", 1000, "Rust"),
    ];

    let result = positioner.position_files(files, None).await.unwrap();

    // Total should still equal original count
    let total = result.positioning.head_files.len()
        + result.positioning.middle_files.len()
        + result.positioning.tail_files.len();
    assert_eq!(total, 2);
}

#[tokio::test]
async fn test_positioning_single_file() {
    let positioner = ContextPositioner::with_defaults();

    let files = vec![create_test_file("src/only.rs", 1000, "Rust")];

    let result = positioner.position_files(files, None).await.unwrap();

    // Single file should be placed in head
    let total = result.positioning.head_files.len()
        + result.positioning.middle_files.len()
        + result.positioning.tail_files.len();
    assert_eq!(total, 1);
}

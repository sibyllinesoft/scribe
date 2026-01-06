//! Tests for incremental scanning module.

use super::*;
use tempfile::TempDir;
use tokio::fs;

async fn create_test_repo() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    // Create some test files
    fs::write(root.join("main.rs"), "fn main() {}")
        .await
        .unwrap();
    fs::write(root.join("lib.rs"), "pub fn hello() {}")
        .await
        .unwrap();

    // Create subdirectory
    fs::create_dir(root.join("src")).await.unwrap();
    fs::write(root.join("src/module.rs"), "mod test;")
        .await
        .unwrap();

    temp_dir
}

#[tokio::test]
async fn test_incremental_scanner_creation() {
    let temp_dir = create_test_repo().await;
    let config = IncrementalConfig::default();

    let scanner = IncrementalScanner::new(temp_dir.path(), config).await;
    assert!(scanner.is_ok());
}

#[tokio::test]
async fn test_file_discovery() {
    let temp_dir = create_test_repo().await;
    let config = IncrementalConfig::default();
    let scanner = IncrementalScanner::new(temp_dir.path(), config)
        .await
        .unwrap();

    let files = scanner.discover_files().await.unwrap();
    assert!(files.len() >= 3); // main.rs, lib.rs, src/module.rs

    let file_names: Vec<_> = files
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap())
        .collect();
    assert!(file_names.contains(&"main.rs"));
    assert!(file_names.contains(&"lib.rs"));
    assert!(file_names.contains(&"module.rs"));
}

#[tokio::test]
async fn test_manifest_serialization() {
    let manifest = FileManifest {
        version: 1,
        created_at: 1640995200,
        updated_at: 1640995200,
        repo_root: "/test/repo".to_string(),
        git_commit: Some("abcdef123456".to_string()),
        entries: FxHashMap::default(),
        stats: ManifestStats {
            total_files: 0,
            cached_files: 0,
            total_bytes: 0,
            manifest_size_bytes: 0,
            last_scan_duration_secs: 0.0,
            cache_hit_rate: 0.0,
        },
    };

    let serialized = bincode::serialize(&manifest).unwrap();
    let deserialized: FileManifest = bincode::deserialize(&serialized).unwrap();

    assert_eq!(manifest.version, deserialized.version);
    assert_eq!(manifest.repo_root, deserialized.repo_root);
    assert_eq!(manifest.git_commit, deserialized.git_commit);
}

#[tokio::test]
async fn test_content_hashing() {
    let temp_dir = create_test_repo().await;
    let config = IncrementalConfig {
        enable_content_hashing: true,
        max_hash_file_size: 1024,
        hash_chunk_size: 256,
        ..Default::default()
    };
    let mut scanner = IncrementalScanner::new(temp_dir.path(), config)
        .await
        .unwrap();

    let test_file = temp_dir.path().join("main.rs");
    let hash1 = scanner.calculate_file_hash(&test_file).await.unwrap();

    // Hash should be consistent
    let hash2 = scanner.calculate_file_hash(&test_file).await.unwrap();
    assert_eq!(hash1, hash2);

    // Modify file and check hash changes
    fs::write(&test_file, "fn main() { println!(\"modified\"); }")
        .await
        .unwrap();
    let hash3 = scanner.calculate_file_hash(&test_file).await.unwrap();
    assert_ne!(hash1, hash3);
}

#[tokio::test]
async fn test_file_change_detection() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let temp_dir = create_test_repo().await;
    let config = IncrementalConfig::default();
    let scanner = IncrementalScanner::new(temp_dir.path(), config)
        .await
        .unwrap();

    let test_file = temp_dir.path().join("main.rs");
    let metadata = fs::metadata(&test_file).await.unwrap();

    // Create manifest entry
    let entry = ManifestEntry {
        path: "main.rs".to_string(),
        size: metadata.len(),
        modified: metadata
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        device: scanner.get_device(&metadata),
        inode: scanner.get_inode(&metadata),
        content_hash: 0,
        git_blob_id: None,
        scanned_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        cached_results: None,
    };

    // Unchanged file
    let change = scanner
        .detect_file_change(&entry, &metadata, &test_file)
        .await
        .unwrap();
    assert_eq!(change, FileChange::Unchanged);

    // Modify file
    fs::write(&test_file, "fn main() { println!(\"changed\"); }")
        .await
        .unwrap();
    let new_metadata = fs::metadata(&test_file).await.unwrap();
    let change = scanner
        .detect_file_change(&entry, &new_metadata, &test_file)
        .await
        .unwrap();
    assert_eq!(change, FileChange::ContentChanged);
}

#[tokio::test]
async fn test_manifest_persistence() {
    let temp_dir = create_test_repo().await;
    let config = IncrementalConfig::default();
    let mut scanner = IncrementalScanner::new(temp_dir.path(), config)
        .await
        .unwrap();

    // Create a manifest
    let manifest = scanner.create_new_manifest().await.unwrap();
    scanner.save_manifest(&manifest).await.unwrap();

    // Load manifest in new scanner instance
    let config2 = IncrementalConfig::default();
    let mut scanner2 = IncrementalScanner::new(temp_dir.path(), config2)
        .await
        .unwrap();

    assert!(scanner2.manifest.is_some());
    let loaded_manifest = scanner2.manifest.unwrap();
    assert_eq!(loaded_manifest.repo_root, manifest.repo_root);
    assert_eq!(loaded_manifest.version, manifest.version);
}

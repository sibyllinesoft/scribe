//! Repository profiling for automatic type detection and configuration optimization.

use crate::api::engine::ScalingConfig;
use crate::core::error::ScalingResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Repository types for classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RepositoryType {
    Personal,
    Library,
    WebApp,
    MobileApp,
    SystemSoftware,
    GameDev,
    DataScience,
    Enterprise,
    Monorepo,
    Documentation,
    Unknown,
}

impl Default for RepositoryType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Repository profile with characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryProfile {
    /// Detected repository type
    pub repository_type: RepositoryType,

    /// Total number of files
    pub file_count: usize,

    /// Total repository size in bytes
    pub total_size: u64,

    /// Average file size
    pub average_file_size: u64,

    /// Primary programming languages
    pub primary_languages: Vec<String>,

    /// Build system type
    pub build_system: String,
}

impl RepositoryProfile {
    /// Convert profile to optimal scaling configuration
    pub fn to_scaling_config(&self) -> ScalingConfig {
        match self.repository_type {
            RepositoryType::Personal if self.file_count < 1000 => ScalingConfig::small_repository(),
            RepositoryType::Enterprise | RepositoryType::Monorepo => {
                ScalingConfig::large_repository()
            }
            _ => ScalingConfig::default(),
        }
    }
}

/// Repository profiler
pub struct RepositoryProfiler {
    // Simple profiler without complex state
}

impl RepositoryProfiler {
    /// Create a new repository profiler
    pub fn new() -> Self {
        Self {}
    }

    /// Profile a repository and return its characteristics
    pub async fn profile_repository(&self, path: &Path) -> ScalingResult<RepositoryProfile> {
        // Basic profiling implementation
        let mut file_count = 0;
        let mut total_size = 0u64;
        let mut languages = std::collections::HashMap::new();

        for entry in walkdir::WalkDir::new(path).follow_links(false) {
            if let Ok(entry) = entry {
                if entry.file_type().is_file() {
                    file_count += 1;
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len();
                    }

                    // Simple language detection
                    if let Some(ext) = entry.path().extension() {
                        if let Some(ext_str) = ext.to_str() {
                            *languages.entry(ext_str.to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        let average_file_size = if file_count > 0 {
            total_size / file_count as u64
        } else {
            0
        };

        // Simple repository type detection
        let repository_type = if file_count < 100 {
            RepositoryType::Personal
        } else if file_count > 10000 {
            RepositoryType::Enterprise
        } else {
            RepositoryType::Library
        };

        // Get primary languages
        let mut lang_vec: Vec<_> = languages.into_iter().collect();
        lang_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let primary_languages = lang_vec.into_iter().take(3).map(|(lang, _)| lang).collect();

        Ok(RepositoryProfile {
            repository_type,
            file_count,
            total_size,
            average_file_size,
            primary_languages,
            build_system: "Unknown".to_string(),
        })
    }

    /// Quick estimate of processing requirements
    pub async fn quick_estimate(
        &self,
        path: &Path,
    ) -> ScalingResult<(usize, std::time::Duration, usize)> {
        let profile = self.profile_repository(path).await?;

        let estimated_duration = std::time::Duration::from_millis(
            (profile.file_count as u64 * 10).min(30000), // Max 30 seconds
        );

        let estimated_memory = profile.file_count * 1024; // 1KB per file

        Ok((profile.file_count, estimated_duration, estimated_memory))
    }
}

impl Default for RepositoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_repository_type_default() {
        let repo_type = RepositoryType::default();
        assert_eq!(repo_type, RepositoryType::Unknown);
    }

    #[test]
    fn test_repository_type_variants() {
        let variants = vec![
            RepositoryType::Personal,
            RepositoryType::Library,
            RepositoryType::WebApp,
            RepositoryType::MobileApp,
            RepositoryType::SystemSoftware,
            RepositoryType::GameDev,
            RepositoryType::DataScience,
            RepositoryType::Enterprise,
            RepositoryType::Monorepo,
            RepositoryType::Documentation,
            RepositoryType::Unknown,
        ];

        // All variants should be clone-able and comparable
        for variant in variants {
            let cloned = variant.clone();
            assert_eq!(variant, cloned);
        }
    }

    #[test]
    fn test_repository_profile_to_scaling_config_personal() {
        let profile = RepositoryProfile {
            repository_type: RepositoryType::Personal,
            file_count: 50,
            total_size: 1024 * 1024,
            average_file_size: 1024,
            primary_languages: vec!["rs".to_string()],
            build_system: "cargo".to_string(),
        };

        let config = profile.to_scaling_config();
        // Personal repo with <1000 files should use small_repository config
        assert!(!config.streaming.enable_streaming);
    }

    #[test]
    fn test_repository_profile_to_scaling_config_enterprise() {
        let profile = RepositoryProfile {
            repository_type: RepositoryType::Enterprise,
            file_count: 50000,
            total_size: 1024 * 1024 * 1024,
            average_file_size: 20480,
            primary_languages: vec!["java".to_string(), "scala".to_string()],
            build_system: "maven".to_string(),
        };

        let config = profile.to_scaling_config();
        // Enterprise should use large_repository config
        assert!(config.streaming.enable_streaming);
    }

    #[test]
    fn test_repository_profile_to_scaling_config_monorepo() {
        let profile = RepositoryProfile {
            repository_type: RepositoryType::Monorepo,
            file_count: 100000,
            total_size: 5 * 1024 * 1024 * 1024,
            average_file_size: 50 * 1024,
            primary_languages: vec!["ts".to_string(), "py".to_string(), "go".to_string()],
            build_system: "bazel".to_string(),
        };

        let config = profile.to_scaling_config();
        // Monorepo should use large_repository config
        assert!(config.streaming.enable_streaming);
    }

    #[test]
    fn test_repository_profile_to_scaling_config_default() {
        let profile = RepositoryProfile {
            repository_type: RepositoryType::Library,
            file_count: 500,
            total_size: 10 * 1024 * 1024,
            average_file_size: 2048,
            primary_languages: vec!["py".to_string()],
            build_system: "pip".to_string(),
        };

        let config = profile.to_scaling_config();
        // Library should use default config
        assert!(config.token_budget.is_none());
    }

    #[test]
    fn test_repository_profiler_new() {
        let _profiler = RepositoryProfiler::new();
        // RepositoryProfiler is a unit struct so it's zero-sized
        // Just verify it can be constructed
    }

    #[test]
    fn test_repository_profiler_default() {
        let _profiler = RepositoryProfiler::default();
        // Just verify default impl works
    }

    #[tokio::test]
    async fn test_repository_profiler_profile_repository() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path();

        // Create some test files
        std::fs::write(path.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(path.join("lib.rs"), "pub fn test() {}").unwrap();
        std::fs::write(path.join("utils.py"), "def helper(): pass").unwrap();

        let profiler = RepositoryProfiler::new();
        let profile = profiler.profile_repository(path).await.unwrap();

        assert_eq!(profile.file_count, 3);
        assert!(profile.total_size > 0);
        assert!(profile.average_file_size > 0);
        assert!(!profile.primary_languages.is_empty());
        assert_eq!(profile.repository_type, RepositoryType::Personal); // <100 files
    }

    #[tokio::test]
    async fn test_repository_profiler_quick_estimate() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path();

        std::fs::write(path.join("file1.txt"), "content").unwrap();
        std::fs::write(path.join("file2.txt"), "content").unwrap();

        let profiler = RepositoryProfiler::new();
        let (file_count, duration, memory) = profiler.quick_estimate(path).await.unwrap();

        assert_eq!(file_count, 2);
        assert!(duration.as_millis() > 0);
        assert!(memory > 0);
    }

    #[test]
    fn test_repository_profile_clone() {
        let profile = RepositoryProfile {
            repository_type: RepositoryType::Library,
            file_count: 100,
            total_size: 1024,
            average_file_size: 10,
            primary_languages: vec!["rs".to_string()],
            build_system: "cargo".to_string(),
        };

        let cloned = profile.clone();
        assert_eq!(profile.file_count, cloned.file_count);
        assert_eq!(profile.repository_type, cloned.repository_type);
    }

    #[test]
    fn test_repository_type_clone_and_copy() {
        let rt = RepositoryType::WebApp;
        let cloned = rt.clone();
        let copied = rt; // Copy trait
        assert_eq!(rt, cloned);
        assert_eq!(rt, copied);
    }

    #[tokio::test]
    async fn test_repository_profiler_empty_dir() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path();

        let profiler = RepositoryProfiler::new();
        let profile = profiler.profile_repository(path).await.unwrap();

        assert_eq!(profile.file_count, 0);
        assert_eq!(profile.total_size, 0);
        assert_eq!(profile.average_file_size, 0);
        assert!(profile.primary_languages.is_empty());
    }
}

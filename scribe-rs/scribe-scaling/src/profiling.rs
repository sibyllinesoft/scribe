//! Repository profiling for automatic type detection and configuration optimization.

use crate::engine::ScalingConfig;
use crate::error::ScalingResult;
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

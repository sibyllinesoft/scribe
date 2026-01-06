//! Type definitions for intelligent scaling selection.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::api::engine::{ProcessingResult, ScalingConfig};
use crate::core::positioning::{ContextPositioningConfig, PositionedSelection};
use crate::io::streaming::FileMetadata;

/// File category classification for quota allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileCategory {
    Config,
    Entry,
    Examples,
    General,
}

/// Selection algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SelectionAlgorithm {
    /// Tiered approach with intelligent selection (V5)
    V5Integrated,
}

/// Configuration for intelligent scaling selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingSelectionConfig {
    /// Token budget for selection (like --token-target)
    pub token_budget: usize,

    /// Selection algorithm variant to use
    pub selection_algorithm: SelectionAlgorithm,

    /// Enable category-based quota allocation
    pub enable_quotas: bool,

    /// Context positioning configuration
    pub positioning_config: ContextPositioningConfig,

    /// Base scaling configuration
    pub scaling_config: ScalingConfig,
}

impl Default for ScalingSelectionConfig {
    fn default() -> Self {
        Self {
            token_budget: 8000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::default(),
        }
    }
}

impl ScalingSelectionConfig {
    /// Create configuration for small token budget (should select ~2 files)
    pub fn small_budget() -> Self {
        Self {
            token_budget: 1000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::small_repository(),
        }
    }

    /// Enable auto-exclusion of test files (focuses on code and docs only)
    pub fn with_test_exclusion(mut self) -> Self {
        self.positioning_config.auto_exclude_tests = true;
        self
    }

    /// Create configuration for medium token budget (should select ~11 files)
    pub fn medium_budget() -> Self {
        Self {
            token_budget: 10000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::default(),
        }
    }

    /// Create configuration for large token budget
    pub fn large_budget() -> Self {
        Self {
            token_budget: 100000,
            selection_algorithm: SelectionAlgorithm::V5Integrated,
            enable_quotas: true,
            positioning_config: ContextPositioningConfig::default(),
            scaling_config: ScalingConfig::large_repository(),
        }
    }
}

/// Results of intelligent scaling selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingSelectionResult {
    /// Selected files with metadata (if positioning disabled)
    pub selected_files: Vec<FileMetadata>,

    /// Context-positioned selection (if positioning enabled)
    pub positioned_selection: Option<PositionedSelection>,

    /// Total files considered during selection
    pub total_files_considered: usize,

    /// Token budget utilization
    pub token_utilization: f64,

    /// Actual tokens used by selected files
    pub tokens_used: usize,

    /// Selection algorithm used
    pub algorithm_used: SelectionAlgorithm,

    /// Selection performance metrics
    pub selection_time: Duration,

    /// Processing performance metrics (from scaling system)
    pub processing_result: ProcessingResult,
}

impl ScalingSelectionResult {
    /// Get all files in optimal order (positioned if available, otherwise selected)
    pub fn get_optimally_ordered_files(&self) -> Vec<&FileMetadata> {
        if let Some(positioned) = &self.positioned_selection {
            let mut files = Vec::new();

            // HEAD files first (query-relevant, high centrality)
            for file in &positioned.positioning.head_files {
                files.push(&file.metadata);
            }

            // MIDDLE files (supporting, low centrality)
            for file in &positioned.positioning.middle_files {
                files.push(&file.metadata);
            }

            // TAIL files last (core functionality, high centrality)
            for file in &positioned.positioning.tail_files {
                files.push(&file.metadata);
            }

            files
        } else {
            self.selected_files.iter().collect()
        }
    }

    /// Get positioning statistics if available
    pub fn get_positioning_stats(&self) -> Option<(usize, usize, usize)> {
        self.positioned_selection.as_ref().map(|p| {
            (
                p.positioning.head_files.len(),
                p.positioning.middle_files.len(),
                p.positioning.tail_files.len(),
            )
        })
    }

    /// Get positioning reasoning if available
    pub fn get_positioning_reasoning(&self) -> Option<&str> {
        self.positioned_selection
            .as_ref()
            .map(|p| p.positioning_reasoning.as_str())
    }

    /// Check if context positioning was applied
    pub fn has_context_positioning(&self) -> bool {
        self.positioned_selection.is_some()
    }
}

/// Main intelligent scaling selector
pub struct ScalingSelector {
    pub(crate) config: ScalingSelectionConfig,
}

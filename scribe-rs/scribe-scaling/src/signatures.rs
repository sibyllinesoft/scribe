//! Advanced signature extraction with multiple levels and budget pressure adaptation.

use serde::{Deserialize, Serialize};

/// Signature extraction levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignatureLevel {
    /// Basic file metadata only
    Minimal,
    /// Structure and imports
    Structural,
    /// Include semantic information
    Semantic,
    /// Detailed analysis
    Detailed,
    /// Complete signature with all information
    Complete,
}

impl Default for SignatureLevel {
    fn default() -> Self {
        Self::Structural
    }
}

/// Configuration for signature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureConfig {
    /// Default signature level
    pub default_level: SignatureLevel,

    /// Whether to enable caching of signatures
    pub enable_caching: bool,

    /// Budget pressure threshold (0.0 to 1.0)
    pub budget_pressure_threshold: f64,
}

impl Default for SignatureConfig {
    fn default() -> Self {
        Self {
            default_level: SignatureLevel::Structural,
            enable_caching: true,
            budget_pressure_threshold: 0.5,
        }
    }
}

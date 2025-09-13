//! Relevance scoring module - stub implementation

use scribe_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceMetrics {
    pub score: f64,
    pub confidence: f64,
}

impl Default for RelevanceMetrics {
    fn default() -> Self {
        Self {
            score: 0.0,
            confidence: 0.0,
        }
    }
}

pub struct RelevanceScorer;

impl RelevanceScorer {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl Default for RelevanceScorer {
    fn default() -> Self {
        Self::new().expect("Failed to create RelevanceScorer")
    }
}
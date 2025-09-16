//! Code selector module - stub implementation
//! This module provides the CodeSelector for intelligent file selection

use scribe_analysis::AnalysisResult;
use scribe_core::Result;
use scribe_graph::CodeGraph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    pub max_files: Option<usize>,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            max_files: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub selected_files: Vec<String>,
    pub scores: Vec<f64>,
    pub total_files_considered: usize,
}

pub struct CodeSelector;

impl CodeSelector {
    pub fn new() -> Self {
        Self
    }

    pub async fn select(
        &self,
        _analysis: &AnalysisResult,
        _graph: &CodeGraph,
        _criteria: &SelectionCriteria,
    ) -> Result<SelectionResult> {
        // Stub implementation
        Ok(SelectionResult {
            selected_files: vec![],
            scores: vec![],
            total_files_considered: 0,
        })
    }
}

impl Default for CodeSelector {
    fn default() -> Self {
        Self::new()
    }
}

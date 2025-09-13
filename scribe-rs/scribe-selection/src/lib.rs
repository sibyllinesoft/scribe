//! # Scribe Selection
//! 
//! Intelligent code selection and context extraction capabilities for the Scribe library.
//! This crate provides advanced algorithms for selecting relevant code sections based on
//! semantic understanding, dependency analysis, and contextual relevance.

pub mod selector;
pub mod context;
pub mod relevance;
pub mod bundler;
pub mod optimizer;
pub mod quota;
pub mod demotion;
pub mod two_pass;
pub mod bandit_router;
pub mod ast_parser;

// Re-export main types
pub use selector::{CodeSelector, SelectionCriteria, SelectionResult};
pub use context::{ContextExtractor, ContextOptions, CodeContext};
pub use relevance::{RelevanceScorer, RelevanceMetrics};
pub use bundler::{CodeBundler, BundleOptions, CodeBundle};
pub use quota::{QuotaManager, FileCategory, CategoryQuota, QuotaAllocation, QuotaScanResult, create_quota_manager};
pub use demotion::{DemotionEngine, FidelityMode, DemotionResult, ChunkInfo, CodeChunker, SignatureExtractor};
pub use two_pass::{TwoPassSelector, TwoPassConfig, TwoPassResult, CoverageGap, SelectionMetrics, SelectionRule, SelectionContext, FileInfo};
pub use bandit_router::{BanditRouter, BanditConfig, SelectionStrategy, RoutingDecision, PerformanceFeedback, BanditStatistics, BanditState};
pub use ast_parser::{AstParser, AstLanguage, AstChunk, AstSignature};

use scribe_core::Result;

/// Main entry point for intelligent code selection
pub struct SelectionEngine {
    selector: CodeSelector,
    context_extractor: ContextExtractor,
    relevance_scorer: RelevanceScorer,
    bundler: CodeBundler,
}

impl SelectionEngine {
    /// Create a new selection engine
    pub fn new() -> Result<Self> {
        Ok(Self {
            selector: CodeSelector::new(),
            context_extractor: ContextExtractor::new(),
            relevance_scorer: RelevanceScorer::new()?,
            bundler: CodeBundler::new(),
        })
    }

    /// Select relevant code based on criteria
    pub async fn select_code(
        &self, 
        criteria: &SelectionCriteria
    ) -> Result<SelectionResult> {
        // TODO: Implement selection logic without heavy dependencies
        todo!("Implement selection logic")
    }

    /// Extract context for selected code
    pub async fn extract_context(
        &self,
        selection: &SelectionResult,
        options: &ContextOptions
    ) -> Result<CodeContext> {
        self.context_extractor.extract(selection, options).await
    }

    /// Create a bundled representation of selected code
    pub async fn create_bundle(
        &self,
        context: &CodeContext,
        options: &BundleOptions
    ) -> Result<CodeBundle> {
        self.bundler.bundle(context, options).await
    }
}

impl Default for SelectionEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create SelectionEngine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_selection_engine_creation() {
        let engine = SelectionEngine::new();
        assert!(engine.is_ok());
    }
}
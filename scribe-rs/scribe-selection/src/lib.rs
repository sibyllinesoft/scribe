//! # Scribe Selection
//!
//! Intelligent code selection and context extraction capabilities for the Scribe library.
//! This crate provides advanced algorithms for selecting relevant code sections based on
//! semantic understanding, dependency analysis, and contextual relevance.

// Module organization
pub mod algorithms;
pub mod ast;
pub mod budget;
pub mod core;

// Re-export AST types
pub use ast::ast_parser::{
    AstChunk, AstLanguage, AstParser, AstSignature, EntityLocation, EntityQuery, EntityType,
};

// Re-export core types
pub use core::bundler::{BundleOptions, CodeBundle, CodeBundler};
pub use core::context::{CodeContext, ContextExtractor, ContextFile, ContextOptions};
pub use core::selector::{CodeSelector, SelectionCriteria, SelectionResult};

// Re-export algorithm types
pub use algorithms::covering_set::{
    CoveringSetComputer, CoveringSetEntity, CoveringSetFile, CoveringSetGranularity,
    CoveringSetOptions, CoveringSetResult, CoveringSetStatistics, InclusionReason, LineRange,
};
pub use algorithms::demotion::{
    ChunkInfo, CodeChunker, DemotionEngine, DemotionResult, FidelityMode, SignatureExtractor,
};
pub use algorithms::quota::{
    create_quota_manager, CategoryQuota, FileCategory, QuotaAllocation, QuotaManager,
    QuotaScanResult,
};
pub use algorithms::simple_router::{
    ProjectSize, RoutingDecision, SelectionStrategy, SimpleRouter, TimeConstraint,
};
pub use algorithms::two_pass::{
    CoverageGap, FileInfo, SelectionContext, SelectionMetrics, SelectionRule, TwoPassConfig,
    TwoPassResult, TwoPassSelector,
};

// Re-export budget types
pub use budget::token_budget::{apply_token_budget_selection, SelectionConfig};
pub use budget::weighting::FileWeights;

use scribe_core::Result;

/// Main entry point for intelligent code selection
pub struct SelectionEngine {
    selector: CodeSelector,
    context_extractor: ContextExtractor,
    bundler: CodeBundler,
}

impl SelectionEngine {
    /// Create a new selection engine
    pub fn new() -> Result<Self> {
        Ok(Self {
            selector: CodeSelector::new(),
            context_extractor: ContextExtractor::new(),
            bundler: CodeBundler::new(),
        })
    }

    /// Select relevant code based on criteria
    pub async fn select_code(&self, criteria: SelectionCriteria<'_>) -> Result<SelectionResult> {
        self.selector.select(criteria).await
    }

    /// Extract context for selected code
    pub async fn extract_context(
        &self,
        selection: &SelectionResult,
        options: &ContextOptions,
    ) -> Result<CodeContext> {
        self.context_extractor.extract(selection, options).await
    }

    /// Create a bundled representation of selected code
    pub async fn create_bundle(
        &self,
        context: &CodeContext,
        options: &BundleOptions,
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

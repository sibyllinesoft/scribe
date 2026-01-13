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

    #[test]
    fn test_selection_engine_default() {
        let engine = SelectionEngine::default();
        // Verify engine is created successfully via Default trait
        let _ = engine;
    }

    #[tokio::test]
    async fn test_selection_engine_select_empty() {
        let engine = SelectionEngine::new().unwrap();
        let config = scribe_core::Config::default();
        let criteria = SelectionCriteria {
            files: vec![],
            token_budget: 1000,
            config: &config,
            weights: None,
        };

        let result = engine.select_code(criteria).await;
        assert!(result.is_ok());
        let selection = result.unwrap();
        assert!(selection.files.is_empty());
    }

    // Test re-exports work
    #[test]
    fn test_reexports_ast() {
        let _: EntityType = EntityType::Function;
        let _: EntityType = EntityType::Class;
        let _: EntityType = EntityType::Module;
        let _: EntityType = EntityType::Interface;
        let _: EntityType = EntityType::Constant;
        let _: EntityType = EntityType::Any;
    }

    #[test]
    fn test_reexports_fidelity_mode() {
        let _: FidelityMode = FidelityMode::Full;
        let _: FidelityMode = FidelityMode::Chunk;
        let _: FidelityMode = FidelityMode::Signature;
    }

    #[test]
    fn test_reexports_selection_config() {
        let default_config = SelectionConfig::default();
        assert!(default_config.signature_boost > 0.0);

        let resolution_config = SelectionConfig::resolution();
        assert_eq!(resolution_config.signature_boost, 1.0);

        let coverage_config = SelectionConfig::coverage();
        assert!(coverage_config.signature_boost > 1.0);
    }

    #[test]
    fn test_reexports_file_weights() {
        let mut weights = FileWeights::new();
        weights.set("file.rs".to_string(), 1.0);
        assert_eq!(weights.get("file.rs"), 1.0);
        assert_eq!(weights.get("nonexistent.rs"), 0.0);
    }

    #[test]
    fn test_reexports_covering_set_granularity() {
        let _: CoveringSetGranularity = CoveringSetGranularity::File;
        let _: CoveringSetGranularity = CoveringSetGranularity::Entity;
    }

    #[test]
    fn test_reexports_routing() {
        let _: ProjectSize = ProjectSize::Small;
        let _: ProjectSize = ProjectSize::Medium;
        let _: ProjectSize = ProjectSize::Large;

        let _: TimeConstraint = TimeConstraint::Tight;
        let _: TimeConstraint = TimeConstraint::Normal;
        let _: TimeConstraint = TimeConstraint::Relaxed;

        // Actual SelectionStrategy variants
        let _: SelectionStrategy = SelectionStrategy::ImportanceGreedy;
        let _: SelectionStrategy = SelectionStrategy::DependencyAware;
        let _: SelectionStrategy = SelectionStrategy::CoverageOptimized;
        let _: SelectionStrategy = SelectionStrategy::Random;
        let _: SelectionStrategy = SelectionStrategy::TwoPassSpeculative;
    }

    #[test]
    fn test_reexports_file_category() {
        // Actual FileCategory variants
        let _: FileCategory = FileCategory::Config;
        let _: FileCategory = FileCategory::Entry;
        let _: FileCategory = FileCategory::Examples;
        let _: FileCategory = FileCategory::General;
    }

    #[tokio::test]
    async fn test_selection_engine_extract_context() {
        let engine = SelectionEngine::new().unwrap();
        let selection = SelectionResult {
            files: vec![],
            total_tokens_used: 0,
            budget: 1000,
            unused_tokens: 1000,
            total_files_considered: 0,
        };
        let options = ContextOptions::default();

        let result = engine.extract_context(&selection, &options).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_selection_engine_create_bundle() {
        let engine = SelectionEngine::new().unwrap();
        let context = CodeContext {
            files: vec![],
            dependencies: vec![],
            total_tokens: 0,
        };
        let options = BundleOptions::default();

        let result = engine.create_bundle(&context, &options).await;
        assert!(result.is_ok());
    }
}

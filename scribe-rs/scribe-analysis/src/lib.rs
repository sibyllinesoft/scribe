//! # Scribe Analysis
//! 
//! Code analysis algorithms and heuristic scoring for the Scribe library.
//! This crate provides sophisticated file prioritization using multi-dimensional 
//! heuristic scoring, template detection, and import graph analysis.

pub mod heuristics;
pub mod ast_import_parser;
pub mod complexity;
pub mod language_support;

// Legacy modules (kept for compatibility)
pub mod ast;
pub mod parser;
pub mod analyzer;
pub mod metrics;
pub mod symbols;
pub mod dependencies;

// Re-export main heuristics types
pub use heuristics::{
    HeuristicSystem, 
    HeuristicScorer, 
    ScoreComponents, 
    HeuristicWeights,
    ScoringFeatures,
    DocumentAnalysis,
    TemplateDetector,
    TemplateEngine,
    ImportGraphBuilder,
    ImportGraph,
    is_template_file,
    get_template_score_boost,
    import_matches_file,
};

// Re-export complexity analysis types
pub use complexity::{
    ComplexityAnalyzer,
    ComplexityMetrics as ComplexityAnalysisMetrics,
    ComplexityConfig,
    ComplexityThresholds,
    LanguageSpecificMetrics,
};

// Re-export language support types
pub use language_support::{
    AstLanguage, LanguageTier, LanguageFeatures,
    FunctionExtractor, FunctionInfo, ClassInfo,
    DocumentationAnalyzer, DocumentationCoverage,
    SymbolAnalyzer, SymbolUsage, SymbolType,
    LanguageMetrics, LanguageSpecificComplexity,
    LanguageSupport, LanguageAnalysisResult,
    analyze_file_language,
};

// Legacy re-exports
pub use ast::{AstNode, AstWalker};
pub use parser::{Parser, ParseResult};
pub use analyzer::{CodeAnalyzer, AnalysisResult};
pub use metrics::{Metrics, ComplexityMetrics as LegacyComplexityMetrics};
pub use symbols::{Symbol, SymbolTable};

use scribe_core::Result;

// Import the types module
pub use ast::types;

/// Main entry point for code analysis
pub struct Analysis {
    parser: Parser,
    analyzer: CodeAnalyzer,
}

impl Analysis {
    /// Create a new analysis instance
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: Parser::new()?,
            analyzer: CodeAnalyzer::new(),
        })
    }

    /// Analyze a piece of code
    pub async fn analyze(&self, code: &str, language: &str) -> Result<AnalysisResult> {
        let ast = self.parser.parse(code, language)?;
        self.analyzer.analyze(&ast).await
    }
}

impl Default for Analysis {
    fn default() -> Self {
        Self::new().expect("Failed to create Analysis")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_analysis_creation() {
        let analysis = Analysis::new();
        assert!(analysis.is_ok());
    }
}
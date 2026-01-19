//! # Scribe Analysis
//!
//! Code analysis algorithms and heuristic scoring for the Scribe library.
//! This crate provides sophisticated file prioritization using multi-dimensional
//! heuristic scoring, template detection, and import graph analysis.

pub mod ast_import_parser;
pub mod heuristics;
pub mod language_support;
pub mod swc_import_extractor;

// Re-export main heuristic types
pub use heuristics::{
    get_template_score_boost, import_matches_file, is_template_file, DocumentAnalysis,
    HeuristicScorer, HeuristicSystem, HeuristicWeights, ImportGraph, ImportGraphBuilder,
    ScoreComponents, ScoringFeatures, TemplateDetector, TemplateEngine,
};

// Re-export language support types
pub use language_support::{
    analyze_file_language, AstLanguage, ClassInfo, DocumentationAnalyzer, DocumentationCoverage,
    FunctionExtractor, FunctionInfo, LanguageAnalysisResult, LanguageFeatures, LanguageMetrics,
    LanguageSpecificComplexity, LanguageSupport, LanguageTier,
};

//! # Core Scoring Algorithms for Heuristic File Prioritization
//!
//! Implements the multi-dimensional scoring system from the Python FastPath heuristics:
//!
//! ## Scoring Formula
//! ```text
//! final_score = Σ(weight_i × normalized_score_i) + priority_boost + template_boost
//! ```
//!
//! Where component scores include:
//! - Documentation importance (doc_score)
//! - README prioritization (readme_score)
//! - Import graph centrality (import_score)
//! - Path depth penalty (path_score)
//! - Test-code relationships (test_link_score)
//! - Git churn recency (churn_score)
//! - PageRank centrality (centrality_score, V2)
//! - Entrypoint detection (entrypoint_score)
//! - Examples detection (examples_score)

use std::collections::HashMap;
use scribe_core::Result;
use super::{ScanResult, import_analysis::ImportGraph};

// Public modules
pub mod types;
pub mod raw_scores;
pub mod normalization;
pub mod final_scoring;

// Re-export main types
pub use types::{ScoreComponents, RawScoreComponents, HeuristicWeights, ScoringFeatures};
pub use normalization::{NormalizedScores, NormalizationStats};

/// Main heuristic scorer that coordinates all scoring components
#[derive(Debug)]
pub struct HeuristicScorer {
    weights: HeuristicWeights,
    import_graph: Option<ImportGraph>,
    norm_stats: Option<NormalizationStats>,
}

impl HeuristicScorer {
    /// Create a new scorer with given weights
    pub fn new(weights: HeuristicWeights) -> Self {
        Self {
            weights,
            import_graph: None,
            norm_stats: None,
        }
    }
    
    /// Create a scorer with default weights
    pub fn default() -> Self {
        Self::new(HeuristicWeights::default())
    }
    
    /// Set the import graph for centrality calculations
    pub fn set_import_graph(&mut self, graph: ImportGraph) {
        self.import_graph = Some(graph);
    }
    
    /// Score a single file within the context of all files
    pub fn score_file<T>(&mut self, file: &T, all_files: &[T]) -> Result<ScoreComponents> 
    where 
        T: ScanResult,
    {
        // Build normalization statistics if not cached
        if self.norm_stats.is_none() {
            self.norm_stats = Some(normalization::build_normalization_stats(all_files));
        }
        
        let norm_stats = self.norm_stats.as_ref().unwrap();
        let raw_scores = raw_scores::calculate_raw_scores(file, &self.weights, self.import_graph.as_ref());
        let normalized_scores = normalization::normalize_scores(&raw_scores, norm_stats);
        
        // Calculate template boost
        let template_boost = if self.weights.features.enable_template_boost {
            super::template_detection::get_template_score_boost(file.path()).unwrap_or(0.0)
        } else {
            0.0
        };
        
        // Apply weighted formula
        let final_score = final_scoring::calculate_final_score(
            &normalized_scores, 
            &self.weights, 
            template_boost, 
            file.priority_boost()
        );
        
        Ok(ScoreComponents {
            final_score,
            doc_score: normalized_scores.doc_score,
            readme_score: normalized_scores.readme_score,
            import_score: normalized_scores.import_score,
            path_score: normalized_scores.path_score,
            test_link_score: normalized_scores.test_link_score,
            churn_score: normalized_scores.churn_score,
            centrality_score: normalized_scores.centrality_score,
            entrypoint_score: normalized_scores.entrypoint_score,
            examples_score: normalized_scores.examples_score,
            priority_boost: file.priority_boost(),
            template_boost,
            raw_scores,
            weights: self.weights.clone(),
        })
    }
    
    /// Score all files and return ranked results
    pub fn score_all_files<T>(&mut self, files: &[T]) -> Result<Vec<(usize, ScoreComponents)>>
    where 
        T: ScanResult,
    {
        let mut scored_files = Vec::new();
        
        for (idx, file) in files.iter().enumerate() {
            let score = self.score_file(file, files)?;
            scored_files.push((idx, score));
        }
        
        // Sort by final score (descending)
        scored_files.sort_by(|a, b| b.1.final_score.partial_cmp(&a.1.final_score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(scored_files)
    }
    
    /// Score files with custom weights for specific use cases
    pub fn score_with_preset<T>(&mut self, files: &[T], preset: WeightPreset) -> Result<Vec<(usize, ScoreComponents)>>
    where 
        T: ScanResult,
    {
        // Temporarily change weights
        let original_weights = self.weights.clone();
        self.weights = match preset {
            WeightPreset::Documentation => HeuristicWeights::for_documentation(),
            WeightPreset::CoreCode => HeuristicWeights::for_core_code(),
            WeightPreset::Tests => HeuristicWeights::for_tests(),
            WeightPreset::Balanced => HeuristicWeights::balanced(),
        };
        
        // Clear cached normalization stats since weights changed
        self.norm_stats = None;
        
        let result = self.score_all_files(files);
        
        // Restore original weights
        self.weights = original_weights;
        self.norm_stats = None;
        
        result
    }
    
    /// Get current weights
    pub fn weights(&self) -> &HeuristicWeights {
        &self.weights
    }
    
    /// Update weights
    pub fn set_weights(&mut self, weights: HeuristicWeights) {
        self.weights = weights;
        self.norm_stats = None; // Clear cache
    }
}

/// Preset weight configurations for common use cases
#[derive(Debug, Clone, Copy)]
pub enum WeightPreset {
    Documentation,
    CoreCode,
    Tests,
    Balanced,
}

impl Default for HeuristicScorer {
    fn default() -> Self {
        Self::new(HeuristicWeights::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    // Mock implementation for testing
    struct MockFile {
        path: String,
        is_docs: bool,
        is_readme: bool,
        depth: usize,
        priority_boost: f64,
    }
    
    impl ScanResult for MockFile {
        fn path(&self) -> &str { &self.path }
        fn relative_path(&self) -> &str { &self.path }
        fn depth(&self) -> usize { self.depth }
        fn is_docs(&self) -> bool { self.is_docs }
        fn is_readme(&self) -> bool { self.is_readme }
        fn is_test(&self) -> bool { false }
        fn is_entrypoint(&self) -> bool { false }
        fn has_examples(&self) -> bool { false }
        fn priority_boost(&self) -> f64 { self.priority_boost }
        fn churn_score(&self) -> f64 { 0.0 }
        fn centrality_in(&self) -> f64 { 0.0 }
        fn imports(&self) -> Option<&[String]> { None }
        fn doc_analysis(&self) -> Option<&crate::heuristics::DocumentAnalysis> { None }
    }
    
    #[test]
    fn test_basic_scoring() {
        let mut scorer = HeuristicScorer::default();
        
        let files = vec![
            MockFile {
                path: "README.md".to_string(),
                is_docs: false,
                is_readme: true,
                depth: 1,
                priority_boost: 0.0,
            },
            MockFile {
                path: "src/lib.rs".to_string(),
                is_docs: false,
                is_readme: false,
                depth: 2,
                priority_boost: 0.0,
            },
        ];
        
        let scores = scorer.score_all_files(&files).unwrap();
        assert_eq!(scores.len(), 2);
        
        // README should score higher
        assert!(scores[0].1.final_score > 0.0);
    }
    
    #[test]
    fn test_weight_presets() {
        let mut scorer = HeuristicScorer::default();
        
        let files = vec![
            MockFile {
                path: "README.md".to_string(),
                is_docs: false,
                is_readme: true,
                depth: 1,
                priority_boost: 0.0,
            },
        ];
        
        let doc_scores = scorer.score_with_preset(&files, WeightPreset::Documentation).unwrap();
        let core_scores = scorer.score_with_preset(&files, WeightPreset::CoreCode).unwrap();
        
        // Documentation preset should give higher scores to README
        assert!(doc_scores[0].1.final_score >= core_scores[0].1.final_score);
    }
    
    #[test]
    fn test_score_explanation() {
        let mut scorer = HeuristicScorer::default();
        
        let files = vec![
            MockFile {
                path: "README.md".to_string(),
                is_docs: false,
                is_readme: true,
                depth: 1,
                priority_boost: 1.0,
            },
        ];
        
        let score = scorer.score_file(&files[0], &files).unwrap();
        let explanation = score.explanation();
        
        assert!(explanation.contains("Score:"));
        assert!(!score.primary_importance_reason().is_empty());
    }
}
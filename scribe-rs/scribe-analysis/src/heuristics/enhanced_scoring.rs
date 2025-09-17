//! # Enhanced Scoring System with Complexity Integration
//!
//! This module extends the basic heuristic scoring with comprehensive complexity analysis,
//! providing deeper insights into code quality and maintainability for better file selection.
//!
//! ## Enhanced Features
//!
//! - **Complexity-Aware Scoring**: Integrates cyclomatic, cognitive, and maintainability metrics
//! - **Quality-Based Prioritization**: Considers code quality alongside importance
//! - **Language-Specific Analysis**: Tailored complexity analysis per programming language
//! - **Maintainability Assessment**: Factors in long-term code maintenance concerns
//! - **Adaptive Weights**: Adjusts scoring based on repository characteristics

use super::scoring::RawScoreComponents;
use super::{HeuristicWeights, ScanResult, ScoreComponents};
use crate::complexity::{ComplexityAnalyzer, ComplexityConfig, ComplexityMetrics};
use scribe_core::{Result, ScribeError};
use std::collections::HashMap;
use rayon::prelude::*;

/// Enhanced score components that include complexity metrics
#[derive(Debug, Clone)]
pub struct EnhancedScoreComponents {
    /// Base score components from standard heuristics
    pub base_score: ScoreComponents,

    /// Complexity-based scores
    pub complexity_score: f64,
    pub maintainability_score: f64,
    pub cognitive_score: f64,
    pub quality_score: f64,

    /// Combined final score
    pub enhanced_final_score: f64,

    /// Detailed complexity metrics
    pub complexity_metrics: Option<ComplexityMetrics>,

    /// Complexity-adjusted weights
    pub adjusted_weights: EnhancedWeights,
}

/// Enhanced weights that include complexity factors
#[derive(Debug, Clone)]
pub struct EnhancedWeights {
    /// Base heuristic weights
    pub base_weights: HeuristicWeights,

    /// Complexity weight factors
    pub complexity_weight: f64,
    pub maintainability_weight: f64,
    pub cognitive_weight: f64,
    pub quality_weight: f64,

    /// Adaptive weight adjustments
    pub adaptive_factors: AdaptiveFactors,
}

/// Adaptive factors that adjust scoring based on repository characteristics
#[derive(Debug, Clone)]
pub struct AdaptiveFactors {
    /// Repository size factor (larger repos may prefer simpler files)
    pub repo_size_factor: f64,

    /// Language complexity factor (some languages naturally more complex)
    pub language_factor: f64,

    /// Project maturity factor (mature projects may prioritize maintainability)
    pub maturity_factor: f64,

    /// Team experience factor (affects complexity tolerance)
    pub experience_factor: f64,
}

/// Enhanced heuristic scorer with complexity integration
#[derive(Debug)]
pub struct EnhancedHeuristicScorer {
    /// Base scorer for standard heuristics
    base_scorer: super::scoring::HeuristicScorer,

    /// Complexity analyzer
    complexity_analyzer: ComplexityAnalyzer,

    /// Enhanced weights configuration
    weights: EnhancedWeights,

    /// Repository characteristics for adaptive scoring
    repo_characteristics: RepositoryCharacteristics,

    /// Content cache for complexity analysis
    content_cache: HashMap<String, ComplexityMetrics>,

    /// Whether to enable expensive complexity analysis (disabled by default for performance)
    enable_complexity_analysis: bool,
}

/// Repository characteristics for adaptive scoring
#[derive(Debug, Clone)]
pub struct RepositoryCharacteristics {
    /// Total number of files in repository
    pub total_files: usize,

    /// Primary programming languages
    pub primary_languages: Vec<String>,

    /// Repository age in months
    pub age_months: usize,

    /// Average team size
    pub team_size: usize,

    /// Project type (library, application, framework, etc.)
    pub project_type: ProjectType,
}

/// Project type classification
#[derive(Debug, Clone)]
pub enum ProjectType {
    Library,
    Application,
    Framework,
    Tool,
    Game,
    WebService,
    EmbeddedSystem,
    Unknown,
}

impl Default for EnhancedWeights {
    fn default() -> Self {
        Self {
            base_weights: HeuristicWeights::default(),
            complexity_weight: 0.15,
            maintainability_weight: 0.20,
            cognitive_weight: 0.10,
            quality_weight: 0.15,
            adaptive_factors: AdaptiveFactors::default(),
        }
    }
}

impl Default for AdaptiveFactors {
    fn default() -> Self {
        Self {
            repo_size_factor: 1.0,
            language_factor: 1.0,
            maturity_factor: 1.0,
            experience_factor: 1.0,
        }
    }
}

impl Default for RepositoryCharacteristics {
    fn default() -> Self {
        Self {
            total_files: 100,
            primary_languages: vec!["rust".to_string()],
            age_months: 12,
            team_size: 3,
            project_type: ProjectType::Application,
        }
    }
}

impl EnhancedHeuristicScorer {
    /// Create a new enhanced scorer with default configuration
    pub fn new() -> Self {
        let base_weights = HeuristicWeights::default();
        let base_scorer = super::scoring::HeuristicScorer::new(base_weights.clone());

        Self {
            base_scorer,
            complexity_analyzer: ComplexityAnalyzer::new(),
            weights: EnhancedWeights::default(),
            repo_characteristics: RepositoryCharacteristics::default(),
            content_cache: HashMap::new(),
            enable_complexity_analysis: false, // TEMPORARILY DISABLED to test baseline performance
        }
    }

    /// Enable complexity analysis (WARNING: This significantly impacts performance)
    pub fn enable_complexity_analysis(&mut self) {
        self.enable_complexity_analysis = true;
    }

    /// Disable complexity analysis for better performance
    pub fn disable_complexity_analysis(&mut self) {
        self.enable_complexity_analysis = false;
        self.content_cache.clear(); // Clear cache to save memory
    }

    /// Create enhanced scorer with custom configuration
    pub fn with_config(
        weights: EnhancedWeights,
        complexity_config: ComplexityConfig,
        repo_characteristics: RepositoryCharacteristics,
    ) -> Self {
        let base_scorer = super::scoring::HeuristicScorer::new(weights.base_weights.clone());
        let complexity_analyzer = ComplexityAnalyzer::with_config(complexity_config);

        Self {
            base_scorer,
            complexity_analyzer,
            weights,
            repo_characteristics,
            content_cache: HashMap::new(),
            enable_complexity_analysis: false, // TEMPORARILY DISABLED to test baseline performance
        }
    }

    /// Score a file with enhanced complexity-aware analysis
    pub fn score_file_enhanced<T>(
        &mut self,
        file: &T,
        file_content: &str,
        all_files: &[T],
    ) -> Result<EnhancedScoreComponents>
    where
        T: ScanResult + Clone,
    {
        // Get base heuristic score
        let base_score = self.base_scorer.score_file(file, all_files)?;

        // Detect language from file path
        let language = self.detect_language(file.path());

        // Analyze complexity only if enabled (with caching)
        let (
            complexity_metrics,
            complexity_score,
            maintainability_score,
            cognitive_score,
            quality_score,
        ) = if self.enable_complexity_analysis {
            let complexity_metrics = if let Some(cached) = self.content_cache.get(file.path()) {
                cached.clone()
            } else {
                let metrics = self
                    .complexity_analyzer
                    .analyze_content(file_content, &language)?;
                self.content_cache
                    .insert(file.path().to_string(), metrics.clone());
                metrics
            };

            // Calculate complexity-based scores
            let complexity_score = self.calculate_complexity_score(&complexity_metrics);
            let maintainability_score = self.calculate_maintainability_score(&complexity_metrics);
            let cognitive_score = self.calculate_cognitive_score(&complexity_metrics);
            let quality_score = self.calculate_quality_score(&complexity_metrics);

            (
                Some(complexity_metrics),
                complexity_score,
                maintainability_score,
                cognitive_score,
                quality_score,
            )
        } else {
            // Skip expensive complexity analysis - use neutral/default scores
            (None, 0.5, 0.5, 0.5, 0.5)
        };

        // Apply adaptive adjustments
        let adjusted_weights = if let Some(ref metrics) = complexity_metrics {
            self.calculate_adaptive_weights(file, metrics)
        } else {
            // Use default weights when complexity analysis is disabled
            self.weights.clone()
        };

        // Calculate enhanced final score
        let enhanced_final_score = self.calculate_enhanced_final_score(
            &base_score,
            complexity_score,
            maintainability_score,
            cognitive_score,
            quality_score,
            &adjusted_weights,
        );

        Ok(EnhancedScoreComponents {
            base_score,
            complexity_score,
            maintainability_score,
            cognitive_score,
            quality_score,
            enhanced_final_score,
            complexity_metrics,
            adjusted_weights,
        })
    }

    /// Score all files with enhanced analysis
    pub fn score_all_files_enhanced<T>(
        &mut self,
        files_with_content: &[(T, String)],
    ) -> Result<Vec<(usize, EnhancedScoreComponents)>>
    where
        T: ScanResult + Clone + Sync + Send,
    {
        let files: Vec<_> = files_with_content.iter().map(|(f, _)| f.clone()).collect();
        
        // PERFORMANCE OPTIMIZATION: Parallelize complexity analysis first
        let complexity_results: Result<HashMap<usize, Option<ComplexityMetrics>>> = if self.enable_complexity_analysis {
            // Create a snapshot of cache for parallel access
            let cache_snapshot: HashMap<String, ComplexityMetrics> = self.content_cache.clone();
            
            // Compute complexity metrics in parallel for all files
            let results: Result<Vec<(usize, Option<ComplexityMetrics>)>> = files_with_content
                .par_iter()
                .enumerate()
                .map(|(idx, (file, content))| {
                    // Check cache first
                    if let Some(cached) = cache_snapshot.get(file.path()) {
                        return Ok((idx, Some(cached.clone())));
                    }
                    
                    // Detect language and analyze complexity
                    let language = Self::detect_language_static(file.path());
                    let analyzer = ComplexityAnalyzer::new();
                    
                    match analyzer.analyze_content(content, &language) {
                        Ok(metrics) => Ok((idx, Some(metrics))),
                        Err(_) => Ok((idx, None)), // Skip files that can't be analyzed
                    }
                })
                .collect();
            
            results.map(|vec| vec.into_iter().collect())
        } else {
            Ok(HashMap::new())
        };

        let complexity_results = complexity_results?;
        
        // Update cache with new results (sequential for cache safety)
        for (idx, metrics_opt) in &complexity_results {
            if let Some(metrics) = metrics_opt {
                let file_path = files_with_content[*idx].0.path().to_string();
                self.content_cache.insert(file_path, metrics.clone());
            }
        }

        // Now score all files sequentially with pre-computed complexity metrics
        let mut scored_files = Vec::new();
        for (idx, (file, content)) in files_with_content.iter().enumerate() {
            let score = self.score_file_enhanced_with_precomputed_complexity(
                file, 
                content, 
                &files, 
                complexity_results.get(&idx).and_then(|opt| opt.as_ref())
            )?;
            scored_files.push((idx, score));
        }

        // Sort by enhanced final score (descending)
        scored_files.sort_by(|a, b| {
            b.1.enhanced_final_score
                .partial_cmp(&a.1.enhanced_final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored_files)
    }

    /// Score a file with pre-computed complexity metrics (for parallel optimization)
    fn score_file_enhanced_with_precomputed_complexity<T>(
        &mut self,
        file: &T,
        file_content: &str,
        all_files: &[T],
        precomputed_complexity: Option<&ComplexityMetrics>,
    ) -> Result<EnhancedScoreComponents>
    where
        T: ScanResult + Clone,
    {
        // Get base heuristic score
        let base_score = self.base_scorer.score_file(file, all_files)?;

        // Use pre-computed complexity metrics or defaults
        let (
            complexity_metrics,
            complexity_score,
            maintainability_score,
            cognitive_score,
            quality_score,
        ) = if let Some(metrics) = precomputed_complexity {
            // Use pre-computed metrics
            let complexity_score = self.calculate_complexity_score(metrics);
            let maintainability_score = self.calculate_maintainability_score(metrics);
            let cognitive_score = self.calculate_cognitive_score(metrics);
            let quality_score = self.calculate_quality_score(metrics);

            (
                Some(metrics.clone()),
                complexity_score,
                maintainability_score,
                cognitive_score,
                quality_score,
            )
        } else {
            // Skip expensive complexity analysis - use neutral/default scores
            (None, 0.5, 0.5, 0.5, 0.5)
        };

        // Apply adaptive adjustments
        let adjusted_weights = if let Some(ref metrics) = complexity_metrics {
            self.calculate_adaptive_weights(file, metrics)
        } else {
            // Use default weights when complexity analysis is disabled
            self.weights.clone()
        };

        // Calculate enhanced final score
        let enhanced_final_score = self.calculate_enhanced_final_score(
            &base_score,
            complexity_score,
            maintainability_score,
            cognitive_score,
            quality_score,
            &adjusted_weights,
        );

        Ok(EnhancedScoreComponents {
            base_score,
            complexity_score,
            maintainability_score,
            cognitive_score,
            quality_score,
            enhanced_final_score,
            complexity_metrics,
            adjusted_weights,
        })
    }

    /// Detect programming language from file path
    fn detect_language(&self, path: &str) -> String {
        Self::detect_language_static(path)
    }

    /// Static version for parallel processing
    fn detect_language_static(path: &str) -> String {
        let extension = std::path::Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "rs" => "rust",
            "py" => "python",
            "js" => "javascript",
            "ts" => "typescript",
            "java" => "java",
            "cs" => "c#",
            "go" => "go",
            "c" => "c",
            "cpp" | "cc" | "cxx" => "cpp",
            "h" | "hpp" => "c",
            "rb" => "ruby",
            "php" => "php",
            "swift" => "swift",
            "kt" => "kotlin",
            "scala" => "scala",
            _ => "unknown",
        }
        .to_string()
    }

    /// Calculate complexity-based score (0-1, where 1 is good)
    fn calculate_complexity_score(&self, metrics: &ComplexityMetrics) -> f64 {
        // Invert complexity score - lower complexity is better
        1.0 - metrics.complexity_score()
    }

    /// Calculate maintainability score (0-1)
    fn calculate_maintainability_score(&self, metrics: &ComplexityMetrics) -> f64 {
        // Maintainability index is 0-100, normalize to 0-1
        metrics.maintainability_index / 100.0
    }

    /// Calculate cognitive load score (0-1, where 1 is good)
    fn calculate_cognitive_score(&self, metrics: &ComplexityMetrics) -> f64 {
        // Lower cognitive complexity is better
        let cognitive_ratio = metrics.cognitive_complexity as f64 / 20.0; // Normalize to rough 0-1 range
        (1.0 - cognitive_ratio.min(1.0)).max(0.0)
    }

    /// Calculate overall quality score (0-1)
    fn calculate_quality_score(&self, metrics: &ComplexityMetrics) -> f64 {
        // Composite quality score
        let complexity_factor = 1.0 - (metrics.cyclomatic_complexity as f64 / 15.0).min(1.0);
        let nesting_factor = 1.0 - (metrics.max_nesting_depth as f64 / 6.0).min(1.0);
        let density_factor = metrics.code_density.min(1.0);
        let comment_factor = (metrics.comment_ratio * 2.0).min(1.0); // Good commenting is valuable

        (complexity_factor * 0.3
            + nesting_factor * 0.2
            + density_factor * 0.3
            + comment_factor * 0.2)
            .min(1.0)
    }

    /// Calculate adaptive weights based on file and repository characteristics
    fn calculate_adaptive_weights<T>(
        &self,
        file: &T,
        metrics: &ComplexityMetrics,
    ) -> EnhancedWeights
    where
        T: ScanResult,
    {
        let mut weights = self.weights.clone();

        // Adjust weights based on repository size
        if self.repo_characteristics.total_files > 1000 {
            // Large repos: prioritize simplicity and maintainability
            weights.maintainability_weight *= 1.3;
            weights.complexity_weight *= 1.2;
        } else if self.repo_characteristics.total_files < 50 {
            // Small repos: focus more on functionality
            weights.base_weights.import_weight *= 1.2;
            weights.base_weights.doc_weight *= 1.1;
        }

        // Adjust based on project type
        match self.repo_characteristics.project_type {
            ProjectType::Library => {
                // Libraries need excellent documentation and maintainability
                weights.base_weights.doc_weight *= 1.4;
                weights.maintainability_weight *= 1.3;
                weights.quality_weight *= 1.2;
            }
            ProjectType::Framework => {
                // Frameworks need clear architecture and examples
                weights.base_weights.entrypoint_weight *= 1.3;
                weights.base_weights.examples_weight *= 1.4;
                weights.quality_weight *= 1.2;
            }
            ProjectType::Tool => {
                // Tools prioritize main functionality and simplicity
                weights.base_weights.entrypoint_weight *= 1.5;
                weights.complexity_weight *= 1.3;
            }
            _ => {
                // Default adjustments for applications
            }
        }

        // Adjust based on file complexity
        if metrics.cyclomatic_complexity > 10 {
            // High complexity files might be core logic - boost importance
            weights.base_weights.import_weight *= 1.2;
        }

        if metrics.maintainability_index < 30.0 {
            // Low maintainability - might indicate technical debt hotspots
            weights.maintainability_weight *= 1.4;
        }

        // Language-specific adjustments
        let language = &metrics.language_metrics.language;
        match language.as_str() {
            "rust" => {
                // Rust: Consider ownership complexity
                if let Some(ownership) = metrics
                    .language_metrics
                    .complexity_factors
                    .get("ownership_complexity")
                {
                    if *ownership > 5.0 {
                        weights.complexity_weight *= 1.2;
                    }
                }
            }
            "python" => {
                // Python: Value documentation and simplicity
                weights.base_weights.doc_weight *= 1.1;
                weights.complexity_weight *= 1.1;
            }
            "javascript" | "typescript" => {
                // JS/TS: Consider async complexity
                if let Some(async_complexity) = metrics
                    .language_metrics
                    .complexity_factors
                    .get("promise_complexity")
                {
                    if *async_complexity > 3.0 {
                        weights.cognitive_weight *= 1.2;
                    }
                }
            }
            _ => {}
        }

        weights
    }

    /// Calculate the final enhanced score
    fn calculate_enhanced_final_score(
        &self,
        base_score: &ScoreComponents,
        complexity_score: f64,
        maintainability_score: f64,
        cognitive_score: f64,
        quality_score: f64,
        weights: &EnhancedWeights,
    ) -> f64 {
        // Combine base score with complexity metrics
        let base_contribution = base_score.final_score * 0.6; // Base heuristics weight

        let complexity_contribution = complexity_score * weights.complexity_weight
            + maintainability_score * weights.maintainability_weight
            + cognitive_score * weights.cognitive_weight
            + quality_score * weights.quality_weight;

        let enhanced_contribution = complexity_contribution * 0.4; // Complexity metrics weight

        // Apply adaptive factors
        let final_score = (base_contribution + enhanced_contribution)
            * weights.adaptive_factors.repo_size_factor
            * weights.adaptive_factors.language_factor
            * weights.adaptive_factors.maturity_factor
            * weights.adaptive_factors.experience_factor;

        final_score.min(2.0) // Cap the score to prevent extreme values
    }

    /// Update repository characteristics
    pub fn update_repository_characteristics(
        &mut self,
        characteristics: RepositoryCharacteristics,
    ) {
        self.repo_characteristics = characteristics;

        // Recalculate adaptive factors based on new characteristics
        self.weights.adaptive_factors = self.calculate_adaptive_factors();
    }

    /// Calculate adaptive factors based on repository characteristics
    fn calculate_adaptive_factors(&self) -> AdaptiveFactors {
        let repo_size_factor = match self.repo_characteristics.total_files {
            0..=50 => 1.1,      // Small repos - boost importance
            51..=500 => 1.0,    // Medium repos - neutral
            501..=2000 => 0.95, // Large repos - slight penalty
            _ => 0.9,           // Very large repos - prefer simpler files
        };

        let language_factor = if self
            .repo_characteristics
            .primary_languages
            .contains(&"rust".to_string())
        {
            1.05 // Rust projects tend to have good practices
        } else if self
            .repo_characteristics
            .primary_languages
            .contains(&"javascript".to_string())
        {
            0.95 // JS can be more complex to analyze
        } else {
            1.0
        };

        let maturity_factor = match self.repo_characteristics.age_months {
            0..=6 => 0.9,   // New projects - focus on functionality
            7..=24 => 1.0,  // Maturing projects - balanced
            25..=60 => 1.1, // Mature projects - prioritize maintainability
            _ => 1.2,       // Very mature projects - heavily prioritize quality
        };

        let experience_factor = match self.repo_characteristics.team_size {
            1 => 1.1,       // Solo projects - prefer simpler code
            2..=5 => 1.0,   // Small teams - balanced
            6..=15 => 0.95, // Medium teams - can handle complexity
            _ => 0.9,       // Large teams - prefer well-structured code
        };

        AdaptiveFactors {
            repo_size_factor,
            language_factor,
            maturity_factor,
            experience_factor,
        }
    }

    /// Clear the complexity metrics cache
    pub fn clear_cache(&mut self) {
        self.content_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.content_cache.len(), self.content_cache.capacity())
    }
}

impl EnhancedScoreComponents {
    /// Get a breakdown of score contributions
    pub fn score_breakdown(&self) -> HashMap<String, f64> {
        let mut breakdown = self.base_score.as_map();

        breakdown.insert("complexity_score".to_string(), self.complexity_score);
        breakdown.insert(
            "maintainability_score".to_string(),
            self.maintainability_score,
        );
        breakdown.insert("cognitive_score".to_string(), self.cognitive_score);
        breakdown.insert("quality_score".to_string(), self.quality_score);
        breakdown.insert(
            "enhanced_final_score".to_string(),
            self.enhanced_final_score,
        );

        breakdown
    }

    /// Get the dominant scoring factor
    pub fn dominant_factor(&self) -> (&'static str, f64) {
        let factors = [
            ("base_heuristics", self.base_score.final_score * 0.6),
            (
                "complexity",
                self.complexity_score * self.adjusted_weights.complexity_weight * 0.4,
            ),
            (
                "maintainability",
                self.maintainability_score * self.adjusted_weights.maintainability_weight * 0.4,
            ),
            (
                "cognitive",
                self.cognitive_score * self.adjusted_weights.cognitive_weight * 0.4,
            ),
            (
                "quality",
                self.quality_score * self.adjusted_weights.quality_weight * 0.4,
            ),
        ];

        factors
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, score)| (*name, *score))
            .unwrap_or(("none", 0.0))
    }

    /// Get a human-readable explanation of the score
    pub fn explanation(&self) -> String {
        let (dominant, _) = self.dominant_factor();
        let complexity_summary = if let Some(metrics) = &self.complexity_metrics {
            metrics.summary()
        } else {
            "No complexity metrics".to_string()
        };

        format!(
            "Score: {:.3} (dominated by {}), Base: {:.3}, Quality: {:.3}, {}",
            self.enhanced_final_score,
            dominant,
            self.base_score.final_score,
            self.quality_score,
            complexity_summary
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristics::DocumentAnalysis;

    // Mock scan result for testing
    #[derive(Debug, Clone)]
    struct MockScanResult {
        path: String,
        relative_path: String,
        depth: usize,
        is_docs: bool,
        is_readme: bool,
        is_test: bool,
        is_entrypoint: bool,
        has_examples: bool,
        priority_boost: f64,
        churn_score: f64,
        centrality_in: f64,
        imports: Option<Vec<String>>,
        doc_analysis: Option<DocumentAnalysis>,
    }

    impl MockScanResult {
        fn new(path: &str) -> Self {
            Self {
                path: path.to_string(),
                relative_path: path.to_string(),
                depth: path.matches('/').count(),
                is_docs: path.contains("doc") || path.ends_with(".md"),
                is_readme: path.to_lowercase().contains("readme"),
                is_test: path.contains("test") || path.contains("spec"),
                is_entrypoint: path.contains("main") || path.contains("index"),
                has_examples: path.contains("example") || path.contains("demo"),
                priority_boost: 0.0,
                churn_score: 0.5,
                centrality_in: 0.3,
                imports: Some(vec!["std::collections::HashMap".to_string()]),
                doc_analysis: Some(DocumentAnalysis::new()),
            }
        }
    }

    impl super::super::ScanResult for MockScanResult {
        fn path(&self) -> &str {
            &self.path
        }
        fn relative_path(&self) -> &str {
            &self.relative_path
        }
        fn depth(&self) -> usize {
            self.depth
        }
        fn is_docs(&self) -> bool {
            self.is_docs
        }
        fn is_readme(&self) -> bool {
            self.is_readme
        }
        fn is_test(&self) -> bool {
            self.is_test
        }
        fn is_entrypoint(&self) -> bool {
            self.is_entrypoint
        }
        fn has_examples(&self) -> bool {
            self.has_examples
        }
        fn priority_boost(&self) -> f64 {
            self.priority_boost
        }
        fn churn_score(&self) -> f64 {
            self.churn_score
        }
        fn centrality_in(&self) -> f64 {
            self.centrality_in
        }
        fn imports(&self) -> Option<&[String]> {
            self.imports.as_deref()
        }
        fn doc_analysis(&self) -> Option<&DocumentAnalysis> {
            self.doc_analysis.as_ref()
        }
    }

    #[test]
    fn test_enhanced_scorer_creation() {
        let scorer = EnhancedHeuristicScorer::new();
        assert!(scorer.weights.complexity_weight > 0.0);
        assert!(scorer.weights.maintainability_weight > 0.0);
    }

    #[test]
    fn test_language_detection() {
        let scorer = EnhancedHeuristicScorer::new();

        assert_eq!(scorer.detect_language("src/main.rs"), "rust");
        assert_eq!(scorer.detect_language("app.py"), "python");
        assert_eq!(scorer.detect_language("script.js"), "javascript");
        assert_eq!(scorer.detect_language("component.ts"), "typescript");
        assert_eq!(scorer.detect_language("Main.java"), "java");
    }

    #[test]
    fn test_enhanced_file_scoring() {
        let mut scorer = EnhancedHeuristicScorer::new();
        scorer.enable_complexity_analysis(); // Enable complexity analysis for testing

        let file = MockScanResult::new("src/main.rs");
        let content = r#"
fn main() {
    if condition() {
        for i in 0..10 {
            println!("Hello {}", i);
        }
    }
}
"#;
        let files = vec![file.clone()];

        let result = scorer.score_file_enhanced(&file, content, &files);
        assert!(result.is_ok());

        let score = result.unwrap();
        assert!(score.enhanced_final_score > 0.0);
        assert!(score.complexity_score >= 0.0 && score.complexity_score <= 1.0);
        assert!(score.quality_score >= 0.0 && score.quality_score <= 1.0);
        assert!(score.complexity_metrics.is_some());
    }

    #[test]
    fn test_adaptive_weights() {
        let weights = EnhancedWeights::default();
        let complexity_config = ComplexityConfig::default();
        let mut repo_chars = RepositoryCharacteristics::default();
        repo_chars.project_type = ProjectType::Library;
        repo_chars.total_files = 1500; // Large repository

        let mut scorer =
            EnhancedHeuristicScorer::with_config(weights, complexity_config, repo_chars);

        let file = MockScanResult::new("src/lib.rs");
        let simple_content = "fn simple() { println!(\"hello\"); }";
        let files = vec![file.clone()];

        let result = scorer.score_file_enhanced(&file, simple_content, &files);
        assert!(result.is_ok());

        let score = result.unwrap();

        // For a library, documentation and maintainability should have higher weights
        assert!(
            score.adjusted_weights.base_weights.doc_weight
                >= score.adjusted_weights.base_weights.import_weight
        );
    }

    #[test]
    fn test_complexity_vs_simple_code() {
        let mut scorer = EnhancedHeuristicScorer::new();
        scorer.enable_complexity_analysis(); // Enable complexity analysis for testing

        let file1 = MockScanResult::new("simple.rs");
        let simple_content = "fn simple() { println!(\"hello\"); }";

        let file2 = MockScanResult::new("complex.rs");
        let complex_content = r#"
fn complex() {
    for i in 0..100 {
        if i % 2 == 0 {
            while condition() {
                match value {
                    1 => { if nested() { deep(); } },
                    2 => { if more_nested() { deeper(); } },
                    _ => { if even_more() { deepest(); } },
                }
            }
        }
    }
}
"#;

        let files = vec![file1.clone(), file2.clone()];

        let simple_score = scorer
            .score_file_enhanced(&file1, simple_content, &files)
            .unwrap();
        let complex_score = scorer
            .score_file_enhanced(&file2, complex_content, &files)
            .unwrap();

        // Simple code should generally score better on complexity metrics
        assert!(simple_score.complexity_score > complex_score.complexity_score);
        assert!(simple_score.cognitive_score > complex_score.cognitive_score);
    }

    #[test]
    fn test_score_breakdown() {
        let mut scorer = EnhancedHeuristicScorer::new();

        let file = MockScanResult::new("test.rs");
        let content = "fn test() { if x > 0 { return 1; } else { return 0; } }";
        let files = vec![file.clone()];

        let score = scorer.score_file_enhanced(&file, content, &files).unwrap();
        let breakdown = score.score_breakdown();

        assert!(breakdown.contains_key("complexity_score"));
        assert!(breakdown.contains_key("maintainability_score"));
        assert!(breakdown.contains_key("cognitive_score"));
        assert!(breakdown.contains_key("quality_score"));
        assert!(breakdown.contains_key("enhanced_final_score"));

        let explanation = score.explanation();
        assert!(explanation.contains("Score:"));
        assert!(explanation.contains("dominated by"));
    }

    #[test]
    fn test_repository_characteristics_update() {
        let mut scorer = EnhancedHeuristicScorer::new();

        let initial_factors = scorer.weights.adaptive_factors.clone();

        let mut new_chars = RepositoryCharacteristics::default();
        new_chars.total_files = 5000; // Much larger
        new_chars.project_type = ProjectType::Framework;
        new_chars.age_months = 48; // Mature project

        scorer.update_repository_characteristics(new_chars);

        let new_factors = &scorer.weights.adaptive_factors;

        // Large, mature projects should have different factors
        assert_ne!(
            initial_factors.repo_size_factor,
            new_factors.repo_size_factor
        );
        assert_ne!(initial_factors.maturity_factor, new_factors.maturity_factor);
    }
}

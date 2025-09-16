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

/// Complete score breakdown for a file
#[derive(Debug, Clone)]
pub struct ScoreComponents {
    /// Final weighted score
    pub final_score: f64,
    
    /// Individual component scores
    pub doc_score: f64,
    pub readme_score: f64,
    pub import_score: f64,
    pub path_score: f64,
    pub test_link_score: f64,
    pub churn_score: f64,
    pub centrality_score: f64,
    pub entrypoint_score: f64,
    pub examples_score: f64,
    
    /// Boost components
    pub priority_boost: f64,
    pub template_boost: f64,
    
    /// Raw component scores before normalization
    pub raw_scores: RawScoreComponents,
    
    /// Applied weights
    pub weights: HeuristicWeights,
}

/// Raw score components before normalization
#[derive(Debug, Clone)]
pub struct RawScoreComponents {
    pub doc_raw: f64,
    pub readme_raw: f64,
    pub import_degree_in: usize,
    pub import_degree_out: usize,
    pub path_depth: usize,
    pub test_links_found: usize,
    pub churn_commits: usize,
    pub centrality_raw: f64,
    pub is_entrypoint: bool,
    pub examples_count: usize,
}

/// Configurable weights for the scoring formula
#[derive(Debug, Clone)]
pub struct HeuristicWeights {
    pub doc_weight: f64,
    pub readme_weight: f64,
    pub import_weight: f64,
    pub path_weight: f64,
    pub test_link_weight: f64,
    pub churn_weight: f64,
    pub centrality_weight: f64, // V2 feature
    pub entrypoint_weight: f64,
    pub examples_weight: f64,
    
    /// Feature flags for advanced capabilities
    pub features: ScoringFeatures,
}

/// Feature flags for scoring system capabilities
#[derive(Debug, Clone)]
pub struct ScoringFeatures {
    /// Enable PageRank centrality calculation (V2)
    pub enable_centrality: bool,
    /// Enable template detection boost
    pub enable_template_boost: bool,
    /// Enable advanced document analysis
    pub enable_doc_analysis: bool,
    /// Enable test-code relationship detection
    pub enable_test_linking: bool,
    /// Enable git churn analysis
    pub enable_churn_analysis: bool,
    /// Enable examples detection
    pub enable_examples_detection: bool,
}

impl Default for HeuristicWeights {
    /// Default V1 weights (matches Python implementation)
    fn default() -> Self {
        Self {
            doc_weight: 0.15,      // Documentation importance
            readme_weight: 0.20,   // README files get priority
            import_weight: 0.20,   // Dependency centrality
            path_weight: 0.10,     // Shallow files preferred
            test_link_weight: 0.10, // Test-code relationships
            churn_weight: 0.15,    // Git activity recency
            centrality_weight: 0.0, // Disabled in V1
            entrypoint_weight: 0.05, // Entry points
            examples_weight: 0.05, // Usage examples
            
            features: ScoringFeatures::v1(),
        }
    }
}

impl HeuristicWeights {
    /// Create V2 weights with advanced features enabled
    pub fn with_v2_features() -> Self {
        Self {
            doc_weight: 0.12,
            readme_weight: 0.18,
            import_weight: 0.15,
            path_weight: 0.08,
            test_link_weight: 0.08,
            churn_weight: 0.12,
            centrality_weight: 0.12, // Enabled in V2
            entrypoint_weight: 0.08,
            examples_weight: 0.07,
            
            features: ScoringFeatures::v2(),
        }
    }
    
    /// Normalize weights to ensure they sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.doc_weight + self.readme_weight + self.import_weight + 
                   self.path_weight + self.test_link_weight + self.churn_weight +
                   self.centrality_weight + self.entrypoint_weight + self.examples_weight;
        
        if total > 0.0 {
            self.doc_weight /= total;
            self.readme_weight /= total;
            self.import_weight /= total;
            self.path_weight /= total;
            self.test_link_weight /= total;
            self.churn_weight /= total;
            self.centrality_weight /= total;
            self.entrypoint_weight /= total;
            self.examples_weight /= total;
        }
    }
    
    /// Get active weight sum (excluding disabled features)
    pub fn active_weight_sum(&self) -> f64 {
        let mut sum = self.doc_weight + self.readme_weight + self.import_weight + 
                     self.path_weight + self.test_link_weight + self.churn_weight +
                     self.entrypoint_weight + self.examples_weight;
        
        if self.features.enable_centrality {
            sum += self.centrality_weight;
        }
        
        sum
    }
}

impl Default for ScoringFeatures {
    fn default() -> Self {
        Self::v1()
    }
}

impl ScoringFeatures {
    /// V1 feature set (stable features only)
    pub fn v1() -> Self {
        Self {
            enable_centrality: false,
            enable_template_boost: true,
            enable_doc_analysis: true,
            enable_test_linking: true,
            enable_churn_analysis: true,
            enable_examples_detection: true,
        }
    }
    
    /// V2 feature set (includes experimental features)
    pub fn v2() -> Self {
        Self {
            enable_centrality: true,
            enable_template_boost: true,
            enable_doc_analysis: true,
            enable_test_linking: true,
            enable_churn_analysis: true,
            enable_examples_detection: true,
        }
    }
    
    /// All features disabled (minimal scoring)
    pub fn minimal() -> Self {
        Self {
            enable_centrality: false,
            enable_template_boost: false,
            enable_doc_analysis: false,
            enable_test_linking: false,
            enable_churn_analysis: false,
            enable_examples_detection: false,
        }
    }
}

/// Core heuristic scoring engine
#[derive(Debug)]
pub struct HeuristicScorer {
    /// Scoring weights configuration
    weights: HeuristicWeights,
    /// Import graph for centrality calculation
    import_graph: Option<ImportGraph>,
    /// Normalization statistics cache
    norm_stats: Option<NormalizationStats>,
}

/// Statistics for score normalization
#[derive(Debug, Clone)]
struct NormalizationStats {
    doc_max: f64,
    import_in_max: f64,
    import_out_max: f64,
    path_max: f64,
    test_links_max: f64,
    churn_max: f64,
    centrality_max: f64,
    examples_max: f64,
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
            self.norm_stats = Some(self.build_normalization_stats(all_files));
        }
        
        let norm_stats = self.norm_stats.as_ref().unwrap();
        let raw_scores = self.calculate_raw_scores(file);
        let normalized_scores = self.normalize_scores(&raw_scores, norm_stats);
        
        // Calculate template boost
        let template_boost = if self.weights.features.enable_template_boost {
            super::template_detection::get_template_score_boost(file.path()).unwrap_or(0.0)
        } else {
            0.0
        };
        
        // Apply weighted formula
        let final_score = self.calculate_final_score(&normalized_scores, template_boost, file.priority_boost());
        
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
    
    /// Calculate raw score components before normalization
    fn calculate_raw_scores<T>(&self, file: &T) -> RawScoreComponents 
    where 
        T: ScanResult,
    {
        // Documentation score
        let doc_raw = if file.is_docs() { 1.0 } else { 0.0 } + 
                     if let Some(doc_analysis) = file.doc_analysis() {
                         doc_analysis.structure_score()
                     } else {
                         0.0
                     };
        
        // README score
        let readme_raw = if file.is_readme() {
            // Root-level README gets higher score
            if file.depth() <= 1 { 1.5 } else { 1.0 }
        } else {
            0.0
        };
        
        // Import degree (in and out)
        let (import_degree_in, import_degree_out) = if let Some(graph) = &self.import_graph {
            graph.get_node_degrees(file.path()).unwrap_or((0, 0))
        } else {
            // Fallback: estimate from imports list
            let import_count = file.imports().map(|imports| imports.len()).unwrap_or(0);
            (0, import_count) // Can't calculate in-degree without full graph
        };
        
        // Path depth (inverted - shallow is better)
        let path_depth = file.depth();
        
        // Test links (heuristic detection)
        let test_links_found = if self.weights.features.enable_test_linking {
            self.count_test_links(file)
        } else {
            0
        };
        
        // Churn commits count
        let churn_commits = if self.weights.features.enable_churn_analysis {
            (file.churn_score() * 10.0) as usize // Convert normalized score to count
        } else {
            0
        };
        
        // Centrality (PageRank)
        let centrality_raw = if self.weights.features.enable_centrality {
            file.centrality_in()
        } else {
            0.0
        };
        
        // Entrypoint detection
        let is_entrypoint = file.is_entrypoint();
        
        // Examples count
        let examples_count = if self.weights.features.enable_examples_detection {
            self.count_examples(file)
        } else {
            0
        };
        
        RawScoreComponents {
            doc_raw,
            readme_raw,
            import_degree_in,
            import_degree_out,
            path_depth,
            test_links_found,
            churn_commits,
            centrality_raw,
            is_entrypoint,
            examples_count,
        }
    }
    
    /// Build normalization statistics from all files
    fn build_normalization_stats<T>(&self, files: &[T]) -> NormalizationStats 
    where 
        T: ScanResult,
    {
        let mut doc_max: f64 = 0.0;
        let mut import_in_max: f64 = 0.0;
        let mut import_out_max: f64 = 0.0;
        let mut path_max: f64 = 0.0;
        let mut test_links_max: f64 = 0.0;
        let mut churn_max: f64 = 0.0;
        let mut centrality_max: f64 = 0.0;
        let mut examples_max: f64 = 0.0;
        
        for file in files {
            let raw = self.calculate_raw_scores(file);
            
            doc_max = doc_max.max(raw.doc_raw);
            import_in_max = import_in_max.max(raw.import_degree_in as f64);
            import_out_max = import_out_max.max(raw.import_degree_out as f64);
            path_max = path_max.max(raw.path_depth as f64);
            test_links_max = test_links_max.max(raw.test_links_found as f64);
            churn_max = churn_max.max(raw.churn_commits as f64);
            centrality_max = centrality_max.max(raw.centrality_raw);
            examples_max = examples_max.max(raw.examples_count as f64);
        }
        
        // Ensure no division by zero
        NormalizationStats {
            doc_max: doc_max.max(1.0),
            import_in_max: import_in_max.max(1.0),
            import_out_max: import_out_max.max(1.0),
            path_max: path_max.max(1.0),
            test_links_max: test_links_max.max(1.0),
            churn_max: churn_max.max(1.0),
            centrality_max: centrality_max.max(1.0),
            examples_max: examples_max.max(1.0),
        }
    }
    
    /// Normalize raw scores to [0, 1] range
    fn normalize_scores(&self, raw: &RawScoreComponents, stats: &NormalizationStats) -> NormalizedScores {
        // Documentation score (already normalized)
        let doc_score = (raw.doc_raw / stats.doc_max).min(1.0);
        
        // README score (already normalized)
        let readme_score = raw.readme_raw.min(1.0);
        
        // Import score (combination of in-degree and out-degree)
        let import_in_norm = raw.import_degree_in as f64 / stats.import_in_max;
        let import_out_norm = raw.import_degree_out as f64 / stats.import_out_max;
        let import_score = (0.7 * import_in_norm + 0.3 * import_out_norm).min(1.0);
        
        // Path score (inverted depth)
        let path_score = if raw.path_depth == 0 {
            1.0
        } else {
            (1.0 / (raw.path_depth as f64)).min(1.0)
        };
        
        // Test link score
        let test_link_score = (raw.test_links_found as f64 / stats.test_links_max).min(1.0);
        
        // Churn score (recency-weighted)
        let churn_score = (raw.churn_commits as f64 / stats.churn_max).min(1.0);
        
        // Centrality score
        let centrality_score = if self.weights.features.enable_centrality {
            (raw.centrality_raw / stats.centrality_max).min(1.0)
        } else {
            0.0
        };
        
        // Entrypoint score
        let entrypoint_score = if raw.is_entrypoint { 1.0 } else { 0.0 };
        
        // Examples score
        let examples_score = (raw.examples_count as f64 / stats.examples_max).min(1.0);
        
        NormalizedScores {
            doc_score,
            readme_score,
            import_score,
            path_score,
            test_link_score,
            churn_score,
            centrality_score,
            entrypoint_score,
            examples_score,
        }
    }
    
    /// Calculate final weighted score
    fn calculate_final_score(&self, scores: &NormalizedScores, template_boost: f64, priority_boost: f64) -> f64 {
        let weighted_sum = 
            self.weights.doc_weight * scores.doc_score +
            self.weights.readme_weight * scores.readme_score +
            self.weights.import_weight * scores.import_score +
            self.weights.path_weight * scores.path_score +
            self.weights.test_link_weight * scores.test_link_score +
            self.weights.churn_weight * scores.churn_score +
            self.weights.centrality_weight * scores.centrality_score +
            self.weights.entrypoint_weight * scores.entrypoint_score +
            self.weights.examples_weight * scores.examples_score;
        
        weighted_sum + template_boost + priority_boost
    }
    
    /// Count test-code relationship heuristics
    fn count_test_links<T>(&self, file: &T) -> usize 
    where 
        T: ScanResult,
    {
        if file.is_test() {
            return 0; // Test files don't get test link boost
        }
        
        let path = file.path();
        let mut links = 0;
        
        // Heuristic: Look for corresponding test files
        let base_name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        
        // Common test naming patterns
        let test_patterns = [
            format!("{}_test", base_name),
            format!("test_{}", base_name),
            format!("{}.test", base_name),
            format!("{}_spec", base_name),
            format!("spec_{}", base_name),
        ];
        
        // This is a simplified heuristic - in practice, we'd check against all files
        for _pattern in &test_patterns {
            // TODO: Implement actual file system lookup
            // For now, just estimate based on naming
            if base_name.len() > 5 && !base_name.starts_with("test") {
                links += 1;
                break;
            }
        }
        
        links
    }
    
    /// Count usage examples in file
    fn count_examples<T>(&self, file: &T) -> usize 
    where 
        T: ScanResult,
    {
        if !file.has_examples() {
            return 0;
        }
        
        // Heuristic based on file name and path
        let path = file.path().to_lowercase();
        let mut count = 0;
        
        if path.contains("example") || path.contains("demo") || path.contains("sample") {
            count += 2;
        }
        
        if path.contains("tutorial") || path.contains("guide") {
            count += 1;
        }
        
        count
    }
}

/// Normalized score components
#[derive(Debug)]
struct NormalizedScores {
    pub doc_score: f64,
    pub readme_score: f64,
    pub import_score: f64,
    pub path_score: f64,
    pub test_link_score: f64,
    pub churn_score: f64,
    pub centrality_score: f64,
    pub entrypoint_score: f64,
    pub examples_score: f64,
}

impl ScoreComponents {
    /// Get total score (final_score + boosts)
    pub fn total_score(&self) -> f64 {
        self.final_score
    }
    
    /// Get component breakdown as a map
    pub fn as_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("doc_score".to_string(), self.doc_score);
        map.insert("readme_score".to_string(), self.readme_score);
        map.insert("import_score".to_string(), self.import_score);
        map.insert("path_score".to_string(), self.path_score);
        map.insert("test_link_score".to_string(), self.test_link_score);
        map.insert("churn_score".to_string(), self.churn_score);
        map.insert("centrality_score".to_string(), self.centrality_score);
        map.insert("entrypoint_score".to_string(), self.entrypoint_score);
        map.insert("examples_score".to_string(), self.examples_score);
        map.insert("priority_boost".to_string(), self.priority_boost);
        map.insert("template_boost".to_string(), self.template_boost);
        map.insert("final_score".to_string(), self.final_score);
        map
    }
    
    /// Get the dominant scoring component
    pub fn dominant_component(&self) -> (&'static str, f64) {
        let components = [
            ("doc", self.doc_score),
            ("readme", self.readme_score),
            ("import", self.import_score),
            ("path", self.path_score),
            ("test_link", self.test_link_score),
            ("churn", self.churn_score),
            ("centrality", self.centrality_score),
            ("entrypoint", self.entrypoint_score),
            ("examples", self.examples_score),
        ];
        
        components.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, score)| (*name, *score))
            .unwrap_or(("none", 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::DocumentAnalysis;
    
    // Mock scan result for testing
    #[derive(Debug)]
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
    
    impl ScanResult for MockScanResult {
        fn path(&self) -> &str { &self.path }
        fn relative_path(&self) -> &str { &self.relative_path }
        fn depth(&self) -> usize { self.depth }
        fn is_docs(&self) -> bool { self.is_docs }
        fn is_readme(&self) -> bool { self.is_readme }
        fn is_test(&self) -> bool { self.is_test }
        fn is_entrypoint(&self) -> bool { self.is_entrypoint }
        fn has_examples(&self) -> bool { self.has_examples }
        fn priority_boost(&self) -> f64 { self.priority_boost }
        fn churn_score(&self) -> f64 { self.churn_score }
        fn centrality_in(&self) -> f64 { self.centrality_in }
        fn imports(&self) -> Option<&[String]> { self.imports.as_deref() }
        fn doc_analysis(&self) -> Option<&DocumentAnalysis> { self.doc_analysis.as_ref() }
    }
    
    #[test]
    fn test_scorer_creation() {
        let weights = HeuristicWeights::default();
        let scorer = HeuristicScorer::new(weights);
        
        assert!(scorer.weights.doc_weight > 0.0);
        assert!(scorer.weights.readme_weight > 0.0);
    }
    
    #[test]
    fn test_v1_vs_v2_weights() {
        let v1 = HeuristicWeights::default();
        let v2 = HeuristicWeights::with_v2_features();
        
        // V1 should have centrality disabled
        assert_eq!(v1.centrality_weight, 0.0);
        assert!(!v1.features.enable_centrality);
        
        // V2 should have centrality enabled
        assert!(v2.centrality_weight > 0.0);
        assert!(v2.features.enable_centrality);
    }
    
    #[test]
    fn test_weight_normalization() {
        let mut weights = HeuristicWeights {
            doc_weight: 2.0,
            readme_weight: 3.0,
            import_weight: 1.0,
            path_weight: 1.0,
            test_link_weight: 1.0,
            churn_weight: 1.0,
            centrality_weight: 1.0,
            entrypoint_weight: 1.0,
            examples_weight: 1.0,
            features: ScoringFeatures::v2(),
        };
        
        weights.normalize();
        
        let total = weights.doc_weight + weights.readme_weight + weights.import_weight +
                   weights.path_weight + weights.test_link_weight + weights.churn_weight +
                   weights.centrality_weight + weights.entrypoint_weight + weights.examples_weight;
        
        assert!((total - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_file_scoring() {
        let weights = HeuristicWeights::default();
        let mut scorer = HeuristicScorer::new(weights);
        
        let files = vec![
            MockScanResult::new("README.md"),
            MockScanResult::new("src/main.rs"),
            MockScanResult::new("src/lib/deep/nested.rs"),
            MockScanResult::new("examples/demo.rs"),
            MockScanResult::new("tests/unit_test.rs"),
        ];
        
        let result = scorer.score_file(&files[0], &files);
        assert!(result.is_ok());
        
        let score = result.unwrap();
        assert!(score.final_score > 0.0);
        assert!(score.readme_score > 0.0); // README should have high readme score
    }
    
    #[test]
    fn test_score_all_files() {
        let weights = HeuristicWeights::default();
        let mut scorer = HeuristicScorer::new(weights);
        
        let files = vec![
            MockScanResult::new("README.md"),
            MockScanResult::new("src/main.rs"),
            MockScanResult::new("src/lib/utils.rs"),
        ];
        
        let result = scorer.score_all_files(&files);
        assert!(result.is_ok());
        
        let scored = result.unwrap();
        assert_eq!(scored.len(), 3);
        
        // Should be sorted by score (descending)
        if scored.len() > 1 {
            assert!(scored[0].1.final_score >= scored[1].1.final_score);
        }
    }
    
    #[test]
    fn test_score_components_map() {
        let score = ScoreComponents {
            final_score: 0.85,
            doc_score: 0.1,
            readme_score: 0.8,
            import_score: 0.3,
            path_score: 0.5,
            test_link_score: 0.2,
            churn_score: 0.4,
            centrality_score: 0.0,
            entrypoint_score: 0.0,
            examples_score: 0.0,
            priority_boost: 0.0,
            template_boost: 0.05,
            raw_scores: RawScoreComponents {
                doc_raw: 1.0,
                readme_raw: 1.0,
                import_degree_in: 3,
                import_degree_out: 5,
                path_depth: 1,
                test_links_found: 2,
                churn_commits: 10,
                centrality_raw: 0.0,
                is_entrypoint: false,
                examples_count: 0,
            },
            weights: HeuristicWeights::default(),
        };
        
        let map = score.as_map();
        assert_eq!(map["final_score"], 0.85);
        assert_eq!(map["readme_score"], 0.8);
        
        let (dominant, _) = score.dominant_component();
        assert_eq!(dominant, "readme");
    }
    
    #[test]
    fn test_scoring_features() {
        let v1_features = ScoringFeatures::v1();
        assert!(!v1_features.enable_centrality);
        assert!(v1_features.enable_template_boost);
        
        let v2_features = ScoringFeatures::v2();
        assert!(v2_features.enable_centrality);
        assert!(v2_features.enable_template_boost);
        
        let minimal_features = ScoringFeatures::minimal();
        assert!(!minimal_features.enable_centrality);
        assert!(!minimal_features.enable_template_boost);
    }
}
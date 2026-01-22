//! Core data types for the heuristic scoring system

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

    /// Applied weights
    pub weights: HeuristicWeights,
}

impl ScoreComponents {
    /// Convert score components to a map for analysis
    pub fn as_map(&self) -> std::collections::HashMap<String, f64> {
        let mut map = std::collections::HashMap::new();
        map.insert("final_score".to_string(), self.final_score);
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
        map
    }
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
    fn default() -> Self {
        Self {
            doc_weight: 1.5,
            readme_weight: 2.0,
            import_weight: 1.2,
            path_weight: -0.3,
            test_link_weight: 0.8,
            churn_weight: 0.5,
            centrality_weight: 1.0,
            entrypoint_weight: 2.5,
            examples_weight: 1.8,
            features: ScoringFeatures::default(),
        }
    }
}

impl HeuristicWeights {
    /// Create weights optimized for documentation discovery
    pub fn for_documentation() -> Self {
        Self {
            doc_weight: 3.0,
            readme_weight: 4.0,
            examples_weight: 2.5,
            ..Default::default()
        }
    }

    /// Create weights optimized for core code discovery
    pub fn for_core_code() -> Self {
        Self {
            import_weight: 2.0,
            centrality_weight: 2.5,
            entrypoint_weight: 3.0,
            path_weight: -0.5,
            ..Default::default()
        }
    }

    /// Create weights optimized for test discovery
    pub fn for_tests() -> Self {
        Self {
            test_link_weight: 3.0,
            path_weight: 0.0, // Don't penalize deep test paths
            doc_weight: 0.5,
            ..Default::default()
        }
    }

    /// Create balanced weights for general analysis
    pub fn balanced() -> Self {
        Self::default()
    }

    /// V2 weights with enhanced features enabled
    pub fn with_v2_features() -> Self {
        Self {
            doc_weight: 0.8,
            readme_weight: 1.0,
            import_weight: 0.6,
            path_weight: 0.4,
            test_link_weight: 0.3,
            churn_weight: 0.4,
            centrality_weight: 0.7, // V2 enables centrality
            entrypoint_weight: 0.9,
            examples_weight: 0.5,
            features: ScoringFeatures {
                enable_centrality: true, // Key V2 feature
                enable_template_boost: true,
                enable_doc_analysis: true,
                enable_test_linking: true,
                enable_churn_analysis: true,
                enable_examples_detection: true,
            },
        }
    }

    /// Normalize all weights to sum to 1.0 for balanced scoring
    pub fn normalized(mut self) -> Self {
        let total_weight = self.doc_weight
            + self.readme_weight
            + self.import_weight
            + self.test_link_weight
            + self.churn_weight
            + self.centrality_weight
            + self.entrypoint_weight
            + self.examples_weight;

        if total_weight > 0.0 {
            self.doc_weight /= total_weight;
            self.readme_weight /= total_weight;
            self.import_weight /= total_weight;
            self.test_link_weight /= total_weight;
            self.churn_weight /= total_weight;
            self.centrality_weight /= total_weight;
            self.entrypoint_weight /= total_weight;
            self.examples_weight /= total_weight;
        }

        self
    }
}

impl Default for ScoringFeatures {
    fn default() -> Self {
        Self {
            enable_centrality: true,
            enable_template_boost: true,
            enable_doc_analysis: true,
            enable_test_linking: true,
            enable_churn_analysis: true,
            enable_examples_detection: true,
        }
    }
}

impl ScoringFeatures {
    /// Enable all features (maximum analysis depth)
    pub fn all_enabled() -> Self {
        Self {
            enable_centrality: true,
            enable_template_boost: true,
            enable_doc_analysis: true,
            enable_test_linking: true,
            enable_churn_analysis: true,
            enable_examples_detection: true,
        }
    }

    /// Minimal feature set for fast analysis
    pub fn minimal() -> Self {
        Self {
            enable_centrality: false,
            enable_template_boost: false,
            enable_doc_analysis: true,
            enable_test_linking: false,
            enable_churn_analysis: false,
            enable_examples_detection: true,
        }
    }

    /// Documentation-focused feature set
    pub fn documentation_focused() -> Self {
        Self {
            enable_centrality: false,
            enable_template_boost: true,
            enable_doc_analysis: true,
            enable_test_linking: false,
            enable_churn_analysis: false,
            enable_examples_detection: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weights() {
        let weights = HeuristicWeights::default();
        assert!(weights.doc_weight > 0.0);
        assert!(weights.readme_weight > 0.0);
        assert!(weights.entrypoint_weight > 0.0);
    }

    #[test]
    fn test_documentation_weights() {
        let weights = HeuristicWeights::for_documentation();
        let default = HeuristicWeights::default();
        assert!(weights.doc_weight > default.doc_weight);
        assert!(weights.readme_weight > default.readme_weight);
    }

    #[test]
    fn test_core_code_weights() {
        let weights = HeuristicWeights::for_core_code();
        let default = HeuristicWeights::default();
        assert!(weights.centrality_weight > default.centrality_weight);
        assert!(weights.entrypoint_weight > default.entrypoint_weight);
    }

    #[test]
    fn test_test_weights() {
        let weights = HeuristicWeights::for_tests();
        let default = HeuristicWeights::default();
        assert!(weights.test_link_weight > default.test_link_weight);
    }

    #[test]
    fn test_normalized_weights() {
        let weights = HeuristicWeights::default().normalized();
        let total = weights.doc_weight
            + weights.readme_weight
            + weights.import_weight
            + weights.test_link_weight
            + weights.churn_weight
            + weights.centrality_weight
            + weights.entrypoint_weight
            + weights.examples_weight;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_v2_features() {
        let weights = HeuristicWeights::with_v2_features();
        assert!(weights.features.enable_centrality);
        assert!(weights.features.enable_template_boost);
    }

    #[test]
    fn test_scoring_features_all_enabled() {
        let features = ScoringFeatures::all_enabled();
        assert!(features.enable_centrality);
        assert!(features.enable_template_boost);
        assert!(features.enable_doc_analysis);
        assert!(features.enable_test_linking);
        assert!(features.enable_churn_analysis);
        assert!(features.enable_examples_detection);
    }

    #[test]
    fn test_scoring_features_minimal() {
        let features = ScoringFeatures::minimal();
        assert!(!features.enable_centrality);
        assert!(!features.enable_template_boost);
        assert!(features.enable_doc_analysis);
        assert!(!features.enable_test_linking);
    }

    #[test]
    fn test_score_components_as_map() {
        let weights = HeuristicWeights::default();
        let components = ScoreComponents {
            final_score: 1.5,
            doc_score: 0.5,
            readme_score: 1.0,
            import_score: 0.3,
            path_score: -0.2,
            test_link_score: 0.0,
            churn_score: 0.1,
            centrality_score: 0.4,
            entrypoint_score: 0.0,
            examples_score: 0.0,
            priority_boost: 0.0,
            template_boost: 0.0,
            weights,
        };

        let map = components.as_map();
        assert_eq!(map.get("final_score"), Some(&1.5));
        assert_eq!(map.get("doc_score"), Some(&0.5));
        assert_eq!(map.get("readme_score"), Some(&1.0));
    }
}

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

    /// Raw component scores before normalization
    pub raw_scores: RawScoreComponents,

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

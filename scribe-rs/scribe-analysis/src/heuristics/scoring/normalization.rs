//! Score normalization logic for consistent heuristic scoring

use super::super::ScanResult;
use super::types::ScoringFeatures;

/// Normalized score components after statistical normalization
#[derive(Debug, Clone)]
pub struct NormalizedScores {
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

/// Statistics for normalization across all files
#[derive(Debug, Clone)]
pub struct NormalizationStats {
    pub max_doc_raw: f64,
    pub max_readme_raw: f64,
    pub max_import_degree_in: usize,
    pub max_import_degree_out: usize,
    pub max_path_depth: usize,
    pub max_test_links: usize,
    pub max_churn_commits: usize,
    pub max_centrality_raw: f64,
    pub max_examples_count: usize,
}

/// Build normalization statistics from all files
pub fn build_normalization_stats<T>(files: &[T]) -> NormalizationStats
where
    T: ScanResult,
{
    let mut stats = NormalizationStats {
        max_doc_raw: 0.0,
        max_readme_raw: 0.0,
        max_import_degree_in: 0,
        max_import_degree_out: 0,
        max_path_depth: 0,
        max_test_links: 0,
        max_churn_commits: 0,
        max_centrality_raw: 0.0,
        max_examples_count: 0,
    };

    for file in files {
        // Documentation stats
        if file.is_docs() {
            stats.max_doc_raw = stats.max_doc_raw.max(1.0);
            if let Some(doc_analysis) = file.doc_analysis() {
                stats.max_doc_raw = stats.max_doc_raw.max(doc_analysis.structure_score());
            }
        }

        // README stats
        if file.is_readme() {
            let readme_score = if file.depth() <= 1 { 1.5 } else { 1.0 };
            stats.max_readme_raw = stats.max_readme_raw.max(readme_score);
        }

        // Path depth
        stats.max_path_depth = stats.max_path_depth.max(file.depth());

        // Test links (use is_test as proxy)
        if file.is_test() {
            stats.max_test_links = stats.max_test_links.max(1);
        }

        // Git churn (use churn_score from trait)
        let churn_score = file.churn_score() as usize;
        stats.max_churn_commits = stats.max_churn_commits.max(churn_score);

        if let Some(imports) = file.imports() {
            stats.max_import_degree_out = stats.max_import_degree_out.max(imports.len());
        }

        let centrality_raw = file.centrality_in();
        stats.max_import_degree_in = stats
            .max_import_degree_in
            .max(centrality_raw.round() as usize);
        stats.max_centrality_raw = stats.max_centrality_raw.max(centrality_raw);

        // Count examples
        let examples_count = if file.has_examples() { 1 } else { 0 };
        stats.max_examples_count = stats.max_examples_count.max(examples_count);
    }

    // Ensure minimums to avoid division by zero
    stats.max_doc_raw = stats.max_doc_raw.max(1.0);
    stats.max_readme_raw = stats.max_readme_raw.max(1.0);
    stats.max_import_degree_in = stats.max_import_degree_in.max(1);
    stats.max_import_degree_out = stats.max_import_degree_out.max(1);
    stats.max_path_depth = stats.max_path_depth.max(1);
    stats.max_test_links = stats.max_test_links.max(1);
    stats.max_churn_commits = stats.max_churn_commits.max(1);
    stats.max_centrality_raw = stats.max_centrality_raw.max(0.1);
    stats.max_examples_count = stats.max_examples_count.max(1);

    stats
}

/// Normalize raw scores using statistics
pub fn normalize_scores<T: ScanResult>(
    file: &T,
    stats: &NormalizationStats,
    features: &ScoringFeatures,
) -> NormalizedScores {
    let mut doc_raw = if file.is_docs() { 1.0 } else { 0.0 };
    if features.enable_doc_analysis {
        if let Some(doc_analysis) = file.doc_analysis() {
            doc_raw += doc_analysis.structure_score();
        }
    }

    let readme_raw = if file.is_readme() {
        if file.depth() <= 1 {
            1.5
        } else {
            1.0
        }
    } else {
        0.0
    };

    let import_out = file.imports().map(|imports| imports.len()).unwrap_or(0);
    let import_in = file.centrality_in().round().max(0.0) as usize;

    let path_depth = file.depth();

    let test_links_found = if features.enable_test_linking && file.is_test() {
        1
    } else {
        0
    };

    let churn_commits = if features.enable_churn_analysis {
        file.churn_score().round().max(0.0) as usize
    } else {
        0
    };

    let centrality_raw = if features.enable_centrality {
        file.centrality_in()
    } else {
        0.0
    };

    let examples_count = if features.enable_examples_detection && file.has_examples() {
        1
    } else {
        0
    };

    NormalizedScores {
        doc_score: doc_raw / stats.max_doc_raw,
        readme_score: readme_raw / stats.max_readme_raw,
        import_score: calculate_import_score(import_in, import_out, stats),
        path_score: calculate_path_score(path_depth, stats),
        test_link_score: test_links_found as f64 / stats.max_test_links as f64,
        churn_score: churn_commits as f64 / stats.max_churn_commits as f64,
        centrality_score: centrality_raw / stats.max_centrality_raw,
        entrypoint_score: if file.is_entrypoint() { 1.0 } else { 0.0 },
        examples_score: examples_count as f64 / stats.max_examples_count as f64,
    }
}

/// Calculate normalized import score combining in and out degree
fn calculate_import_score(import_in: usize, import_out: usize, stats: &NormalizationStats) -> f64 {
    let in_score = import_in as f64 / stats.max_import_degree_in as f64;
    let out_score = import_out as f64 / stats.max_import_degree_out as f64;

    // Weight incoming imports higher (more important files are imported more)
    0.7 * in_score + 0.3 * out_score
}

/// Calculate normalized path score (inverted - deeper paths get lower scores)
fn calculate_path_score(path_depth: usize, stats: &NormalizationStats) -> f64 {
    // Invert path depth (deeper = lower score)
    1.0 - (path_depth as f64 / stats.max_path_depth as f64)
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
        content: Option<String>,
    }

    impl ScanResult for MockFile {
        fn path(&self) -> &str {
            &self.path
        }
        fn relative_path(&self) -> &str {
            &self.path
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
            false
        }
        fn is_entrypoint(&self) -> bool {
            false
        }
        fn has_examples(&self) -> bool {
            false
        }
        fn priority_boost(&self) -> f64 {
            0.0
        }
        fn churn_score(&self) -> f64 {
            0.0
        }
        fn centrality_in(&self) -> f64 {
            0.0
        }
        fn imports(&self) -> Option<&[String]> {
            None
        }
        fn doc_analysis(&self) -> Option<&crate::heuristics::DocumentAnalysis> {
            None
        }
    }

    #[test]
    fn test_normalization_stats() {
        let files = vec![
            MockFile {
                path: "README.md".to_string(),
                is_docs: false,
                is_readme: true,
                depth: 1,
                content: None,
            },
            MockFile {
                path: "docs/guide.md".to_string(),
                is_docs: true,
                is_readme: false,
                depth: 2,
                content: None,
            },
        ];

        let stats = build_normalization_stats(&files);
        assert!(stats.max_readme_raw > 0.0);
        assert!(stats.max_doc_raw > 0.0);
        assert_eq!(stats.max_path_depth, 2);
    }

    #[test]
    fn test_path_score_inversion() {
        let stats = NormalizationStats {
            max_doc_raw: 1.0,
            max_readme_raw: 1.0,
            max_import_degree_in: 1,
            max_import_degree_out: 1,
            max_path_depth: 5, // Max depth is 5
            max_test_links: 1,
            max_churn_commits: 1,
            max_centrality_raw: 1.0,
            max_examples_count: 1,
        };

        let path_score = calculate_path_score(3, &stats);
        // Path depth 3/5 = 0.6, so inverted score should be 1.0 - 0.6 = 0.4
        assert!((path_score - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_import_score_calculation() {
        let stats = NormalizationStats {
            max_doc_raw: 1.0,
            max_readme_raw: 1.0,
            max_import_degree_in: 10,
            max_import_degree_out: 10,
            max_path_depth: 5,
            max_test_links: 1,
            max_churn_commits: 1,
            max_centrality_raw: 1.0,
            max_examples_count: 1,
        };

        // Test with 5 imports in, 5 out
        let import_score = calculate_import_score(5, 5, &stats);
        // 0.7 * 0.5 + 0.3 * 0.5 = 0.35 + 0.15 = 0.5
        assert!((import_score - 0.5).abs() < 0.01);

        // Test with 10 imports in, 0 out (highly imported file)
        let import_score_in = calculate_import_score(10, 0, &stats);
        // 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert!((import_score_in - 0.7).abs() < 0.01);

        // Test with 0 imports in, 10 out (file that imports many others)
        let import_score_out = calculate_import_score(0, 10, &stats);
        // 0.7 * 0.0 + 0.3 * 1.0 = 0.3
        assert!((import_score_out - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_normalization_stats_empty_files() {
        let files: Vec<MockFile> = vec![];
        let stats = build_normalization_stats(&files);

        // Should have minimum values to avoid division by zero
        assert!(stats.max_doc_raw >= 1.0);
        assert!(stats.max_readme_raw >= 1.0);
        assert!(stats.max_path_depth >= 1);
        assert!(stats.max_import_degree_in >= 1);
        assert!(stats.max_import_degree_out >= 1);
    }

    #[test]
    fn test_normalized_scores_struct_clone() {
        let scores = NormalizedScores {
            doc_score: 0.5,
            readme_score: 0.8,
            import_score: 0.3,
            path_score: 0.7,
            test_link_score: 0.2,
            churn_score: 0.4,
            centrality_score: 0.6,
            entrypoint_score: 1.0,
            examples_score: 0.5,
        };
        let cloned = scores.clone();
        assert_eq!(scores.doc_score, cloned.doc_score);
        assert_eq!(scores.readme_score, cloned.readme_score);
    }

    #[test]
    fn test_normalized_scores_debug() {
        let scores = NormalizedScores {
            doc_score: 0.5,
            readme_score: 0.8,
            import_score: 0.3,
            path_score: 0.7,
            test_link_score: 0.2,
            churn_score: 0.4,
            centrality_score: 0.6,
            entrypoint_score: 1.0,
            examples_score: 0.5,
        };
        let debug_str = format!("{:?}", scores);
        assert!(debug_str.contains("NormalizedScores"));
    }

    #[test]
    fn test_normalization_stats_clone() {
        let stats = NormalizationStats {
            max_doc_raw: 1.0,
            max_readme_raw: 1.5,
            max_import_degree_in: 10,
            max_import_degree_out: 5,
            max_path_depth: 4,
            max_test_links: 2,
            max_churn_commits: 3,
            max_centrality_raw: 0.8,
            max_examples_count: 1,
        };
        let cloned = stats.clone();
        assert_eq!(stats.max_doc_raw, cloned.max_doc_raw);
        assert_eq!(stats.max_path_depth, cloned.max_path_depth);
    }

    #[test]
    fn test_normalization_stats_debug() {
        let stats = NormalizationStats {
            max_doc_raw: 1.0,
            max_readme_raw: 1.5,
            max_import_degree_in: 10,
            max_import_degree_out: 5,
            max_path_depth: 4,
            max_test_links: 2,
            max_churn_commits: 3,
            max_centrality_raw: 0.8,
            max_examples_count: 1,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("NormalizationStats"));
    }

    #[test]
    fn test_path_score_edge_cases() {
        let stats = NormalizationStats {
            max_doc_raw: 1.0,
            max_readme_raw: 1.0,
            max_import_degree_in: 1,
            max_import_degree_out: 1,
            max_path_depth: 10,
            max_test_links: 1,
            max_churn_commits: 1,
            max_centrality_raw: 1.0,
            max_examples_count: 1,
        };

        // Depth 0 should give score 1.0
        let path_score_0 = calculate_path_score(0, &stats);
        assert!((path_score_0 - 1.0).abs() < 0.01);

        // Max depth should give score 0.0
        let path_score_max = calculate_path_score(10, &stats);
        assert!((path_score_max - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_scores_basic() {
        let stats = NormalizationStats {
            max_doc_raw: 2.0,
            max_readme_raw: 1.5,
            max_import_degree_in: 10,
            max_import_degree_out: 10,
            max_path_depth: 5,
            max_test_links: 3,
            max_churn_commits: 10,
            max_centrality_raw: 1.0,
            max_examples_count: 2,
        };

        let features = ScoringFeatures::default();

        let file = MockFile {
            path: "README.md".to_string(),
            is_docs: false,
            is_readme: true,
            depth: 1,
            content: None,
        };

        let scores = normalize_scores(&file, &stats, &features);

        // README should get a readme score
        assert!(scores.readme_score > 0.0);

        // Path score should be high (depth 1 out of 5)
        assert!(scores.path_score > 0.5);
    }

    #[test]
    fn test_normalize_scores_docs_file() {
        let stats = NormalizationStats {
            max_doc_raw: 2.0,
            max_readme_raw: 1.5,
            max_import_degree_in: 10,
            max_import_degree_out: 10,
            max_path_depth: 5,
            max_test_links: 3,
            max_churn_commits: 10,
            max_centrality_raw: 1.0,
            max_examples_count: 2,
        };

        let features = ScoringFeatures::default();

        let file = MockFile {
            path: "docs/guide.md".to_string(),
            is_docs: true,
            is_readme: false,
            depth: 2,
            content: None,
        };

        let scores = normalize_scores(&file, &stats, &features);

        // Docs file should get a doc score
        assert!(scores.doc_score > 0.0);

        // Not a README
        assert_eq!(scores.readme_score, 0.0);
    }

    // Mock with more features
    struct MockFileWithFeatures {
        path: String,
        is_docs: bool,
        is_readme: bool,
        is_test: bool,
        is_entrypoint: bool,
        has_examples: bool,
        depth: usize,
        churn_score: f64,
        centrality_in: f64,
        imports: Vec<String>,
    }

    impl ScanResult for MockFileWithFeatures {
        fn path(&self) -> &str {
            &self.path
        }
        fn relative_path(&self) -> &str {
            &self.path
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
            0.0
        }
        fn churn_score(&self) -> f64 {
            self.churn_score
        }
        fn centrality_in(&self) -> f64 {
            self.centrality_in
        }
        fn imports(&self) -> Option<&[String]> {
            if self.imports.is_empty() {
                None
            } else {
                Some(&self.imports)
            }
        }
        fn doc_analysis(&self) -> Option<&crate::heuristics::DocumentAnalysis> {
            None
        }
    }

    #[test]
    fn test_build_normalization_stats_with_all_features() {
        let files = vec![
            MockFileWithFeatures {
                path: "README.md".to_string(),
                is_docs: false,
                is_readme: true,
                is_test: false,
                is_entrypoint: false,
                has_examples: false,
                depth: 0,
                churn_score: 5.0,
                centrality_in: 10.0,
                imports: vec!["import1".to_string(), "import2".to_string()],
            },
            MockFileWithFeatures {
                path: "tests/test.rs".to_string(),
                is_docs: false,
                is_readme: false,
                is_test: true,
                is_entrypoint: false,
                has_examples: false,
                depth: 1,
                churn_score: 3.0,
                centrality_in: 5.0,
                imports: vec!["import1".to_string()],
            },
            MockFileWithFeatures {
                path: "examples/demo.rs".to_string(),
                is_docs: false,
                is_readme: false,
                is_test: false,
                is_entrypoint: false,
                has_examples: true,
                depth: 1,
                churn_score: 1.0,
                centrality_in: 2.0,
                imports: vec![],
            },
        ];

        let stats = build_normalization_stats(&files);

        // Check stats are properly computed
        assert!(stats.max_churn_commits >= 5);
        assert!(stats.max_centrality_raw >= 10.0);
        assert_eq!(stats.max_import_degree_in, 10);
        assert!(stats.max_import_degree_out >= 2);
        assert!(stats.max_test_links >= 1);
        assert!(stats.max_examples_count >= 1);
    }

    #[test]
    fn test_normalize_scores_with_all_features_enabled() {
        let stats = NormalizationStats {
            max_doc_raw: 2.0,
            max_readme_raw: 1.5,
            max_import_degree_in: 10,
            max_import_degree_out: 10,
            max_path_depth: 5,
            max_test_links: 3,
            max_churn_commits: 10,
            max_centrality_raw: 1.0,
            max_examples_count: 2,
        };

        let features = ScoringFeatures {
            enable_doc_analysis: true,
            enable_test_linking: true,
            enable_churn_analysis: true,
            enable_centrality: true,
            enable_examples_detection: true,
            enable_template_boost: true,
        };

        let file = MockFileWithFeatures {
            path: "src/main.rs".to_string(),
            is_docs: false,
            is_readme: false,
            is_test: true,
            is_entrypoint: true,
            has_examples: true,
            depth: 1,
            churn_score: 5.0,
            centrality_in: 0.5,
            imports: vec!["import1".to_string()],
        };

        let scores = normalize_scores(&file, &stats, &features);

        // Test that all scores are computed when features are enabled
        assert!(scores.test_link_score > 0.0);
        assert!(scores.churn_score > 0.0);
        assert!(scores.centrality_score > 0.0);
        assert!(scores.entrypoint_score > 0.0);
        assert!(scores.examples_score > 0.0);
    }

    #[test]
    fn test_normalize_scores_with_features_disabled() {
        let stats = NormalizationStats {
            max_doc_raw: 2.0,
            max_readme_raw: 1.5,
            max_import_degree_in: 10,
            max_import_degree_out: 10,
            max_path_depth: 5,
            max_test_links: 3,
            max_churn_commits: 10,
            max_centrality_raw: 1.0,
            max_examples_count: 2,
        };

        let features = ScoringFeatures {
            enable_doc_analysis: false,
            enable_test_linking: false,
            enable_churn_analysis: false,
            enable_centrality: false,
            enable_examples_detection: false,
            enable_template_boost: false,
        };

        let file = MockFileWithFeatures {
            path: "src/main.rs".to_string(),
            is_docs: false,
            is_readme: false,
            is_test: true,
            is_entrypoint: true,
            has_examples: true,
            depth: 1,
            churn_score: 5.0,
            centrality_in: 0.5,
            imports: vec!["import1".to_string()],
        };

        let scores = normalize_scores(&file, &stats, &features);

        // When features are disabled, these should be 0
        assert_eq!(scores.test_link_score, 0.0);
        assert_eq!(scores.churn_score, 0.0);
        assert_eq!(scores.centrality_score, 0.0);
        assert_eq!(scores.examples_score, 0.0);
    }

    #[test]
    fn test_normalize_scores_readme_depth_greater_than_one() {
        let stats = NormalizationStats {
            max_doc_raw: 2.0,
            max_readme_raw: 1.5,
            max_import_degree_in: 10,
            max_import_degree_out: 10,
            max_path_depth: 5,
            max_test_links: 3,
            max_churn_commits: 10,
            max_centrality_raw: 1.0,
            max_examples_count: 2,
        };

        let features = ScoringFeatures::default();

        let file = MockFileWithFeatures {
            path: "docs/subdir/README.md".to_string(),
            is_docs: false,
            is_readme: true,
            is_test: false,
            is_entrypoint: false,
            has_examples: false,
            depth: 2,
            churn_score: 0.0,
            centrality_in: 0.0,
            imports: vec![],
        };

        let scores = normalize_scores(&file, &stats, &features);

        // README at depth > 1 gets score of 1.0 (not 1.5)
        // Normalized: 1.0 / 1.5 = 0.667
        assert!(scores.readme_score > 0.0);
        assert!(scores.readme_score < 1.0);
    }
}

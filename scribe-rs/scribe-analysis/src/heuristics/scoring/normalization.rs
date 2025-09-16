//! Score normalization logic for consistent heuristic scoring

use super::super::ScanResult;
use super::types::{HeuristicWeights, RawScoreComponents};

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

        // Count examples
        let examples_count = count_examples_in_file(file);
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
pub fn normalize_scores(
    raw_scores: &RawScoreComponents,
    stats: &NormalizationStats,
) -> NormalizedScores {
    NormalizedScores {
        doc_score: raw_scores.doc_raw / stats.max_doc_raw,
        readme_score: raw_scores.readme_raw / stats.max_readme_raw,
        import_score: calculate_import_score(raw_scores, stats),
        path_score: calculate_path_score(raw_scores, stats),
        test_link_score: raw_scores.test_links_found as f64 / stats.max_test_links as f64,
        churn_score: raw_scores.churn_commits as f64 / stats.max_churn_commits as f64,
        centrality_score: raw_scores.centrality_raw / stats.max_centrality_raw,
        entrypoint_score: if raw_scores.is_entrypoint { 1.0 } else { 0.0 },
        examples_score: raw_scores.examples_count as f64 / stats.max_examples_count as f64,
    }
}

/// Calculate normalized import score combining in and out degree
fn calculate_import_score(raw_scores: &RawScoreComponents, stats: &NormalizationStats) -> f64 {
    let in_score = raw_scores.import_degree_in as f64 / stats.max_import_degree_in as f64;
    let out_score = raw_scores.import_degree_out as f64 / stats.max_import_degree_out as f64;

    // Weight incoming imports higher (more important files are imported more)
    0.7 * in_score + 0.3 * out_score
}

/// Calculate normalized path score (inverted - deeper paths get lower scores)
fn calculate_path_score(raw_scores: &RawScoreComponents, stats: &NormalizationStats) -> f64 {
    // Invert path depth (deeper = lower score)
    1.0 - (raw_scores.path_depth as f64 / stats.max_path_depth as f64)
}

/// Count examples or example-like content in a file
fn count_examples_in_file<T: ScanResult>(file: &T) -> usize {
    // Use the built-in method from ScanResult trait
    if file.has_examples() {
        1
    } else {
        0
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
        let raw_scores = RawScoreComponents {
            doc_raw: 0.0,
            readme_raw: 0.0,
            import_degree_in: 0,
            import_degree_out: 0,
            path_depth: 3, // Deep path
            test_links_found: 0,
            churn_commits: 0,
            centrality_raw: 0.0,
            is_entrypoint: false,
            examples_count: 0,
        };

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

        let path_score = calculate_path_score(&raw_scores, &stats);
        // Path depth 3/5 = 0.6, so inverted score should be 1.0 - 0.6 = 0.4
        assert!((path_score - 0.4).abs() < 0.01);
    }
}

//! # Centrality Calculator with Heuristics Integration
//!
//! Main interface for PageRank centrality calculation and integration with the
//! heuristic scoring system used by Scribe.

use rayon::prelude::*;
use scribe_analysis::heuristics::ScanResult;
use scribe_core::{file, Result};
use std::collections::HashMap;
use std::path::Path;

use crate::graph::DependencyGraph;
use crate::pagerank::{PageRankComputer, PageRankConfig};
use crate::statistics::{GraphAnalysisResults, GraphStatisticsAnalyzer};

// Re-export types from centrality_types for API compatibility
pub use crate::centrality_types::{
    CentralityConfig, CentralityResults, ImportDetectionStats, ImportPatternStats,
    ImportResolutionConfig, IntegrationConfig, IntegrationMetadata, NormalizationMethod,
};

// Re-export ImportDetector for external use
pub use crate::import_detector::ImportDetector;

/// Main centrality calculator with heuristics integration
#[derive(Debug)]
pub struct CentralityCalculator {
    /// Configuration
    config: CentralityConfig,

    /// PageRank computer
    pagerank_computer: PageRankComputer,

    /// Graph statistics analyzer
    stats_analyzer: GraphStatisticsAnalyzer,

    /// Import detector
    import_detector: ImportDetector,
}

impl CentralityCalculator {
    /// Create a new centrality calculator with default configuration
    pub fn new() -> Result<Self> {
        let config = CentralityConfig::default();
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: CentralityConfig) -> Result<Self> {
        let pagerank_computer = PageRankComputer::with_config(config.pagerank_config.clone())?;

        let stats_analyzer = if config.analyze_graph_structure {
            GraphStatisticsAnalyzer::new()
        } else {
            GraphStatisticsAnalyzer::for_large_graphs()
        };

        let import_detector = ImportDetector::with_config(config.import_resolution.clone());

        Ok(Self {
            config,
            pagerank_computer,
            stats_analyzer,
            import_detector,
        })
    }

    /// Create optimized for large codebases
    pub fn for_large_codebases() -> Result<Self> {
        let config = CentralityConfig {
            pagerank_config: PageRankConfig::for_large_codebases(),
            analyze_graph_structure: false,
            ..CentralityConfig::default()
        };
        Self::with_config(config)
    }

    /// Calculate centrality scores for a collection of scan results
    pub fn calculate_centrality<T>(&self, scan_results: &[T]) -> Result<CentralityResults>
    where
        T: ScanResult + Sync,
    {
        let start_time = std::time::Instant::now();

        // Build dependency graph from scan results
        let (graph, import_stats) = self.build_dependency_graph(scan_results)?;

        // Compute PageRank centrality
        let pagerank_results = self.pagerank_computer.compute(&graph)?;

        // Perform graph analysis if enabled
        let graph_analysis = if self.config.analyze_graph_structure {
            self.stats_analyzer.analyze(&graph)?
        } else {
            self.create_minimal_analysis(&graph)?
        };

        // Create integration metadata
        let computation_time = start_time.elapsed().as_millis() as u64;
        let integration_metadata = IntegrationMetadata {
            timestamp: chrono::Utc::now(),
            computation_time_ms: computation_time,
            integration_successful: true,
            centrality_weight: self.config.integration.centrality_weight,
            files_with_centrality: pagerank_results.scores.len(),
            config: self.config.clone(),
        };

        Ok(CentralityResults {
            pagerank_scores: pagerank_results.scores.clone(),
            graph_analysis,
            pagerank_details: pagerank_results,
            import_stats,
            integration_metadata,
        })
    }

    /// Build a dependency graph from scan results without computing PageRank.
    pub fn build_graph_only<T>(&self, scan_results: &[T]) -> Result<DependencyGraph>
    where
        T: ScanResult + Sync,
    {
        let (graph, _stats) = self.build_dependency_graph(scan_results)?;
        Ok(graph)
    }

    /// Apply entrypoint boost to centrality score if configured
    fn apply_entrypoint_boost(&self, centrality_score: f64, file_path: &str) -> f64 {
        if self.config.integration.boost_entrypoints && self.is_entrypoint_file(file_path) {
            centrality_score * self.config.integration.entrypoint_boost_factor
        } else {
            centrality_score
        }
    }

    /// Combine a single file's heuristic and centrality scores
    fn combine_scores(&self, heuristic_score: f64, centrality_score: f64) -> f64 {
        let centrality_weight = self.config.integration.centrality_weight;
        let heuristic_weight = 1.0 - centrality_weight;
        heuristic_weight * heuristic_score + centrality_weight * centrality_score
    }

    /// Add centrality-only files (not in heuristic scores)
    fn add_centrality_only_files(
        &self,
        integrated_scores: &mut HashMap<String, f64>,
        normalized_centrality: &HashMap<String, f64>,
    ) {
        let centrality_weight = self.config.integration.centrality_weight;

        for (file_path, &centrality_score) in normalized_centrality {
            if !integrated_scores.contains_key(file_path) {
                let boosted = self.apply_entrypoint_boost(centrality_score, file_path);
                integrated_scores.insert(file_path.clone(), centrality_weight * boosted);
            }
        }
    }

    /// Integrate centrality scores with existing heuristic scores
    pub fn integrate_with_heuristics(
        &self,
        centrality_results: &CentralityResults,
        heuristic_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let normalized_centrality = self
            .normalize_centrality_scores(&centrality_results.pagerank_scores, heuristic_scores)?;

        let mut integrated_scores = HashMap::new();

        // Combine heuristic and centrality scores
        for (file_path, heuristic_score) in heuristic_scores {
            let centrality_score = normalized_centrality.get(file_path).copied().unwrap_or(0.0);
            let boosted = self.apply_entrypoint_boost(centrality_score, file_path);
            let integrated_score = self.combine_scores(*heuristic_score, boosted);
            integrated_scores.insert(file_path.clone(), integrated_score);
        }

        self.add_centrality_only_files(&mut integrated_scores, &normalized_centrality);

        Ok(integrated_scores)
    }

    /// Build dependency graph from scan results
    fn build_dependency_graph<T>(
        &self,
        scan_results: &[T],
    ) -> Result<(DependencyGraph, ImportDetectionStats)>
    where
        T: ScanResult + Sync,
    {
        let mut graph = DependencyGraph::with_capacity(scan_results.len());

        // Create optimized import detector with pre-computed lookup maps
        let optimized_detector =
            ImportDetector::with_file_index(self.import_detector.config.clone(), scan_results);

        // Add all files as nodes first
        for result in scan_results {
            graph.add_node(result.path().to_string())?;
        }

        // Detect imports and build edges using optimized detector
        let import_stats = if self.config.pagerank_config.use_parallel {
            self.build_edges_parallel(&mut graph, scan_results, &optimized_detector)?
        } else {
            self.build_edges_sequential(&mut graph, scan_results, &optimized_detector)?
        };

        Ok((graph, import_stats))
    }

    /// Build graph edges sequentially
    fn build_edges_sequential<T>(
        &self,
        graph: &mut DependencyGraph,
        scan_results: &[T],
        optimized_detector: &ImportDetector,
    ) -> Result<ImportDetectionStats>
    where
        T: ScanResult,
    {
        let mut stats = ImportDetectionStats {
            files_processed: 0,
            imports_detected: 0,
            imports_resolved: 0,
            resolution_rate: 0.0,
            language_breakdown: HashMap::new(),
            import_patterns: HashMap::new(),
        };

        // Create file path lookup for resolution
        let file_path_map: HashMap<&str, &T> = scan_results
            .iter()
            .map(|result| (result.path(), result))
            .collect();

        for result in scan_results {
            stats.files_processed += 1;

            // Track language
            if let Some(lang) = optimized_detector.detect_language(result.path()) {
                *stats.language_breakdown.entry(lang.clone()).or_insert(0) += 1;
            }

            // Extract and resolve imports using optimized detector
            if let Some(imports) = result.imports() {
                stats.imports_detected += imports.len();

                for import_str in imports {
                    if let Some(resolved_path) =
                        optimized_detector.resolve_import(import_str, result.path(), &file_path_map)
                    {
                        graph.add_edge(result.path().to_string(), resolved_path)?;
                        stats.imports_resolved += 1;
                    }
                }
            }
        }

        stats.resolution_rate = if stats.imports_detected > 0 {
            stats.imports_resolved as f64 / stats.imports_detected as f64
        } else {
            0.0
        };

        Ok(stats)
    }

    /// Build graph edges in parallel
    fn build_edges_parallel<T>(
        &self,
        graph: &mut DependencyGraph,
        scan_results: &[T],
        optimized_detector: &ImportDetector,
    ) -> Result<ImportDetectionStats>
    where
        T: ScanResult + Sync,
    {
        // Create file path lookup
        let file_path_map: HashMap<&str, &T> = scan_results
            .iter()
            .map(|result| (result.path(), result))
            .collect();

        // Process imports in parallel using optimized detector
        let import_edges: Vec<_> = scan_results
            .par_iter()
            .flat_map(|result| {
                let mut edges = Vec::new();

                if let Some(imports) = result.imports() {
                    for import_str in imports {
                        if let Some(resolved_path) = optimized_detector.resolve_import(
                            import_str,
                            result.path(),
                            &file_path_map,
                        ) {
                            edges.push((result.path().to_string(), resolved_path));
                        }
                    }
                }

                edges
            })
            .collect();

        // Add edges to graph
        for (from, to) in &import_edges {
            graph.add_edge(from.clone(), to.clone())?;
        }

        // Calculate statistics
        let total_imports: usize = scan_results
            .iter()
            .map(|result| result.imports().map_or(0, |imports| imports.len()))
            .sum();

        let language_breakdown: HashMap<String, usize> = scan_results
            .iter()
            .filter_map(|result| {
                optimized_detector
                    .detect_language(result.path())
                    .map(|lang| (lang, 1))
            })
            .fold(HashMap::new(), |mut acc, (lang, count)| {
                *acc.entry(lang).or_insert(0) += count;
                acc
            });

        let stats = ImportDetectionStats {
            files_processed: scan_results.len(),
            imports_detected: total_imports,
            imports_resolved: import_edges.len(),
            resolution_rate: if total_imports > 0 {
                import_edges.len() as f64 / total_imports as f64
            } else {
                0.0
            },
            language_breakdown,
            import_patterns: HashMap::new(),
        };

        Ok(stats)
    }

    /// Normalize centrality scores for integration with heuristics
    fn normalize_centrality_scores(
        &self,
        centrality_scores: &HashMap<String, f64>,
        heuristic_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        if centrality_scores.is_empty() {
            return Ok(HashMap::new());
        }

        match self.config.integration.normalization_method {
            NormalizationMethod::MinMax => {
                self.normalize_min_max(centrality_scores, heuristic_scores)
            }
            NormalizationMethod::ZScore => self.normalize_z_score(centrality_scores),
            NormalizationMethod::Rank => self.normalize_rank(centrality_scores),
            NormalizationMethod::None => Ok(centrality_scores.clone()),
        }
    }

    /// Compute min/max values from a slice of f64 values
    fn compute_min_max(values: &[f64]) -> (f64, f64) {
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min, max)
    }

    /// Get max heuristic score or default to 1.0
    fn get_max_heuristic(heuristic_scores: &HashMap<String, f64>) -> f64 {
        if heuristic_scores.is_empty() {
            1.0
        } else {
            heuristic_scores
                .values()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        }
    }

    /// Normalize scores when all centrality values are identical
    fn normalize_uniform_scores(
        centrality_scores: &HashMap<String, f64>,
        target_value: f64,
    ) -> HashMap<String, f64> {
        centrality_scores
            .keys()
            .map(|path| (path.clone(), target_value))
            .collect()
    }

    /// Min-max normalization to match heuristic score range
    fn normalize_min_max(
        &self,
        centrality_scores: &HashMap<String, f64>,
        heuristic_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let centrality_values: Vec<f64> = centrality_scores.values().copied().collect();
        let (min_centrality, max_centrality) = Self::compute_min_max(&centrality_values);
        let max_heuristic = Self::get_max_heuristic(heuristic_scores);
        let range = max_centrality - min_centrality;

        if range.abs() < f64::EPSILON {
            return Ok(Self::normalize_uniform_scores(centrality_scores, max_heuristic * 0.5));
        }

        let threshold = self.config.integration.min_centrality_threshold;
        let normalized = centrality_scores
            .iter()
            .filter_map(|(path, &score)| {
                let normalized_score = ((score - min_centrality) / range) * max_heuristic;
                (normalized_score >= threshold).then(|| (path.clone(), normalized_score))
            })
            .collect();

        Ok(normalized)
    }

    /// Compute mean and standard deviation from a slice of values
    fn compute_mean_std(values: &[f64]) -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        (mean, variance.sqrt())
    }

    /// Z-score normalization
    fn normalize_z_score(
        &self,
        centrality_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let values: Vec<f64> = centrality_scores.values().copied().collect();
        let (mean, std_dev) = Self::compute_mean_std(&values);

        if std_dev <= f64::EPSILON {
            return Ok(Self::normalize_uniform_scores(centrality_scores, 0.5));
        }

        let threshold = self.config.integration.min_centrality_threshold;
        let normalized = centrality_scores
            .iter()
            .filter_map(|(path, &score)| {
                let z_score = (score - mean) / std_dev;
                let normalized_score = (z_score + 3.0) / 6.0;
                (normalized_score >= threshold).then(|| (path.clone(), normalized_score))
            })
            .collect();

        Ok(normalized)
    }

    /// Rank-based normalization
    fn normalize_rank(
        &self,
        centrality_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut scored_files: Vec<_> = centrality_scores
            .iter()
            .map(|(path, &score)| (path.clone(), score))
            .collect();

        scored_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut normalized = HashMap::new();
        let total_files = scored_files.len();

        for (rank, (path, _)) in scored_files.into_iter().enumerate() {
            let normalized_score = 1.0 - (rank as f64 / total_files as f64);
            if normalized_score >= self.config.integration.min_centrality_threshold {
                normalized.insert(path, normalized_score);
            }
        }

        Ok(normalized)
    }

    /// Create minimal analysis for large graphs (performance optimization)
    fn create_minimal_analysis(&self, graph: &DependencyGraph) -> Result<GraphAnalysisResults> {
        let minimal_analyzer = GraphStatisticsAnalyzer::for_large_graphs();
        minimal_analyzer.analyze(graph)
    }

    /// Check if a file is an entrypoint
    fn is_entrypoint_file(&self, file_path: &str) -> bool {
        let path = Path::new(file_path);
        let language = file::detect_language_from_path(path);
        file::is_entrypoint_path(path, &language)
    }
}

impl Default for CentralityCalculator {
    fn default() -> Self {
        Self::new().expect("Failed to create CentralityCalculator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scribe_analysis::heuristics::DocumentAnalysis;

    // Mock scan result for testing
    #[derive(Debug, Clone)]
    struct MockScanResult {
        path: String,
        relative_path: String,
        depth: usize,
        imports: Option<Vec<String>>,
        is_docs: bool,
        is_readme: bool,
        is_test: bool,
        is_entrypoint: bool,
        has_examples: bool,
        priority_boost: f64,
        churn_score: f64,
        centrality_in: f64,
        doc_analysis: Option<DocumentAnalysis>,
    }

    impl MockScanResult {
        fn new(path: &str) -> Self {
            Self {
                path: path.to_string(),
                relative_path: path.to_string(),
                depth: path.matches('/').count(),
                imports: None,
                is_docs: path.contains("doc") || path.ends_with(".md"),
                is_readme: path.to_lowercase().contains("readme"),
                is_test: path.contains("test"),
                is_entrypoint: path.contains("main") || path.contains("index"),
                has_examples: path.contains("example"),
                priority_boost: 0.0,
                churn_score: 0.5,
                centrality_in: 0.0,
                doc_analysis: Some(DocumentAnalysis::new()),
            }
        }

        fn with_imports(mut self, imports: Vec<String>) -> Self {
            self.imports = Some(imports);
            self
        }
    }

    impl ScanResult for MockScanResult {
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
    fn test_centrality_calculator_creation() {
        let calculator = CentralityCalculator::new();
        assert!(calculator.is_ok());

        let large_calc = CentralityCalculator::for_large_codebases();
        assert!(large_calc.is_ok());
    }

    #[test]
    fn test_centrality_calculation() {
        let calculator = CentralityCalculator::new().unwrap();

        let scan_results = vec![
            MockScanResult::new("main.py")
                .with_imports(vec!["utils".to_string(), "config".to_string()]),
            MockScanResult::new("utils.py").with_imports(vec!["config".to_string()]),
            MockScanResult::new("config.py"),
            MockScanResult::new("test.py").with_imports(vec!["main".to_string()]),
        ];

        let results = calculator.calculate_centrality(&scan_results).unwrap();

        assert!(!results.pagerank_scores.is_empty());
        assert!(results.integration_metadata.integration_successful);
    }

    #[test]
    fn test_heuristics_integration() {
        let calculator = CentralityCalculator::new().unwrap();

        let scan_results = vec![
            MockScanResult::new("main.py").with_imports(vec!["utils".to_string()]),
            MockScanResult::new("utils.py"),
        ];

        let centrality_results = calculator.calculate_centrality(&scan_results).unwrap();

        let mut heuristic_scores = HashMap::new();
        heuristic_scores.insert("main.py".to_string(), 0.8);
        heuristic_scores.insert("utils.py".to_string(), 0.6);

        let integrated_scores = calculator
            .integrate_with_heuristics(&centrality_results, &heuristic_scores)
            .unwrap();

        assert!(!integrated_scores.is_empty());
    }

    #[test]
    fn test_entrypoint_detection() {
        let calculator = CentralityCalculator::new().unwrap();

        assert!(calculator.is_entrypoint_file("main.py"));
        assert!(calculator.is_entrypoint_file("src/main.rs"));
        assert!(calculator.is_entrypoint_file("index.js"));
    }
}

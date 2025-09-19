//! # Centrality Calculator with Heuristics Integration
//!
//! Main interface for PageRank centrality calculation and integration with the
//! heuristic scoring system used by Scribe. This module provides the high-level API for:
//!
//! ## Key Features
//! - **PageRank Centrality Computation**: Research-grade algorithm with convergence detection
//! - **Import Graph Construction**: Builds dependency graphs from file scan results  
//! - **Heuristics Integration**: Seamless integration with V2 scoring system
//! - **Performance Optimization**: Efficient computation for large codebases
//! - **Multi-language Support**: Import detection across programming languages
//! - **Comprehensive Analysis**: Full graph statistics and structural insights
//!
//! ## Integration with Scribe Heuristics
//! The centrality scores are integrated into the heuristic scoring formula:
//! ```text
//! final_score = Σ(weight_i × normalized_score_i) + priority_boost + template_boost
//! ```
//! Where `centrality_score` becomes a weighted component when V2 features are enabled.

use rayon::prelude::*;
use scribe_analysis::heuristics::ScanResult;
use scribe_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::graph::{DependencyGraph, NodeId};
use crate::pagerank::{PageRankComputer, PageRankConfig, PageRankResults};
use crate::statistics::{GraphAnalysisResults, GraphStatisticsAnalyzer};

/// Complete centrality calculation results with comprehensive metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentralityResults {
    /// PageRank scores (file path -> centrality score)
    pub pagerank_scores: HashMap<NodeId, f64>,

    /// Graph analysis results
    pub graph_analysis: GraphAnalysisResults,

    /// PageRank computation details
    pub pagerank_details: PageRankResults,

    /// Import detection statistics
    pub import_stats: ImportDetectionStats,

    /// Integration metadata
    pub integration_metadata: IntegrationMetadata,
}

/// Statistics about import detection and graph construction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportDetectionStats {
    /// Number of files processed for import detection
    pub files_processed: usize,

    /// Number of import relationships detected
    pub imports_detected: usize,

    /// Number of resolved imports (mapped to actual files)
    pub imports_resolved: usize,

    /// Import resolution success rate
    pub resolution_rate: f64,

    /// Language breakdown of processed files
    pub language_breakdown: HashMap<String, usize>,

    /// Import patterns by language
    pub import_patterns: HashMap<String, ImportPatternStats>,
}

/// Import pattern statistics for a specific language
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportPatternStats {
    /// Total imports found
    pub total_imports: usize,

    /// Relative imports (./,../)
    pub relative_imports: usize,

    /// Absolute imports
    pub absolute_imports: usize,

    /// Standard library imports
    pub stdlib_imports: usize,

    /// Third-party imports
    pub third_party_imports: usize,
}

/// Metadata about centrality-heuristics integration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrationMetadata {
    /// When the analysis was performed
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Total computation time
    pub computation_time_ms: u64,

    /// Whether centrality was successfully integrated
    pub integration_successful: bool,

    /// Centrality weight used in integration
    pub centrality_weight: f64,

    /// Number of files with centrality scores
    pub files_with_centrality: usize,

    /// Configuration used
    pub config: CentralityConfig,
}

/// Configuration for centrality calculation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentralityConfig {
    /// PageRank algorithm configuration
    pub pagerank_config: PageRankConfig,

    /// Whether to perform expensive graph analysis
    pub analyze_graph_structure: bool,

    /// Import resolution configuration
    pub import_resolution: ImportResolutionConfig,

    /// Integration parameters
    pub integration: IntegrationConfig,
}

/// Configuration for import resolution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportResolutionConfig {
    /// Maximum search depth for import resolution
    pub max_search_depth: usize,

    /// Whether to resolve relative imports
    pub resolve_relative_imports: bool,

    /// Whether to resolve absolute imports
    pub resolve_absolute_imports: bool,

    /// Whether to exclude standard library imports
    pub exclude_stdlib_imports: bool,

    /// Custom import path mappings
    pub path_mappings: HashMap<String, String>,
}

/// Configuration for heuristics integration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Weight for centrality in final score
    pub centrality_weight: f64,

    /// Normalization method for centrality scores
    pub normalization_method: NormalizationMethod,

    /// Minimum centrality score threshold
    pub min_centrality_threshold: f64,

    /// Whether to boost entrypoint centrality
    pub boost_entrypoints: bool,

    /// Entrypoint boost factor
    pub entrypoint_boost_factor: f64,
}

/// Methods for normalizing centrality scores
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Normalize to \[0,1\] range
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Rank-based normalization
    Rank,
    /// No normalization
    None,
}

impl Default for CentralityConfig {
    fn default() -> Self {
        Self {
            pagerank_config: PageRankConfig::for_code_analysis(),
            analyze_graph_structure: true,
            import_resolution: ImportResolutionConfig::default(),
            integration: IntegrationConfig::default(),
        }
    }
}

impl Default for ImportResolutionConfig {
    fn default() -> Self {
        Self {
            max_search_depth: 3,
            resolve_relative_imports: true,
            resolve_absolute_imports: true,
            exclude_stdlib_imports: true,
            path_mappings: HashMap::new(),
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            centrality_weight: 0.15, // 15% weight in V2 scoring
            normalization_method: NormalizationMethod::MinMax,
            min_centrality_threshold: 1e-6,
            boost_entrypoints: true,
            entrypoint_boost_factor: 1.5,
        }
    }
}

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
            // Create minimal analysis for large graphs
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

    /// Integrate centrality scores with existing heuristic scores
    pub fn integrate_with_heuristics(
        &self,
        centrality_results: &CentralityResults,
        heuristic_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let normalized_centrality = self
            .normalize_centrality_scores(&centrality_results.pagerank_scores, heuristic_scores)?;

        let mut integrated_scores = HashMap::new();
        let centrality_weight = self.config.integration.centrality_weight;
        let heuristic_weight = 1.0 - centrality_weight;

        // Combine heuristic and centrality scores
        for (file_path, heuristic_score) in heuristic_scores {
            let centrality_score = normalized_centrality.get(file_path).copied().unwrap_or(0.0);

            // Apply entrypoint boost if configured
            let boosted_centrality = if self.config.integration.boost_entrypoints
                && self.is_entrypoint_file(file_path)
            {
                centrality_score * self.config.integration.entrypoint_boost_factor
            } else {
                centrality_score
            };

            let integrated_score =
                heuristic_weight * heuristic_score + centrality_weight * boosted_centrality;

            integrated_scores.insert(file_path.clone(), integrated_score);
        }

        // Add centrality-only files (not in heuristic scores)
        for (file_path, centrality_score) in &normalized_centrality {
            if !integrated_scores.contains_key(file_path) {
                let boosted_centrality = if self.config.integration.boost_entrypoints
                    && self.is_entrypoint_file(file_path)
                {
                    centrality_score * self.config.integration.entrypoint_boost_factor
                } else {
                    *centrality_score
                };

                integrated_scores.insert(file_path.clone(), centrality_weight * boosted_centrality);
            }
        }

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
        let mut optimized_detector =
            ImportDetector::with_file_index(self.import_detector.config.clone(), scan_results);

        // Add all files as nodes first
        for result in scan_results {
            graph.add_node(result.path().to_string())?;
        }

        // Detect imports and build edges using optimized detector
        let import_stats = if self.config.pagerank_config.use_parallel {
            self.build_edges_parallel_optimized(&mut graph, scan_results, &optimized_detector)?
        } else {
            self.build_edges_sequential_optimized(&mut graph, scan_results, &optimized_detector)?
        };

        Ok((graph, import_stats))
    }

    /// Build graph edges sequentially - OPTIMIZED
    fn build_edges_sequential_optimized<T>(
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

    /// Build graph edges sequentially - LEGACY
    fn build_edges_sequential<T>(
        &self,
        graph: &mut DependencyGraph,
        scan_results: &[T],
    ) -> Result<ImportDetectionStats>
    where
        T: ScanResult,
    {
        let optimized_detector =
            ImportDetector::with_file_index(self.import_detector.config.clone(), scan_results);
        self.build_edges_sequential_optimized(graph, scan_results, &optimized_detector)
    }

    /// Build graph edges in parallel - OPTIMIZED
    fn build_edges_parallel_optimized<T>(
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
            import_patterns: HashMap::new(), // TODO: Implement detailed pattern analysis
        };

        Ok(stats)
    }

    /// Build graph edges in parallel - LEGACY
    fn build_edges_parallel<T>(
        &self,
        graph: &mut DependencyGraph,
        scan_results: &[T],
    ) -> Result<ImportDetectionStats>
    where
        T: ScanResult + Sync,
    {
        let optimized_detector =
            ImportDetector::with_file_index(self.import_detector.config.clone(), scan_results);
        self.build_edges_parallel_optimized(graph, scan_results, &optimized_detector)
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

    /// Min-max normalization to match heuristic score range
    fn normalize_min_max(
        &self,
        centrality_scores: &HashMap<String, f64>,
        heuristic_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let centrality_values: Vec<f64> = centrality_scores.values().copied().collect();
        let min_centrality = centrality_values
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_centrality = centrality_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Target range based on heuristic scores
        let heuristic_values: Vec<f64> = heuristic_scores.values().copied().collect();
        let max_heuristic = if heuristic_values.is_empty() {
            1.0
        } else {
            heuristic_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        };

        let mut normalized = HashMap::new();

        if (max_centrality - min_centrality).abs() < f64::EPSILON {
            // All scores are the same
            for (path, _) in centrality_scores {
                normalized.insert(path.clone(), max_heuristic * 0.5); // Use half of max heuristic
            }
        } else {
            for (path, &score) in centrality_scores {
                let normalized_score =
                    ((score - min_centrality) / (max_centrality - min_centrality)) * max_heuristic;
                if normalized_score >= self.config.integration.min_centrality_threshold {
                    normalized.insert(path.clone(), normalized_score);
                }
            }
        }

        Ok(normalized)
    }

    /// Z-score normalization
    fn normalize_z_score(
        &self,
        centrality_scores: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let values: Vec<f64> = centrality_scores.values().copied().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut normalized = HashMap::new();

        if std_dev > f64::EPSILON {
            for (path, &score) in centrality_scores {
                let z_score = (score - mean) / std_dev;
                // Shift and scale to positive range
                let normalized_score = (z_score + 3.0) / 6.0; // Roughly [0,1] for most values
                if normalized_score >= self.config.integration.min_centrality_threshold {
                    normalized.insert(path.clone(), normalized_score);
                }
            }
        } else {
            // All scores are the same
            for (path, _) in centrality_scores {
                normalized.insert(path.clone(), 0.5);
            }
        }

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
        // Use a simplified analyzer for large graphs
        let minimal_analyzer = GraphStatisticsAnalyzer::for_large_graphs();
        minimal_analyzer.analyze(graph)
    }

    /// Check if a file is an entrypoint
    fn is_entrypoint_file(&self, file_path: &str) -> bool {
        let path = Path::new(file_path);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("")
            .to_lowercase();

        matches!(
            file_name.as_str(),
            "main.py"
                | "main.rs"
                | "main.go"
                | "main.js"
                | "main.ts"
                | "index.py"
                | "index.rs"
                | "index.go"
                | "index.js"
                | "index.ts"
                | "app.py"
                | "app.rs"
                | "app.go"
                | "app.js"
                | "app.ts"
                | "server.py"
                | "server.rs"
                | "server.go"
                | "server.js"
                | "server.ts"
                | "lib.rs"
                | "__init__.py"
        )
    }
}

impl Default for CentralityCalculator {
    fn default() -> Self {
        Self::new().expect("Failed to create CentralityCalculator")
    }
}

/// Import detection and resolution engine with pre-computed lookup optimization
#[derive(Debug, Clone)]
pub struct ImportDetector {
    config: ImportResolutionConfig,
    /// Pre-computed lookup map: file stem -> full paths (massive performance improvement)
    stem_to_paths: HashMap<String, Vec<String>>,
    /// Pre-computed lookup map: filename -> full paths
    filename_to_paths: HashMap<String, Vec<String>>,
    /// Set of all available file paths for quick existence checks
    available_paths: HashSet<String>,
}

const PYTHON_FILE_EXTENSIONS: &[&str] = &["py"];
const PYTHON_SUFFIXES: &[&str] = &[".py"];
const JS_FILE_EXTENSIONS: &[&str] = &["js", "jsx", "ts", "tsx", "mjs", "cjs"];
const JS_SUFFIXES: &[&str] = &[".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"];
const RUST_FILE_EXTENSIONS: &[&str] = &["rs"];
const RUST_SUFFIXES: &[&str] = &[".rs"];

fn strip_known_suffix<'a>(value: &'a str, suffixes: &[&str]) -> &'a str {
    for suffix in suffixes {
        if value.ends_with(suffix) {
            return &value[..value.len() - suffix.len()];
        }
    }
    value
}

impl ImportDetector {
    /// Create with configuration
    pub fn with_config(config: ImportResolutionConfig) -> Self {
        Self {
            config,
            stem_to_paths: HashMap::new(),
            filename_to_paths: HashMap::new(),
            available_paths: HashSet::new(),
        }
    }

    /// Create with pre-computed lookup maps for massive performance improvement
    pub fn with_file_index<T>(config: ImportResolutionConfig, scan_results: &[T]) -> Self
    where
        T: ScanResult,
    {
        let mut detector = Self::with_config(config);
        detector.build_lookup_maps(scan_results);
        detector
    }

    /// Build inverted index mapping file stems/names to full paths
    /// This eliminates the O(n) scan-all-files bottleneck
    fn build_lookup_maps<T>(&mut self, scan_results: &[T])
    where
        T: ScanResult,
    {
        self.stem_to_paths.clear();
        self.filename_to_paths.clear();
        self.available_paths.clear();

        for result in scan_results {
            let full_path = result.path().to_string();
            self.available_paths.insert(full_path.clone());

            let path = Path::new(result.path());

            // Index by file stem (name without extension)
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let stem_lower = stem.to_lowercase();
                self.stem_to_paths
                    .entry(stem_lower)
                    .or_insert_with(Vec::new)
                    .push(full_path.clone());
            }

            // Index by full filename
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                let filename_lower = filename.to_lowercase();
                self.filename_to_paths
                    .entry(filename_lower)
                    .or_insert_with(Vec::new)
                    .push(full_path);
            }
        }
    }

    /// Detect programming language from file extension
    pub fn detect_language(&self, file_path: &str) -> Option<String> {
        let path = Path::new(file_path);
        let ext = path.extension()?.to_str()?.to_lowercase();

        match ext.as_str() {
            "py" => Some("python".to_string()),
            "js" | "jsx" | "mjs" => Some("javascript".to_string()),
            "ts" | "tsx" => Some("typescript".to_string()),
            "rs" => Some("rust".to_string()),
            "go" => Some("go".to_string()),
            "java" | "kt" => Some("java".to_string()),
            "cpp" | "cc" | "cxx" | "hpp" | "h" => Some("cpp".to_string()),
            "c" => Some("c".to_string()),
            "rb" => Some("ruby".to_string()),
            "php" => Some("php".to_string()),
            "cs" => Some("csharp".to_string()),
            "swift" => Some("swift".to_string()),
            _ => None,
        }
    }

    /// Resolve import string to actual file path
    pub fn resolve_import<T>(
        &self,
        import_str: &str,
        current_file: &str,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        // Check custom path mappings first
        if let Some(mapped_path) = self.config.path_mappings.get(import_str) {
            if file_map.contains_key(mapped_path.as_str()) {
                return Some(mapped_path.clone());
            }
        }

        let current_path = Path::new(current_file);
        let language = self.detect_language(current_file);

        match language.as_deref() {
            Some("python") => self.resolve_python_import(import_str, current_path, file_map),
            Some("javascript") | Some("typescript") => {
                self.resolve_js_import(import_str, current_path, file_map)
            }
            Some("rust") => self.resolve_rust_import(import_str, current_path, file_map),
            Some("go") => self.resolve_go_import(import_str, current_path, file_map),
            _ => self.resolve_generic_import(import_str, current_path, file_map),
        }
    }

    /// Resolve Python import
    fn resolve_python_import<T>(
        &self,
        import_str: &str,
        current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        if cleaned_import.is_empty() {
            return None;
        }

        if self.config.exclude_stdlib_imports && self.is_python_stdlib(cleaned_import) {
            return None;
        }

        let mut module = cleaned_import;
        if let Some(alias_index) = module.find(" as ") {
            module = &module[..alias_index];
        }

        let mut base_dir = current_path.parent().unwrap_or(current_path).to_path_buf();
        let mut relative_levels = 0;
        while module.starts_with('.') {
            relative_levels += 1;
            module = &module[1..];
        }

        for _ in 0..relative_levels {
            if let Some(parent) = base_dir.parent() {
                base_dir = parent.to_path_buf();
            }
        }

        module = module.trim();
        let module = strip_known_suffix(module, PYTHON_SUFFIXES);
        let module_parts: Vec<&str> = if module.is_empty() {
            Vec::new()
        } else {
            module.split('.').filter(|part| !part.is_empty()).collect()
        };

        if !module_parts.is_empty() {
            if let Some(resolved) = self.resolve_relative_python(&base_dir, &module_parts, file_map)
            {
                return Some(resolved);
            }
        }

        if module_parts.is_empty() {
            return None;
        }

        self.find_module_candidate(&module_parts, PYTHON_FILE_EXTENSIONS)
    }

    /// Resolve JavaScript/TypeScript import
    fn resolve_js_import<T>(
        &self,
        import_str: &str,
        current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        if cleaned_import.is_empty() {
            return None;
        }

        let parent_dir = current_path.parent().unwrap_or(current_path);

        if cleaned_import.starts_with("./") || cleaned_import.starts_with("../") {
            if !self.config.resolve_relative_imports {
                return None;
            }

            if let Some(resolved) = self.resolve_relative_js(parent_dir, cleaned_import, file_map) {
                return Some(resolved);
            }
        } else {
            // Attempt to resolve within the same directory first
            if let Some(resolved) = self.resolve_relative_js(parent_dir, cleaned_import, file_map) {
                return Some(resolved);
            }

            if !self.config.resolve_absolute_imports {
                return None;
            }

            let normalized = strip_known_suffix(cleaned_import, JS_SUFFIXES);
            let module_parts: Vec<&str> = normalized
                .split('/')
                .filter(|segment| !segment.is_empty())
                .collect();

            if module_parts.is_empty() {
                return None;
            }

            return self.find_module_candidate(&module_parts, JS_FILE_EXTENSIONS);
        }

        None
    }

    /// Resolve Rust import (use/mod statements)
    fn resolve_rust_import<T>(
        &self,
        import_str: &str,
        current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        if cleaned_import.is_empty() {
            return None;
        }

        if self.config.exclude_stdlib_imports && self.is_rust_stdlib(cleaned_import) {
            return None;
        }

        let mut module = cleaned_import;

        if let Some(stripped) = module.strip_prefix("crate::") {
            module = stripped;
        }

        while let Some(stripped) = module.strip_prefix("self::") {
            module = stripped;
        }

        let mut base_dir = current_path.parent().unwrap_or(current_path).to_path_buf();
        while let Some(stripped) = module.strip_prefix("super::") {
            module = stripped;
            if let Some(parent) = base_dir.parent() {
                base_dir = parent.to_path_buf();
            }
        }

        module = strip_known_suffix(module, RUST_SUFFIXES);
        let module_parts: Vec<&str> = module
            .split("::")
            .filter(|segment| !segment.is_empty())
            .collect();

        if module_parts.is_empty() {
            return None;
        }

        if let Some(resolved) = self.resolve_relative_rust(&base_dir, &module_parts, file_map) {
            return Some(resolved);
        }

        if module_parts.len() == 1 {
            let crate_lib = base_dir.join("lib.rs");
            if let Some(candidate_str) = crate_lib.to_str() {
                if file_map.contains_key(candidate_str) {
                    return Some(candidate_str.to_string());
                }
            }
        }

        self.find_module_candidate(&module_parts, RUST_FILE_EXTENSIONS)
    }

    /// Resolve Go import
    fn resolve_go_import<T>(
        &self,
        import_str: &str,
        _current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim().trim_matches('"');

        // Skip standard library
        if self.config.exclude_stdlib_imports && !cleaned_import.contains('.') {
            return None;
        }

        let parts: Vec<&str> = cleaned_import.split('/').collect();

        // Try various Go file patterns
        let mut candidates = Vec::new();

        // Package directory
        candidates.push(format!("{}.go", parts.last()?));
        candidates.push(format!("{}/main.go", cleaned_import));
        candidates.push(format!("{}/{}.go", cleaned_import, parts.last()?));

        for candidate in &candidates {
            if file_map.contains_key(candidate.as_str()) {
                return Some(candidate.clone());
            }
        }

        self.fuzzy_match_import(&parts, file_map)
    }

    /// Generic import resolution
    fn resolve_generic_import<T>(
        &self,
        import_str: &str,
        _current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        let parts: Vec<&str> = cleaned_import.split(&['/', '.', ':']).collect();
        self.fuzzy_match_import(&parts, file_map)
    }

    fn resolve_relative_python<T>(
        &self,
        base_dir: &Path,
        module_parts: &[&str],
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        if module_parts.is_empty() {
            return None;
        }

        let mut module_path = base_dir.to_path_buf();
        for part in module_parts {
            module_path.push(part);
        }

        let mut candidate = module_path.clone();
        candidate.set_extension("py");
        if let Some(candidate_str) = candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        let init_candidate = module_path.join("__init__.py");
        if let Some(candidate_str) = init_candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        None
    }

    fn resolve_relative_js<T>(
        &self,
        base_dir: &Path,
        import_path: &str,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let normalized = strip_known_suffix(import_path, JS_SUFFIXES);
        let target = self.build_relative_js_path(base_dir, normalized);

        for ext in JS_FILE_EXTENSIONS {
            let mut candidate = target.clone();
            candidate.set_extension(ext);
            if let Some(candidate_str) = candidate.to_str() {
                if file_map.contains_key(candidate_str) {
                    return Some(candidate_str.to_string());
                }
            }
        }

        for ext in JS_FILE_EXTENSIONS {
            let index_candidate = target.join(format!("index.{}", ext));
            if let Some(candidate_str) = index_candidate.to_str() {
                if file_map.contains_key(candidate_str) {
                    return Some(candidate_str.to_string());
                }
            }
        }

        None
    }

    fn build_relative_js_path(&self, base_dir: &Path, import_path: &str) -> PathBuf {
        let mut resolved = base_dir.to_path_buf();
        for segment in import_path.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    if let Some(parent) = resolved.parent() {
                        resolved = parent.to_path_buf();
                    }
                }
                _ => resolved.push(segment),
            }
        }
        resolved
    }

    fn resolve_relative_rust<T>(
        &self,
        base_dir: &Path,
        module_parts: &[&str],
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        if module_parts.is_empty() {
            return None;
        }

        let mut module_path = base_dir.to_path_buf();
        for part in module_parts {
            module_path.push(part);
        }

        let mut candidate = module_path.clone();
        candidate.set_extension("rs");
        if let Some(candidate_str) = candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        let mod_candidate = module_path.join("mod.rs");
        if let Some(candidate_str) = mod_candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        None
    }

    fn find_module_candidate(&self, module_parts: &[&str], extensions: &[&str]) -> Option<String> {
        if module_parts.is_empty() {
            return None;
        }

        let stem = module_parts.last().unwrap().to_lowercase();
        let candidates = self.stem_to_paths.get(&stem)?;

        for candidate in candidates {
            if self.module_path_matches(candidate, module_parts, extensions) {
                return Some(candidate.clone());
            }
        }

        None
    }

    fn module_path_matches(
        &self,
        candidate: &str,
        module_parts: &[&str],
        extensions: &[&str],
    ) -> bool {
        let path = Path::new(candidate);
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => return false,
        };

        let lower_file = file_name.to_lowercase();
        if lower_file == "__init__.py" {
            return self.dir_path_matches(path.parent(), module_parts);
        }

        let ext = Path::new(file_name)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        if !extensions
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(&ext))
        {
            return false;
        }

        let stem = Path::new(file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        if stem == "index" && !module_parts.is_empty() {
            return self.dir_path_matches(path.parent(), module_parts);
        }

        if module_parts.is_empty() {
            return false;
        }

        if stem != module_parts.last().unwrap().to_lowercase() {
            return false;
        }

        self.dir_path_matches(
            path.parent(),
            &module_parts[..module_parts.len().saturating_sub(1)],
        )
    }

    fn dir_path_matches(&self, dir: Option<&Path>, module_parts: &[&str]) -> bool {
        if module_parts.is_empty() {
            return true;
        }

        let mut current = dir;
        for expected in module_parts.iter().rev() {
            match current {
                Some(path) => {
                    let name = path.file_name().and_then(|n| n.to_str());
                    match name {
                        Some(name) if name.eq_ignore_ascii_case(expected) => {
                            current = path.parent();
                        }
                        _ => return false,
                    }
                }
                None => return false,
            }
        }

        true
    }

    /// Fuzzy matching for import resolution - OPTIMIZED with pre-computed maps
    fn fuzzy_match_import<T>(
        &self,
        import_parts: &[&str],
        _file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        if import_parts.is_empty() {
            return None;
        }

        let last_part = import_parts.last()?.to_lowercase();

        // MASSIVE PERFORMANCE IMPROVEMENT: Use pre-computed lookup maps instead of O(n) scan
        // 1. First try exact stem match (most common case)
        if let Some(paths) = self.stem_to_paths.get(&last_part) {
            // Return first match (could be made smarter with scoring)
            if let Some(first_path) = paths.first() {
                return Some(first_path.clone());
            }
        }

        // 2. Try filename match
        if let Some(paths) = self.filename_to_paths.get(&last_part) {
            if let Some(first_path) = paths.first() {
                return Some(first_path.clone());
            }
        }

        // 3. Try partial matching against stems
        for (stem, paths) in &self.stem_to_paths {
            if stem.contains(&last_part) || last_part.contains(stem) {
                if let Some(first_path) = paths.first() {
                    return Some(first_path.clone());
                }
            }
        }

        // 4. Fallback: check if path contains all import parts
        for path in &self.available_paths {
            let path_lower = path.to_lowercase();
            if import_parts
                .iter()
                .all(|&part| path_lower.contains(&part.to_lowercase()))
            {
                return Some(path.clone());
            }
        }

        None
    }

    /// Check if import is Python standard library
    fn is_python_stdlib(&self, import_str: &str) -> bool {
        let stdlib_modules = [
            "os",
            "sys",
            "re",
            "json",
            "collections",
            "itertools",
            "functools",
            "typing",
            "datetime",
            "math",
            "random",
            "string",
            "pathlib",
            "io",
            "csv",
            "xml",
            "html",
            "urllib",
            "http",
            "email",
            "logging",
            "unittest",
            "asyncio",
            "concurrent",
            "multiprocessing",
            "threading",
            "subprocess",
        ];

        let first_part = import_str.split('.').next().unwrap_or(import_str);
        stdlib_modules.contains(&first_part)
    }

    /// Check if import is Rust standard library
    fn is_rust_stdlib(&self, import_str: &str) -> bool {
        import_str.starts_with("std::")
            || import_str.starts_with("core::")
            || import_str.starts_with("alloc::")
    }
}

/// Utility functions for centrality results analysis
impl CentralityResults {
    /// Get files sorted by centrality score (descending)
    pub fn top_files_by_centrality(&self, k: usize) -> Vec<(String, f64)> {
        let mut scored_files: Vec<_> = self
            .pagerank_scores
            .iter()
            .map(|(path, &score)| (path.clone(), score))
            .collect();

        scored_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_files.into_iter().take(k).collect()
    }

    /// Get summary statistics about centrality computation
    pub fn summary(&self) -> String {
        format!(
            "Centrality Analysis Summary:\n\
             - Files with centrality scores: {}\n\
             - PageRank iterations: {} (converged: {})\n\
             - Graph: {} nodes, {} edges (density: {:.4})\n\
             - Import resolution: {:.1}% ({}/{})\n\
             - Top languages: {}\n\
             - Computation time: {}ms\n\
             - Integration weight: {:.2}",
            self.pagerank_scores.len(),
            self.pagerank_details.iterations_converged,
            self.pagerank_details.converged(),
            self.graph_analysis.basic_stats.total_nodes,
            self.graph_analysis.basic_stats.total_edges,
            self.graph_analysis.basic_stats.graph_density,
            self.import_stats.resolution_rate * 100.0,
            self.import_stats.imports_resolved,
            self.import_stats.imports_detected,
            self.import_stats
                .language_breakdown
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(lang, count)| format!("{} ({})", lang, count))
                .unwrap_or_else(|| "None".to_string()),
            self.integration_metadata.computation_time_ms,
            self.integration_metadata.centrality_weight,
        )
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
    fn test_import_detection() {
        let detector = ImportDetector::with_config(ImportResolutionConfig::default());

        // Test language detection
        assert_eq!(
            detector.detect_language("main.py"),
            Some("python".to_string())
        );
        assert_eq!(
            detector.detect_language("app.js"),
            Some("javascript".to_string())
        );
        assert_eq!(detector.detect_language("lib.rs"), Some("rust".to_string()));

        // Test Python stdlib detection
        assert!(detector.is_python_stdlib("os"));
        assert!(detector.is_python_stdlib("sys.path"));
        assert!(!detector.is_python_stdlib("custom_module"));

        // Test Rust stdlib detection
        assert!(detector.is_rust_stdlib("std::collections::HashMap"));
        assert!(detector.is_rust_stdlib("core::fmt"));
        assert!(!detector.is_rust_stdlib("serde::Deserialize"));
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

        // Basic checks
        assert!(!results.pagerank_scores.is_empty());
        assert!(results.integration_metadata.integration_successful);
        assert_eq!(
            results.integration_metadata.files_with_centrality,
            results.pagerank_scores.len()
        );

        // config.py should have high centrality (imported by main.py and utils.py)
        let config_score = results.pagerank_scores.get("config.py");
        assert!(config_score.is_some());

        println!("Centrality scores:");
        for (file, score) in &results.pagerank_scores {
            println!("  {}: {:.6}", file, score);
        }

        println!("\n{}", results.summary());
    }

    #[test]
    fn test_heuristics_integration() {
        let calculator = CentralityCalculator::new().unwrap();

        let scan_results = vec![
            MockScanResult::new("main.py").with_imports(vec!["utils".to_string()]),
            MockScanResult::new("utils.py"),
        ];

        let centrality_results = calculator.calculate_centrality(&scan_results).unwrap();

        // Mock heuristic scores
        let mut heuristic_scores = HashMap::new();
        heuristic_scores.insert("main.py".to_string(), 0.8);
        heuristic_scores.insert("utils.py".to_string(), 0.6);

        let integrated_scores = calculator
            .integrate_with_heuristics(&centrality_results, &heuristic_scores)
            .unwrap();

        assert!(!integrated_scores.is_empty());

        // Integrated scores should be different from original heuristic scores
        for (file, integrated_score) in &integrated_scores {
            let original_score = heuristic_scores.get(file).unwrap();
            println!(
                "File {}: heuristic={:.3}, integrated={:.3}",
                file, original_score, integrated_score
            );
        }
    }

    #[test]
    fn test_normalization_methods() {
        let calculator = CentralityCalculator::new().unwrap();

        let centrality_scores = vec![
            ("file1".to_string(), 0.1),
            ("file2".to_string(), 0.3),
            ("file3".to_string(), 0.6),
            ("file4".to_string(), 1.0),
        ]
        .into_iter()
        .collect();

        let heuristic_scores = vec![
            ("file1".to_string(), 0.5),
            ("file2".to_string(), 0.7),
            ("file3".to_string(), 0.9),
            ("file4".to_string(), 1.2),
        ]
        .into_iter()
        .collect();

        // Test min-max normalization
        let normalized = calculator
            .normalize_min_max(&centrality_scores, &heuristic_scores)
            .unwrap();
        assert!(!normalized.is_empty());

        // Test z-score normalization
        let z_normalized = calculator.normalize_z_score(&centrality_scores).unwrap();
        assert!(!z_normalized.is_empty());

        // Test rank normalization
        let rank_normalized = calculator.normalize_rank(&centrality_scores).unwrap();
        assert!(!rank_normalized.is_empty());

        println!("Original scores: {:?}", centrality_scores);
        println!("Min-max normalized: {:?}", normalized);
        println!("Z-score normalized: {:?}", z_normalized);
        println!("Rank normalized: {:?}", rank_normalized);
    }

    #[test]
    fn test_import_resolution() {
        let detector = ImportDetector::with_config(ImportResolutionConfig::default());

        // Create mock file map
        let scan_results = vec![
            MockScanResult::new("src/main.py"),
            MockScanResult::new("src/utils.py"),
            MockScanResult::new("src/config.py"),
            MockScanResult::new("tests/test_main.py"),
        ];

        let file_map: HashMap<&str, &MockScanResult> = scan_results
            .iter()
            .map(|result| (result.path(), result))
            .collect();

        // Test Python import resolution
        let resolved = detector.resolve_import("utils", "src/main.py", &file_map);
        assert!(resolved.is_some());

        // Test module path resolution
        let resolved_config = detector.resolve_import("src.config", "src/main.py", &file_map);
        // Should resolve to src/config.py through fuzzy matching
        assert!(resolved_config.is_some());

        println!("Resolved imports:");
        if let Some(path) = resolved {
            println!("  utils -> {}", path);
        }
        if let Some(path) = resolved_config {
            println!("  src.config -> {}", path);
        }
    }

    #[test]
    fn test_entrypoint_detection() {
        let calculator = CentralityCalculator::new().unwrap();

        assert!(calculator.is_entrypoint_file("main.py"));
        assert!(calculator.is_entrypoint_file("src/main.rs"));
        assert!(calculator.is_entrypoint_file("index.js"));
        assert!(calculator.is_entrypoint_file("app.py"));
        assert!(calculator.is_entrypoint_file("lib.rs"));
        assert!(calculator.is_entrypoint_file("__init__.py"));

        assert!(!calculator.is_entrypoint_file("utils.py"));
        assert!(!calculator.is_entrypoint_file("config.rs"));
        assert!(!calculator.is_entrypoint_file("helper.js"));
    }

    #[test]
    fn test_top_files_by_centrality() {
        let mut pagerank_scores = HashMap::new();
        pagerank_scores.insert("file1.py".to_string(), 0.4);
        pagerank_scores.insert("file2.py".to_string(), 0.6);
        pagerank_scores.insert("file3.py".to_string(), 0.2);
        pagerank_scores.insert("file4.py".to_string(), 0.8);

        let results = CentralityResults {
            pagerank_scores,
            graph_analysis: GraphAnalysisResults {
                basic_stats: crate::graph::GraphStatistics::empty(),
                degree_distribution: Default::default(),
                connectivity: Default::default(),
                structural_patterns: Default::default(),
                import_insights: Default::default(),
                performance_profile: Default::default(),
                analysis_metadata: Default::default(),
            },
            pagerank_details: PageRankResults {
                scores: HashMap::new(),
                iterations_converged: 10,
                convergence_epsilon: 1e-6,
                graph_stats: crate::graph::GraphStatistics::empty(),
                parameters: PageRankConfig::default(),
                performance_metrics: Default::default(),
            },
            import_stats: ImportDetectionStats {
                files_processed: 4,
                imports_detected: 0,
                imports_resolved: 0,
                resolution_rate: 0.0,
                language_breakdown: HashMap::new(),
                import_patterns: HashMap::new(),
            },
            integration_metadata: IntegrationMetadata {
                timestamp: chrono::Utc::now(),
                computation_time_ms: 100,
                integration_successful: true,
                centrality_weight: 0.15,
                files_with_centrality: 4,
                config: CentralityConfig::default(),
            },
        };

        let top_files = results.top_files_by_centrality(2);
        assert_eq!(top_files.len(), 2);
        assert_eq!(top_files[0].0, "file4.py");
        assert_eq!(top_files[0].1, 0.8);
        assert_eq!(top_files[1].0, "file2.py");
        assert_eq!(top_files[1].1, 0.6);
    }
}

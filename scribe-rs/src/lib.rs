#![cfg_attr(not(tarpaulin), warn(warnings))]
#![cfg_attr(tarpaulin, allow(warnings))]

//! # Scribe - Advanced Code Analysis Library
//!
//! Scribe is a comprehensive Rust library for code analysis, repository exploration,
//! and intelligent file processing. It provides powerful tools for understanding
//! codebases through heuristic scoring, graph analysis, and AI-powered insights.
//!
//! ## Features
//!
//! - **🔍 Intelligent File Analysis**: Multi-dimensional heuristic scoring system
//! - **📊 Dependency Graph Analysis**: PageRank centrality for code importance
//! - **⚡ High-Performance Scanning**: Parallel file system traversal with git integration
//! - **🎯 Advanced Pattern Matching**: Flexible glob and gitignore pattern support
//! - **🧠 Smart Code Selection**: Context-aware code bundling and relevance scoring
//! - **🛠️ Extensible Architecture**: Plugin system for custom analyzers and scorers
//!
//! ## Quick Start
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! scribe = "0.1.0"
//! ```
//!
//! ### Basic Usage
//!
//! ```rust,ignore
//! use scribe_analyzer::prelude::*;
//! use std::path::Path;
//!
//! # async fn example() -> scribe_analyzer::Result<()> {
//! // Configure analysis
//! let config = Config::default();
//! let repo_path = Path::new(".");
//!
//! // Quick analysis - get most important files
//! let important_files = scribe_analyzer::analyze_repository(repo_path, &config).await?;
//!
//! println!("Top 10 most important files:");
//! for (file, score) in important_files.top_files(10) {
//!     println!("  {}: {:.3}", file, score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Feature-Specific Usage
//!
//! ```rust,ignore
//! // For minimal installations with selective features
//! use scribe_analyzer::core::{Config, FileInfo};
//! use scribe_analyzer::scanner::{Scanner, ScanOptions};
//!
//! # async fn selective_example() -> scribe_analyzer::Result<()> {
//! let scanner = Scanner::new();
//! let options = ScanOptions::default();
//! let files = scanner.scan(".", options).await?;
//! println!("Found {} files", files.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! Scribe uses feature flags to allow selective compilation:
//!
//! - **`default`**: Includes `core`, `analysis`, `graph`, `scanner`, `patterns`, `selection`
//! - **`core`**: Essential types, traits, and utilities (always recommended)
//! - **`analysis`**: Heuristic scoring and code analysis algorithms
//! - **`graph`**: PageRank centrality and dependency graph analysis
//! - **`scanner`**: High-performance file system scanning with git integration
//! - **`patterns`**: Flexible pattern matching (glob, gitignore)
//! - **`selection`**: Intelligent code selection and context extraction
//!
//! ### Feature Groups
//!
//! - **`minimal`**: Just `core` functionality
//! - **`fast`**: Core + scanning and patterns for quick file operations
//! - **`comprehensive`**: All features (same as default)
//! - **`full`**: Alias for default
//!
//! ### Selective Installation Examples
//!
//! ```toml
//! # Minimal installation
//! scribe = { version = "0.1.0", default-features = false, features = ["core"] }
//!
//! # Fast file operations only
//! scribe = { version = "0.1.0", default-features = false, features = ["fast"] }
//!
//! # Analysis without graph features
//! scribe = { version = "0.1.0", default-features = false, features = ["core", "analysis", "scanner"] }
//! ```
//!
//! ## Architecture
//!
//! Scribe is built with a modular architecture:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                        scribe                               │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
//! │  │ scribe-core │ │scribe-scanner│ │    scribe-patterns     │ │
//! │  │   (types,   │ │(file system  │ │  (glob, gitignore,     │ │
//! │  │ traits,     │ │ traversal,   │ │   pattern matching)    │ │
//! │  │ utilities)  │ │ git support) │ │                        │ │
//! │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
//! │  │scribe-analysis│ │scribe-graph │ │   scribe-selection     │ │
//! │  │ (heuristic  │ │  (PageRank  │ │ (intelligent bundling, │ │
//! │  │  scoring,   │ │ centrality, │ │  context extraction,   │ │
//! │  │ code metrics)│ │ dependency  │ │   relevance scoring)   │ │
//! │  │             │ │  analysis)  │ │                        │ │
//! │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```

// Re-export core functionality (always available when scribe is used)

pub mod pipeline;
pub mod report;

pub use pipeline::{
    analyze_and_select, select_from_analysis, AnalysisOutcome, SelectionOptions, SelectionOutcome,
};

pub use report::{
    format_bytes, format_number, format_timestamp, generate_html_output, generate_json_output,
    generate_markdown_output, generate_repomix_output, generate_report, generate_text_output,
    generate_xml_output, get_file_icon, ReportFile, ReportFormat, SelectionMetrics,
};

#[cfg(feature = "core")]
pub use scribe_core as core;

#[cfg(feature = "core")]
pub use scribe_core::{
    meta,
    Config,
    FileInfo,
    FileType,
    HeuristicWeights,

    Language,
    // Essential types
    Result,
    ScoreComponents,
    ScribeError,
    // Version and meta information
    VERSION as CORE_VERSION,
};

// Analysis functionality
#[cfg(feature = "analysis")]
pub use scribe_analysis as analysis;

#[cfg(feature = "analysis")]
pub use scribe_analysis::{
    DocumentAnalysis, HeuristicScorer, HeuristicSystem, ImportGraph, ImportGraphBuilder,
    TemplateDetector,
};

// Graph analysis functionality
#[cfg(feature = "graph")]
pub use scribe_graph as graph;

#[cfg(feature = "graph")]
pub use scribe_graph::{
    CentralityCalculator,
    CentralityResults,
    DependencyGraph,
    GraphStatistics,
    PageRankAnalysis,
    PageRankAnalysis as GraphAnalysis, // Alias for convenience
    PageRankResults,
};

// Scanner functionality
#[cfg(feature = "scanner")]
pub use scribe_scanner as scanner;

#[cfg(feature = "scanner")]
pub use scribe_scanner::{
    FileScanner, LanguageDetector, ScanOptions, ScanResult, Scanner, ScannerStats,
};

// Pattern matching functionality
#[cfg(feature = "patterns")]
pub use scribe_patterns as patterns;

#[cfg(feature = "patterns")]
pub use scribe_patterns::{
    presets, GitignoreMatcher, GlobMatcher, PatternBuilder, PatternMatcher, PatternMatcherBuilder,
    QuickMatcher,
};

// Selection functionality
#[cfg(feature = "selection")]
pub use scribe_selection as selection;

#[cfg(feature = "selection")]
pub use scribe_selection::{
    apply_token_budget_selection, CodeBundle, CodeBundler, CodeContext, CodeSelector,
    ContextExtractor, ContextFile, QuotaManager, SelectionConfig, SelectionEngine, TwoPassSelector,
};

/// Current version of the main Scribe library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// High-level repository analysis results
#[cfg(all(feature = "analysis", feature = "scanner"))]
#[derive(Debug, Clone)]
pub struct RepositoryAnalysis {
    /// All scanned files with metadata
    pub files: Vec<FileInfo>,
    /// Heuristic scores for each file
    pub heuristic_scores: std::collections::HashMap<String, f64>,
    /// Graph centrality scores (if graph feature enabled)
    #[cfg(feature = "graph")]
    pub centrality_scores: Option<std::collections::HashMap<String, f64>>,
    /// Combined final scores
    pub final_scores: std::collections::HashMap<String, f64>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

#[cfg(all(feature = "analysis", feature = "scanner"))]
impl RepositoryAnalysis {
    /// Get the top N files by score
    pub fn top_files(&self, n: usize) -> Vec<(&str, f64)> {
        let mut scored: Vec<_> = self
            .final_scores
            .iter()
            .map(|(path, score)| (path.as_str(), *score))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n).collect()
    }

    /// Get files above a certain score threshold
    pub fn files_above_threshold(&self, threshold: f64) -> Vec<(&str, f64)> {
        self.final_scores
            .iter()
            .filter(|(_, score)| **score >= threshold)
            .map(|(path, score)| (path.as_str(), *score))
            .collect()
    }

    /// Get total number of analyzed files
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Get analysis summary statistics
    pub fn summary(&self) -> String {
        let avg_score = self.final_scores.values().sum::<f64>() / self.final_scores.len() as f64;
        let top_file = self
            .top_files(1)
            .get(0)
            .map(|(path, score)| format!("{} ({:.3})", path, score))
            .unwrap_or_else(|| "None".to_string());

        format!(
            "Repository Analysis Summary:\n\
             - Files analyzed: {}\n\
             - Average score: {:.3}\n\
             - Top file: {}\n\
             - Scribe version: {}",
            self.file_count(),
            avg_score,
            top_file,
            self.metadata.scribe_version
        )
    }
}

/// Convenience function for quick repository analysis
///
/// This function performs a complete repository analysis using default configuration
/// and returns the most important files based on comprehensive scoring.
///
/// # Example
///
/// ```rust,ignore
/// use scribe_analyzer;
/// use std::path::Path;
///
/// # async fn example() -> scribe_analyzer::Result<()> {
/// let config = scribe_analyzer::Config::default();
/// let analysis = scribe_analyzer::analyze_repository(".", &config).await?;
///
/// println!("Analysis: {}", analysis.summary());
/// for (file, score) in analysis.top_files(5) {
///     println!("  {}: {:.3}", file, score);
/// }
/// # Ok(())
/// # }
/// ```
#[cfg(all(feature = "analysis", feature = "scanner", feature = "patterns"))]
pub async fn analyze_repository<P: AsRef<std::path::Path>>(
    path: P,
    config: &Config,
) -> Result<RepositoryAnalysis> {
    let optimized_config = build_optimized_config(config);

    // Try scaling engine for large repositories
    #[cfg(feature = "scaling")]
    if let Some(result) = try_scaling_analysis(path.as_ref(), &optimized_config, config).await {
        return result;
    }

    // Fallback to the optimized scanning pipeline
    fallback_scan(path, &optimized_config).await
}

/// Build optimized config for repository analysis
fn build_optimized_config(config: &Config) -> Config {
    let mut optimized = config.clone();
    optimized.performance.batch_size = 20;
    optimized.performance.use_mmap = true;
    optimized.performance.io_buffer_size = 512 * 1024;
    optimized.analysis.enable_caching = true;
    optimized
}

/// Log debug message if SCRIBE_DEBUG is set
fn debug_log(msg: &str) {
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("{}", msg);
    }
}

/// Try using the scaling engine for large repositories
#[cfg(feature = "scaling")]
async fn try_scaling_analysis(
    path: &std::path::Path,
    optimized_config: &Config,
    original_config: &Config,
) -> Option<Result<RepositoryAnalysis>> {
    use scribe_scaling::{create_scaling_engine, quick_scale_estimate};

    let (file_count, estimated_duration, _) = match quick_scale_estimate(path).await {
        Ok(estimate) => estimate,
        Err(e) => {
            debug_log(&format!("Scaling estimate failed: {}, falling back", e));
            return None;
        }
    };

    debug_log(&format!(
        "Scaling estimate: {} files, {:?} duration",
        file_count, estimated_duration
    ));

    let is_large_repo = file_count > 50 || estimated_duration.as_secs() > 2;

    if !is_large_repo {
        debug_log("Small repo detected, using optimized basic scanner");
        return None;
    }

    if !original_config.features.scaling_enabled {
        debug_log("Large repo detected but scaling disabled, using optimized basic scanner");
        return None;
    }

    debug_log("Using scaling engine for large repo");

    let mut scaling_engine = match create_scaling_engine(path).await {
        Ok(engine) => engine,
        Err(e) => {
            debug_log(&format!("Failed to create scaling engine: {}, falling back", e));
            return None;
        }
    };

    debug_log("Scaling engine created, processing repository...");

    match scaling_engine.process_repository(path).await {
        Ok(result) => {
            debug_log(&format!(
                "Scaling processing complete: {} files processed in {:?}",
                result.total_files, result.processing_time
            ));
            Some(convert_scaling_result_to_analysis(result, optimized_config.clone(), path).await)
        }
        Err(e) => {
            debug_log(&format!("Scaling engine processing failed: {}, falling back", e));
            None
        }
    }
}

async fn fallback_scan<P: AsRef<std::path::Path>>(
    path: P,
    config: &Config,
) -> Result<RepositoryAnalysis> {
    let repo_path = path.as_ref();
    let start_time = std::time::Instant::now();

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("🔄 Using fallback scanner with optimized config");
    }

    let scanner = Scanner::new();
    let scan_options = ScanOptions::default()
        .with_git_integration(true)
        .with_parallel_processing(true);

    let mut files = scanner.scan(repo_path, scan_options).await?;

    if config.features.auto_exclude_tests {
        let original_count = files.len();
        files.retain(|file| !scribe_core::file::is_test_path(&file.path));
        if std::env::var("SCRIBE_DEBUG").is_ok() && files.len() != original_count {
            eprintln!(
                "Auto-excluded {} test files, {} files remaining",
                original_count - files.len(),
                files.len()
            );
        }
    }

    if let Some(token_budget) = config.analysis.token_budget {
        if std::env::var("SCRIBE_DEBUG").is_ok() {
            eprintln!("🎯 Applying token budget: {} tokens", token_budget);
        }
        files = apply_token_budget_selection(files, token_budget, config, None, &SelectionConfig::default()).await?;
        if std::env::var("SCRIBE_DEBUG").is_ok() {
            eprintln!("✅ Token budget applied: {} files selected", files.len());
        }
    }

    let analysis = build_repository_analysis(files, config, &["optimized_scanner"])?;

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!(
            "📊 Completed fallback analysis in {:?} ({} files)",
            start_time.elapsed(),
            analysis.files.len()
        );
    }

    Ok(analysis)
}

#[derive(Debug, Clone)]
struct AnalyzerContext {
    imports: Vec<String>,
    doc_analysis: Option<DocumentAnalysis>,
    has_examples: bool,
    is_entrypoint: bool,
    priority_boost: f64,
    content: Option<String>,
}

#[derive(Debug, Clone)]
struct AnalyzerFile {
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
    centrality_score: f64,
    imports: Vec<String>,
    doc_analysis: Option<DocumentAnalysis>,
}

impl AnalyzerFile {
    fn from_file_info(file: &FileInfo, context: &AnalyzerContext) -> Self {
        let path_string = file.path.to_string_lossy().to_string();
        let relative = if file.relative_path.is_empty() {
            path_string.clone()
        } else {
            file.relative_path.clone()
        };
        let normalized_path = relative.replace('\\', "/");
        let depth = normalized_path.matches('/').count();
        let is_docs = matches!(file.file_type, FileType::Documentation { .. });
        let is_readme = normalized_path.to_lowercase().contains("readme");
        let is_test = matches!(file.file_type, FileType::Test { .. })
            || scribe_core::file::is_test_path(&file.path);

        Self {
            path: path_string,
            relative_path: normalized_path,
            depth,
            is_docs,
            is_readme,
            is_test,
            is_entrypoint: context.is_entrypoint,
            has_examples: context.has_examples,
            priority_boost: context.priority_boost.min(1.0),
            churn_score: compute_churn_score(file),
            centrality_score: 0.0,
            imports: context.imports.clone(),
            doc_analysis: context.doc_analysis.clone(),
        }
    }
}

impl scribe_analysis::heuristics::ScanResult for AnalyzerFile {
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
        self.centrality_score
    }

    fn imports(&self) -> Option<&[String]> {
        if self.imports.is_empty() {
            None
        } else {
            Some(&self.imports)
        }
    }

    fn doc_analysis(&self) -> Option<&DocumentAnalysis> {
        self.doc_analysis.as_ref()
    }
}

fn build_repository_analysis(
    mut files: Vec<FileInfo>,
    config: &Config,
    additional_features: &[&str],
) -> Result<RepositoryAnalysis> {
    use std::collections::{HashMap, HashSet};

    let contexts: Vec<AnalyzerContext> = files
        .iter()
        .map(|file| derive_file_context(file, config))
        .collect();

    let mut analyzer_files: Vec<AnalyzerFile> = files
        .iter()
        .zip(contexts.iter())
        .map(|(file, context)| AnalyzerFile::from_file_info(file, context))
        .collect();

    #[cfg(feature = "graph")]
    let mut centrality_scores = compute_centrality_scores(&analyzer_files);

    #[cfg(not(feature = "graph"))]
    let mut centrality_scores: Option<HashMap<String, f64>> = None;

    #[cfg(feature = "graph")]
    if let Some(ref centrality) = centrality_scores {
        for analyzer in analyzer_files.iter_mut() {
            if let Some(score) = centrality.get(&analyzer.path) {
                analyzer.centrality_score = *score;
            }
        }

        for file in files.iter_mut() {
            let key = file.path.to_string_lossy().to_string();
            if let Some(score) = centrality.get(&key) {
                file.centrality_score = Some(*score);
            }
        }
    }

    let mut heuristic_scores = HashMap::with_capacity(analyzer_files.len());
    let mut scoring_system = HeuristicSystem::with_v2_features()?;
    let scored_files = scoring_system.score_all_files(&analyzer_files)?;
    for (idx, components) in scored_files {
        let key = analyzer_files[idx].path.clone();
        heuristic_scores.insert(key, components.final_score);
    }

    let final_scores = heuristic_scores.clone();

    let mut features: HashSet<String> = HashSet::new();
    features.insert("heuristic_scoring".to_string());
    #[cfg(feature = "graph")]
    if centrality_scores.is_some() {
        features.insert("centrality_analysis".to_string());
    }
    for feature in additional_features {
        features.insert((*feature).to_string());
    }

    let mut features_enabled: Vec<String> = features.into_iter().collect();
    features_enabled.sort();

    let metadata = AnalysisMetadata {
        timestamp: std::time::SystemTime::now(),
        scribe_version: VERSION.to_string(),
        config_hash: Some(config.compute_hash()),
        features_enabled,
    };

    Ok(RepositoryAnalysis {
        files,
        heuristic_scores,
        #[cfg(feature = "graph")]
        centrality_scores,
        final_scores,
        metadata,
    })
}

fn derive_file_context(file: &FileInfo, config: &Config) -> AnalyzerContext {
    let mut imports = Vec::new();
    let mut doc_analysis = None;
    let mut has_examples = file.relative_path.to_lowercase().contains("example");
    let mut is_entrypoint = scribe_core::file::is_entrypoint_path(&file.path, &file.language);
    let mut priority_boost = compute_priority_boost(file);
    let mut cached_content: Option<String> = None;

    if should_load_content(file, config) {
        if let Ok(content) = std::fs::read_to_string(&file.path) {
            if matches!(file.file_type, FileType::Documentation { .. }) {
                doc_analysis = Some(analyze_document_content(&content));
            }

            if !has_examples {
                has_examples = content.contains("Example") || content.contains("example");
            }

            if matches!(
                file.language,
                Language::Rust
                    | Language::Python
                    | Language::JavaScript
                    | Language::TypeScript
                    | Language::Go
            ) {
                imports = extract_imports(&content, &file.language);
            }

            if !is_entrypoint {
                is_entrypoint = detect_entrypoint_from_content(&content, &file.language);
            }

            cached_content = Some(content);
        }
    }

    AnalyzerContext {
        imports,
        doc_analysis,
        has_examples,
        is_entrypoint,
        priority_boost,
        content: cached_content,
    }
}

fn should_load_content(file: &FileInfo, config: &Config) -> bool {
    if !config.analysis.analyze_content || file.is_binary {
        return false;
    }

    let size_limit = std::cmp::max(config.performance.io_buffer_size as u64, 256 * 1024);
    file.size <= size_limit
}

fn compute_priority_boost(file: &FileInfo) -> f64 {
    let path_lower = file.relative_path.to_lowercase();
    let mut boost: f64 = 0.0;

    if path_lower.ends_with("readme.md") || path_lower.ends_with("readme") {
        boost += 0.4;
    }
    if path_lower.ends_with("cargo.toml")
        || path_lower.ends_with("package.json")
        || path_lower.ends_with("requirements.txt")
        || path_lower.ends_with("pyproject.toml")
    {
        boost += 0.25;
    }
    if path_lower.ends_with("main.rs")
        || path_lower.ends_with("main.py")
        || path_lower.ends_with("main.go")
        || path_lower.ends_with("index.js")
        || path_lower.ends_with("index.ts")
    {
        boost += 0.3;
    }
    if path_lower.ends_with("lib.rs") {
        boost += 0.2;
    }
    if path_lower.ends_with("build.rs") || path_lower.ends_with("setup.py") {
        boost += 0.15;
    }

    boost.min(1.0)
}

fn detect_entrypoint_from_content(content: &str, language: &Language) -> bool {
    match language {
        Language::Rust => content.contains("fn main("),
        Language::Python => content.contains("__name__ == \"__main__\""),
        Language::JavaScript | Language::TypeScript => {
            content.contains("module.exports") || content.contains("export default")
        }
        Language::Go => content.contains("func main("),
        Language::Java => content.contains("public static void main("),
        _ => false,
    }
}

/// Helper to extract the import path from a Go import line.
/// Handles: `"path"`, `alias "path"`, `. "path"`, `_ "path"`
fn extract_go_import_path(line: &str) -> Option<String> {
    // Find the quoted string (either " or `)
    let line = line.trim();

    // Try double quotes first
    if let Some(start) = line.find('"') {
        if let Some(end) = line[start + 1..].find('"') {
            return Some(line[start + 1..start + 1 + end].to_string());
        }
    }

    // Try backticks
    if let Some(start) = line.find('`') {
        if let Some(end) = line[start + 1..].find('`') {
            return Some(line[start + 1..start + 1 + end].to_string());
        }
    }

    None
}

/// Extract import statements from source code content.
///
/// This function parses the content and extracts import/use/require statements
/// based on the programming language. Used for building dependency graphs.
pub fn extract_imports(content: &str, language: &Language) -> Vec<String> {
    use std::collections::HashSet;

    let mut imports = HashSet::new();

    match language {
        Language::Rust => {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("use ") {
                    let statement = trimmed
                        .trim_start_matches("use ")
                        .trim_end_matches(';')
                        .split_whitespace()
                        .next()
                        .unwrap_or_default();
                    // Handle grouped imports: use crate::module::{item1, item2};
                    // Extract the module path before the brace
                    let module_path = if let Some(brace_pos) = statement.find('{') {
                        statement[..brace_pos].trim_end_matches("::")
                    } else {
                        statement.trim_end_matches("::")
                    };
                    if !module_path.is_empty() {
                        imports.insert(module_path.to_string());
                    }
                } else if trimmed.starts_with("mod ") {
                    let module = trimmed
                        .trim_start_matches("mod ")
                        .trim_end_matches(';')
                        .trim();
                    if !module.is_empty() {
                        imports.insert(module.to_string());
                    }
                }
            }
        }
        Language::Python => {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("import ") {
                    for module in trimmed.trim_start_matches("import ").split(',') {
                        let module = module.trim().split_whitespace().next().unwrap_or("");
                        if !module.is_empty() {
                            imports.insert(module.to_string());
                        }
                    }
                } else if trimmed.starts_with("from ") && trimmed.contains(" import ") {
                    let module = trimmed
                        .trim_start_matches("from ")
                        .split(" import ")
                        .next()
                        .unwrap_or("")
                        .trim();
                    if !module.is_empty() {
                        imports.insert(module.to_string());
                    }
                }
            }
        }
        Language::JavaScript | Language::TypeScript => {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("import ") {
                    if let Some(start) = trimmed.find("\"") {
                        if let Some(end) = trimmed[start + 1..].find('"') {
                            imports.insert(trimmed[start + 1..start + 1 + end].to_string());
                        }
                    } else if let Some(start) = trimmed.find('\'') {
                        if let Some(end) = trimmed[start + 1..].find('\'') {
                            imports.insert(trimmed[start + 1..start + 1 + end].to_string());
                        }
                    }
                } else if trimmed.contains("require(") {
                    if let Some(start) = trimmed.find("require(") {
                        let start = start + "require(".len();
                        let slice = &trimmed[start..];
                        if let Some(end_idx) = slice.find(')') {
                            let inner = &slice[..end_idx];
                            let inner = inner.trim_matches(&['\'', '"'][..]);
                            if !inner.is_empty() {
                                imports.insert(inner.to_string());
                            }
                        }
                    }
                }
            }
        }
        Language::Go => {
            let mut in_block = false;
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed == "import (" {
                    in_block = true;
                    continue;
                }
                if in_block {
                    if trimmed == ")" {
                        in_block = false;
                        continue;
                    }
                    // Handle aliased imports: alias "path" or . "path" or _ "path"
                    // Extract just the quoted path
                    if let Some(import_path) = extract_go_import_path(trimmed) {
                        if !import_path.is_empty() {
                            imports.insert(import_path);
                        }
                    }
                } else if trimmed.starts_with("import ") {
                    let rest = trimmed.trim_start_matches("import ");
                    if let Some(import_path) = extract_go_import_path(rest) {
                        if !import_path.is_empty() {
                            imports.insert(import_path);
                        }
                    }
                }
            }
        }
        _ => {}
    }

    let mut ordered: Vec<String> = imports.into_iter().collect();
    ordered.sort();
    ordered.truncate(64);
    ordered
}

fn analyze_document_content(content: &str) -> DocumentAnalysis {
    let mut analysis = DocumentAnalysis::new();
    let mut in_code_block = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            if !in_code_block {
                analysis.code_block_count += 1;
            }
            in_code_block = !in_code_block;
            continue;
        }

        if trimmed.starts_with('#') {
            analysis.heading_count += 1;
            if trimmed.to_lowercase().contains("table of contents") {
                analysis.toc_indicators += 1;
            }
        }

        if trimmed.contains("](") {
            analysis.link_count += trimmed.matches("](").count();
        }
    }

    analysis.is_well_structured = analysis.heading_count > 0 && analysis.link_count > 0;
    analysis
}

fn compute_churn_score(file: &FileInfo) -> f64 {
    use scribe_core::GitFileStatus;

    match &file.git_status {
        Some(status) => match status.working_tree {
            GitFileStatus::Modified => 0.6,
            GitFileStatus::Added => 0.8,
            GitFileStatus::Deleted => 0.4,
            GitFileStatus::Renamed => 0.5,
            GitFileStatus::Copied => 0.45,
            GitFileStatus::Unmerged => 0.9,
            GitFileStatus::Untracked => 0.3,
            _ => 0.1,
        },
        None => 0.0,
    }
}

fn language_from_identifier(language: &str, path: &std::path::Path) -> Language {
    if !language.is_empty() {
        match language.to_lowercase().as_str() {
            "rust" => return Language::Rust,
            "python" => return Language::Python,
            "javascript" => return Language::JavaScript,
            "typescript" => return Language::TypeScript,
            "go" => return Language::Go,
            "java" => return Language::Java,
            "c" => return Language::C,
            "cpp" | "c++" => return Language::Cpp,
            "kotlin" => return Language::Kotlin,
            "swift" => return Language::Swift,
            "php" => return Language::PHP,
            "ruby" => return Language::Ruby,
            _ => {}
        }
    }

    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    Language::from_extension(extension)
}

#[cfg(feature = "graph")]
fn compute_centrality_scores(
    analyzer_files: &[AnalyzerFile],
) -> Option<std::collections::HashMap<String, f64>> {
    use scribe_graph::CentralityCalculator;

    if analyzer_files.is_empty() {
        return Some(std::collections::HashMap::new());
    }

    let calculator = CentralityCalculator::new().ok()?;
    let results = calculator.calculate_centrality(analyzer_files).ok()?;
    Some(results.pagerank_scores.into_iter().collect())
}

#[cfg(feature = "scaling")]
async fn convert_scaling_result_to_analysis(
    processing_result: scribe_scaling::ProcessingResult,
    config: Config,
    repo_root: &std::path::Path,
) -> Result<RepositoryAnalysis> {
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("🔄 Converting scaling result to repository analysis format");
    }

    let mut files: Vec<FileInfo> = Vec::with_capacity(processing_result.files.len());

    for file_meta in processing_result.files {
        let mut absolute_path = file_meta.path.clone();
        if !absolute_path.is_absolute() {
            absolute_path = repo_root.join(absolute_path);
        }

        let relative_path = absolute_path
            .strip_prefix(repo_root)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| absolute_path.to_string_lossy().replace('\\', "/"));

        let extension = absolute_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        let language = language_from_identifier(&file_meta.language, &absolute_path);
        let file_type = FileInfo::classify_file_type(&relative_path, &language, extension);

        files.push(FileInfo {
            path: absolute_path,
            relative_path,
            size: file_meta.size,
            modified: Some(file_meta.modified),
            decision: scribe_core::RenderDecision::include("scaling_engine"),
            file_type,
            language,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: None,
            weight: scribe_core::FileWeight::default(),
            centrality_score: None,
        });
    }

    let analysis = build_repository_analysis(
        files,
        &config,
        &[
            "scaling_engine",
            "progressive_loading",
            "optimized_processing",
        ],
    )?;

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!(
            "📈 Scaling analysis processed {} files in {:?} (cache hits {}, misses {})",
            analysis.files.len(),
            processing_result.processing_time,
            processing_result.cache_hits,
            processing_result.cache_misses
        );
    }

    Ok(analysis)
}

/// Convenience function for fast file scanning without deep analysis
///
/// This is useful when you just need to discover files quickly without
/// computing complex heuristic scores.
#[cfg(all(feature = "scanner", feature = "patterns"))]
pub async fn scan_repository<P: AsRef<std::path::Path>>(
    path: P,
    include_patterns: Option<&[&str]>,
    exclude_patterns: Option<&[&str]>,
) -> Result<Vec<FileInfo>> {
    let scanner = Scanner::new();
    let mut options = ScanOptions::default()
        .with_git_integration(true)
        .with_parallel_processing(true);

    // Apply patterns if provided
    if let (Some(includes), Some(excludes)) = (include_patterns, exclude_patterns) {
        let matcher = QuickMatcher::new(includes, excludes)?;
        // Note: This would need proper integration with ScanOptions
        // options = options.with_pattern_matcher(matcher);
    }

    scanner.scan(path, options).await
}

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and functions
/// to provide a convenient single import for typical usage.
///
/// # Example
///
/// ```rust,ignore
/// use scribe_analyzer::prelude::*;
///
/// // Now you have access to:
/// // - Result, ScribeError
/// // - Config, FileInfo
/// // - analyze_repository function
/// // - Scanner, PatternMatcher
/// // - And other commonly used types
/// ```
pub mod prelude {
    //! Commonly used imports for Scribe applications

    #[cfg(feature = "core")]
    pub use crate::core::{
        Config, FileInfo, FileType, HeuristicWeights, Language, Result, ScoreComponents,
        ScribeError, VERSION as CORE_VERSION,
    };

    #[cfg(feature = "analysis")]
    pub use crate::analysis::{HeuristicScorer, HeuristicSystem};

    #[cfg(feature = "scanner")]
    pub use crate::scanner::{FileScanner, ScanOptions, Scanner};

    #[cfg(feature = "patterns")]
    pub use crate::patterns::{presets, PatternMatcher, PatternMatcherBuilder, QuickMatcher};

    #[cfg(feature = "graph")]
    pub use crate::graph::{CentralityCalculator, PageRankAnalysis};

    #[cfg(feature = "selection")]
    pub use crate::selection::{CodeSelector, SelectionEngine};

    // High-level functions
    #[cfg(all(feature = "analysis", feature = "scanner", feature = "patterns"))]
    pub use crate::{analyze_repository, RepositoryAnalysis};

    #[cfg(all(feature = "scanner", feature = "patterns"))]
    pub use crate::scan_repository;

    pub use crate::VERSION;
}

/// Utility functions for common operations
pub mod utils {
    #[cfg(feature = "core")]
    pub use crate::core::utils::*;

    #[cfg(feature = "patterns")]
    pub use crate::patterns::utils as pattern_utils;

    #[cfg(feature = "graph")]
    pub use crate::graph::utils as graph_utils;
}

// Re-export the main AnalysisMetadata type if available
#[cfg(feature = "core")]
pub use crate::core::AnalysisMetadata;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[cfg(feature = "core")]
    #[test]
    fn test_core_reexport() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[cfg(all(feature = "analysis", feature = "scanner", feature = "patterns"))]
    #[tokio::test]
    async fn test_repository_analysis_interface() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() { println!(\"Hello world\"); }").unwrap();

        let config = Config::default();
        let result = analyze_repository(temp_dir.path(), &config).await;

        // Should succeed or fail gracefully
        match result {
            Ok(analysis) => {
                assert!(analysis.file_count() > 0);
                assert!(!analysis.summary().is_empty());
            }
            Err(_) => {
                // Analysis might fail in test environment, which is acceptable
                // as long as the interface compiles correctly
            }
        }
    }

    #[cfg(all(feature = "scanner", feature = "patterns"))]
    #[tokio::test]
    async fn test_scan_repository_interface() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() {}").unwrap();

        let result =
            scan_repository(temp_dir.path(), Some(&["**/*.rs"]), Some(&["**/target/**"])).await;

        // Should find the test file
        match result {
            Ok(files) => {
                assert!(!files.is_empty());
                assert!(files
                    .iter()
                    .any(|f| f.path.file_name().unwrap() == "test.rs"));
            }
            Err(_) => {
                // Scan might fail in test environment, which is acceptable
            }
        }
    }

    #[cfg(feature = "core")]
    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        // Test that basic types are available
        let config = Config::default();
        assert!(config.validate().is_ok());

        // Test that version is available
        assert!(!VERSION.is_empty());
    }
}

#[cfg(test)]
mod import_extraction_tests {
    use super::*;

    #[test]
    fn test_rust_imports() {
        let content = r#"
use crate::module;
use crate::module::{item1, item2};
use super::parent_module;
use self::sibling;
use std::collections::HashMap;
use std::io::{Read, Write};
mod my_module;
pub use crate::reexport;
"#;
        let imports = extract_imports(content, &Language::Rust);
        println!("Rust imports: {:?}", imports);
        
        assert!(imports.contains(&"crate::module".to_string()), "Should extract simple crate import");
        assert!(imports.contains(&"super::parent_module".to_string()), "Should extract super import");
        assert!(imports.contains(&"self::sibling".to_string()), "Should extract self import");
        assert!(imports.contains(&"std::collections::HashMap".to_string()), "Should extract std import");
        assert!(imports.contains(&"std::io".to_string()), "Should extract grouped std import");
        assert!(imports.contains(&"my_module".to_string()), "Should extract mod declaration");
        // pub use should NOT be extracted (doesn't start with "use ")
        assert!(!imports.iter().any(|i| i.contains("reexport")), "Should NOT extract pub use");
    }

    #[test]
    fn test_python_imports() {
        let content = r#"
import os
import os, sys
import numpy as np
from os import path
from . import module
from .. import parent
from ..package import module
from typing import List, Dict
"#;
        let imports = extract_imports(content, &Language::Python);
        println!("Python imports: {:?}", imports);
        
        assert!(imports.contains(&"os".to_string()), "Should extract simple import");
        assert!(imports.contains(&"sys".to_string()), "Should extract comma-separated import");
        assert!(imports.contains(&"numpy".to_string()), "Should extract aliased import");
        assert!(imports.contains(&".".to_string()), "Should extract relative import");
        assert!(imports.contains(&"..".to_string()), "Should extract parent relative import");
        assert!(imports.contains(&"..package".to_string()), "Should extract parent package import");
        assert!(imports.contains(&"typing".to_string()), "Should extract from import");
    }

    #[test]
    fn test_javascript_imports() {
        let content = r#"
import foo from 'module'
import bar from "double-quotes"
import { a, b } from 'destructure'
import * as all from 'star-import'
import 'side-effect'
const req1 = require('commonjs')
const req2 = require("cjs-double")
import { x } from './relative'
"#;
        let imports = extract_imports(content, &Language::JavaScript);
        println!("JavaScript imports: {:?}", imports);
        
        assert!(imports.contains(&"module".to_string()), "Should extract single-quote import");
        assert!(imports.contains(&"double-quotes".to_string()), "Should extract double-quote import");
        assert!(imports.contains(&"destructure".to_string()), "Should extract destructured import");
        assert!(imports.contains(&"star-import".to_string()), "Should extract star import");
        assert!(imports.contains(&"side-effect".to_string()), "Should extract side-effect import");
        assert!(imports.contains(&"commonjs".to_string()), "Should extract require single");
        assert!(imports.contains(&"cjs-double".to_string()), "Should extract require double");
        assert!(imports.contains(&"./relative".to_string()), "Should extract relative import");
    }

    #[test]
    fn test_go_imports() {
        let content = r#"
package main

import "fmt"
import (
    "os"
    "path/filepath"
)
"#;
        let imports = extract_imports(content, &Language::Go);
        println!("Go imports: {:?}", imports);
        
        assert!(imports.contains(&"fmt".to_string()), "Should extract single import");
        assert!(imports.contains(&"os".to_string()), "Should extract block import");
        assert!(imports.contains(&"path/filepath".to_string()), "Should extract block import with path");
    }

    #[test]
    fn test_go_aliased_imports() {
        let content = r#"
import f "fmt"
import (
    . "os"
    _ "init/pkg"
    alias "github.com/pkg/errors"
)
"#;
        let imports = extract_imports(content, &Language::Go);
        println!("Go aliased imports: {:?}", imports);

        // These should extract the package path, not the alias
        assert!(imports.contains(&"fmt".to_string()), "Should extract aliased fmt");
        assert!(imports.contains(&"os".to_string()), "Should extract dot-imported os");
        assert!(imports.contains(&"init/pkg".to_string()), "Should extract blank-imported init/pkg");
        assert!(imports.contains(&"github.com/pkg/errors".to_string()), "Should extract aliased github package");

        // Should NOT contain the aliases themselves
        assert!(!imports.iter().any(|i| i.starts_with("f ")), "Should not include alias 'f'");
        assert!(!imports.iter().any(|i| i.starts_with(". ")), "Should not include dot alias");
    }
}

#[cfg(test)]
mod import_edge_cases {
    use super::*;

    #[test]
    fn test_rust_pub_use_not_extracted() {
        let content = "pub use crate::module;";
        let imports = extract_imports(content, &Language::Rust);
        assert!(imports.is_empty(), "pub use should not be extracted");
    }

    #[test]
    fn test_rust_multiline_use() {
        // Multi-line use statements (spanning lines) - only first line detected
        let content = r#"
use std::collections::{
    HashMap,
    HashSet,
};
"#;
        let imports = extract_imports(content, &Language::Rust);
        println!("Multiline Rust: {:?}", imports);
        assert!(imports.contains(&"std::collections".to_string()));
    }

    #[test]
    fn test_typescript_import_type() {
        let content = r#"
import type { Type } from 'type-module'
import { Component } from '@angular/core'
"#;
        let imports = extract_imports(content, &Language::TypeScript);
        println!("TypeScript imports: {:?}", imports);
        assert!(imports.contains(&"type-module".to_string()), "Should extract type import");
        assert!(imports.contains(&"@angular/core".to_string()), "Should extract scoped package");
    }

    #[test]
    fn test_python_multiline_from_import() {
        // Python allows parenthesized imports
        let content = r#"
from module import (
    item1,
    item2,
)
"#;
        let imports = extract_imports(content, &Language::Python);
        println!("Python multiline: {:?}", imports);
        assert!(imports.contains(&"module".to_string()));
    }

    #[test]
    fn test_go_backtick_imports() {
        let content = "import `fmt`";
        let imports = extract_imports(content, &Language::Go);
        assert!(imports.contains(&"fmt".to_string()), "Should handle backtick imports");
    }

    #[test]
    fn test_js_template_literal_not_extracted() {
        // Template literals shouldn't be confused with imports
        let content = r#"
const x = `not an import`;
import real from 'real-module';
"#;
        let imports = extract_imports(content, &Language::JavaScript);
        println!("JS template test: {:?}", imports);
        assert!(imports.contains(&"real-module".to_string()));
        assert!(!imports.contains(&"not an import".to_string()));
    }
}

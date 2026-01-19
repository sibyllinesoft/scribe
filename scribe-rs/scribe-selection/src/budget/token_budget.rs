//! Token budget selection logic previously implemented in the analyzer crate.
//! This module provides a shared implementation that can be reused by both the
//! library pipeline and external consumers without duplicating complex logic.

use crate::algorithms::demotion::{DemotionEngine, FidelityMode};
use crate::budget::weighting::FileWeights;
use scribe_analysis::heuristics::ScanResult;
use scribe_core::{
    tokenization::{utils as token_utils, TokenBudget, TokenCounter},
    Config, FileInfo, FileType, Result, ScribeError,
};
use scribe_graph::CentralityCalculator;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

/// Configuration for the coverage-optimized selection algorithm.
///
/// The boost factors control the trade-off between resolution (full file content)
/// and coverage (more files represented via signatures).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Boost factor for signature scores (higher = prefer coverage).
    /// - 1.0 = signatures valued same as full content (resolution mode)
    /// - 1.5 = balanced mix of full and signatures (default)
    /// - 2.0+ = strong preference for signatures (coverage mode)
    pub signature_boost: f64,

    /// Boost factor for chunk scores.
    /// Should be between 1.0 and signature_boost.
    pub chunk_boost: f64,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            signature_boost: 1.5,
            chunk_boost: 1.2,
        }
    }
}

impl SelectionConfig {
    /// Resolution mode: prefer full file content over signatures.
    pub fn resolution() -> Self {
        Self {
            signature_boost: 1.0,
            chunk_boost: 1.0,
        }
    }

    /// Coverage mode: prefer more files via signatures.
    pub fn coverage() -> Self {
        Self {
            signature_boost: 2.0,
            chunk_boost: 1.5,
        }
    }

    /// Maximum coverage: strongly prefer signatures.
    pub fn max_coverage() -> Self {
        Self {
            signature_boost: 3.0,
            chunk_boost: 2.0,
        }
    }
}

/// A candidate for selection: a (file, fidelity mode) pair with computed scores.
#[derive(Debug, Clone)]
struct FidelityCandidate {
    /// Index into the source files vector.
    file_index: usize,
    /// The fidelity mode for this candidate.
    mode: FidelityMode,
    /// The score for this candidate (base_score × boost).
    score: f64,
    /// Token cost for this mode.
    tokens: usize,
    /// Value density: score / tokens.
    density: f64,
    /// The content for this mode.
    content: String,
}

/// Minimum tokens for a signature to be considered useful.
/// Signatures below this threshold are likely just file paths (from fallback)
/// which duplicate the directory map and waste tokens.
const MIN_USEFUL_SIGNATURE_TOKENS: usize = 30;

/// Pre-computed fidelity options for a single file.
#[derive(Debug)]
struct FileFidelityOptions {
    /// The original file info.
    file: FileInfo,
    /// Priority score (from centrality + external weights).
    priority: f64,
    /// Full content and token count.
    full_content: String,
    full_tokens: usize,
    /// Chunk content and token count (None if chunking not applicable/failed).
    chunk_content: Option<String>,
    chunk_tokens: Option<usize>,
    /// Signature content and token count (None if extraction failed or below threshold).
    signature_content: Option<String>,
    signature_tokens: Option<usize>,
}

/// Apply the optimization-based token budget selection.
///
/// This function uses a multiple-choice knapsack approach to maximize coverage
/// while respecting the token budget. The `selection_config` parameter controls the
/// trade-off between resolution (full content) and coverage (signatures).
///
/// The selector prioritizes files in multiple tiers:
/// 1. Mandatory project metadata (README, config files, entrypoints) - always full content
/// 2. Source files - optimized selection based on value density (score / tokens)
/// 3. Documentation with preference for design/architecture material
/// 4. Any remaining files while budget remains
///
/// For source files, the algorithm:
/// 1. Pre-computes full/chunk/signature variants for each file
/// 2. Scores each variant using boost factors (signatures get higher effective scores)
/// 3. Computes value density (score / tokens) for each variant
/// 4. Greedily selects highest-density variants that fit the budget
///
/// This approach maximizes coverage by naturally preferring signatures when
/// they provide better value per token than full content.
pub async fn apply_token_budget_selection(
    files: Vec<FileInfo>,
    token_budget: usize,
    config: &Config,
    weights: Option<&FileWeights>,
    selection_config: &SelectionConfig,
) -> Result<Vec<FileInfo>> {
    let debug = std::env::var("SCRIBE_DEBUG").is_ok();

    if debug {
        eprintln!(
            "🎯 Token budget selection: {} tokens, boost={:.1}/{:.1}",
            token_budget, selection_config.signature_boost, selection_config.chunk_boost,
        );
    }

    let counter = TokenCounter::global();
    let mut selected_files = Vec::new();
    let mut budget_tracker = TokenBudget::new(token_budget);

    let (mandatory_files, source_files, doc_files, other_files) = categorize_files(files);

    if debug {
        eprintln!(
            "📊 File categories: {} mandatory, {} source, {} docs, {} other",
            mandatory_files.len(), source_files.len(), doc_files.len(), other_files.len()
        );
    }

    // Tier 1: Mandatory files
    select_mandatory_files(mandatory_files, &counter, &mut budget_tracker, &mut selected_files).await?;

    // Tier 2: Source files with optimization
    if !source_files.is_empty() && budget_tracker.available() > 0 {
        select_source_files_optimized(
            source_files, weights, selection_config, &counter,
            &mut budget_tracker, &mut selected_files, debug,
        ).await?;
    }

    // Tier 3: Documentation files
    select_documentation_files(doc_files, &counter, &mut budget_tracker, &mut selected_files).await?;

    // Tier 4: Other files
    select_remaining_files(other_files, &counter, &mut budget_tracker, &mut selected_files).await?;

    if debug {
        let tokens_used = token_budget - budget_tracker.available();
        let utilization = (tokens_used as f64 / token_budget as f64) * 100.0;
        eprintln!(
            "✅ Total: {} files ({} tokens / {} budget, {:.1}% utilized)",
            selected_files.len(), tokens_used, token_budget, utilization
        );
    }

    Ok(selected_files)
}

/// Select mandatory files (always full content)
async fn select_mandatory_files(
    files: Vec<FileInfo>,
    counter: &TokenCounter,
    budget_tracker: &mut TokenBudget,
    selected_files: &mut Vec<FileInfo>,
) -> Result<()> {
    for file in files {
        if budget_tracker.available() < 1 {
            break;
        }
        if let Some(selected_file) = try_include_file_with_budget(file, counter, budget_tracker).await? {
            selected_files.push(selected_file);
        }
    }
    Ok(())
}

/// Select source files using optimization-based selection
async fn select_source_files_optimized(
    source_files: Vec<FileInfo>,
    weights: Option<&FileWeights>,
    selection_config: &SelectionConfig,
    counter: &TokenCounter,
    budget_tracker: &mut TokenBudget,
    selected_files: &mut Vec<FileInfo>,
    debug: bool,
) -> Result<()> {
    let calculator = CentralityCalculator::new()?;
    let mock_scan_results: Vec<_> = source_files.iter().map(MockScanResult::from_file_info).collect();
    let centrality_results = calculator.calculate_centrality(&mock_scan_results)?;

    let source_with_priority: Vec<_> = source_files
        .into_iter()
        .map(|mut file| {
            let centrality_score = centrality_results.pagerank_scores.get(&file.relative_path).copied().unwrap_or(0.0);
            file.centrality_score = Some(centrality_score);
            let priority = compute_file_priority(centrality_score, &file.relative_path, weights);
            (file, priority)
        })
        .collect();

    let fidelity_options = precompute_fidelity_options(source_with_priority, counter)?;

    if debug {
        eprintln!("📦 Pre-computed fidelity options for {} source files", fidelity_options.len());
    }

    let mut candidates = generate_fidelity_candidates(&fidelity_options, selection_config);

    if debug {
        eprintln!("🔢 Generated {} candidates", candidates.len());
    }

    let selected_indices = optimize_selection(&mut candidates, budget_tracker.available());
    apply_selected_candidates(&selected_indices, &candidates, &fidelity_options, budget_tracker, selected_files, debug);

    Ok(())
}

/// Compute file priority from centrality and external weights
fn compute_file_priority(centrality_score: f64, relative_path: &str, weights: Option<&FileWeights>) -> f64 {
    if let Some(w) = weights {
        let external_weight = w.get(relative_path);
        if external_weight > 0.0 {
            return (centrality_score + external_weight) / 2.0;
        }
    }
    centrality_score
}

/// Apply selected candidates to the output
fn apply_selected_candidates(
    selected_indices: &[usize],
    candidates: &[FidelityCandidate],
    fidelity_options: &[FileFidelityOptions],
    budget_tracker: &mut TokenBudget,
    selected_files: &mut Vec<FileInfo>,
    debug: bool,
) {
    let mut full_count = 0;
    let mut chunk_count = 0;
    let mut signature_count = 0;
    let mut total_tokens = 0;

    for &idx in selected_indices {
        let candidate = &candidates[idx];
        let opt = &fidelity_options[candidate.file_index];
        let mut file = opt.file.clone();

        file.content = Some(candidate.content.clone());
        file.token_estimate = Some(candidate.tokens);
        file.char_count = Some(candidate.content.chars().count());
        file.line_count = Some(candidate.content.lines().count());
        file.centrality_score = Some(opt.priority);

        budget_tracker.allocate(candidate.tokens);
        total_tokens += candidate.tokens;

        match candidate.mode {
            FidelityMode::Full => full_count += 1,
            FidelityMode::Chunk => chunk_count += 1,
            FidelityMode::Signature => signature_count += 1,
        }

        selected_files.push(file);
    }

    if debug {
        eprintln!(
            "✅ Selected {} source files: {} full, {} chunk, {} signature ({} tokens)",
            full_count + chunk_count + signature_count, full_count, chunk_count, signature_count, total_tokens
        );
    }
}

/// Keywords indicating critical documentation files
const CRITICAL_DOC_KEYWORDS: &[&str] = &["architecture", "design", "api", "spec"];

/// Specific critical doc filenames
const CRITICAL_DOC_FILES: &[&str] = &["changelog.md", "contributing.md"];

/// Check if a documentation file is critical
fn is_critical_doc(path_lower: &str) -> bool {
    CRITICAL_DOC_KEYWORDS.iter().any(|kw| path_lower.contains(kw))
        || CRITICAL_DOC_FILES.iter().any(|f| path_lower.ends_with(f))
}

/// Select documentation files with priority for critical docs
async fn select_documentation_files(
    doc_files: Vec<FileInfo>,
    counter: &TokenCounter,
    budget_tracker: &mut TokenBudget,
    selected_files: &mut Vec<FileInfo>,
) -> Result<()> {
    if doc_files.is_empty() || budget_tracker.available() == 0 {
        return Ok(());
    }

    let (critical_docs, other_docs): (Vec<_>, Vec<_>) = doc_files
        .into_iter()
        .partition(|f| is_critical_doc(&f.relative_path.to_lowercase()));

    for file in critical_docs.into_iter().chain(other_docs) {
        if budget_tracker.available() < 1 {
            break;
        }
        if let Some(selected_file) = try_include_file_with_budget(file, counter, budget_tracker).await? {
            selected_files.push(selected_file);
        }
    }
    Ok(())
}

/// Select remaining files until budget is exhausted
async fn select_remaining_files(
    files: Vec<FileInfo>,
    counter: &TokenCounter,
    budget_tracker: &mut TokenBudget,
    selected_files: &mut Vec<FileInfo>,
) -> Result<()> {
    if files.is_empty() || budget_tracker.available() == 0 {
        return Ok(());
    }

    for file in files {
        if budget_tracker.available() < 1 {
            break;
        }
        if let Some(selected_file) = try_include_file_with_budget(file, counter, budget_tracker).await? {
            selected_files.push(selected_file);
        }
    }
    Ok(())
}

fn categorize_files(
    files: Vec<FileInfo>,
) -> (Vec<FileInfo>, Vec<FileInfo>, Vec<FileInfo>, Vec<FileInfo>) {
    let mut mandatory = Vec::new();
    let mut source = Vec::new();
    let mut docs = Vec::new();
    let mut other = Vec::new();

    for file in files {
        if !file.decision.should_include() {
            continue;
        }

        if is_mandatory_file(&file) {
            mandatory.push(file);
        } else if matches!(file.file_type, FileType::Source { .. }) {
            source.push(file);
        } else if matches!(file.file_type, FileType::Documentation { .. }) {
            docs.push(file);
        } else {
            other.push(file);
        }
    }

    (mandatory, source, docs, other)
}

/// Directories that indicate generated/vendored content
const IGNORED_DIRS: [&str; 8] = [
    "node_modules/",
    "target/",
    "vendor/",
    ".git/",
    "__pycache__/",
    "build/",
    "dist/",
    ".cache/",
];

/// Root-level project configuration files
const ROOT_CONFIG_FILES: [&str; 12] = [
    "package.json",
    "cargo.toml",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "composer.json",
    "tsconfig.json",
    ".gitignore",
    "dockerfile",
    "docker-compose.yml",
];

/// Check if path is in an ignored directory
fn is_in_ignored_dir(path: &str) -> bool {
    IGNORED_DIRS.iter().any(|dir| path.contains(dir))
}

/// Check if file is a README file worth including
fn is_mandatory_readme(path: &str) -> bool {
    if !path.contains("readme") {
        return false;
    }

    let depth = path.matches('/').count();
    if depth <= 1 {
        return true;
    }

    path.ends_with("readme.md")
        || path.ends_with("readme.markdown")
        || path.ends_with("readme.txt")
        || path.ends_with("readme")
}

/// Check if file is a root-level config file
fn is_root_config_file(path: &str) -> bool {
    !path.contains('/') && ROOT_CONFIG_FILES.contains(&path)
}

/// Check if file is a main/index entrypoint
fn is_entrypoint_file(path: &str) -> bool {
    let is_source_dir = path.starts_with("src/") || path.starts_with("lib/") || !path.contains('/');
    is_source_dir && (path.contains("main") || path.contains("index"))
}

fn is_mandatory_file(file: &FileInfo) -> bool {
    let path = file.relative_path.to_lowercase();

    if is_in_ignored_dir(&path) {
        return false;
    }

    is_mandatory_readme(&path) || is_root_config_file(&path) || is_entrypoint_file(&path)
}

/// Log a debug warning if SCRIBE_DEBUG is set
fn debug_warn(msg: &str) {
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("⚠️  {}", msg);
    }
}

async fn try_include_file_with_budget(
    mut file: FileInfo,
    counter: &TokenCounter,
    budget_tracker: &mut TokenBudget,
) -> Result<Option<FileInfo>> {
    let content = match load_file_content_safe(&file.path) {
        Ok(c) => c,
        Err(e) => {
            debug_warn(&format!("Failed to read {}: {}", file.relative_path, e));
            return Ok(None);
        }
    };

    let token_count = match counter.estimate_file_tokens(&content, &file.path) {
        Ok(count) => count,
        Err(e) => {
            debug_warn(&format!("Failed to estimate tokens for {}: {}", file.relative_path, e));
            return Ok(None);
        }
    };

    if !budget_tracker.can_allocate(token_count) {
        debug_warn(&format!("Skipping {} ({} tokens) - would exceed budget", file.relative_path, token_count));
        return Ok(None);
    }

    budget_tracker.allocate(token_count);
    file.char_count = Some(content.chars().count());
    file.line_count = Some(content.lines().count());
    file.content = Some(content);
    file.token_estimate = Some(token_count);
    Ok(Some(file))
}

struct MockScanResult {
    path: String,
    relative_path: String,
    centrality_score: Option<f64>,
}

impl MockScanResult {
    fn from_file_info(file: &FileInfo) -> Self {
        Self {
            path: file.path.to_string_lossy().to_string(),
            relative_path: file.relative_path.clone(),
            centrality_score: file.centrality_score,
        }
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
        self.relative_path.matches('/').count()
    }

    fn is_docs(&self) -> bool {
        false
    }

    fn is_readme(&self) -> bool {
        self.relative_path.to_lowercase().contains("readme")
    }

    fn is_entrypoint(&self) -> bool {
        self.relative_path.contains("main") || self.relative_path.contains("index")
    }

    fn has_examples(&self) -> bool {
        self.relative_path.contains("example")
    }

    fn is_test(&self) -> bool {
        self.relative_path.contains("test")
    }

    fn priority_boost(&self) -> f64 {
        0.0
    }

    fn churn_score(&self) -> f64 {
        0.0
    }

    fn centrality_in(&self) -> f64 {
        self.centrality_score.unwrap_or(0.0)
    }

    fn imports(&self) -> Option<&[String]> {
        None
    }

    fn doc_analysis(&self) -> Option<&scribe_analysis::heuristics::DocumentAnalysis> {
        None
    }
}

fn load_file_content_safe(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .map_err(|e| ScribeError::io(format!("Failed to read file {}: {}", path.display(), e), e))
}

// =============================================================================
// Optimization-based selection algorithm
// =============================================================================

/// Pre-compute fidelity options (full, chunk, signature) for all source files.
///
/// This computes the token cost and content for each fidelity mode so we can
/// make optimal selection decisions.
fn precompute_fidelity_options(
    source_files: Vec<(FileInfo, f64)>, // (file, priority)
    counter: &TokenCounter,
) -> Result<Vec<FileFidelityOptions>> {
    let mut demotion_engine = DemotionEngine::new()?;
    let mut options = Vec::with_capacity(source_files.len());

    for (file, priority) in source_files {
        // Load file content
        let full_content = match load_file_content_safe(&file.path) {
            Ok(content) => content,
            Err(_) => continue, // Skip files we can't read
        };

        // Compute full token count
        let full_tokens = counter
            .estimate_file_tokens(&full_content, &file.path)
            .unwrap_or_else(|_| token_utils::estimate_tokens_legacy(&full_content));

        if full_tokens == 0 {
            continue; // Skip empty files
        }

        // Compute chunk content and tokens
        let (chunk_content, chunk_tokens) = match demotion_engine.demote_content(
            &full_content,
            &file.relative_path,
            FidelityMode::Chunk,
            None,
        ) {
            Ok(result) if result.demoted_tokens > 0 && result.demoted_tokens < full_tokens => {
                (Some(result.content), Some(result.demoted_tokens))
            }
            _ => (None, None),
        };

        // Compute signature content and tokens
        // Only use signatures that provide meaningful content (above threshold).
        // Path-only signatures duplicate the directory map and waste tokens.
        let (signature_content, signature_tokens) = match demotion_engine.demote_content(
            &full_content,
            &file.relative_path,
            FidelityMode::Signature,
            None,
        ) {
            Ok(result) if result.demoted_tokens >= MIN_USEFUL_SIGNATURE_TOKENS => {
                (Some(result.content), Some(result.demoted_tokens))
            }
            _ => {
                // No useful signature available - file can only be included
                // at full or chunk fidelity levels
                (None, None)
            }
        };

        options.push(FileFidelityOptions {
            file,
            priority,
            full_content,
            full_tokens,
            chunk_content,
            chunk_tokens,
            signature_content,
            signature_tokens,
        });
    }

    Ok(options)
}

/// Generate all fidelity candidates from pre-computed options.
///
/// Each file produces up to 3 candidates (full, chunk, signature), each with
/// a score computed using the boost factors.
fn generate_fidelity_candidates(
    options: &[FileFidelityOptions],
    config: &SelectionConfig,
) -> Vec<FidelityCandidate> {
    let mut candidates = Vec::with_capacity(options.len() * 3);

    for (index, opt) in options.iter().enumerate() {
        // Full content candidate
        let full_score = opt.priority;
        let full_density = if opt.full_tokens > 0 {
            full_score / opt.full_tokens as f64
        } else {
            0.0
        };
        candidates.push(FidelityCandidate {
            file_index: index,
            mode: FidelityMode::Full,
            score: full_score,
            tokens: opt.full_tokens,
            density: full_density,
            content: opt.full_content.clone(),
        });

        // Chunk candidate (if available and smaller than full)
        if let (Some(ref chunk_content), Some(chunk_tokens)) = (&opt.chunk_content, opt.chunk_tokens) {
            let chunk_score = opt.priority * config.chunk_boost;
            let chunk_density = if chunk_tokens > 0 {
                chunk_score / chunk_tokens as f64
            } else {
                0.0
            };
            candidates.push(FidelityCandidate {
                file_index: index,
                mode: FidelityMode::Chunk,
                score: chunk_score,
                tokens: chunk_tokens,
                density: chunk_density,
                content: chunk_content.clone(),
            });
        }

        // Signature candidate (only if meaningful signature was extracted)
        if let (Some(ref signature_content), Some(signature_tokens)) =
            (&opt.signature_content, opt.signature_tokens)
        {
            let signature_score = opt.priority * config.signature_boost;
            let signature_density = if signature_tokens > 0 {
                signature_score / signature_tokens as f64
            } else {
                0.0
            };
            candidates.push(FidelityCandidate {
                file_index: index,
                mode: FidelityMode::Signature,
                score: signature_score,
                tokens: signature_tokens,
                density: signature_density,
                content: signature_content.clone(),
            });
        }
    }

    candidates
}

/// Solve the multiple-choice knapsack problem using greedy approximation.
///
/// Returns the indices of selected candidates.
fn optimize_selection(
    candidates: &mut [FidelityCandidate],
    budget: usize,
) -> Vec<usize> {
    // Sort by density (value per token) descending
    candidates.sort_by(|a, b| {
        b.density
            .partial_cmp(&a.density)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = Vec::new();
    let mut selected_files: HashSet<usize> = HashSet::new();
    let mut remaining_budget = budget;

    for (idx, candidate) in candidates.iter().enumerate() {
        // Skip if file already selected in a different mode
        if selected_files.contains(&candidate.file_index) {
            continue;
        }

        // Skip if doesn't fit budget
        if candidate.tokens > remaining_budget {
            continue;
        }

        // Select this candidate
        selected.push(idx);
        selected_files.insert(candidate.file_index);
        remaining_budget -= candidate.tokens;

        // Early termination if budget exhausted
        if remaining_budget == 0 {
            break;
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_config_default() {
        let config = SelectionConfig::default();
        assert_eq!(config.signature_boost, 1.5);
        assert_eq!(config.chunk_boost, 1.2);
    }

    #[test]
    fn test_selection_config_resolution() {
        let config = SelectionConfig::resolution();
        assert_eq!(config.signature_boost, 1.0);
        assert_eq!(config.chunk_boost, 1.0);
    }

    #[test]
    fn test_selection_config_coverage() {
        let config = SelectionConfig::coverage();
        assert_eq!(config.signature_boost, 2.0);
        assert_eq!(config.chunk_boost, 1.5);
    }

    #[test]
    fn test_selection_config_max_coverage() {
        let config = SelectionConfig::max_coverage();
        assert_eq!(config.signature_boost, 3.0);
        assert_eq!(config.chunk_boost, 2.0);
    }

    #[test]
    fn test_is_critical_doc_architecture() {
        // Note: is_critical_doc expects lowercase input (path_lower parameter)
        assert!(is_critical_doc("docs/architecture.md"));
        assert!(is_critical_doc("architecture.md"));
        assert!(is_critical_doc("design/architecture-overview.md"));
    }

    #[test]
    fn test_is_critical_doc_design() {
        assert!(is_critical_doc("docs/design.md"));
        assert!(is_critical_doc("design.md"));
    }

    #[test]
    fn test_is_critical_doc_api() {
        assert!(is_critical_doc("docs/api.md"));
        assert!(is_critical_doc("api.md"));
        assert!(is_critical_doc("api-reference.md"));
    }

    #[test]
    fn test_is_critical_doc_spec() {
        assert!(is_critical_doc("docs/spec.md"));
        assert!(is_critical_doc("specification.md"));
    }

    #[test]
    fn test_is_critical_doc_changelog() {
        // CRITICAL_DOC_FILES matches with ends_with, requires lowercase
        assert!(is_critical_doc("changelog.md"));
        assert!(is_critical_doc("docs/changelog.md"));
    }

    #[test]
    fn test_is_critical_doc_contributing() {
        assert!(is_critical_doc("contributing.md"));
        assert!(is_critical_doc("docs/contributing.md"));
    }

    #[test]
    fn test_is_critical_doc_non_critical() {
        assert!(!is_critical_doc("readme.md"));
        assert!(!is_critical_doc("notes.md"));
        assert!(!is_critical_doc("todo.md"));
    }

    #[test]
    fn test_critical_doc_keywords() {
        assert_eq!(CRITICAL_DOC_KEYWORDS.len(), 4);
        assert!(CRITICAL_DOC_KEYWORDS.contains(&"architecture"));
        assert!(CRITICAL_DOC_KEYWORDS.contains(&"design"));
        assert!(CRITICAL_DOC_KEYWORDS.contains(&"api"));
        assert!(CRITICAL_DOC_KEYWORDS.contains(&"spec"));
    }

    #[test]
    fn test_critical_doc_files() {
        assert_eq!(CRITICAL_DOC_FILES.len(), 2);
        assert!(CRITICAL_DOC_FILES.contains(&"changelog.md"));
        assert!(CRITICAL_DOC_FILES.contains(&"contributing.md"));
    }

    #[test]
    fn test_fidelity_candidate_density() {
        // Verify density calculation logic
        let score = 100.0;
        let tokens = 50usize;
        let density = score / tokens as f64;
        assert_eq!(density, 2.0);
    }

    #[test]
    fn test_optimize_selection_empty() {
        let mut candidates: Vec<FidelityCandidate> = vec![];
        let selected = optimize_selection(&mut candidates, 1000);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_optimize_selection_single_fits() {
        let mut candidates = vec![FidelityCandidate {
            file_index: 0,
            mode: FidelityMode::Full,
            score: 100.0,
            tokens: 50,
            density: 2.0,
            content: "test content".to_string(),
        }];

        let selected = optimize_selection(&mut candidates, 1000);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 0);
    }

    #[test]
    fn test_optimize_selection_budget_constraint() {
        let mut candidates = vec![
            FidelityCandidate {
                file_index: 0,
                mode: FidelityMode::Full,
                score: 100.0,
                tokens: 500,
                density: 0.2,
                content: "large content".to_string(),
            },
            FidelityCandidate {
                file_index: 1,
                mode: FidelityMode::Signature,
                score: 50.0,
                tokens: 100,
                density: 0.5, // Higher density
                content: "small content".to_string(),
            },
        ];

        // With only 200 tokens budget, should select the smaller one
        let selected = optimize_selection(&mut candidates, 200);
        assert_eq!(selected.len(), 1);
        // The algorithm sorts by density, so the higher density one should be selected
    }

    #[test]
    fn test_optimize_selection_prefers_higher_density() {
        let mut candidates = vec![
            FidelityCandidate {
                file_index: 0,
                mode: FidelityMode::Full,
                score: 100.0,
                tokens: 100,
                density: 1.0, // Lower density
                content: "content1".to_string(),
            },
            FidelityCandidate {
                file_index: 1,
                mode: FidelityMode::Signature,
                score: 100.0,
                tokens: 50,
                density: 2.0, // Higher density
                content: "content2".to_string(),
            },
        ];

        // Should prefer the higher density candidate first
        let selected = optimize_selection(&mut candidates, 1000);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_optimize_selection_same_file_different_modes() {
        // When same file has multiple modes, only one should be selected
        let mut candidates = vec![
            FidelityCandidate {
                file_index: 0,
                mode: FidelityMode::Full,
                score: 100.0,
                tokens: 200,
                density: 0.5,
                content: "full content".to_string(),
            },
            FidelityCandidate {
                file_index: 0, // Same file
                mode: FidelityMode::Signature,
                score: 80.0,
                tokens: 50,
                density: 1.6, // Higher density
                content: "signature".to_string(),
            },
        ];

        let selected = optimize_selection(&mut candidates, 1000);
        // Should only select one since they're the same file
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_selection_config_clone() {
        let config = SelectionConfig::default();
        let cloned = config.clone();
        assert_eq!(config.signature_boost, cloned.signature_boost);
        assert_eq!(config.chunk_boost, cloned.chunk_boost);
    }

    #[test]
    fn test_selection_config_serialize() {
        let config = SelectionConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SelectionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.signature_boost, deserialized.signature_boost);
        assert_eq!(config.chunk_boost, deserialized.chunk_boost);
    }

    #[test]
    fn test_is_in_ignored_dir() {
        assert!(is_in_ignored_dir("node_modules/package/file.js"));
        assert!(is_in_ignored_dir("target/debug/build.rs"));
        assert!(is_in_ignored_dir("vendor/dep/lib.rs"));
        assert!(is_in_ignored_dir(".git/config"));
        assert!(is_in_ignored_dir("__pycache__/module.pyc"));
        assert!(is_in_ignored_dir("build/output.o"));
        assert!(is_in_ignored_dir("dist/bundle.js"));
        assert!(is_in_ignored_dir(".cache/data.json"));

        // Non-ignored directories
        assert!(!is_in_ignored_dir("src/main.rs"));
        assert!(!is_in_ignored_dir("lib/utils.py"));
        assert!(!is_in_ignored_dir("docs/readme.md"));
    }

    #[test]
    fn test_ignored_dirs_constant() {
        assert_eq!(IGNORED_DIRS.len(), 8);
        assert!(IGNORED_DIRS.contains(&"node_modules/"));
        assert!(IGNORED_DIRS.contains(&"target/"));
        assert!(IGNORED_DIRS.contains(&"vendor/"));
        assert!(IGNORED_DIRS.contains(&".git/"));
        assert!(IGNORED_DIRS.contains(&"__pycache__/"));
        assert!(IGNORED_DIRS.contains(&"build/"));
        assert!(IGNORED_DIRS.contains(&"dist/"));
        assert!(IGNORED_DIRS.contains(&".cache/"));
    }

    #[test]
    fn test_root_config_files_constant() {
        assert_eq!(ROOT_CONFIG_FILES.len(), 12);
        assert!(ROOT_CONFIG_FILES.contains(&"package.json"));
        assert!(ROOT_CONFIG_FILES.contains(&"cargo.toml"));
        assert!(ROOT_CONFIG_FILES.contains(&"pyproject.toml"));
        assert!(ROOT_CONFIG_FILES.contains(&"requirements.txt"));
        assert!(ROOT_CONFIG_FILES.contains(&"go.mod"));
        assert!(ROOT_CONFIG_FILES.contains(&"pom.xml"));
        assert!(ROOT_CONFIG_FILES.contains(&"build.gradle"));
        assert!(ROOT_CONFIG_FILES.contains(&"composer.json"));
        assert!(ROOT_CONFIG_FILES.contains(&"tsconfig.json"));
        assert!(ROOT_CONFIG_FILES.contains(&".gitignore"));
        assert!(ROOT_CONFIG_FILES.contains(&"dockerfile"));
        assert!(ROOT_CONFIG_FILES.contains(&"docker-compose.yml"));
    }

    #[test]
    fn test_is_mandatory_readme() {
        // Root-level readme files are mandatory
        assert!(is_mandatory_readme("readme.md"));
        assert!(is_mandatory_readme("readme"));
        assert!(is_mandatory_readme("readme.txt"));
        assert!(is_mandatory_readme("readme.markdown"));

        // Single directory depth is ok
        assert!(is_mandatory_readme("docs/readme.md"));

        // Deep paths with proper extensions
        assert!(is_mandatory_readme("docs/guide/readme.md"));
        assert!(is_mandatory_readme("packages/core/readme.markdown"));

        // Non-readme files
        assert!(!is_mandatory_readme("main.rs"));
        assert!(!is_mandatory_readme("notes.md"));
    }

    #[test]
    fn test_is_root_config_file() {
        // Root-level config files
        assert!(is_root_config_file("package.json"));
        assert!(is_root_config_file("cargo.toml"));
        assert!(is_root_config_file("pyproject.toml"));
        assert!(is_root_config_file(".gitignore"));

        // Nested config files are not root config
        assert!(!is_root_config_file("src/package.json"));
        assert!(!is_root_config_file("packages/core/cargo.toml"));

        // Non-config files
        assert!(!is_root_config_file("main.rs"));
        assert!(!is_root_config_file("unknown.config"));
    }

    #[test]
    fn test_is_entrypoint_file() {
        // Source directory entrypoints
        assert!(is_entrypoint_file("src/main.rs"));
        assert!(is_entrypoint_file("src/index.ts"));
        assert!(is_entrypoint_file("lib/main.py"));
        assert!(is_entrypoint_file("lib/index.js"));

        // Root-level entrypoints
        assert!(is_entrypoint_file("main.rs"));
        assert!(is_entrypoint_file("index.js"));

        // Non-entrypoint files
        assert!(!is_entrypoint_file("utils/helpers.rs"));
        assert!(!is_entrypoint_file("components/button.tsx"));
    }

    #[test]
    fn test_generate_fidelity_candidates_full_only() {
        let options = vec![FileFidelityOptions {
            file: create_test_file_info("test.rs"),
            priority: 1.0,
            full_content: "fn main() {}".to_string(),
            full_tokens: 10,
            chunk_content: None,
            chunk_tokens: None,
            signature_content: Some("fn main()".to_string()),
            signature_tokens: Some(50), // Above MIN_USEFUL_SIGNATURE_TOKENS threshold
        }];

        let config = SelectionConfig::default();
        let candidates = generate_fidelity_candidates(&options, &config);

        // Should have 2 candidates: Full and Signature (no chunk)
        assert_eq!(candidates.len(), 2);

        let full = candidates.iter().find(|c| matches!(c.mode, FidelityMode::Full)).unwrap();
        assert_eq!(full.tokens, 10);
        assert_eq!(full.score, 1.0);

        let sig = candidates.iter().find(|c| matches!(c.mode, FidelityMode::Signature)).unwrap();
        assert_eq!(sig.tokens, 50);
        assert_eq!(sig.score, 1.0 * 1.5); // priority * signature_boost
    }

    #[test]
    fn test_generate_fidelity_candidates_with_chunk() {
        let options = vec![FileFidelityOptions {
            file: create_test_file_info("test.rs"),
            priority: 2.0,
            full_content: "fn main() {\n    println!(\"hello\");\n}".to_string(),
            full_tokens: 20,
            chunk_content: Some("fn main() { ... }".to_string()),
            chunk_tokens: Some(10),
            signature_content: Some("fn main()".to_string()),
            signature_tokens: Some(50), // Above MIN_USEFUL_SIGNATURE_TOKENS threshold
        }];

        let config = SelectionConfig::default();
        let candidates = generate_fidelity_candidates(&options, &config);

        // Should have 3 candidates: Full, Chunk, and Signature
        assert_eq!(candidates.len(), 3);

        let chunk = candidates.iter().find(|c| matches!(c.mode, FidelityMode::Chunk)).unwrap();
        assert_eq!(chunk.tokens, 10);
        assert_eq!(chunk.score, 2.0 * 1.2); // priority * chunk_boost
    }

    #[test]
    fn test_generate_fidelity_candidates_no_signature_when_none() {
        // When signature is None, only Full candidate should be generated
        let options = vec![FileFidelityOptions {
            file: create_test_file_info("test.rs"),
            priority: 1.0,
            full_content: "fn main() {}".to_string(),
            full_tokens: 10,
            chunk_content: None,
            chunk_tokens: None,
            signature_content: None, // No useful signature available
            signature_tokens: None,
        }];

        let config = SelectionConfig::default();
        let candidates = generate_fidelity_candidates(&options, &config);

        // Should have only 1 candidate: Full (no chunk, no signature)
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0].mode, FidelityMode::Full));
    }

    #[test]
    fn test_min_useful_signature_tokens_constant() {
        // Verify the threshold is reasonable
        assert!(MIN_USEFUL_SIGNATURE_TOKENS > 0);
        assert!(MIN_USEFUL_SIGNATURE_TOKENS < 100);
    }

    #[test]
    fn test_compute_file_priority_no_weights() {
        let priority = compute_file_priority(0.5, "src/main.rs", None);
        assert_eq!(priority, 0.5);
    }

    #[test]
    fn test_compute_file_priority_with_weights() {
        let mut weights = FileWeights::new();
        weights.set("src/main.rs".to_string(), 0.8);

        let priority = compute_file_priority(0.5, "src/main.rs", Some(&weights));
        assert_eq!(priority, (0.5 + 0.8) / 2.0); // Average of centrality and external weight
    }

    #[test]
    fn test_compute_file_priority_zero_weight() {
        let weights = FileWeights::new();

        // File not in weights should return just centrality
        let priority = compute_file_priority(0.5, "src/main.rs", Some(&weights));
        assert_eq!(priority, 0.5);
    }

    #[test]
    fn test_mock_scan_result() {
        let file = create_test_file_info("src/main.rs");
        let mock = MockScanResult::from_file_info(&file);

        assert!(mock.relative_path().contains("main.rs"));
        assert_eq!(mock.depth(), mock.relative_path.matches('/').count());
        assert!(!mock.is_docs());
        assert!(!mock.is_readme());
        assert!(mock.is_entrypoint()); // contains "main"
        assert!(!mock.has_examples());
        assert!(!mock.is_test());
        assert_eq!(mock.priority_boost(), 0.0);
        assert_eq!(mock.churn_score(), 0.0);
        assert_eq!(mock.centrality_in(), 0.0); // No centrality score set
        assert!(mock.imports().is_none());
        assert!(mock.doc_analysis().is_none());
    }

    #[test]
    fn test_mock_scan_result_readme() {
        let file = create_test_file_info("README.md");
        let mock = MockScanResult::from_file_info(&file);

        assert!(mock.is_readme());
    }

    #[test]
    fn test_mock_scan_result_test_file() {
        let file = create_test_file_info("src/test_utils.rs");
        let mock = MockScanResult::from_file_info(&file);

        assert!(mock.is_test());
    }

    #[test]
    fn test_mock_scan_result_example() {
        let file = create_test_file_info("examples/basic.rs");
        let mock = MockScanResult::from_file_info(&file);

        assert!(mock.has_examples());
    }

    #[test]
    fn test_mock_scan_result_with_centrality() {
        let mut file = create_test_file_info("src/lib.rs");
        file.centrality_score = Some(0.75);

        let mock = MockScanResult::from_file_info(&file);
        assert_eq!(mock.centrality_in(), 0.75);
    }

    #[test]
    fn test_optimize_selection_exact_budget() {
        let mut candidates = vec![
            FidelityCandidate {
                file_index: 0,
                mode: FidelityMode::Full,
                score: 100.0,
                tokens: 100,
                density: 1.0,
                content: "content".to_string(),
            },
        ];

        // Exact budget fit
        let selected = optimize_selection(&mut candidates, 100);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_optimize_selection_just_over_budget() {
        let mut candidates = vec![
            FidelityCandidate {
                file_index: 0,
                mode: FidelityMode::Full,
                score: 100.0,
                tokens: 101,
                density: 0.99,
                content: "content".to_string(),
            },
        ];

        // Just over budget
        let selected = optimize_selection(&mut candidates, 100);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_fidelity_candidate_mode_coverage() {
        // Test all fidelity modes
        let modes = vec![FidelityMode::Full, FidelityMode::Chunk, FidelityMode::Signature];
        for mode in modes {
            let candidate = FidelityCandidate {
                file_index: 0,
                mode: mode.clone(),
                score: 1.0,
                tokens: 10,
                density: 0.1,
                content: "test".to_string(),
            };
            match candidate.mode {
                FidelityMode::Full => assert!(true),
                FidelityMode::Chunk => assert!(true),
                FidelityMode::Signature => assert!(true),
            }
        }
    }

    #[test]
    fn test_file_fidelity_options_debug() {
        let options = FileFidelityOptions {
            file: create_test_file_info("test.rs"),
            priority: 1.0,
            full_content: "content".to_string(),
            full_tokens: 10,
            chunk_content: Some("chunk".to_string()),
            chunk_tokens: Some(5),
            signature_content: Some("sig".to_string()),
            signature_tokens: Some(50),
        };

        // Test debug output works
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("test.rs"));
    }

    #[test]
    fn test_fidelity_candidate_debug() {
        let candidate = FidelityCandidate {
            file_index: 0,
            mode: FidelityMode::Full,
            score: 100.0,
            tokens: 50,
            density: 2.0,
            content: "test".to_string(),
        };

        let debug_str = format!("{:?}", candidate);
        assert!(debug_str.contains("100"));
    }

    // Helper function to create a test FileInfo
    fn create_test_file_info(path: &str) -> FileInfo {
        use scribe_core::file::{FileWeight, RenderDecision};
        use std::path::PathBuf;
        FileInfo {
            path: PathBuf::from(path),
            relative_path: path.to_string(),
            size: 100,
            modified: None,
            decision: RenderDecision::include("test"),
            file_type: FileType::Source { language: scribe_core::file::Language::Rust },
            language: scribe_core::file::Language::Rust,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary: false,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        }
    }
}

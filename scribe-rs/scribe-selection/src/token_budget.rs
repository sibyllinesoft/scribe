//! Token budget selection logic previously implemented in the analyzer crate.
//! This module provides a shared implementation that can be reused by both the
//! library pipeline and external consumers without duplicating complex logic.

use crate::demotion::{DemotionEngine, FidelityMode};
use crate::weighting::FileWeights;
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
    /// Signature content and token count.
    signature_content: String,
    signature_tokens: usize,
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

/// Check if a documentation file is critical
fn is_critical_doc(path_lower: &str) -> bool {
    path_lower.contains("architecture")
        || path_lower.contains("design")
        || path_lower.contains("api")
        || path_lower.contains("spec")
        || path_lower.ends_with("changelog.md")
        || path_lower.ends_with("contributing.md")
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

fn is_mandatory_file(file: &FileInfo) -> bool {
    let path = file.relative_path.to_lowercase();

    // Skip files in dependency/build directories
    if path.contains("node_modules/")
        || path.contains("target/")
        || path.contains("vendor/")
        || path.contains(".git/")
        || path.contains("__pycache__/")
        || path.contains("build/")
        || path.contains("dist/")
        || path.contains(".cache/")
    {
        return false;
    }

    // README files (only in project root and first-level directories)
    if path.contains("readme") {
        let depth = path.matches('/').count();
        // Treat all READMEs (including subfolders) as mandatory unless they're in ignored dirs
        if depth <= 1 {
            return true;
        }
        if path.ends_with("readme.md")
            || path.ends_with("readme.markdown")
            || path.ends_with("readme.txt")
            || path.ends_with("readme")
        {
            return true;
        }
    }

    // Project configuration files (only at root level)
    if !path.contains('/')
        && matches!(
            path.as_str(),
            "package.json"
                | "cargo.toml"
                | "pyproject.toml"
                | "requirements.txt"
                | "go.mod"
                | "pom.xml"
                | "build.gradle"
                | "composer.json"
                | "tsconfig.json"
                | ".gitignore"
                | "dockerfile"
                | "docker-compose.yml"
        )
    {
        return true;
    }

    // Main/index files in root or src
    if (path.starts_with("src/") || path.starts_with("lib/") || !path.contains('/'))
        && (path.contains("main") || path.contains("index"))
    {
        return true;
    }

    false
}

async fn try_include_file_with_budget(
    mut file: FileInfo,
    counter: &TokenCounter,
    budget_tracker: &mut TokenBudget,
) -> Result<Option<FileInfo>> {
    match load_file_content_safe(&file.path) {
        Ok(content) => match counter.estimate_file_tokens(&content, &file.path) {
            Ok(token_count) => {
                if budget_tracker.can_allocate(token_count) {
                    budget_tracker.allocate(token_count);
                    file.content = Some(content);
                    file.token_estimate = Some(token_count);
                    file.char_count = Some(file.content.as_ref().unwrap().chars().count());
                    file.line_count = Some(file.content.as_ref().unwrap().lines().count());
                    Ok(Some(file))
                } else {
                    if std::env::var("SCRIBE_DEBUG").is_ok() {
                        eprintln!(
                            "⚠️  Skipping {} ({} tokens) - would exceed budget",
                            file.relative_path, token_count
                        );
                    }
                    Ok(None)
                }
            }
            Err(e) => {
                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!(
                        "⚠️  Failed to estimate tokens for {}: {}",
                        file.relative_path, e
                    );
                }
                Ok(None)
            }
        },
        Err(e) => {
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!("⚠️  Failed to read {}: {}", file.relative_path, e);
            }
            Ok(None)
        }
    }
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
        let (signature_content, signature_tokens) = match demotion_engine.demote_content(
            &full_content,
            &file.relative_path,
            FidelityMode::Signature,
            None,
        ) {
            Ok(result) if result.demoted_tokens > 0 => {
                (result.content, result.demoted_tokens)
            }
            _ => {
                // Fallback: use a minimal representation
                let fallback = format!("// {}\n", file.relative_path);
                let fallback_tokens = counter
                    .count_tokens(&fallback)
                    .unwrap_or(1);
                (fallback, fallback_tokens.max(1))
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

        // Signature candidate
        let signature_score = opt.priority * config.signature_boost;
        let signature_density = if opt.signature_tokens > 0 {
            signature_score / opt.signature_tokens as f64
        } else {
            0.0
        };
        candidates.push(FidelityCandidate {
            file_index: index,
            mode: FidelityMode::Signature,
            score: signature_score,
            tokens: opt.signature_tokens,
            density: signature_density,
            content: opt.signature_content.clone(),
        });
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

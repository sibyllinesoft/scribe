use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use globset::{Glob, GlobSet, GlobSetBuilder};
use tracing::{debug, info, warn};

use crate::report::SelectionMetrics;
use crate::{
    analyze_repository, apply_token_budget_selection, format_timestamp, report::ReportFile, Config,
    RepositoryAnalysis, SelectionConfig,
};
use scribe_core::tokenization::{utils as token_utils, TokenCounter};
use scribe_core::{FileInfo, Result};
use scribe_selection::FileWeights;

#[cfg(feature = "scaling")]
use scribe_index::{CodeDocument, CodeIndex};

/// Apply BM25 reranking to files based on query hint.
/// Returns FileWeights containing the BM25 boost scores for use in token budget selection.
/// Files matching the query get a strong boost (up to 10.0) to ensure they're prioritized.
#[cfg(feature = "scaling")]
fn apply_bm25_reranking(
    repo_path: &Path,
    files: &mut [FileInfo],
    query: &str,
    final_scores: &std::collections::HashMap<String, f64>,
) -> FileWeights {
    info!("Applying BM25 reranking for query: '{}'", query);

    // Try to create BM25 index
    let index = match CodeIndex::open_for_repo(repo_path) {
        Ok(idx) => idx,
        Err(e) => {
            warn!("Failed to open BM25 index: {}, skipping reranking", e);
            return FileWeights::new();
        }
    };

    // Build documents for indexing
    let docs: Vec<CodeDocument> = files
        .iter()
        .filter_map(|file| {
            let content = file.content.clone().or_else(|| fs::read_to_string(&file.path).ok())?;
            let content_hash = scribe_cache::ContentHash::from_content(content.as_bytes());

            // Extract symbols from content
            let lang_name = file.language.display_name();
            let symbols = extract_symbols_simple(&content, lang_name);

            Some(CodeDocument {
                path: file.path.to_string_lossy().to_string(),
                content_hash: content_hash.as_u64(),
                content,
                symbols,
                imports: vec![],
                language: lang_name.to_string(),
            })
        })
        .collect();

    if docs.is_empty() {
        return FileWeights::new();
    }

    // Index the documents
    if let Err(e) = index.index_documents(&docs) {
        warn!("Failed to index documents: {}, skipping reranking", e);
        return FileWeights::new();
    }

    if let Err(e) = index.reload() {
        warn!("Failed to reload index: {}", e);
        return FileWeights::new();
    }

    // Get BM25 scores
    let file_paths: Vec<PathBuf> = files.iter().map(|f| f.path.clone()).collect();
    let bm25_scores: std::collections::HashMap<PathBuf, f32> = match index.score_files(query, &file_paths) {
        Ok(scores) => scores.into_iter().collect(),
        Err(e) => {
            warn!("Failed to get BM25 scores: {}", e);
            return FileWeights::new();
        }
    };

    // Build FileWeights with strong BM25 boost for matching files.
    // Use a high multiplier (10x) so query-matched files are strongly prioritized.
    let mut weights = FileWeights::new();
    let max_bm25 = bm25_scores.values().copied().fold(0.0f32, f32::max);

    for file in files.iter() {
        let path_str = file.path.to_string_lossy().to_string();
        let bm25 = bm25_scores.get(&file.path).copied().unwrap_or(0.0);

        // Normalize BM25 score (0-1) and apply strong boost (up to 10.0)
        // Files with any BM25 match get at least 2.0 boost
        let normalized = if max_bm25 > 0.0 { bm25 / max_bm25 } else { 0.0 };
        let boost = if bm25 > 0.0 {
            2.0 + 8.0 * normalized as f64  // Range: 2.0 to 10.0
        } else {
            0.0
        };

        if boost > 0.0 {
            weights.set(path_str, boost);
        }
    }

    // Sort files by combined score (base + BM25) for ordering
    files.sort_by(|a, b| {
        let a_key = a.path.to_string_lossy().to_string();
        let b_key = b.path.to_string_lossy().to_string();

        let a_base = final_scores.get(&a_key).copied().unwrap_or(0.0);
        let b_base = final_scores.get(&b_key).copied().unwrap_or(0.0);

        let a_bm25 = bm25_scores.get(&a.path).copied().unwrap_or(0.0) as f64;
        let b_bm25 = bm25_scores.get(&b.path).copied().unwrap_or(0.0) as f64;

        // Normalize and combine: base + 2.0 * normalized_bm25
        let a_combined = a_base + 2.0 * (a_bm25 / 10.0).min(3.0);
        let b_combined = b_base + 2.0 * (b_bm25 / 10.0).min(3.0);

        b_combined.partial_cmp(&a_combined).unwrap_or(Ordering::Equal)
    });

    // Log top results
    let matched_count = weights.len();
    info!("BM25 reranking complete: {} files matched query", matched_count);
    for (i, file) in files.iter().take(5).enumerate() {
        let bm25 = bm25_scores.get(&file.path).copied().unwrap_or(0.0);
        let boost = weights.get_path(&file.path);
        debug!(
            "BM25 rank {}: {} (bm25={:.2}, boost={:.1})",
            i + 1,
            file.path.display(),
            bm25,
            boost
        );
    }

    weights
}

/// Simple symbol extraction for BM25 indexing
#[cfg(feature = "scaling")]
fn extract_symbols_simple(content: &str, language: &str) -> Vec<String> {
    let mut symbols = Vec::new();

    let patterns: &[&str] = match language.to_lowercase().as_str() {
        "rust" => &[r"fn\s+(\w+)", r"struct\s+(\w+)", r"enum\s+(\w+)", r"trait\s+(\w+)"],
        "python" => &[r"def\s+(\w+)", r"class\s+(\w+)"],
        "go" => &[r"func\s+(\w+)", r"type\s+(\w+)\s+struct"],
        "javascript" | "typescript" => &[r"function\s+(\w+)", r"class\s+(\w+)"],
        "java" => &[r"class\s+(\w+)", r"interface\s+(\w+)"],
        _ => &[],
    };

    for pattern in patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            for cap in re.captures_iter(content) {
                if let Some(name) = cap.get(1) {
                    symbols.push(name.as_str().to_string());
                }
            }
        }
    }

    symbols
}

/// Configuration options controlling how selection behaves when generating
/// analysis reports. These options capture the CLI behaviour but remain general
/// enough for other front-ends (e.g. the web service) to reuse.
#[derive(Debug, Clone)]
pub struct SelectionOptions {
    /// Target number of tokens to keep within. `0` means unlimited.
    pub token_target: usize,
    /// When true the selector skips token-budget pruning and returns everything.
    pub force_traditional: bool,
    /// Human friendly label for the active algorithm (used in metrics output).
    pub algorithm_name: Option<String>,
    /// Whether to inject the directory inventory map into the final bundle.
    pub include_directory_map: bool,
    /// Query hint for BM25-based file relevance scoring.
    pub query_hint: Option<String>,
}

impl Default for SelectionOptions {
    fn default() -> Self {
        Self {
            token_target: 128_000,
            force_traditional: false,
            algorithm_name: None,
            include_directory_map: true,
            query_hint: None,
        }
    }
}

/// Result of running the selection step against a repository analysis.
#[derive(Debug, Clone)]
pub struct SelectionOutcome {
    /// Files that were selected for inclusion in the final bundle.
    pub selected_files: Vec<ReportFile>,
    /// The underlying `FileInfo` records corresponding to the selected files.
    pub selected_file_infos: Vec<FileInfo>,
    /// Summary statistics describing the selection.
    pub metrics: SelectionMetrics,
    /// Number of files that were eligible after filtering and ignore handling.
    pub eligible_file_count: usize,
    /// Indicates whether a token budget was applied.
    pub unlimited_budget: bool,
}

/// Combined result containing the raw repository analysis and the derived
/// selection outcome.
#[derive(Debug, Clone)]
pub struct AnalysisOutcome {
    pub analysis: RepositoryAnalysis,
    pub selection: SelectionOutcome,
}

/// Run a full repository analysis followed by intelligent selection using the
/// provided configuration.
pub async fn analyze_and_select<P: AsRef<Path>>(
    repo_path: P,
    config: &Config,
    options: &SelectionOptions,
) -> Result<AnalysisOutcome> {
    let repo_path = repo_path.as_ref();
    let analysis = analyze_repository(repo_path, config).await?;
    let selection = select_from_analysis(repo_path, config, &analysis, options).await?;

    Ok(AnalysisOutcome {
        analysis,
        selection,
    })
}

/// Derive a selection outcome from an existing repository analysis.
pub async fn select_from_analysis(
    repo_path: &Path,
    config: &Config,
    analysis: &RepositoryAnalysis,
    options: &SelectionOptions,
) -> Result<SelectionOutcome> {
    let selection_start = Instant::now();
    let token_counter = TokenCounter::global();

    let total_files_discovered = analysis.files.len();
    let include_filter = build_include_filter(&config.filtering.include_patterns);

    let mut filtered_infos: Vec<FileInfo> = analysis
        .files
        .iter()
        .filter(|info| info.decision.should_include())
        .filter(|info| match &include_filter {
            Some(filter) => filter.is_match(info.relative_path.as_str()),
            None => true,
        })
        .cloned()
        .collect();

    // Apply BM25 reranking if query_hint is provided
    #[cfg(feature = "scaling")]
    let bm25_weights: Option<FileWeights> = if let Some(ref query) = options.query_hint {
        Some(apply_bm25_reranking(repo_path, &mut filtered_infos, query, &analysis.final_scores))
    } else {
        None
    };

    #[cfg(not(feature = "scaling"))]
    let bm25_weights: Option<FileWeights> = None;

    let unlimited_budget = options.force_traditional || options.token_target == 0;

    let mut selected_infos = if unlimited_budget {
        filtered_infos.clone()
    } else {
        apply_token_budget_selection(
            filtered_infos.clone(),
            options.token_target,
            config,
            bm25_weights.as_ref(),
            &SelectionConfig::default(),
        ).await?
    };

    selected_infos.sort_by(|a, b| {
        let a_key = a.path.to_string_lossy();
        let b_key = b.path.to_string_lossy();
        let a_score = analysis
            .final_scores
            .get(&a_key.to_string())
            .copied()
            .unwrap_or(0.0);
        let b_score = analysis
            .final_scores
            .get(&b_key.to_string())
            .copied()
            .unwrap_or(0.0);

        b_score
            .partial_cmp(&a_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });

    let mut selected_file_infos = selected_infos.clone();

    let mut selected_files = Vec::new();
    let mut budget_consumed = 0usize;

    // Always attempt to include the directory map first so subsequent selection respects
    // the remaining budget. This keeps the structural overview available in every bundle.
    if options.include_directory_map {
        if let Some(directory_map) = build_directory_map_for_analysis(repo_path, &analysis.files) {
            let map_tokens = directory_map.estimated_tokens;

            if !unlimited_budget {
                budget_consumed = budget_consumed.saturating_add(map_tokens);

                if map_tokens > options.token_target && std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!(
                        "Directory map ({} tokens) exceeds the token budget {}; proceeding regardless",
                        map_tokens, options.token_target
                    );
                }
            }

            selected_files.push(directory_map);
        }
    }

    for info in selected_infos {
        let mut content = info.content.clone();
        if content.is_none() && !info.is_binary {
            if let Ok(read) = fs::read_to_string(&info.path) {
                content = Some(read);
            }
        }

        let text = content.unwrap_or_else(|| String::from("<binary or unavailable content>"));
        let estimated_tokens = info.token_estimate.unwrap_or_else(|| {
            token_counter
                .estimate_file_tokens(&text, &info.path)
                .unwrap_or_else(|_| token_utils::estimate_tokens_legacy(&text))
                .max(1)
        });

        if !unlimited_budget {
            if budget_consumed.saturating_add(estimated_tokens) > options.token_target {
                continue;
            }
            budget_consumed = budget_consumed.saturating_add(estimated_tokens);
        }

        let path_key = info.path.to_string_lossy().to_string();
        let importance_score = analysis.final_scores.get(&path_key).copied().unwrap_or(0.0);

        let display_path = info
            .path
            .strip_prefix(repo_path)
            .unwrap_or(&info.path)
            .to_string_lossy()
            .to_string();

        selected_files.push(ReportFile {
            path: info.path.clone(),
            relative_path: display_path,
            content: text,
            size: info.size,
            estimated_tokens,
            importance_score,
            centrality_score: info.centrality_score.unwrap_or(0.0),
            query_relevance_score: 0.0,
            entry_point_proximity: 0.0,
            content_quality_score: 0.0,
            repository_role_score: 0.0,
            recency_score: 0.0,
            modified: info.modified,
        });
    }

    if selected_files.is_empty() {
        if let Some(first) = filtered_infos.first().or_else(|| analysis.files.first()) {
            let fallback_content = fs::read_to_string(&first.path).unwrap_or_default();
            let estimated_tokens = token_counter
                .estimate_file_tokens(&fallback_content, &first.path)
                .unwrap_or_else(|_| token_utils::estimate_tokens_legacy(&fallback_content))
                .max(1);

            let fallback_display = first
                .path
                .strip_prefix(repo_path)
                .unwrap_or(&first.path)
                .to_string_lossy()
                .to_string();

            selected_files.push(ReportFile {
                path: first.path.clone(),
                relative_path: fallback_display.clone(),
                content: fallback_content,
                size: first.size,
                estimated_tokens,
                importance_score: analysis
                    .final_scores
                    .get(&first.path.to_string_lossy().to_string())
                    .copied()
                    .unwrap_or(0.0),
                centrality_score: first.centrality_score.unwrap_or(0.0),
                query_relevance_score: 0.0,
                entry_point_proximity: 0.0,
                content_quality_score: 0.0,
                repository_role_score: 0.0,
                recency_score: 0.0,
                modified: first.modified,
            });
            selected_file_infos.push(first.clone());
        }
    }

    let total_tokens_estimated: usize = selected_files.iter().map(|f| f.estimated_tokens).sum();
    let selection_time_ms = selection_start.elapsed().as_millis() as u128;

    let coverage_score = if total_files_discovered > 0 {
        selected_files.len() as f64 / total_files_discovered as f64
    } else {
        1.0
    };

    let relevance_score = if selected_files.is_empty() {
        0.0
    } else {
        selected_files
            .iter()
            .map(|f| f.importance_score)
            .sum::<f64>()
            / selected_files.len() as f64
    };

    let algorithm_label = match (&options.algorithm_name, unlimited_budget) {
        (Some(name), true) => format!("{} (unlimited)", name),
        (Some(name), false) => name.clone(),
        (None, true) => "Tiered (unlimited budget)".to_string(),
        (None, false) => "Tiered (token-budget)".to_string(),
    };

    let metrics = SelectionMetrics {
        total_files_discovered,
        files_selected: selected_files.len(),
        total_tokens_estimated,
        selection_time_ms,
        algorithm_used: algorithm_label,
        coverage_score,
        relevance_score,
    };

    Ok(SelectionOutcome {
        selected_files,
        selected_file_infos,
        metrics,
        eligible_file_count: filtered_infos.len(),
        unlimited_budget,
    })
}

fn build_include_filter(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() {
        return None;
    }

    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        if let Ok(glob) = Glob::new(pattern) {
            builder.add(glob);
        }
    }

    builder.build().ok()
}

fn build_directory_map_for_analysis(repo_path: &Path, files: &[FileInfo]) -> Option<ReportFile> {
    let inventory = gather_inventory_entries(repo_path, files);
    if inventory.is_empty() {
        return None;
    }

    let directory_map = build_directory_map(&inventory)?;
    if directory_map.is_empty() {
        return None;
    }

    let estimated_tokens = TokenCounter::global()
        .count_tokens(&directory_map)
        .unwrap_or_else(|_| token_utils::estimate_tokens_legacy(&directory_map));
    let tokens = estimated_tokens.max(1);
    let size = directory_map.len() as u64;

    Some(ReportFile {
        path: repo_path.join("DIRECTORY_MAP.txt"),
        relative_path: "DIRECTORY_MAP.txt".to_string(),
        content: directory_map,
        size,
        estimated_tokens: tokens,
        importance_score: 1.0,
        centrality_score: 0.0,
        query_relevance_score: 0.0,
        entry_point_proximity: 0.0,
        content_quality_score: 0.0,
        repository_role_score: 0.0,
        recency_score: 0.0,
        modified: None,
    })
}

fn gather_inventory_entries(repo_path: &Path, files: &[FileInfo]) -> Vec<InventoryEntry> {
    if files.is_empty() {
        return Vec::new();
    }

    let mut entries = Vec::with_capacity(files.len() + 16);
    let mut directories: HashSet<String> = HashSet::new();

    // Canonicalize repo_path for consistent prefix stripping
    let repo_prefix = repo_path
        .canonicalize()
        .unwrap_or_else(|_| repo_path.to_path_buf());

    for file in files {
        // Try to make the path relative to the repo root
        let file_path = Path::new(&file.relative_path);
        let relative = if file_path.is_absolute() {
            // Strip the repo prefix if present
            file_path
                .strip_prefix(&repo_prefix)
                .or_else(|_| file_path.strip_prefix(repo_path))
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|_| file_path.to_path_buf())
        } else {
            file_path.to_path_buf()
        };

        let mut ancestor = relative.parent();
        while let Some(parent) = ancestor {
            let parent_str = parent.to_string_lossy().to_string();
            if parent_str.is_empty() {
                break;
            }
            directories.insert(parent_str);
            ancestor = parent.parent();
        }
    }

    for dir in directories {
        if dir.is_empty() {
            continue;
        }
        entries.push(InventoryEntry { path: dir });
    }

    entries
}

#[derive(Debug, Clone)]
struct InventoryEntry {
    path: String,
}

/// Tree node for building hierarchical directory representation
#[derive(Debug, Default)]
struct DirTreeNode {
    name: String,
    children: Vec<DirTreeNode>,
}

impl DirTreeNode {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            children: Vec::new(),
        }
    }

    /// Insert a path into the tree, creating intermediate nodes as needed
    fn insert(&mut self, parts: &[&str]) {
        if parts.is_empty() {
            return;
        }

        let name = parts[0];
        let rest = &parts[1..];

        // Find or create child
        let child_idx = self.children.iter().position(|c| c.name == name);
        let child = match child_idx {
            Some(idx) => &mut self.children[idx],
            None => {
                self.children.push(DirTreeNode::new(name));
                self.children.last_mut().unwrap()
            }
        };

        child.insert(rest);
    }

    /// Render this node and its children using brace notation
    /// e.g., "src/{bin/cli,lib}" for siblings, "src/bin/cli" for single paths
    fn render_brace(&self) -> String {
        if self.children.is_empty() {
            return self.name.clone();
        }

        let children_rendered: Vec<String> =
            self.children.iter().map(|c| c.render_brace()).collect();

        if children_rendered.len() == 1 {
            // Single child: path/child
            format!("{}/{}", self.name, children_rendered[0])
        } else {
            // Multiple children: path/{a,b,c}
            format!("{}/{{{}}}", self.name, children_rendered.join(","))
        }
    }
}

/// Build a tree from directory paths
fn build_dir_tree(entries: &[InventoryEntry]) -> DirTreeNode {
    let mut root = DirTreeNode::new(".");

    for entry in entries {
        if entry.path.is_empty() {
            continue;
        }
        let parts: Vec<&str> = entry.path.split('/').collect();
        root.insert(&parts);
    }

    // Sort children alphabetically for consistent output
    root.sort_children();
    root
}

impl DirTreeNode {
    /// Recursively sort all children alphabetically
    fn sort_children(&mut self) {
        self.children.sort_by(|a, b| a.name.cmp(&b.name));
        for child in &mut self.children {
            child.sort_children();
        }
    }
}

fn build_directory_map(entries: &[InventoryEntry]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }

    let tree = build_dir_tree(entries);

    if tree.children.is_empty() {
        return None;
    }

    // Render each top-level directory using brace notation
    let lines: Vec<String> = tree.children.iter().map(|c| c.render_brace()).collect();

    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_options_default() {
        let options = SelectionOptions::default();
        assert_eq!(options.token_target, 128_000);
        assert!(!options.force_traditional);
        assert!(options.algorithm_name.is_none());
        assert!(options.include_directory_map);
    }

    #[test]
    fn test_selection_options_custom() {
        let options = SelectionOptions {
            token_target: 50_000,
            force_traditional: true,
            algorithm_name: Some("custom".to_string()),
            include_directory_map: false,
            query_hint: Some("test query".to_string()),
        };
        assert_eq!(options.token_target, 50_000);
        assert!(options.force_traditional);
        assert_eq!(options.algorithm_name, Some("custom".to_string()));
        assert!(!options.include_directory_map);
        assert_eq!(options.query_hint, Some("test query".to_string()));
    }

    #[test]
    fn test_inventory_entry_creation() {
        let entry = InventoryEntry {
            path: "src/lib".to_string(),
        };
        assert_eq!(entry.path, "src/lib");
    }

    #[test]
    fn test_build_directory_map_empty() {
        let entries: Vec<InventoryEntry> = vec![];
        let result = build_directory_map(&entries);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_directory_map_single_root_only() {
        // Root entry only (empty path) should return None - no actual directories
        let entries = vec![InventoryEntry {
            path: String::new(),
        }];
        let result = build_directory_map(&entries);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_directory_map_single_dir() {
        let entries = vec![InventoryEntry {
            path: "src".to_string(),
        }];
        let result = build_directory_map(&entries);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map, "src");
    }

    #[test]
    fn test_build_directory_map_multiple() {
        let entries = vec![
            InventoryEntry {
                path: String::new(),
            },
            InventoryEntry {
                path: "src".to_string(),
            },
            InventoryEntry {
                path: "src/lib".to_string(),
            },
            InventoryEntry {
                path: "tests".to_string(),
            },
        ];
        let result = build_directory_map(&entries);
        assert!(result.is_some());
        let map = result.unwrap();
        // Brace notation: src with child lib, tests as sibling
        assert!(map.contains("src/lib"));
        assert!(map.contains("tests"));
    }

    #[test]
    fn test_build_include_filter_empty() {
        let patterns: Vec<String> = vec![];
        let result = build_include_filter(&patterns);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_include_filter_single() {
        let patterns = vec!["*.rs".to_string()];
        let result = build_include_filter(&patterns);
        assert!(result.is_some());

        let filter = result.unwrap();
        assert!(filter.is_match("main.rs"));
        assert!(filter.is_match("lib.rs"));
        assert!(!filter.is_match("main.py"));
    }

    #[test]
    fn test_build_include_filter_multiple() {
        let patterns = vec!["*.rs".to_string(), "*.py".to_string()];
        let result = build_include_filter(&patterns);
        assert!(result.is_some());

        let filter = result.unwrap();
        assert!(filter.is_match("main.rs"));
        assert!(filter.is_match("app.py"));
        assert!(!filter.is_match("index.js"));
    }

    #[test]
    fn test_build_include_filter_glob_pattern() {
        let patterns = vec!["src/**/*.rs".to_string()];
        let result = build_include_filter(&patterns);
        assert!(result.is_some());

        let filter = result.unwrap();
        assert!(filter.is_match("src/main.rs"));
        assert!(filter.is_match("src/lib/utils.rs"));
        assert!(!filter.is_match("tests/test.rs"));
    }

    #[test]
    fn test_selection_outcome_structure() {
        let outcome = SelectionOutcome {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: crate::report::SelectionMetrics {
                total_files_discovered: 100,
                files_selected: 10,
                total_tokens_estimated: 5000,
                selection_time_ms: 150,
                algorithm_used: "test".to_string(),
                coverage_score: 0.8,
                relevance_score: 0.9,
            },
            eligible_file_count: 50,
            unlimited_budget: false,
        };

        assert!(outcome.selected_files.is_empty());
        assert_eq!(outcome.eligible_file_count, 50);
        assert!(!outcome.unlimited_budget);
        assert_eq!(outcome.metrics.files_selected, 10);
    }

    #[test]
    fn test_gather_inventory_entries_empty() {
        let repo_path = PathBuf::from("/tmp/test");
        let files: Vec<scribe_core::FileInfo> = vec![];
        let entries = gather_inventory_entries(&repo_path, &files);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_selection_options_clone() {
        let options = SelectionOptions {
            token_target: 50_000,
            force_traditional: true,
            algorithm_name: Some("test_algo".to_string()),
            include_directory_map: false,
            query_hint: Some("test".to_string()),
        };
        let cloned = options.clone();
        assert_eq!(options.token_target, cloned.token_target);
        assert_eq!(options.force_traditional, cloned.force_traditional);
        assert_eq!(options.algorithm_name, cloned.algorithm_name);
        assert_eq!(options.query_hint, cloned.query_hint);
    }

    #[test]
    fn test_selection_options_debug() {
        let options = SelectionOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("SelectionOptions"));
        assert!(debug_str.contains("token_target"));
    }

    #[test]
    fn test_selection_outcome_clone() {
        let outcome = SelectionOutcome {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: crate::report::SelectionMetrics {
                total_files_discovered: 10,
                files_selected: 5,
                total_tokens_estimated: 1000,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 0.5,
                relevance_score: 0.8,
            },
            eligible_file_count: 8,
            unlimited_budget: true,
        };
        let cloned = outcome.clone();
        assert_eq!(outcome.eligible_file_count, cloned.eligible_file_count);
        assert_eq!(outcome.unlimited_budget, cloned.unlimited_budget);
    }

    #[test]
    fn test_selection_outcome_debug() {
        let outcome = SelectionOutcome {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: crate::report::SelectionMetrics {
                total_files_discovered: 10,
                files_selected: 5,
                total_tokens_estimated: 1000,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 0.5,
                relevance_score: 0.8,
            },
            eligible_file_count: 8,
            unlimited_budget: true,
        };
        let debug_str = format!("{:?}", outcome);
        assert!(debug_str.contains("SelectionOutcome"));
        assert!(debug_str.contains("eligible_file_count"));
    }

    fn create_test_analysis() -> RepositoryAnalysis {
        use std::collections::HashMap;
        use scribe_core::AnalysisMetadata;

        RepositoryAnalysis {
            files: vec![],
            heuristic_scores: HashMap::new(),
            #[cfg(feature = "graph")]
            centrality_scores: None,
            final_scores: HashMap::new(),
            metadata: AnalysisMetadata {
                timestamp: SystemTime::now(),
                scribe_version: "test".to_string(),
                features_enabled: vec![],
                config_hash: None,
            },
        }
    }

    #[test]
    fn test_analysis_outcome_clone() {
        let outcome = AnalysisOutcome {
            analysis: create_test_analysis(),
            selection: SelectionOutcome {
                selected_files: vec![],
                selected_file_infos: vec![],
                metrics: crate::report::SelectionMetrics {
                    total_files_discovered: 0,
                    files_selected: 0,
                    total_tokens_estimated: 0,
                    selection_time_ms: 0,
                    algorithm_used: "test".to_string(),
                    coverage_score: 0.0,
                    relevance_score: 0.0,
                },
                eligible_file_count: 0,
                unlimited_budget: false,
            },
        };
        let cloned = outcome.clone();
        assert_eq!(outcome.selection.unlimited_budget, cloned.selection.unlimited_budget);
    }

    #[test]
    fn test_analysis_outcome_debug() {
        let outcome = AnalysisOutcome {
            analysis: create_test_analysis(),
            selection: SelectionOutcome {
                selected_files: vec![],
                selected_file_infos: vec![],
                metrics: crate::report::SelectionMetrics {
                    total_files_discovered: 0,
                    files_selected: 0,
                    total_tokens_estimated: 0,
                    selection_time_ms: 0,
                    algorithm_used: "test".to_string(),
                    coverage_score: 0.0,
                    relevance_score: 0.0,
                },
                eligible_file_count: 0,
                unlimited_budget: false,
            },
        };
        let debug_str = format!("{:?}", outcome);
        assert!(debug_str.contains("AnalysisOutcome"));
    }

    #[test]
    fn test_inventory_entry_clone() {
        let entry = InventoryEntry {
            path: "src/main.rs".to_string(),
        };
        let cloned = entry.clone();
        assert_eq!(entry.path, cloned.path);
    }

    #[test]
    fn test_inventory_entry_debug() {
        let entry = InventoryEntry {
            path: "test".to_string(),
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("InventoryEntry"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_build_directory_map_top_level_dirs() {
        // Top-level directories should each appear on their own line
        let entries = vec![
            InventoryEntry { path: "z_dir".to_string() },
            InventoryEntry { path: "a_dir".to_string() },
            InventoryEntry { path: "m_dir".to_string() },
        ];
        let result = build_directory_map(&entries);
        assert!(result.is_some());
        let map = result.unwrap();
        let lines: Vec<&str> = map.lines().collect();

        // All three directories should be present as separate lines
        assert!(lines.contains(&"a_dir"));
        assert!(lines.contains(&"m_dir"));
        assert!(lines.contains(&"z_dir"));
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_build_include_filter_invalid_pattern() {
        // Invalid glob patterns should be silently ignored
        let patterns = vec!["[invalid".to_string(), "*.rs".to_string()];
        let result = build_include_filter(&patterns);
        // Should still create a filter with the valid pattern
        assert!(result.is_some());
        let filter = result.unwrap();
        assert!(filter.is_match("main.rs"));
    }

    #[test]
    fn test_build_directory_map_with_nested_paths() {
        let entries = vec![
            InventoryEntry { path: String::new() },
            InventoryEntry { path: "src".to_string() },
            InventoryEntry { path: "src/lib".to_string() },
            InventoryEntry { path: "src/lib/utils".to_string() },
            InventoryEntry { path: "src/bin".to_string() },
        ];
        let result = build_directory_map(&entries);
        assert!(result.is_some());
        let map = result.unwrap();
        // Brace notation: src/{bin,lib/utils}
        assert_eq!(map, "src/{bin,lib/utils}");
    }

    #[test]
    fn test_build_directory_map_brace_notation() {
        // Test full brace notation rendering
        let entries = vec![
            InventoryEntry { path: "packages".to_string() },
            InventoryEntry { path: "packages/core".to_string() },
            InventoryEntry { path: "packages/core/src".to_string() },
            InventoryEntry { path: "packages/core/tests".to_string() },
            InventoryEntry { path: "packages/cli".to_string() },
            InventoryEntry { path: "packages/cli/src".to_string() },
        ];
        let result = build_directory_map(&entries);
        assert!(result.is_some());
        let map = result.unwrap();
        // packages/{cli/src,core/{src,tests}}
        assert!(map.contains("packages/"));
        assert!(map.contains("cli/src"));
        assert!(map.contains("core/"));
    }

    #[test]
    fn test_selection_options_unlimited_budget_conditions() {
        // Unlimited when force_traditional is true
        let options1 = SelectionOptions {
            token_target: 50_000,
            force_traditional: true,
            algorithm_name: None,
            include_directory_map: true,
            query_hint: None,
        };
        let unlimited1 = options1.force_traditional || options1.token_target == 0;
        assert!(unlimited1);

        // Unlimited when token_target is 0
        let options2 = SelectionOptions {
            token_target: 0,
            force_traditional: false,
            algorithm_name: None,
            include_directory_map: true,
            query_hint: None,
        };
        let unlimited2 = options2.force_traditional || options2.token_target == 0;
        assert!(unlimited2);

        // Not unlimited otherwise
        let options3 = SelectionOptions {
            token_target: 50_000,
            force_traditional: false,
            algorithm_name: None,
            include_directory_map: true,
            query_hint: None,
        };
        let unlimited3 = options3.force_traditional || options3.token_target == 0;
        assert!(!unlimited3);
    }

    #[test]
    fn test_selection_metrics_coverage_calculation() {
        // With files discovered
        let total_discovered = 100;
        let files_selected = 25;
        let coverage = files_selected as f64 / total_discovered as f64;
        assert!((coverage - 0.25).abs() < 0.01);

        // Edge case: no files discovered
        let zero_discovered = 0;
        let coverage_zero = if zero_discovered > 0 {
            files_selected as f64 / zero_discovered as f64
        } else {
            1.0
        };
        assert_eq!(coverage_zero, 1.0);
    }

    #[test]
    fn test_algorithm_label_generation() {
        // Case 1: Named algorithm, unlimited budget
        let name = Some("Custom".to_string());
        let unlimited = true;
        let label1 = match (&name, unlimited) {
            (Some(n), true) => format!("{} (unlimited)", n),
            (Some(n), false) => n.clone(),
            (None, true) => "Tiered (unlimited budget)".to_string(),
            (None, false) => "Tiered (token-budget)".to_string(),
        };
        assert_eq!(label1, "Custom (unlimited)");

        // Case 2: Named algorithm, limited budget
        let label2 = match (&name, false) {
            (Some(n), true) => format!("{} (unlimited)", n),
            (Some(n), false) => n.clone(),
            (None, true) => "Tiered (unlimited budget)".to_string(),
            (None, false) => "Tiered (token-budget)".to_string(),
        };
        assert_eq!(label2, "Custom");

        // Case 3: No name, unlimited budget
        let no_name: Option<String> = None;
        let label3 = match (&no_name, true) {
            (Some(n), true) => format!("{} (unlimited)", n),
            (Some(n), false) => n.clone(),
            (None, true) => "Tiered (unlimited budget)".to_string(),
            (None, false) => "Tiered (token-budget)".to_string(),
        };
        assert_eq!(label3, "Tiered (unlimited budget)");

        // Case 4: No name, limited budget
        let label4 = match (&no_name, false) {
            (Some(n), true) => format!("{} (unlimited)", n),
            (Some(n), false) => n.clone(),
            (None, true) => "Tiered (unlimited budget)".to_string(),
            (None, false) => "Tiered (token-budget)".to_string(),
        };
        assert_eq!(label4, "Tiered (token-budget)");
    }

    #[test]
    fn test_build_directory_map_for_analysis_empty() {
        let repo_path = PathBuf::from("/tmp/test");
        let files: Vec<scribe_core::FileInfo> = vec![];
        let result = build_directory_map_for_analysis(&repo_path, &files);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_include_filter_complex_patterns() {
        let patterns = vec![
            "**/test*.rs".to_string(),
            "!**/vendor/**".to_string(),
            "src/**".to_string(),
        ];
        let result = build_include_filter(&patterns);
        assert!(result.is_some());
        let filter = result.unwrap();
        assert!(filter.is_match("src/main.rs"));
    }
}

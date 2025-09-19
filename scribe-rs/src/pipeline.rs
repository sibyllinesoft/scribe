use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::report::SelectionMetrics;
use crate::{
    analyze_repository, apply_token_budget_selection, format_bytes, format_timestamp,
    report::ReportFile, Config, RepositoryAnalysis,
};
use scribe_core::tokenization::{utils as token_utils, TokenCounter};
use scribe_core::{FileInfo, Result};

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
}

impl Default for SelectionOptions {
    fn default() -> Self {
        Self {
            token_target: 128_000,
            force_traditional: false,
            algorithm_name: None,
            include_directory_map: true,
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

    let filtered_infos: Vec<FileInfo> = analysis
        .files
        .iter()
        .filter(|info| info.decision.should_include())
        .filter(|info| match &include_filter {
            Some(filter) => filter.is_match(info.relative_path.as_str()),
            None => true,
        })
        .cloned()
        .collect();

    let unlimited_budget = options.force_traditional || options.token_target == 0;

    let mut selected_infos = if unlimited_budget {
        filtered_infos.clone()
    } else {
        apply_token_budget_selection(filtered_infos.clone(), options.token_target, config).await?
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
            });
            selected_file_infos.push(first.clone());
        }
    }

    if options.include_directory_map
        && !selected_files
            .iter()
            .any(|file| file.relative_path == "DIRECTORY_MAP.txt")
    {
        if let Some(directory_map) = build_directory_map_for_analysis(repo_path, &analysis.files) {
            selected_files.insert(0, directory_map);
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

    let estimated_tokens = directory_map.split_whitespace().count() * 4 / 3;
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
    })
}

fn gather_inventory_entries(repo_path: &Path, files: &[FileInfo]) -> Vec<InventoryEntry> {
    if files.is_empty() {
        return Vec::new();
    }

    let mut entries = Vec::with_capacity(files.len() + 16);
    let total_size: u64 = files.iter().map(|f| f.size).sum();
    let root_modified = files.iter().filter_map(|f| f.modified).max();

    entries.push(InventoryEntry {
        path: String::new(),
        is_dir: true,
        size: total_size,
        modified: root_modified,
    });

    let mut directories: HashSet<String> = HashSet::new();

    for file in files {
        entries.push(InventoryEntry {
            path: file.relative_path.clone(),
            is_dir: false,
            size: file.size,
            modified: file.modified,
        });

        let mut ancestor = Path::new(&file.relative_path).parent();
        while let Some(parent) = ancestor {
            let parent_str = parent.to_string_lossy().to_string();
            if parent_str.is_empty() {
                break;
            }
            directories.insert(parent_str.clone());
            ancestor = parent.parent();
        }
    }

    for dir in directories {
        if dir.is_empty() {
            continue;
        }

        let dir_path = repo_path.join(&dir);
        let metadata = fs::metadata(dir_path).ok();
        let modified = metadata.as_ref().and_then(|meta| meta.modified().ok());
        let size = metadata.map(|meta| meta.len()).unwrap_or(0);

        entries.push(InventoryEntry {
            path: dir,
            is_dir: true,
            size,
            modified,
        });
    }

    entries
}

#[derive(Debug, Clone)]
struct InventoryEntry {
    path: String,
    is_dir: bool,
    size: u64,
    modified: Option<SystemTime>,
}

fn build_directory_map(entries: &[InventoryEntry]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }

    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.path.cmp(&b.path).then_with(|| b.is_dir.cmp(&a.is_dir)));

    let mut lines = Vec::with_capacity(sorted.len() + 4);
    lines.push("Repository Inventory".to_string());
    lines.push("====================".to_string());
    lines.push(format!(
        "{:<60} {:<4} {:>10} {}",
        "Path", "Type", "Size", "Modified"
    ));
    lines.push(format!("{:-<60} {:-<4} {:-<10} {:-<19}", "", "", "", ""));

    for entry in sorted {
        let display_path = if entry.path.is_empty() {
            "."
        } else {
            entry.path.as_str()
        };
        let type_label = if entry.is_dir { "dir" } else { "file" };
        let size_label = format_bytes(entry.size);
        let modified_label = format_timestamp(entry.modified);

        lines.push(format!(
            "{:<60} {:<4} {:>10} {}",
            display_path, type_label, size_label, modified_label
        ));
    }

    lines.push(String::new());
    Some(lines.join("\n"))
}

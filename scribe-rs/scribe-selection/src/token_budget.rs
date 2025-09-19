//! Token budget selection logic previously implemented in the analyzer crate.
//! This module provides a shared implementation that can be reused by both the
//! library pipeline and external consumers without duplicating complex logic.

use crate::demotion::{DemotionEngine, FidelityMode};
use scribe_analysis::heuristics::ScanResult;
use scribe_core::{
    tokenization::{TokenBudget, TokenCounter},
    Config, FileInfo, FileType, Result, ScribeError,
};
use scribe_graph::CentralityCalculator;
use std::collections::HashSet;
use std::path::Path;

/// Apply the library's tiered token budget selection to a set of files.
///
/// The selector prioritizes files in multiple tiers:
/// 1. Mandatory project metadata (README, config files, entrypoints)
/// 2. Source files ordered by graph centrality with demotion fallback
/// 3. Documentation with preference for design/architecture material
/// 4. Any remaining files while budget remains
///
/// The function loads file content and token estimates for the selected files
/// and will attempt demotion (chunk/signature extraction) when a source file
/// would otherwise exceed the available budget.
pub async fn apply_token_budget_selection(
    files: Vec<FileInfo>,
    token_budget: usize,
    config: &Config,
) -> Result<Vec<FileInfo>> {
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!(
            "🎯 Intelligent token budget selection: {} tokens across {} files",
            token_budget,
            files.len()
        );
    }

    let counter = TokenCounter::global();
    let mut selected_files = Vec::new();

    // Split files into categories for prioritized selection
    let (mandatory_files, source_files, doc_files, other_files) = categorize_files(files.clone());

    // Keep a reference to all files for final optimization pass
    let all_files = files;

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!(
            "📊 File categories: {} mandatory, {} source, {} docs, {} other",
            mandatory_files.len(),
            source_files.len(),
            doc_files.len(),
            other_files.len()
        );
    }

    let mut budget_tracker = TokenBudget::new(token_budget);

    // Tier 1: Mandatory files (README, project config, main/index files)
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("📌 Tier 1: Processing mandatory files");
    }
    for file in mandatory_files {
        if budget_tracker.available() < 1 {
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!("🛑 Budget exhausted, stopping mandatory file selection");
            }
            break;
        }
        if let Some(selected_file) =
            try_include_file_with_budget(file, &counter, &mut budget_tracker).await?
        {
            selected_files.push(selected_file);
        }
    }

    // Tier 2: Source files (prioritized by centrality)
    if !source_files.is_empty() && budget_tracker.available() > 0 {
        if std::env::var("SCRIBE_DEBUG").is_ok() {
            eprintln!("🧠 Tier 2: Processing source files with centrality analysis");
        }

        // Calculate centrality scores for source files
        let calculator = CentralityCalculator::new()?;
        let mock_scan_results: Vec<_> = source_files
            .iter()
            .map(MockScanResult::from_file_info)
            .collect();
        let centrality_results = calculator.calculate_centrality(&mock_scan_results)?;

        let mut source_with_centrality: Vec<_> = source_files
            .into_iter()
            .map(|mut file| {
                let centrality_score = centrality_results
                    .pagerank_scores
                    .get(&file.relative_path)
                    .copied()
                    .unwrap_or(0.0);
                file.centrality_score = Some(centrality_score);
                (file, centrality_score)
            })
            .collect();

        // Sort by centrality score (highest first)
        source_with_centrality
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if std::env::var("SCRIBE_DEBUG").is_ok() && !source_with_centrality.is_empty() {
            eprintln!("🔍 Top 10 source files by centrality:");
            for (i, (file, score)) in source_with_centrality.iter().enumerate().take(10) {
                eprintln!("  {}. {} (score: {:.6})", i + 1, file.relative_path, score);
            }
        }

        for (file, centrality_score) in source_with_centrality {
            if budget_tracker.available() < 1 {
                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!("🛑 Budget exhausted, stopping source selection");
                }
                break;
            }

            if let Some(selected_file) = try_include_file_with_budget_and_demotion(
                file,
                &counter,
                &mut budget_tracker,
                centrality_score,
            )
            .await?
            {
                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!(
                        "✅ Selected {} (centrality: {:.4})",
                        selected_file.relative_path, centrality_score
                    );
                }
                selected_files.push(selected_file);
            }
        }
    }

    // Tier 3: Documentation files
    if !doc_files.is_empty() && budget_tracker.available() > 0 {
        if std::env::var("SCRIBE_DEBUG").is_ok() {
            eprintln!("📚 Tier 3: Processing documentation files");
        }

        // Sort docs by importance - prioritize architecture/design docs
        let mut critical_docs = Vec::new();
        let mut other_docs = Vec::new();

        for file in doc_files {
            let path_lower = file.relative_path.to_lowercase();
            if path_lower.contains("architecture")
                || path_lower.contains("design")
                || path_lower.contains("api")
                || path_lower.contains("spec")
                || path_lower.ends_with("changelog.md")
                || path_lower.ends_with("contributing.md")
            {
                critical_docs.push(file);
            } else {
                other_docs.push(file);
            }
        }

        // Process critical docs first, then others
        for file in critical_docs.into_iter().chain(other_docs.into_iter()) {
            if budget_tracker.available() < 1 {
                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!("🛑 Budget exhausted, stopping documentation selection");
                }
                break;
            }

            if let Some(selected_file) =
                try_include_file_with_budget(file, &counter, &mut budget_tracker).await?
            {
                selected_files.push(selected_file);
            }
        }
    }

    // Tier 4: Other files (if budget remains)
    if !other_files.is_empty() && budget_tracker.available() > 0 {
        if std::env::var("SCRIBE_DEBUG").is_ok() {
            eprintln!("📄 Tier 4: Processing other files");
        }

        for file in other_files {
            if budget_tracker.available() < 1 {
                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!("🛑 Budget exhausted, stopping other file selection");
                }
                break;
            }

            if let Some(selected_file) =
                try_include_file_with_budget(file, &counter, &mut budget_tracker).await?
            {
                selected_files.push(selected_file);
            }
        }
    }

    // Final optimization pass: try to fill remaining budget with smaller files
    if budget_tracker.available() > 1 {
        if std::env::var("SCRIBE_DEBUG").is_ok() {
            eprintln!(
                "🔧 Final optimization pass: {} tokens remaining, searching for small files",
                budget_tracker.available()
            );
        }

        let included_paths: HashSet<String> = selected_files
            .iter()
            .map(|f| f.relative_path.clone())
            .collect();

        // Try to find any remaining files that could fit
        for file in &all_files {
            if budget_tracker.available() < 1 {
                break;
            }

            if included_paths.contains(&file.relative_path) || !file.decision.should_include() {
                continue;
            }

            // Quick estimate - try small files that might fit
            if file.size <= (budget_tracker.available() * 4) as u64 {
                if let Some(selected_file) =
                    try_include_file_with_budget(file.clone(), &counter, &mut budget_tracker)
                        .await?
                {
                    if std::env::var("SCRIBE_DEBUG").is_ok() {
                        eprintln!(
                            "🎯 Final pass: included {} ({} tokens)",
                            selected_file.relative_path,
                            selected_file.token_estimate.unwrap_or(0)
                        );
                    }
                    selected_files.push(selected_file);
                }
            }
        }
    }

    let tokens_used = token_budget - budget_tracker.available();
    let utilization = (tokens_used as f64 / token_budget as f64) * 100.0;

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!(
            "✅ Selected {} files ({} tokens / {} budget, {:.1}% utilized)",
            selected_files.len(),
            tokens_used,
            token_budget,
            utilization
        );

        if utilization < 90.0 {
            eprintln!(
                "⚠️  Budget utilization below 90% - {} tokens unused",
                budget_tracker.available()
            );
        }
    }

    Ok(selected_files)
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
        return depth <= 1;
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

async fn try_include_file_with_budget_and_demotion(
    mut file: FileInfo,
    counter: &TokenCounter,
    budget_tracker: &mut TokenBudget,
    centrality_score: f64,
) -> Result<Option<FileInfo>> {
    match load_file_content_safe(&file.path) {
        Ok(content) => match counter.estimate_file_tokens(&content, &file.path) {
            Ok(full_tokens) => {
                // Try full content first
                if budget_tracker.can_allocate(full_tokens) {
                    budget_tracker.allocate(full_tokens);
                    file.content = Some(content);
                    file.token_estimate = Some(full_tokens);
                    file.char_count = Some(file.content.as_ref().unwrap().chars().count());
                    file.line_count = Some(file.content.as_ref().unwrap().lines().count());
                    return Ok(Some(file));
                }

                // Full content doesn't fit - try demotion for source files
                if matches!(file.file_type, FileType::Source { .. }) {
                    if std::env::var("SCRIBE_DEBUG").is_ok() {
                        eprintln!(
                            "🔧 Trying demotion for {} ({} tokens → chunks/signatures)",
                            file.relative_path, full_tokens
                        );
                    }

                    if let Ok(mut demotion_engine) = DemotionEngine::new() {
                        if let Ok(chunk_result) = demotion_engine.demote_content(
                            &content,
                            &file.relative_path,
                            FidelityMode::Chunk,
                            Some(budget_tracker.available()),
                        ) {
                            if budget_tracker.can_allocate(chunk_result.demoted_tokens) {
                                budget_tracker.allocate(chunk_result.demoted_tokens);
                                file.content = Some(chunk_result.content);
                                file.token_estimate = Some(chunk_result.demoted_tokens);
                                file.char_count =
                                    Some(file.content.as_ref().unwrap().chars().count());
                                file.line_count =
                                    Some(file.content.as_ref().unwrap().lines().count());
                                if std::env::var("SCRIBE_DEBUG").is_ok() {
                                    eprintln!(
                                        "✅ Demoted {} to chunks ({} → {} tokens, {:.1}% compression, centrality: {:.4})",
                                        file.relative_path,
                                        full_tokens,
                                        chunk_result.demoted_tokens,
                                        chunk_result.compression_ratio * 100.0,
                                        centrality_score
                                    );
                                }
                                return Ok(Some(file));
                            }
                        }

                        if let Ok(sig_result) = demotion_engine.demote_content(
                            &content,
                            &file.relative_path,
                            FidelityMode::Signature,
                            None,
                        ) {
                            if budget_tracker.can_allocate(sig_result.demoted_tokens) {
                                budget_tracker.allocate(sig_result.demoted_tokens);
                                file.content = Some(sig_result.content);
                                file.token_estimate = Some(sig_result.demoted_tokens);
                                file.char_count =
                                    Some(file.content.as_ref().unwrap().chars().count());
                                file.line_count =
                                    Some(file.content.as_ref().unwrap().lines().count());
                                if std::env::var("SCRIBE_DEBUG").is_ok() {
                                    eprintln!(
                                        "✅ Demoted {} to signatures ({} → {} tokens, {:.1}% compression, centrality: {:.4})",
                                        file.relative_path,
                                        full_tokens,
                                        sig_result.demoted_tokens,
                                        sig_result.compression_ratio * 100.0,
                                        centrality_score
                                    );
                                }
                                return Ok(Some(file));
                            }
                        }
                    }
                }

                if std::env::var("SCRIBE_DEBUG").is_ok() {
                    eprintln!(
                        "⚠️  Skipping {} ({} tokens) - no demotion method fits budget",
                        file.relative_path, full_tokens
                    );
                }
                Ok(None)
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

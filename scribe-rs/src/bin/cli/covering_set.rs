//! Covering set computation and display for CLI

use git2::{DiffOptions, Repository};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use scribe::{analyze_and_select, extract_imports, Config, SelectionOptions};
use scribe_selection::{
    CoveringSetComputer, CoveringSetGranularity, CoveringSetOptions, CoveringSetResult,
    EntityQuery, EntityType,
};

use super::import_extraction::extract_imports_for_diff;
use super::xml_output::output_covering_set_xml;

/// Simple struct for building dependency graph from file info
pub struct CoveringScanFile {
    pub path: String,
    pub relative_path: String,
    pub imports: Vec<String>,
}

impl scribe_analysis::heuristics::ScanResult for CoveringScanFile {
    fn path(&self) -> &str { &self.path }
    fn relative_path(&self) -> &str { &self.relative_path }
    fn depth(&self) -> usize { self.relative_path.matches('/').count() }
    fn is_docs(&self) -> bool { false }
    fn is_readme(&self) -> bool { false }
    fn is_test(&self) -> bool { false }
    fn is_entrypoint(&self) -> bool { false }
    fn has_examples(&self) -> bool { false }
    fn priority_boost(&self) -> f64 { 0.0 }
    fn churn_score(&self) -> f64 { 0.0 }
    fn centrality_in(&self) -> f64 { 0.0 }
    fn imports(&self) -> Option<&[String]> {
        if self.imports.is_empty() { None } else { Some(&self.imports) }
    }
    fn doc_analysis(&self) -> Option<&scribe_analysis::heuristics::DocumentAnalysis> { None }
}

/// File metadata for diff covering set analysis (implements ScanResult)
#[derive(Debug, Clone)]
pub struct DiffScanFile {
    pub path: String,
    pub relative_path: String,
    pub depth: usize,
    pub is_docs: bool,
    pub is_readme: bool,
    pub is_test: bool,
    pub is_entrypoint: bool,
    pub has_examples: bool,
    pub priority_boost: f64,
    pub churn_score: f64,
    pub imports: Vec<String>,
}

impl scribe_analysis::heuristics::ScanResult for DiffScanFile {
    fn path(&self) -> &str { &self.path }
    fn relative_path(&self) -> &str { &self.relative_path }
    fn depth(&self) -> usize { self.depth }
    fn is_docs(&self) -> bool { self.is_docs }
    fn is_readme(&self) -> bool { self.is_readme }
    fn is_test(&self) -> bool { self.is_test }
    fn is_entrypoint(&self) -> bool { self.is_entrypoint }
    fn has_examples(&self) -> bool { self.has_examples }
    fn priority_boost(&self) -> f64 { self.priority_boost }
    fn churn_score(&self) -> f64 { self.churn_score }
    fn centrality_in(&self) -> f64 { 0.0 }
    fn imports(&self) -> Option<&[String]> { Some(&self.imports) }
    fn doc_analysis(&self) -> Option<&scribe_analysis::heuristics::DocumentAnalysis> { None }
}

/// Log message conditionally based on mode and verbosity
pub fn log_covering_set_progress(msg: &str, stdout_mode: bool, verbose_level: u8, use_info: bool) {
    if stdout_mode {
        return;
    }
    if use_info && verbose_level > 0 {
        info!("{}", msg);
    } else if !use_info || verbose_level == 0 {
        eprintln!("{}", msg);
    }
}

/// File count threshold for applying adaptive depth limiting.
/// Repos with more than this many files get an automatic max_depth cap.
const LARGE_REPO_FILE_THRESHOLD: usize = 1000;

/// Default max_depth for large repos when user doesn't specify one.
/// Depth 3 captures direct deps, their deps, and one more level.
const LARGE_REPO_DEFAULT_DEPTH: usize = 3;

/// Default max_depth for entity-level granularity (always applied).
/// Entity traversal is much more expensive, so we always limit depth.
const ENTITY_GRANULARITY_DEFAULT_DEPTH: usize = 3;

/// Compute adaptive max_depth based on repo size and granularity.
///
/// If user explicitly specified a depth, use it. Otherwise:
/// - For entity granularity: always cap at depth 3 (entity traversal is expensive)
/// - For repos with > 1000 files: cap at depth 3 to prevent exponential slowdown
/// - For smaller repos with file granularity: allow unlimited traversal
fn adaptive_max_depth(
    user_specified: Option<usize>,
    file_count: usize,
    is_entity_granularity: bool,
) -> Option<usize> {
    // If user explicitly set a depth, respect it
    if user_specified.is_some() {
        return user_specified;
    }

    // Entity-level granularity is always expensive, cap depth regardless of repo size
    if is_entity_granularity {
        return Some(ENTITY_GRANULARITY_DEFAULT_DEPTH);
    }

    // For large repos, apply a sensible default to prevent timeouts
    if file_count > LARGE_REPO_FILE_THRESHOLD {
        Some(LARGE_REPO_DEFAULT_DEPTH)
    } else {
        None // Allow unlimited for smaller repos
    }
}

/// Build entity query from name and options
pub fn build_entity_query(
    entity_name: &str,
    entity_type: Option<&str>,
    _exact_match: bool, // Now defaults to true in EntityQuery, kept for CLI compat
) -> EntityQuery {
    let mut query = EntityQuery::parse(entity_name);
    // exact_match defaults to true in for_file_entity to avoid substring false positives
    // The --exact-match CLI flag is now a no-op (always exact), preserved for backwards compat

    if let Some(t) = entity_type {
        query.entity_type = match t.to_lowercase().as_str() {
            "function" => Some(EntityType::Function),
            "class" => Some(EntityType::Class),
            "module" => Some(EntityType::Module),
            "interface" => Some(EntityType::Interface),
            "constant" => Some(EntityType::Constant),
            _ => None,
        };
    }
    query
}

/// Display covering set result in interactive mode
pub fn display_covering_set_result(
    result: &CoveringSetResult,
    entity_name: &str,
    granularity: CoveringSetGranularity,
) {
    if result.files.is_empty() && result.entities.is_empty() {
        display_not_found_error(entity_name);
        return;
    }

    display_target_info(result);

    if granularity == CoveringSetGranularity::Entity {
        display_entity_results(result);
    } else {
        display_file_results(result);
    }
}

/// Display error when target not found
fn display_not_found_error(entity_name: &str) {
    let has_entity = entity_name.contains(':') &&
        !(entity_name.len() > 1 && entity_name.chars().nth(1) == Some(':') &&
          entity_name.chars().next().unwrap().is_ascii_alphabetic());

    if has_entity {
        println!("\n❌ Target '{}' not found", entity_name);
        println!("   The file may not exist or the entity wasn't found in the file.");
    } else {
        println!("\n❌ File '{}' not found", entity_name);
        println!("   Try using a more specific path or check the file exists.");
    }
}

/// Display target entity/file info
fn display_target_info(result: &CoveringSetResult) {
    if let Some(target) = &result.target_entity {
        println!("\n✅ Found target entity:");
        println!("  • File     : {}", target.file_path);
        println!("  • Type     : {}", target.entity_type);
        println!("  • Name     : {}", target.entity_name);
        println!("  • Lines    : {}-{}", target.start_line, target.end_line);
        println!("  • Public   : {}", if target.is_public { "yes" } else { "no" });
    } else if let Some(first_file) = result.files.first() {
        println!("\n✅ Found target file: {}", first_file.path);
    }
}

/// Display entity-level covering set results
fn display_entity_results(result: &CoveringSetResult) {
    println!("\n📦 Covering set ({} entities):", result.entities.len());
    for (idx, entity) in result.entities.iter().enumerate() {
        let explanation = result
            .inclusion_reasons
            .get(&format!("{}::{}", entity.file_path, entity.name))
            .map(|s| s.as_str())
            .unwrap_or("Included");

        println!(
            "  {}. {}::{} ({}, distance: {}, reason: {})",
            idx + 1, entity.file_path, entity.name, entity.entity_type, entity.distance, explanation
        );
    }
    println!("\n📊 Statistics:");
    println!("  • Files examined    : {}", result.statistics.files_examined);
    println!("  • Entities selected : {}", result.statistics.entities_selected);
    println!("  • Max depth         : {}", result.statistics.max_depth_reached);
    println!("  • Limits reached    : {}", if result.statistics.limits_reached { "yes" } else { "no" });
}

/// Display file-level covering set results
fn display_file_results(result: &CoveringSetResult) {
    println!("\n📦 Covering set ({} files):", result.files.len());
    for (idx, file) in result.files.iter().enumerate() {
        let explanation = result.inclusion_reasons.get(&file.path).map(|s| s.as_str()).unwrap_or("Included");
        println!("  {}. {} (distance: {}, reason: {})", idx + 1, file.path, file.distance, explanation);
    }
    println!("\n📊 Statistics:");
    println!("  • Files examined  : {}", result.statistics.files_examined);
    println!("  • Files selected  : {}", result.statistics.files_selected);
    println!("  • Files excluded  : {}", result.statistics.files_excluded);
    println!("  • Max depth       : {}", result.statistics.max_depth_reached);
    println!("  • Limits reached  : {}", if result.statistics.limits_reached { "yes" } else { "no" });
}

pub async fn run_covering_set_mode(
    repo_dir: &Path,
    entity_name: &str,
    entity_type: Option<&str>,
    exact_match: bool,
    include_dependents: bool,
    max_depth: Option<usize>,
    max_files: Option<usize>,
    granularity: &str,
    stdout_mode: bool,
    verbose_level: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    log_covering_set_progress(&format!("🎯 Finding covering set for: {}", entity_name), stdout_mode, verbose_level, false);

    // Analyze repository
    let mut config = Config::default();
    config.general.working_dir = Some(repo_dir.to_path_buf());
    config.analysis.token_budget = None;

    let selection_options = SelectionOptions {
        token_target: 0,
        force_traditional: false,
        algorithm_name: Some("covering-set".to_string()),
        include_directory_map: false,
        query_hint: None,
    };

    log_covering_set_progress("📊 Scanning repository...", stdout_mode, verbose_level, true);
    let analysis_outcome = analyze_and_select(repo_dir, &config, &selection_options).await?;

    // Collect file contents and build scan files
    let mut file_contents = HashMap::new();
    let mut scan_files = Vec::new();

    for file_info in &analysis_outcome.analysis.files {
        if let Ok(content) = std::fs::read_to_string(&file_info.path) {
            let path_str = file_info.path.display().to_string();
            let imports = extract_imports(&content, &file_info.language);
            scan_files.push(CoveringScanFile {
                path: path_str.clone(),
                relative_path: file_info.relative_path.clone(),
                imports,
            });
            file_contents.insert(path_str, content);
        }
    }

    log_covering_set_progress(&format!("📁 Loaded {} files", file_contents.len()), stdout_mode, verbose_level, true);

    // Build dependency graph
    log_covering_set_progress("🔗 Building dependency graph...", stdout_mode, verbose_level, true);
    use scribe_graph::CentralityCalculator;
    let calculator = CentralityCalculator::new()?;
    let graph = calculator.build_graph_only(&scan_files)?;

    if !stdout_mode && verbose_level > 0 {
        eprintln!("🔗 Graph built with {} nodes, {} edges", graph.node_count(), graph.edge_count());
    }

    // Build query and options
    let query = build_entity_query(entity_name, entity_type, exact_match);
    let granularity_option = match granularity {
        "entity" => CoveringSetGranularity::Entity,
        _ => CoveringSetGranularity::File,
    };

    // Apply adaptive depth limit for large repos or entity granularity
    let is_entity_granularity = granularity_option == CoveringSetGranularity::Entity;
    let effective_depth = adaptive_max_depth(max_depth, file_contents.len(), is_entity_granularity);

    // Log if we applied adaptive depth limiting
    if max_depth.is_none() && effective_depth.is_some() {
        let reason = if is_entity_granularity {
            "entity granularity".to_string()
        } else {
            format!("{} files", file_contents.len())
        };
        log_covering_set_progress(
            &format!("⚡ Limiting depth to {} for performance ({})",
                effective_depth.unwrap(), reason),
            stdout_mode, verbose_level, true
        );
    }

    let options = CoveringSetOptions {
        include_dependencies: true,
        include_dependents,
        max_depth: effective_depth,
        max_files,
        min_importance: None,
        granularity: granularity_option,
    };

    log_covering_set_progress("🔍 Computing covering set...", stdout_mode, verbose_level, true);

    // Compute covering set
    let mut computer = CoveringSetComputer::new()?;
    let result = computer.compute_covering_set(&query, &file_contents, &graph, &options)?;

    if stdout_mode {
        return output_covering_set_xml(&result, &file_contents, granularity_option);
    }

    display_covering_set_result(&result, entity_name, granularity_option);

    if verbose_level > 0 {
        info!("✨ Covering set computation complete");
    }

    Ok(())
}

/// Collect changed files from git diff
fn collect_changed_files_from_diff(
    repo: &Repository,
    workdir: &Path,
    diff_against: Option<&str>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut diff_opts = DiffOptions::new();
    diff_opts.include_untracked(true).recurse_untracked_dirs(true);

    let mut changed_files = std::collections::HashSet::new();

    if let Some(reference) = diff_against {
        let obj = repo.revparse_single(reference)?;
        let commit = obj.peel_to_commit()?;
        let tree = commit.tree()?;
        let diff = repo.diff_tree_to_workdir_with_index(Some(&tree), Some(&mut diff_opts))?;
        for delta in diff.deltas() {
            if let Some(path) = delta.new_file().path().or_else(|| delta.old_file().path()) {
                changed_files.insert(workdir.join(path).to_string_lossy().to_string());
            }
        }
    } else {
        let diff = repo.diff_index_to_workdir(None, Some(&mut diff_opts))?;
        for delta in diff.deltas() {
            if let Some(path) = delta.new_file().path().or_else(|| delta.old_file().path()) {
                changed_files.insert(workdir.join(path).to_string_lossy().to_string());
            }
        }
    }

    Ok(changed_files.into_iter().collect())
}

/// Build DiffScanFile metadata from file info
fn build_diff_scan_file(file: &scribe::FileInfo) -> DiffScanFile {
    use scribe_core::file::{is_entrypoint_path, is_test_path, FileType};
    use scribe_core::Language;

    let extension = file.path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    let language = Language::from_extension(extension);
    let content = if file.is_binary {
        String::new()
    } else {
        file.content
            .clone()
            .or_else(|| std::fs::read_to_string(&file.path).ok())
            .unwrap_or_default()
    };

    let imports = if file.is_binary {
        Vec::new()
    } else {
        extract_imports_for_diff(&content, &language)
    };

    let relative_path = file.relative_path.clone();
    let depth = relative_path.matches('/').count();
    let path_lower = relative_path.to_lowercase();

    DiffScanFile {
        path: file.path.to_string_lossy().to_string(),
        relative_path,
        depth,
        is_docs: matches!(file.file_type, FileType::Documentation { .. }),
        is_readme: path_lower.contains("readme"),
        is_test: is_test_path(&file.path),
        is_entrypoint: is_entrypoint_path(&file.path, &language),
        has_examples: path_lower.contains("example"),
        priority_boost: 0.0,
        churn_score: 0.0,
        imports,
    }
}

/// Log message with appropriate verbosity
fn log_covering_set_msg(verbose: bool, info_msg: &str, normal_msg: &str) {
    if verbose {
        info!("{}", info_msg);
    } else {
        println!("{}", normal_msg);
    }
}

/// Build dependency graph from scan files
fn build_dependency_graph(
    diff_scan_files: &[DiffScanFile],
) -> Result<scribe_graph::DependencyGraph, Box<dyn std::error::Error>> {
    use scribe_analysis::heuristics::ScanResult;
    use scribe_graph::centrality::{ImportDetector, ImportResolutionConfig};
    use scribe_graph::DependencyGraph;

    let mut graph = DependencyGraph::with_capacity(diff_scan_files.len());
    for file in diff_scan_files {
        graph.add_node(file.path.clone())?;
    }

    let detector =
        ImportDetector::with_file_index(ImportResolutionConfig::default(), diff_scan_files);
    let file_map: HashMap<&str, &DiffScanFile> = diff_scan_files
        .iter()
        .map(|f| (f.path.as_str(), f))
        .collect();

    for file in diff_scan_files {
        if let Some(imports) = file.imports() {
            for import_str in imports {
                if let Some(resolved) = detector.resolve_import(import_str, &file.path, &file_map) {
                    graph.add_edge(file.path.clone(), resolved)?;
                }
            }
        }
    }

    Ok(graph)
}

/// Create covering set options from parameters (for diff mode, always file granularity)
fn create_covering_set_options(
    include_dependents: bool,
    max_depth: Option<usize>,
    max_files: Option<usize>,
    file_count: usize,
) -> CoveringSetOptions {
    // Apply adaptive depth limit for large repos to prevent exponential slowdown
    // Diff mode is always file granularity, so pass false for is_entity_granularity
    let effective_depth = adaptive_max_depth(max_depth, file_count, false);

    CoveringSetOptions {
        include_dependencies: true,
        include_dependents,
        max_depth: effective_depth,
        max_files,
        min_importance: None,
        granularity: CoveringSetGranularity::File,
    }
}

pub async fn run_covering_set_diff_mode(
    repo_dir: &Path,
    diff_against: Option<&str>,
    include_dependents: bool,
    max_depth: Option<usize>,
    max_files: Option<usize>,
    verbose_level: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = verbose_level > 0;
    log_covering_set_msg(verbose, "🎯 Covering set (diff) mode", "🎯 Computing covering set for git diff");

    let repo = Repository::open(repo_dir)?;
    let workdir = repo.workdir().unwrap_or(repo_dir);
    let changed_files = collect_changed_files_from_diff(&repo, workdir, diff_against)?;

    if changed_files.is_empty() {
        println!("❌ No changes detected in the diff");
        return Ok(());
    }

    log_covering_set_msg(
        verbose,
        &format!("📁 {} changed files detected", changed_files.len()),
        &format!("📁 {} changed files detected", changed_files.len()),
    );

    // Run analysis pipeline
    let mut config = Config::default();
    config.general.working_dir = Some(repo_dir.to_path_buf());
    config.analysis.token_budget = None;
    let selection_options = SelectionOptions {
        token_target: 0,
        force_traditional: true,
        algorithm_name: Some("covering-set-diff".to_string()),
        include_directory_map: false,
        query_hint: None,
    };
    let analysis_outcome = analyze_and_select(repo_dir, &config, &selection_options).await?;

    // Build scan files and dependency graph
    let diff_scan_files: Vec<DiffScanFile> = analysis_outcome
        .analysis
        .files
        .iter()
        .map(build_diff_scan_file)
        .collect();
    let graph = build_dependency_graph(&diff_scan_files)?;

    // Compute and display covering set
    let options = create_covering_set_options(include_dependents, max_depth, max_files, diff_scan_files.len());
    let computer = CoveringSetComputer::new()?;
    let result = computer.compute_covering_set_for_files(&changed_files, &graph, None, &options)?;
    print_covering_set_results(&result, verbose_level);

    Ok(())
}

/// Print covering set results to stdout
fn print_covering_set_results(result: &CoveringSetResult, verbose_level: u8) {
    println!("\n📦 Covering set for diff ({} files):", result.files.len());
    for (idx, file) in result.files.iter().enumerate() {
        let explanation = result
            .inclusion_reasons
            .get(&file.path)
            .map(|s| s.as_str())
            .unwrap_or("Included");

        println!(
            "  {}. {} (distance: {}, reason: {})",
            idx + 1,
            file.path,
            file.distance,
            explanation
        );
    }

    println!("\n📊 Statistics:");
    println!("  • Files examined  : {}", result.statistics.files_examined);
    println!("  • Files selected  : {}", result.statistics.files_selected);
    println!("  • Files excluded  : {}", result.statistics.files_excluded);
    println!(
        "  • Max depth       : {}",
        result.statistics.max_depth_reached
    );
    println!(
        "  • Limits reached  : {}",
        if result.statistics.limits_reached {
            "yes"
        } else {
            "no"
        }
    );

    if verbose_level > 0 {
        info!("✨ Diff covering set computation complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_max_depth_user_specified() {
        // User-specified depth should always be respected regardless of repo size or granularity
        assert_eq!(adaptive_max_depth(Some(5), 500, false), Some(5));
        assert_eq!(adaptive_max_depth(Some(5), 2000, false), Some(5));
        assert_eq!(adaptive_max_depth(Some(0), 2000, true), Some(0));
        assert_eq!(adaptive_max_depth(Some(10), 100, true), Some(10));
    }

    #[test]
    fn test_adaptive_max_depth_small_repo_file_granularity() {
        // Small repos (<=1000 files) with file granularity should have unlimited depth
        assert_eq!(adaptive_max_depth(None, 100, false), None);
        assert_eq!(adaptive_max_depth(None, 500, false), None);
        assert_eq!(adaptive_max_depth(None, 1000, false), None);
    }

    #[test]
    fn test_adaptive_max_depth_large_repo() {
        // Large repos (>1000 files) should be capped at default depth
        assert_eq!(adaptive_max_depth(None, 1001, false), Some(LARGE_REPO_DEFAULT_DEPTH));
        assert_eq!(adaptive_max_depth(None, 5000, false), Some(LARGE_REPO_DEFAULT_DEPTH));
        assert_eq!(adaptive_max_depth(None, 12000, false), Some(LARGE_REPO_DEFAULT_DEPTH));
    }

    #[test]
    fn test_adaptive_max_depth_entity_granularity() {
        // Entity granularity should always be capped regardless of repo size
        assert_eq!(adaptive_max_depth(None, 100, true), Some(ENTITY_GRANULARITY_DEFAULT_DEPTH));
        assert_eq!(adaptive_max_depth(None, 500, true), Some(ENTITY_GRANULARITY_DEFAULT_DEPTH));
        assert_eq!(adaptive_max_depth(None, 2000, true), Some(ENTITY_GRANULARITY_DEFAULT_DEPTH));
    }

    #[test]
    fn test_threshold_constants() {
        // Verify constants are sensible
        assert_eq!(LARGE_REPO_FILE_THRESHOLD, 1000);
        assert_eq!(LARGE_REPO_DEFAULT_DEPTH, 3);
        assert_eq!(ENTITY_GRANULARITY_DEFAULT_DEPTH, 3);
    }
}

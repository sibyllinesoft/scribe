use clap::{Arg, ArgAction, Command, ValueEnum};
use git2::{DiffOptions, Repository};
use serde_json::{self, json};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use tempfile::TempDir;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};
use url::Url;

#[cfg(feature = "web")]
use async_trait::async_trait;
#[cfg(feature = "web")]
use scribe_webservice::{
    AnalysisOutput, AnalysisProvider, WebReportFile, WebSelectionMetrics, WebService,
    WebServiceConfig, WebServiceError,
};

// Import the main library functions
use scribe::{
    analyze_and_select, format_bytes, format_timestamp, generate_report, get_file_icon, Config,
    ReportFile, ReportFormat, SelectionMetrics, SelectionOptions,
};

async fn clone_github_repo(
    url: &str,
) -> Result<(PathBuf, Option<TempDir>), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    Repository::clone(url, temp_dir.path())?;
    Ok((temp_dir.path().to_path_buf(), Some(temp_dir)))
}

/// Simple struct for building dependency graph from file info
struct CoveringScanFile {
    path: String,
    relative_path: String,
    imports: Vec<String>,
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

/// Log message conditionally based on mode and verbosity
fn log_covering_set_progress(msg: &str, stdout_mode: bool, verbose_level: u8, use_info: bool) {
    if stdout_mode {
        return;
    }
    if use_info && verbose_level > 0 {
        info!("{}", msg);
    } else if !use_info || verbose_level == 0 {
        eprintln!("{}", msg);
    }
}

/// Build entity query from name and options
fn build_entity_query(
    entity_name: &str,
    entity_type: Option<&str>,
    exact_match: bool,
) -> scribe_selection::EntityQuery {
    use scribe_selection::{EntityQuery, EntityType};

    let mut query = EntityQuery::parse(entity_name);
    query.exact_match = exact_match;

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
fn display_covering_set_result(
    result: &scribe_selection::CoveringSetResult,
    entity_name: &str,
    granularity: scribe_selection::CoveringSetGranularity,
) {
    if result.files.is_empty() && result.entities.is_empty() {
        display_not_found_error(entity_name);
        return;
    }

    display_target_info(result);

    if granularity == scribe_selection::CoveringSetGranularity::Entity {
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
fn display_target_info(result: &scribe_selection::CoveringSetResult) {
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
fn display_entity_results(result: &scribe_selection::CoveringSetResult) {
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
fn display_file_results(result: &scribe_selection::CoveringSetResult) {
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

async fn run_covering_set_mode(
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
    use scribe_selection::{CoveringSetComputer, CoveringSetGranularity, CoveringSetOptions};
    use scribe::{analyze_and_select, extract_imports};
    use std::collections::HashMap;

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

    let options = CoveringSetOptions {
        include_dependencies: true,
        include_dependents,
        max_depth,
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

/// Output covering set result as XML to stdout (for agent consumption)
fn output_covering_set_xml(
    result: &scribe_selection::CoveringSetResult,
    file_contents: &std::collections::HashMap<String, String>,
    granularity: scribe_selection::CoveringSetGranularity,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    writeln!(handle, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(handle, "<covering_set>")?;

    write_xml_target(&mut handle, &result.target_entity)?;

    if granularity == scribe_selection::CoveringSetGranularity::Entity {
        write_xml_entities(&mut handle, &result.entities)?;
    } else {
        write_xml_files(&mut handle, &result.files, file_contents)?;
    }

    write_xml_statistics(&mut handle, &result.statistics)?;
    writeln!(handle, "</covering_set>")?;

    Ok(())
}

/// Write XML target element
fn write_xml_target<W: std::io::Write>(
    handle: &mut W,
    target: &Option<scribe_selection::EntityLocation>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    if let Some(target) = target {
        writeln!(handle, "  <target>")?;
        writeln!(handle, "    <file>{}</file>", escape_xml(&target.file_path))?;
        writeln!(handle, "    <name>{}</name>", escape_xml(&target.entity_name))?;
        writeln!(handle, "    <type>{}</type>", escape_xml(&target.entity_type))?;
        writeln!(handle, "    <lines start=\"{}\" end=\"{}\"/>", target.start_line, target.end_line)?;
        writeln!(handle, "  </target>")?;
    }
    Ok(())
}

/// Write XML entities element
fn write_xml_entities<W: std::io::Write>(
    handle: &mut W,
    entities: &[scribe_selection::CoveringSetEntity],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    writeln!(handle, "  <entities count=\"{}\">", entities.len())?;
    for entity in entities {
        writeln!(handle, "    <entity>")?;
        writeln!(handle, "      <file>{}</file>", escape_xml(&entity.file_path))?;
        writeln!(handle, "      <name>{}</name>", escape_xml(&entity.name))?;
        writeln!(handle, "      <type>{}</type>", escape_xml(&entity.entity_type))?;
        writeln!(handle, "      <lines start=\"{}\" end=\"{}\"/>", entity.start_line, entity.end_line)?;
        writeln!(handle, "      <distance>{}</distance>", entity.distance)?;
        writeln!(handle, "      <reason>{:?}</reason>", entity.reason)?;
        writeln!(handle, "      <content><![CDATA[{}]]></content>", entity.content)?;
        writeln!(handle, "    </entity>")?;
    }
    writeln!(handle, "  </entities>")?;
    Ok(())
}

/// Write XML files element
fn write_xml_files<W: std::io::Write>(
    handle: &mut W,
    files: &[scribe_selection::CoveringSetFile],
    file_contents: &std::collections::HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    writeln!(handle, "  <files count=\"{}\">", files.len())?;
    for file in files {
        let content = file_contents.get(&file.path).map(|s| s.as_str()).unwrap_or("");
        writeln!(handle, "    <file>")?;
        writeln!(handle, "      <path>{}</path>", escape_xml(&file.path))?;
        writeln!(handle, "      <distance>{}</distance>", file.distance)?;
        writeln!(handle, "      <reason>{:?}</reason>", file.reason)?;
        writeln!(handle, "      <content><![CDATA[{}]]></content>", content)?;
        writeln!(handle, "    </file>")?;
    }
    writeln!(handle, "  </files>")?;
    Ok(())
}

/// Write XML statistics element
fn write_xml_statistics<W: std::io::Write>(
    handle: &mut W,
    stats: &scribe_selection::CoveringSetStatistics,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    writeln!(handle, "  <statistics>")?;
    writeln!(handle, "    <files_examined>{}</files_examined>", stats.files_examined)?;
    writeln!(handle, "    <files_selected>{}</files_selected>", stats.files_selected)?;
    writeln!(handle, "    <entities_selected>{}</entities_selected>", stats.entities_selected)?;
    writeln!(handle, "    <max_depth_reached>{}</max_depth_reached>", stats.max_depth_reached)?;
    writeln!(handle, "    <limits_reached>{}</limits_reached>", stats.limits_reached)?;
    writeln!(handle, "  </statistics>")?;
    Ok(())
}

/// Escape special XML characters
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// File metadata for diff covering set analysis (implements ScanResult)
#[derive(Debug, Clone)]
struct DiffScanFile {
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
    imports: Vec<String>,
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

/// Extract imports from file content based on language
fn extract_imports_for_diff(content: &str, language: &scribe_core::Language) -> Vec<String> {
    use scribe_core::Language;
    use std::collections::HashSet;
    let mut imports = HashSet::new();

    match language {
        Language::Rust => extract_rust_imports(content, &mut imports),
        Language::Python => extract_python_imports(content, &mut imports),
        Language::JavaScript | Language::TypeScript => extract_js_imports(content, &mut imports),
        Language::Go => extract_go_imports(content, &mut imports),
        _ => {}
    }

    let mut ordered: Vec<String> = imports.into_iter().collect();
    ordered.sort();
    ordered.truncate(64);
    ordered
}

fn extract_rust_imports(content: &str, imports: &mut std::collections::HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("use ") {
            let statement = trimmed
                .trim_start_matches("use ")
                .trim_end_matches(';')
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_end_matches("::");
            if !statement.is_empty() {
                imports.insert(statement.to_string());
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

fn extract_python_imports(content: &str, imports: &mut std::collections::HashSet<String>) {
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

fn extract_js_imports(content: &str, imports: &mut std::collections::HashSet<String>) {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            if let Some(start) = trimmed.find('"') {
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

fn extract_go_imports(content: &str, imports: &mut std::collections::HashSet<String>) {
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
            let import_path = trimmed.trim_matches(&['"', '`'][..]);
            if !import_path.is_empty() {
                imports.insert(import_path.to_string());
            }
        } else if trimmed.starts_with("import ") {
            let import_path = trimmed
                .trim_start_matches("import ")
                .trim_matches(&['"', '`'][..]);
            if !import_path.is_empty() {
                imports.insert(import_path.to_string());
            }
        }
    }
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
    let file_map: std::collections::HashMap<&str, &DiffScanFile> = diff_scan_files
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

/// Create covering set options from parameters
fn create_covering_set_options(
    include_dependents: bool,
    max_depth: Option<usize>,
    max_files: Option<usize>,
) -> scribe_selection::CoveringSetOptions {
    scribe_selection::CoveringSetOptions {
        include_dependencies: true,
        include_dependents,
        max_depth,
        max_files,
        min_importance: None,
        granularity: scribe_selection::CoveringSetGranularity::File,
    }
}

async fn run_covering_set_diff_mode(
    repo_dir: &Path,
    diff_against: Option<&str>,
    include_dependents: bool,
    max_depth: Option<usize>,
    max_files: Option<usize>,
    verbose_level: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    use scribe_selection::CoveringSetComputer;

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
    let options = create_covering_set_options(include_dependents, max_depth, max_files);
    let computer = CoveringSetComputer::new()?;
    let result = computer.compute_covering_set_for_files(&changed_files, &graph, None, &options)?;
    print_covering_set_results(&result, verbose_level);

    Ok(())
}

/// Print covering set results to stdout
fn print_covering_set_results(result: &scribe_selection::CoveringSetResult, verbose_level: u8) {
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

#[cfg(feature = "web")]
struct CliAnalysisProvider;

#[cfg(feature = "web")]
#[async_trait]
impl AnalysisProvider for CliAnalysisProvider {
    async fn analyze(
        &self,
        config: &WebServiceConfig,
    ) -> std::result::Result<AnalysisOutput, WebServiceError> {
        let mut scribe_config = Config::default();
        scribe_config.filtering.max_file_size = config.max_file_size as u64;
        scribe_config.features.auto_exclude_tests = config.auto_exclude_tests;
        scribe_config.analysis.token_budget = None;
        scribe_config.general.working_dir = Some(config.repo_path.clone());

        let selection_options = SelectionOptions {
            token_target: config.token_budget,
            force_traditional: config.token_budget == 0,
            algorithm_name: Some("web-service".to_string()),
            include_directory_map: true,
        };

        let outcome = analyze_and_select(&config.repo_path, &scribe_config, &selection_options)
            .await
            .map_err(|err| WebServiceError::ScribeCore(err.to_string()))?;

        let selected_files = outcome
            .selection
            .selected_files
            .into_iter()
            .map(convert_report_file)
            .collect();

        let metrics = convert_selection_metrics(outcome.selection.metrics);

        Ok(AnalysisOutput {
            selected_files,
            selected_file_infos: outcome.selection.selected_file_infos,
            metrics,
            repository_files: outcome.analysis.files,
            token_budget: config.token_budget,
        })
    }
}

#[cfg(feature = "web")]
fn convert_report_file(file: ReportFile) -> WebReportFile {
    WebReportFile {
        path: file.path,
        relative_path: file.relative_path,
        content: file.content,
        size: file.size,
        estimated_tokens: file.estimated_tokens,
        importance_score: file.importance_score,
        centrality_score: file.centrality_score,
        query_relevance_score: file.query_relevance_score,
        entry_point_proximity: file.entry_point_proximity,
        content_quality_score: file.content_quality_score,
        repository_role_score: file.repository_role_score,
        recency_score: file.recency_score,
        modified: format_timestamp(file.modified),
    }
}

#[cfg(feature = "web")]
fn convert_selection_metrics(metrics: SelectionMetrics) -> WebSelectionMetrics {
    WebSelectionMetrics {
        total_files_discovered: metrics.total_files_discovered,
        files_selected: metrics.files_selected,
        total_tokens_estimated: metrics.total_tokens_estimated,
        selection_time_ms: metrics.selection_time_ms,
        algorithm_used: metrics.algorithm_used,
        coverage_score: metrics.coverage_score,
        relevance_score: metrics.relevance_score,
    }
}

#[cfg(feature = "web")]
async fn launch_editor_mode(
    repo_dir: &Path,
    token_budget: usize,
    max_bytes: usize,
    no_exclude_tests: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::net::TcpListener;

    info!("Launching embedded web editor for {}", repo_dir.display());

    let host = "127.0.0.1";
    let mut candidate_port = 5000u16;
    let chosen = loop {
        match TcpListener::bind((host, candidate_port)) {
            Ok(listener) => break Some((candidate_port, listener)),
            Err(_) => {
                candidate_port = candidate_port.saturating_add(1);
                if candidate_port >= 6000 {
                    break None;
                }
            }
        }
    };

    let (port, listener) = match chosen {
        Some(value) => value,
        None => return Err("No available ports in range 5000-5999".into()),
    };
    drop(listener);

    let config = WebServiceConfig {
        port,
        host: host.to_string(),
        repo_path: repo_dir.to_path_buf(),
        token_budget,
        auto_open_browser: true,
        max_file_size: max_bytes,
        auto_exclude_tests: !no_exclude_tests,
        ..WebServiceConfig::default()
    };

    info!(
        "Starting web editor at http://{}:{} (token budget: {}, max bytes: {})",
        config.host, config.port, token_budget, max_bytes
    );

    let provider = Arc::new(CliAnalysisProvider);
    let mut service = WebService::new(config, provider)?;
    service.start().await?;

    info!("Web editor session finished");
    Ok(())
}

#[cfg(not(feature = "web"))]
async fn launch_editor_mode(
    _repo_dir: &Path,
    _token_budget: usize,
    _max_bytes: usize,
    _no_exclude_tests: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    Err(
        "The --editor option requires the `web` feature. Rebuild Scribe with --features web."
            .into(),
    )
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Html,
    Repomix,
    Xml,
    Json,
    Text,
    Markdown,
}

impl From<OutputFormat> for ReportFormat {
    fn from(value: OutputFormat) -> Self {
        match value {
            OutputFormat::Html => ReportFormat::Html,
            OutputFormat::Repomix => ReportFormat::Repomix,
            OutputFormat::Xml => ReportFormat::Xml,
            OutputFormat::Json => ReportFormat::Json,
            OutputFormat::Text => ReportFormat::Text,
            OutputFormat::Markdown => ReportFormat::Markdown,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Algorithm {
    #[value(name = "v1-baseline")]
    V1Baseline,
    #[value(name = "v3-centrality")]
    V3Centrality,
    #[value(name = "v4-demotion")]
    V4Demotion,
    #[value(name = "v5-integrated")]
    V5Integrated,
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        info!("CLI main started in debug mode");
    }
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let app = Command::new("scribe")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Nathan Rice <nathan@sibylline.dev>")
        .about("Scribe: Intelligent repository tool")
        .long_about("Scribe is a comprehensive tool that intelligently selects and processes repository files for AI consumption. It provides multiple output formats and uses advanced algorithms to optimize file selection within token budgets.")
        .arg(
            Arg::new("repo_path")
                .help("Repository path to analyze (local directory or GitHub URL)")
                .value_name("PATH_OR_URL")
                .default_value(".")
                .index(1),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("out")
                .alias("output")
                .help("Output file path (auto-generated if not specified)")
                .value_name("FILE"),
        )
        .arg(
            Arg::new("output_format")
                .long("output-format")
                .help("Output format: html, xml, json, text, markdown, repomix (default: html)")
                .value_parser(clap::value_parser!(OutputFormat))
                .default_value("html"),
        )
        .arg(
            Arg::new("line_numbers")
                .long("line-numbers")
                .help("Prefix each line of bundled files with its line number")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("token_target")
                .long("token-target")
                .alias("token-budget")
                .help("Target token count for intelligent selection (default: 128000)")
                .value_name("TOKENS")
                .default_value("128000")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("max_bytes")
                .long("max-bytes")
                .help("Maximum file size to consider (in bytes)")
                .value_name("BYTES")
                .default_value("204800") // 200KB
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("include")
                .long("include")
                .help("Comma-separated glob patterns for files to include")
                .value_name("PATTERNS"),
        )
        .arg(
            Arg::new("exclude")
                .long("exclude")
                .help("Comma-separated glob patterns for files to exclude")
                .value_name("PATTERNS"),
        )
        .arg(
            Arg::new("exclude_tests")
                .long("exclude-tests")
                .help("Exclude test files from selection (tests/, *_test.*, *.test.*, *.spec.*)")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no_exclude_tests")
                .long("no-exclude-tests")
                .help("Include test files even when they would normally be excluded")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ignore")
                .long("ignore")
                .help("Comma-separated glob patterns for files to ignore")
                .value_name("PATTERNS"),
        )
        .arg(
            Arg::new("no_gitignore")
                .long("no-gitignore")
                .help("Disable .gitignore handling during scanning")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no_default_patterns")
                .long("no-default-patterns")
                .help("Disable built-in ignore patterns like node_modules or target")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(ArgAction::Count),
        )
        // Advanced mode selection
        .arg(
            Arg::new("force_traditional")
                .long("force-traditional")
                .help("Force traditional file filtering instead of intelligent selection")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("editor")
                .long("editor")
                .help("Launch interactive bundle editor in browser")
                .action(ArgAction::SetTrue),
        )
        // Intelligent selection algorithm options
        .arg(
            Arg::new("algorithm")
                .long("algorithm")
                .alias("variant")
                .help("Selection algorithm")
                .value_parser(clap::value_parser!(Algorithm))
                .default_value("v5-integrated"),
        )
        .arg(
            Arg::new("query_hint")
                .long("query-hint")
                .help("Query hint to guide file selection (e.g., authentication, database)")
                .value_name("HINT"),
        )
        .arg(
            Arg::new("show_metrics")
                .long("show-metrics")
                .help("Show detailed performance and quality metrics")
                .action(ArgAction::SetTrue),
        )
        // Entry point relevance
        .arg(
            Arg::new("entry_points")
                .long("entry-points")
                .help("Focus on specific entry point files")
                .value_name("FILES")
                .num_args(0..),
        )
        .arg(
            Arg::new("entry_functions")
                .long("entry-functions")
                .help("Focus on specific functions (format: file.py:function_name)")
                .value_name("FUNCTIONS")
                .num_args(0..),
        )
        .arg(
            Arg::new("personalization_alpha")
                .long("personalization-alpha")
                .help("Entry point focus strength (0.0-1.0)")
                .value_name("ALPHA")
                .default_value("0.15")
                .value_parser(clap::value_parser!(f64)),
        )
        // Git integration
        .arg(
            Arg::new("include_diffs")
                .long("include-diffs")
                .help("Include relevant Git diffs")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("diff_commits")
                .long("diff-commits")
                .help("Number of recent commits to analyze")
                .value_name("COUNT")
                .default_value("1")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("diff_branch")
                .long("diff-branch")
                .help("Compare with specific branch")
                .value_name("BRANCH"),
        )
        .arg(
            Arg::new("diff_relevance_threshold")
                .long("diff-relevance-threshold")
                .help("Minimum relevance score for including diffs")
                .value_name("THRESHOLD")
                .default_value("0.1")
                .value_parser(clap::value_parser!(f64)),
        )
        // Scaling optimization flag
        .arg(
            Arg::new("scaling")
                .long("scaling")
                .help("Enable advanced scaling optimizations for large repositories")
                .action(ArgAction::SetTrue),
        )
        // Covering set mode
        .arg(
            Arg::new("covering_set")
                .long("covering-set")
                .help("Find covering set for a file or entity. Use 'file' or 'file:entity' syntax (e.g., 'src/auth.rs' or 'src/auth.rs:login')")
                .value_name("TARGET"),
        )
        .arg(
            Arg::new("covering_set_diff")
                .long("covering-set-diff")
                .help("Compute covering set for the current git diff")
                .action(ArgAction::SetTrue)
                .conflicts_with("covering_set"),
        )
        .arg(
            Arg::new("diff_against")
                .long("diff-against")
                .help("Git ref to diff against (defaults to HEAD)")
                .value_name("REF")
                .requires("covering_set_diff"),
        )
        .arg(
            Arg::new("entity_type")
                .long("entity-type")
                .help("Type of entity to find: function, class, module, interface, constant")
                .value_name("TYPE")
                .requires("covering_set"),
        )
        .arg(
            Arg::new("exact_match")
                .long("exact-match")
                .help("Match entity name exactly (vs substring match)")
                .action(ArgAction::SetTrue)
                .requires("covering_set"),
        )
        .arg(
            Arg::new("include_dependents")
                .long("include-dependents")
                .help("Include files that depend on the target (for impact analysis)")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("max_depth")
                .long("max-depth")
                .help("Maximum dependency traversal depth")
                .value_name("DEPTH")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("max_files_covering")
                .long("max-files")
                .help("Maximum number of files in covering set")
                .value_name("COUNT")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("granularity")
                .long("granularity")
                .help("Covering set granularity: 'file' returns whole files, 'entity' returns only specific functions/classes")
                .value_parser(["file", "entity"])
                .default_value("file"),
        )
        // Agent/stdout mode
        .arg(
            Arg::new("stdout")
                .long("stdout")
                .help("Output directly to stdout (for agent/pipeline use). Defaults to XML format")
                .action(ArgAction::SetTrue),
        )
        .after_help(r#"AGENT USAGE:
  Scribe can be used by AI agents to quickly retrieve relevant code context.

  EXAMPLES:
    # Get covering set for a file (all dependencies)
    scribe --covering-set "src/auth.rs" --stdout

    # Get covering set for a specific function within a file
    scribe --covering-set "src/auth.rs:login" --stdout

    # Entity-level granularity (returns only specific functions/classes)
    scribe --covering-set "src/service.rs:UserService" --granularity entity --stdout

    # Windows paths work too (rightmost colon is the separator)
    scribe --covering-set "C:\project\auth.rs:login" --stdout

    # Analyze what code is affected by recent changes
    scribe --covering-set-diff --stdout

    # Get full repository context within token budget
    scribe --token-target 50000 --stdout --output-format xml

  COVERING SET OPTIONS:
    --granularity file    Return whole files (default, faster)
    --granularity entity  Return only specific functions/classes (more precise)
    --include-dependents  Include code that depends on target (impact analysis)
    --max-depth N         Limit dependency traversal depth
    --max-files N         Limit number of results

  TARGET FORMAT:
    file                  Covering set for entire file
    file:entity           Covering set for specific entity within file

  OUTPUT FORMATS FOR AGENTS:
    xml      Structured XML with metadata (recommended)
    json     JSON array of files with content
    text     Plain text with file separators
"#);

    let matches = app.get_matches();

    // Parse arguments
    let repo_path_or_url = matches.get_one::<String>("repo_path").unwrap();
    let output_format = matches.get_one::<OutputFormat>("output_format").unwrap();
    let report_format: ReportFormat = (*output_format).into();
    let token_target = *matches.get_one::<usize>("token_target").unwrap();
    let max_bytes = *matches.get_one::<usize>("max_bytes").unwrap();
    let verbose_level = matches.get_count("verbose");
    let include_line_numbers = matches.get_flag("line_numbers");

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        info!("Verbose level set to {}", verbose_level);
    }

    // Normalize repository location (local path or cloned GitHub temp dir)
    let (repo_dir, _temp_repo_guard) =
        if repo_path_or_url.starts_with("http://") || repo_path_or_url.starts_with("https://") {
            info!("🌐 Detected GitHub URL: {}", repo_path_or_url);
            clone_github_repo(repo_path_or_url).await?
        } else {
            let path = PathBuf::from(repo_path_or_url);
            if !path.exists() {
                error!("Repository path does not exist: {}", repo_path_or_url);
                process::exit(1);
            }
            if !path.is_dir() {
                error!("Repository path is not a directory: {}", repo_path_or_url);
                process::exit(1);
            }
            (path.canonicalize()?, None)
        };

    // Check for editor mode IMMEDIATELY - before any analysis
    let editor_mode = matches.get_flag("editor");
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        info!("Editor mode flag: {}", editor_mode);
    }
    if editor_mode {
        return launch_editor_mode(
            &repo_dir,
            token_target,
            max_bytes,
            matches.get_flag("no_exclude_tests"),
        )
        .await;
    }

    // Covering set for git diff
    if matches.get_flag("covering_set_diff") {
        return run_covering_set_diff_mode(
            &repo_dir,
            matches.get_one::<String>("diff_against").map(|s| s.as_str()),
            matches.get_flag("include_dependents"),
            matches.get_one::<usize>("max_depth").copied(),
            matches.get_one::<usize>("max_files_covering").copied(),
            verbose_level,
        )
        .await;
    }

    // Check for covering set mode
    if let Some(entity_name) = matches.get_one::<String>("covering_set") {
        let granularity = matches.get_one::<String>("granularity").map(|s| s.as_str()).unwrap_or("file");
        let stdout_mode = matches.get_flag("stdout");
        return run_covering_set_mode(
            &repo_dir,
            entity_name,
            matches.get_one::<String>("entity_type").map(|s| s.as_str()),
            matches.get_flag("exact_match"),
            matches.get_flag("include_dependents"),
            matches.get_one::<usize>("max_depth").copied(),
            matches.get_one::<usize>("max_files_covering").copied(),
            granularity,
            stdout_mode,
            verbose_level,
        )
        .await;
    }

    // New arguments
    let force_traditional = matches.get_flag("force_traditional");
    let algorithm = matches.get_one::<Algorithm>("algorithm").unwrap();
    let query_hint = matches.get_one::<String>("query_hint").cloned();
    let show_metrics = matches.get_flag("show_metrics");
    let entry_points: Vec<String> = matches
        .get_many::<String>("entry_points")
        .map(|vals| vals.cloned().collect())
        .unwrap_or_default();
    let entry_functions: Vec<String> = matches
        .get_many::<String>("entry_functions")
        .map(|vals| vals.cloned().collect())
        .unwrap_or_default();
    let personalization_alpha = *matches.get_one::<f64>("personalization_alpha").unwrap();
    let include_diffs = matches.get_flag("include_diffs");
    let diff_commits = *matches.get_one::<usize>("diff_commits").unwrap();
    let diff_branch = matches.get_one::<String>("diff_branch").cloned();
    let diff_relevance_threshold = *matches.get_one::<f64>("diff_relevance_threshold").unwrap();
    let use_scaling = matches.get_flag("scaling");
    let exclude_tests = matches.get_flag("exclude_tests");
    let include_tests_override = matches.get_flag("no_exclude_tests");
    let include_patterns_cli = matches
        .get_one::<String>("include")
        .map(|value| normalize_patterns(parse_pattern_list(value)));
    let exclude_patterns_cli = matches
        .get_one::<String>("exclude")
        .map(|value| normalize_patterns(parse_pattern_list(value)));
    let ignore_patterns_cli = matches
        .get_one::<String>("ignore")
        .map(|value| normalize_patterns(parse_pattern_list(value)));
    let disable_gitignore = matches.get_flag("no_gitignore");
    let disable_default_patterns = matches.get_flag("no_default_patterns");

    // Set up verbose logging and debug output
    if verbose_level > 0 {
        std::env::set_var("SCRIBE_DEBUG", "1");
        info!("Verbose mode enabled (level: {})", verbose_level);
    }

    // Load repository configuration (.scribe.json or scribe.config.json)
    let mut config = load_repository_config(&repo_dir);

    // Load .scribeignore patterns
    let repo_ignore_patterns = load_ignore_patterns(&repo_dir);

    if verbose_level > 0 {
        info!("Analyzing repository: {}", repo_dir.display());
    }

    // Determine output file path with config file support
    let output_path = determine_output_path(
        matches.get_one::<String>("output"),
        config.output.file_path.as_ref(),
        &repo_dir,
        report_format,
    );

    // Use the library function for proper intelligent analysis
    config.filtering.max_file_size = max_bytes as u64;
    config.analysis.token_budget = None;

    // Enable scaling optimizations if requested
    config.features.scaling_enabled = use_scaling;

    // Apply filtering configuration
    apply_filter_config(
        &mut config,
        exclude_patterns_cli,
        ignore_patterns_cli,
        include_patterns_cli,
        repo_ignore_patterns,
        disable_default_patterns,
        disable_gitignore,
        exclude_tests,
        include_tests_override,
    );

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("Include patterns: {:?}", config.filtering.include_patterns);
        eprintln!("Exclude patterns: {:?}", config.filtering.exclude_patterns);
    }

    if verbose_level > 0 {
        info!("🎯 Token budget configured: {} tokens", token_target);
        info!("📏 Max file size limit: {} bytes", max_bytes);
    }

    let algorithm_name = match algorithm {
        Algorithm::V1Baseline => "v1-baseline",
        Algorithm::V3Centrality => "v3-centrality",
        Algorithm::V4Demotion => "v4-demotion",
        Algorithm::V5Integrated => "v5-integrated",
    }
    .to_string();

    if verbose_level > 0 {
        info!("Algorithm: {}", algorithm_name);
        info!("Force traditional: {}", force_traditional);
        if let Some(hint) = &query_hint {
            info!("Query hint: {}", hint);
        }
        if !entry_points.is_empty() {
            info!("Entry points: {:?}", entry_points);
        }
        if !entry_functions.is_empty() {
            info!("Entry functions: {:?}", entry_functions);
        }
        if include_diffs {
            info!("Including diffs from {} commits", diff_commits);
            if let Some(branch) = &diff_branch {
                info!("Diff branch: {}", branch);
            }
        }
        if use_scaling {
            info!("Scaling optimizations: ENABLED");
        }
        if exclude_tests {
            info!("Auto-exclude tests: ENABLED");
        }
    }

    let selection_options = SelectionOptions {
        token_target,
        force_traditional,
        algorithm_name: Some(algorithm_name.clone()),
        include_directory_map: true,
    };

    let analysis_outcome = analyze_and_select(&repo_dir, &config, &selection_options).await?;
    let mut selected_files = analysis_outcome.selection.selected_files;
    let metrics = analysis_outcome.selection.metrics;
    let eligible_file_count = analysis_outcome.selection.eligible_file_count;
    let unlimited_budget = analysis_outcome.selection.unlimited_budget;
    let total_files_discovered = metrics.total_files_discovered;

    if verbose_level > 0 {
        info!(
            "Selected {} files ({} tokens)",
            metrics.files_selected, metrics.total_tokens_estimated
        );
    } else {
        print_selection_summary(&metrics, eligible_file_count, token_target, unlimited_budget);
    }

    if show_metrics {
        if verbose_level > 0 {
            info!("Enhanced Selection Metrics:");
        } else {
            println!(
                "
📈 Additional metrics"
            );
        }

        let repository_complexity_factor = if total_files_discovered > 0 {
            eligible_file_count as f64 / total_files_discovered as f64
        } else {
            0.0
        };

        if verbose_level > 0 {
            info!("  - Algorithm: {}", metrics.algorithm_used);
            info!(
                "  - Files: {} / {}",
                metrics.files_selected, metrics.total_files_discovered
            );
            info!("  - Tokens: {}", metrics.total_tokens_estimated);
            info!("  - Coverage: {:.1}%", metrics.coverage_score * 100.0);
            info!("  - Relevance: {:.2}", metrics.relevance_score);
            info!("  - Selection time: {}ms", metrics.selection_time_ms);
            info!(
                "  - Repository complexity: {:.2}",
                repository_complexity_factor
            );
        } else {
            println!("  • Algorithm        : {}", metrics.algorithm_used);
            println!(
                "  • Coverage         : {:.1}%",
                metrics.coverage_score * 100.0
            );
            println!("  • Relevance score  : {:.2}", metrics.relevance_score);
        }

        if !entry_points.is_empty() {
            let avg_entry_proximity = selected_files
                .iter()
                .map(|f| f.entry_point_proximity)
                .sum::<f64>()
                / selected_files.len().max(1) as f64;
            info!("  - Entry point influence: {:.2}", avg_entry_proximity);
        }

        if query_hint.is_some() {
            let avg_query_relevance = selected_files
                .iter()
                .map(|f| f.query_relevance_score)
                .sum::<f64>()
                / selected_files.len().max(1) as f64;
            info!("  - Query relevance: {:.2}", avg_query_relevance);
        }

        if include_diffs {
            let avg_recency = selected_files.iter().map(|f| f.recency_score).sum::<f64>()
                / selected_files.len().max(1) as f64;
            info!("  - Recency score: {:.2}", avg_recency);
        }

        let avg_content_quality = selected_files
            .iter()
            .map(|f| f.content_quality_score)
            .sum::<f64>()
            / selected_files.len().max(1) as f64;
        let avg_centrality = selected_files
            .iter()
            .map(|f| f.centrality_score)
            .sum::<f64>()
            / selected_files.len().max(1) as f64;
        info!("  - Content quality: {:.2}", avg_content_quality);
        info!("  - Centrality: {:.2}", avg_centrality);
    }

    // Generate output
    let format_label = report_format_label(report_format);

    if verbose_level == 0 {
        println!("📝 Generating {} output...", format_label);
    } else {
        info!("📝 Generating {} output", format_label);
    }

    let mut selected_files = selected_files;

    if include_line_numbers {
        apply_line_numbers_to_files(&mut selected_files);
    }

    let report_content = generate_report(report_format, &selected_files, &metrics)?;
    fs::write(&output_path, report_content)?;

    if verbose_level > 0 {
        info!(
            "🎉 Analysis complete! Output saved to: {}",
            output_path.display()
        );
    } else {
        println!("  • Output location : {}", output_path.display());
        println!(
            "
🎉 Analysis complete"
        );
    }

    // Show configuration source info
    if config.output.file_path.is_some() && matches.get_one::<String>("output").is_none() {
        info!("📋 Output path from configuration file");
    }

    Ok(())
}

/// Configuration file candidates to check in repository
const CONFIG_FILE_CANDIDATES: [&str; 2] = [".scribe.json", "scribe.config.json"];

fn load_repository_config(repo_dir: &Path) -> Config {
    for candidate in &CONFIG_FILE_CANDIDATES {
        let candidate_path = repo_dir.join(candidate);
        if let Some(config) = try_load_config_file(&candidate_path) {
            return config;
        }
    }
    Config::default()
}

/// Attempt to load config from a specific file path
fn try_load_config_file(path: &Path) -> Option<Config> {
    if !path.exists() {
        return None;
    }
    match Config::load_from_file(path) {
        Ok(config) => {
            info!("📋 Loaded repository configuration from: {}", path.display());
            Some(config)
        }
        Err(err) => {
            warn!("Failed to load configuration from {}: {}", path.display(), err);
            None
        }
    }
}

fn load_ignore_patterns(repo_dir: &Path) -> Vec<String> {
    let mut patterns = Vec::new();
    let ignore_file = repo_dir.join(".scribeignore");
    if ignore_file.exists() {
        match fs::read_to_string(&ignore_file) {
            Ok(content) => {
                info!("📋 Loaded ignore patterns from: {}", ignore_file.display());
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }
                    if !trimmed.starts_with('!') {
                        patterns.push(trimmed.to_string());
                    }
                }
            }
            Err(err) => {
                warn!("Failed to read {}: {}", ignore_file.display(), err);
            }
        }
    }

    patterns
}

fn parse_pattern_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .flat_map(|segment| segment.split_whitespace())
        .map(str::trim)
        .filter(|pattern| !pattern.is_empty())
        .map(|pattern| pattern.to_string())
        .collect()
}

fn normalize_patterns(patterns: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for pattern in patterns {
        let trimmed = pattern.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut normalized = trimmed.to_string();
        if trimmed.ends_with('/') {
            normalized.push_str("**");
        } else if !trimmed.contains('/') && !trimmed.contains('\\') && !trimmed.contains("**") {
            normalized = format!("**/{}", trimmed);
        }

        if seen.insert(normalized.clone()) {
            result.push(normalized);
        }
    }

    result
}

fn apply_line_numbers_to_files(files: &mut [ReportFile]) {
    for file in files {
        file.content = add_line_numbers(&file.content);
    }
}

fn add_line_numbers(content: &str) -> String {
    let lines: Vec<&str> = content.split('\n').collect();
    let width = lines.len().max(1).to_string().len().max(3);

    let mut numbered = String::with_capacity(content.len() + lines.len() * (width + 3));
    for (idx, line) in lines.iter().enumerate() {
        let line_no = idx + 1;
        numbered.push_str(&format!("{:width$} | {}", line_no, line, width = width));
        if idx + 1 < lines.len() {
            numbered.push('\n');
        }
    }

    numbered
}

/// Determine output file path from CLI args, config, or auto-generate
fn determine_output_path(
    cli_output: Option<&String>,
    config_path: Option<&String>,
    repo_dir: &Path,
    report_format: ReportFormat,
) -> PathBuf {
    if let Some(output) = cli_output {
        return PathBuf::from(output);
    }

    if let Some(config_path) = config_path {
        let path = PathBuf::from(config_path);
        return if path.is_absolute() { path } else { repo_dir.join(path) };
    }

    auto_generate_output_path(repo_dir, report_format)
}

/// Auto-generate output filename based on repository name and format
fn auto_generate_output_path(repo_dir: &Path, format: ReportFormat) -> PathBuf {
    let base_name = repo_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("repository");

    let extension = report_format_extension(format);
    PathBuf::from(format!("{}.{}", base_name, extension))
}

/// Get file extension for a report format
fn report_format_extension(format: ReportFormat) -> &'static str {
    match format {
        ReportFormat::Html => "html",
        ReportFormat::Repomix => "repomix",
        ReportFormat::Xml => "xml",
        ReportFormat::Json => "json",
        ReportFormat::Text => "txt",
        ReportFormat::Markdown => "md",
    }
}

/// Get human-readable label for a report format
fn report_format_label(format: ReportFormat) -> &'static str {
    match format {
        ReportFormat::Html => "HTML",
        ReportFormat::Repomix => "Repomix",
        ReportFormat::Xml => "XML",
        ReportFormat::Json => "JSON",
        ReportFormat::Text => "Text",
        ReportFormat::Markdown => "Markdown",
    }
}

/// Apply filtering configuration from CLI arguments
fn apply_filter_config(
    config: &mut Config,
    exclude_patterns_cli: Option<Vec<String>>,
    ignore_patterns_cli: Option<Vec<String>>,
    include_patterns_cli: Option<Vec<String>>,
    repo_ignore_patterns: Vec<String>,
    disable_default_patterns: bool,
    disable_gitignore: bool,
    exclude_tests: bool,
    include_tests_override: bool,
) {
    config.filtering.include_patterns =
        normalize_patterns(std::mem::take(&mut config.filtering.include_patterns));
    let mut exclude_patterns =
        normalize_patterns(std::mem::take(&mut config.filtering.exclude_patterns));

    if disable_default_patterns {
        exclude_patterns.clear();
    }

    if !repo_ignore_patterns.is_empty() {
        exclude_patterns.extend(normalize_patterns(repo_ignore_patterns));
    }

    if let Some(patterns) = exclude_patterns_cli {
        exclude_patterns.extend(patterns);
    }

    if let Some(patterns) = ignore_patterns_cli {
        exclude_patterns.extend(patterns);
    }

    config.filtering.exclude_patterns = normalize_patterns(exclude_patterns);

    if disable_gitignore {
        config.filtering.respect_gitignore = false;
    }

    if let Some(patterns) = include_patterns_cli {
        if !patterns.is_empty() {
            config.filtering.include_patterns = patterns;
        }
    }

    config.features.auto_exclude_tests = if include_tests_override {
        false
    } else if exclude_tests {
        true
    } else {
        config.features.auto_exclude_tests
    };
}

/// Print selection summary to stdout
fn print_selection_summary(
    metrics: &SelectionMetrics,
    eligible_file_count: usize,
    token_target: usize,
    unlimited_budget: bool,
) {
    println!("📊 Selection summary");
    println!("  • Files scanned   : {}", metrics.total_files_discovered);
    println!("  • Eligible files  : {}", eligible_file_count);
    println!(
        "  • Files selected  : {} ({} tokens)",
        metrics.files_selected, metrics.total_tokens_estimated
    );
    println!(
        "  • Files excluded  : {}",
        eligible_file_count.saturating_sub(metrics.files_selected)
    );
    println!("  • Coverage        : {:.1}%", metrics.coverage_score * 100.0);
    if unlimited_budget || token_target == 0 {
        println!("  • Token usage     : unlimited");
    } else {
        println!(
            "  • Token usage     : {} / {}",
            metrics.total_tokens_estimated, token_target
        );
    }
}

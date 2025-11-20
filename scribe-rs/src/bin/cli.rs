use clap::{Arg, ArgAction, Command, ValueEnum};
use git2::Repository;
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

async fn run_covering_set_mode(
    repo_dir: &Path,
    entity_name: &str,
    entity_type: Option<&str>,
    exact_match: bool,
    include_dependents: bool,
    max_depth: Option<usize>,
    max_files: Option<usize>,
    verbose_level: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    use scribe_selection::{CoveringSetComputer, CoveringSetOptions, EntityQuery, EntityType};
    use scribe::analyze_and_select;
    use std::collections::HashMap;

    if verbose_level > 0 {
        info!("🎯 Covering set mode: finding '{}'", entity_name);
    } else {
        println!("🎯 Finding covering set for: {}", entity_name);
    }

    // First, scan and analyze the repository to get file info and build dependency graph
    let mut config = Config::default();
    config.general.working_dir = Some(repo_dir.to_path_buf());
    config.analysis.token_budget = None; // Analyze all files

    let selection_options = SelectionOptions {
        token_target: 0, // No budget limit for initial analysis
        force_traditional: false,
        algorithm_name: Some("covering-set".to_string()),
        include_directory_map: false,
    };

    if verbose_level > 0 {
        info!("📊 Scanning repository...");
    } else {
        println!("📊 Scanning repository...");
    }

    let analysis_outcome = analyze_and_select(repo_dir, &config, &selection_options).await?;

    // Collect file contents with String keys (required by covering set computer)
    let mut file_contents = HashMap::new();
    for file_info in &analysis_outcome.analysis.files {
        if let Ok(content) = std::fs::read_to_string(&file_info.path) {
            // Convert PathBuf to String for the covering set API
            file_contents.insert(file_info.path.display().to_string(), content);
        }
    }

    if verbose_level > 0 {
        info!("📁 Loaded {} files", file_contents.len());
    }

    // Build dependency graph for covering set computation
    // For now, create an empty graph - the covering set will primarily rely on
    // AST-based entity search rather than dependency traversal
    if verbose_level > 0 {
        info!("🔗 Preparing dependency graph...");
    }

    use scribe_graph::DependencyGraph;
    let graph = DependencyGraph::new();

    // TODO: Implement full dependency graph construction
    // This would require import extraction from file contents

    // Build entity query
    let parsed_entity_type = entity_type.and_then(|t| match t.to_lowercase().as_str() {
        "function" => Some(EntityType::Function),
        "class" => Some(EntityType::Class),
        "module" => Some(EntityType::Module),
        "interface" => Some(EntityType::Interface),
        "constant" => Some(EntityType::Constant),
        _ => None,
    });

    let query = EntityQuery {
        entity_type: parsed_entity_type,
        name_pattern: Some(entity_name.to_string()),
        exact_match,
        public_only: None,
    };

    // Build covering set options
    let options = CoveringSetOptions {
        include_dependencies: true,
        include_dependents,
        max_depth,
        max_files,
        min_importance: None,
    };

    if verbose_level > 0 {
        info!("🔍 Computing covering set...");
    } else {
        println!("🔍 Computing covering set...");
    }

    // Compute covering set
    let mut computer = CoveringSetComputer::new()?;
    let result = computer.compute_covering_set(
        &query,
        &file_contents,
        &graph,
        &options,
    )?;

    // Display results
    if let Some(target) = &result.target_entity {
        println!("\n✅ Found target entity:");
        println!("  • File     : {}", target.file_path);
        println!("  • Type     : {}", target.entity_type);
        println!("  • Name     : {}", target.entity_name);
        println!("  • Lines    : {}-{}", target.start_line, target.end_line);
        println!("  • Public   : {}", if target.is_public { "yes" } else { "no" });
    } else {
        println!("\n❌ Entity '{}' not found", entity_name);
        println!("   Try:");
        println!("   - Using a different name pattern");
        println!("   - Removing --exact-match flag for fuzzy search");
        println!("   - Specifying --entity-type (function, class, module, etc.)");
        return Ok(());
    }

    println!("\n📦 Covering set ({} files):", result.files.len());
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
    println!("  • Max depth       : {}", result.statistics.max_depth_reached);
    println!("  • Limits reached  : {}", if result.statistics.limits_reached { "yes" } else { "no" });

    if verbose_level > 0 {
        info!("✨ Covering set computation complete");
    }

    Ok(())
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
    Cxml,
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
            OutputFormat::Cxml => ReportFormat::Cxml,
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
                .help("Output format: html for web page, cxml for LLM, repomix for repomix format, xml for standard XML (default: html)")
                .value_parser(clap::value_parser!(OutputFormat))
                .default_value("html"),
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
                .help("Find covering set for a specific entity (function, class, module)")
                .value_name("ENTITY_NAME"),
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
                .action(ArgAction::SetTrue)
                .requires("covering_set"),
        )
        .arg(
            Arg::new("max_depth")
                .long("max-depth")
                .help("Maximum dependency traversal depth")
                .value_name("DEPTH")
                .value_parser(clap::value_parser!(usize))
                .requires("covering_set"),
        )
        .arg(
            Arg::new("max_files_covering")
                .long("max-files")
                .help("Maximum number of files in covering set")
                .value_name("COUNT")
                .value_parser(clap::value_parser!(usize))
                .requires("covering_set"),
        );

    let matches = app.get_matches();

    // Parse arguments
    let repo_path_or_url = matches.get_one::<String>("repo_path").unwrap();
    let output_format = matches.get_one::<OutputFormat>("output_format").unwrap();
    let report_format: ReportFormat = (*output_format).into();
    let token_target = *matches.get_one::<usize>("token_target").unwrap();
    let max_bytes = *matches.get_one::<usize>("max_bytes").unwrap();
    let verbose_level = matches.get_count("verbose");

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

    // Check for covering set mode
    if let Some(entity_name) = matches.get_one::<String>("covering_set") {
        return run_covering_set_mode(
            &repo_dir,
            entity_name,
            matches.get_one::<String>("entity_type").map(|s| s.as_str()),
            matches.get_flag("exact_match"),
            matches.get_flag("include_dependents"),
            matches.get_one::<usize>("max_depth").copied(),
            matches.get_one::<usize>("max_files_covering").copied(),
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
    let output_path = if let Some(output) = matches.get_one::<String>("output") {
        // CLI argument takes priority
        PathBuf::from(output)
    } else if let Some(config_path) = &config.output.file_path {
        // Use path from config file
        let path = PathBuf::from(config_path);
        if path.is_absolute() {
            path
        } else {
            // Resolve relative paths against repository directory
            repo_dir.join(path)
        }
    } else {
        // Auto-generate output filename
        let base_name = repo_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("repository");

        let extension = match report_format {
            ReportFormat::Html => "html",
            ReportFormat::Cxml => "cxml",
            ReportFormat::Repomix => "repomix",
            ReportFormat::Xml => "xml",
            ReportFormat::Json => "json",
            ReportFormat::Text => "txt",
            ReportFormat::Markdown => "md",
        };

        PathBuf::from(format!("{}.{}", base_name, extension))
    };

    // Use the library function for proper intelligent analysis
    config.filtering.max_file_size = max_bytes as u64;
    config.analysis.token_budget = None;

    // Enable scaling optimizations if requested
    config.features.scaling_enabled = use_scaling;

    // Start from configuration-defined patterns
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

    // Apply CLI overrides for filtering behaviour
    if disable_gitignore {
        config.filtering.respect_gitignore = false;
    }

    if let Some(patterns) = include_patterns_cli {
        if !patterns.is_empty() {
            config.filtering.include_patterns = patterns;
        }
    }

    // Enable auto-exclude tests if requested
    config.features.auto_exclude_tests = if include_tests_override {
        false
    } else if exclude_tests {
        true
    } else {
        config.features.auto_exclude_tests
    };

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
        println!("📊 Selection summary");
        println!("  • Files scanned   : {}", total_files_discovered);
        println!("  • Eligible files  : {}", eligible_file_count);
        println!(
            "  • Files selected  : {} ({} tokens)",
            metrics.files_selected, metrics.total_tokens_estimated
        );
        println!(
            "  • Files excluded  : {}",
            eligible_file_count.saturating_sub(metrics.files_selected)
        );
        println!(
            "  • Coverage        : {:.1}%",
            metrics.coverage_score * 100.0
        );
        if unlimited_budget || token_target == 0 {
            println!("  • Token usage     : unlimited");
        } else {
            println!(
                "  • Token usage     : {} / {}",
                metrics.total_tokens_estimated, token_target
            );
        }
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
    let format_label = match report_format {
        ReportFormat::Html => "HTML",
        ReportFormat::Cxml => "CXML",
        ReportFormat::Repomix => "Repomix",
        ReportFormat::Xml => "XML",
        ReportFormat::Json => "JSON",
        ReportFormat::Text => "Text",
        ReportFormat::Markdown => "Markdown",
    };

    if verbose_level == 0 {
        println!("📝 Generating {} output...", format_label);
    } else {
        info!("📝 Generating {} output", format_label);
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

fn load_repository_config(repo_dir: &Path) -> Config {
    let candidates = [".scribe.json", "scribe.config.json"];

    for candidate in &candidates {
        let candidate_path = repo_dir.join(candidate);
        if candidate_path.exists() {
            match Config::load_from_file(&candidate_path) {
                Ok(config) => {
                    info!(
                        "📋 Loaded repository configuration from: {}",
                        candidate_path.display()
                    );
                    return config;
                }
                Err(err) => {
                    warn!(
                        "Failed to load configuration from {}: {}",
                        candidate_path.display(),
                        err
                    );
                }
            }
        }
    }

    Config::default()
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

use clap::{Arg, ArgAction, Command, ValueEnum};
use git2::Repository;
use handlebars::Handlebars;
use serde_json::{self, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use tempfile::TempDir;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};
use url::Url;

// Import the main library functions
use scribe_analyzer::{
    analyze_and_select, default_include_patterns, format_bytes, format_timestamp, generate_report,
    get_file_icon, load_ignore_patterns, load_scribe_config, normalize_patterns,
    parse_pattern_list, Config, ReportFile, ReportFormat, SelectionMetrics, SelectionOptions,
};

// HTML Editor mode generation
#[allow(dead_code)]
fn generate_interactive_editor(
    files: &[ReportFile],
    metrics: &SelectionMetrics,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "🚀 Starting interactive editor generation with {} files",
        files.len()
    );
    let mut handlebars = Handlebars::new();

    // Use the bundled template with React tree and checkboxes
    let template_path = Path::new("templates/report_bundled.html");
    let template_content = if template_path.exists() {
        info!(
            "📄 Loading bundled template from: {}",
            template_path.display()
        );
        fs::read_to_string(template_path)?
    } else {
        warn!(
            "⚠️  Template file not found at: {}, using embedded template",
            template_path.display()
        );
        // Fallback to embedded template content if file doesn't exist
        include_str!("../../templates/report_bundled.html").to_string()
    };

    info!(
        "📝 Template content length: {} characters",
        template_content.len()
    );
    handlebars.register_template_string("editor", &template_content)?;

    // Generate current timestamp
    let generated_time = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    let template_data = serde_json::json!({
        "repository_name": "Scribe Analysis",
        "algorithm": metrics.algorithm_used,
        "generated_time": generated_time,
        "selection_time_ms": 0, // We don't track this in editor mode
        "total_files": files.len(),
        "total_tokens": metrics.total_tokens_estimated,
        "total_size": format_bytes(files.iter().map(|f| f.size).sum::<u64>()),
        "coverage_percentage": (metrics.coverage_score * 100.0) as u32,
        "files": files.iter().map(|f| serde_json::json!({
            "relative_path": f.relative_path,
            "content": f.content,
            "size": format_bytes(f.size),
            "estimated_tokens": f.estimated_tokens,
            "importance_score": format!("{:.2}", f.importance_score),
            "icon": get_file_icon(&f.relative_path)
        })).collect::<Vec<_>>()
    });

    let rendered = handlebars.render("editor", &template_data)?;
    fs::write(output_path, rendered)?;

    // Copy the JavaScript bundle to the output directory
    if let Some(output_dir) = output_path.parent() {
        let assets_dir = output_dir.join("assets");
        fs::create_dir_all(&assets_dir)?;

        let bundle_source = Path::new("templates/assets/scribe-tree-bundle.js");
        let bundle_dest = assets_dir.join("scribe-tree-bundle.js");

        if bundle_source.exists() {
            fs::copy(bundle_source, &bundle_dest)?;
            info!("📦 Copied bundle to: {}", bundle_dest.display());
        } else {
            warn!("⚠️  Bundle not found at: {}", bundle_source.display());
        }
    }

    info!("📝 Interactive editor generated: {}", output_path.display());
    println!("📝 Interactive editor saved to: {}", output_path.display());
    Ok(())
}
async fn clone_github_repo(
    url: &str,
) -> Result<(PathBuf, Option<TempDir>), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    Repository::clone(url, temp_dir.path())?;
    Ok((temp_dir.path().to_path_buf(), Some(temp_dir)))
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
        eprintln!("🚀 MAIN FUNCTION STARTED - DEBUG MODE");
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
        eprintln!("DEBUG: verbose_level = {}", verbose_level);
    }

    // Check for editor mode IMMEDIATELY - before any analysis
    let editor_mode = matches.get_flag("editor");
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("DEBUG: editor_mode = {}", editor_mode);
    }
    if editor_mode {
        eprintln!("🚀 EDITOR MODE DETECTED - Launching web service immediately...");

        // Find first available port starting at 5000
        let mut port = 5000u16;
        eprintln!("🔍 Looking for available port starting at 5000...");
        while port < 6000 {
            eprintln!("🔍 Testing port {}...", port);
            if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
                eprintln!("✅ Port {} is available!", port);
                break;
            }
            eprintln!("❌ Port {} is in use", port);
            port += 1;
        }

        if port >= 6000 {
            return Err("No available ports in range 5000-5999".into());
        }

        eprintln!("🎯 Selected port: {}", port);

        let mut web_service_cmd = std::process::Command::new("scribe-web");
        web_service_cmd
            .arg(&repo_path_or_url)
            .arg("--token-budget")
            .arg(&token_target.to_string())
            .arg("--port")
            .arg(&port.to_string());

        eprintln!(
            "🌐 Starting: scribe-web {} --token-budget {} --port {}",
            repo_path_or_url, token_target, port
        );

        let status = web_service_cmd.status()?;

        if !status.success() {
            return Err(format!("Web service failed with exit code: {:?}", status.code()).into());
        }

        return Ok(());
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

    // Handle GitHub URLs vs local paths
    let (repo_dir, _cleanup_temp) =
        if repo_path_or_url.starts_with("http://") || repo_path_or_url.starts_with("https://") {
            info!("🌐 Detected GitHub URL: {}", repo_path_or_url);
            clone_github_repo(repo_path_or_url).await?
        } else {
            // Local path handling
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

    // Load configuration from scribe.config.json if available
    let scribe_config = load_scribe_config(&repo_dir);

    // Load .scribeignore patterns
    let repo_ignore_patterns = load_ignore_patterns(&repo_dir);

    if verbose_level > 0 {
        info!("Analyzing repository: {}", repo_dir.display());
    }

    // Determine output file path with config file support
    let output_path = if let Some(output) = matches.get_one::<String>("output") {
        // CLI argument takes priority
        PathBuf::from(output)
    } else if let Some(config_path) = &scribe_config.output_file_path {
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
    let mut config = Config::default();
    config.filtering.max_file_size = max_bytes as u64;
    config.analysis.token_budget = None;

    // Enable scaling optimizations if requested
    config.features.scaling_enabled = use_scaling;

    // Respect configuration file include/exclude rules if present
    if scribe_config.include != default_include_patterns() && !scribe_config.include.is_empty() {
        config.filtering.include_patterns = normalize_patterns(scribe_config.include.clone());
    }

    if !scribe_config.ignore_use_gitignore {
        config.filtering.respect_gitignore = false;
    }

    let mut exclude_patterns = if !scribe_config.ignore_use_default_patterns {
        Vec::new()
    } else {
        config.filtering.exclude_patterns.clone()
    };

    if disable_default_patterns {
        exclude_patterns.clear();
    }

    if !scribe_config.ignore_custom_patterns.is_empty() {
        exclude_patterns.extend(normalize_patterns(
            scribe_config.ignore_custom_patterns.clone(),
        ));
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
    if scribe_config.output_file_path.is_some() && matches.get_one::<String>("output").is_none() {
        info!("📋 Output path from configuration file");
    }

    Ok(())
}

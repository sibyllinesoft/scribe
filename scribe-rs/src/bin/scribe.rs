use clap::{Arg, ArgAction, Command, ValueEnum};
use git2::Repository;
use std::fs;
use std::path::PathBuf;
use std::process;
use tempfile::TempDir;
use tracing::{error, info};
use tracing_subscriber::{fmt, EnvFilter};

mod cli;

use cli::config::{
    apply_filter_config, load_ignore_patterns, load_repository_config, normalize_patterns,
    parse_pattern_list,
};
use cli::covering_set::{run_covering_set_diff_mode, run_covering_set_mode};
use cli::output::{
    apply_line_numbers_to_files, determine_output_path, print_selection_summary,
    report_format_label,
};
use cli::web_service::launch_editor_mode;

use scribe::{analyze_and_select, generate_report, Config, ReportFormat, SelectionOptions};

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

fn build_cli() -> Command {
    Command::new("scribe")
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
                .help("Output format: text, html, xml, json, markdown, repomix (default: text)")
                .value_parser(clap::value_parser!(OutputFormat))
                .default_value("text"),
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
                .default_value("204800")
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
                .help("Exclude test files from selection")
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
                .help("Query hint to guide file selection")
                .value_name("HINT"),
        )
        .arg(
            Arg::new("show_metrics")
                .long("show-metrics")
                .help("Show detailed performance and quality metrics")
                .action(ArgAction::SetTrue),
        )
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
                .help("Focus on specific functions")
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
        .arg(
            Arg::new("scaling")
                .long("scaling")
                .help("Enable advanced scaling optimizations for large repositories")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("covering_set")
                .long("covering-set")
                .help("Find covering set for a file or entity")
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
                .help("Type of entity to find")
                .value_name("TYPE")
                .requires("covering_set"),
        )
        .arg(
            Arg::new("exact_match")
                .long("exact-match")
                .help("Match entity name exactly")
                .action(ArgAction::SetTrue)
                .requires("covering_set"),
        )
        .arg(
            Arg::new("include_dependents")
                .long("include-dependents")
                .help("Include files that depend on the target")
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
                .help("Covering set granularity: 'file' or 'entity'")
                .value_parser(["file", "entity"])
                .default_value("file"),
        )
        .arg(
            Arg::new("stdout")
                .long("stdout")
                .help("Output directly to stdout (for agent/pipeline use)")
                .action(ArgAction::SetTrue),
        )
        .after_help(include_str!("cli/help_text.txt"))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        info!("CLI main started in debug mode");
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let matches = build_cli().get_matches();

    // Parse arguments
    let repo_path_or_url = matches.get_one::<String>("repo_path").unwrap();
    let output_format = matches.get_one::<OutputFormat>("output_format").unwrap();
    let report_format: ReportFormat = (*output_format).into();
    let token_target = *matches.get_one::<usize>("token_target").unwrap();
    let max_bytes = *matches.get_one::<usize>("max_bytes").unwrap();
    let verbose_level = matches.get_count("verbose");
    let include_line_numbers = matches.get_flag("line_numbers");

    // Normalize repository location
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

    // Check for editor mode
    if matches.get_flag("editor") {
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

    // Standard analysis mode
    let stdout_mode = matches.get_flag("stdout");
    run_standard_analysis(
        &repo_dir,
        &matches,
        report_format,
        token_target,
        max_bytes,
        verbose_level,
        include_line_numbers,
        stdout_mode,
    )
    .await
}

async fn run_standard_analysis(
    repo_dir: &std::path::Path,
    matches: &clap::ArgMatches,
    report_format: ReportFormat,
    token_target: usize,
    max_bytes: usize,
    verbose_level: u8,
    include_line_numbers: bool,
    stdout_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let include_diffs = matches.get_flag("include_diffs");
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

    if verbose_level > 0 {
        std::env::set_var("SCRIBE_DEBUG", "1");
        info!("Verbose mode enabled (level: {})", verbose_level);
    }

    let mut config = load_repository_config(repo_dir);
    let repo_ignore_patterns = load_ignore_patterns(repo_dir);

    if verbose_level > 0 {
        info!("Analyzing repository: {}", repo_dir.display());
    }

    let output_path = determine_output_path(
        matches.get_one::<String>("output"),
        config.output.file_path.as_ref(),
        repo_dir,
        report_format,
    );

    config.filtering.max_file_size = max_bytes as u64;
    config.analysis.token_budget = None;
    config.features.scaling_enabled = use_scaling;

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
            info!("Including diffs");
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
        query_hint: query_hint.clone(),
    };

    let analysis_outcome = analyze_and_select(repo_dir, &config, &selection_options).await?;
    let mut selected_files = analysis_outcome.selection.selected_files;
    let metrics = analysis_outcome.selection.metrics;
    let eligible_file_count = analysis_outcome.selection.eligible_file_count;
    let unlimited_budget = analysis_outcome.selection.unlimited_budget;
    let total_files_discovered = metrics.total_files_discovered;

    if !stdout_mode {
        if verbose_level > 0 {
            info!(
                "Selected {} files ({} tokens)",
                metrics.files_selected, metrics.total_tokens_estimated
            );
        } else {
            print_selection_summary(&metrics, eligible_file_count, token_target, unlimited_budget);
        }
    }

    if show_metrics && !stdout_mode {
        print_detailed_metrics(
            &metrics,
            &selected_files,
            total_files_discovered,
            eligible_file_count,
            &entry_points,
            &query_hint,
            include_diffs,
            verbose_level,
        );
    }

    let format_label = report_format_label(report_format);

    if !stdout_mode {
        if verbose_level == 0 {
            println!("📝 Generating {} output...", format_label);
        } else {
            info!("📝 Generating {} output", format_label);
        }
    }

    if include_line_numbers {
        apply_line_numbers_to_files(&mut selected_files);
    }

    let report_content = generate_report(report_format, &selected_files, &metrics)?;

    if stdout_mode {
        // Write directly to stdout for agent/pipeline use
        print!("{}", report_content);
    } else {
        fs::write(&output_path, report_content)?;

        if verbose_level > 0 {
            info!(
                "🎉 Analysis complete! Output saved to: {}",
                output_path.display()
            );
        } else {
            println!("  • Output location : {}", output_path.display());
            println!("\n🎉 Analysis complete");
        }
    }

    if config.output.file_path.is_some() && matches.get_one::<String>("output").is_none() {
        info!("📋 Output path from configuration file");
    }

    Ok(())
}

fn print_detailed_metrics(
    metrics: &scribe::SelectionMetrics,
    selected_files: &[scribe::ReportFile],
    total_files_discovered: usize,
    eligible_file_count: usize,
    entry_points: &[String],
    query_hint: &Option<String>,
    include_diffs: bool,
    verbose_level: u8,
) {
    if verbose_level > 0 {
        info!("Enhanced Selection Metrics:");
    } else {
        println!("\n📈 Additional metrics");
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

//! CLI binary for launching the Scribe web service

use clap::{Arg, Command};
use scribe_webservice::{WebService, WebServiceConfig};
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let app = Command::new("scribe-web")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Nathan Rice <nathan@sibylline.dev>")
        .about("Scribe Web Service - Interactive repository analysis with automatic browser opening")
        .long_about("
Scribe Web Service provides a modern web interface for repository analysis and bundle generation.
Unlike the old static HTML approach, this creates a real HTTP server with:

✅ Automatic browser opening
✅ Real-time bundle generation and saving  
✅ Interactive file selection
✅ Direct download capabilities
✅ REST API endpoints

The web service automatically opens your browser and provides a much better user experience
than the previous static file approach.
        ")
        .arg(
            Arg::new("repo")
                .help("Repository path to analyze")
                .value_name("PATH")
                .default_value(".")
                .index(1),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .short('p')
                .help("Port to bind to")
                .value_name("PORT")
                .default_value("8080"),
        )
        .arg(
            Arg::new("host")
                .long("host")
                .help("Host to bind to")
                .value_name("HOST")  
                .default_value("127.0.0.1"),
        )
        .arg(
            Arg::new("token-budget")
                .long("token-budget")
                .short('t')
                .help("Token budget for file selection")
                .value_name("TOKENS")
                .default_value("50000"),
        )
        .arg(
            Arg::new("no-browser")
                .long("no-browser")
                .help("Don't automatically open browser")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("max-file-size")
                .long("max-file-size")
                .help("Maximum file size to consider (in bytes)")
                .value_name("BYTES")
                .default_value("1048576"), // 1MB
        )
        .arg(
            Arg::new("no-exclude-tests")
                .long("no-exclude-tests")
                .help("Don't automatically exclude test files")
                .action(clap::ArgAction::SetTrue),
        );

    let matches = app.get_matches();

    // Parse arguments
    let repo_path = PathBuf::from(matches.get_one::<String>("repo").unwrap());
    let port = matches
        .get_one::<String>("port")
        .unwrap()
        .parse::<u16>()
        .map_err(|_| "Invalid port number")?;
    let host = matches.get_one::<String>("host").unwrap().to_string();
    let token_budget = matches
        .get_one::<String>("token-budget")
        .unwrap()
        .parse::<usize>()
        .map_err(|_| "Invalid token budget")?;
    let auto_open_browser = !matches.get_flag("no-browser");
    let max_file_size = matches
        .get_one::<String>("max-file-size")
        .unwrap()
        .parse::<usize>()
        .map_err(|_| "Invalid max file size")?;
    let auto_exclude_tests = !matches.get_flag("no-exclude-tests");

    // Validate repository path
    if !repo_path.exists() {
        error!("Repository path does not exist: {}", repo_path.display());
        std::process::exit(1);
    }

    if !repo_path.is_dir() {
        error!("Repository path is not a directory: {}", repo_path.display());
        std::process::exit(1);
    }

    // Create configuration
    let config = WebServiceConfig {
        port,
        host,
        repo_path,
        token_budget,
        auto_open_browser,
        max_file_size,
        auto_exclude_tests,
    };

    info!("Starting Scribe web service...");
    info!("Repository: {}", config.repo_path.display());
    info!("Token budget: {}", config.token_budget);
    info!("Auto-exclude tests: {}", config.auto_exclude_tests);
    info!("Max file size: {} MB", config.max_file_size / 1024 / 1024);
    
    if config.auto_open_browser {
        info!("Browser will open automatically when ready");
    } else {
        info!("Browser auto-opening disabled");
        info!("Navigate to http://{}:{} when ready", config.host, config.port);
    }

    // Create and start web service
    match WebService::new(config) {
        Ok(service) => {
            if let Err(e) = service.start().await {
                error!("Web service failed: {}", e);
                std::process::exit(1);
            }
        }
        Err(e) => {
            error!("Failed to create web service: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
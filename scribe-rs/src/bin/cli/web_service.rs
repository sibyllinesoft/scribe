//! Web service integration for the CLI editor mode

use std::path::Path;
use std::sync::Arc;
use tracing::info;

#[cfg(feature = "web")]
use async_trait::async_trait;
#[cfg(feature = "web")]
use scribe_webservice::{
    AnalysisOutput, AnalysisProvider, WebReportFile, WebSelectionMetrics, WebService,
    WebServiceConfig, WebServiceError,
};

use scribe::{
    analyze_and_select, format_timestamp, Config, ReportFile, SelectionMetrics, SelectionOptions,
};

#[cfg(feature = "web")]
pub struct CliAnalysisProvider;

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
pub async fn launch_editor_mode(
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
pub async fn launch_editor_mode(
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

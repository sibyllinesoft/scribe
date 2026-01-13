//! Analysis caching infrastructure for handlers.

use crate::{
    handler_helpers::{
        file_entry_from_fileinfo, file_entry_from_report_file, prepare_template_data, FileEntry,
        TemplateData,
    },
    AnalysisOutput, AppState,
};
use handlebars::{Context as HbContext, Handlebars, Helper, HelperResult, Output, RenderContext};
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::types::ScanResult;

pub const ANALYSIS_CACHE_TTL: Duration = Duration::from_secs(5);

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct CacheKey {
    pub repo_path: String,
    pub token_budget: usize,
    pub max_file_size: usize,
    pub auto_exclude_tests: bool,
}

#[derive(Clone)]
pub struct CachedAnalysis {
    pub generated_at: Instant,
    pub analysis: AnalysisOutput,
    pub scan_result: ScanResult,
    pub template_data: TemplateData,
    pub rendered_html: Option<String>,
}

pub static ANALYSIS_CACHE: Lazy<Mutex<HashMap<CacheKey, CachedAnalysis>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn build_cache_key(config: &crate::WebServiceConfig) -> CacheKey {
    CacheKey {
        repo_path: config.repo_path.to_string_lossy().into_owned(),
        token_budget: config.token_budget,
        max_file_size: config.max_file_size,
        auto_exclude_tests: config.auto_exclude_tests,
    }
}

pub fn build_render_data(
    analysis: &AnalysisOutput,
    config: &crate::WebServiceConfig,
) -> (ScanResult, TemplateData, Option<String>) {
    let selected_paths: HashSet<&str> = analysis
        .selected_files
        .iter()
        .map(|file| file.relative_path.as_str())
        .collect();

    let included_entries: Vec<FileEntry> = analysis
        .selected_files
        .iter()
        .map(|file| {
            let matching_info = analysis
                .repository_files
                .iter()
                .find(|info| info.relative_path == file.relative_path);
            file_entry_from_report_file(file, matching_info)
        })
        .collect();

    let excluded_entries: Vec<FileEntry> = analysis
        .repository_files
        .iter()
        .filter(|file| !selected_paths.contains(file.relative_path.as_str()))
        .map(|file| file_entry_from_fileinfo(file, false))
        .collect();

    let mut categories = HashMap::new();
    categories.insert("included".to_string(), included_entries.clone());
    categories.insert("excluded".to_string(), excluded_entries.clone());

    let total_size: usize = included_entries.iter().map(|entry| entry.size).sum();

    let template_data = prepare_template_data(
        &config.repo_path,
        &analysis.selected_files,
        &analysis.metrics,
    );

    let rendered_html = render_template(&template_data).ok();

    (
        ScanResult {
            total_files: analysis.repository_files.len(),
            selected_files: included_entries.len(),
            excluded_files: excluded_entries.len(),
            token_estimate: analysis.metrics.total_tokens_estimated,
            total_size,
            categories,
            rendered_html: rendered_html.clone(),
        },
        template_data,
        rendered_html,
    )
}

pub async fn get_or_compute_analysis(state: &AppState) -> crate::Result<CachedAnalysis> {
    let config_snapshot = { state.config.read().await.clone() };
    let key = build_cache_key(&config_snapshot);

    if let Some(entry) = {
        let cache = ANALYSIS_CACHE.lock().unwrap();
        cache.get(&key).cloned()
    } {
        if entry.generated_at.elapsed() < ANALYSIS_CACHE_TTL {
            return Ok(entry);
        }
    }

    let analysis = state.analysis_provider.analyze(&config_snapshot).await?;
    let (scan_result, template_data, rendered_html) =
        build_render_data(&analysis, &config_snapshot);

    let cached = CachedAnalysis {
        generated_at: Instant::now(),
        analysis: analysis.clone(),
        scan_result: scan_result.clone(),
        template_data: template_data.clone(),
        rendered_html: rendered_html.clone(),
    };

    let mut cache = ANALYSIS_CACHE.lock().unwrap();
    cache.insert(key, cached.clone());

    Ok(cached)
}

pub fn render_template(
    template_data: &TemplateData,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let template_content = include_str!("../../../templates/report_bundled.html");

    let mut handlebars = Handlebars::new();
    handlebars.register_helper(
        "add",
        Box::new(
            |helper: &Helper<'_, '_>,
             _hb: &Handlebars<'_>,
             _ctx: &HbContext,
             _rc: &mut RenderContext<'_, '_>,
             out: &mut dyn Output|
             -> HelperResult {
                let lhs = helper
                    .param(0)
                    .and_then(|v| v.value().as_i64())
                    .unwrap_or(0);
                let rhs = helper
                    .param(1)
                    .and_then(|v| v.value().as_i64())
                    .unwrap_or(0);
                let sum = lhs + rhs;
                out.write(sum.to_string().as_ref())?;
                Ok(())
            },
        ),
    );
    handlebars.register_template_string("report", template_content)?;

    let mut rendered = handlebars.render("report", template_data)?;

    // Fix static asset path
    rendered = rendered.replace(
        "src=\"assets/scribe-tree-bundle.js\"",
        "src=\"/static/scribe-tree-bundle.js\"",
    );

    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WebReportFile, WebSelectionMetrics};
    use std::path::PathBuf;

    fn create_test_config() -> crate::WebServiceConfig {
        crate::WebServiceConfig {
            repo_path: PathBuf::from("/test/repo"),
            port: 8080,
            host: "127.0.0.1".to_string(),
            token_budget: 10000,
            auto_open_browser: false,
            max_file_size: 1024,
            auto_exclude_tests: true,
            auto_shutdown: false,
            auto_shutdown_timeout: 60,
        }
    }

    fn create_test_analysis() -> AnalysisOutput {
        AnalysisOutput {
            selected_files: vec![],
            selected_file_infos: vec![],
            metrics: WebSelectionMetrics {
                total_files_discovered: 10,
                files_selected: 5,
                total_tokens_estimated: 1000,
                selection_time_ms: 50,
                algorithm_used: "test".to_string(),
                coverage_score: 0.5,
                relevance_score: 0.8,
            },
            repository_files: vec![],
            token_budget: 10000,
        }
    }

    #[test]
    fn test_cache_key_creation() {
        let config = create_test_config();
        let key = build_cache_key(&config);

        assert_eq!(key.repo_path, "/test/repo");
        assert_eq!(key.token_budget, 10000);
        assert_eq!(key.max_file_size, 1024);
        assert!(key.auto_exclude_tests);
    }

    #[test]
    fn test_cache_key_equality() {
        let config1 = create_test_config();
        let config2 = create_test_config();

        let key1 = build_cache_key(&config1);
        let key2 = build_cache_key(&config2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality() {
        let mut config1 = create_test_config();
        let mut config2 = create_test_config();
        config2.token_budget = 5000;

        let key1 = build_cache_key(&config1);
        let key2 = build_cache_key(&config2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_clone() {
        let config = create_test_config();
        let key = build_cache_key(&config);
        let cloned = key.clone();

        assert_eq!(key.repo_path, cloned.repo_path);
        assert_eq!(key.token_budget, cloned.token_budget);
    }

    #[test]
    fn test_build_render_data_empty() {
        let analysis = create_test_analysis();
        let config = create_test_config();

        let (scan_result, template_data, _rendered) = build_render_data(&analysis, &config);

        assert_eq!(scan_result.total_files, 0);
        assert_eq!(scan_result.selected_files, 0);
        assert_eq!(scan_result.excluded_files, 0);
        assert_eq!(template_data.repository_name, "repo");
    }

    #[test]
    fn test_build_render_data_with_files() {
        let mut analysis = create_test_analysis();
        analysis.selected_files = vec![WebReportFile {
            path: PathBuf::from("/test/repo/src/main.rs"),
            relative_path: "src/main.rs".to_string(),
            content: "fn main() {}".to_string(),
            size: 1024,
            estimated_tokens: 256,
            importance_score: 0.9,
            centrality_score: 0.8,
            query_relevance_score: 0.0,
            entry_point_proximity: 1.0,
            content_quality_score: 0.7,
            repository_role_score: 0.6,
            recency_score: 0.5,
            modified: "2024-01-01".to_string(),
        }];

        let config = create_test_config();
        let (scan_result, template_data, _rendered) = build_render_data(&analysis, &config);

        assert_eq!(scan_result.selected_files, 1);
        assert_eq!(template_data.total_files, 1);
    }

    #[test]
    fn test_analysis_cache_ttl_constant() {
        assert_eq!(ANALYSIS_CACHE_TTL, Duration::from_secs(5));
    }

    #[test]
    fn test_cached_analysis_clone() {
        let analysis = create_test_analysis();
        let config = create_test_config();
        let (scan_result, template_data, rendered_html) = build_render_data(&analysis, &config);

        let cached = CachedAnalysis {
            generated_at: Instant::now(),
            analysis: analysis.clone(),
            scan_result,
            template_data,
            rendered_html,
        };

        let cloned = cached.clone();
        assert_eq!(
            cached.analysis.metrics.total_files_discovered,
            cloned.analysis.metrics.total_files_discovered
        );
    }

    #[test]
    fn test_render_template_success() {
        let template_data = TemplateData {
            repository_name: "test-repo".to_string(),
            algorithm: "two-pass".to_string(),
            generated_time: "2024-01-01".to_string(),
            selection_time_ms: 100,
            total_files: 5,
            total_tokens: "1000".to_string(),
            total_size: "10 KB".to_string(),
            coverage_percentage: 50,
            files: vec![],
        };

        let result = render_template(&template_data);
        assert!(result.is_ok());

        let html = result.unwrap();
        assert!(html.contains("test-repo"));
        // Check that static asset path was replaced
        assert!(html.contains("/static/scribe-tree-bundle.js") || !html.contains("assets/"));
    }

    #[test]
    fn test_render_template_with_files() {
        let template_data = TemplateData {
            repository_name: "my-project".to_string(),
            algorithm: "quota".to_string(),
            generated_time: "2024-01-15".to_string(),
            selection_time_ms: 250,
            total_files: 3,
            total_tokens: "500".to_string(),
            total_size: "5 KB".to_string(),
            coverage_percentage: 30,
            files: vec![
                crate::handler_helpers::TemplateFile {
                    relative_path: "src/main.rs".to_string(),
                    icon: "🦀".to_string(),
                    size: "1 KB".to_string(),
                    estimated_tokens: "200".to_string(),
                    importance_score: "0.90".to_string(),
                    content: "fn main() {}".to_string(),
                },
            ],
        };

        let result = render_template(&template_data);
        assert!(result.is_ok());

        let html = result.unwrap();
        assert!(html.contains("my-project"));
    }

    #[test]
    fn test_scan_result_categories() {
        let analysis = create_test_analysis();
        let config = create_test_config();

        let (scan_result, _, _) = build_render_data(&analysis, &config);

        assert!(scan_result.categories.contains_key("included"));
        assert!(scan_result.categories.contains_key("excluded"));
    }
}

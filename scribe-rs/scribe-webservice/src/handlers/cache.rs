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

#[derive(Hash, Eq, PartialEq, Clone)]
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

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

/// User-configurable settings loaded from `scribe.config.json` or compatible
/// configuration files. This mirrors the legacy CLI structure so both the CLI
/// and web service can rely on the same parsing logic.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScribeConfig {
    // Core settings
    #[serde(default = "default_max_file_size")]
    pub input_max_file_size: u64,

    // Output settings
    #[serde(default = "default_output_style")]
    pub output_style: String,
    pub output_file_path: Option<String>,
    #[serde(default)]
    pub output_parsable_style: bool,
    pub output_header_text: Option<String>,
    #[serde(default)]
    pub output_show_line_numbers: bool,
    #[serde(default = "default_true")]
    pub output_file_summary: bool,
    #[serde(default = "default_true")]
    pub output_directory_structure: bool,
    #[serde(default = "default_true")]
    pub output_files: bool,
    #[serde(default)]
    pub output_copy_to_clipboard: bool,

    // Pattern settings
    #[serde(default = "default_include_patterns")]
    pub include: Vec<String>,
    #[serde(default = "default_true")]
    pub ignore_use_gitignore: bool,
    #[serde(default = "default_true")]
    pub ignore_use_default_patterns: bool,
    #[serde(default)]
    pub ignore_custom_patterns: Vec<String>,

    // Git integration
    #[serde(default)]
    pub git_sort_by_changes: bool,
    #[serde(default = "default_git_max_commits")]
    pub git_sort_by_changes_max_commits: u32,
    #[serde(default)]
    pub git_include_diffs: bool,
    #[serde(default)]
    pub git_include_logs: bool,
    #[serde(default = "default_git_logs_count")]
    pub git_include_logs_count: u32,

    // Remote repository
    pub remote_url: Option<String>,
    pub remote_branch: Option<String>,

    // Token settings
    #[serde(default = "default_token_encoding")]
    pub token_count_encoding: String,

    // Security
    #[serde(default = "default_true")]
    pub security_enable_security_check: bool,
}

impl Default for ScribeConfig {
    fn default() -> Self {
        Self {
            input_max_file_size: default_max_file_size(),
            output_style: default_output_style(),
            output_file_path: None,
            output_parsable_style: false,
            output_header_text: None,
            output_show_line_numbers: false,
            output_file_summary: true,
            output_directory_structure: true,
            output_files: true,
            output_copy_to_clipboard: false,
            include: default_include_patterns(),
            ignore_use_gitignore: true,
            ignore_use_default_patterns: true,
            ignore_custom_patterns: Vec::new(),
            git_sort_by_changes: false,
            git_sort_by_changes_max_commits: default_git_max_commits(),
            git_include_diffs: false,
            git_include_logs: false,
            git_include_logs_count: default_git_logs_count(),
            remote_url: None,
            remote_branch: None,
            token_count_encoding: default_token_encoding(),
            security_enable_security_check: true,
        }
    }
}

/// Load configuration by walking up the repository tree and looking for
/// `scribe.config.json` (preferred) or `repomix.config.json` (converted).
pub fn load_scribe_config(repo_dir: &Path) -> ScribeConfig {
    let mut current_dir = repo_dir;

    loop {
        let scribe_config_path = current_dir.join("scribe.config.json");
        if scribe_config_path.exists() {
            if let Ok(config) = load_scribe_config_file(&scribe_config_path) {
                info!("📋 Loaded config from: {}", scribe_config_path.display());
                return config;
            }
        }

        let repomix_config_path = current_dir.join("repomix.config.json");
        if repomix_config_path.exists() {
            if let Ok(config) = load_repomix_config(&repomix_config_path) {
                info!(
                    "📋 Loaded repomix config from: {}",
                    repomix_config_path.display()
                );
                return config;
            }
        }

        if let Some(parent) = current_dir.parent() {
            current_dir = parent;
        } else {
            break;
        }
    }

    ScribeConfig::default()
}

/// Parse a configuration file and convert it into a `ScribeConfig`.
pub fn load_scribe_config_file(
    config_path: &Path,
) -> Result<ScribeConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(config_path)?;
    let json_value: Value = serde_json::from_str(&content)?;

    if is_repomix_style_config(&json_value) {
        convert_repomix_config(&json_value)
    } else {
        let config: ScribeConfig = serde_json::from_value(json_value)?;
        Ok(config)
    }
}

fn load_repomix_config(config_path: &Path) -> Result<ScribeConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(config_path)?;
    let json_value: Value = serde_json::from_str(&content)?;
    convert_repomix_config(&json_value)
}

fn is_repomix_style_config(config: &Value) -> bool {
    let repomix_indicators = [
        "output.filePath",
        "output.style",
        "ignore.customPatterns",
        "ignore.useGitignore",
        "tokenCount.encoding",
    ];

    repomix_indicators
        .iter()
        .any(|indicator| has_nested_key(config, indicator))
}

fn has_nested_key(data: &Value, key: &str) -> bool {
    let keys: Vec<&str> = key.split('.').collect();
    let mut current = data;

    for k in keys {
        if let Some(obj) = current.as_object() {
            if let Some(value) = obj.get(k) {
                current = value;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

fn convert_repomix_config(config_data: &Value) -> Result<ScribeConfig, Box<dyn std::error::Error>> {
    let mut config = ScribeConfig::default();

    if let Some(input) = config_data.get("input") {
        if let Some(max_file_size) = input.get("maxFileSize").and_then(|v| v.as_u64()) {
            config.input_max_file_size = max_file_size;
        }
    }

    if let Some(output) = config_data.get("output") {
        if let Some(style) = output.get("style").and_then(|v| v.as_str()) {
            config.output_style = style.to_string();
        }
        if let Some(file_path) = output.get("filePath").and_then(|v| v.as_str()) {
            config.output_file_path = Some(file_path.to_string());
        }
        if let Some(parsable) = output.get("parsableStyle").and_then(|v| v.as_bool()) {
            config.output_parsable_style = parsable;
        }
        if let Some(header) = output.get("headerText").and_then(|v| v.as_str()) {
            config.output_header_text = Some(header.to_string());
        }
        if let Some(line_numbers) = output.get("showLineNumbers").and_then(|v| v.as_bool()) {
            config.output_show_line_numbers = line_numbers;
        }
        if let Some(summary) = output.get("fileSummary").and_then(|v| v.as_bool()) {
            config.output_file_summary = summary;
        }
        if let Some(structure) = output.get("directoryStructure").and_then(|v| v.as_bool()) {
            config.output_directory_structure = structure;
        }
        if let Some(files) = output.get("files").and_then(|v| v.as_bool()) {
            config.output_files = files;
        }
        if let Some(clipboard) = output.get("copyToClipboard").and_then(|v| v.as_bool()) {
            config.output_copy_to_clipboard = clipboard;
        }

        if let Some(git) = output.get("git") {
            if let Some(sort_changes) = git.get("sortByChanges").and_then(|v| v.as_bool()) {
                config.git_sort_by_changes = sort_changes;
            }
            if let Some(include_diffs) = git.get("includeDiffs").and_then(|v| v.as_bool()) {
                config.git_include_diffs = include_diffs;
            }
        }
    }

    if let Some(ignore) = config_data.get("ignore") {
        if let Some(use_gitignore) = ignore.get("useGitignore").and_then(|v| v.as_bool()) {
            config.ignore_use_gitignore = use_gitignore;
        }
        if let Some(use_defaults) = ignore.get("useDefaultPatterns").and_then(|v| v.as_bool()) {
            config.ignore_use_default_patterns = use_defaults;
        }
        if let Some(custom_patterns) = ignore.get("customPatterns").and_then(|v| v.as_array()) {
            config.ignore_custom_patterns = custom_patterns
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();
        }
    }

    if let Some(git) = config_data.get("git") {
        if let Some(sort_changes) = git.get("sortByChanges").and_then(|v| v.as_bool()) {
            config.git_sort_by_changes = sort_changes;
        }
        if let Some(max_commits) = git.get("sortByChangesMaxCommits").and_then(|v| v.as_u64()) {
            config.git_sort_by_changes_max_commits = max_commits as u32;
        }
        if let Some(include_diffs) = git.get("includeDiffs").and_then(|v| v.as_bool()) {
            config.git_include_diffs = include_diffs;
        }
        if let Some(include_logs) = git.get("includeLogs").and_then(|v| v.as_bool()) {
            config.git_include_logs = include_logs;
        }
        if let Some(logs_count) = git.get("includeLogsCount").and_then(|v| v.as_u64()) {
            config.git_include_logs_count = logs_count as u32;
        }
    }

    if let Some(remote) = config_data.get("remote") {
        if let Some(url) = remote.get("url").and_then(|v| v.as_str()) {
            config.remote_url = Some(url.to_string());
        }
        if let Some(branch) = remote.get("branch").and_then(|v| v.as_str()) {
            config.remote_branch = Some(branch.to_string());
        }
    }

    if let Some(token_count) = config_data.get("tokenCount") {
        if let Some(encoding) = token_count.get("encoding").and_then(|v| v.as_str()) {
            config.token_count_encoding = encoding.to_string();
        }
    }

    if let Some(security) = config_data.get("security") {
        if let Some(enable_check) = security
            .get("enableSecurityCheck")
            .and_then(|v| v.as_bool())
        {
            config.security_enable_security_check = enable_check;
        }
    }

    Ok(config)
}

/// Load additional ignore patterns from `.scribeignore` or `.repomixignore`.
pub fn load_ignore_patterns(repo_dir: &Path) -> Vec<String> {
    let mut patterns = Vec::new();
    let ignore_files = [
        repo_dir.join(".scribeignore"),
        repo_dir.join(".repomixignore"),
    ];

    for ignore_file in &ignore_files {
        if ignore_file.exists() {
            if let Ok(content) = fs::read_to_string(ignore_file) {
                info!("📋 Loading ignore patterns from: {}", ignore_file.display());
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }
                    if !trimmed.starts_with('!') {
                        patterns.push(trimmed.to_string());
                    }
                }
                break;
            }
        }
    }

    patterns
}

/// Determine whether a relative path should be ignored given the configured
/// ignore globs.
pub fn should_ignore_file(relative_path: &str, ignore_patterns: &[String]) -> bool {
    ignore_patterns
        .iter()
        .any(|pattern| matches_glob_pattern(relative_path, pattern))
}

fn matches_glob_pattern(path: &str, pattern: &str) -> bool {
    let mut glob_pattern = pattern.to_string();

    if glob_pattern.ends_with('/') {
        glob_pattern.push_str("**");
    } else if !glob_pattern.contains('/')
        && !glob_pattern.contains('\\')
        && !glob_pattern.contains("**")
    {
        glob_pattern = format!("**/{}", glob_pattern);
    }

    if let Ok(glob) = globset::Glob::new(&glob_pattern) {
        glob.compile_matcher().is_match(path)
    } else {
        path.contains(pattern)
    }
}

/// Parse a comma or whitespace separated list of patterns from CLI/config.
pub fn parse_pattern_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .flat_map(|segment| segment.split_whitespace())
        .map(str::trim)
        .filter(|pattern| !pattern.is_empty())
        .map(|pattern| pattern.to_string())
        .collect()
}

/// Deduplicate and normalize patterns (adds recursive form for bare filenames).
pub fn normalize_patterns(patterns: Vec<String>) -> Vec<String> {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for pattern in patterns {
        let trimmed = pattern.trim();
        if trimmed.is_empty() {
            continue;
        }

        let normalized = trimmed.to_string();
        if seen.insert(normalized.clone()) {
            result.push(normalized.clone());
        }

        if !normalized.contains('/') && !normalized.contains('\\') && !normalized.contains("**") {
            let recursive = format!("**/{}", normalized);
            if seen.insert(recursive.clone()) {
                result.push(recursive);
            }
        }
    }

    result
}

fn default_max_file_size() -> u64 {
    204_800 // 200 KB
}

fn default_output_style() -> String {
    "html".to_string()
}

fn default_true() -> bool {
    true
}

pub fn default_include_patterns() -> Vec<String> {
    vec!["**/*".to_string()]
}

fn default_git_max_commits() -> u32 {
    100
}

fn default_git_logs_count() -> u32 {
    50
}

fn default_token_encoding() -> String {
    "o200k_base".to_string()
}

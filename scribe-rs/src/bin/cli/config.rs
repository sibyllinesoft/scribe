//! Configuration loading and management for the CLI

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use tracing::{info, warn};

use scribe::Config;

/// Configuration file candidates to check in repository
const CONFIG_FILE_CANDIDATES: [&str; 2] = [".scribe.json", "scribe.config.json"];

pub fn load_repository_config(repo_dir: &Path) -> Config {
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

pub fn load_ignore_patterns(repo_dir: &Path) -> Vec<String> {
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

pub fn parse_pattern_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .flat_map(|segment| segment.split_whitespace())
        .map(str::trim)
        .filter(|pattern| !pattern.is_empty())
        .map(|pattern| pattern.to_string())
        .collect()
}

pub fn normalize_patterns(patterns: Vec<String>) -> Vec<String> {
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

/// Apply filtering configuration from CLI arguments
pub fn apply_filter_config(
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

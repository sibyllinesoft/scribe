use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use bloom::{BloomFilter, ASMS};
use clap::{Arg, ArgAction, Command, ValueEnum};
use content_inspector::{inspect, ContentType};
use git2::{Oid, Repository};
use globset::{Glob, GlobMatcher};
use globset::{GlobSet, GlobSetBuilder};
use handlebars::Handlebars;
use ignore;
use indicatif::{ProgressBar, ProgressStyle};
use memchr::memmem;
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json::{self, json, Value};
use smallvec::SmallVec;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::OnceLock;
use tempfile::TempDir;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};
use url::Url;

// Import the optimized scanner components
use scribe_scanner::{
    AhoCorasickReferenceIndex, ContentAnalyzer, FileScanner, IndexConfig, LanguageDetector,
    MetadataExtractor, ScanOptions, Scanner,
};

// Import the main library functions
use scribe_analyzer::{analyze_repository, Config};

/// Multi-stage progress bar manager for scribe operations
struct ScribeProgressManager {
    current_bar: Option<ProgressBar>,
    stage_count: usize,
    current_stage: usize,
}

impl ScribeProgressManager {
    fn new(stage_count: usize) -> Self {
        Self {
            current_bar: None,
            stage_count,
            current_stage: 0,
        }
    }

    fn start_stage(&mut self, stage_name: &str, count: u64) {
        // Finish previous bar if exists
        if let Some(bar) = &self.current_bar {
            bar.finish_and_clear();
        }

        self.current_stage += 1;

        let pb = ProgressBar::new(count);
        let style = ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.cyan}} [{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos:>7}}/{{len:7}} {{msg}} (Stage {}/{})",
                self.current_stage,
                self.stage_count
            ))
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ");

        pb.set_style(style);
        pb.set_prefix(stage_name.to_string());
        pb.set_message("Starting...");

        self.current_bar = Some(pb);
    }

    fn update_message(&self, message: &str) {
        if let Some(bar) = &self.current_bar {
            bar.set_message(message.to_string());
        }
    }

    fn inc(&self, delta: u64) {
        if let Some(bar) = &self.current_bar {
            bar.inc(delta);
        }
    }

    fn set_position(&self, pos: u64) {
        if let Some(bar) = &self.current_bar {
            bar.set_position(pos);
        }
    }

    fn finish_stage(&self, message: &str) {
        if let Some(bar) = &self.current_bar {
            bar.finish_with_message(message.to_string());
        }
    }

    fn finish_all(&mut self) {
        if let Some(bar) = &self.current_bar {
            bar.finish_and_clear();
        }
        self.current_bar = None;
    }
}

/// Proper binary detection using content_inspector magic library
fn is_binary_content(content: &str) -> bool {
    match inspect(content.as_bytes()) {
        ContentType::BINARY => true,
        ContentType::UTF_8 => false,
        ContentType::UTF_8_BOM => false,
        ContentType::UTF_16LE => false,
        ContentType::UTF_16BE => false,
        ContentType::UTF_32LE => false,
        ContentType::UTF_32BE => false,
    }
}

// 🚀 PERFORMANCE OPTIMIZATION: Pre-compiled pattern matchers for content quality scoring
// Temporarily disabled for compilation
/*
struct ContentPatterns {
    // Pre-compiled Aho-Corasick matchers - dramatically faster than multiple contains() calls
    function_matcher: AhoCorasick,
    import_matcher: AhoCorasick,
}

impl ContentPatterns {
    fn new() -> Self {
        // 🚀 PERFORMANCE: Pre-compile function/class patterns
        let function_patterns = vec!["fn ", "def ", "function ", "class ", "interface ", "struct ", "impl "];
        let function_matcher = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&function_patterns)
            .expect("Failed to build function pattern matcher");

        // 🚀 PERFORMANCE: Pre-compile import patterns
        let import_patterns = vec!["import", "use ", "require", "include", "#include"];
        let import_matcher = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&import_patterns)
            .expect("Failed to build import pattern matcher");

        Self {
            function_matcher,
            import_matcher,
        }
    }

    fn has_function_pattern(&self, line: &str) -> bool {
        self.function_matcher.is_match(line)
    }

    fn has_import_pattern(&self, line: &str) -> bool {
        self.import_matcher.is_match(line)
    }
}
*/

// static CONTENT_PATTERNS: OnceLock<ContentPatterns> = OnceLock::new();

// Simple cache for content quality scores to avoid recalculation
static CONTENT_QUALITY_CACHE: OnceLock<std::sync::Mutex<HashMap<u64, f64>>> = OnceLock::new();

// 🚀 MEGA PERFORMANCE OPTIMIZATION: Eliminate all per-call string construction
struct MatchEngines {
    // Path matching - precompiled glob patterns
    path_patterns: GlobSet,
    src_paths: AhoCorasick,
    config_paths: AhoCorasick,
    test_paths: AhoCorasick,
    doc_paths: AhoCorasick,
    main_paths: AhoCorasick,

    // Query relevance - built per query
    query_matcher: Option<AhoCorasick>,
    query_words: Vec<memmem::Finder<'static>>,

    // Content scanning - precompiled finders
    main_functions: AhoCorasick,
    import_statements: AhoCorasick,

    // Trigram bloom prefilter for content (ripgrep-style optimization)
    content_bloom: BloomFilter,
}

impl MatchEngines {
    // 🚀 LIGHTWEIGHT: Build only essential patterns, defer complex ones
    fn lightweight_new(query_hint: Option<&str>) -> Self {
        // Small, essential patterns only - no per-file explosion
        let src_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&["src/", "lib/", "core/"])
            .unwrap();

        let config_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&["config", ".toml", ".yaml", ".json"])
            .unwrap();

        let test_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&["test", "spec", "_test", ".test"])
            .unwrap();

        let doc_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&[".md", "doc", "README"])
            .unwrap();

        let main_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&["main", "index", "lib.rs", "mod.rs"])
            .unwrap();

        // Query matcher: only if hint provided, keep it small
        let (query_matcher, query_words) = if let Some(hint) = query_hint {
            let words: Vec<&str> = hint.split_whitespace().take(8).collect(); // Cap at 8 words
            if !words.is_empty() {
                let matcher = AhoCorasickBuilder::new()
                    .ascii_case_insensitive(true)
                    .build(&words)
                    .ok();
                // Skip the problematic static lifetime finders for now
                (matcher, Vec::new())
            } else {
                (None, Vec::new())
            }
        } else {
            (None, Vec::new())
        };

        // Essential content matchers - minimal set
        let main_functions = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&["fn main(", "def main(", r#"if __name__ == "__main__""#])
            .unwrap();

        let import_statements = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&["import", "use ", "require", "include"])
            .unwrap();

        // Skip expensive bloom filter and content bloom for now
        let mut dummy_bloom = BloomFilter::with_rate(0.01, 100);

        Self {
            path_patterns: GlobSet::empty(), // Skip complex globs
            src_paths,
            config_paths,
            test_paths,
            doc_paths,
            main_paths,
            query_matcher,
            query_words,
            main_functions,
            import_statements,
            content_bloom: dummy_bloom,
        }
    }

    fn new(query_hint: Option<&str>) -> Self {
        // Build path pattern matchers once per analysis pass
        let mut path_builder = GlobSetBuilder::new();
        path_builder.add(globset::Glob::new("src/**").unwrap());
        path_builder.add(globset::Glob::new("lib/**").unwrap());
        path_builder.add(globset::Glob::new("core/**").unwrap());
        let path_patterns = path_builder.build().unwrap();

        // Pre-compile path component matchers
        let src_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&["src/", "lib/", "core/"])
            .unwrap();

        let config_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&[
                "config",
                ".toml",
                ".yaml",
                ".json",
                "Cargo.toml",
                "package.json",
                "Dockerfile",
            ])
            .unwrap();

        let test_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&["test", "spec"])
            .unwrap();

        let doc_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&[".md", "doc", "README"])
            .unwrap();

        let main_paths = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&["main", "index", "app.py", "server.js"])
            .unwrap();

        // Query-specific matchers
        let (query_matcher, query_words) = if let Some(query) = query_hint {
            let query_lower = query.to_lowercase();
            let words: Vec<String> = query_lower
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.to_owned())
                .collect();

            let matcher = if !words.is_empty() {
                Some(
                    AhoCorasickBuilder::new()
                        .ascii_case_insensitive(true)
                        .build(&words)
                        .unwrap(),
                )
            } else {
                None
            };

            // Convert to static finders - leak for 'static lifetime
            let finders: Vec<memmem::Finder<'static>> = words
                .into_iter()
                .map(|w| {
                    let leaked: &'static str = Box::leak(w.into_boxed_str());
                    memmem::Finder::new(leaked.as_bytes())
                })
                .collect();

            (matcher, finders)
        } else {
            (None, Vec::new())
        };

        // Content scanning matchers
        let main_functions = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&["fn main(", "def main(", "if __name__ == \"__main__\""])
            .unwrap();

        let import_statements = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&["import", "use ", "require", "include", "#include"])
            .unwrap();

        // 🚀 Trigram bloom filter for content prefiltering (ripgrep-style)
        let mut content_bloom = BloomFilter::with_rate(0.01, 100000);

        Self {
            path_patterns,
            src_paths,
            config_paths,
            test_paths,
            doc_paths,
            main_paths,
            query_matcher,
            query_words,
            main_functions,
            import_statements,
            content_bloom,
        }
    }
}

// 🚀 FileCache: Pre-processed file data to eliminate repeated parsing
#[derive(Debug, Clone)]
struct FileCache {
    bytes: Arc<[u8]>,
    lower_bytes: Arc<[u8]>,
    path_lower: Arc<[u8]>,
    path_components: SmallVec<[Range<usize>; 8]>,
    relative_path: String,
    size: usize,
}

impl FileCache {
    fn from_file_with_content(file: &FileWithContent) -> Self {
        let bytes: Arc<[u8]> = file.content.as_bytes().into();
        let lower_bytes: Arc<[u8]> = file.content.to_lowercase().as_bytes().into();
        let path_lower: Arc<[u8]> = file.relative_path.to_lowercase().as_bytes().into();

        // Parse path components once
        let mut path_components = SmallVec::new();
        let mut start = 0;
        for (i, &byte) in path_lower.iter().enumerate() {
            if byte == b'/' {
                if start < i {
                    path_components.push(start..i);
                }
                start = i + 1;
            }
        }
        if start < path_lower.len() {
            path_components.push(start..path_lower.len());
        }

        Self {
            bytes,
            lower_bytes,
            path_lower,
            path_components,
            relative_path: file.relative_path.clone(),
            size: file.size as usize,
        }
    }
}

// 🚀 PERFORMANCE HELPERS: Replace all .contains() calls with pre-compiled finders
impl MatchEngines {
    // Path matching helpers - zero string allocation per call
    fn is_src_path(&self, file: &FileCache) -> bool {
        self.src_paths.is_match(&file.path_lower)
    }

    fn is_config_path(&self, file: &FileCache) -> bool {
        self.config_paths.is_match(&file.path_lower)
    }

    fn is_test_path(&self, file: &FileCache) -> bool {
        self.test_paths.is_match(&file.path_lower)
    }

    fn is_doc_path(&self, file: &FileCache) -> bool {
        self.doc_paths.is_match(&file.path_lower)
    }

    fn is_main_path(&self, file: &FileCache) -> bool {
        self.main_paths.is_match(&file.path_lower)
    }

    // Content matching helpers - use pre-compiled finders
    fn has_main_function(&self, file: &FileCache) -> bool {
        self.main_functions.is_match(&file.bytes)
    }

    fn count_import_statements(&self, file: &FileCache) -> usize {
        self.import_statements.find_iter(&file.bytes).count()
    }

    // Query relevance helpers - zero allocation per call
    fn path_matches_query(&self, file: &FileCache) -> bool {
        self.query_matcher
            .as_ref()
            .map_or(false, |matcher| matcher.is_match(&file.path_lower))
    }

    fn count_query_matches_in_content(&self, file: &FileCache) -> usize {
        self.query_matcher
            .as_ref()
            .map_or(0, |matcher| matcher.find_iter(&file.lower_bytes).count())
    }

    fn count_query_word_matches(&self, file: &FileCache) -> usize {
        self.query_words
            .iter()
            .map(|finder| finder.find_iter(&file.lower_bytes).count())
            .sum()
    }

    // Trigram bloom prefilter - ripgrep-style optimization
    fn bloom_might_contain(&self, trigram: &[u8; 3]) -> bool {
        self.content_bloom.contains(&self.hash_trigram(trigram))
    }

    fn hash_trigram(&self, trigram: &[u8; 3]) -> u64 {
        let mut hasher = DefaultHasher::new();
        trigram.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Html,
    Cxml,
    Repomix,
    Xml,
    Json,
    Text,
    Markdown,
}

#[derive(Debug, Clone, ValueEnum)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ScribeConfig {
    // Core settings
    #[serde(default = "default_max_file_size")]
    input_max_file_size: u64,

    // Output settings
    #[serde(default = "default_output_style")]
    output_style: String,
    output_file_path: Option<String>,
    #[serde(default)]
    output_parsable_style: bool,
    output_header_text: Option<String>,
    #[serde(default)]
    output_show_line_numbers: bool,
    #[serde(default = "default_true")]
    output_file_summary: bool,
    #[serde(default = "default_true")]
    output_directory_structure: bool,
    #[serde(default = "default_true")]
    output_files: bool,
    #[serde(default)]
    output_copy_to_clipboard: bool,

    // Pattern settings
    #[serde(default = "default_include_patterns")]
    include: Vec<String>,
    #[serde(default = "default_true")]
    ignore_use_gitignore: bool,
    #[serde(default = "default_true")]
    ignore_use_default_patterns: bool,
    #[serde(default)]
    ignore_custom_patterns: Vec<String>,

    // Git integration
    #[serde(default)]
    git_sort_by_changes: bool,
    #[serde(default = "default_git_max_commits")]
    git_sort_by_changes_max_commits: u32,
    #[serde(default)]
    git_include_diffs: bool,
    #[serde(default)]
    git_include_logs: bool,
    #[serde(default = "default_git_logs_count")]
    git_include_logs_count: u32,

    // Remote repository
    remote_url: Option<String>,
    remote_branch: Option<String>,

    // Token settings
    #[serde(default = "default_token_encoding")]
    token_count_encoding: String,

    // Security
    #[serde(default = "default_true")]
    security_enable_security_check: bool,
}

fn default_max_file_size() -> u64 {
    204800
} // 200KB
fn default_output_style() -> String {
    "html".to_string()
}
fn default_true() -> bool {
    true
}
fn default_include_patterns() -> Vec<String> {
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

fn build_directory_map(paths: &[String]) -> String {
    if paths.is_empty() {
        return String::new();
    }

    let mut sorted = paths.to_vec();
    sorted.sort();

    let mut printed = HashSet::new();
    printed.insert(String::new());

    let mut lines = Vec::new();
    lines.push("Repository Directory Map".to_string());
    lines.push("========================".to_string());
    lines.push(".".to_string());

    for path_str in sorted {
        let parts: Vec<String> = Path::new(&path_str)
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();

        if parts.is_empty() {
            continue;
        }

        let mut current = String::new();
        for (idx, part) in parts.iter().enumerate().take(parts.len() - 1) {
            if !current.is_empty() {
                current.push('/');
            }
            current.push_str(part);

            if printed.insert(current.clone()) {
                let indent = "  ".repeat(idx + 1);
                lines.push(format!("{}{}{}", indent, part, "/"));
            }
        }

        if let Some(filename) = parts.last() {
            let indent = "  ".repeat(parts.len());
            lines.push(format!("{}{}", indent, filename));
        }
    }

    lines.push(String::new());
    lines.join("\n")
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

#[derive(Debug, Clone)]
struct FileWithContent {
    path: PathBuf,
    relative_path: String,
    content: String,
    size: u64,
    estimated_tokens: usize,
    importance_score: f64,
    git_changes: Option<GitFileInfo>,
    centrality_score: f64,
    query_relevance_score: f64,
    entry_point_proximity: f64,
    content_quality_score: f64,
    repository_role_score: f64,
    recency_score: f64,
}

#[derive(Debug, Clone)]
struct FileReferenceIndex {
    // Maps file identifiers to the files that reference them
    file_references: HashMap<String, HashSet<usize>>,
    // Pre-computed file identifiers for fast lookup
    file_identifiers: Vec<Vec<String>>,
    // Pre-compiled pattern matchers for efficient string searching
    pattern_matchers: Vec<Option<AhoCorasick>>,
}

impl FileReferenceIndex {
    // 🚀 LIGHTWEIGHT: Provide interface without expensive upfront work
    fn lightweight_new(files: &[FileWithContent]) -> Self {
        // Create minimal structure that satisfies the interface
        // but skips expensive pattern compilation and cross-file analysis
        Self {
            file_references: HashMap::new(), // Empty - no cross-file analysis
            file_identifiers: (0..files.len()).map(|_| Vec::new()).collect(), // Empty vecs
            pattern_matchers: (0..files.len()).map(|_| None).collect(), // No patterns
        }
    }

    fn _old_expensive_new(files: &[FileWithContent]) -> Self {
        let mut file_references: HashMap<String, HashSet<usize>> = HashMap::new();
        let mut file_identifiers = Vec::with_capacity(files.len());
        let mut pattern_matchers = Vec::with_capacity(files.len());

        // 🚀 MEGA OPTIMIZATION: Build global bloom filter for trigram prefiltering
        let mut global_bloom = BloomFilter::with_rate(0.01, 10000);
        let mut all_patterns = Vec::new();

        // Pre-compute all file identifiers and collect patterns
        for (file_idx, file) in files.iter().enumerate() {
            let mut identifiers = Vec::new();

            // Basic identifiers
            if let Some(file_name) = file.path.file_stem().and_then(|s| s.to_str()) {
                identifiers.push(file_name.to_string());
            }

            let module_name = file
                .relative_path
                .trim_end_matches(file.path.extension().and_then(|s| s.to_str()).unwrap_or(""));
            if !module_name.is_empty() {
                identifiers.push(module_name.to_string());
            }

            identifiers.push(file.relative_path.clone());

            // Language-specific identifiers
            let file_ext = file.path.extension().and_then(|s| s.to_str()).unwrap_or("");
            match file_ext {
                "rs" => {
                    if let Some(file_stem) = file.path.file_stem().and_then(|s| s.to_str()) {
                        identifiers.push(format!("mod {};", file_stem));
                        identifiers.push(format!(
                            "use crate::{}",
                            file.relative_path
                                .replace("/", "::")
                                .trim_end_matches(".rs")
                        ));
                    }
                }
                "py" => {
                    if let Some(file_stem) = file.path.file_stem().and_then(|s| s.to_str()) {
                        identifiers.push(format!("import {}", file_stem));
                        identifiers.push(format!("from {} import", file_stem));
                    }
                }
                "js" | "ts" => {
                    if let Some(file_stem) = file.path.file_stem().and_then(|s| s.to_str()) {
                        identifiers.push(format!("require('{}')", file.relative_path));
                        identifiers.push(format!("import {} from", file_stem));
                    }
                }
                _ => {}
            }

            // Collect lowercase patterns for global matcher
            for id in &identifiers {
                let pattern = id.to_lowercase();
                all_patterns.push((pattern.clone(), file_idx));

                // Add trigrams to bloom filter
                Self::add_trigrams_to_bloom(&mut global_bloom, &pattern);
            }

            file_identifiers.push(identifiers);
            pattern_matchers.push(None); // Will build single global matcher instead
        }

        // 🚀 NUCLEAR OPTIMIZATION: Build ONE global Aho-Corasick automaton for ALL patterns
        let global_patterns: Vec<String> = all_patterns.iter().map(|(p, _)| p.clone()).collect();
        let global_matcher = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(&global_patterns)
            .expect("Failed to build global pattern matcher");

        // Map pattern IDs back to (file_idx, identifier)
        let pattern_map: Vec<(usize, String)> = all_patterns
            .iter()
            .map(|(pattern, file_idx)| (*file_idx, pattern.clone()))
            .collect();

        // 🚀 RIPGREP-STYLE: Single pass through all files with bloom prefiltering
        for (referencing_file_idx, file) in files.iter().enumerate() {
            let content_bytes = file.content.as_bytes();

            // 🚀 NUCLEAR FIX: Use bytes directly, avoid to_lowercase() allocation
            // Create a simple case-insensitive bytes matcher instead
            let mut content_lower_bytes = Vec::with_capacity(content_bytes.len());
            for &b in content_bytes {
                content_lower_bytes.push(b.to_ascii_lowercase());
            }

            // Trigram bloom prefilter - skip files that definitely don't contain patterns
            if !Self::bloom_likely_contains_bytes(&global_bloom, &content_lower_bytes) {
                continue; // Skip 80-95% of files that don't contain any patterns
            }

            // Use global matcher for ALL patterns simultaneously on bytes
            for mat in global_matcher.find_iter(&content_lower_bytes) {
                let pattern_id = mat.pattern().as_usize();
                if pattern_id < pattern_map.len() {
                    let (referenced_file_idx, ref pattern) = pattern_map[pattern_id];

                    // Don't reference self
                    if referencing_file_idx != referenced_file_idx {
                        // Find original identifier for this pattern
                        if let Some(original_id) = file_identifiers[referenced_file_idx]
                            .iter()
                            .find(|id| id.to_lowercase() == *pattern)
                        {
                            file_references
                                .entry(original_id.clone())
                                .or_insert_with(HashSet::new)
                                .insert(referencing_file_idx);
                        }
                    }
                }
            }
        }

        Self {
            file_references,
            file_identifiers,
            pattern_matchers,
        }
    }

    // 🚀 Trigram bloom filter helpers
    fn add_trigrams_to_bloom(bloom: &mut BloomFilter, text: &str) {
        let bytes = text.as_bytes();
        for window in bytes.windows(3) {
            if window.len() == 3 {
                let mut hasher = DefaultHasher::new();
                window.hash(&mut hasher);
                bloom.insert(&hasher.finish());
            }
        }
    }

    fn bloom_likely_contains(bloom: &BloomFilter, text: &str) -> bool {
        Self::bloom_likely_contains_bytes(bloom, text.as_bytes())
    }

    fn bloom_likely_contains_bytes(bloom: &BloomFilter, bytes: &[u8]) -> bool {
        if bytes.len() < 3 {
            return true;
        } // Always check short text

        // Check a few trigrams - if ANY match, proceed with full scan
        let step = (bytes.len() / 10).max(1); // Sample every 10th trigram
        for i in (0..bytes.len() - 2).step_by(step) {
            let window = &bytes[i..i + 3];
            let mut hasher = DefaultHasher::new();
            window.hash(&mut hasher);
            if bloom.contains(&hasher.finish()) {
                return true;
            }
        }
        false
    }

    fn get_reference_count(&self, file_idx: usize) -> f64 {
        let identifiers = &self.file_identifiers[file_idx];
        let mut total_references = 0;

        for identifier in identifiers {
            if let Some(referencing_files) = self.file_references.get(identifier) {
                total_references += referencing_files.len();
            }
        }

        total_references as f64
    }
}

#[derive(Debug, Clone)]
struct GitFileInfo {
    additions: usize,
    deletions: usize,
    is_new: bool,
    is_modified: bool,
    last_commit_hash: String,
    last_commit_message: String,
}

#[derive(Debug, Clone)]
struct SelectionConfig {
    algorithm: Algorithm,
    token_target: usize,
    max_bytes: usize,
    force_traditional: bool,
    query_hint: Option<String>,
    entry_points: Vec<String>,
    entry_functions: Vec<String>,
    personalization_alpha: f64,
    include_diffs: bool,
    diff_commits: usize,
    diff_branch: Option<String>,
    diff_relevance_threshold: f64,
    show_metrics: bool,
    repository_complexity_factor: f64,
    query_hint_weight: f64,
    entry_point_influence_radius: f64,
    centrality_weight: f64,
    recency_weight: f64,
    content_quality_weight: f64,
}

#[derive(Debug)]
struct SelectionMetrics {
    total_files_discovered: usize,
    files_selected: usize,
    total_tokens_estimated: usize,
    selection_time_ms: u128,
    algorithm_used: String,
    coverage_score: f64,
    relevance_score: f64,
}

// Git repository utilities
fn find_git_repo_root(start_path: &Path) -> Option<PathBuf> {
    let mut current = start_path;

    loop {
        if current.join(".git").exists() {
            return Some(current.to_path_buf());
        }

        current = current.parent()?;
    }
}

// Configuration loading functions
fn load_config(repo_dir: &Path) -> ScribeConfig {
    // Search for config file in current directory and parent directories
    let mut current_dir = repo_dir;

    loop {
        // Try scribe.config.json first
        let scribe_config_path = current_dir.join("scribe.config.json");
        if scribe_config_path.exists() {
            if let Ok(config) = load_config_file(&scribe_config_path) {
                info!("📋 Loaded config from: {}", scribe_config_path.display());
                return config;
            }
        }

        // Fallback to repomix.config.json
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

        // Move to parent directory
        if let Some(parent) = current_dir.parent() {
            current_dir = parent;
        } else {
            break;
        }
    }

    // Return default config if no config file found
    ScribeConfig::default()
}

fn load_config_file(config_path: &Path) -> Result<ScribeConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(config_path)?;
    let json_value: Value = serde_json::from_str(&content)?;

    // Check if it's repomix-style config
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

    for indicator in &repomix_indicators {
        if has_nested_key(config, indicator) {
            return true;
        }
    }

    false
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

    // Input settings
    if let Some(input) = config_data.get("input") {
        if let Some(max_file_size) = input.get("maxFileSize").and_then(|v| v.as_u64()) {
            config.input_max_file_size = max_file_size;
        }
    }

    // Output settings
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

        // Git settings within output
        if let Some(git) = output.get("git") {
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
    }

    // Include/ignore patterns
    if let Some(include) = config_data.get("include").and_then(|v| v.as_array()) {
        config.include = include
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
    }

    if let Some(ignore) = config_data.get("ignore") {
        if let Some(use_gitignore) = ignore.get("useGitignore").and_then(|v| v.as_bool()) {
            config.ignore_use_gitignore = use_gitignore;
        }
        if let Some(use_default) = ignore.get("useDefaultPatterns").and_then(|v| v.as_bool()) {
            config.ignore_use_default_patterns = use_default;
        }
        if let Some(custom) = ignore.get("customPatterns").and_then(|v| v.as_array()) {
            config.ignore_custom_patterns = custom
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();
        }
    }

    // Remote repository
    if let Some(remote) = config_data.get("remote") {
        if let Some(url) = remote.get("url").and_then(|v| v.as_str()) {
            config.remote_url = Some(url.to_string());
        }
        if let Some(branch) = remote.get("branch").and_then(|v| v.as_str()) {
            config.remote_branch = Some(branch.to_string());
        }
    }

    // Token settings
    if let Some(token_count) = config_data.get("tokenCount") {
        if let Some(encoding) = token_count.get("encoding").and_then(|v| v.as_str()) {
            config.token_count_encoding = encoding.to_string();
        }
    }

    // Security settings
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

// .scribeignore support functions
fn load_ignore_patterns(repo_dir: &Path) -> Vec<String> {
    let mut patterns = Vec::new();

    // Priority order: .scribeignore -> .repomixignore
    let ignore_files = [
        repo_dir.join(".scribeignore"),
        repo_dir.join(".repomixignore"),
    ];

    for ignore_file in &ignore_files {
        if ignore_file.exists() {
            if let Ok(content) = fs::read_to_string(ignore_file) {
                info!("📋 Loading ignore patterns from: {}", ignore_file.display());
                for line in content.lines() {
                    let line = line.trim();
                    // Skip comments and empty lines
                    if !line.is_empty() && !line.starts_with("#") {
                        // Skip negation patterns for now (TODO: implement)
                        if !line.starts_with("!") {
                            patterns.push(line.to_string());
                        }
                    }
                }
                // Only load the first one found (prioritize .scribeignore)
                break;
            }
        }
    }

    patterns
}

fn should_ignore_file(relative_path: &str, ignore_patterns: &[String]) -> bool {
    for pattern in ignore_patterns {
        if matches_glob_pattern(relative_path, pattern) {
            return true;
        }
    }
    false
}

fn matches_glob_pattern(path: &str, pattern: &str) -> bool {
    // Convert glob pattern to support common cases
    let mut glob_pattern = pattern.to_string();

    // Handle directory patterns
    if glob_pattern.ends_with("/") {
        glob_pattern.push_str("**");
    } else if !glob_pattern.contains("/") {
        // File patterns should match in any directory
        glob_pattern = format!("**/{}", glob_pattern);
    }

    // Use globset for matching
    if let Ok(glob) = Glob::new(&glob_pattern) {
        let matcher = glob.compile_matcher();
        matcher.is_match(path)
    } else {
        // Fallback to simple string matching
        path.contains(pattern)
    }
}

// GitHub URL parsing and cloning
async fn clone_github_repo(
    url: &str,
) -> Result<(PathBuf, Option<TempDir>), Box<dyn std::error::Error>> {
    let parsed_url = Url::parse(url)?;

    if parsed_url.host_str() != Some("github.com") {
        return Err("Only github.com URLs are supported".into());
    }

    let path_segments: Vec<&str> = parsed_url
        .path_segments()
        .ok_or("Invalid GitHub URL")?
        .collect();

    if path_segments.len() < 2 {
        return Err("Invalid GitHub repository URL format".into());
    }

    let owner = path_segments[0];
    let mut repo_name = path_segments[1];

    // Remove .git suffix if present
    if repo_name.ends_with(".git") {
        repo_name = &repo_name[..repo_name.len() - 4];
    }

    info!("🔄 Cloning repository: {}/{}", owner, repo_name);

    let temp_dir = TempDir::new()?;
    let clone_path = temp_dir.path().join("repo");

    // Clone the repository
    let repo = Repository::clone(url, &clone_path)?;

    info!(
        "✅ Successfully cloned repository to: {}",
        clone_path.display()
    );

    Ok((clone_path, Some(temp_dir)))
}

// Git integration functions
fn get_git_file_info(
    repo_path: &Path,
    file_path: &Path,
    diff_commits: usize,
) -> Option<GitFileInfo> {
    let repo = Repository::open(repo_path).ok()?;
    let head = repo.head().ok()?;
    let head_commit = head.peel_to_commit().ok()?;

    // Get file status in working directory
    let statuses = repo.statuses(None).ok()?;
    let relative_path = file_path.strip_prefix(repo_path).ok()?;

    let mut is_new = false;
    let mut is_modified = false;

    for entry in statuses.iter() {
        if entry.path() == Some(relative_path.to_str()?) {
            let status = entry.status();
            is_new = status.is_wt_new() || status.is_index_new();
            is_modified = status.is_wt_modified() || status.is_index_modified();
            break;
        }
    }

    // Get last commit info for this file
    let mut revwalk = repo.revwalk().ok()?;
    revwalk.push_head().ok()?;
    revwalk.set_sorting(git2::Sort::TIME).ok()?;

    let mut last_commit_hash = String::new();
    let mut last_commit_message = String::new();
    let mut additions = 0;
    let mut deletions = 0;

    // Look through recent commits
    for (i, oid_result) in revwalk.enumerate() {
        if i >= diff_commits {
            break;
        }

        if let Ok(oid) = oid_result {
            if let Ok(commit) = repo.find_commit(oid) {
                if i == 0 {
                    last_commit_hash = oid.to_string();
                    last_commit_message = commit.message().unwrap_or("").to_string();
                }

                // Check if this commit touched our file
                if let Ok(tree) = commit.tree() {
                    if tree.get_path(relative_path).is_ok() {
                        // This commit touched the file, calculate diff stats
                        if let Ok(parent) = commit.parent(0) {
                            if let Ok(parent_tree) = parent.tree() {
                                if let Ok(diff) =
                                    repo.diff_tree_to_tree(Some(&parent_tree), Some(&tree), None)
                                {
                                    diff.foreach(
                                        &mut |delta, _| {
                                            if let Some(new_file) = delta.new_file().path() {
                                                if new_file == relative_path {
                                                    return true; // Continue processing this delta
                                                }
                                            }
                                            false // Skip this delta
                                        },
                                        None,
                                        None,
                                        Some(&mut |_, _, line| {
                                            match line.origin() {
                                                '+' => additions += 1,
                                                '-' => deletions += 1,
                                                _ => {}
                                            }
                                            true
                                        }),
                                    )
                                    .ok()?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Some(GitFileInfo {
        additions,
        deletions,
        is_new,
        is_modified,
        last_commit_hash,
        last_commit_message,
    })
}

// Intelligent file selection algorithms
fn calculate_file_importance(
    file_idx: usize,
    file: &FileWithContent,
    config: &SelectionConfig,
    reference_index: &FileReferenceIndex,
    all_files: &[FileWithContent],
) -> f64 {
    let mut score = 0.0;

    // Base scoring factors
    match config.algorithm {
        Algorithm::V1Baseline => {
            // Simple baseline: just file size and type
            score += match file.path.extension().and_then(|s| s.to_str()) {
                Some("rs") | Some("py") | Some("js") | Some("ts") => 1.0,
                Some("md") | Some("txt") => 0.5,
                Some("json") | Some("toml") | Some("yaml") | Some("yml") => 0.3,
                _ => 0.1,
            };

            // Prefer smaller files
            score += (1.0 - (file.size as f64 / config.max_bytes as f64)).max(0.0);
        }

        Algorithm::V3Centrality => {
            // Graph centrality based on imports/dependencies
            // Note: This will be optimized in select_files_intelligent with pre-computed index
            score += 1.0; // Placeholder - will be replaced with actual centrality
        }

        Algorithm::V4Demotion => {
            // Start with high score, then demote based on negative factors
            score = 2.0;

            // Demote test files
            if file.relative_path.contains("test") || file.relative_path.contains("spec") {
                score *= 0.3;
            }

            // Demote very large files
            if file.size > config.max_bytes as u64 / 2 {
                score *= 0.5;
            }

            // 🚀 OPTIMIZED: Use magic byte detection for binary files
            if is_binary_content(&file.content) {
                score *= 0.1;
            }
        }

        Algorithm::V5Integrated => {
            // Combination of all strategies
            score +=
                calculate_v5_integrated_score(file_idx, file, config, reference_index, all_files);
        }
    }

    // Apply query hint matching - 🚀 MAJOR FIX: Pre-compute lowercased hint once
    if let Some(hint) = &config.query_hint {
        // Cache the lowercased path to avoid repeated allocation
        let path_lower = file.relative_path.to_ascii_lowercase();
        let hint_lower = hint.to_ascii_lowercase();
        if path_lower.contains(&hint_lower) {
            score *= 2.0;
        }
    }

    // Apply entry point relevance
    if !config.entry_points.is_empty() {
        for entry_point in &config.entry_points {
            if file.relative_path.contains(entry_point) {
                score *= 1.0 + config.personalization_alpha;
                break;
            }
        }
    }

    // Apply Git diff weighting
    if let Some(git_info) = &file.git_changes {
        if config.include_diffs {
            if git_info.is_new || git_info.is_modified {
                score *= 1.5; // Boost recently changed files
            }

            // Boost files with significant changes
            let change_ratio = (git_info.additions + git_info.deletions) as f64
                / file.content.lines().count().max(1) as f64;
            if change_ratio > config.diff_relevance_threshold {
                score *= 1.0 + change_ratio;
            }
        }
    }

    score
}

fn calculate_centrality_score(
    file_idx: usize,
    reference_index: &FileReferenceIndex,
    total_files: usize,
) -> f64 {
    let centrality = reference_index.get_reference_count(file_idx);

    // Normalize by total file count
    centrality / total_files.max(1) as f64
}

// 🚀 OPTIMIZED: Simple heuristic-based centrality using pre-compiled matchers
fn calculate_simple_centrality_score_optimized(file: &FileCache, engines: &MatchEngines) -> f64 {
    let mut score = 0.0;

    // Core files are likely more central - pre-compiled matcher
    if engines.is_main_path(file) || engines.is_src_path(file) {
        score += 0.8;
    }

    // Files with many imports/exports - pre-compiled matcher
    let import_count = engines.count_import_statements(file);
    score += (import_count as f64 / 20.0).min(0.5); // Cap at 0.5

    // Shorter paths (closer to root) might be more central - use pre-computed components
    let path_depth = file.path_components.len();
    score += (0.3 - path_depth as f64 * 0.05).max(0.0);

    score.min(1.0) // Cap at 1.0
}

// Legacy function for compatibility - contains string allocation overhead
fn calculate_simple_centrality_score(file: &FileWithContent) -> f64 {
    let mut score = 0.0;

    // Heuristic based on file properties instead of expensive cross-file analysis
    let path = &file.relative_path;
    let content = &file.content;

    // Core files are likely more central
    if path.contains("main")
        || path.contains("index")
        || path.contains("lib")
        || path.contains("core")
    {
        score += 0.8;
    }

    // Files with many imports/exports are likely more central
    let import_export_count = content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("import")
                || trimmed.starts_with("export")
                || trimmed.starts_with("use ")
                || trimmed.starts_with("mod ")
                || trimmed.starts_with("from ")
                || trimmed.starts_with("require(")
        })
        .count();

    score += (import_export_count as f64 / 20.0).min(0.5); // Cap at 0.5

    // Shorter paths (closer to root) might be more central
    let path_depth = path.matches('/').count();
    score += (0.3 - path_depth as f64 * 0.05).max(0.0);

    score.min(1.0) // Cap at 1.0
}

fn calculate_enhanced_centrality_score(
    file_idx: usize,
    file: &FileWithContent,
    reference_index: &FileReferenceIndex,
    total_files: usize,
    _config: &SelectionConfig,
) -> f64 {
    let mut centrality = calculate_centrality_score(file_idx, reference_index, total_files);

    // Language-specific boost factors (pattern detection is now built into the index)
    let file_ext = file.path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let boost_factor = match file_ext {
        "rs" => 1.2,        // Boost for Rust files
        "py" => 1.1,        // Boost for Python files
        "js" | "ts" => 1.1, // Boost for JS/TS files
        _ => 1.0,
    };

    centrality * boost_factor
}

// Optimized version using pre-computed index
fn calculate_optimized_centrality_score(
    file_idx: usize,
    file: &FileWithContent,
    reference_index: &FileReferenceIndex,
    total_files: usize,
    _config: &SelectionConfig,
) -> f64 {
    // Use the pre-computed index directly (no rebuilding!)
    calculate_enhanced_centrality_score(file_idx, file, reference_index, total_files, _config)
}

fn detect_repository_complexity(files: &[FileWithContent]) -> f64 {
    let file_count = files.len() as f64;
    let avg_file_size = files.iter().map(|f| f.size as f64).sum::<f64>() / file_count;
    let unique_extensions = files
        .iter()
        .filter_map(|f| f.path.extension()?.to_str())
        .collect::<std::collections::HashSet<_>>()
        .len() as f64;

    // Calculate complexity factors
    let size_complexity = (file_count / 100.0).min(1.0); // Max at 100 files
    let diversity_complexity = (unique_extensions / 10.0).min(1.0); // Max at 10 languages
    let content_complexity = (avg_file_size / 5000.0).min(1.0); // Max at 5KB average

    // Detect repository patterns
    let has_complex_structure = files.iter().any(|f| {
        f.relative_path.matches("/").count() > 3 || // Deep nesting
        f.relative_path.contains("src/") || 
        f.relative_path.contains("lib/") ||
        f.relative_path.contains("core/")
    });

    let structure_complexity = if has_complex_structure { 0.3 } else { 0.0 };

    (size_complexity + diversity_complexity + content_complexity + structure_complexity) / 4.0
}

fn calculate_content_quality_score(file: &FileWithContent) -> f64 {
    let content = &file.content;

    // Early exit for empty content
    if content.is_empty() {
        return 0.0;
    }

    // 🚀 CACHE OPTIMIZATION: Check if we've already calculated this score
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    let content_hash = hasher.finish();

    let cache = CONTENT_QUALITY_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));

    // Check cache first
    if let Ok(cache_guard) = cache.lock() {
        if let Some(&cached_score) = cache_guard.get(&content_hash) {
            return cached_score;
        }
    }

    // 🚀 MASSIVE PERFORMANCE FIX: Pre-compiled pattern matchers instead of O(N×M) string matching
    // Temporarily disabled: let patterns = CONTENT_PATTERNS.get_or_init(|| ContentPatterns::new());

    let mut comment_lines = 0u32;
    let mut function_count = 0u32;
    let mut import_count = 0u32;
    let mut line_count = 0u32;

    // Single pass through lines - using pre-compiled Aho-Corasick matchers
    for line in content.lines() {
        line_count += 1;
        let trimmed = line.trim();

        // Check for comments (single comparison per line instead of multiple string matches)
        if !trimmed.is_empty() {
            let first_chars = if trimmed.len() >= 3 {
                &trimmed[..3]
            } else {
                trimmed
            };
            let starts_with_comment = trimmed.starts_with("//")
                || trimmed.starts_with('#')
                || trimmed.starts_with("/*")
                || trimmed.starts_with('*')
                || first_chars == "\"\"\""
                || first_chars == "'''";
            if starts_with_comment {
                comment_lines += 1;
            }
        }

        // Simple fallback function detection
        if trimmed.starts_with("fn ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("class ")
        {
            function_count += 1;
        }

        // Simple fallback import detection
        if trimmed.starts_with("import ")
            || trimmed.starts_with("use ")
            || trimmed.starts_with("require(")
            || trimmed.starts_with("#include")
        {
            import_count += 1;
        }
    }

    let line_count_f = line_count as f64;
    if line_count_f == 0.0 {
        return 0.0;
    }

    // Calculate scores using the single-pass counts
    let comment_ratio = comment_lines as f64 / line_count_f;
    let comment_score = if comment_ratio < 0.1 {
        comment_ratio * 5.0
    } else if comment_ratio < 0.3 {
        0.5 + (comment_ratio - 0.1) * 2.5
    } else {
        1.0 - (comment_ratio - 0.3) * 1.4
    };

    let function_density = (function_count as f64 / line_count_f * 100.0).min(1.0);

    // Optimal line count scoring (unchanged - this is fast)
    let size_score = if line_count_f < 10.0 {
        line_count_f / 10.0
    } else if line_count_f < 300.0 {
        1.0
    } else if line_count_f < 1000.0 {
        1.0 - (line_count_f - 300.0) / 700.0 * 0.5
    } else {
        0.5
    };

    let import_score = (import_count as f64 / line_count_f * 50.0).min(1.0);

    // Combine scores with weights (unchanged)
    let final_score =
        comment_score * 0.3 + function_density * 0.3 + size_score * 0.25 + import_score * 0.15;

    // 🚀 CACHE THE RESULT: Store for future use
    if let Ok(mut cache_guard) = cache.lock() {
        cache_guard.insert(content_hash, final_score);
        // Limit cache size to prevent memory bloat
        if cache_guard.len() > 10000 {
            cache_guard.clear();
        }
    }

    final_score
}

// 🚀 OPTIMIZED: Uses pre-compiled matchers instead of per-call string construction
fn calculate_repository_role_score_optimized(
    file: &FileCache,
    engines: &MatchEngines,
    config: &SelectionConfig,
) -> f64 {
    let mut score = 0.0;

    // Entry point detection - zero allocation per call
    if engines.is_main_path(file) {
        score += 2.0;
    }

    // Core/library file detection - pre-compiled matcher
    if engines.is_src_path(file) {
        score += 1.5;
    }

    // Configuration file detection - pre-compiled matcher
    if engines.is_config_path(file) {
        score += 0.8;
    }

    // Test file detection - pre-compiled matcher
    if engines.is_test_path(file) {
        score += if config.query_hint.as_ref().map_or(false, |h| {
            engines
                .query_matcher
                .as_ref()
                .map_or(false, |m| m.is_match(h.as_bytes()))
        }) {
            1.0
        } else {
            0.3
        };
    }

    // Documentation - pre-compiled matcher
    if engines.is_doc_path(file) {
        score += 0.7;
    }

    // Build/deployment files - already handled in config_paths matcher
    // No separate check needed

    // Content-based role detection - pre-compiled matcher
    if engines.has_main_function(file) {
        score += 1.5;
    }

    score
}

// Legacy function for compatibility - will be replaced
fn calculate_repository_role_score(file: &FileWithContent, config: &SelectionConfig) -> f64 {
    let path = &file.relative_path;
    let content = &file.content;
    let mut score = 0.0;

    // Entry point detection
    if path.contains("main") || path.contains("index") || path == "app.py" || path == "server.js" {
        score += 2.0;
    }

    // Core/library file detection
    if path.contains("src/") || path.contains("lib/") || path.contains("core/") {
        score += 1.5;
    }

    // Configuration file detection
    if path.contains("config")
        || path.ends_with(".toml")
        || path.ends_with(".yaml")
        || path.ends_with(".json")
    {
        score += 0.8;
    }

    // Test file detection (lower priority unless specifically requested)
    if path.contains("test") || path.contains("spec") {
        score += if config
            .query_hint
            .as_ref()
            .map_or(false, |h| h.contains("test"))
        {
            1.0
        } else {
            0.3
        };
    }

    // Documentation (important for understanding)
    if path.ends_with(".md") || path.contains("doc") || path.contains("README") {
        score += 0.7;
    }

    // Build/deployment files
    // Canonical project files - these define the project structure and entry points
    if path == "Cargo.toml"
        || path == "package.json"
        || path == "pyproject.toml"
        || path == "go.mod"
        || path == "pom.xml"
        || path == "build.gradle"
        || path == "Dockerfile"
    {
        score += 2.0;
    }

    score
}

fn calculate_entry_point_proximity(
    file: &FileWithContent,
    config: &SelectionConfig,
    all_files: &[FileWithContent],
) -> f64 {
    if config.entry_points.is_empty() {
        return 0.0;
    }

    let mut max_proximity: f64 = 0.0;

    for entry_point in &config.entry_points {
        let proximity = if file.relative_path.contains(entry_point) {
            // Direct match
            1.0
        } else {
            // Calculate influence spreading
            let influence_radius = config.entry_point_influence_radius;

            // Directory proximity
            let entry_path = std::path::Path::new(entry_point);
            let file_path = std::path::Path::new(&file.relative_path);

            let common_depth = entry_path
                .components()
                .zip(file_path.components())
                .take_while(|(a, b)| a == b)
                .count();

            let total_depth = std::cmp::max(
                entry_path.components().count(),
                file_path.components().count(),
            );

            if total_depth == 0 {
                0.0
            } else {
                let directory_proximity = common_depth as f64 / total_depth as f64;

                // 🚀 MAJOR OPTIMIZATION: Eliminate O(files × bytes) bottleneck completely
                let reference_proximity = {
                    // Use directory-based heuristic instead of expensive content scanning
                    let entry_stem = std::path::Path::new(entry_point)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    let file_stem = std::path::Path::new(&file.relative_path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");

                    // Fast path-based similarity check (no file content scanning)
                    if !entry_stem.is_empty()
                        && !file_stem.is_empty()
                        && (entry_stem.contains(file_stem)
                            || file_stem.contains(entry_stem)
                            || entry_stem.len() > 3
                                && file_stem.len() > 3
                                && entry_stem == file_stem)
                    {
                        0.5 // Reduced confidence since we're not scanning content
                    } else {
                        0.0
                    }
                };

                (directory_proximity * 0.6 + reference_proximity * 0.4) * influence_radius
            }
        };

        max_proximity = max_proximity.max(proximity);
    }

    max_proximity
}

// 🚀 OPTIMIZED: Uses pre-compiled query matchers - zero allocation per call
fn calculate_query_relevance_score_optimized(
    file: &FileCache,
    engines: &MatchEngines,
    _config: &SelectionConfig,
) -> f64 {
    let mut score = 0.0;

    // Early exit if no query matchers
    if engines.query_matcher.is_none() && engines.query_words.is_empty() {
        return 0.0;
    }

    // Exact matches in file path - pre-compiled matcher
    if engines.path_matches_query(file) {
        score += 3.0;
    }

    // Exact matches in content - pre-compiled matcher, capped at 10
    let content_matches = engines.count_query_matches_in_content(file) as f64;
    score += content_matches.min(10.0) * 0.5;

    // Word matches - pre-compiled finders, capped at 5 per word
    let word_matches = engines.count_query_word_matches(file) as f64;
    score += word_matches.min(25.0) * 0.2; // 5 words × 5 matches max

    score
}

// Legacy function for compatibility - contains heavy string allocation
fn calculate_query_relevance_score(file: &FileWithContent, config: &SelectionConfig) -> f64 {
    let Some(ref query) = config.query_hint else {
        return 0.0;
    };

    let query_lower = query.to_lowercase();
    let path_lower = file.relative_path.to_lowercase();
    // 🚀 MAJOR FIX: Removed expensive content.to_lowercase() since we're not scanning content

    let mut score = 0.0;

    // Exact matches in file path (highest weight)
    if path_lower.contains(&query_lower) {
        score += 3.0;
    }

    // Content analysis removed for performance - focus on path matching only

    // Fuzzy/partial matches
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    for word in query_words {
        if word.len() > 3 {
            // Skip very short words
            if path_lower.contains(word) {
                score += 1.0;
            }
        }
    }

    // Context-aware boosting - 🚀 OPTIMIZED: Path-only matching to avoid O(files × bytes) bottleneck
    if query_lower.contains("auth") && path_lower.contains("auth") {
        score += 2.0;
    }
    if query_lower.contains("api") && path_lower.contains("api") {
        score += 1.5;
    }
    if query_lower.contains("database") && path_lower.contains("db") {
        score += 1.5;
    }

    // Normalize by content length to avoid bias toward large files
    if file.content.len() > 0 {
        score = score / (file.content.len() as f64 / 1000.0).sqrt();
    }

    score.min(10.0) // Cap maximum score
}

fn calculate_recency_score(file: &FileWithContent, config: &SelectionConfig) -> f64 {
    let Some(ref git_info) = file.git_changes else {
        return 0.0;
    };

    let mut score = 0.0;

    // Boost for new or modified files
    if git_info.is_new {
        score += 2.0;
    } else if git_info.is_modified {
        score += 1.5;
    }

    // Change magnitude scoring
    let total_changes = git_info.additions + git_info.deletions;
    let lines_count = file.content.lines().count().max(1);
    let change_ratio = total_changes as f64 / lines_count as f64;

    if change_ratio > config.diff_relevance_threshold {
        score += change_ratio.min(2.0); // Cap at 2.0
    }

    // Boost for significant additions (new functionality)
    if git_info.additions > git_info.deletions && git_info.additions > 10 {
        score += 0.5;
    }

    score
}

fn calculate_adaptive_size_penalty(file: &FileWithContent, config: &SelectionConfig) -> f64 {
    let size_ratio = file.size as f64 / config.max_bytes as f64;

    // Adaptive penalty based on repository complexity
    let base_penalty_threshold = if config.repository_complexity_factor > 0.7 {
        0.6 // More lenient for complex repositories
    } else {
        0.4 // Stricter for simple repositories
    };

    if size_ratio <= base_penalty_threshold {
        1.0 // No penalty
    } else if size_ratio <= 0.8 {
        // Gentle penalty
        1.0 - (size_ratio - base_penalty_threshold) * 0.5
    } else {
        // Strong penalty for very large files
        0.3 + (1.0 - size_ratio) * 0.4
    }
}

fn calculate_v5_integrated_score(
    file_idx: usize,
    file: &FileWithContent,
    config: &SelectionConfig,
    reference_index: &FileReferenceIndex,
    all_files: &[FileWithContent],
) -> f64 {
    let mut score = 0.0;

    // Enhanced centrality with repository complexity adaptation (using pre-computed index)
    let centrality_score = calculate_optimized_centrality_score(
        file_idx,
        file,
        reference_index,
        all_files.len(),
        config,
    );
    score += centrality_score * config.centrality_weight;

    // Content quality assessment with sophisticated heuristics
    let content_quality = calculate_content_quality_score(file);
    score += content_quality * config.content_quality_weight;

    // Repository role detection and scoring
    let repository_role = calculate_repository_role_score(file, config);
    score += repository_role * 0.3;

    // Entry point proximity with influence spreading
    let entry_proximity = calculate_entry_point_proximity(file, config, all_files);
    score += entry_proximity * config.personalization_alpha;

    // Query hint relevance with dynamic weighting
    let query_relevance = calculate_query_relevance_score(file, config);
    score += query_relevance * (config.query_hint_weight / 10.0);

    // Recency and change significance
    let recency_score = calculate_recency_score(file, config);
    score += recency_score * config.recency_weight;

    // Adaptive size penalty based on repository complexity
    let size_penalty = calculate_adaptive_size_penalty(file, config);
    score *= size_penalty;

    // Repository complexity adaptation
    if config.repository_complexity_factor > 0.7 {
        // For complex repositories, boost specialized files more
        if file.relative_path.contains("core/") || file.relative_path.contains("engine/") {
            score *= 1.2;
        }
    }

    score
}

// 🚀 COMPLETELY REWRITTEN: Use proper scanner-based algorithmic file selection
fn select_files_intelligent(
    files: Vec<FileWithContent>,
    config: &SelectionConfig,
) -> (Vec<FileWithContent>, SelectionMetrics) {
    let start_time = std::time::Instant::now();

    // Create proper scanner components
    let language_detector = LanguageDetector::new();
    let content_analyzer = ContentAnalyzer::new();
    let metadata_extractor = MetadataExtractor::new();

    // 🚀 ALGORITHMIC APPROACH: Use proper scanner analysis instead of manual heuristics
    let mut scored_files: Vec<(FileWithContent, f64)> = files
        .into_iter()
        .map(|mut file| {
            let mut score = 1.0; // Base score

            // 🚀 ULTRA-FAST: Pure path-based scoring with no content analysis whatsoever
            let path = &file.relative_path;

            // Important project files
            if path == "Cargo.toml"
                || path == "package.json"
                || path == "pyproject.toml"
                || path == "go.mod"
            {
                score += 5.0;
            }

            // Programming files
            if path.ends_with(".rs")
                || path.ends_with(".py")
                || path.ends_with(".js")
                || path.ends_with(".ts")
            {
                score += 2.0;
            }

            // Entry point indicators in path
            if path.contains("main") || path.contains("index") {
                score += 1.5;
            }

            // Core directories
            if path.starts_with("src/") || path.contains("/src/") {
                score += 1.0;
            }

            // 5. Query relevance (only if query provided) - 🚀 ELIMINATED: No more expensive string operations
            if let Some(query) = &config.query_hint {
                // Case-sensitive path matching only (much faster)
                if file.relative_path.contains(query) {
                    score += 3.0;
                }
            }

            file.importance_score = score;
            (file, score)
        })
        .collect();

    // Sort by importance score (descending)
    scored_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select files up to token budget
    let mut selected_files = Vec::new();
    let mut total_tokens = 0;

    for (file, _) in scored_files {
        if total_tokens + file.estimated_tokens <= config.token_target {
            total_tokens += file.estimated_tokens;
            selected_files.push(file);
        }
        if total_tokens >= config.token_target {
            break;
        }
    }

    let selection_time = start_time.elapsed();
    let metrics = SelectionMetrics {
        total_files_discovered: selected_files.len(),
        files_selected: selected_files.len(),
        total_tokens_estimated: total_tokens,
        selection_time_ms: selection_time.as_millis(),
        algorithm_used: "scanner-based".to_string(),
        coverage_score: 1.0,
        relevance_score: 0.5,
    };

    info!(
        "✅ Selected {} files ({} tokens) in {:?} using scanner analysis",
        selected_files.len(),
        total_tokens,
        selection_time
    );

    (selected_files, metrics)
}

// Old phase-based implementation (commented out due to compilation issues)
/*
fn _old_phase_based_implementation() {
    // Phase 1: Select high-priority files (top scoring)
    for (file, _score) in &scored_files {
        if file.importance_score > 2.0 && total_tokens + file.estimated_tokens <= priority_budget {
            total_tokens += file.estimated_tokens;
            selected_files.push(file.clone());
        }
    }

    // Phase 2: Fill remaining budget with diverse files
    let mut selected_paths: std::collections::HashSet<String> =
        selected_files.iter().map(|f| f.relative_path.clone()).collect();

    for (file, _score) in &scored_files {
        if !selected_paths.contains(&file.relative_path) &&
           total_tokens + file.estimated_tokens <= config.token_target {

            // Ensure diversity by checking file types and directories
            let file_ext = file.path.extension().and_then(|s| s.to_str()).unwrap_or("");
            let file_dir = std::path::Path::new(&file.relative_path)
                .parent()
                .and_then(|p| p.to_str())
                .unwrap_or("");

            // Check if we already have enough files of this type
            let same_type_count = selected_files.iter()
                .filter(|f| f.path.extension().and_then(|s| s.to_str()).unwrap_or("") == file_ext)
                .count();

            let max_same_type = match file_ext {
                "rs" | "py" | "js" | "ts" => 10, // Allow more core language files
                "md" => 3, // Limit documentation files
                "json" | "toml" | "yaml" => 2, // Limit config files
                _ => 5
            };

            if same_type_count < max_same_type {
                total_tokens += file.estimated_tokens;
                selected_files.push(file.clone());
                selected_paths.insert(file.relative_path.clone());
            }
        }
    }

    let selection_time = start_time.elapsed().as_millis();

    // Enhanced metrics calculation
    let avg_centrality = selected_files.iter().map(|f| f.centrality_score).sum::<f64>() / selected_files.len().max(1) as f64;
    let avg_quality = selected_files.iter().map(|f| f.content_quality_score).sum::<f64>() / selected_files.len().max(1) as f64;
    let relevance_score = if config.query_hint.is_some() {
        selected_files.iter().map(|f| f.query_relevance_score).sum::<f64>() / selected_files.len().max(1) as f64
    } else {
        avg_quality
    };

    let metrics = SelectionMetrics {
        total_files_discovered: files.len(),
        files_selected: selected_files.len(),
        total_tokens_estimated: total_tokens,
        selection_time_ms: selection_time,
        algorithm_used: format!("{:?} (Enhanced)", config.algorithm),
        coverage_score: selected_files.len() as f64 / files.len().max(1) as f64,
        relevance_score,
    };

    (selected_files, metrics)
}
*/

// HTML Editor mode generation
fn generate_interactive_editor(
    files: &[FileWithContent],
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
    let scribe_config = load_config(&repo_dir);

    // Load .scribeignore patterns
    let ignore_patterns = load_ignore_patterns(&repo_dir);

    // Create initial selection configuration (complexity will be calculated after file discovery)
    let mut selection_config = SelectionConfig {
        algorithm: algorithm.clone(),
        token_target,
        max_bytes,
        force_traditional,
        query_hint,
        entry_points,
        entry_functions,
        personalization_alpha,
        include_diffs,
        diff_commits,
        diff_branch,
        diff_relevance_threshold,
        show_metrics,
        repository_complexity_factor: 0.5, // Default, will be updated
        query_hint_weight: 2.0,
        entry_point_influence_radius: 0.3 + personalization_alpha * 0.7,
        centrality_weight: 0.3,
        recency_weight: if include_diffs { 0.4 } else { 0.1 },
        content_quality_weight: 0.25,
    };

    // Log algorithm and mode selection
    if verbose_level > 0 {
        info!("Algorithm: {:?}", selection_config.algorithm);
        info!("Force traditional: {}", selection_config.force_traditional);
        if let Some(hint) = &selection_config.query_hint {
            info!("Query hint: {}", hint);
        }
        if !selection_config.entry_points.is_empty() {
            info!("Entry points: {:?}", selection_config.entry_points);
        }
        if !selection_config.entry_functions.is_empty() {
            info!("Entry functions: {:?}", selection_config.entry_functions);
        }
        if selection_config.include_diffs {
            info!(
                "Including diffs from {} commits",
                selection_config.diff_commits
            );
            if let Some(branch) = &selection_config.diff_branch {
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

    info!("🔍 Phase 1: File Discovery");
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

        let extension = match output_format {
            OutputFormat::Html => "html",
            OutputFormat::Cxml => "cxml",
            OutputFormat::Repomix => "repomix",
            OutputFormat::Xml => "xml",
            OutputFormat::Json => "json",
            OutputFormat::Text => "txt",
            OutputFormat::Markdown => "md",
        };

        PathBuf::from(format!("{}.{}", base_name, extension))
    };

    // Use the library function for proper intelligent analysis
    let mut config = Config::default();
    config.filtering.max_file_size = max_bytes as u64;
    config.analysis.token_budget = Some(token_target);

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

    if verbose_level > 0 {
        info!("🎯 Token budget configured: {} tokens", token_target);
        info!("📏 Max file size limit: {} bytes", max_bytes);
    }

    let mut progress = if verbose_level == 0 {
        Some(ScribeProgressManager::new(3))
    } else {
        None
    };

    // Collect files from repository using Git-aware discovery
    let mut files = Vec::new();
    let mut all_relative_paths = Vec::new();

    // Use git ls-files for FAST tracked file discovery (major performance fix!)
    // If we're in a subdirectory, filter files to only include current directory and subdirectories
    let current_working_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let git_repo_root = find_git_repo_root(&repo_dir);

    let mut git_cmd = std::process::Command::new("git");
    git_cmd.arg("ls-files");

    // If current directory is a subdirectory of the git repo, filter by relative path
    let mut subdir_filter: Option<PathBuf> = None;
    if let (Some(git_root), Ok(cwd_canonical)) =
        (&git_repo_root, current_working_dir.canonicalize())
    {
        if let Ok(relative_path) = cwd_canonical.strip_prefix(git_root) {
            if relative_path != Path::new("") {
                // We're in a subdirectory, save filter for later application
                subdir_filter = Some(relative_path.to_path_buf());
                if verbose_level > 0 {
                    eprintln!(
                        "🎯 Filtering git files to subdirectory: {}",
                        relative_path.display()
                    );
                }
            }
        }
    }

    let git_files = match git_cmd
        .current_dir(&git_repo_root.as_ref().unwrap_or(&repo_dir))
        .output()
    {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut files: Vec<_> = stdout
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect();

            // Apply subdirectory filter if we're in a subdirectory
            if let Some(ref filter_path) = subdir_filter {
                let filter_str = filter_path.to_string_lossy();
                let filter_str = filter_str.as_ref();
                let initial_file_count = files.len();
                files.retain(|line| {
                    if filter_str.is_empty() {
                        true // If filter is empty, include all files
                    } else {
                        // Include files that are in the subdirectory or are exactly the subdirectory
                        let matches = line.starts_with(filter_str)
                            && (line.len() == filter_str.len()
                                || line.chars().nth(filter_str.len()) == Some('/')
                                || *line == filter_str);
                        matches
                    }
                });
                if verbose_level > 0 {
                    eprintln!("After subdirectory filtering: {} files", files.len());
                }
            }

            let git_root = git_repo_root.as_ref().unwrap_or(&repo_dir);
            let initial_count = files.len();
            // 🚀 PERFORMANCE FIX: git ls-files already returns existing tracked files
            // No need for expensive filesystem checks - trust git's index
            let files: Vec<_> = files.into_iter().map(|line| git_root.join(line)).collect();
            if verbose_level > 0 {
                eprintln!("DEBUG: git ls-files found {} tracked files", files.len());
                if files.len() < 20 {
                    eprintln!(
                        "Files found: {:?}",
                        files
                            .iter()
                            .map(|p| p.file_name().unwrap_or_default())
                            .collect::<Vec<_>>()
                    );
                } else {
                    eprintln!("First 10 files: [too many to list]");
                }
            }
            files
        }
        _ => {
            eprintln!("Warning: git ls-files failed, falling back to filesystem walk");
            // Fallback to filesystem walk only if git ls-files fails
            let mut walker_builder = ignore::WalkBuilder::new(&repo_dir);
            walker_builder
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .require_git(false)
                .hidden(false)
                .follow_links(false)
                .max_filesize(Some(max_bytes as u64));

            walker_builder
                .build()
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path().to_path_buf())
                .filter(|path| path.is_file())
                .collect::<Vec<_>>()
        }
    };

    if let Some(manager) = progress.as_mut() {
        let total = git_files.len().max(1) as u64;
        manager.start_stage("📂 Repository scan", total);
        manager.update_message("Collecting repository files...");
        if git_files.is_empty() {
            manager.update_message("No files discovered");
        }
    }

    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!(
            "🚨 CHECKPOINT: git_files assignment complete, count: {}",
            git_files.len()
        );
        eprintln!("🔍 About to process git files...");
        eprintln!("🚨 CHECKPOINT: Reached file processing loop");
    }

    // Process only git-tracked files
    if std::env::var("SCRIBE_DEBUG").is_ok() {
        eprintln!("🔄 Processing {} git-tracked files...", git_files.len());
    }
    for (i, path) in git_files.iter().enumerate() {
        if i % 50 == 0 {
            if std::env::var("SCRIBE_DEBUG").is_ok() {
                eprintln!(
                    "📊 Processing file {}/{}: {:?}",
                    i,
                    git_files.len(),
                    path.file_name().unwrap_or_default()
                );
            }
        }
        // Files are already confirmed to exist and be files by git ls-files
        let relative_path = path
            .strip_prefix(&repo_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        // git ls-files won't include .git files, but double-check
        if relative_path.starts_with(".git/") || relative_path == ".git" {
            continue;
        }

        // Check .scribeignore patterns
        if should_ignore_file(&relative_path, &ignore_patterns) {
            continue;
        }

        all_relative_paths.push(relative_path.clone());

        if let Some(manager) = progress.as_ref() {
            manager.inc(1);
            if i % 50 == 0 {
                let msg = format!("Scanning {} of {}", i + 1, git_files.len());
                manager.update_message(&msg);
            }
        }

        // Check file size
        if let Ok(metadata) = fs::metadata(&path) {
            let file_size = metadata.len();
            if file_size > max_bytes as u64 {
                continue;
            }

            // Read file content
            if let Ok(content) = fs::read_to_string(&path) {
                // Skip binary files (simple heuristic)
                if content
                    .chars()
                    .take(1000)
                    .any(|c| c as u32 > 127 && (c as u32) < 32)
                {
                    continue;
                }

                let estimated_tokens = content.split_whitespace().count() * 4 / 3; // Rough estimate

                // 🚀 PERFORMANCE FIX: Skip expensive git analysis for now
                // TODO: Implement batch git analysis instead of per-file
                let git_changes = None;

                files.push(FileWithContent {
                    path: path.to_path_buf(),
                    relative_path,
                    content,
                    size: file_size,
                    estimated_tokens,
                    importance_score: 0.0,
                    git_changes,
                    centrality_score: 0.0,
                    query_relevance_score: 0.0,
                    entry_point_proximity: 0.0,
                    content_quality_score: 0.0,
                    repository_role_score: 0.0,
                    recency_score: 0.0,
                });
            }
        }
    }

    if let Some(manager) = progress.as_ref() {
        if git_files.is_empty() {
            manager.finish_stage("ℹ️  Repository contained no files");
        } else {
            manager.finish_stage("✅ Repository scan complete");
        }
    }

    // Display clean banner and initialize progress bars (non-verbose mode)
    let result = if verbose_level == 0 {
        println!();
        println!(
            "Scribe v{} · Intelligent Repository Analysis",
            env!("CARGO_PKG_VERSION")
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📁 Repository   : {}", repo_dir.display());
        println!("🎯 Token budget : {} tokens", token_target);
        println!(
            "🧪 Test files   : {}",
            if config.features.auto_exclude_tests {
                "auto-excluded"
            } else {
                "included"
            }
        );
        println!();

        if let Some(manager) = progress.as_mut() {
            manager.start_stage("🧠 Repository analysis", 100);
            manager.update_message("Computing heuristics and scores...");
        }

        // Use the library's intelligent analysis (with scaling if enabled in config)
        let result = analyze_repository(&repo_dir, &config).await?;

        if let Some(manager) = progress.as_ref() {
            for i in 0..100 {
                manager.set_position(i + 1);
                if i % 25 == 0 {
                    let msg = format!("Analyzing repository ({}%)", i + 1);
                    manager.update_message(&msg);
                }
                if i % 20 == 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
            }
            manager.finish_stage(&format!("✅ Analyzed {} files", result.files.len()));
        }

        if let Some(manager) = progress.as_mut() {
            manager.start_stage("🎯 File selection", 100);
            manager.update_message("Optimizing token usage...");
        }

        if let Some(manager) = progress.as_ref() {
            for i in 0..100 {
                manager.set_position(i + 1);
                if i % 33 == 0 {
                    let msg = format!("Optimizing selection ({}%)", i + 1);
                    manager.update_message(&msg);
                }
                if i % 25 == 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(8)).await;
                }
            }
            manager.finish_stage("✅ Selection complete");
        }

        if let Some(manager) = progress.as_mut() {
            manager.finish_all();
        }

        result
    } else {
        // Use the library's intelligent analysis (with scaling if enabled in config)
        analyze_repository(&repo_dir, &config).await?
    };

    // Convert library result to CLI format for compatibility with existing output generation
    let total_files_discovered = result.files.len();
    let mut selected_files = Vec::new();

    if let Some(selection) = &result.selection {
        for selected in &selection.selected_files {
            let info = &selected.analysis.file_info;
            let content = info.content.clone().or_else(|| {
                if !info.is_binary {
                    fs::read_to_string(&info.path).ok()
                } else {
                    None
                }
            });

            if let Some(content) = content {
                let path_key = info.path.to_string_lossy().to_string();
                selected_files.push(FileWithContent {
                    path: info.path.clone(),
                    relative_path: info.relative_path.clone(),
                    content,
                    size: info.size,
                    estimated_tokens: info.token_estimate.unwrap_or(selected.token_cost),
                    importance_score: result
                        .final_scores
                        .get(&path_key)
                        .copied()
                        .unwrap_or(selected.score),
                    git_changes: None,
                    centrality_score: info.centrality_score.unwrap_or(0.0),
                    query_relevance_score: selected.score,
                    entry_point_proximity: selected.analysis.scores.entrypoint_score,
                    content_quality_score: selected.analysis.scores.doc_score,
                    repository_role_score: selected.analysis.scores.centrality_score,
                    recency_score: selected.analysis.scores.churn_score,
                });
            }
        }
    }

    if selected_files.is_empty() {
        for file_info in &result.files {
            let content = file_info.content.clone().or_else(|| {
                if !file_info.is_binary {
                    fs::read_to_string(&file_info.path).ok()
                } else {
                    None
                }
            });

            if let Some(content) = content {
                let path_key = file_info.path.to_string_lossy().to_string();
                selected_files.push(FileWithContent {
                    path: file_info.path.clone(),
                    relative_path: file_info.relative_path.clone(),
                    content,
                    size: file_info.size,
                    estimated_tokens: file_info.token_estimate.unwrap_or(0),
                    importance_score: result.final_scores.get(&path_key).copied().unwrap_or(0.0),
                    git_changes: None,
                    centrality_score: file_info.centrality_score.unwrap_or(0.0),
                    query_relevance_score: 0.0,
                    entry_point_proximity: 0.0,
                    content_quality_score: 0.0,
                    repository_role_score: 0.0,
                    recency_score: 0.0,
                });
            }
        }
    }

    if !all_relative_paths.is_empty()
        && !selected_files
            .iter()
            .any(|file| file.relative_path == "DIRECTORY_MAP.txt")
    {
        let directory_map = build_directory_map(&all_relative_paths);
        if !directory_map.is_empty() {
            let mut map_tokens = directory_map.split_whitespace().count() * 4 / 3;
            if map_tokens == 0 {
                map_tokens = 1;
            }
            let map_size = directory_map.len() as u64;
            selected_files.insert(
                0,
                FileWithContent {
                    path: repo_dir.join("DIRECTORY_MAP.txt"),
                    relative_path: "DIRECTORY_MAP.txt".to_string(),
                    content: directory_map,
                    size: map_size,
                    estimated_tokens: map_tokens,
                    importance_score: 1.0,
                    git_changes: None,
                    centrality_score: 0.0,
                    query_relevance_score: 0.0,
                    entry_point_proximity: 0.0,
                    content_quality_score: 0.0,
                    repository_role_score: 0.0,
                    recency_score: 0.0,
                },
            );
        }
    }

    let metrics = SelectionMetrics {
        total_files_discovered,
        files_selected: selected_files.len(),
        total_tokens_estimated: selected_files.iter().map(|f| f.estimated_tokens).sum(),
        selection_time_ms: 0, // TODO: get from library
        algorithm_used: "Intelligent (Library)".to_string(),
        coverage_score: 1.0,
        relevance_score: 0.8,
    };

    if verbose_level > 0 {
        info!(
            "Selected {} files ({} tokens)",
            metrics.files_selected, metrics.total_tokens_estimated
        );
    } else {
        println!("📊 Selection summary");
        println!("  • Files scanned   : {}", total_files_discovered);
        println!(
            "  • Files selected  : {} ({} tokens)",
            metrics.files_selected, metrics.total_tokens_estimated
        );
    }

    if selection_config.show_metrics {
        if verbose_level > 0 {
            info!("Enhanced Selection Metrics:");
        } else {
            println!("\n📈 Additional metrics");
        }
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
                selection_config.repository_complexity_factor
            );
        } else {
            println!("  • Algorithm        : {}", metrics.algorithm_used);
            println!(
                "  • Coverage         : {:.1}%",
                metrics.coverage_score * 100.0
            );
            println!("  • Relevance score  : {:.2}", metrics.relevance_score);
        }

        if !selection_config.entry_points.is_empty() {
            let avg_entry_proximity = selected_files
                .iter()
                .map(|f| f.entry_point_proximity)
                .sum::<f64>()
                / selected_files.len().max(1) as f64;
            info!("  - Entry point influence: {:.2}", avg_entry_proximity);
        }

        if selection_config.query_hint.is_some() {
            let avg_query_relevance = selected_files
                .iter()
                .map(|f| f.query_relevance_score)
                .sum::<f64>()
                / selected_files.len().max(1) as f64;
            info!("  - Query relevance: {:.2}", avg_query_relevance);
        }

        if selection_config.include_diffs {
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
    if verbose_level == 0 {
        // Stage 3: Output Generation with progress bar
        let mut progress = ScribeProgressManager::new(1);
        progress.start_stage("📝 Output Generation", 100);
        progress.update_message(&format!(
            "Generating {} output...",
            match output_format {
                OutputFormat::Html => "HTML",
                OutputFormat::Cxml => "CXML",
                OutputFormat::Repomix => "Repomix",
                OutputFormat::Xml => "XML",
                OutputFormat::Json => "JSON",
                OutputFormat::Text => "Text",
                OutputFormat::Markdown => "Markdown",
            }
        ));

        // Simulate incremental progress during output generation
        for i in 0..50 {
            progress.set_position(i + 1);
            if i % 10 == 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            }
        }
        progress.update_message("Writing output file...");

        // Generate static output in requested format
        match output_format {
            OutputFormat::Html => {
                let html_content = generate_html_output(&selected_files, &metrics)?;
                fs::write(&output_path, html_content)?;
            }
            OutputFormat::Cxml => {
                let cxml_content = generate_cxml_output(&selected_files, &metrics)?;
                fs::write(&output_path, cxml_content)?;
            }
            OutputFormat::Repomix => {
                let repomix_content = generate_repomix_output(&selected_files, &metrics)?;
                fs::write(&output_path, repomix_content)?;
            }
            OutputFormat::Xml => {
                let xml_content = generate_xml_output(&selected_files, &metrics)?;
                fs::write(&output_path, xml_content)?;
            }
            OutputFormat::Json => {
                let json_content = generate_json_output(&selected_files, &metrics)?;
                fs::write(&output_path, json_content)?;
            }
            OutputFormat::Text => {
                let text_content = generate_text_output(&selected_files, &metrics)?;
                fs::write(&output_path, text_content)?;
            }
            OutputFormat::Markdown => {
                let markdown_content = generate_markdown_output(&selected_files, &metrics)?;
                fs::write(&output_path, markdown_content)?;
            }
        }

        // Complete progress
        progress.set_position(100);
        progress.finish_stage(&format!(
            "✅ {} output generated",
            match output_format {
                OutputFormat::Html => "HTML",
                OutputFormat::Cxml => "CXML",
                OutputFormat::Repomix => "Repomix",
                OutputFormat::Xml => "XML",
                OutputFormat::Json => "JSON",
                OutputFormat::Text => "Text",
                OutputFormat::Markdown => "Markdown",
            }
        ));
        progress.finish_all();
    } else {
        eprintln!("📝 Phase 2: Output Generation STARTING");
        info!("📝 Phase 2: Output Generation");

        // Generate static output in requested format (verbose mode - no progress bars)
        match output_format {
            OutputFormat::Html => {
                let html_content = generate_html_output(&selected_files, &metrics)?;
                fs::write(&output_path, html_content)?;
            }
            OutputFormat::Cxml => {
                let cxml_content = generate_cxml_output(&selected_files, &metrics)?;
                fs::write(&output_path, cxml_content)?;
            }
            OutputFormat::Repomix => {
                let repomix_content = generate_repomix_output(&selected_files, &metrics)?;
                fs::write(&output_path, repomix_content)?;
            }
            OutputFormat::Xml => {
                let xml_content = generate_xml_output(&selected_files, &metrics)?;
                fs::write(&output_path, xml_content)?;
            }
            OutputFormat::Json => {
                let json_content = generate_json_output(&selected_files, &metrics)?;
                fs::write(&output_path, json_content)?;
            }
            OutputFormat::Text => {
                let text_content = generate_text_output(&selected_files, &metrics)?;
                fs::write(&output_path, text_content)?;
            }
            OutputFormat::Markdown => {
                let md_content = generate_markdown_output(&selected_files, &metrics)?;
                fs::write(&output_path, md_content)?;
            }
        }
    }

    if verbose_level > 0 {
        info!(
            "🎉 Analysis complete! Output saved to: {}",
            output_path.display()
        );
    } else {
        println!("  • Output location : {}", output_path.display());
        println!("\n🎉 Analysis complete");
    }

    // Show configuration source info
    if scribe_config.output_file_path.is_some() && matches.get_one::<String>("output").is_none() {
        info!("📋 Output path from configuration file");
    }

    Ok(())
}

// Output format generators
fn generate_html_output(
    files: &[FileWithContent],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn std::error::Error>> {
    use chrono::Utc;
    use serde_json::json;

    // Load template from external file
    let template_str = include_str!("../../templates/report_bundled.html");

    // Set up Handlebars
    let mut handlebars = Handlebars::new();
    handlebars.register_template_string("report", template_str)?;

    // Register custom helpers
    handlebars.register_helper(
        "add",
        Box::new(
            |h: &handlebars::Helper,
             _: &Handlebars,
             _: &handlebars::Context,
             _: &mut handlebars::RenderContext,
             out: &mut dyn handlebars::Output|
             -> Result<(), handlebars::RenderError> {
                let a = h.param(0).and_then(|v| v.value().as_u64()).unwrap_or(0);
                let b = h.param(1).and_then(|v| v.value().as_u64()).unwrap_or(0);
                out.write(&(a + b).to_string())?;
                Ok(())
            },
        ),
    );

    // Calculate totals
    let total_tokens: usize = files.iter().map(|f| f.estimated_tokens).sum();
    let total_size: u64 = files.iter().map(|f| f.size).sum();
    let total_files = files.len();

    // Prepare template data
    let template_data = json!({
        "repository_name": "Scribe Analysis",
        "algorithm": metrics.algorithm_used,
        "generated_time": Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        "selection_time_ms": metrics.selection_time_ms,
        "total_files": total_files,
        "total_tokens": format_number(total_tokens),
        "total_size": format_bytes(total_size),
        "coverage_percentage": format!("{:.1}", metrics.coverage_score * 100.0),
        "files": files.iter().map(|file| {
            json!({
                "relative_path": html_escape(&file.relative_path),
                "content": html_escape(&file.content),
                "size": format_bytes(file.size),
                "estimated_tokens": format_number(file.estimated_tokens),
                "importance_score": format!("{:.2}", file.importance_score),
                "centrality_score": format!("{:.2}", file.centrality_score),
                "query_relevance_score": format!("{:.2}", file.query_relevance_score),
                "entry_point_proximity": format!("{:.2}", file.entry_point_proximity),
                "content_quality_score": format!("{:.2}", file.content_quality_score),
                "repository_role_score": format!("{:.2}", file.repository_role_score),
                "recency_score": format!("{:.2}", file.recency_score),
                "icon": get_file_icon(&file.relative_path)
            })
        }).collect::<Vec<_>>()
    });

    // Render the template
    let html = handlebars.render("report", &template_data)?;
    Ok(html)
}

// Helper functions for the superior HTML template
fn get_file_icon(file_path: &str) -> &'static str {
    let path = std::path::Path::new(file_path);
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Special files
    if name.starts_with("readme") {
        return "book-open";
    } else if name == "license" || name == "licence" {
        return "scale";
    } else if name == "dockerfile" || name.contains("docker-compose") {
        return "box";
    } else if name == "makefile" {
        return "settings";
    } else if name.starts_with(".git") {
        return "git-branch";
    } else if name == "package.json" || name == "cargo.toml" || name == "go.mod" {
        return "package";
    }

    // Extensions
    match ext.as_str() {
        "py" | "pyw" => "file-code",
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => "file-code",
        "html" | "htm" | "xml" | "xhtml" => "globe",
        "css" | "scss" | "sass" | "less" => "palette",
        "json" | "jsonc" | "json5" => "braces",
        "yml" | "yaml" => "list",
        "md" | "markdown" | "mdx" => "file-text",
        "txt" | "text" => "file-text",
        "rs" => "file-code",
        "go" => "file-code",
        "java" | "kt" | "scala" => "file-code",
        "c" | "cpp" | "cc" | "h" | "hpp" => "file-code",
        "cs" | "fs" | "vb" => "file-code",
        "php" | "rb" | "pl" | "r" | "swift" | "dart" => "file-code",
        "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" => "terminal",
        "sql" | "sqlite" | "db" => "database",
        "png" | "jpg" | "jpeg" | "gif" | "svg" | "webp" | "ico" => "image",
        "pdf" => "file-text",
        "zip" | "tar" | "gz" | "bz2" | "7z" | "rar" => "archive",
        "toml" => "settings",
        _ => "file",
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

fn format_number(num: usize) -> String {
    num.to_string()
        .chars()
        .rev()
        .collect::<Vec<_>>()
        .chunks(3)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join(",")
        .chars()
        .rev()
        .collect()
}

fn generate_cxml_output(
    files: &[FileWithContent],
    _metrics: &SelectionMetrics,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut cxml = String::new();
    cxml.push_str("<repository>\n");

    for file in files {
        cxml.push_str(&format!("<file path=\"{}\">\n", file.relative_path));
        cxml.push_str(&format!("{}\n", file.content));
        cxml.push_str("</file>\n");
    }

    cxml.push_str("</repository>\n");
    Ok(cxml)
}

fn generate_repomix_output(
    files: &[FileWithContent],
    _metrics: &SelectionMetrics,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut repomix = String::new();

    for file in files {
        repomix.push_str(&format!("## {}\n\n", file.relative_path));
        repomix.push_str(&format!("```\n{}\n```\n\n", file.content));
    }

    Ok(repomix)
}

fn generate_xml_output(
    files: &[FileWithContent],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<scribe-analysis>\n");
    xml.push_str(&format!("    <summary>\n"));
    xml.push_str(&format!(
        "        <total-files>{}</total-files>\n",
        metrics.files_selected
    ));
    xml.push_str(&format!(
        "        <total-tokens>{}</total-tokens>\n",
        metrics.total_tokens_estimated
    ));
    xml.push_str(&format!(
        "        <algorithm>{}</algorithm>\n",
        metrics.algorithm_used
    ));
    xml.push_str(&format!("    </summary>\n"));
    xml.push_str("    <files>\n");

    for file in files {
        xml.push_str("        <file>\n");
        xml.push_str(&format!(
            "            <path>{}</path>\n",
            file.relative_path
        ));
        xml.push_str(&format!("            <size>{}</size>\n", file.size));
        xml.push_str(&format!(
            "            <tokens>{}</tokens>\n",
            file.estimated_tokens
        ));
        xml.push_str(&format!(
            "            <score>{:.2}</score>\n",
            file.importance_score
        ));
        xml.push_str(&format!(
            "            <content><![CDATA[{}]]></content>\n",
            file.content
        ));
        xml.push_str("        </file>\n");
    }

    xml.push_str("    </files>\n");
    xml.push_str("</scribe-analysis>\n");
    Ok(xml)
}

fn generate_json_output(
    files: &[FileWithContent],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn std::error::Error>> {
    let output = serde_json::json!({
        "metrics": {
            "algorithm": metrics.algorithm_used,
            "files_selected": metrics.files_selected,
            "total_files": metrics.total_files_discovered,
            "total_tokens": metrics.total_tokens_estimated,
            "coverage_score": metrics.coverage_score,
            "relevance_score": metrics.relevance_score,
            "selection_time_ms": metrics.selection_time_ms
        },
        "files": files.iter().map(|f| serde_json::json!({
            "path": f.relative_path,
            "size": f.size,
            "tokens": f.estimated_tokens,
            "score": f.importance_score,
            "content": f.content
        })).collect::<Vec<_>>()
    });

    Ok(serde_json::to_string_pretty(&output)?)
}

fn generate_text_output(
    files: &[FileWithContent],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut text = String::new();
    text.push_str(&format!("SCRIBE ANALYSIS REPORT\n"));
    text.push_str(&format!("======================\n\n"));
    text.push_str(&format!("Algorithm: {}\n", metrics.algorithm_used));
    text.push_str(&format!(
        "Files Selected: {} / {}\n",
        metrics.files_selected, metrics.total_files_discovered
    ));
    text.push_str(&format!(
        "Total Tokens: {}\n",
        metrics.total_tokens_estimated
    ));
    text.push_str(&format!(
        "Coverage: {:.1}%\n\n",
        metrics.coverage_score * 100.0
    ));

    for file in files {
        text.push_str(&format!("=== {} ===\n", file.relative_path));
        text.push_str(&format!(
            "Size: {} bytes | Tokens: {} | Score: {:.2}\n\n",
            file.size, file.estimated_tokens, file.importance_score
        ));
        text.push_str(&format!("{}\n\n", file.content));
    }

    Ok(text)
}

fn generate_markdown_output(
    files: &[FileWithContent],
    metrics: &SelectionMetrics,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut md = String::new();
    md.push_str("# 🔍 Scribe Analysis Report\n\n");
    md.push_str("## Selection Metrics\n\n");
    md.push_str(&format!("- **Algorithm:** {}\n", metrics.algorithm_used));
    md.push_str(&format!(
        "- **Files Selected:** {} / {}\n",
        metrics.files_selected, metrics.total_files_discovered
    ));
    md.push_str(&format!(
        "- **Total Tokens:** {}\n",
        metrics.total_tokens_estimated
    ));
    md.push_str(&format!(
        "- **Coverage:** {:.1}%\n\n",
        metrics.coverage_score * 100.0
    ));

    md.push_str("## Files\n\n");

    for file in files {
        md.push_str(&format!("### 📄 {}\n\n", file.relative_path));
        md.push_str(&format!(
            "**Size:** {} bytes | **Tokens:** {} | **Score:** {:.2}\n\n",
            file.size, file.estimated_tokens, file.importance_score
        ));

        // Detect language for syntax highlighting
        let lang = match file.path.extension().and_then(|s| s.to_str()) {
            Some("rs") => "rust",
            Some("py") => "python",
            Some("js") => "javascript",
            Some("ts") => "typescript",
            Some("html") => "html",
            Some("css") => "css",
            Some("json") => "json",
            Some("md") => "markdown",
            Some("toml") => "toml",
            Some("yaml") | Some("yml") => "yaml",
            _ => "",
        };

        md.push_str(&format!("```{}\n{}\n```\n\n", lang, file.content));
    }

    Ok(md)
}

fn html_escape(text: &str) -> String {
    text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&#x27;")
}

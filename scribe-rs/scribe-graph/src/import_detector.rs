//! Import detection and resolution engine
//!
//! This module provides multi-language import detection and resolution
//! with pre-computed lookup optimization for efficient graph construction.

use scribe_analysis::heuristics::ScanResult;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::centrality_types::ImportResolutionConfig;

// File extension constants for each supported language
const PYTHON_FILE_EXTENSIONS: &[&str] = &["py"];
const PYTHON_SUFFIXES: &[&str] = &[".py"];
const JS_FILE_EXTENSIONS: &[&str] = &["js", "jsx", "ts", "tsx", "mjs", "cjs"];
const JS_SUFFIXES: &[&str] = &[".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"];
const RUST_FILE_EXTENSIONS: &[&str] = &["rs"];
const RUST_SUFFIXES: &[&str] = &[".rs"];

/// Strip known file extension suffixes from an import string
fn strip_known_suffix<'a>(value: &'a str, suffixes: &[&str]) -> &'a str {
    for suffix in suffixes {
        if value.ends_with(suffix) {
            return &value[..value.len() - suffix.len()];
        }
    }
    value
}

/// Import detection and resolution engine with pre-computed lookup optimization
#[derive(Debug, Clone)]
pub struct ImportDetector {
    pub(crate) config: ImportResolutionConfig,
    /// Pre-computed lookup map: file stem -> full paths (massive performance improvement)
    stem_to_paths: HashMap<String, Vec<String>>,
    /// Pre-computed lookup map: filename -> full paths
    filename_to_paths: HashMap<String, Vec<String>>,
    /// Set of all available file paths for quick existence checks
    available_paths: HashSet<String>,
}

impl ImportDetector {
    /// Create with configuration
    pub fn with_config(config: ImportResolutionConfig) -> Self {
        Self {
            config,
            stem_to_paths: HashMap::new(),
            filename_to_paths: HashMap::new(),
            available_paths: HashSet::new(),
        }
    }

    /// Create with pre-computed lookup maps for massive performance improvement
    pub fn with_file_index<T>(config: ImportResolutionConfig, scan_results: &[T]) -> Self
    where
        T: ScanResult,
    {
        let mut detector = Self::with_config(config);
        detector.build_lookup_maps(scan_results);
        detector
    }

    /// Build inverted index mapping file stems/names to full paths
    /// This eliminates the O(n) scan-all-files bottleneck
    fn build_lookup_maps<T>(&mut self, scan_results: &[T])
    where
        T: ScanResult,
    {
        self.stem_to_paths.clear();
        self.filename_to_paths.clear();
        self.available_paths.clear();

        for result in scan_results {
            let full_path = result.path().to_string();
            self.available_paths.insert(full_path.clone());

            let path = Path::new(result.path());

            // Index by file stem (name without extension)
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let stem_lower = stem.to_lowercase();
                self.stem_to_paths
                    .entry(stem_lower)
                    .or_insert_with(Vec::new)
                    .push(full_path.clone());
            }

            // Index by full filename
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                let filename_lower = filename.to_lowercase();
                self.filename_to_paths
                    .entry(filename_lower)
                    .or_insert_with(Vec::new)
                    .push(full_path);
            }
        }
    }

    /// Detect programming language from file extension
    pub fn detect_language(&self, file_path: &str) -> Option<String> {
        let path = Path::new(file_path);
        let ext = path.extension()?.to_str()?.to_lowercase();

        match ext.as_str() {
            "py" => Some("python".to_string()),
            "js" | "jsx" | "mjs" => Some("javascript".to_string()),
            "ts" | "tsx" => Some("typescript".to_string()),
            "rs" => Some("rust".to_string()),
            "go" => Some("go".to_string()),
            "java" | "kt" => Some("java".to_string()),
            "cpp" | "cc" | "cxx" | "hpp" | "h" => Some("cpp".to_string()),
            "c" => Some("c".to_string()),
            "rb" => Some("ruby".to_string()),
            "php" => Some("php".to_string()),
            "cs" => Some("csharp".to_string()),
            "swift" => Some("swift".to_string()),
            _ => None,
        }
    }

    /// Resolve import string to actual file path
    pub fn resolve_import<T>(
        &self,
        import_str: &str,
        current_file: &str,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        // Check custom path mappings first
        if let Some(mapped_path) = self.config.path_mappings.get(import_str) {
            if file_map.contains_key(mapped_path.as_str()) {
                return Some(mapped_path.clone());
            }
        }

        let current_path = Path::new(current_file);
        let language = self.detect_language(current_file);

        match language.as_deref() {
            Some("python") => self.resolve_python_import(import_str, current_path, file_map),
            Some("javascript") | Some("typescript") => {
                self.resolve_js_import(import_str, current_path, file_map)
            }
            Some("rust") => self.resolve_rust_import(import_str, current_path, file_map),
            Some("go") => self.resolve_go_import(import_str, current_path, file_map),
            _ => self.resolve_generic_import(import_str, current_path, file_map),
        }
    }

    /// Resolve Python import
    fn resolve_python_import<T>(
        &self,
        import_str: &str,
        current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        if cleaned_import.is_empty() {
            return None;
        }

        if self.config.exclude_stdlib_imports && self.is_python_stdlib(cleaned_import) {
            return None;
        }

        let mut module = cleaned_import;
        if let Some(alias_index) = module.find(" as ") {
            module = &module[..alias_index];
        }

        let mut base_dir = current_path.parent().unwrap_or(current_path).to_path_buf();
        let mut relative_levels = 0;
        while module.starts_with('.') {
            relative_levels += 1;
            module = &module[1..];
        }

        for _ in 0..relative_levels {
            if let Some(parent) = base_dir.parent() {
                base_dir = parent.to_path_buf();
            }
        }

        module = module.trim();
        let module = strip_known_suffix(module, PYTHON_SUFFIXES);
        let module_parts: Vec<&str> = if module.is_empty() {
            Vec::new()
        } else {
            module.split('.').filter(|part| !part.is_empty()).collect()
        };

        if !module_parts.is_empty() {
            if let Some(resolved) = self.resolve_relative_python(&base_dir, &module_parts, file_map)
            {
                return Some(resolved);
            }
        }

        if module_parts.is_empty() {
            return None;
        }

        self.find_module_candidate(&module_parts, PYTHON_FILE_EXTENSIONS)
    }

    /// Resolve JavaScript/TypeScript import
    fn resolve_js_import<T>(
        &self,
        import_str: &str,
        current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        if cleaned_import.is_empty() {
            return None;
        }

        let parent_dir = current_path.parent().unwrap_or(current_path);

        if cleaned_import.starts_with("./") || cleaned_import.starts_with("../") {
            if !self.config.resolve_relative_imports {
                return None;
            }

            if let Some(resolved) = self.resolve_relative_js(parent_dir, cleaned_import, file_map) {
                return Some(resolved);
            }
        } else {
            // Attempt to resolve within the same directory first
            if let Some(resolved) = self.resolve_relative_js(parent_dir, cleaned_import, file_map) {
                return Some(resolved);
            }

            if !self.config.resolve_absolute_imports {
                return None;
            }

            let normalized = strip_known_suffix(cleaned_import, JS_SUFFIXES);
            let module_parts: Vec<&str> = normalized
                .split('/')
                .filter(|segment| !segment.is_empty())
                .collect();

            if module_parts.is_empty() {
                return None;
            }

            return self.find_module_candidate(&module_parts, JS_FILE_EXTENSIONS);
        }

        None
    }

    /// Resolve Rust import (use/mod statements)
    fn resolve_rust_import<T>(
        &self,
        import_str: &str,
        current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        if cleaned_import.is_empty() {
            return None;
        }

        if self.config.exclude_stdlib_imports && self.is_rust_stdlib(cleaned_import) {
            return None;
        }

        let mut module = cleaned_import;

        if let Some(stripped) = module.strip_prefix("crate::") {
            module = stripped;
        }

        while let Some(stripped) = module.strip_prefix("self::") {
            module = stripped;
        }

        let mut base_dir = current_path.parent().unwrap_or(current_path).to_path_buf();
        while let Some(stripped) = module.strip_prefix("super::") {
            module = stripped;
            if let Some(parent) = base_dir.parent() {
                base_dir = parent.to_path_buf();
            }
        }

        let module = strip_known_suffix(module, RUST_SUFFIXES);
        let module_parts: Vec<&str> = module
            .split("::")
            .filter(|segment| !segment.is_empty())
            .collect();

        if module_parts.is_empty() {
            return None;
        }

        if let Some(resolved) = self.resolve_relative_rust(&base_dir, &module_parts, file_map) {
            return Some(resolved);
        }

        if module_parts.len() == 1 {
            let crate_lib = base_dir.join("lib.rs");
            if let Some(candidate_str) = crate_lib.to_str() {
                if file_map.contains_key(candidate_str) {
                    return Some(candidate_str.to_string());
                }
            }
        }

        self.find_module_candidate(&module_parts, RUST_FILE_EXTENSIONS)
    }

    /// Resolve Go import
    fn resolve_go_import<T>(
        &self,
        import_str: &str,
        _current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim().trim_matches('"');

        // Skip standard library
        if self.config.exclude_stdlib_imports && !cleaned_import.contains('.') {
            return None;
        }

        let parts: Vec<&str> = cleaned_import.split('/').collect();

        // Try various Go file patterns
        let mut candidates = Vec::new();

        // Package directory
        candidates.push(format!("{}.go", parts.last()?));
        candidates.push(format!("{}/main.go", cleaned_import));
        candidates.push(format!("{}/{}.go", cleaned_import, parts.last()?));

        for candidate in &candidates {
            if file_map.contains_key(candidate.as_str()) {
                return Some(candidate.clone());
            }
        }

        self.fuzzy_match_import(&parts, file_map)
    }

    /// Generic import resolution
    fn resolve_generic_import<T>(
        &self,
        import_str: &str,
        _current_path: &Path,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let cleaned_import = import_str.trim();
        let parts: Vec<&str> = cleaned_import.split(&['/', '.', ':']).collect();
        self.fuzzy_match_import(&parts, file_map)
    }

    fn resolve_relative_python<T>(
        &self,
        base_dir: &Path,
        module_parts: &[&str],
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        if module_parts.is_empty() {
            return None;
        }

        let mut module_path = base_dir.to_path_buf();
        for part in module_parts {
            module_path.push(part);
        }

        let mut candidate = module_path.clone();
        candidate.set_extension("py");
        if let Some(candidate_str) = candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        let init_candidate = module_path.join("__init__.py");
        if let Some(candidate_str) = init_candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        None
    }

    fn resolve_relative_js<T>(
        &self,
        base_dir: &Path,
        import_path: &str,
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        let normalized = strip_known_suffix(import_path, JS_SUFFIXES);
        let target = self.build_relative_js_path(base_dir, normalized);

        for ext in JS_FILE_EXTENSIONS {
            let mut candidate = target.clone();
            candidate.set_extension(ext);
            if let Some(candidate_str) = candidate.to_str() {
                if file_map.contains_key(candidate_str) {
                    return Some(candidate_str.to_string());
                }
            }
        }

        for ext in JS_FILE_EXTENSIONS {
            let index_candidate = target.join(format!("index.{}", ext));
            if let Some(candidate_str) = index_candidate.to_str() {
                if file_map.contains_key(candidate_str) {
                    return Some(candidate_str.to_string());
                }
            }
        }

        None
    }

    fn build_relative_js_path(&self, base_dir: &Path, import_path: &str) -> PathBuf {
        let mut resolved = base_dir.to_path_buf();
        for segment in import_path.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    if let Some(parent) = resolved.parent() {
                        resolved = parent.to_path_buf();
                    }
                }
                _ => resolved.push(segment),
            }
        }
        resolved
    }

    fn resolve_relative_rust<T>(
        &self,
        base_dir: &Path,
        module_parts: &[&str],
        file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        if module_parts.is_empty() {
            return None;
        }

        let mut module_path = base_dir.to_path_buf();
        for part in module_parts {
            module_path.push(part);
        }

        let mut candidate = module_path.clone();
        candidate.set_extension("rs");
        if let Some(candidate_str) = candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        let mod_candidate = module_path.join("mod.rs");
        if let Some(candidate_str) = mod_candidate.to_str() {
            if file_map.contains_key(candidate_str) {
                return Some(candidate_str.to_string());
            }
        }

        None
    }

    fn find_module_candidate(&self, module_parts: &[&str], extensions: &[&str]) -> Option<String> {
        if module_parts.is_empty() {
            return None;
        }

        let stem = module_parts.last().unwrap().to_lowercase();
        let candidates = self.stem_to_paths.get(&stem)?;

        for candidate in candidates {
            if self.module_path_matches(candidate, module_parts, extensions) {
                return Some(candidate.clone());
            }
        }

        None
    }

    fn module_path_matches(
        &self,
        candidate: &str,
        module_parts: &[&str],
        extensions: &[&str],
    ) -> bool {
        let path = Path::new(candidate);
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => return false,
        };

        let lower_file = file_name.to_lowercase();
        if lower_file == "__init__.py" {
            return self.dir_path_matches(path.parent(), module_parts);
        }

        let ext = Path::new(file_name)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        if !extensions
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(&ext))
        {
            return false;
        }

        let stem = Path::new(file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        if stem == "index" && !module_parts.is_empty() {
            return self.dir_path_matches(path.parent(), module_parts);
        }

        if module_parts.is_empty() {
            return false;
        }

        if stem != module_parts.last().unwrap().to_lowercase() {
            return false;
        }

        self.dir_path_matches(
            path.parent(),
            &module_parts[..module_parts.len().saturating_sub(1)],
        )
    }

    fn dir_path_matches(&self, dir: Option<&Path>, module_parts: &[&str]) -> bool {
        if module_parts.is_empty() {
            return true;
        }

        let mut current = dir;
        for expected in module_parts.iter().rev() {
            match current {
                Some(path) => {
                    let name = path.file_name().and_then(|n| n.to_str());
                    match name {
                        Some(name) if name.eq_ignore_ascii_case(expected) => {
                            current = path.parent();
                        }
                        _ => return false,
                    }
                }
                None => return false,
            }
        }

        true
    }

    /// Fuzzy matching for import resolution - OPTIMIZED with pre-computed maps
    fn fuzzy_match_import<T>(
        &self,
        import_parts: &[&str],
        _file_map: &HashMap<&str, &T>,
    ) -> Option<String>
    where
        T: ScanResult,
    {
        if import_parts.is_empty() {
            return None;
        }

        let last_part = import_parts.last()?.to_lowercase();

        // MASSIVE PERFORMANCE IMPROVEMENT: Use pre-computed lookup maps instead of O(n) scan
        // 1. First try exact stem match (most common case)
        if let Some(paths) = self.stem_to_paths.get(&last_part) {
            // Return first match (could be made smarter with scoring)
            if let Some(first_path) = paths.first() {
                return Some(first_path.clone());
            }
        }

        // 2. Try filename match
        if let Some(paths) = self.filename_to_paths.get(&last_part) {
            if let Some(first_path) = paths.first() {
                return Some(first_path.clone());
            }
        }

        // 3. Try partial matching against stems
        for (stem, paths) in &self.stem_to_paths {
            if stem.contains(&last_part) || last_part.contains(stem) {
                if let Some(first_path) = paths.first() {
                    return Some(first_path.clone());
                }
            }
        }

        // 4. Fallback: check if path contains all import parts
        for path in &self.available_paths {
            let path_lower = path.to_lowercase();
            if import_parts
                .iter()
                .all(|&part| path_lower.contains(&part.to_lowercase()))
            {
                return Some(path.clone());
            }
        }

        None
    }

    /// Check if import is Python standard library
    pub fn is_python_stdlib(&self, import_str: &str) -> bool {
        let stdlib_modules = [
            "os",
            "sys",
            "re",
            "json",
            "collections",
            "itertools",
            "functools",
            "typing",
            "datetime",
            "math",
            "random",
            "string",
            "pathlib",
            "io",
            "csv",
            "xml",
            "html",
            "urllib",
            "http",
            "email",
            "logging",
            "unittest",
            "asyncio",
            "concurrent",
            "multiprocessing",
            "threading",
            "subprocess",
        ];

        let first_part = import_str.split('.').next().unwrap_or(import_str);
        stdlib_modules.contains(&first_part)
    }

    /// Check if import is Rust standard library
    pub fn is_rust_stdlib(&self, import_str: &str) -> bool {
        import_str.starts_with("std::")
            || import_str.starts_with("core::")
            || import_str.starts_with("alloc::")
    }
}

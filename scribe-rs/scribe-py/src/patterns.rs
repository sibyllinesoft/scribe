//! Pattern matching Python interface
//!
//! Provides Python bindings for the Scribe pattern matching system,
//! enabling advanced code pattern detection and analysis in Python applications.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use scribe_core::{Config, FileInfo, Language, Position, Range};
use scribe_patterns::{MatchResult, MatcherOptions, PatternMatcher as ScribePatternMatcher};

use crate::error::{rust_error_to_py_err, rust_result_to_py};
use crate::utils::*;

/// Python wrapper for pattern matching engine
#[pyclass]
pub struct PatternMatcher {
    /// Configuration
    config: Arc<RwLock<Config>>,
    /// Language-specific matchers cache
    matchers: Arc<RwLock<HashMap<Language, ScribePatternMatcher>>>,
}

#[pymethods]
impl PatternMatcher {
    /// Create a new pattern matcher
    ///
    /// Args:
    ///     config: Optional configuration dict
    ///
    /// Returns:
    ///     PatternMatcher instance
    #[new]
    pub fn new(config: Option<&PyDict>) -> PyResult<Self> {
        // Load configuration
        let matcher_config = Config::default();

        Ok(PatternMatcher {
            config: Arc::new(RwLock::new(matcher_config)),
            matchers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Load pattern rules from file
    ///
    /// Args:
    ///     rules_path: Path to pattern rules file (JSON, YAML, or TOML)
    ///
    /// Returns:
    ///     Number of rules loaded
    pub fn load_rules_from_file<'py>(
        &self,
        py: Python<'py>,
        rules_path: &str,
    ) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let rules = self.rules.clone();
        let path = PathBuf::from(rules_path);

        future_into_py(py, async move {
            let mut engine_guard = engine.write().await;

            let loaded_rules = engine_guard
                .load_rules_from_file(&path)
                .await
                .map_err(rust_error_to_py_err)?;

            let rule_count = loaded_rules.len();

            // Cache the rules
            {
                let mut rules_guard = rules.write().await;
                *rules_guard = loaded_rules;
            }

            Ok(rule_count)
        })
    }

    /// Load pattern rules from JSON string
    ///
    /// Args:
    ///     rules_json: JSON string containing pattern rules
    ///
    /// Returns:
    ///     Number of rules loaded
    pub fn load_rules_from_json<'py>(
        &self,
        py: Python<'py>,
        rules_json: &str,
    ) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let rules = self.rules.clone();
        let json_str = rules_json.to_string();

        future_into_py(py, async move {
            let mut engine_guard = engine.write().await;

            let loaded_rules = engine_guard
                .load_rules_from_json(&json_str)
                .await
                .map_err(rust_error_to_py_err)?;

            let rule_count = loaded_rules.len();

            // Cache the rules
            {
                let mut rules_guard = rules.write().await;
                *rules_guard = loaded_rules;
            }

            Ok(rule_count)
        })
    }

    /// Add a custom pattern rule
    ///
    /// Args:
    ///     rule_dict: Dictionary containing pattern rule definition
    ///               Required keys: name, pattern, language
    ///               Optional keys: description, category, severity, examples
    ///
    /// Returns:
    ///     Success status
    pub fn add_rule(&self, rule_dict: &PyDict) -> PyResult<()> {
        // Extract rule components
        let name: String = rule_dict
            .get_item("name")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'name' key"))?
            .unwrap()
            .extract()?;

        let pattern: String = rule_dict
            .get_item("pattern")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'pattern' key"))?
            .unwrap()
            .extract()?;

        let language_str: String = rule_dict
            .get_item("language")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'language' key"))?
            .unwrap()
            .extract()?;

        let language = py_to_language(&pyo3::types::PyString::new(
            pyo3::Python::with_gil(|py| py),
            &language_str,
        ))?;

        let description = rule_dict
            .get_item("description")?
            .map(|v| v.extract::<String>())
            .transpose()?
            .unwrap_or_else(|| format!("Pattern rule: {}", name));

        let category = rule_dict
            .get_item("category")?
            .map(|v| v.extract::<String>())
            .transpose()?
            .unwrap_or_else(|| "custom".to_string());

        let severity = rule_dict
            .get_item("severity")?
            .map(|v| v.extract::<String>())
            .transpose()?
            .unwrap_or_else(|| "medium".to_string());

        // Create pattern rule
        let mut rule = PatternRule::new(name, pattern, language);
        rule.description = Some(description);
        rule.category = Some(category);
        rule.severity = Some(severity);

        // Add examples if provided
        if let Some(examples_obj) = rule_dict.get_item("examples")? {
            let examples: Vec<String> = examples_obj.extract()?;
            rule.examples = examples;
        }

        // Add rule to engine
        futures::executor::block_on(async {
            let mut engine_guard = self.engine.write().await;
            engine_guard
                .add_rule(rule.clone())
                .await
                .map_err(rust_error_to_py_err)?;

            // Cache the rule
            let mut rules_guard = self.rules.write().await;
            rules_guard.push(rule);

            Ok(())
        })
    }

    /// Find matches for a specific file
    ///
    /// Args:
    ///     file_path: Path to the file to analyze
    ///     file_content: Optional file content (if not provided, will read from disk)
    ///     rule_filter: Optional list of rule names to apply (if None, apply all rules)
    ///
    /// Returns:
    ///     List of match result dictionaries
    pub fn find_matches<'py>(
        &self,
        py: Python<'py>,
        file_path: &str,
        file_content: Option<&str>,
        rule_filter: Option<Vec<String>>,
    ) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let matchers = self.matchers.clone();
        let path = PathBuf::from(file_path);
        let content = file_content.map(|s| s.to_string());
        let filter = rule_filter.unwrap_or_default();

        future_into_py(py, async move {
            let engine_guard = engine.read().await;

            // Get file content if not provided
            let file_content = if let Some(content) = content {
                content
            } else {
                std::fs::read_to_string(&path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to read file {}: {}",
                        path.display(),
                        e
                    ))
                })?
            };

            // Create FileInfo for analysis
            let file_info = FileInfo::from_path_and_content(&path, &file_content)
                .map_err(rust_error_to_py_err)?;

            // Get or create language-specific matcher
            let language_matcher = {
                let mut matchers_guard = matchers.write().await;

                if !matchers_guard.contains_key(&file_info.language) {
                    let matcher =
                        scribe_patterns::PatternMatcher::for_language(file_info.language.clone())
                            .map_err(rust_error_to_py_err)?;
                    matchers_guard.insert(file_info.language.clone(), matcher);
                }

                matchers_guard.get(&file_info.language).unwrap().clone()
            };

            // Find matches
            let matches = if filter.is_empty() {
                engine_guard
                    .find_all_matches(&file_info, &language_matcher)
                    .await
                    .map_err(rust_error_to_py_err)?
            } else {
                engine_guard
                    .find_matches_with_filter(&file_info, &language_matcher, &filter)
                    .await
                    .map_err(rust_error_to_py_err)?
            };

            // Convert matches to Python format
            Python::with_gil(|py| {
                let result_list = PyList::empty(py);

                for match_result in matches {
                    let match_dict = PyDict::new(py);
                    match_dict.set_item("rule_name", &match_result.rule_name)?;
                    match_dict.set_item("pattern", &match_result.pattern)?;
                    match_dict.set_item("matched_text", &match_result.matched_text)?;
                    match_dict.set_item(
                        "start_position",
                        position_to_py(py, &match_result.start_position)?,
                    )?;
                    match_dict.set_item(
                        "end_position",
                        position_to_py(py, &match_result.end_position)?,
                    )?;
                    match_dict.set_item("confidence", match_result.confidence)?;

                    if let Some(ref category) = match_result.category {
                        match_dict.set_item("category", category)?;
                    }

                    if let Some(ref severity) = match_result.severity {
                        match_dict.set_item("severity", severity)?;
                    }

                    if let Some(ref description) = match_result.description {
                        match_dict.set_item("description", description)?;
                    }

                    // Add captured groups if any
                    if !match_result.captured_groups.is_empty() {
                        let groups_dict = PyDict::new(py);
                        for (name, value) in &match_result.captured_groups {
                            groups_dict.set_item(name, value)?;
                        }
                        match_dict.set_item("captured_groups", groups_dict)?;
                    }

                    result_list.append(match_dict)?;
                }

                Ok(result_list.into())
            })
        })
    }

    /// Find matches across multiple files
    ///
    /// Args:
    ///     files: List of file paths to analyze
    ///     rule_filter: Optional list of rule names to apply
    ///     progress_callback: Optional progress callback function
    ///     batch_size: Number of files to process in each batch
    ///
    /// Returns:
    ///     Dictionary mapping file paths to lists of match results
    pub fn find_matches_batch<'py>(
        &self,
        py: Python<'py>,
        files: &PyList,
        rule_filter: Option<Vec<String>>,
        progress_callback: Option<PyObject>,
        batch_size: Option<usize>,
    ) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let matchers = self.matchers.clone();
        let filter = rule_filter.unwrap_or_default();
        let batch_size = batch_size.unwrap_or(50);

        // Convert Python list to file paths
        let mut file_paths = Vec::new();
        for item in files {
            let path_str: String = item.extract()?;
            file_paths.push(PathBuf::from(path_str));
        }

        future_into_py(py, async move {
            let engine_guard = engine.read().await;
            let total_files = file_paths.len();
            let mut results = HashMap::new();

            // Process files in batches
            for (batch_idx, batch) in file_paths.chunks(batch_size).enumerate() {
                for path in batch {
                    let file_content = match std::fs::read_to_string(path) {
                        Ok(content) => content,
                        Err(_) => continue, // Skip files we can't read
                    };

                    let file_info = match FileInfo::from_path_and_content(path, &file_content) {
                        Ok(info) => info,
                        Err(_) => continue, // Skip files we can't analyze
                    };

                    // Get or create language-specific matcher
                    let language_matcher = {
                        let mut matchers_guard = matchers.write().await;

                        if !matchers_guard.contains_key(&file_info.language) {
                            let matcher = match scribe_patterns::PatternMatcher::for_language(
                                file_info.language.clone(),
                            ) {
                                Ok(matcher) => matcher,
                                Err(_) => continue, // Skip if we can't create matcher for this language
                            };
                            matchers_guard.insert(file_info.language.clone(), matcher);
                        }

                        matchers_guard.get(&file_info.language).unwrap().clone()
                    };

                    // Find matches for this file
                    let matches = if filter.is_empty() {
                        match engine_guard
                            .find_all_matches(&file_info, &language_matcher)
                            .await
                        {
                            Ok(matches) => matches,
                            Err(_) => continue, // Skip files with match errors
                        }
                    } else {
                        match engine_guard
                            .find_matches_with_filter(&file_info, &language_matcher, &filter)
                            .await
                        {
                            Ok(matches) => matches,
                            Err(_) => continue, // Skip files with match errors
                        }
                    };

                    let path_str = path.to_string_lossy().to_string();
                    results.insert(path_str, matches);
                }

                // Call progress callback
                if let Some(ref callback) = progress_callback {
                    let current = (batch_idx + 1) * batch_size;
                    let current = if current > total_files {
                        total_files
                    } else {
                        current
                    };

                    Python::with_gil(|py| {
                        let _ = callback.call1(py, (current, total_files));
                    });
                }
            }

            // Convert results to Python format
            Python::with_gil(|py| {
                let result_dict = PyDict::new(py);

                for (path, matches) in results {
                    let matches_list = PyList::empty(py);

                    for match_result in matches {
                        let match_dict = PyDict::new(py);
                        match_dict.set_item("rule_name", &match_result.rule_name)?;
                        match_dict.set_item("pattern", &match_result.pattern)?;
                        match_dict.set_item("matched_text", &match_result.matched_text)?;
                        match_dict.set_item(
                            "start_position",
                            position_to_py(py, &match_result.start_position)?,
                        )?;
                        match_dict.set_item(
                            "end_position",
                            position_to_py(py, &match_result.end_position)?,
                        )?;
                        match_dict.set_item("confidence", match_result.confidence)?;

                        if let Some(ref category) = match_result.category {
                            match_dict.set_item("category", category)?;
                        }

                        if let Some(ref severity) = match_result.severity {
                            match_dict.set_item("severity", severity)?;
                        }

                        if let Some(ref description) = match_result.description {
                            match_dict.set_item("description", description)?;
                        }

                        if !match_result.captured_groups.is_empty() {
                            let groups_dict = PyDict::new(py);
                            for (name, value) in &match_result.captured_groups {
                                groups_dict.set_item(name, value)?;
                            }
                            match_dict.set_item("captured_groups", groups_dict)?;
                        }

                        matches_list.append(match_dict)?;
                    }

                    result_dict.set_item(path, matches_list)?;
                }

                Ok(result_dict.into())
            })
        })
    }

    /// Get available pattern rules
    ///
    /// Args:
    ///     language_filter: Optional language to filter rules by
    ///     category_filter: Optional category to filter rules by
    ///
    /// Returns:
    ///     List of rule dictionaries
    pub fn get_rules(
        &self,
        language_filter: Option<&str>,
        category_filter: Option<&str>,
    ) -> PyResult<PyObject> {
        let rules_guard = futures::executor::block_on(self.rules.read());

        Python::with_gil(|py| {
            let result_list = PyList::empty(py);

            let lang_filter = language_filter
                .map(|s| py_to_language(&pyo3::types::PyString::new(py, s)))
                .transpose()?;

            for rule in rules_guard.iter() {
                // Apply language filter
                if let Some(ref filter_lang) = lang_filter {
                    if rule.language != *filter_lang {
                        continue;
                    }
                }

                // Apply category filter
                if let Some(filter_cat) = category_filter {
                    if let Some(ref rule_cat) = rule.category {
                        if rule_cat != filter_cat {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                let rule_dict = PyDict::new(py);
                rule_dict.set_item("name", &rule.name)?;
                rule_dict.set_item("pattern", &rule.pattern)?;
                rule_dict.set_item("language", language_to_py(py, &rule.language)?)?;

                if let Some(ref description) = rule.description {
                    rule_dict.set_item("description", description)?;
                }

                if let Some(ref category) = rule.category {
                    rule_dict.set_item("category", category)?;
                }

                if let Some(ref severity) = rule.severity {
                    rule_dict.set_item("severity", severity)?;
                }

                if !rule.examples.is_empty() {
                    let examples_list = PyList::empty(py);
                    for example in &rule.examples {
                        examples_list.append(example)?;
                    }
                    rule_dict.set_item("examples", examples_list)?;
                }

                result_list.append(rule_dict)?;
            }

            Ok(result_list.into())
        })
    }

    /// Remove a pattern rule
    ///
    /// Args:
    ///     rule_name: Name of the rule to remove
    ///
    /// Returns:
    ///     Success status
    pub fn remove_rule<'py>(&self, py: Python<'py>, rule_name: &str) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let rules = self.rules.clone();
        let name = rule_name.to_string();

        future_into_py(py, async move {
            let mut engine_guard = engine.write().await;

            engine_guard
                .remove_rule(&name)
                .await
                .map_err(rust_error_to_py_err)?;

            // Remove from cache
            {
                let mut rules_guard = rules.write().await;
                rules_guard.retain(|rule| rule.name != name);
            }

            Ok(true)
        })
    }

    /// Clear all pattern rules
    pub fn clear_rules<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let rules = self.rules.clone();

        future_into_py(py, async move {
            let mut engine_guard = engine.write().await;
            engine_guard.clear_rules().await;

            // Clear cache
            {
                let mut rules_guard = rules.write().await;
                rules_guard.clear();
            }

            Ok(())
        })
    }

    /// Export pattern rules to file
    ///
    /// Args:
    ///     output_path: Path to save the rules file
    ///     format: Export format ("json", "yaml", "toml")
    ///
    /// Returns:
    ///     Success status
    pub fn export_rules<'py>(
        &self,
        py: Python<'py>,
        output_path: &str,
        format: &str,
    ) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let path = PathBuf::from(output_path);
        let export_format = format.to_string();

        future_into_py(py, async move {
            let engine_guard = engine.read().await;

            match export_format.as_str() {
                "json" => {
                    engine_guard
                        .export_rules_json(&path)
                        .await
                        .map_err(rust_error_to_py_err)?;
                }
                "yaml" => {
                    engine_guard
                        .export_rules_yaml(&path)
                        .await
                        .map_err(rust_error_to_py_err)?;
                }
                "toml" => {
                    engine_guard
                        .export_rules_toml(&path)
                        .await
                        .map_err(rust_error_to_py_err)?;
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unsupported export format: {}",
                        export_format
                    )));
                }
            }

            Ok(true)
        })
    }
}

/// Create a default pattern matcher
#[pyfunction]
pub fn create_pattern_matcher() -> PyResult<PatternMatcher> {
    PatternMatcher::new(None)
}

/// Get list of supported languages for pattern matching
#[pyfunction]
pub fn get_supported_languages() -> PyResult<PyObject> {
    let languages = scribe_patterns::get_supported_languages();

    Python::with_gil(|py| {
        let lang_list = PyList::empty(py);
        for lang in languages {
            lang_list.append(language_to_py(py, &lang)?)?;
        }
        Ok(lang_list.into())
    })
}

/// Validate a regex pattern
#[pyfunction]
pub fn validate_pattern(pattern: &str) -> PyResult<bool> {
    match scribe_patterns::validate_regex_pattern(pattern) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

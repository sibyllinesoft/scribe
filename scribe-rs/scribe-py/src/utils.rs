//! Utility functions for Python bindings
//!
//! Provides type conversions, helper functions, and common utilities for efficient
//! data exchange between Rust and Python, with support for NumPy integration.

use crate::error::rust_result_to_py;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use scribe_core::{
    AnalysisResult, FileInfo, FileType, GitStatus, HeuristicWeights, Language, Position, Range,
    RepositoryInfo, ScoreComponents,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Convert Rust PathBuf to Python string
pub fn pathbuf_to_py(py: Python, path: &PathBuf) -> PyResult<PyObject> {
    Ok(path.to_string_lossy().to_py(py))
}

/// Convert Python string to Rust PathBuf
pub fn py_to_pathbuf(obj: &PyAny) -> PyResult<PathBuf> {
    let path_str: String = obj.extract()?;
    Ok(PathBuf::from(path_str))
}

/// Convert Rust Vec<String> to Python list
pub fn vec_string_to_py(py: Python, vec: &[String]) -> PyResult<PyObject> {
    let py_list = PyList::empty(py);
    for item in vec {
        py_list.append(item)?;
    }
    Ok(py_list.into())
}

/// Convert Python list to Rust Vec<String>
pub fn py_to_vec_string(obj: &PyAny) -> PyResult<Vec<String>> {
    let py_list: &PyList = obj.extract()?;
    let mut vec = Vec::new();
    for item in py_list {
        let s: String = item.extract()?;
        vec.push(s);
    }
    Ok(vec)
}

/// Convert Rust Vec<PathBuf> to Python list of strings
pub fn vec_pathbuf_to_py(py: Python, paths: &[PathBuf]) -> PyResult<PyObject> {
    let py_list = PyList::empty(py);
    for path in paths {
        py_list.append(path.to_string_lossy().as_ref())?;
    }
    Ok(py_list.into())
}

/// Convert Python list to Rust Vec<PathBuf>
pub fn py_to_vec_pathbuf(obj: &PyAny) -> PyResult<Vec<PathBuf>> {
    let py_list: &PyList = obj.extract()?;
    let mut vec = Vec::new();
    for item in py_list {
        let path_str: String = item.extract()?;
        vec.push(PathBuf::from(path_str));
    }
    Ok(vec)
}

/// Convert Rust HashMap<String, T> to Python dict
pub fn hashmap_to_py_dict<T>(py: Python, map: &HashMap<String, T>) -> PyResult<PyObject>
where
    T: ToPyObject,
{
    let py_dict = PyDict::new(py);
    for (key, value) in map {
        py_dict.set_item(key, value)?;
    }
    Ok(py_dict.into())
}

/// Convert Python dict to Rust HashMap<String, T>
pub fn py_dict_to_hashmap<T>(obj: &PyAny) -> PyResult<HashMap<String, T>>
where
    for<'a> T: FromPyObject<'a>,
{
    let py_dict: &PyDict = obj.extract()?;
    let mut map = HashMap::new();
    for (key, value) in py_dict {
        let key_str: String = key.extract()?;
        let val: T = value.extract()?;
        map.insert(key_str, val);
    }
    Ok(map)
}

/// Convert Rust Language enum to Python string
pub fn language_to_py(py: Python, lang: &Language) -> PyResult<PyObject> {
    let lang_str = match lang {
        Language::Rust => "rust",
        Language::Python => "python",
        Language::JavaScript => "javascript",
        Language::TypeScript => "typescript",
        Language::Go => "go",
        Language::Java => "java",
        Language::Cpp => "cpp",
        Language::C => "c",
        Language::CSharp => "csharp",
        Language::Ruby => "ruby",
        Language::Php => "php",
        Language::Swift => "swift",
        Language::Kotlin => "kotlin",
        Language::Scala => "scala",
        Language::Shell => "shell",
        Language::Html => "html",
        Language::Css => "css",
        Language::Json => "json",
        Language::Xml => "xml",
        Language::Yaml => "yaml",
        Language::Toml => "toml",
        Language::Markdown => "markdown",
        Language::Text => "text",
        Language::Binary => "binary",
        Language::Unknown => "unknown",
    };
    Ok(lang_str.to_py(py))
}

/// Convert Python string to Rust Language enum
pub fn py_to_language(obj: &PyAny) -> PyResult<Language> {
    let lang_str: String = obj.extract()?;
    let lang = match lang_str.as_str() {
        "rust" => Language::Rust,
        "python" => Language::Python,
        "javascript" => Language::JavaScript,
        "typescript" => Language::TypeScript,
        "go" => Language::Go,
        "java" => Language::Java,
        "cpp" => Language::Cpp,
        "c" => Language::C,
        "csharp" => Language::CSharp,
        "ruby" => Language::Ruby,
        "php" => Language::Php,
        "swift" => Language::Swift,
        "kotlin" => Language::Kotlin,
        "scala" => Language::Scala,
        "shell" => Language::Shell,
        "html" => Language::Html,
        "css" => Language::Css,
        "json" => Language::Json,
        "xml" => Language::Xml,
        "yaml" => Language::Yaml,
        "toml" => Language::Toml,
        "markdown" => Language::Markdown,
        "text" => Language::Text,
        "binary" => Language::Binary,
        _ => Language::Unknown,
    };
    Ok(lang)
}

/// Convert Rust Position to Python tuple (line, column)
pub fn position_to_py(py: Python, pos: &Position) -> PyResult<PyObject> {
    let tuple = PyTuple::new(py, &[pos.line, pos.column]);
    Ok(tuple.into())
}

/// Convert Python tuple to Rust Position
pub fn py_to_position(obj: &PyAny) -> PyResult<Position> {
    let tuple: &PyTuple = obj.extract()?;
    if tuple.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Position must be a tuple of (line, column)",
        ));
    }
    let line: usize = tuple.get_item(0)?.extract()?;
    let column: usize = tuple.get_item(1)?.extract()?;
    Ok(Position { line, column })
}

/// Convert Rust Range to Python dict
pub fn range_to_py_dict(py: Python, range: &Range) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("start", position_to_py(py, &range.start)?)?;
    dict.set_item("end", position_to_py(py, &range.end)?)?;
    Ok(dict.into())
}

/// Convert Python dict to Rust Range
pub fn py_dict_to_range(obj: &PyAny) -> PyResult<Range> {
    let dict: &PyDict = obj.extract()?;
    let start = py_to_position(dict.get_item("start")?.unwrap())?;
    let end = py_to_position(dict.get_item("end")?.unwrap())?;
    Ok(Range { start, end })
}

/// Convert Rust ScoreComponents to Python dict
pub fn score_components_to_py_dict(py: Python, scores: &ScoreComponents) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("doc_score", scores.doc_score)?;
    dict.set_item("import_score", scores.import_score)?;
    dict.set_item("export_score", scores.export_score)?;
    dict.set_item("function_score", scores.function_score)?;
    dict.set_item("class_score", scores.class_score)?;
    dict.set_item("complexity_score", scores.complexity_score)?;
    dict.set_item("test_score", scores.test_score)?;
    dict.set_item("config_score", scores.config_score)?;
    dict.set_item("size_score", scores.size_score)?;
    dict.set_item("age_score", scores.age_score)?;
    dict.set_item("churn_score", scores.churn_score)?;
    dict.set_item("centrality_score", scores.centrality_score)?;
    dict.set_item("final_score", scores.final_score)?;
    Ok(dict.into())
}

/// Convert Python dict to Rust ScoreComponents
pub fn py_dict_to_score_components(obj: &PyAny) -> PyResult<ScoreComponents> {
    let dict: &PyDict = obj.extract()?;

    let mut scores = ScoreComponents::zero();

    if let Some(val) = dict.get_item("doc_score")? {
        scores.doc_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("import_score")? {
        scores.import_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("export_score")? {
        scores.export_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("function_score")? {
        scores.function_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("class_score")? {
        scores.class_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("complexity_score")? {
        scores.complexity_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("test_score")? {
        scores.test_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("config_score")? {
        scores.config_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("size_score")? {
        scores.size_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("age_score")? {
        scores.age_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("churn_score")? {
        scores.churn_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("centrality_score")? {
        scores.centrality_score = val.extract()?;
    }
    if let Some(val) = dict.get_item("final_score")? {
        scores.final_score = val.extract()?;
    }

    Ok(scores)
}

/// Convert Rust HeuristicWeights to Python dict
pub fn heuristic_weights_to_py_dict(py: Python, weights: &HeuristicWeights) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("documentation", weights.documentation)?;
    dict.set_item("imports", weights.imports)?;
    dict.set_item("exports", weights.exports)?;
    dict.set_item("functions", weights.functions)?;
    dict.set_item("classes", weights.classes)?;
    dict.set_item("complexity", weights.complexity)?;
    dict.set_item("tests", weights.tests)?;
    dict.set_item("config", weights.config)?;
    dict.set_item("size", weights.size)?;
    dict.set_item("age", weights.age)?;
    dict.set_item("churn", weights.churn)?;
    dict.set_item("centrality", weights.centrality)?;
    Ok(dict.into())
}

/// Convert Python dict to Rust HeuristicWeights
pub fn py_dict_to_heuristic_weights(obj: &PyAny) -> PyResult<HeuristicWeights> {
    let dict: &PyDict = obj.extract()?;

    let mut weights = HeuristicWeights::default();

    if let Some(val) = dict.get_item("documentation")? {
        weights.documentation = val.extract()?;
    }
    if let Some(val) = dict.get_item("imports")? {
        weights.imports = val.extract()?;
    }
    if let Some(val) = dict.get_item("exports")? {
        weights.exports = val.extract()?;
    }
    if let Some(val) = dict.get_item("functions")? {
        weights.functions = val.extract()?;
    }
    if let Some(val) = dict.get_item("classes")? {
        weights.classes = val.extract()?;
    }
    if let Some(val) = dict.get_item("complexity")? {
        weights.complexity = val.extract()?;
    }
    if let Some(val) = dict.get_item("tests")? {
        weights.tests = val.extract()?;
    }
    if let Some(val) = dict.get_item("config")? {
        weights.config = val.extract()?;
    }
    if let Some(val) = dict.get_item("size")? {
        weights.size = val.extract()?;
    }
    if let Some(val) = dict.get_item("age")? {
        weights.age = val.extract()?;
    }
    if let Some(val) = dict.get_item("churn")? {
        weights.churn = val.extract()?;
    }
    if let Some(val) = dict.get_item("centrality")? {
        weights.centrality = val.extract()?;
    }

    Ok(weights)
}

/// Helper function to create a progress callback from Python
pub fn create_progress_callback(callback: PyObject) -> impl Fn(usize, usize) -> bool + Send + Sync {
    move |current: usize, total: usize| -> bool {
        Python::with_gil(|py| {
            match callback.call1(py, (current, total)) {
                Ok(result) => {
                    // If callback returns false, stop processing
                    result.extract::<bool>(py).unwrap_or(true)
                }
                Err(_) => {
                    // If callback fails, continue processing
                    true
                }
            }
        })
    }
}

/// Format file size in human readable format
pub fn format_file_size(size: u64) -> String {
    scribe_core::bytes_to_human(size)
}

/// Validate and normalize a file path
pub fn normalize_file_path(path: &str) -> PyResult<String> {
    let path_buf = PathBuf::from(path);
    let normalized = scribe_core::normalize_path(&path_buf);
    Ok(normalized.to_string_lossy().to_string())
}

/// Check if a path is under a given directory
pub fn is_path_under_directory(path: &str, directory: &str) -> bool {
    let path = Path::new(path);
    let dir = Path::new(directory);
    scribe_core::is_under_directory(path, dir)
}

/// Extract numeric statistics from a collection
pub fn extract_numeric_stats(values: &[f64]) -> HashMap<String, f64> {
    let mut stats = HashMap::new();

    if values.is_empty() {
        return stats;
    }

    let mean = scribe_core::mean(values);
    let median = scribe_core::median(values);
    let std_dev = scribe_core::std_deviation(values);
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    stats.insert("mean".to_string(), mean);
    stats.insert("median".to_string(), median);
    stats.insert("std_dev".to_string(), std_dev);
    stats.insert("min".to_string(), min);
    stats.insert("max".to_string(), max);
    stats.insert("count".to_string(), values.len() as f64);

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_pathbuf_conversions() {
        Python::with_gil(|py| {
            let path = PathBuf::from("/home/user/test.rs");
            let py_path = pathbuf_to_py(py, &path).unwrap();
            let py_str: &PyString = py_path.downcast(py).unwrap();
            assert_eq!(py_str.to_string_lossy(), "/home/user/test.rs");
        });
    }

    #[test]
    fn test_language_conversions() {
        Python::with_gil(|py| {
            let lang = Language::Rust;
            let py_lang = language_to_py(py, &lang).unwrap();
            let py_str: &PyString = py_lang.downcast(py).unwrap();
            assert_eq!(py_str.to_string_lossy(), "rust");

            let converted_back = py_to_language(py_str).unwrap();
            assert_eq!(converted_back, Language::Rust);
        });
    }

    #[test]
    fn test_position_conversions() {
        Python::with_gil(|py| {
            let pos = Position {
                line: 10,
                column: 5,
            };
            let py_pos = position_to_py(py, &pos).unwrap();
            let py_tuple: &PyTuple = py_pos.downcast(py).unwrap();

            assert_eq!(py_tuple.len(), 2);
            assert_eq!(
                py_tuple.get_item(0).unwrap().extract::<usize>().unwrap(),
                10
            );
            assert_eq!(py_tuple.get_item(1).unwrap().extract::<usize>().unwrap(), 5);

            let converted_back = py_to_position(py_tuple).unwrap();
            assert_eq!(converted_back.line, 10);
            assert_eq!(converted_back.column, 5);
        });
    }

    #[test]
    fn test_score_components_conversion() {
        Python::with_gil(|py| {
            let mut scores = ScoreComponents::zero();
            scores.doc_score = 0.8;
            scores.import_score = 0.6;
            scores.final_score = 0.7;

            let py_scores = score_components_to_py_dict(py, &scores).unwrap();
            let py_dict: &PyDict = py_scores.downcast(py).unwrap();

            let doc_score: f64 = py_dict
                .get_item("doc_score")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(doc_score, 0.8);

            let converted_back = py_dict_to_score_components(py_dict).unwrap();
            assert_eq!(converted_back.doc_score, 0.8);
            assert_eq!(converted_back.import_score, 0.6);
            assert_eq!(converted_back.final_score, 0.7);
        });
    }

    #[test]
    fn test_vec_string_conversions() {
        Python::with_gil(|py| {
            let vec = vec!["hello".to_string(), "world".to_string()];
            let py_list = vec_string_to_py(py, &vec).unwrap();
            let py_list_ref: &PyList = py_list.downcast(py).unwrap();

            assert_eq!(py_list_ref.len(), 2);
            assert_eq!(
                py_list_ref
                    .get_item(0)
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "hello"
            );
            assert_eq!(
                py_list_ref
                    .get_item(1)
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "world"
            );

            let converted_back = py_to_vec_string(py_list_ref).unwrap();
            assert_eq!(converted_back, vec!["hello", "world"]);
        });
    }
}

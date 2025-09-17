//! Heuristic scoring system Python interface
//!
//! Provides Python bindings for the Scribe heuristic scoring system,
//! enabling high-performance file analysis and ranking in Python applications.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use scribe_analysis::{
    heuristics::{
        HeuristicScorer as AnalysisScorer, HeuristicWeights, ScanResult, ScoreComponents,
    },
    CodeAnalyzer,
};
use scribe_core::{Config, FileInfo, Result as ScribeResult};

use crate::error::{rust_error_to_py_err, rust_result_to_py};
use crate::utils::*;

/// Python wrapper for the heuristic scoring engine
#[pyclass]
pub struct PyHeuristicScorer {
    /// Scoring engine instance
    scorer: Arc<RwLock<AnalysisScorer>>,
    /// Configuration
    config: Arc<RwLock<Config>>,
    /// Custom weights
    weights: HeuristicWeights,
}

#[pymethods]
impl PyHeuristicScorer {
    /// Create a new heuristic scorer
    ///
    /// Args:
    ///     config: Optional configuration dict
    ///     weights: Optional custom scoring weights dict
    ///
    /// Returns:
    ///     PyHeuristicScorer instance
    #[new]
    pub fn new(config: Option<&PyDict>, weights: Option<&PyDict>) -> PyResult<Self> {
        // Load base configuration
        let mut scorer_config = Config::default();
        if let Some(py_config) = config {
            // TODO: Convert Python config dict to Rust Config
        }

        // Load custom weights
        let custom_weights = if let Some(py_weights) = weights {
            py_dict_to_heuristic_weights(py_weights)?
        } else {
            HeuristicWeights::default()
        };

        // Create scoring engine
        let scorer = AnalysisScorer::new(custom_weights.clone());

        Ok(PyHeuristicScorer {
            scorer: Arc::new(RwLock::new(scorer)),
            config: Arc::new(RwLock::new(scorer_config)),
            weights: custom_weights,
        })
    }

    /// Score a single file
    ///
    /// Args:
    ///     file_path: Path to the file to score
    ///     file_content: Optional file content (if not provided, will read from disk)
    ///
    /// Returns:
    ///     Dictionary containing score components
    pub fn score_file<'py>(
        &self,
        py: Python<'py>,
        file_path: &str,
        file_content: Option<&str>,
    ) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let weights = self.weights.clone();
        let path = PathBuf::from(file_path);
        let content = file_content.map(|s| s.to_string());

        future_into_py(py, async move {
            let engine_guard = engine.read().await;
            let weights_guard = weights.read().await;

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

            // Create FileInfo for scoring
            let file_info = FileInfo::from_path_and_content(&path, &file_content)
                .map_err(rust_error_to_py_err)?;

            // Perform scoring
            let scores = engine_guard
                .score_file(&file_info, &weights_guard)
                .await
                .map_err(rust_error_to_py_err)?;

            // Convert to Python dict
            Python::with_gil(|py| score_components_to_py_dict(py, &scores))
        })
    }

    /// Score multiple files
    ///
    /// Args:
    ///     files: List of FileInfo objects or file paths
    ///     batch_size: Number of files to process in each batch
    ///     progress_callback: Optional progress callback function
    ///
    /// Returns:
    ///     Dictionary mapping file paths to score components
    pub fn score_files<'py>(
        &self,
        py: Python<'py>,
        files: &PyList,
        batch_size: Option<usize>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<&'py PyAny> {
        let engine = self.engine.clone();
        let weights = self.weights.clone();
        let batch_size = batch_size.unwrap_or(100);

        // Convert Python list to Rust Vec
        let mut file_paths = Vec::new();
        for item in files {
            let path_str: String = item.extract()?;
            file_paths.push(PathBuf::from(path_str));
        }

        future_into_py(py, async move {
            let engine_guard = engine.read().await;
            let weights_guard = weights.read().await;

            let total_files = file_paths.len();
            let mut results = HashMap::new();

            // Process files in batches
            for (batch_idx, batch) in file_paths.chunks(batch_size).enumerate() {
                let mut batch_results = Vec::new();

                // Process batch in parallel
                for path in batch {
                    let file_content = match std::fs::read_to_string(path) {
                        Ok(content) => content,
                        Err(_) => continue, // Skip files we can't read
                    };

                    let file_info = match FileInfo::from_path_and_content(path, &file_content) {
                        Ok(info) => info,
                        Err(_) => continue, // Skip files we can't analyze
                    };

                    let scores = match engine_guard.score_file(&file_info, &weights_guard).await {
                        Ok(scores) => scores,
                        Err(_) => continue, // Skip files we can't score
                    };

                    batch_results.push((path.clone(), scores));
                }

                // Add batch results to final results
                for (path, scores) in batch_results {
                    let path_str = path.to_string_lossy().to_string();
                    results.insert(path_str, scores);
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

            // Convert results to Python dict
            Python::with_gil(|py| {
                let py_dict = PyDict::new(py);
                for (path, scores) in results {
                    let py_scores = score_components_to_py_dict(py, &scores)?;
                    py_dict.set_item(path, py_scores)?;
                }
                Ok(py_dict.into())
            })
        })
    }

    /// Combine file scores with centrality scores
    ///
    /// Args:
    ///     file_scores: Dictionary of file paths to score components
    ///     centrality_scores: Dictionary of file paths to centrality values
    ///     centrality_weight: Weight for centrality in final score (default: 0.2)
    ///
    /// Returns:
    ///     Dictionary of file paths to updated score components
    pub fn combine_with_centrality(
        &self,
        file_scores: &PyDict,
        centrality_scores: &PyDict,
        centrality_weight: Option<f64>,
    ) -> PyResult<PyObject> {
        let weight = centrality_weight.unwrap_or(0.2);

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);

            for (file_path, scores_obj) in file_scores {
                let path_str: String = file_path.extract()?;
                let mut scores = py_dict_to_score_components(scores_obj)?;

                // Get centrality score for this file
                if let Some(centrality_obj) = centrality_scores.get_item(&path_str)? {
                    let centrality: f64 = centrality_obj.extract()?;
                    scores.centrality_score = centrality;

                    // Recompute final score with centrality
                    let weights_guard = futures::executor::block_on(self.weights.read());
                    scores.compute_final_score(&weights_guard);
                }

                let py_scores = score_components_to_py_dict(py, &scores)?;
                result_dict.set_item(path_str, py_scores)?;
            }

            Ok(result_dict.into())
        })
    }

    /// Get top N files by score
    ///
    /// Args:
    ///     scored_files: Dictionary of file paths to score components
    ///     n: Number of top files to return
    ///     score_field: Score field to sort by (default: "final_score")
    ///
    /// Returns:
    ///     List of tuples (file_path, score_value) sorted by score descending
    pub fn get_top_files(
        &self,
        scored_files: &PyDict,
        n: usize,
        score_field: Option<&str>,
    ) -> PyResult<PyObject> {
        let field = score_field.unwrap_or("final_score");

        Python::with_gil(|py| {
            let mut file_scores: Vec<(String, f64)> = Vec::new();

            for (file_path, scores_obj) in scored_files {
                let path_str: String = file_path.extract()?;
                let scores_dict: &PyDict = scores_obj.extract()?;

                if let Some(score_obj) = scores_dict.get_item(field)? {
                    let score: f64 = score_obj.extract()?;
                    file_scores.push((path_str, score));
                }
            }

            // Sort by score descending
            file_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top N
            let top_files: Vec<(String, f64)> = file_scores.into_iter().take(n).collect();

            // Convert to Python list
            let py_list = PyList::empty(py);
            for (path, score) in top_files {
                let tuple =
                    pyo3::types::PyTuple::new(py, &[path.to_object(py), score.to_object(py)]);
                py_list.append(tuple)?;
            }

            Ok(py_list.into())
        })
    }

    /// Update scoring weights
    ///
    /// Args:
    ///     weights: Dictionary of weight values
    pub fn update_weights(&self, weights: &PyDict) -> PyResult<()> {
        let new_weights = py_dict_to_heuristic_weights(weights)?;

        // Update weights
        futures::executor::block_on(async {
            let mut weights_guard = self.weights.write().await;
            *weights_guard = new_weights;
        });

        Ok(())
    }

    /// Get current scoring weights
    pub fn get_weights(&self) -> PyResult<PyObject> {
        let weights_guard = futures::executor::block_on(self.weights.read());

        Python::with_gil(|py| heuristic_weights_to_py_dict(py, &weights_guard))
    }

    /// Calculate statistics for a set of scored files
    ///
    /// Args:
    ///     scored_files: Dictionary of file paths to score components
    ///     score_field: Score field to analyze (default: "final_score")
    ///
    /// Returns:
    ///     Dictionary containing statistical metrics
    pub fn calculate_score_statistics(
        &self,
        scored_files: &PyDict,
        score_field: Option<&str>,
    ) -> PyResult<PyObject> {
        let field = score_field.unwrap_or("final_score");

        Python::with_gil(|py| {
            let mut scores: Vec<f64> = Vec::new();

            for (_, scores_obj) in scored_files {
                let scores_dict: &PyDict = scores_obj.extract()?;

                if let Some(score_obj) = scores_dict.get_item(field)? {
                    let score: f64 = score_obj.extract()?;
                    scores.push(score);
                }
            }

            if scores.is_empty() {
                return Ok(PyDict::new(py).into());
            }

            let stats = extract_numeric_stats(&scores);
            hashmap_to_py_dict(py, &stats)
        })
    }

    /// Filter files by score threshold
    ///
    /// Args:
    ///     scored_files: Dictionary of file paths to score components
    ///     threshold: Minimum score threshold
    ///     score_field: Score field to filter by (default: "final_score")
    ///
    /// Returns:
    ///     Dictionary of files meeting the threshold
    pub fn filter_by_score_threshold(
        &self,
        scored_files: &PyDict,
        threshold: f64,
        score_field: Option<&str>,
    ) -> PyResult<PyObject> {
        let field = score_field.unwrap_or("final_score");

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);

            for (file_path, scores_obj) in scored_files {
                let scores_dict: &PyDict = scores_obj.extract()?;

                if let Some(score_obj) = scores_dict.get_item(field)? {
                    let score: f64 = score_obj.extract()?;
                    if score >= threshold {
                        result_dict.set_item(file_path, scores_obj)?;
                    }
                }
            }

            Ok(result_dict.into())
        })
    }
}

/// Create a default heuristic scorer
#[pyfunction]
pub fn create_default_scorer() -> PyResult<HeuristicScorer> {
    HeuristicScorer::new(None, None)
}

/// Create scorer with custom weights
#[pyfunction]
pub fn create_scorer_with_weights(weights: &PyDict) -> PyResult<HeuristicScorer> {
    HeuristicScorer::new(None, Some(weights))
}

/// Get default scoring weights as Python dict
#[pyfunction]
pub fn get_default_weights() -> PyResult<PyObject> {
    let weights = HeuristicWeights::default();
    Python::with_gil(|py| heuristic_weights_to_py_dict(py, &weights))
}

//! Graph analysis and PageRank Python interface
//!
//! Provides Python bindings for dependency graph analysis, PageRank centrality calculation,
//! and other graph-based metrics for code analysis.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use scribe_core::{Config, FileInfo, CentralityScores, GraphStats};
use scribe_graph::{DependencyGraph, PageRankComputer, GraphStatisticsAnalyzer};

use crate::error::{rust_result_to_py, rust_error_to_py_err};
use crate::utils::*;

/// Python wrapper for PageRank analysis engine
#[pyclass]
pub struct PageRankAnalyzer {
    /// PageRank engine instance
    engine: Arc<RwLock<PageRankEngine>>,
    /// Graph analyzer instance
    analyzer: Arc<RwLock<GraphAnalyzer>>,
    /// Configuration
    config: Arc<RwLock<Config>>,
    /// Cached dependency graph
    graph: Arc<RwLock<Option<DependencyGraph>>>,
}

#[pymethods]
impl PageRankAnalyzer {
    /// Create a new PageRank analyzer
    /// 
    /// Args:
    ///     damping_factor: PageRank damping factor (default: 0.85)
    ///     max_iterations: Maximum iterations for convergence (default: 100)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    ///     config: Optional configuration dict
    /// 
    /// Returns:
    ///     PageRankAnalyzer instance
    #[new]
    pub fn new(
        damping_factor: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
        config: Option<&PyDict>,
    ) -> PyResult<Self> {
        let damping = damping_factor.unwrap_or(0.85);
        let max_iter = max_iterations.unwrap_or(100);
        let tol = tolerance.unwrap_or(1e-6);
        
        // Load configuration
        let analyzer_config = Config::default();
        
        // Create PageRank engine with custom parameters
        let mut engine_config = scribe_graph::PageRankConfig::default();
        engine_config.damping_factor = damping;
        engine_config.max_iterations = max_iter;
        engine_config.tolerance = tol;
        
        let engine = PageRankEngine::with_config(engine_config)
            .map_err(rust_error_to_py_err)?;
        
        let analyzer = GraphAnalyzer::new(analyzer_config.clone())
            .map_err(rust_error_to_py_err)?;
        
        Ok(PageRankAnalyzer {
            engine: Arc::new(RwLock::new(engine)),
            analyzer: Arc::new(RwLock::new(analyzer)),
            config: Arc::new(RwLock::new(analyzer_config)),
            graph: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Analyze dependencies and calculate PageRank scores
    /// 
    /// Args:
    ///     files: List of FileInfo objects or file paths
    ///     include_external: Whether to include external dependencies
    ///     progress_callback: Optional progress callback function
    /// 
    /// Returns:
    ///     Dictionary mapping file paths to PageRank scores
    pub fn analyze_dependencies<'py>(
        &self,
        py: Python<'py>,
        files: &PyList,
        include_external: Option<bool>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<&'py PyAny> {
        let analyzer = self.analyzer.clone();
        let engine = self.engine.clone();
        let graph = self.graph.clone();
        let include_ext = include_external.unwrap_or(false);
        
        // Convert Python list to file paths
        let mut file_paths = Vec::new();
        for item in files {
            let path_str: String = item.extract()?;
            file_paths.push(PathBuf::from(path_str));
        }
        
        future_into_py(py, async move {
            let analyzer_guard = analyzer.read().await;
            let engine_guard = engine.read().await;
            
            // Build dependency graph
            let mut dep_graph = DependencyGraph::new();
            let total_files = file_paths.len();
            
            for (idx, path) in file_paths.iter().enumerate() {
                // Read file content
                let content = match std::fs::read_to_string(path) {
                    Ok(content) => content,
                    Err(_) => continue, // Skip files we can't read
                };
                
                // Create FileInfo
                let file_info = match FileInfo::from_path_and_content(path, &content) {
                    Ok(info) => info,
                    Err(_) => continue,
                };
                
                // Analyze dependencies for this file
                let dependencies = analyzer_guard.analyze_file_dependencies(&file_info)
                    .await
                    .map_err(rust_error_to_py_err)?;
                
                // Add to graph
                let node_id = path.to_string_lossy().to_string();
                dep_graph.add_node(node_id.clone(), file_info);
                
                for dep_path in dependencies {
                    if include_ext || file_paths.iter().any(|p| p == &dep_path) {
                        let dep_id = dep_path.to_string_lossy().to_string();
                        dep_graph.add_edge(node_id.clone(), dep_id);
                    }
                }
                
                // Call progress callback
                if let Some(ref callback) = progress_callback {
                    Python::with_gil(|py| {
                        let _ = callback.call1(py, (idx + 1, total_files));
                    });
                }
            }
            
            // Cache the graph
            {
                let mut graph_guard = graph.write().await;
                *graph_guard = Some(dep_graph.clone());
            }
            
            // Calculate PageRank scores
            let scores = engine_guard.calculate_pagerank(&dep_graph)
                .await
                .map_err(rust_error_to_py_err)?;
                
            Ok(scores)
        })
    }
    
    /// Calculate various centrality measures
    /// 
    /// Args:
    ///     files: List of FileInfo objects or file paths
    ///     measures: List of centrality measures to calculate
    ///               Options: ["pagerank", "betweenness", "closeness", "degree", "eigenvector"]
    /// 
    /// Returns:
    ///     Dictionary mapping file paths to centrality score objects
    pub fn calculate_centrality_measures<'py>(
        &self,
        py: Python<'py>,
        files: &PyList,
        measures: Option<Vec<String>>,
    ) -> PyResult<&'py PyAny> {
        let analyzer = self.analyzer.clone();
        let graph = self.graph.clone();
        let requested_measures = measures.unwrap_or_else(|| vec!["pagerank".to_string()]);
        
        // Convert Python list to file paths
        let mut file_paths = Vec::new();
        for item in files {
            let path_str: String = item.extract()?;
            file_paths.push(PathBuf::from(path_str));
        }
        
        future_into_py(py, async move {
            let analyzer_guard = analyzer.read().await;
            
            // Get or build the dependency graph
            let dep_graph = {
                let graph_guard = graph.read().await;
                if let Some(existing_graph) = graph_guard.as_ref() {
                    existing_graph.clone()
                } else {
                    drop(graph_guard);
                    
                    // Build the graph if it doesn't exist
                    let mut new_graph = DependencyGraph::new();
                    
                    for path in &file_paths {
                        let content = match std::fs::read_to_string(path) {
                            Ok(content) => content,
                            Err(_) => continue,
                        };
                        
                        let file_info = match FileInfo::from_path_and_content(path, &content) {
                            Ok(info) => info,
                            Err(_) => continue,
                        };
                        
                        let dependencies = analyzer_guard.analyze_file_dependencies(&file_info)
                            .await
                            .map_err(rust_error_to_py_err)?;
                        
                        let node_id = path.to_string_lossy().to_string();
                        new_graph.add_node(node_id.clone(), file_info);
                        
                        for dep_path in dependencies {
                            let dep_id = dep_path.to_string_lossy().to_string();
                            new_graph.add_edge(node_id.clone(), dep_id);
                        }
                    }
                    
                    // Cache the new graph
                    {
                        let mut graph_guard = graph.write().await;
                        *graph_guard = Some(new_graph.clone());
                    }
                    
                    new_graph
                }
            };
            
            // Calculate requested centrality measures
            let mut results = HashMap::new();
            
            for measure in requested_measures {
                let centrality_scores = match measure.as_str() {
                    "pagerank" => {
                        analyzer_guard.calculate_pagerank_centrality(&dep_graph)
                            .await
                            .map_err(rust_error_to_py_err)?
                    },
                    "betweenness" => {
                        analyzer_guard.calculate_betweenness_centrality(&dep_graph)
                            .await
                            .map_err(rust_error_to_py_err)?
                    },
                    "closeness" => {
                        analyzer_guard.calculate_closeness_centrality(&dep_graph)
                            .await
                            .map_err(rust_error_to_py_err)?
                    },
                    "degree" => {
                        analyzer_guard.calculate_degree_centrality(&dep_graph)
                            .await
                            .map_err(rust_error_to_py_err)?
                    },
                    "eigenvector" => {
                        analyzer_guard.calculate_eigenvector_centrality(&dep_graph)
                            .await
                            .map_err(rust_error_to_py_err)?
                    },
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Unknown centrality measure: {}", measure)
                        ));
                    }
                };
                
                results.insert(measure, centrality_scores);
            }
            
            // Convert to Python format
            Python::with_gil(|py| {
                let result_dict = PyDict::new(py);
                
                // Get all file paths from the graph
                let all_nodes: Vec<String> = dep_graph.get_all_nodes()
                    .into_iter()
                    .collect();
                
                for node_id in all_nodes {
                    let node_dict = PyDict::new(py);
                    
                    for (measure, scores) in &results {
                        if let Some(score) = scores.get(&node_id) {
                            node_dict.set_item(measure, *score)?;
                        }
                    }
                    
                    result_dict.set_item(node_id, node_dict)?;
                }
                
                Ok(result_dict.into())
            })
        })
    }
    
    /// Get graph statistics
    /// 
    /// Returns:
    ///     Dictionary containing graph statistics
    pub fn get_graph_statistics<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let graph = self.graph.clone();
        let analyzer = self.analyzer.clone();
        
        future_into_py(py, async move {
            let graph_guard = graph.read().await;
            
            if let Some(dep_graph) = graph_guard.as_ref() {
                let analyzer_guard = analyzer.read().await;
                let stats = analyzer_guard.calculate_graph_statistics(dep_graph)
                    .await
                    .map_err(rust_error_to_py_err)?;
                
                // Convert to Python dict
                Python::with_gil(|py| {
                    let stats_dict = PyDict::new(py);
                    stats_dict.set_item("node_count", stats.node_count)?;
                    stats_dict.set_item("edge_count", stats.edge_count)?;
                    stats_dict.set_item("density", stats.density)?;
                    stats_dict.set_item("diameter", stats.diameter)?;
                    stats_dict.set_item("average_path_length", stats.average_path_length)?;
                    stats_dict.set_item("clustering_coefficient", stats.clustering_coefficient)?;
                    stats_dict.set_item("connected_components", stats.connected_components)?;
                    Ok(stats_dict.into())
                })
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "No dependency graph available. Run analyze_dependencies first."
                ))
            }
        })
    }
    
    /// Export dependency graph to various formats
    /// 
    /// Args:
    ///     format: Export format ("graphml", "dot", "json", "csv")
    ///     output_path: Path to save the exported graph
    ///     include_metadata: Whether to include node metadata
    /// 
    /// Returns:
    ///     Success status
    pub fn export_graph<'py>(
        &self,
        py: Python<'py>,
        format: &str,
        output_path: &str,
        include_metadata: Option<bool>,
    ) -> PyResult<&'py PyAny> {
        let graph = self.graph.clone();
        let export_format = format.to_string();
        let output = PathBuf::from(output_path);
        let include_meta = include_metadata.unwrap_or(true);
        
        future_into_py(py, async move {
            let graph_guard = graph.read().await;
            
            if let Some(dep_graph) = graph_guard.as_ref() {
                match export_format.as_str() {
                    "graphml" => {
                        dep_graph.export_graphml(&output, include_meta)
                            .await
                            .map_err(rust_error_to_py_err)?;
                    },
                    "dot" => {
                        dep_graph.export_dot(&output, include_meta)
                            .await
                            .map_err(rust_error_to_py_err)?;
                    },
                    "json" => {
                        dep_graph.export_json(&output, include_meta)
                            .await
                            .map_err(rust_error_to_py_err)?;
                    },
                    "csv" => {
                        dep_graph.export_csv(&output)
                            .await
                            .map_err(rust_error_to_py_err)?;
                    },
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Unsupported export format: {}", export_format)
                        ));
                    }
                }
                Ok(true)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "No dependency graph available. Run analyze_dependencies first."
                ))
            }
        })
    }
    
    /// Find strongly connected components in the dependency graph
    /// 
    /// Returns:
    ///     List of strongly connected components (each is a list of file paths)
    pub fn find_strongly_connected_components<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let graph = self.graph.clone();
        let analyzer = self.analyzer.clone();
        
        future_into_py(py, async move {
            let graph_guard = graph.read().await;
            
            if let Some(dep_graph) = graph_guard.as_ref() {
                let analyzer_guard = analyzer.read().await;
                let components = analyzer_guard.find_strongly_connected_components(dep_graph)
                    .await
                    .map_err(rust_error_to_py_err)?;
                
                // Convert to Python list of lists
                Python::with_gil(|py| {
                    let result_list = PyList::empty(py);
                    
                    for component in components {
                        let component_list = PyList::empty(py);
                        for node_id in component {
                            component_list.append(node_id)?;
                        }
                        result_list.append(component_list)?;
                    }
                    
                    Ok(result_list.into())
                })
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "No dependency graph available. Run analyze_dependencies first."
                ))
            }
        })
    }
    
    /// Find circular dependencies in the graph
    /// 
    /// Returns:
    ///     List of circular dependency chains (each is a list of file paths)
    pub fn find_circular_dependencies<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let graph = self.graph.clone();
        let analyzer = self.analyzer.clone();
        
        future_into_py(py, async move {
            let graph_guard = graph.read().await;
            
            if let Some(dep_graph) = graph_guard.as_ref() {
                let analyzer_guard = analyzer.read().await;
                let cycles = analyzer_guard.find_circular_dependencies(dep_graph)
                    .await
                    .map_err(rust_error_to_py_err)?;
                
                // Convert to Python list of lists
                Python::with_gil(|py| {
                    let result_list = PyList::empty(py);
                    
                    for cycle in cycles {
                        let cycle_list = PyList::empty(py);
                        for node_id in cycle {
                            cycle_list.append(node_id)?;
                        }
                        result_list.append(cycle_list)?;
                    }
                    
                    Ok(result_list.into())
                })
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "No dependency graph available. Run analyze_dependencies first."
                ))
            }
        })
    }
    
    /// Clear cached graph data
    pub fn clear_cache(&self) -> PyResult<()> {
        futures::executor::block_on(async {
            let mut graph_guard = self.graph.write().await;
            *graph_guard = None;
        });
        Ok(())
    }
    
    /// Update PageRank configuration
    /// 
    /// Args:
    ///     damping_factor: New damping factor (optional)
    ///     max_iterations: New max iterations (optional)
    ///     tolerance: New convergence tolerance (optional)
    pub fn update_config(
        &self,
        damping_factor: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<()> {
        futures::executor::block_on(async {
            let mut engine_guard = self.engine.write().await;
            
            if let Some(damping) = damping_factor {
                engine_guard.set_damping_factor(damping);
            }
            if let Some(max_iter) = max_iterations {
                engine_guard.set_max_iterations(max_iter);
            }
            if let Some(tol) = tolerance {
                engine_guard.set_tolerance(tol);
            }
        });
        Ok(())
    }
    
    /// Get current PageRank configuration
    pub fn get_config(&self) -> PyResult<PyObject> {
        let engine_guard = futures::executor::block_on(self.engine.read());
        let config = engine_guard.get_config();
        
        Python::with_gil(|py| {
            let config_dict = PyDict::new(py);
            config_dict.set_item("damping_factor", config.damping_factor)?;
            config_dict.set_item("max_iterations", config.max_iterations)?;
            config_dict.set_item("tolerance", config.tolerance)?;
            Ok(config_dict.into())
        })
    }
}

/// Create a default PageRank analyzer
#[pyfunction]
pub fn create_pagerank_analyzer() -> PyResult<PageRankAnalyzer> {
    PageRankAnalyzer::new(None, None, None, None)
}

/// Create PageRank analyzer with custom configuration
#[pyfunction]
pub fn create_pagerank_analyzer_with_config(
    damping_factor: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PageRankAnalyzer> {
    PageRankAnalyzer::new(
        Some(damping_factor),
        Some(max_iterations), 
        Some(tolerance),
        None
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    use pyo3::types::PyList;

    #[test]
    fn test_pagerank_analyzer_creation() {
        Python::with_gil(|py| {
            let analyzer = PageRankAnalyzer::new(None, None, None, None);
            assert!(analyzer.is_ok());
            
            // Test with custom parameters
            let custom_analyzer = PageRankAnalyzer::new(
                Some(0.9),
                Some(200),
                Some(1e-8),
                None
            );
            assert!(custom_analyzer.is_ok());
        });
    }

    #[test]
    fn test_create_pagerank_analyzer() {
        let analyzer = create_pagerank_analyzer();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_create_pagerank_analyzer_with_config() {
        let analyzer = create_pagerank_analyzer_with_config(0.9, 200, 1e-8);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_config_operations() {
        Python::with_gil(|py| {
            let analyzer = PageRankAnalyzer::new(None, None, None, None).unwrap();
            
            // Test getting config
            let config = analyzer.get_config();
            assert!(config.is_ok());
            
            // Test updating config
            let result = analyzer.update_config(Some(0.9), Some(150), Some(1e-7));
            assert!(result.is_ok());
            
            // Test clearing cache
            let result = analyzer.clear_cache();
            assert!(result.is_ok());
        });
    }
}
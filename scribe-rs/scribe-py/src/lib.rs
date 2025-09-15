//! # Scribe Python Bindings
//! 
//! High-performance Python bindings for the Scribe code analysis library, providing
//! comprehensive file analysis, dependency graphing, pattern matching, and heuristic
//! scoring capabilities with full async support.
//!
//! ## Features
//!
//! - **Repository Analysis**: Fast, parallel file scanning and analysis
//! - **Heuristic Scoring**: Advanced scoring system for file importance
//! - **Dependency Graphs**: PageRank centrality and graph analysis  
//! - **Pattern Matching**: Flexible regex-based code pattern detection
//! - **Async Support**: Full async/await support with progress callbacks
//! - **NumPy Integration**: Efficient data exchange with NumPy arrays
//! - **Memory Efficient**: Zero-copy data structures where possible
//!
//! ## Example Usage
//!
//! ```python
//! import asyncio
//! from scribe_rs import Repository, HeuristicScorer, PageRankAnalyzer
//!
//! async def analyze_repository():
//!     # Initialize repository analysis
//!     repo = Repository("/path/to/repo")
//!     files = await repo.scan_files(max_files=1000)
//!
//!     # Score files using Rust heuristics  
//!     scorer = HeuristicScorer()
//!     scored_files = await scorer.score_files(files)
//!
//!     # Calculate PageRank centrality
//!     analyzer = PageRankAnalyzer()
//!     centrality_scores = await analyzer.analyze_dependencies(files)
//!
//!     # Combined analysis
//!     final_scores = scorer.combine_with_centrality(scored_files, centrality_scores)
//!     
//!     return final_scores
//!
//! # Run the analysis
//! results = asyncio.run(analyze_repository())
//! ```

use pyo3::prelude::*;
use pyo3_asyncio::tokio::future_into_py;

// Module declarations
mod error;
mod utils;
mod config;
mod repository;
mod scoring;
mod graph;
mod patterns;

// Re-export public APIs
pub use error::{
    rust_error_to_py_err, rust_result_to_py, ScribeException, AnalysisException,
    PatternException, ConfigurationException, ToPyResult
};

pub use config::{
    PyConfig as Config
};

pub use repository::{
    Repository, create_repository, is_valid_repository, find_repository_root
};

pub use scoring::{
    HeuristicScorer, create_default_scorer, create_scorer_with_weights, get_default_weights
};

pub use graph::{
    PageRankAnalyzer, create_pagerank_analyzer, create_pagerank_analyzer_with_config
};

pub use patterns::{
    PatternMatcher, create_pattern_matcher, get_supported_languages, validate_pattern
};

/// Python module for the Scribe library
#[pymodule]
fn _scribe_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize the async runtime
    pyo3_asyncio::tokio::init_multi_thread();
    
    // Configuration
    m.add_class::<config::PyConfig>()?;
    
    // Core classes - Repository Analysis
    m.add_class::<Repository>()?;
    m.add_function(wrap_pyfunction!(create_repository, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid_repository, m)?)?;
    m.add_function(wrap_pyfunction!(find_repository_root, m)?)?;
    
    // Heuristic Scoring
    m.add_class::<HeuristicScorer>()?;
    m.add_function(wrap_pyfunction!(create_default_scorer, m)?)?;
    m.add_function(wrap_pyfunction!(create_scorer_with_weights, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_weights, m)?)?;
    
    // Graph Analysis and PageRank
    m.add_class::<PageRankAnalyzer>()?;
    m.add_function(wrap_pyfunction!(create_pagerank_analyzer, m)?)?;
    m.add_function(wrap_pyfunction!(create_pagerank_analyzer_with_config, m)?)?;
    
    // Pattern Matching
    m.add_class::<PatternMatcher>()?;
    m.add_function(wrap_pyfunction!(create_pattern_matcher, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_languages, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pattern, m)?)?;
    
    // Custom exceptions
    m.add("ScribeException", _py.get_type::<ScribeException>())?;
    m.add("AnalysisException", _py.get_type::<AnalysisException>())?;
    m.add("PatternException", _py.get_type::<PatternException>())?;
    m.add("ConfigurationException", _py.get_type::<ConfigurationException>())?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(initialize_async_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(get_version_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    
    // Version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Module docstring
    m.add("__doc__", "High-performance code analysis library with Rust backend")?;

    Ok(())
}

/// Initialize the async runtime for Python bindings
/// This should be called once at module import time
#[pyfunction]
fn initialize_async_runtime() -> PyResult<()> {
    pyo3_asyncio::tokio::init_multi_thread();
    Ok(())
}

/// Get version information
#[pyfunction]
fn get_version_info() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let version_dict = pyo3::types::PyDict::new(py);
        version_dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
        version_dict.set_item("name", env!("CARGO_PKG_NAME"))?;
        version_dict.set_item("description", env!("CARGO_PKG_DESCRIPTION"))?;
        version_dict.set_item("authors", env!("CARGO_PKG_AUTHORS"))?;
        version_dict.set_item("repository", env!("CARGO_PKG_REPOSITORY"))?;
        version_dict.set_item("license", env!("CARGO_PKG_LICENSE"))?;
        Ok(version_dict.into())
    })
}

/// Get detailed build information
#[pyfunction]  
fn get_build_info() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let build_dict = pyo3::types::PyDict::new(py);
        build_dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
        build_dict.set_item("name", env!("CARGO_PKG_NAME"))?;
        
        // Add optional build-time information if available
        if let Some(target) = option_env!("TARGET") {
            build_dict.set_item("target", target)?;
        }
        
        if let Some(git_hash) = option_env!("GIT_HASH") {
            build_dict.set_item("git_hash", &git_hash[..8.min(git_hash.len())])?;
        }
        
        if let Some(timestamp) = option_env!("BUILD_TIMESTAMP") {
            build_dict.set_item("build_timestamp", timestamp)?;
        }
        
        // Rust version used to build
        build_dict.set_item("rustc_version", env!("CARGO_PKG_RUST_VERSION"))?;
        
        Ok(build_dict.into())
    })
}

/// High-level analysis function combining multiple analyses
#[pyfunction]
fn analyze_repository_comprehensive<'py>(
    py: Python<'py>,
    repo_path: &str,
    max_files: Option<usize>,
    include_patterns: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
    scoring_weights: Option<&pyo3::types::PyDict>,
    pagerank_damping: Option<f64>,
    progress_callback: Option<PyObject>,
) -> PyResult<&'py PyAny> {
    future_into_py(py, async move {
        // Create repository
        let repo = Repository::new(repo_path, None)?;
        
        // Scan files
        let py_list = Python::with_gil(|py| pyo3::types::PyList::empty(py).into());
        let files = repo.scan_files(
            Python::with_gil(|py| py), 
            max_files,
            include_patterns,
            exclude_patterns,
            progress_callback.clone()
        ).await?;
        
        // Create scorer  
        let scorer = if let Some(weights) = scoring_weights {
            HeuristicScorer::new(None, Some(weights))?
        } else {
            HeuristicScorer::new(None, None)?
        };
        
        // Score files
        let py_list_ref = Python::with_gil(|py| {
            let files_list = pyo3::types::PyList::empty(py);
            // Convert files to list - this would need actual file data
            files_list
        });
        
        let scored_files = scorer.score_files(
            Python::with_gil(|py| py),
            py_list_ref,
            None,
            progress_callback.clone()
        ).await?;
        
        // Create PageRank analyzer
        let analyzer = PageRankAnalyzer::new(
            pagerank_damping,
            None,
            None,
            None
        )?;
        
        // Analyze dependencies  
        let centrality_scores = analyzer.analyze_dependencies(
            Python::with_gil(|py| py),
            py_list_ref,
            Some(false),
            progress_callback
        ).await?;
        
        // Combine results
        let final_results = Python::with_gil(|py| {
            let scored_dict = scored_files.downcast(py)?;
            let centrality_dict = centrality_scores.downcast(py)?;
            
            scorer.combine_with_centrality(
                scored_dict,
                centrality_dict,
                None
            )
        })?;
        
        // Create comprehensive results
        Python::with_gil(|py| {
            let results_dict = pyo3::types::PyDict::new(py);
            results_dict.set_item("scored_files", final_results)?;
            results_dict.set_item("repository_info", repo.get_repository_info(py)?)?;
            results_dict.set_item("language_stats", repo.get_language_stats(py)?)?;
            results_dict.set_item("size_stats", repo.get_size_stats(py)?)?;
            
            if repo.has_git() {
                results_dict.set_item("git_stats", repo.get_git_stats(py)?)?;
            }
            
            Ok(results_dict.into())
        })
    })
}

/// Add the comprehensive analysis function to the module
#[pymodule]
fn scribe_py_extended(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_repository_comprehensive, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_module_creation() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_scribe_py").unwrap();
            _scribe_py(py, module).unwrap();
            
            // Test that the module has expected attributes
            assert!(module.getattr("__version__").is_ok());
            assert!(module.getattr("__doc__").is_ok());
            
            // Test classes are available
            assert!(module.getattr("Repository").is_ok());
            assert!(module.getattr("HeuristicScorer").is_ok());
            assert!(module.getattr("PageRankAnalyzer").is_ok());
            assert!(module.getattr("PatternMatcher").is_ok());
            
            // Test functions are available
            assert!(module.getattr("create_repository").is_ok());
            assert!(module.getattr("get_default_weights").is_ok());
            assert!(module.getattr("create_pagerank_analyzer").is_ok());
            assert!(module.getattr("validate_pattern").is_ok());
            
            // Test exceptions are available
            assert!(module.getattr("ScribeException").is_ok());
            assert!(module.getattr("AnalysisException").is_ok());
            assert!(module.getattr("PatternException").is_ok());
            assert!(module.getattr("ConfigurationException").is_ok());
        });
    }

    #[test]
    fn test_version_info() {
        Python::with_gil(|py| {
            let version_info = get_version_info().unwrap();
            let version_dict: &pyo3::types::PyDict = version_info.downcast(py).unwrap();
            
            assert!(version_dict.contains("version").unwrap());
            assert!(version_dict.contains("name").unwrap());
            assert!(version_dict.contains("description").unwrap());
        });
    }

    #[test]
    fn test_build_info() {
        Python::with_gil(|py| {
            let build_info = get_build_info().unwrap();
            let build_dict: &pyo3::types::PyDict = build_info.downcast(py).unwrap();
            
            assert!(build_dict.contains("version").unwrap());
            assert!(build_dict.contains("name").unwrap());
        });
    }

    #[test]
    fn test_initialize_async_runtime() {
        let result = initialize_async_runtime();
        assert!(result.is_ok());
    }
}
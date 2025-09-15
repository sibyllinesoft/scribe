//! Python bindings for Scribe configuration management.
//!
//! This module exposes the Rust `Config` struct to Python, ensuring that the Rust
//! configuration system is the single source of truth for all settings.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::path::PathBuf;
use std::collections::HashMap;

use scribe_core::config::{
    Config as RustConfig, 
    GeneralConfig, FilteringConfig, AnalysisConfig, 
    ScoringConfig, PerformanceConfig, GitConfig, 
    FeatureFlags, OutputConfig
};
use crate::error::ToPyResult;

/// Python wrapper for the Rust Config struct
#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyConfig {
    inner: RustConfig,
}

#[pymethods]
impl PyConfig {
    /// Create a new Config with default values
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustConfig::default(),
        }
    }

    /// Load configuration from a file
    #[staticmethod]
    pub fn load_from_file(path: &str) -> PyResult<Self> {
        let config = RustConfig::load_from_file(path).to_py_result()?;
        Ok(Self { inner: config })
    }

    /// Save configuration to a file
    pub fn save_to_file(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_file(path).to_py_result()
    }

    /// Validate the configuration
    pub fn validate(&self) -> PyResult<()> {
        self.inner.validate().to_py_result()
    }

    /// Compute a hash of the configuration for cache invalidation
    pub fn compute_hash(&self) -> String {
        self.inner.compute_hash()
    }

    /// Get verbosity level (0-4)
    #[getter]
    pub fn verbosity(&self) -> u8 {
        self.inner.general.verbosity
    }

    /// Set verbosity level (0-4)
    #[setter]
    pub fn set_verbosity(&mut self, value: u8) {
        self.inner.general.verbosity = value;
    }

    /// Get show progress flag
    #[getter]
    pub fn show_progress(&self) -> bool {
        self.inner.general.show_progress
    }

    /// Set show progress flag
    #[setter]
    pub fn set_show_progress(&mut self, value: bool) {
        self.inner.general.show_progress = value;
    }

    /// Get use colors flag
    #[getter]
    pub fn use_colors(&self) -> bool {
        self.inner.general.use_colors
    }

    /// Set use colors flag
    #[setter]
    pub fn set_use_colors(&mut self, value: bool) {
        self.inner.general.use_colors = value;
    }

    /// Get max threads setting
    #[getter]
    pub fn max_threads(&self) -> usize {
        self.inner.general.max_threads
    }

    /// Set max threads setting (0 = auto-detect)
    #[setter]
    pub fn set_max_threads(&mut self, value: usize) {
        self.inner.general.max_threads = value;
    }

    /// Get working directory
    #[getter]
    pub fn working_dir(&self) -> Option<String> {
        self.inner.general.working_dir.as_ref().map(|p| p.to_string_lossy().to_string())
    }

    /// Set working directory
    #[setter]
    pub fn set_working_dir(&mut self, value: Option<&str>) {
        self.inner.general.working_dir = value.map(PathBuf::from);
    }

    /// Get maximum file size for analysis
    #[getter]
    pub fn max_file_size(&self) -> u64 {
        self.inner.filtering.max_file_size
    }

    /// Set maximum file size for analysis
    #[setter]
    pub fn set_max_file_size(&mut self, value: u64) {
        self.inner.filtering.max_file_size = value;
    }

    /// Enable/disable dependency analysis
    #[getter]
    pub fn enable_dependency_analysis(&self) -> bool {
        self.inner.analysis.enable_dependency_analysis
    }

    /// Set dependency analysis flag
    #[setter]
    pub fn set_enable_dependency_analysis(&mut self, value: bool) {
        self.inner.analysis.enable_dependency_analysis = value;
    }

    /// Get PageRank damping factor
    #[getter]
    pub fn pagerank_damping(&self) -> f64 {
        self.inner.scoring.pagerank_damping
    }

    /// Set PageRank damping factor
    #[setter]
    pub fn set_pagerank_damping(&mut self, value: f64) {
        self.inner.scoring.pagerank_damping = value;
    }

    /// Create a deep copy of the configuration
    pub fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Merge with another configuration (other takes priority)
    pub fn merge_with(&self, other: &PyConfig) -> Self {
        Self {
            inner: self.inner.clone().merge_with(other.inner.clone()),
        }
    }

    /// Update configuration from a Python dictionary
    pub fn update_from_dict(&mut self, py_dict: &PyDict) -> PyResult<()> {
        // General settings
        if let Some(verbosity) = py_dict.get_item("verbosity")? {
            self.inner.general.verbosity = verbosity.extract()?;
        }
        if let Some(show_progress) = py_dict.get_item("show_progress")? {
            self.inner.general.show_progress = show_progress.extract()?;
        }
        if let Some(use_colors) = py_dict.get_item("use_colors")? {
            self.inner.general.use_colors = use_colors.extract()?;
        }
        if let Some(max_threads) = py_dict.get_item("max_threads")? {
            self.inner.general.max_threads = max_threads.extract()?;
        }
        if let Some(working_dir) = py_dict.get_item("working_dir")? {
            let path_str: Option<String> = working_dir.extract()?;
            self.inner.general.working_dir = path_str.map(PathBuf::from);
        }

        // Filtering settings
        if let Some(max_file_size) = py_dict.get_item("max_file_size")? {
            self.inner.filtering.max_file_size = max_file_size.extract()?;
        }

        // Analysis settings
        if let Some(enable_deps) = py_dict.get_item("enable_dependency_analysis")? {
            self.inner.analysis.enable_dependency_analysis = enable_deps.extract()?;
        }

        // Scoring settings
        if let Some(damping) = py_dict.get_item("pagerank_damping")? {
            self.inner.scoring.pagerank_damping = damping.extract()?;
        }

        Ok(())
    }

    /// Convert configuration to a Python dictionary
    pub fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        // General settings
        dict.set_item("verbosity", self.inner.general.verbosity)?;
        dict.set_item("show_progress", self.inner.general.show_progress)?;
        dict.set_item("use_colors", self.inner.general.use_colors)?;
        dict.set_item("max_threads", self.inner.general.max_threads)?;
        dict.set_item("working_dir", self.inner.general.working_dir.as_ref().map(|p| p.to_string_lossy().to_string()))?;

        // Filtering settings
        dict.set_item("max_file_size", self.inner.filtering.max_file_size)?;

        // Analysis settings
        dict.set_item("enable_dependency_analysis", self.inner.analysis.enable_dependency_analysis)?;

        // Scoring settings
        dict.set_item("pagerank_damping", self.inner.scoring.pagerank_damping)?;

        Ok(dict.into())
    }

    /// String representation for debugging
    pub fn __repr__(&self) -> String {
        format!("Config(verbosity={}, max_threads={}, hash={})", 
                self.inner.general.verbosity, 
                self.inner.general.max_threads,
                self.inner.compute_hash())
    }
}

impl PyConfig {
    /// Get the inner Rust config (for internal use)
    pub fn inner(&self) -> &RustConfig {
        &self.inner
    }

    /// Get mutable reference to inner Rust config (for internal use)
    pub fn inner_mut(&mut self) -> &mut RustConfig {
        &mut self.inner
    }

    /// Create from Rust config (for internal use)
    pub fn from_rust_config(config: RustConfig) -> Self {
        Self { inner: config }
    }
}
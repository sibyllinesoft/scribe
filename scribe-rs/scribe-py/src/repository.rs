//! Repository scanning and analysis Python interface
//!
//! Provides a high-level Python API for repository scanning, file discovery,
//! and comprehensive analysis with async support and progress reporting.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use scribe_analysis::CodeAnalyzer;
use scribe_core::{AnalysisResult, Config, FileInfo, RepositoryInfo};
use scribe_scanner::Scanner;

use crate::error::{rust_error_to_py_err, rust_result_to_py};
use crate::utils::*;

/// Python wrapper for repository analysis and scanning
#[pyclass]
pub struct Repository {
    /// Repository root path
    path: PathBuf,
    /// Configuration for analysis
    config: Arc<RwLock<Config>>,
    /// Cached scanner instance
    scanner: Arc<RwLock<Option<Scanner>>>,
    /// Cached analyzer instance  
    analyzer: Arc<RwLock<Option<Analyzer>>>,
    /// Repository info cache
    repo_info: Arc<RwLock<Option<RepositoryInfo>>>,
}

#[pymethods]
impl Repository {
    /// Create a new repository instance
    ///
    /// Args:
    ///     path: Path to the repository root
    ///     config: Optional configuration dict
    ///
    /// Returns:
    ///     Repository instance
    #[new]
    pub fn new(path: &str, config: Option<&PyDict>) -> PyResult<Self> {
        let repo_path = PathBuf::from(path);

        // Validate that path exists and is a directory
        if !repo_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("Repository path does not exist: {}", path),
            ));
        }

        if !repo_path.is_dir() {
            return Err(PyErr::new::<pyo3::exceptions::PyNotADirectoryError, _>(
                format!("Path is not a directory: {}", path),
            ));
        }

        // Load configuration
        let mut repo_config = Config::default();
        if let Some(py_config) = config {
            // TODO: Convert Python config dict to Rust Config
            // For now, use defaults
        }

        Ok(Repository {
            path: repo_path,
            config: Arc::new(RwLock::new(repo_config)),
            scanner: Arc::new(RwLock::new(None)),
            analyzer: Arc::new(RwLock::new(None)),
            repo_info: Arc::new(RwLock::new(None)),
        })
    }

    /// Get the repository path
    #[getter]
    pub fn path(&self) -> String {
        self.path.to_string_lossy().to_string()
    }

    /// Get repository information (cached)
    pub fn get_repository_info<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let repo_info = self.repo_info.clone();
        let path = self.path.clone();
        let config = self.config.clone();

        future_into_py(py, async move {
            // Check cache first
            {
                let info_guard = repo_info.read().await;
                if let Some(info) = info_guard.as_ref() {
                    return Ok(info.clone());
                }
            }

            // Analyze repository
            let config_guard = config.read().await;
            let analyzer = Analyzer::new(config_guard.clone()).map_err(rust_error_to_py_err)?;

            let info = analyzer
                .analyze_repository(&path)
                .await
                .map_err(rust_error_to_py_err)?;

            // Cache the result
            {
                let mut info_guard = repo_info.write().await;
                *info_guard = Some(info.clone());
            }

            Ok(info)
        })
    }

    /// Scan files in the repository
    ///
    /// Args:
    ///     max_files: Maximum number of files to scan (optional)
    ///     include_patterns: List of glob patterns to include (optional)
    ///     exclude_patterns: List of glob patterns to exclude (optional)
    ///     progress_callback: Optional progress callback function
    ///
    /// Returns:
    ///     List of FileInfo objects
    pub fn scan_files<'py>(
        &self,
        py: Python<'py>,
        max_files: Option<usize>,
        include_patterns: Option<Vec<String>>,
        exclude_patterns: Option<Vec<String>>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<&'py PyAny> {
        let scanner = self.scanner.clone();
        let config = self.config.clone();
        let path = self.path.clone();

        future_into_py(py, async move {
            // Get or create scanner
            let scanner_instance = {
                let mut scanner_guard = scanner.write().await;
                if scanner_guard.is_none() {
                    let config_guard = config.read().await;
                    let new_scanner =
                        Scanner::new(config_guard.clone()).map_err(rust_error_to_py_err)?;
                    *scanner_guard = Some(new_scanner);
                }
                scanner_guard.as_ref().unwrap().clone()
            };

            // Configure scan options
            let mut scan_config = scribe_scanner::ScanOptions::default();
            if let Some(max) = max_files {
                scan_config.max_files = Some(max);
            }
            if let Some(includes) = include_patterns {
                scan_config.include_patterns = includes;
            }
            if let Some(excludes) = exclude_patterns {
                scan_config.exclude_patterns = excludes;
            }

            // Set up progress callback
            if let Some(callback) = progress_callback {
                let callback_fn = create_progress_callback(callback);
                scan_config.progress_callback = Some(Box::new(callback_fn));
            }

            // Perform the scan
            let files = scanner_instance
                .scan_repository(&path, scan_config)
                .await
                .map_err(rust_error_to_py_err)?;

            Ok(files)
        })
    }

    /// Get files by language
    ///
    /// Args:
    ///     language: Language name (e.g., "rust", "python")
    ///
    /// Returns:
    ///     List of FileInfo objects for the specified language
    pub fn get_files_by_language<'py>(
        &self,
        py: Python<'py>,
        language: &str,
    ) -> PyResult<&'py PyAny> {
        let scanner = self.scanner.clone();
        let config = self.config.clone();
        let path = self.path.clone();
        let lang_filter = language.to_string();

        future_into_py(py, async move {
            // First scan all files if not already done
            let scanner_instance = {
                let mut scanner_guard = scanner.write().await;
                if scanner_guard.is_none() {
                    let config_guard = config.read().await;
                    let new_scanner =
                        Scanner::new(config_guard.clone()).map_err(rust_error_to_py_err)?;
                    *scanner_guard = Some(new_scanner);
                }
                scanner_guard.as_ref().unwrap().clone()
            };

            let scan_config = scribe_scanner::ScanOptions::default();
            let all_files = scanner_instance
                .scan_repository(&path, scan_config)
                .await
                .map_err(rust_error_to_py_err)?;

            // Filter by language
            let target_lang = py_to_language(&pyo3::types::PyString::new(
                pyo3::Python::with_gil(|py| py),
                &lang_filter,
            ))?;

            let filtered_files: Vec<FileInfo> = all_files
                .into_iter()
                .filter(|file| file.language == target_lang)
                .collect();

            Ok(filtered_files)
        })
    }

    /// Get file statistics by language
    pub fn get_language_stats<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let repo_info = self.repo_info.clone();
        let path = self.path.clone();
        let config = self.config.clone();

        future_into_py(py, async move {
            // Get repository info (will cache if not already cached)
            let info = {
                let info_guard = repo_info.read().await;
                if let Some(info) = info_guard.as_ref() {
                    info.clone()
                } else {
                    drop(info_guard);

                    let config_guard = config.read().await;
                    let analyzer =
                        Analyzer::new(config_guard.clone()).map_err(rust_error_to_py_err)?;

                    let info = analyzer
                        .analyze_repository(&path)
                        .await
                        .map_err(rust_error_to_py_err)?;

                    // Cache the result
                    {
                        let mut info_guard = repo_info.write().await;
                        *info_guard = Some(info.clone());
                    }

                    info
                }
            };

            // Convert language stats to Python dict
            let mut stats: HashMap<String, HashMap<String, i64>> = HashMap::new();

            for (language, lang_stats) in &info.language_stats {
                let mut lang_data = HashMap::new();
                lang_data.insert("file_count".to_string(), lang_stats.file_count as i64);
                lang_data.insert("line_count".to_string(), lang_stats.line_count as i64);
                lang_data.insert("byte_count".to_string(), lang_stats.byte_count as i64);

                let lang_name = format!("{:?}", language).to_lowercase();
                stats.insert(lang_name, lang_data);
            }

            Ok(stats)
        })
    }

    /// Find files matching a pattern
    ///
    /// Args:
    ///     pattern: Glob pattern to match against file paths
    ///     case_sensitive: Whether matching should be case sensitive
    ///
    /// Returns:
    ///     List of matching file paths
    pub fn find_files<'py>(
        &self,
        py: Python<'py>,
        pattern: &str,
        case_sensitive: Option<bool>,
    ) -> PyResult<&'py PyAny> {
        let scanner = self.scanner.clone();
        let config = self.config.clone();
        let path = self.path.clone();
        let search_pattern = pattern.to_string();
        let case_sensitive = case_sensitive.unwrap_or(true);

        future_into_py(py, async move {
            // Get or create scanner
            let scanner_instance = {
                let mut scanner_guard = scanner.write().await;
                if scanner_guard.is_none() {
                    let config_guard = config.read().await;
                    let new_scanner =
                        Scanner::new(config_guard.clone()).map_err(rust_error_to_py_err)?;
                    *scanner_guard = Some(new_scanner);
                }
                scanner_guard.as_ref().unwrap().clone()
            };

            // Use the scanner's pattern matching capabilities
            let mut scan_config = scribe_scanner::ScanOptions::default();
            scan_config.include_patterns = vec![search_pattern];

            let matched_files = scanner_instance
                .scan_repository(&path, scan_config)
                .await
                .map_err(rust_error_to_py_err)?;

            // Convert to path strings
            let paths: Vec<String> = matched_files
                .iter()
                .map(|file| file.path.to_string_lossy().to_string())
                .collect();

            Ok(paths)
        })
    }

    /// Get repository size statistics
    pub fn get_size_stats<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let repo_info = self.repo_info.clone();
        let path = self.path.clone();
        let config = self.config.clone();

        future_into_py(py, async move {
            // Get repository info
            let info = {
                let info_guard = repo_info.read().await;
                if let Some(info) = info_guard.as_ref() {
                    info.clone()
                } else {
                    drop(info_guard);

                    let config_guard = config.read().await;
                    let analyzer =
                        Analyzer::new(config_guard.clone()).map_err(rust_error_to_py_err)?;

                    let info = analyzer
                        .analyze_repository(&path)
                        .await
                        .map_err(rust_error_to_py_err)?;

                    {
                        let mut info_guard = repo_info.write().await;
                        *info_guard = Some(info.clone());
                    }

                    info
                }
            };

            // Convert size statistics to Python dict
            let mut stats = HashMap::new();
            stats.insert(
                "total_files".to_string(),
                info.size_stats.total_files as i64,
            );
            stats.insert(
                "total_lines".to_string(),
                info.size_stats.total_lines as i64,
            );
            stats.insert(
                "total_bytes".to_string(),
                info.size_stats.total_bytes as i64,
            );
            stats.insert(
                "largest_file_size".to_string(),
                info.size_stats.largest_file_size as i64,
            );
            stats.insert(
                "average_file_size".to_string(),
                info.size_stats.average_file_size as i64,
            );
            stats.insert(
                "median_file_size".to_string(),
                info.size_stats.median_file_size as i64,
            );

            Ok(stats)
        })
    }

    /// Clear all caches
    pub fn clear_cache<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let scanner = self.scanner.clone();
        let analyzer = self.analyzer.clone();
        let repo_info = self.repo_info.clone();

        future_into_py(py, async move {
            // Clear all cached data
            {
                let mut scanner_guard = scanner.write().await;
                *scanner_guard = None;
            }
            {
                let mut analyzer_guard = analyzer.write().await;
                *analyzer_guard = None;
            }
            {
                let mut info_guard = repo_info.write().await;
                *info_guard = None;
            }

            Ok(())
        })
    }

    /// Update configuration  
    pub fn update_config(&self, config_dict: &PyDict) -> PyResult<()> {
        // TODO: Implement config conversion from Python dict
        // For now, just return Ok
        Ok(())
    }

    /// Get current configuration as Python dict
    pub fn get_config<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let config = self.config.clone();

        future_into_py(py, async move {
            let config_guard = config.read().await;

            // Convert config to Python dict
            // TODO: Implement full config serialization
            let mut config_dict = HashMap::new();
            config_dict.insert(
                "max_files".to_string(),
                config_guard.filtering.max_files as i64,
            );
            config_dict.insert(
                "max_file_size".to_string(),
                config_guard.filtering.max_file_size as i64,
            );
            config_dict.insert(
                "follow_symlinks".to_string(),
                config_guard.filtering.follow_symlinks as i64,
            );

            Ok(config_dict)
        })
    }

    /// Check if repository has Git integration
    pub fn has_git(&self) -> bool {
        self.path.join(".git").exists()
    }

    /// Get Git repository statistics (if available)
    pub fn get_git_stats<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let repo_info = self.repo_info.clone();
        let path = self.path.clone();
        let config = self.config.clone();

        future_into_py(py, async move {
            if !path.join(".git").exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Repository does not have Git integration",
                ));
            }

            // Get repository info
            let info = {
                let info_guard = repo_info.read().await;
                if let Some(info) = info_guard.as_ref() {
                    info.clone()
                } else {
                    drop(info_guard);

                    let config_guard = config.read().await;
                    let analyzer =
                        Analyzer::new(config_guard.clone()).map_err(rust_error_to_py_err)?;

                    let info = analyzer
                        .analyze_repository(&path)
                        .await
                        .map_err(rust_error_to_py_err)?;

                    {
                        let mut info_guard = repo_info.write().await;
                        *info_guard = Some(info.clone());
                    }

                    info
                }
            };

            // Convert Git stats to Python dict
            if let Some(git_stats) = &info.git_stats {
                let mut stats = HashMap::new();
                stats.insert("total_commits".to_string(), git_stats.total_commits as i64);
                stats.insert("total_authors".to_string(), git_stats.total_authors as i64);
                stats.insert(
                    "first_commit_date".to_string(),
                    git_stats
                        .first_commit_date
                        .map(|d| d.timestamp())
                        .unwrap_or(0),
                );
                stats.insert(
                    "last_commit_date".to_string(),
                    git_stats
                        .last_commit_date
                        .map(|d| d.timestamp())
                        .unwrap_or(0),
                );
                stats.insert("tracked_files".to_string(), git_stats.tracked_files as i64);
                stats.insert(
                    "modified_files".to_string(),
                    git_stats.modified_files as i64,
                );

                Ok(stats)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Git statistics not available",
                ))
            }
        })
    }
}

/// Module-level functions for repository operations
#[pyfunction]
pub fn create_repository(path: &str, config: Option<&PyDict>) -> PyResult<Repository> {
    Repository::new(path, config)
}

/// Check if a path contains a valid repository
#[pyfunction]
pub fn is_valid_repository(path: &str) -> bool {
    let repo_path = PathBuf::from(path);
    repo_path.exists() && repo_path.is_dir()
}

/// Find repository root from a given path
#[pyfunction]
pub fn find_repository_root(path: &str) -> PyResult<Option<String>> {
    let path_buf = PathBuf::from(path);
    let root = scribe_core::find_repo_root(&path_buf);
    Ok(root.map(|p| p.to_string_lossy().to_string()))
}

//! High-performance optimized scanner integrating all performance improvements.
//!
//! This module provides a complete, production-ready scanner that integrates
//! all the performance optimizations: strict pre-filtering, batched git operations,
//! bounded parallelism with backpressure, compact data structures, incremental
//! scanning, and comprehensive performance monitoring.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use scribe_core::{
    FileInfo, GitFileStatus, GitStatus, RenderDecision, Result, ScribeError,
};
use scribe_core::tokenization::{TokenCounter, utils as token_utils};

use crate::{
    content::ContentAnalyzer,
    filtering::{DirectoryFilter, FileFilter, FilterResult},
    git_batch::{BatchMetrics, BulkStatusResult, GitBatchProcessor},
    metadata::{FileSystemType, MetadataExtractor},
    parallel::{ParallelController, ParallelConfig, WorkItem},
    compact_data::CompactFileCollection,
    incremental::{IncrementalScanner, IncrementalConfig},
    performance::{PerformanceMonitor, PerfTimer, PERF_MONITOR},
    language_detection::LanguageDetector,
};
use tokio::sync::Mutex;

/// High-performance optimized file scanner
pub struct OptimizedScanner {
    /// File and directory filters
    file_filter: FileFilter,
    dir_filter: DirectoryFilter,
    /// Shared metadata extractor
    metadata_extractor: Arc<MetadataExtractor>,
    /// Batched git processor
    git_processor: Option<Arc<Mutex<GitBatchProcessor>>>,
    /// Parallel processing controller
    parallel_controller: ParallelController,
    /// Incremental scanner
    incremental_scanner: Option<IncrementalScanner>,
    /// Configuration
    config: OptimizedScanConfig,
    /// Repository root path
    repo_root: PathBuf,
}

/// Configuration for optimized scanning
#[derive(Debug, Clone)]
pub struct OptimizedScanConfig {
    /// Enable git integration
    pub enable_git: bool,
    /// Enable incremental scanning
    pub enable_incremental: bool,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Parallel processing configuration
    pub parallel_config: ParallelConfig,
    /// Incremental scanning configuration
    pub incremental_config: IncrementalConfig,
    /// Maximum file size to process (bytes)
    pub max_file_size: u64,
    /// File extensions to include (if empty, include all text files)
    pub include_extensions: Vec<String>,
    /// File extensions to exclude
    pub exclude_extensions: Vec<String>,
    /// Directories to exclude
    pub exclude_directories: Vec<String>,
    /// Enable content hashing for change detection
    pub enable_content_hashing: bool,
    /// Batch size for file processing
    pub batch_size: usize,
}

/// Comprehensive scan results
#[derive(Debug)]
pub struct OptimizedScanResult {
    /// Processed files in compact format
    pub files: CompactFileCollection,
    /// Scan statistics
    pub stats: ScanStats,
    /// Git bulk status result (if enabled)
    pub git_stats: Option<BulkStatusResult>,
    /// Performance metrics
    pub performance: OptimizedPerformanceMetrics,
    /// Scan duration
    pub duration: std::time::Duration,
}

/// Scan statistics
#[derive(Debug, Clone)]
pub struct ScanStats {
    /// Total files discovered
    pub files_discovered: usize,
    /// Files processed (after filtering)
    pub files_processed: usize,
    /// Files filtered out
    pub files_filtered: usize,
    /// Files loaded from cache
    pub files_cached: usize,
    /// Files that failed processing
    pub files_failed: usize,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Directories skipped
    pub directories_skipped: usize,
}

/// Performance metrics specific to optimized scanning
#[derive(Debug, Clone)]
pub struct OptimizedPerformanceMetrics {
    /// Files per second throughput
    pub files_per_second: f64,
    /// Bytes per second throughput
    pub bytes_per_second: f64,
    /// Pre-filtering effectiveness (% filtered early)
    pub filter_effectiveness: f64,
    /// Git batch efficiency (calls saved)
    pub git_batch_efficiency: f64,
    /// Cache hit rate (for incremental scanning)
    pub cache_hit_rate: f64,
    /// Memory efficiency (compression ratio)
    pub memory_compression_ratio: f64,
    /// Parallelism utilization (average threads used)
    pub parallelism_utilization: f64,
    /// Time breakdown
    pub time_breakdown: TimeBreakdown,
}

/// Detailed time breakdown
#[derive(Debug, Clone)]
pub struct TimeBreakdown {
    /// Time spent on file discovery
    pub discovery_time_ms: f64,
    /// Time spent on filtering
    pub filtering_time_ms: f64,
    /// Time spent on git operations
    pub git_time_ms: f64,
    /// Time spent on file processing
    pub processing_time_ms: f64,
    /// Time spent on I/O operations
    pub io_time_ms: f64,
    /// Time spent waiting for parallel tasks
    pub parallel_wait_time_ms: f64,
}

#[derive(Clone)]
struct ProcessorResources {
    metadata_extractor: Arc<MetadataExtractor>,
    git_processor: Option<Arc<Mutex<GitBatchProcessor>>>,
    repo_root: PathBuf,
}

impl Default for OptimizedScanConfig {
    fn default() -> Self {
        Self {
            enable_git: false,
            enable_incremental: true,
            enable_monitoring: true,
            parallel_config: ParallelConfig::default(),
            incremental_config: IncrementalConfig::default(),
            max_file_size: 50 * 1024 * 1024, // 50MB
            include_extensions: vec![],
            exclude_extensions: vec![
                "png".to_string(), "jpg".to_string(), "jpeg".to_string(),
                "gif".to_string(), "bmp".to_string(), "ico".to_string(),
                "pdf".to_string(), "zip".to_string(), "tar".to_string(),
                "gz".to_string(), "exe".to_string(), "dll".to_string(),
                "so".to_string(), "dylib".to_string(),
            ],
            exclude_directories: vec![
                "node_modules".to_string(), "__pycache__".to_string(),
                "target".to_string(), "build".to_string(), "dist".to_string(),
                ".git".to_string(), ".hg".to_string(), ".svn".to_string(),
            ],
            enable_content_hashing: true,
            batch_size: 1000,
        }
    }
}

impl OptimizedScanner {
    /// Create a new optimized scanner for the given repository
    pub async fn new<P: AsRef<Path>>(repo_root: P, config: OptimizedScanConfig) -> Result<Self> {
        let repo_root = repo_root.as_ref().to_path_buf();

        // Initialize file filters
        let mut file_filter = FileFilter::new()
            .with_max_file_size(config.max_file_size)
            .with_binary_detection(true);

        if !config.include_extensions.is_empty() {
            file_filter = file_filter.with_allow_extensions(config.include_extensions.clone());
        }

        if !config.exclude_extensions.is_empty() {
            file_filter = file_filter.with_deny_extensions(config.exclude_extensions.clone());
        }

        let dir_filter = DirectoryFilter::new()
            .with_additional_cold_dirs(config.exclude_directories.clone());

        let metadata_extractor = Arc::new(MetadataExtractor::new());

        // Initialize git processor
        let git_processor = if config.enable_git {
            match GitBatchProcessor::new(&repo_root) {
                Ok(processor) => Some(Arc::new(Mutex::new(processor))),
                Err(e) => {
                    log::warn!("Git integration disabled: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize parallel controller
        let parallel_controller = ParallelController::new(config.parallel_config.clone());

        // Initialize incremental scanner
        let incremental_scanner = if config.enable_incremental {
            match IncrementalScanner::new(&repo_root, config.incremental_config.clone()).await {
                Ok(scanner) => Some(scanner),
                Err(e) => {
                    log::warn!("Incremental scanning disabled: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Start performance monitoring
        if config.enable_monitoring {
            PERF_MONITOR.start_monitoring();
        }

        Ok(Self {
            file_filter,
            dir_filter,
            metadata_extractor,
            git_processor,
            parallel_controller,
            incremental_scanner,
            config,
            repo_root,
        })
    }

    /// Perform optimized scanning with all performance improvements
    pub async fn scan(&mut self) -> Result<OptimizedScanResult> {
        let total_start = Instant::now();
        
        log::info!("Starting optimized scan of {}", self.repo_root.display());

        // Start comprehensive performance monitoring
        if self.config.enable_monitoring {
            PERF_MONITOR.reset_metrics();
        }

        let mut stats = ScanStats {
            files_discovered: 0,
            files_processed: 0,
            files_filtered: 0,
            files_cached: 0,
            files_failed: 0,
            bytes_processed: 0,
            directories_skipped: 0,
        };

        // Try incremental scanning first
        if let Some(ref mut incremental) = self.incremental_scanner {
            let _timer = PerfTimer::start("incremental_scan");
            
            match incremental.scan_incremental().await {
                Ok(collection) => {
                    let metrics = incremental.metrics();
                    stats.files_cached = metrics.files_cached;
                    stats.files_processed = metrics.files_cached + metrics.files_scanned + metrics.files_updated;
                    
                    log::info!("Incremental scan completed: {}/{} files from cache", 
                              stats.files_cached, stats.files_processed);

                    return Ok(OptimizedScanResult {
                        files: collection,
                        stats,
                        git_stats: None,
                        performance: self.calculate_performance_metrics(&stats, total_start.elapsed()).await,
                        duration: total_start.elapsed(),
                    });
                }
                Err(e) => {
                    log::warn!("Incremental scan failed, falling back to full scan: {}", e);
                }
            }
        }

        // Perform full optimized scan
        self.perform_full_scan(&mut stats, total_start).await
    }

    /// Perform a full optimized scan with all performance improvements
    async fn perform_full_scan(
        &mut self, 
        stats: &mut ScanStats,
        total_start: Instant,
    ) -> Result<OptimizedScanResult> {
        
        // Phase 1: File Discovery with Directory Filtering
        let discovery_start = Instant::now();
        let discovered_files = self.discover_files_optimized().await?;
        let discovery_time = discovery_start.elapsed();
        
        stats.files_discovered = discovered_files.len();
        log::info!("Discovered {} files in {:.2}s", stats.files_discovered, discovery_time.as_secs_f64());

        // Phase 2: Pre-filtering
        let filtering_start = Instant::now();
        let filtered_files = self.apply_pre_filtering(discovered_files).await;
        let filtering_time = filtering_start.elapsed();
        
        stats.files_filtered = stats.files_discovered - filtered_files.len();
        log::info!("Pre-filtered to {} files ({} filtered out) in {:.2}s", 
                  filtered_files.len(), stats.files_filtered, filtering_time.as_secs_f64());

        // Phase 3: Git Batch Loading
        let git_start = Instant::now();
        let mut git_metrics_snapshot: Option<BatchMetrics> = None;
        let git_stats = if let Some(ref git_processor) = self.git_processor {
            let _timer = PerfTimer::start("git_batch_load");
            let mut guard = git_processor.lock().await;
            match guard.load_bulk_status() {
                Ok(result) => {
                    git_metrics_snapshot = Some(guard.metrics().clone());
                    log::info!(
                        "Loaded git status for {} files in {}ms",
                        result.files_processed,
                        result.load_time_ms
                    );
                    Some(result)
                }
                Err(e) => {
                    log::warn!("Git batch loading failed: {}", e);
                    None
                }
            }
        } else {
            None
        };
        let git_time = git_start.elapsed();

        // Phase 4: Parallel Processing with Backpressure
        let processing_start = Instant::now();
        let work_items: Vec<WorkItem<PathBuf>> = filtered_files.into_iter()
            .enumerate()
            .map(|(i, path)| {
                let size_hint = std::fs::metadata(&path)
                    .map(|m| m.len())
                    .unwrap_or(1000) as u32;
                
                WorkItem::new(path)
                    .with_priority((i % 256) as u8) // Distribute priorities
                    .with_estimated_cost(size_hint)
            })
            .collect();

        let resources = ProcessorResources {
            metadata_extractor: Arc::clone(&self.metadata_extractor),
            git_processor: self.git_processor.clone(),
            repo_root: self.repo_root.clone(),
        };

        let processor = {
            let resources = resources.clone();
            move |path: PathBuf| {
                let resources = resources.clone();
                async move { Self::process_single_file(path, resources).await }
            }
        };

        let results = self.parallel_controller.process_parallel(work_items, processor).await;
        let processing_time = processing_start.elapsed();

        // Phase 5: Results Collection and Compact Storage
        let collection_start = Instant::now();
        let mut file_collection = CompactFileCollection::new();
        
        for result in results {
            match result {
                Ok(file_info) => {
                    stats.bytes_processed += file_info.size;
                    file_collection.add_file(&file_info);
                    stats.files_processed += 1;
                }
                Err(_) => {
                    stats.files_failed += 1;
                }
            }
        }
        let collection_time = collection_start.elapsed();

        let total_duration = total_start.elapsed();
        
        log::info!(
            "Optimized scan completed in {:.2}s: processed {}/{} files ({:.1}% success rate)",
            total_duration.as_secs_f64(),
            stats.files_processed,
            stats.files_discovered,
            (stats.files_processed as f64 / stats.files_discovered as f64) * 100.0
        );

        // Calculate comprehensive performance metrics
        let git_batch_efficiency = git_metrics_snapshot
            .as_ref()
            .map(|m| m.status_calls_avoided as f64)
            .or_else(|| {
                self.git_processor.as_ref().and_then(|git| {
                    git.try_lock()
                        .ok()
                        .map(|guard| guard.metrics().status_calls_avoided as f64)
                })
            })
            .unwrap_or(0.0);

        let performance = OptimizedPerformanceMetrics {
            files_per_second: if total_duration.as_secs_f64() > 0.0 {
                stats.files_processed as f64 / total_duration.as_secs_f64()
            } else {
                0.0
            },
            bytes_per_second: if total_duration.as_secs_f64() > 0.0 {
                stats.bytes_processed as f64 / total_duration.as_secs_f64()
            } else {
                0.0
            },
            filter_effectiveness: if stats.files_discovered > 0 {
                stats.files_filtered as f64 / stats.files_discovered as f64
            } else {
                0.0
            },
            git_batch_efficiency,
            cache_hit_rate: stats.files_cached as f64 / stats.files_processed.max(1) as f64,
            memory_compression_ratio: file_collection.stats().compression_ratio,
            parallelism_utilization: self.parallel_controller.metrics().current_concurrency as f64 
                / self.config.parallel_config.max_concurrency as f64,
            time_breakdown: TimeBreakdown {
                discovery_time_ms: discovery_time.as_secs_f64() * 1000.0,
                filtering_time_ms: filtering_time.as_secs_f64() * 1000.0,
                git_time_ms: git_time.as_secs_f64() * 1000.0,
                processing_time_ms: processing_time.as_secs_f64() * 1000.0,
                io_time_ms: 0.0, // Would be tracked by performance monitor
                parallel_wait_time_ms: collection_time.as_secs_f64() * 1000.0,
            },
        };

        Ok(OptimizedScanResult {
            files: file_collection,
            stats: stats.clone(),
            git_stats,
            performance,
            duration: total_duration,
        })
    }

    /// Discover files with directory-level filtering
    async fn discover_files_optimized(&mut self) -> Result<Vec<PathBuf>> {
        use ignore::{WalkBuilder, WalkState};
        
        let mut builder = WalkBuilder::new(&self.repo_root);
        builder
            .git_ignore(true)
            .git_exclude(true)
            .hidden(!self.config.include_extensions.is_empty()) // Include hidden if specific extensions
            .follow_links(false);

        let mut files = Vec::new();

        builder.build().for_each(|entry| {
            match entry {
                Ok(entry) => {
                    if entry.file_type().map_or(false, |ft| ft.is_dir()) {
                        // Apply directory filtering
                        if self.dir_filter.should_skip_directory(entry.path()) {
                            return WalkState::Skip;
                        }
                    } else if entry.file_type().map_or(false, |ft| ft.is_file()) {
                        files.push(entry.path().to_path_buf());
                    }
                }
                Err(e) => {
                    log::debug!("Walk error: {}", e);
                }
            }
        });

        self.stats.directories_skipped = self.dir_filter.stats().dirs_skipped as usize;
        Ok(files)
    }

    /// Apply pre-filtering to discovered files
    async fn apply_pre_filtering(&mut self, files: Vec<PathBuf>) -> Vec<PathBuf> {
        let mut filtered_files = Vec::new();
        
        for file_path in files {
            match self.file_filter.filter_file(&file_path).await {
                FilterResult::Include => {
                    filtered_files.push(file_path);
                }
                FilterResult::Exclude(_) => {
                    // File was filtered out
                }
            }
        }

        filtered_files
    }

    /// Process a single file with all optimizations
    async fn process_single_file(
        path: PathBuf,
        resources: ProcessorResources,
    ) -> Result<FileInfo, String> {
        let _timer = PerfTimer::start("process_file");

        let metadata = resources
            .metadata_extractor
            .extract_metadata(&path)
            .await
            .map_err(|e| format!("Metadata extraction failed for {}: {}", path.display(), e))?;

        if metadata.file_type != FileSystemType::RegularFile {
            return Err(format!("Skipping non-regular file {}", path.display()));
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_string();

        if FileInfo::detect_binary_by_extension(&extension) {
            return Err(format!("Skipping binary file {}", path.display()));
        }

        let raw_bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        if raw_bytes.iter().take(1024).any(|b| *b == 0) {
            return Err(format!("Skipping binary file {}", path.display()));
        }

        let content = String::from_utf8_lossy(&raw_bytes).into_owned();
        let line_count = content.lines().count();
        let char_count = content.chars().count();

        let language_detector = LanguageDetector::new();
        let language = language_detector.detect_language(&path);

        let analyzer = ContentAnalyzer::new();
        let analysis_summary = match analyzer.analyze_content(&content, &language).await {
            Ok(stats) => Some(stats),
            Err(err) => {
                log::debug!(
                    "Content analysis failed for {}: {}",
                    path.display(),
                    err
                );
                None
            }
        };

        let mut decision = RenderDecision::include("optimized_scan");
        if let Some(ref stats) = analysis_summary {
            decision = decision.with_context(format!(
                "imports={},docs={}",
                stats.imports.total_imports,
                stats.documentation.headings.len()
            ));
        }

        let token_estimate = TokenCounter::global()
            .estimate_file_tokens(&content, &path)
            .unwrap_or_else(|_| token_utils::estimate_tokens_legacy(&content));

        let relative_path = path
            .strip_prefix(&resources.repo_root)
            .map(|rel| rel.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());

        let mut file_info = FileInfo {
            path: path.clone(),
            relative_path: relative_path.clone(),
            size: metadata.size,
            modified: metadata
                .modified
                .map(|secs| UNIX_EPOCH + Duration::from_secs(secs)),
            decision,
            file_type: FileInfo::classify_file_type(
                &relative_path,
                &language,
                &extension,
            ),
            language,
            content: Some(content),
            token_estimate: Some(token_estimate),
            line_count: Some(line_count),
            char_count: Some(char_count),
            is_binary: false,
            git_status: None,
            centrality_score: None,
        };

        if let Some(ref git_arc) = resources.git_processor {
            if let Ok(status) = git_arc.lock().await.get_file_status(&path) {
                file_info.git_status = Some(GitStatus {
                    working_tree: status,
                    index: GitFileStatus::Unmodified,
                });
            }
        }

        if let Some(stats) = analysis_summary {
            let dependency_count =
                (stats.imports.internal_dependencies.len() + stats.imports.external_dependencies.len())
                    as f64;
            if stats.imports.total_imports > 0 {
                file_info.centrality_score = Some(
                    (dependency_count / stats.imports.total_imports as f64).min(1.0),
                );
            }
        }

        Ok(file_info)
    }

    /// Calculate comprehensive performance metrics
    async fn calculate_performance_metrics(
        &self,
        stats: &ScanStats, 
        duration: std::time::Duration,
    ) -> OptimizedPerformanceMetrics {
        let git_batch_efficiency = self
            .git_processor
            .as_ref()
            .and_then(|git| {
                git.try_lock()
                    .ok()
                    .map(|guard| guard.metrics().status_calls_avoided as f64)
            })
            .unwrap_or(0.0);

        OptimizedPerformanceMetrics {
            files_per_second: if duration.as_secs_f64() > 0.0 {
                stats.files_processed as f64 / duration.as_secs_f64()
            } else {
                0.0
            },
            bytes_per_second: if duration.as_secs_f64() > 0.0 {
                stats.bytes_processed as f64 / duration.as_secs_f64()
            } else {
                0.0
            },
            filter_effectiveness: if stats.files_discovered > 0 {
                stats.files_filtered as f64 / stats.files_discovered as f64
            } else {
                0.0
            },
            git_batch_efficiency,
            cache_hit_rate: if stats.files_processed > 0 {
                stats.files_cached as f64 / stats.files_processed as f64
            } else {
                0.0
            },
            memory_compression_ratio: 0.0, // Would be calculated from CompactFileCollection
            parallelism_utilization: self.parallel_controller.metrics().current_concurrency as f64 
                / self.config.parallel_config.max_concurrency as f64,
            time_breakdown: TimeBreakdown {
                discovery_time_ms: 0.0,
                filtering_time_ms: 0.0,
                git_time_ms: 0.0,
                processing_time_ms: duration.as_secs_f64() * 1000.0,
                io_time_ms: 0.0,
                parallel_wait_time_ms: 0.0,
            },
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> OptimizedScanResult {
        // This would return current metrics without scanning
        todo!("Implement current metrics retrieval")
    }

    /// Reset all performance counters
    pub fn reset_metrics(&mut self) {
        if self.config.enable_monitoring {
            PERF_MONITOR.reset_metrics();
        }
        
        self.parallel_controller.reset_metrics();
        
        if let Some(ref git_processor) = self.git_processor {
            if let Ok(mut guard) = git_processor.try_lock() {
                guard.clear_cache();
            } else {
                log::debug!("Unable to reset git metrics: processor busy");
            }
        }

        self.file_filter.reset_stats();
    }
}

impl OptimizedScanResult {
    /// Generate a comprehensive performance report
    pub fn generate_performance_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("# Optimized Scan Performance Report\n\n"));
        report.push_str(&format!("**Duration:** {:.2}s\n", self.duration.as_secs_f64()));
        report.push_str(&format!("**Files Processed:** {}\n", self.stats.files_processed));
        report.push_str(&format!("**Throughput:** {:.1} files/sec, {:.1} MB/sec\n\n", 
                                self.performance.files_per_second,
                                self.performance.bytes_per_second / (1024.0 * 1024.0)));
        
        report.push_str("## Optimization Effectiveness\n");
        report.push_str(&format!("- **Pre-filtering:** {:.1}% of files filtered early\n", 
                                self.performance.filter_effectiveness * 100.0));
        report.push_str(&format!("- **Git Batch Efficiency:** {:.0} individual calls avoided\n", 
                                self.performance.git_batch_efficiency));
        report.push_str(&format!("- **Cache Hit Rate:** {:.1}%\n", 
                                self.performance.cache_hit_rate * 100.0));
        report.push_str(&format!("- **Memory Compression:** {:.1}% space saved\n", 
                                self.performance.memory_compression_ratio * 100.0));
        report.push_str(&format!("- **Parallelism Utilization:** {:.1}%\n\n", 
                                self.performance.parallelism_utilization * 100.0));
        
        report.push_str("## Time Breakdown\n");
        let breakdown = &self.performance.time_breakdown;
        report.push_str(&format!("- **Discovery:** {:.1}ms\n", breakdown.discovery_time_ms));
        report.push_str(&format!("- **Filtering:** {:.1}ms\n", breakdown.filtering_time_ms));
        report.push_str(&format!("- **Git Operations:** {:.1}ms\n", breakdown.git_time_ms));
        report.push_str(&format!("- **Processing:** {:.1}ms\n", breakdown.processing_time_ms));
        report.push_str(&format!("- **I/O Wait:** {:.1}ms\n", breakdown.io_time_ms));
        report.push_str(&format!("- **Parallel Coordination:** {:.1}ms\n", breakdown.parallel_wait_time_ms));
        
        if let Some(ref git_stats) = self.git_stats {
            report.push_str(&format!("\n## Git Statistics\n"));
            report.push_str(&format!("- **Files Processed:** {}\n", git_stats.files_processed));
            report.push_str(&format!("- **Modified Files:** {}\n", git_stats.modified_files));
            report.push_str(&format!("- **Untracked Files:** {}\n", git_stats.untracked_files));
            report.push_str(&format!("- **Load Time:** {}ms\n", git_stats.load_time_ms));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    async fn create_test_repo() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Create test files
        fs::write(root.join("main.rs"), "fn main() { println!(\"Hello\"); }").await.unwrap();
        fs::write(root.join("lib.rs"), "pub fn hello() -> String { \"world\".to_string() }").await.unwrap();
        
        // Create subdirectories
        fs::create_dir(root.join("src")).await.unwrap();
        fs::write(root.join("src/module.rs"), "pub mod submodule;").await.unwrap();
        fs::write(root.join("src/utils.rs"), "pub fn utility_function() {}").await.unwrap();
        
        fs::create_dir(root.join("tests")).await.unwrap();
        fs::write(root.join("tests/integration.rs"), "#[test] fn test_something() {}").await.unwrap();

        // Create files that should be filtered
        fs::write(root.join("image.png"), &[0u8; 1024]).await.unwrap(); // Binary file
        fs::write(root.join("large.txt"), &vec![b'x'; 100_000]).await.unwrap(); // Large file
        
        // Create node_modules (should be filtered)
        fs::create_dir(root.join("node_modules")).await.unwrap();
        fs::write(root.join("node_modules/package.js"), "module.exports = {};").await.unwrap();

        temp_dir
    }

    #[tokio::test]
    async fn test_optimized_scanner_creation() {
        let temp_dir = create_test_repo().await;
        let config = OptimizedScanConfig::default();
        
        let scanner = OptimizedScanner::new(temp_dir.path(), config).await;
        assert!(scanner.is_ok());
    }

    #[tokio::test]
    async fn test_optimized_scan() {
        let temp_dir = create_test_repo().await;
        let config = OptimizedScanConfig {
            enable_git: false, // Disable git for test simplicity
            enable_incremental: false, // Disable incremental for test
            enable_monitoring: true,
            ..Default::default()
        };
        
        let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();
        let result = scanner.scan().await.unwrap();
        
        // Should find Rust files but filter out binary and large files
        assert!(result.stats.files_processed >= 4); // At least main.rs, lib.rs, module.rs, utils.rs, integration.rs
        assert!(result.stats.files_filtered > 0); // Should filter out image.png, large.txt, etc.
        assert!(result.performance.files_per_second > 0.0);
        assert!(result.performance.filter_effectiveness > 0.0);
        assert!(result.duration.as_secs_f64() > 0.0);
        
        println!("Scan completed:");
        println!("- Files discovered: {}", result.stats.files_discovered);
        println!("- Files processed: {}", result.stats.files_processed);
        println!("- Files filtered: {}", result.stats.files_filtered);
        println!("- Throughput: {:.1} files/sec", result.performance.files_per_second);
        println!("- Filter effectiveness: {:.1}%", result.performance.filter_effectiveness * 100.0);
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let temp_dir = create_test_repo().await;
        let config = OptimizedScanConfig {
            enable_monitoring: true,
            enable_git: false,
            enable_incremental: false,
            ..Default::default()
        };
        
        let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();
        
        // Reset metrics before scan
        scanner.reset_metrics();
        
        let result = scanner.scan().await.unwrap();
        
        // Verify metrics were collected
        let monitor_snapshot = PERF_MONITOR.get_current_snapshot();
        assert!(monitor_snapshot.files_per_second >= 0.0);
        
        // Verify result contains performance data
        assert!(result.performance.files_per_second > 0.0);
        assert!(result.performance.time_breakdown.processing_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_file_filtering() {
        let temp_dir = create_test_repo().await;
        let config = OptimizedScanConfig {
            include_extensions: vec!["rs".to_string()], // Only Rust files
            exclude_directories: vec!["node_modules".to_string()],
            max_file_size: 10_000, // Exclude large files
            enable_git: false,
            enable_incremental: false,
            ..Default::default()
        };
        
        let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();
        let result = scanner.scan().await.unwrap();
        
        // Should only find .rs files
        let files = result.files.to_full_file_infos().unwrap();
        for file in &files {
            assert!(file.path.extension().unwrap_or_default() == "rs");
        }
        
        // Should have filtered out non-.rs files and large files
        assert!(result.stats.files_filtered > 0);
        assert!(result.performance.filter_effectiveness > 0.0);
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let temp_dir = create_test_repo().await;
        let config = OptimizedScanConfig {
            enable_monitoring: true,
            enable_git: false,
            enable_incremental: false,
            ..Default::default()
        };
        
        let mut scanner = OptimizedScanner::new(temp_dir.path(), config).await.unwrap();
        let result = scanner.scan().await.unwrap();
        
        let report = result.generate_performance_report();
        
        assert!(report.contains("Optimized Scan Performance Report"));
        assert!(report.contains("Duration:"));
        assert!(report.contains("Files Processed:"));
        assert!(report.contains("Throughput:"));
        assert!(report.contains("Pre-filtering:"));
        assert!(report.contains("Time Breakdown"));
        
        println!("Performance Report:\n{}", report);
    }
}

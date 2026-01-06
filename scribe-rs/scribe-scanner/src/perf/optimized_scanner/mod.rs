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
    FileInfo, FileWeight, GitFileStatus, GitStatus, RenderDecision, Result, ScribeError,
};
use scribe_core::tokenization::{TokenCounter, utils as token_utils};

use crate::{
    core::filtering::{DirectoryFilter, FileFilter, FilterResult},
    core::metadata::{FileSystemType, MetadataExtractor},
    git::git_batch::{BatchMetrics, BulkStatusResult, GitBatchProcessor},
    perf::parallel::{ParallelController, ParallelConfig, WorkItem},
    perf::compact_data::CompactFileCollection,
    perf::incremental::{IncrementalScanner, IncrementalConfig},
    perf::performance::{PerformanceMonitor, PerfTimer, PERF_MONITOR},
    analysis::language_detection::LanguageDetector,
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
    /// Statistics (mutable during scan)
    stats: ScanStats,
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
#[derive(Debug, Clone, Default)]
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
            stats: ScanStats::default(),
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

        if FileInfo::detect_binary_with_hint(&path, extension.as_str()) {
            return Err(format!("Skipping binary file {}", path.display()));
        }

        let raw_bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        if FileInfo::detect_binary_from_bytes(&raw_bytes, Some(extension.as_str())) {
            return Err(format!("Skipping binary file {}", path.display()));
        }

        let content = String::from_utf8_lossy(&raw_bytes).into_owned();
        let line_count = content.lines().count();
        let char_count = content.chars().count();

        let language_detector = LanguageDetector::new();
        let language = language_detector.detect_language(&path);

        let decision = RenderDecision::include("optimized_scan");

        let token_estimate = TokenCounter::global()
            .estimate_file_tokens(&content, &path)
            .unwrap_or_else(|_| token_utils::estimate_tokens_legacy(&content));

        let relative_path = path
            .strip_prefix(&resources.repo_root)
            .map(|rel| rel.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());

        let file_info = FileInfo {
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
            weight: FileWeight::default(),
            centrality_score: None,
        };

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
mod tests;

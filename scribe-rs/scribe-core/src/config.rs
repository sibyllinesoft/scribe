//! Configuration types and management for Scribe.
//!
//! Provides comprehensive configuration structures with validation,
//! serialization, and environment-based overrides.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::error::{Result, ScribeError};
use crate::types::HeuristicWeights;
use crate::file::Language;

/// Main configuration structure for Scribe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// General settings
    pub general: GeneralConfig,
    
    /// File filtering configuration
    pub filtering: FilteringConfig,
    
    /// Analysis configuration
    pub analysis: AnalysisConfig,
    
    /// Scoring configuration
    pub scoring: ScoringConfig,
    
    /// Performance and resource limits
    pub performance: PerformanceConfig,
    
    /// Git integration settings
    pub git: GitConfig,
    
    /// Feature flags
    pub features: FeatureFlags,
    
    /// Output format configuration
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            filtering: FilteringConfig::default(),
            analysis: AnalysisConfig::default(),
            scoring: ScoringConfig::default(),
            performance: PerformanceConfig::default(),
            git: GitConfig::default(),
            features: FeatureFlags::default(),
            output: OutputConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ScribeError::path_with_source(
                "Failed to read config file", 
                path.as_ref(), 
                e
            ))?;

        let config: Config = match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::from_str(&content)?,
            Some("yaml") | Some("yml") => {
                return Err(ScribeError::config("YAML support not yet implemented"));
            }
            Some("toml") => {
                return Err(ScribeError::config("TOML support not yet implemented"));
            }
            _ => {
                return Err(ScribeError::config(
                    "Unsupported config file format. Use .json, .yaml, or .toml"
                ));
            }
        };

        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::to_string_pretty(self)?,
            Some("yaml") | Some("yml") => {
                return Err(ScribeError::config("YAML support not yet implemented"));
            }
            Some("toml") => {
                return Err(ScribeError::config("TOML support not yet implemented"));
            }
            _ => {
                return Err(ScribeError::config(
                    "Unsupported config file format. Use .json, .yaml, or .toml"
                ));
            }
        };

        std::fs::write(path.as_ref(), content)
            .map_err(|e| ScribeError::path_with_source(
                "Failed to write config file", 
                path.as_ref(), 
                e
            ))?;

        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        self.general.validate()?;
        self.filtering.validate()?;
        self.analysis.validate()?;
        self.scoring.validate()?;
        self.performance.validate()?;
        self.git.validate()?;
        self.features.validate()?;
        self.output.validate()?;
        Ok(())
    }

    /// Merge with another configuration (other takes priority)
    pub fn merge_with(mut self, other: Config) -> Self {
        // Note: This is a simplified merge - in practice you might want
        // more sophisticated merging logic
        self.general = other.general;
        self.filtering = other.filtering;
        self.analysis = other.analysis;
        self.scoring = other.scoring;
        self.performance = other.performance;
        self.git = other.git;
        self.features = other.features;
        self.output = other.output;
        self
    }

    /// Create a configuration hash for cache invalidation
    pub fn compute_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let serialized = serde_json::to_string(self).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        serialized.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// General application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Verbosity level (0-4)
    pub verbosity: u8,
    
    /// Enable progress reporting
    pub show_progress: bool,
    
    /// Enable colored output
    pub use_colors: bool,
    
    /// Maximum number of worker threads (0 = auto-detect)
    pub max_threads: usize,
    
    /// Working directory override
    pub working_dir: Option<PathBuf>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            verbosity: 1,
            show_progress: true,
            use_colors: true,
            max_threads: 0, // Auto-detect
            working_dir: None,
        }
    }
}

impl GeneralConfig {
    fn validate(&self) -> Result<()> {
        if self.verbosity > 4 {
            return Err(ScribeError::config_field(
                "Verbosity must be between 0 and 4", 
                "verbosity"
            ));
        }
        Ok(())
    }
}

/// File filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteringConfig {
    /// Include patterns (glob format)
    pub include_patterns: Vec<String>,
    
    /// Exclude patterns (glob format)
    pub exclude_patterns: Vec<String>,
    
    /// Maximum file size in bytes
    pub max_file_size: u64,
    
    /// Minimum file size in bytes
    pub min_file_size: u64,
    
    /// Languages to include (empty = all)
    pub include_languages: HashSet<Language>,
    
    /// Languages to exclude
    pub exclude_languages: HashSet<Language>,
    
    /// Whether to follow symbolic links
    pub follow_symlinks: bool,
    
    /// Whether to include hidden files (starting with .)
    pub include_hidden: bool,
    
    /// Whether to respect .gitignore files
    pub respect_gitignore: bool,
    
    /// Additional ignore files to respect
    pub ignore_files: Vec<PathBuf>,
}

impl Default for FilteringConfig {
    fn default() -> Self {
        Self {
            include_patterns: vec![],
            exclude_patterns: vec![
                "node_modules/**".to_string(),
                "target/**".to_string(),
                ".git/**".to_string(),
                "build/**".to_string(),
                "dist/**".to_string(),
                "__pycache__/**".to_string(),
                "*.pyc".to_string(),
                ".DS_Store".to_string(),
            ],
            max_file_size: 10 * 1024 * 1024, // 10MB
            min_file_size: 0,
            include_languages: HashSet::new(), // Empty = all languages
            exclude_languages: HashSet::new(),
            follow_symlinks: false,
            include_hidden: false,
            respect_gitignore: true,
            ignore_files: vec![],
        }
    }
}

impl FilteringConfig {
    fn validate(&self) -> Result<()> {
        if self.max_file_size < self.min_file_size {
            return Err(ScribeError::config(
                "max_file_size must be >= min_file_size"
            ));
        }

        // Validate glob patterns
        for pattern in &self.include_patterns {
            Glob::new(pattern).map_err(|e| ScribeError::pattern(
                format!("Invalid include pattern: {}", e), 
                pattern
            ))?;
        }

        for pattern in &self.exclude_patterns {
            Glob::new(pattern).map_err(|e| ScribeError::pattern(
                format!("Invalid exclude pattern: {}", e), 
                pattern
            ))?;
        }

        Ok(())
    }

    /// Build a GlobSet for include patterns
    pub fn build_include_set(&self) -> Result<Option<GlobSet>> {
        if self.include_patterns.is_empty() {
            return Ok(None);
        }

        let mut builder = GlobSetBuilder::new();
        for pattern in &self.include_patterns {
            builder.add(Glob::new(pattern)?);
        }
        Ok(Some(builder.build()?))
    }

    /// Build a GlobSet for exclude patterns
    pub fn build_exclude_set(&self) -> Result<GlobSet> {
        let mut builder = GlobSetBuilder::new();
        for pattern in &self.exclude_patterns {
            builder.add(Glob::new(pattern)?);
        }
        Ok(builder.build()?)
    }
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Whether to analyze file content (not just metadata)
    pub analyze_content: bool,
    
    /// Whether to compute token estimates
    pub compute_tokens: bool,
    
    /// Whether to count lines
    pub count_lines: bool,
    
    /// Whether to detect binary files by content (in addition to extension)
    pub detect_binary_content: bool,
    
    /// Languages that require special handling
    pub language_overrides: HashMap<String, Language>,
    
    /// Custom file type mappings (extension -> language)
    pub custom_extensions: HashMap<String, Language>,
    
    /// Whether to cache analysis results
    pub enable_caching: bool,
    
    /// Cache directory (relative to project root or absolute)
    pub cache_dir: PathBuf,
    
    /// Cache TTL in seconds
    pub cache_ttl: u64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            analyze_content: true,
            compute_tokens: true,
            count_lines: true,
            detect_binary_content: false,
            language_overrides: HashMap::new(),
            custom_extensions: HashMap::new(),
            enable_caching: false,
            cache_dir: PathBuf::from(".scribe-cache"),
            cache_ttl: 3600, // 1 hour
        }
    }
}

impl AnalysisConfig {
    fn validate(&self) -> Result<()> {
        if self.cache_ttl == 0 {
            return Err(ScribeError::config_field(
                "cache_ttl must be > 0", 
                "cache_ttl"
            ));
        }
        Ok(())
    }
}

/// Scoring system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Heuristic weights
    pub weights: HeuristicWeights,
    
    /// Whether to enable advanced scoring features
    pub enable_advanced: bool,
    
    /// Custom scoring rules
    pub custom_rules: Vec<CustomScoringRule>,
    
    /// Minimum score threshold for inclusion
    pub min_score_threshold: f64,
    
    /// Maximum number of files to return (0 = unlimited)
    pub max_results: usize,
    
    /// Whether to normalize scores to 0-1 range
    pub normalize_scores: bool,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            weights: HeuristicWeights::default(),
            enable_advanced: false,
            custom_rules: vec![],
            min_score_threshold: 0.0,
            max_results: 0, // Unlimited
            normalize_scores: true,
        }
    }
}

impl ScoringConfig {
    fn validate(&self) -> Result<()> {
        if self.min_score_threshold < 0.0 || self.min_score_threshold > 1.0 {
            return Err(ScribeError::config_field(
                "min_score_threshold must be between 0.0 and 1.0", 
                "min_score_threshold"
            ));
        }
        Ok(())
    }
}

/// Custom scoring rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScoringRule {
    /// Rule name/description
    pub name: String,
    
    /// File pattern to match
    pub pattern: String,
    
    /// Score modifier type
    pub modifier: ScoreModifier,
}

/// Score modifier operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreModifier {
    /// Add a constant value
    Add(f64),
    /// Multiply by a factor
    Multiply(f64),
    /// Set to a specific value
    Set(f64),
    /// Add bonus based on condition
    ConditionalBonus { condition: String, bonus: f64 },
}

/// Performance and resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum memory usage in MB (0 = unlimited)
    pub max_memory_mb: usize,
    
    /// Analysis timeout per file in seconds
    pub analysis_timeout: u64,
    
    /// Global timeout in seconds
    pub global_timeout: u64,
    
    /// Batch size for parallel processing
    pub batch_size: usize,
    
    /// Whether to use memory mapping for large files
    pub use_mmap: bool,
    
    /// Buffer size for I/O operations
    pub io_buffer_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 0, // Unlimited
            analysis_timeout: 30,
            global_timeout: 300, // 5 minutes
            batch_size: 100,
            use_mmap: false,
            io_buffer_size: 64 * 1024, // 64KB
        }
    }
}

impl PerformanceConfig {
    fn validate(&self) -> Result<()> {
        if self.analysis_timeout == 0 {
            return Err(ScribeError::config_field(
                "analysis_timeout must be > 0", 
                "analysis_timeout"
            ));
        }
        if self.global_timeout == 0 {
            return Err(ScribeError::config_field(
                "global_timeout must be > 0", 
                "global_timeout"
            ));
        }
        if self.batch_size == 0 {
            return Err(ScribeError::config_field(
                "batch_size must be > 0", 
                "batch_size"
            ));
        }
        Ok(())
    }

    /// Get analysis timeout as Duration
    pub fn analysis_timeout_duration(&self) -> Duration {
        Duration::from_secs(self.analysis_timeout)
    }

    /// Get global timeout as Duration
    pub fn global_timeout_duration(&self) -> Duration {
        Duration::from_secs(self.global_timeout)
    }
}

/// Git integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitConfig {
    /// Whether to use git information
    pub enabled: bool,
    
    /// Whether to respect .gitignore
    pub respect_gitignore: bool,
    
    /// Whether to include git status in analysis
    pub include_status: bool,
    
    /// Whether to analyze git history for churn
    pub analyze_history: bool,
    
    /// Number of commits to analyze for churn (0 = all)
    pub history_depth: usize,
    
    /// Whether to include untracked files
    pub include_untracked: bool,
    
    /// Git command timeout in seconds
    pub git_timeout: u64,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            respect_gitignore: true,
            include_status: true,
            analyze_history: false,
            history_depth: 100,
            include_untracked: false,
            git_timeout: 30,
        }
    }
}

impl GitConfig {
    fn validate(&self) -> Result<()> {
        if self.git_timeout == 0 {
            return Err(ScribeError::config_field(
                "git_timeout must be > 0", 
                "git_timeout"
            ));
        }
        Ok(())
    }
}

/// Feature flags for experimental features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable PageRank centrality computation
    pub centrality_enabled: bool,
    
    /// Enable entrypoint detection
    pub entrypoint_detection: bool,
    
    /// Enable examples/usage analysis
    pub examples_analysis: bool,
    
    /// Enable semantic analysis (if available)
    pub semantic_analysis: bool,
    
    /// Enable machine learning features
    pub ml_features: bool,
    
    /// Enable experimental scoring algorithms
    pub experimental_scoring: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            centrality_enabled: false,
            entrypoint_detection: false,
            examples_analysis: false,
            semantic_analysis: false,
            ml_features: false,
            experimental_scoring: false,
        }
    }
}

impl FeatureFlags {
    fn validate(&self) -> Result<()> {
        // Currently no validation needed
        Ok(())
    }

    /// Check if any V2 features are enabled
    pub fn has_v2_features(&self) -> bool {
        self.centrality_enabled || self.entrypoint_detection || self.examples_analysis
    }

    /// Get list of enabled feature names
    pub fn enabled_features(&self) -> Vec<&'static str> {
        let mut features = Vec::new();
        
        if self.centrality_enabled {
            features.push("centrality");
        }
        if self.entrypoint_detection {
            features.push("entrypoint_detection");
        }
        if self.examples_analysis {
            features.push("examples_analysis");
        }
        if self.semantic_analysis {
            features.push("semantic_analysis");
        }
        if self.ml_features {
            features.push("ml_features");
        }
        if self.experimental_scoring {
            features.push("experimental_scoring");
        }
        
        features
    }
}

/// Output format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format
    pub format: OutputFormat,
    
    /// Whether to include file content in output
    pub include_content: bool,
    
    /// Whether to include detailed scores breakdown
    pub include_score_breakdown: bool,
    
    /// Whether to include repository statistics
    pub include_repo_stats: bool,
    
    /// Whether to sort results by score
    pub sort_by_score: bool,
    
    /// Pretty print JSON output
    pub pretty_json: bool,
    
    /// Custom output fields to include
    pub custom_fields: Vec<String>,
}

/// Output format options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    JsonLines,
    Csv,
    Table,
    Summary,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            include_content: false,
            include_score_breakdown: true,
            include_repo_stats: true,
            sort_by_score: true,
            pretty_json: true,
            custom_fields: vec![],
        }
    }
}

impl OutputConfig {
    fn validate(&self) -> Result<()> {
        // Currently no validation needed
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.general.verbosity, 1);
        assert!(config.filtering.respect_gitignore);
        assert!(config.git.enabled);
        assert!(!config.features.centrality_enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        // Test invalid verbosity
        config.general.verbosity = 10;
        assert!(config.validate().is_err());

        // Reset and test invalid file sizes
        config = Config::default();
        config.filtering.max_file_size = 100;
        config.filtering.min_file_size = 200;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_file_io() {
        let config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();
        
        // Test JSON serialization
        let json_path = temp_file.path().with_extension("json");
        config.save_to_file(&json_path).unwrap();
        let loaded_config = Config::load_from_file(&json_path).unwrap();
        
        assert_eq!(config.general.verbosity, loaded_config.general.verbosity);
    }

    #[test]
    fn test_filtering_patterns() {
        let mut config = FilteringConfig::default();
        config.include_patterns.push("*.rs".to_string());
        config.exclude_patterns.push("target/**".to_string());
        
        assert!(config.validate().is_ok());
        
        let include_set = config.build_include_set().unwrap();
        assert!(include_set.is_some());
        
        let exclude_set = config.build_exclude_set().unwrap();
        assert!(exclude_set.is_match("target/debug/file.o"));
    }

    #[test]
    fn test_feature_flags() {
        let mut flags = FeatureFlags::default();
        assert!(!flags.has_v2_features());
        assert!(flags.enabled_features().is_empty());
        
        flags.centrality_enabled = true;
        flags.entrypoint_detection = true;
        
        assert!(flags.has_v2_features());
        let enabled = flags.enabled_features();
        assert!(enabled.contains(&"centrality"));
        assert!(enabled.contains(&"entrypoint_detection"));
    }

    #[test]
    fn test_performance_config_timeouts() {
        let config = PerformanceConfig::default();
        assert_eq!(config.analysis_timeout_duration(), Duration::from_secs(30));
        assert_eq!(config.global_timeout_duration(), Duration::from_secs(300));
    }

    #[test]
    fn test_config_hash() {
        let config1 = Config::default();
        let config2 = Config::default();
        
        let hash1 = config1.compute_hash();
        let hash2 = config2.compute_hash();
        
        assert_eq!(hash1, hash2);
        
        let mut config3 = Config::default();
        config3.general.verbosity = 2;
        let hash3 = config3.compute_hash();
        
        assert_ne!(hash1, hash3);
    }
}
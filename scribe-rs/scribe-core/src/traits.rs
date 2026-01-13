//! Core traits for extensibility and plugin architecture.
//!
//! Defines the essential traits that enable customization and extension
//! of Scribe's analysis pipeline, scoring system, and output formatting.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::config::Config;
use crate::error::Result;
use crate::file::FileInfo;
use crate::types::{AnalysisResult, CentralityScores, RepositoryInfo, ScoreComponents};

/// Core trait for file analysis implementations
#[async_trait]
pub trait FileAnalyzer: Send + Sync {
    /// Analyze a single file and return metadata
    async fn analyze_file(&self, file_path: &Path, config: &Config) -> Result<FileInfo>;

    /// Load and analyze file content
    async fn analyze_content(&self, file_info: &mut FileInfo, config: &Config) -> Result<()>;

    /// Batch analyze multiple files for efficiency
    async fn analyze_batch(&self, files: Vec<&Path>, config: &Config) -> Result<Vec<FileInfo>> {
        let mut results = Vec::new();
        for file in files {
            results.push(self.analyze_file(file, config).await?);
        }
        Ok(results)
    }

    /// Get analyzer name/version for caching
    fn name(&self) -> &'static str;

    /// Get analyzer version for cache invalidation
    fn version(&self) -> &'static str;
}

/// Trait for heuristic scoring implementations
pub trait HeuristicScorer: Send + Sync {
    /// Compute heuristic scores for a file
    fn score_file(
        &self,
        file_info: &FileInfo,
        repo_info: &RepositoryInfo,
    ) -> Result<ScoreComponents>;

    /// Batch score multiple files (can be optimized for cross-file analysis)
    fn score_batch(
        &self,
        files: &[&FileInfo],
        repo_info: &RepositoryInfo,
    ) -> Result<Vec<ScoreComponents>> {
        files
            .iter()
            .map(|file| self.score_file(file, repo_info))
            .collect()
    }

    /// Get scorer name for identification
    fn name(&self) -> &'static str;

    /// Get list of score components this scorer produces
    fn score_components(&self) -> Vec<&'static str>;

    /// Check if scorer supports advanced features (V2)
    fn supports_advanced_features(&self) -> bool {
        false
    }
}

/// Trait for repository analysis implementations
#[async_trait]
pub trait RepositoryAnalyzer: Send + Sync {
    /// Analyze repository structure and metadata
    async fn analyze_repository(&self, root_path: &Path, config: &Config)
        -> Result<RepositoryInfo>;

    /// Get repository statistics
    async fn get_statistics(&self, root_path: &Path, files: &[FileInfo]) -> Result<RepositoryInfo>;

    /// Check if this analyzer can handle the given repository
    fn can_analyze(&self, root_path: &Path) -> bool;

    /// Get analyzer priority (higher = preferred)
    fn priority(&self) -> u8 {
        0
    }
}

/// Trait for git integration implementations
#[async_trait]
pub trait GitIntegration: Send + Sync {
    /// Check if path is in a git repository
    async fn is_git_repository(&self, path: &Path) -> Result<bool>;

    /// Get git status for files
    async fn get_file_status(&self, files: &[&Path]) -> Result<Vec<crate::file::GitStatus>>;

    /// Get repository information
    async fn get_repo_info(&self, root_path: &Path) -> Result<GitRepositoryInfo>;

    /// Analyze file churn (commit history)
    async fn analyze_churn(
        &self,
        root_path: &Path,
        depth: usize,
    ) -> Result<Vec<crate::types::ChurnInfo>>;

    /// Check if file should be ignored by git
    async fn should_ignore(&self, file_path: &Path, root_path: &Path) -> Result<bool>;
}

/// Git repository information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRepositoryInfo {
    /// Repository root path
    pub root: std::path::PathBuf,
    /// Current branch
    pub branch: Option<String>,
    /// Remote URL
    pub remote_url: Option<String>,
    /// Last commit hash
    pub last_commit: Option<String>,
    /// Whether repository has uncommitted changes
    pub has_changes: bool,
}

/// Trait for centrality computation implementations
#[async_trait]
pub trait CentralityComputer: Send + Sync {
    /// Build dependency graph from files
    async fn build_dependency_graph(&self, files: &[&FileInfo]) -> Result<DependencyGraph>;

    /// Compute PageRank centrality scores
    async fn compute_centrality(&self, graph: &DependencyGraph) -> Result<CentralityScores>;

    /// Get supported languages for dependency analysis
    fn supported_languages(&self) -> Vec<crate::file::Language>;

    /// Check if file can be analyzed for dependencies
    fn can_analyze_file(&self, file_info: &FileInfo) -> bool;
}

/// Dependency graph representation
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Node IDs (file paths)
    pub nodes: Vec<String>,

    /// Adjacency list (node_id -> \[dependent_node_ids\])
    pub edges: Vec<Vec<usize>>,

    /// Reverse adjacency list (node_id -> \[dependency_node_ids\])
    pub reverse_edges: Vec<Vec<usize>>,

    /// Node metadata
    pub node_metadata: Vec<DependencyNodeMetadata>,
}

/// Metadata for dependency graph nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyNodeMetadata {
    /// File path
    pub path: String,
    /// Programming language
    pub language: crate::file::Language,
    /// File size
    pub size: u64,
    /// Whether this is a test file
    pub is_test: bool,
    /// Whether this is an entrypoint
    pub is_entrypoint: bool,
}

impl DependencyGraph {
    /// Create empty dependency graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            reverse_edges: Vec::new(),
            node_metadata: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, path: String, metadata: DependencyNodeMetadata) -> usize {
        let node_id = self.nodes.len();
        self.nodes.push(path);
        self.edges.push(Vec::new());
        self.reverse_edges.push(Vec::new());
        self.node_metadata.push(metadata);
        node_id
    }

    /// Add an edge from source to target
    pub fn add_edge(&mut self, source: usize, target: usize) {
        if source < self.edges.len() && target < self.reverse_edges.len() {
            self.edges[source].push(target);
            self.reverse_edges[target].push(source);
        }
    }

    /// Get graph statistics
    pub fn stats(&self) -> crate::types::GraphStats {
        let total_nodes = self.nodes.len();
        let total_edges: usize = self.edges.iter().map(|adj| adj.len()).sum();

        let in_degree_sum: usize = self.reverse_edges.iter().map(|adj| adj.len()).sum();
        let out_degree_sum: usize = self.edges.iter().map(|adj| adj.len()).sum();

        let in_degree_avg = if total_nodes > 0 {
            in_degree_sum as f64 / total_nodes as f64
        } else {
            0.0
        };
        let out_degree_avg = if total_nodes > 0 {
            out_degree_sum as f64 / total_nodes as f64
        } else {
            0.0
        };

        let in_degree_max = self
            .reverse_edges
            .iter()
            .map(|adj| adj.len())
            .max()
            .unwrap_or(0);
        let out_degree_max = self.edges.iter().map(|adj| adj.len()).max().unwrap_or(0);

        let possible_edges = if total_nodes > 1 {
            total_nodes * (total_nodes - 1)
        } else {
            0
        };
        let graph_density = if possible_edges > 0 {
            total_edges as f64 / possible_edges as f64
        } else {
            0.0
        };

        crate::types::GraphStats {
            total_nodes,
            total_edges,
            in_degree_avg,
            in_degree_max,
            out_degree_avg,
            out_degree_max,
            strongly_connected_components: 0, // TODO: Implement SCC computation
            graph_density,
        }
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for pattern matching implementations (glob, regex, etc.)
pub trait PatternMatcher: Send + Sync {
    /// Check if a path matches the pattern
    fn matches(&self, path: &Path) -> bool;

    /// Get pattern string for debugging
    fn pattern(&self) -> &str;

    /// Check if pattern is case sensitive
    fn is_case_sensitive(&self) -> bool {
        true
    }
}

/// Trait for output formatting implementations
pub trait OutputFormatter: Send + Sync {
    /// Format analysis results
    fn format_results(&self, results: &[AnalysisResult], config: &Config) -> Result<String>;

    /// Format repository information
    fn format_repository_info(&self, repo_info: &RepositoryInfo, config: &Config)
        -> Result<String>;

    /// Get supported output format name
    fn format_name(&self) -> &'static str;

    /// Get file extension for output format
    fn file_extension(&self) -> &'static str;

    /// Check if format supports streaming output
    fn supports_streaming(&self) -> bool {
        false
    }
}

/// Trait for caching implementations
#[async_trait]
pub trait CacheStorage: Send + Sync {
    /// Get cached result
    async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send;

    /// Store result in cache
    async fn put<T>(&self, key: &str, value: &T, ttl: Option<std::time::Duration>) -> Result<()>
    where
        T: Serialize + Send + Sync;

    /// Remove item from cache
    async fn remove(&self, key: &str) -> Result<()>;

    /// Clear entire cache
    async fn clear(&self) -> Result<()>;

    /// Get cache statistics
    async fn stats(&self) -> Result<CacheStats>;
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of items in cache
    pub item_count: usize,
    /// Total cache size in bytes
    pub size_bytes: u64,
    /// Cache hit rate (0.0 - 1.0)
    pub hit_rate: f64,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
}

/// Trait for progress reporting implementations
pub trait ProgressReporter: Send + Sync {
    /// Start a new progress bar/indicator
    fn start(&self, total: u64, message: &str);

    /// Update progress
    fn update(&self, current: u64, message: Option<&str>);

    /// Finish progress reporting
    fn finish(&self, message: &str);

    /// Report an error
    fn error(&self, message: &str);

    /// Report a warning
    fn warning(&self, message: &str);

    /// Check if progress reporting is enabled
    fn is_enabled(&self) -> bool;
}

/// Trait for language-specific analysis extensions
#[async_trait]
pub trait LanguageExtension: Send + Sync {
    /// Get supported languages
    fn supported_languages(&self) -> Vec<crate::file::Language>;

    /// Extract dependencies from file content
    async fn extract_dependencies(
        &self,
        content: &str,
        language: crate::file::Language,
    ) -> Result<Vec<String>>;

    /// Detect if file is an entrypoint (main, index, etc.)
    async fn is_entrypoint(&self, file_info: &FileInfo) -> Result<bool>;

    /// Extract documentation/comments
    async fn extract_documentation(
        &self,
        content: &str,
        language: crate::file::Language,
    ) -> Result<Vec<DocumentationBlock>>;

    /// Get extension priority (higher = preferred for language)
    fn priority(&self) -> u8 {
        0
    }
}

/// Documentation block extracted from source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationBlock {
    /// Documentation text
    pub text: String,
    /// Position in file
    pub position: crate::types::Range,
    /// Type of documentation (comment, docstring, etc.)
    pub doc_type: DocumentationType,
}

/// Types of documentation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentationType {
    /// Single line comment (// or #)
    LineComment,
    /// Block comment (/* */ or """ """)
    BlockComment,
    /// Documentation comment (/// or /** */)
    DocComment,
    /// Docstring (Python, etc.)
    Docstring,
    /// Module-level documentation
    ModuleDoc,
    /// README or markdown documentation
    Readme,
}

/// Trait for plugin registration and discovery
pub trait PluginRegistry: Send + Sync {
    /// Register a file analyzer
    fn register_analyzer(&mut self, analyzer: Box<dyn FileAnalyzer>);

    /// Register a scorer
    fn register_scorer(&mut self, scorer: Box<dyn HeuristicScorer>);

    /// Register a repository analyzer
    fn register_repository_analyzer(&mut self, analyzer: Box<dyn RepositoryAnalyzer>);

    /// Register an output formatter
    fn register_formatter(&mut self, formatter: Box<dyn OutputFormatter>);

    /// Register a language extension
    fn register_language_extension(&mut self, extension: Box<dyn LanguageExtension>);

    /// Get registered analyzers
    fn get_analyzers(&self) -> Vec<&dyn FileAnalyzer>;

    /// Get registered scorers
    fn get_scorers(&self) -> Vec<&dyn HeuristicScorer>;

    /// Get registered formatters
    fn get_formatters(&self) -> Vec<&dyn OutputFormatter>;

    /// Load plugins from directory
    fn load_plugins_from_dir(&mut self, dir: &Path) -> Result<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAnalyzer;

    #[async_trait]
    impl FileAnalyzer for MockAnalyzer {
        async fn analyze_file(&self, _file_path: &Path, _config: &Config) -> Result<FileInfo> {
            unimplemented!()
        }

        async fn analyze_content(&self, _file_info: &mut FileInfo, _config: &Config) -> Result<()> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "mock"
        }

        fn version(&self) -> &'static str {
            "1.0.0"
        }
    }

    #[test]
    fn mock_analyzer_metadata() {
        let analyzer = MockAnalyzer;
        assert_eq!(analyzer.name(), "mock");
        assert_eq!(analyzer.version(), "1.0.0");
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();

        let metadata = DependencyNodeMetadata {
            path: "test.rs".to_string(),
            language: crate::file::Language::Rust,
            size: 100,
            is_test: false,
            is_entrypoint: false,
        };

        let node1 = graph.add_node("file1.rs".to_string(), metadata.clone());
        let node2 = graph.add_node("file2.rs".to_string(), metadata);

        graph.add_edge(node1, node2);

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges[node1].len(), 1);
        assert_eq!(graph.reverse_edges[node2].len(), 1);

        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 1);
    }

    #[test]
    fn test_documentation_block() {
        use crate::types::{Position, Range};

        let doc_block = DocumentationBlock {
            text: "This is a test function".to_string(),
            position: Range::new(Position::new(0, 0), Position::new(0, 23)),
            doc_type: DocumentationType::DocComment,
        };

        assert_eq!(doc_block.doc_type, DocumentationType::DocComment);
        assert!(doc_block.text.contains("test function"));
    }

    #[test]
    fn test_dependency_graph_default() {
        let graph = DependencyGraph::default();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
        assert!(graph.reverse_edges.is_empty());
        assert!(graph.node_metadata.is_empty());
    }

    #[test]
    fn test_dependency_graph_stats_empty() {
        let graph = DependencyGraph::new();
        let stats = graph.stats();

        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.in_degree_avg, 0.0);
        assert_eq!(stats.out_degree_avg, 0.0);
        assert_eq!(stats.in_degree_max, 0);
        assert_eq!(stats.out_degree_max, 0);
        assert_eq!(stats.graph_density, 0.0);
    }

    #[test]
    fn test_dependency_graph_stats_single_node() {
        let mut graph = DependencyGraph::new();
        let metadata = DependencyNodeMetadata {
            path: "single.rs".to_string(),
            language: crate::file::Language::Rust,
            size: 50,
            is_test: false,
            is_entrypoint: true,
        };
        graph.add_node("single.rs".to_string(), metadata);

        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.graph_density, 0.0); // No possible edges with single node
    }

    #[test]
    fn test_dependency_graph_multiple_edges() {
        let mut graph = DependencyGraph::new();
        let metadata = DependencyNodeMetadata {
            path: "test.rs".to_string(),
            language: crate::file::Language::Rust,
            size: 100,
            is_test: false,
            is_entrypoint: false,
        };

        let node0 = graph.add_node("a.rs".to_string(), metadata.clone());
        let node1 = graph.add_node("b.rs".to_string(), metadata.clone());
        let node2 = graph.add_node("c.rs".to_string(), metadata.clone());
        let node3 = graph.add_node("d.rs".to_string(), metadata);

        // a -> b, a -> c, b -> d, c -> d
        graph.add_edge(node0, node1);
        graph.add_edge(node0, node2);
        graph.add_edge(node1, node3);
        graph.add_edge(node2, node3);

        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 4);
        assert_eq!(stats.total_edges, 4);
        assert_eq!(stats.out_degree_max, 2); // node0 has 2 outgoing edges
        assert_eq!(stats.in_degree_max, 2);  // node3 has 2 incoming edges
    }

    #[test]
    fn test_dependency_graph_add_edge_invalid() {
        let mut graph = DependencyGraph::new();
        let metadata = DependencyNodeMetadata {
            path: "test.rs".to_string(),
            language: crate::file::Language::Rust,
            size: 100,
            is_test: false,
            is_entrypoint: false,
        };
        let node0 = graph.add_node("a.rs".to_string(), metadata);

        // Try to add edge to non-existent node - should not panic
        graph.add_edge(node0, 999);
        graph.add_edge(999, node0);

        // Graph should remain unchanged
        assert!(graph.edges[node0].is_empty());
    }

    #[test]
    fn test_dependency_node_metadata() {
        let metadata = DependencyNodeMetadata {
            path: "src/main.rs".to_string(),
            language: crate::file::Language::Rust,
            size: 500,
            is_test: false,
            is_entrypoint: true,
        };

        assert_eq!(metadata.path, "src/main.rs");
        assert_eq!(metadata.language, crate::file::Language::Rust);
        assert_eq!(metadata.size, 500);
        assert!(!metadata.is_test);
        assert!(metadata.is_entrypoint);
    }

    #[test]
    fn test_dependency_node_metadata_clone() {
        let metadata = DependencyNodeMetadata {
            path: "test.py".to_string(),
            language: crate::file::Language::Python,
            size: 200,
            is_test: true,
            is_entrypoint: false,
        };

        let cloned = metadata.clone();
        assert_eq!(metadata.path, cloned.path);
        assert_eq!(metadata.language, cloned.language);
        assert_eq!(metadata.is_test, cloned.is_test);
    }

    #[test]
    fn test_documentation_type_equality() {
        assert_eq!(DocumentationType::LineComment, DocumentationType::LineComment);
        assert_eq!(DocumentationType::BlockComment, DocumentationType::BlockComment);
        assert_eq!(DocumentationType::DocComment, DocumentationType::DocComment);
        assert_eq!(DocumentationType::Docstring, DocumentationType::Docstring);
        assert_eq!(DocumentationType::ModuleDoc, DocumentationType::ModuleDoc);
        assert_eq!(DocumentationType::Readme, DocumentationType::Readme);

        assert_ne!(DocumentationType::LineComment, DocumentationType::BlockComment);
        assert_ne!(DocumentationType::DocComment, DocumentationType::Docstring);
    }

    #[test]
    fn test_documentation_type_clone() {
        let doc_type = DocumentationType::DocComment;
        let cloned = doc_type.clone();
        assert_eq!(doc_type, cloned);
    }

    #[test]
    fn test_git_repository_info() {
        let info = GitRepositoryInfo {
            root: std::path::PathBuf::from("/repo"),
            branch: Some("main".to_string()),
            remote_url: Some("https://github.com/user/repo.git".to_string()),
            last_commit: Some("abc123".to_string()),
            has_changes: true,
        };

        assert_eq!(info.root, std::path::PathBuf::from("/repo"));
        assert_eq!(info.branch, Some("main".to_string()));
        assert!(info.has_changes);
    }

    #[test]
    fn test_git_repository_info_minimal() {
        let info = GitRepositoryInfo {
            root: std::path::PathBuf::from("."),
            branch: None,
            remote_url: None,
            last_commit: None,
            has_changes: false,
        };

        assert!(info.branch.is_none());
        assert!(info.remote_url.is_none());
        assert!(!info.has_changes);
    }

    #[test]
    fn test_git_repository_info_clone() {
        let info = GitRepositoryInfo {
            root: std::path::PathBuf::from("/repo"),
            branch: Some("develop".to_string()),
            remote_url: Some("git@github.com:user/repo.git".to_string()),
            last_commit: Some("def456".to_string()),
            has_changes: false,
        };

        let cloned = info.clone();
        assert_eq!(info.root, cloned.root);
        assert_eq!(info.branch, cloned.branch);
        assert_eq!(info.remote_url, cloned.remote_url);
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            item_count: 100,
            size_bytes: 1024 * 1024,
            hit_rate: 0.85,
            hits: 850,
            misses: 150,
        };

        assert_eq!(stats.item_count, 100);
        assert_eq!(stats.size_bytes, 1024 * 1024);
        assert!((stats.hit_rate - 0.85).abs() < 0.001);
        assert_eq!(stats.hits, 850);
        assert_eq!(stats.misses, 150);
    }

    #[test]
    fn test_cache_stats_clone() {
        let stats = CacheStats {
            item_count: 50,
            size_bytes: 512,
            hit_rate: 0.9,
            hits: 90,
            misses: 10,
        };

        let cloned = stats.clone();
        assert_eq!(stats.item_count, cloned.item_count);
        assert_eq!(stats.hits, cloned.hits);
    }

    #[test]
    fn test_documentation_block_clone() {
        use crate::types::{Position, Range};

        let doc = DocumentationBlock {
            text: "Test documentation".to_string(),
            position: Range::new(Position::new(10, 0), Position::new(15, 0)),
            doc_type: DocumentationType::BlockComment,
        };

        let cloned = doc.clone();
        assert_eq!(doc.text, cloned.text);
        assert_eq!(doc.doc_type, cloned.doc_type);
    }

    #[test]
    fn test_dependency_graph_graph_density() {
        let mut graph = DependencyGraph::new();
        let metadata = DependencyNodeMetadata {
            path: "test.rs".to_string(),
            language: crate::file::Language::Rust,
            size: 100,
            is_test: false,
            is_entrypoint: false,
        };

        // Create 3 nodes with all possible edges (complete graph)
        let n0 = graph.add_node("a.rs".to_string(), metadata.clone());
        let n1 = graph.add_node("b.rs".to_string(), metadata.clone());
        let n2 = graph.add_node("c.rs".to_string(), metadata);

        // Add all possible directed edges (n*(n-1) = 6 edges)
        graph.add_edge(n0, n1);
        graph.add_edge(n0, n2);
        graph.add_edge(n1, n0);
        graph.add_edge(n1, n2);
        graph.add_edge(n2, n0);
        graph.add_edge(n2, n1);

        let stats = graph.stats();
        assert_eq!(stats.total_edges, 6);
        // Complete graph should have density = 1.0
        assert!((stats.graph_density - 1.0).abs() < 0.001);
    }

    // Test default trait implementations
    struct MockScorer;

    impl HeuristicScorer for MockScorer {
        fn score_file(
            &self,
            _file_info: &FileInfo,
            _repo_info: &RepositoryInfo,
        ) -> Result<ScoreComponents> {
            Ok(ScoreComponents::zero())
        }

        fn name(&self) -> &'static str {
            "mock_scorer"
        }

        fn score_components(&self) -> Vec<&'static str> {
            vec!["component1", "component2"]
        }
    }

    #[test]
    fn test_heuristic_scorer_defaults() {
        let scorer = MockScorer;
        assert!(!scorer.supports_advanced_features());
        assert_eq!(scorer.name(), "mock_scorer");
        assert_eq!(scorer.score_components().len(), 2);
    }

    struct MockPatternMatcher {
        pattern: String,
    }

    impl PatternMatcher for MockPatternMatcher {
        fn matches(&self, path: &Path) -> bool {
            path.to_string_lossy().contains(&self.pattern)
        }

        fn pattern(&self) -> &str {
            &self.pattern
        }
    }

    #[test]
    fn test_pattern_matcher_defaults() {
        let matcher = MockPatternMatcher {
            pattern: "*.rs".to_string(),
        };
        assert!(matcher.is_case_sensitive());
        assert_eq!(matcher.pattern(), "*.rs");
    }

    #[test]
    fn test_pattern_matcher_matches() {
        let matcher = MockPatternMatcher {
            pattern: "test".to_string(),
        };
        assert!(matcher.matches(Path::new("test.rs")));
        assert!(!matcher.matches(Path::new("main.rs")));
    }

    struct MockOutputFormatter;

    impl OutputFormatter for MockOutputFormatter {
        fn format_results(&self, _results: &[AnalysisResult], _config: &Config) -> Result<String> {
            Ok("formatted results".to_string())
        }

        fn format_repository_info(&self, _repo_info: &RepositoryInfo, _config: &Config) -> Result<String> {
            Ok("formatted repo info".to_string())
        }

        fn format_name(&self) -> &'static str {
            "mock"
        }

        fn file_extension(&self) -> &'static str {
            "txt"
        }
    }

    #[test]
    fn test_output_formatter_defaults() {
        let formatter = MockOutputFormatter;
        assert!(!formatter.supports_streaming());
        assert_eq!(formatter.format_name(), "mock");
        assert_eq!(formatter.file_extension(), "txt");
    }

    struct MockRepositoryAnalyzer;

    #[async_trait]
    impl RepositoryAnalyzer for MockRepositoryAnalyzer {
        async fn analyze_repository(&self, _root_path: &Path, _config: &Config) -> Result<RepositoryInfo> {
            unimplemented!()
        }

        async fn get_statistics(&self, _root_path: &Path, _files: &[FileInfo]) -> Result<RepositoryInfo> {
            unimplemented!()
        }

        fn can_analyze(&self, _root_path: &Path) -> bool {
            true
        }
    }

    #[test]
    fn test_repository_analyzer_defaults() {
        let analyzer = MockRepositoryAnalyzer;
        assert_eq!(analyzer.priority(), 0);
        assert!(analyzer.can_analyze(Path::new("/tmp")));
    }
}

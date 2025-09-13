"""
Scribe-RS: High-performance code analysis library with Rust backend

A Python library that provides fast, comprehensive code analysis capabilities
powered by a Rust backend. Features include repository scanning, heuristic
scoring, dependency graph analysis, and pattern matching.

Example:
    Basic usage example:

    >>> import asyncio
    >>> from scribe_rs import Repository, HeuristicScorer, PageRankAnalyzer
    >>> 
    >>> async def analyze_repo():
    ...     # Create repository instance
    ...     repo = Repository("/path/to/repo")
    ...     
    ...     # Scan files
    ...     files = await repo.scan_files(max_files=1000)
    ...     
    ...     # Score files
    ...     scorer = HeuristicScorer()
    ...     scores = await scorer.score_files(files)
    ...     
    ...     # Analyze dependencies
    ...     analyzer = PageRankAnalyzer()
    ...     centrality = await analyzer.analyze_dependencies(files)
    ...     
    ...     # Combine results
    ...     final_scores = scorer.combine_with_centrality(scores, centrality)
    ...     return final_scores
    >>> 
    >>> # Run analysis
    >>> results = asyncio.run(analyze_repo())
"""

# Import the compiled Rust extension module
from ._scribe_py import (
    # Core classes
    Repository,
    HeuristicScorer, 
    PageRankAnalyzer,
    PatternMatcher,
    
    # Factory functions
    create_repository,
    create_default_scorer,
    create_scorer_with_weights,
    create_pagerank_analyzer,
    create_pagerank_analyzer_with_config,
    create_pattern_matcher,
    
    # Utility functions
    get_default_weights,
    get_supported_languages,
    validate_pattern,
    is_valid_repository,
    find_repository_root,
    initialize_async_runtime,
    get_version_info,
    get_build_info,
    
    # Exceptions
    ScribeException,
    AnalysisException, 
    PatternException,
    ConfigurationException,
    
    # Version info
    __version__ as _version,
    __doc__ as _doc,
)

# Initialize async runtime on import
initialize_async_runtime()

# Package metadata
__version__ = _version
__author__ = "Nathan Rice"
__email__ = "nathan.alexander.rice@gmail.com"
__license__ = "MIT OR Apache-2.0"
__description__ = "High-performance code analysis library with Rust backend"

# Public API
__all__ = [
    # Core classes
    "Repository",
    "HeuristicScorer",
    "PageRankAnalyzer", 
    "PatternMatcher",
    
    # Factory functions
    "create_repository",
    "create_default_scorer",
    "create_scorer_with_weights", 
    "create_pagerank_analyzer",
    "create_pagerank_analyzer_with_config",
    "create_pattern_matcher",
    
    # Utility functions
    "get_default_weights",
    "get_supported_languages",
    "validate_pattern",
    "is_valid_repository",
    "find_repository_root",
    "get_version_info",
    "get_build_info",
    
    # Exceptions
    "ScribeException",
    "AnalysisException",
    "PatternException", 
    "ConfigurationException",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
]


def get_info():
    """Get comprehensive library information.
    
    Returns:
        dict: Dictionary containing version, build info, and capabilities
    """
    info = get_version_info()
    build = get_build_info()
    
    return {
        "version": info,
        "build": build,
        "capabilities": {
            "async_support": True,
            "parallel_processing": True,
            "graph_analysis": True,
            "pattern_matching": True,
            "heuristic_scoring": True,
            "repository_scanning": True,
            "supported_languages": get_supported_languages(),
        }
    }


class AnalysisConfig:
    """Configuration helper for analysis operations.
    
    This class provides a convenient way to configure various analysis
    parameters across different components.
    """
    
    def __init__(self):
        self.max_files = 10000
        self.max_file_size = 1024 * 1024  # 1MB
        self.include_patterns = []
        self.exclude_patterns = [
            "*.pyc", "*.pyo", "*.pyd", "__pycache__", ".git", 
            "node_modules", "target", "build", "dist"
        ]
        self.parallel_workers = None  # Auto-detect
        self.batch_size = 100
        
        # Scoring weights
        self.scoring_weights = {
            "documentation": 0.15,
            "complexity": 0.20,
            "imports": 0.10,
            "exports": 0.10,
            "functions": 0.10,
            "classes": 0.10,
            "tests": 0.05,
            "config": 0.05,
            "size": 0.05,
            "age": 0.05,
            "churn": 0.05,
            "centrality": 0.20,
        }
        
        # PageRank parameters
        self.pagerank_damping = 0.85
        self.pagerank_max_iterations = 100
        self.pagerank_tolerance = 1e-6
    
    def to_dict(self):
        """Convert configuration to dictionary format."""
        return {
            "max_files": self.max_files,
            "max_file_size": self.max_file_size,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "parallel_workers": self.parallel_workers,
            "batch_size": self.batch_size,
            "scoring_weights": self.scoring_weights,
            "pagerank_damping": self.pagerank_damping,
            "pagerank_max_iterations": self.pagerank_max_iterations,
            "pagerank_tolerance": self.pagerank_tolerance,
        }


async def analyze_repository_complete(
    repo_path: str,
    config: AnalysisConfig = None,
    progress_callback=None
):
    """Perform comprehensive repository analysis.
    
    This high-level function combines all analysis capabilities:
    repository scanning, heuristic scoring, dependency analysis,
    and pattern matching.
    
    Args:
        repo_path: Path to the repository to analyze
        config: Analysis configuration (uses defaults if None)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        dict: Comprehensive analysis results including:
            - file_scores: Dictionary of file paths to score components
            - repository_info: Repository metadata and statistics
            - language_stats: File counts and sizes by programming language  
            - centrality_scores: PageRank centrality scores for files
            - git_stats: Git repository statistics (if available)
            - analysis_metadata: Metadata about the analysis run
    """
    if config is None:
        config = AnalysisConfig()
    
    # Create repository instance
    repo = Repository(repo_path)
    
    # Scan files
    files = await repo.scan_files(
        max_files=config.max_files,
        include_patterns=config.include_patterns if config.include_patterns else None,
        exclude_patterns=config.exclude_patterns if config.exclude_patterns else None,
        progress_callback=progress_callback
    )
    
    # Create scorer with custom weights
    scorer = create_scorer_with_weights(config.scoring_weights)
    
    # Score files 
    file_scores = await scorer.score_files(
        files,
        batch_size=config.batch_size,
        progress_callback=progress_callback
    )
    
    # Create PageRank analyzer
    analyzer = create_pagerank_analyzer_with_config(
        config.pagerank_damping,
        config.pagerank_max_iterations, 
        config.pagerank_tolerance
    )
    
    # Analyze dependencies
    centrality_scores = await analyzer.analyze_dependencies(
        files,
        include_external=False,
        progress_callback=progress_callback
    )
    
    # Combine scores with centrality
    final_scores = scorer.combine_with_centrality(
        file_scores,
        centrality_scores,
        config.scoring_weights.get("centrality", 0.2)
    )
    
    # Gather additional repository information
    repo_info = await repo.get_repository_info()
    language_stats = await repo.get_language_stats()
    size_stats = await repo.get_size_stats()
    
    results = {
        "file_scores": final_scores,
        "repository_info": repo_info,
        "language_stats": language_stats,
        "size_stats": size_stats,
        "centrality_scores": centrality_scores,
        "analysis_metadata": {
            "scribe_version": __version__,
            "files_analyzed": len(final_scores) if final_scores else 0,
            "config_used": config.to_dict(),
        }
    }
    
    # Add Git statistics if available
    if repo.has_git():
        try:
            git_stats = await repo.get_git_stats()
            results["git_stats"] = git_stats
        except Exception:
            # Git stats not available
            pass
    
    return results


# Add the comprehensive analysis function to __all__
__all__.append("analyze_repository_complete")
__all__.append("AnalysisConfig")
__all__.append("get_info")
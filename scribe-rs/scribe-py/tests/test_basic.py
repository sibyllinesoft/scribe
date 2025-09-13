"""
Basic tests for Scribe-RS Python bindings.

These tests verify that the bindings can be imported and basic functionality works.
"""

import pytest
import sys
from pathlib import Path

# Add the python package to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import scribe_rs
    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_import_basic():
    """Test that basic imports work."""
    from scribe_rs import (
        Repository, HeuristicScorer, PageRankAnalyzer, PatternMatcher,
        get_version_info, get_build_info, get_info
    )
    
    assert Repository is not None
    assert HeuristicScorer is not None
    assert PageRankAnalyzer is not None
    assert PatternMatcher is not None


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")  
def test_version_info():
    """Test version information functions."""
    from scribe_rs import get_version_info, get_build_info, get_info
    
    version_info = get_version_info()
    assert isinstance(version_info, dict)
    assert "version" in version_info
    assert "name" in version_info
    
    build_info = get_build_info()
    assert isinstance(build_info, dict)
    assert "version" in build_info
    
    info = get_info()
    assert isinstance(info, dict)
    assert "version" in info
    assert "build" in info
    assert "capabilities" in info


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_factory_functions():
    """Test factory functions work."""
    from scribe_rs import (
        create_default_scorer, create_pagerank_analyzer, 
        create_pattern_matcher, get_default_weights
    )
    
    scorer = create_default_scorer()
    assert scorer is not None
    
    analyzer = create_pagerank_analyzer()
    assert analyzer is not None
    
    matcher = create_pattern_matcher()
    assert matcher is not None
    
    weights = get_default_weights()
    assert isinstance(weights, dict)


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_utility_functions():
    """Test utility functions."""
    from scribe_rs import (
        get_supported_languages, validate_pattern,
        is_valid_repository, find_repository_root
    )
    
    languages = get_supported_languages()
    assert isinstance(languages, list)
    assert len(languages) > 0
    
    # Test pattern validation
    assert validate_pattern(r"\d+") == True
    assert validate_pattern(r"[a-zA-Z]+") == True
    assert validate_pattern(r"[") == False  # Invalid pattern
    
    # Test repository validation
    assert is_valid_repository(".") == True  # Current directory should exist
    assert is_valid_repository("/nonexistent/path") == False


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_exceptions():
    """Test custom exceptions are available."""
    from scribe_rs import (
        ScribeException, AnalysisException, 
        PatternException, ConfigurationException
    )
    
    assert ScribeException is not None
    assert AnalysisException is not None  
    assert PatternException is not None
    assert ConfigurationException is not None


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_analysis_config():
    """Test AnalysisConfig helper class."""
    from scribe_rs import AnalysisConfig
    
    config = AnalysisConfig()
    assert config.max_files == 10000
    assert isinstance(config.scoring_weights, dict)
    assert config.pagerank_damping == 0.85
    
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "max_files" in config_dict
    assert "scoring_weights" in config_dict


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_repository_creation():
    """Test repository creation with valid path."""
    from scribe_rs import Repository
    
    # Test with current directory
    repo = Repository(".")
    assert repo is not None
    assert repo.path == str(Path(".").resolve())
    
    # Test invalid path should raise exception
    with pytest.raises(Exception):  # Should raise FileNotFoundError
        Repository("/definitely/nonexistent/path")


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_scorer_creation():
    """Test heuristic scorer creation."""
    from scribe_rs import HeuristicScorer, get_default_weights
    
    # Test default scorer
    scorer = HeuristicScorer()
    assert scorer is not None
    
    # Test scorer with custom weights
    weights = get_default_weights()
    weights["documentation"] = 0.5  # Custom weight
    
    custom_scorer = HeuristicScorer(weights=weights)
    assert custom_scorer is not None
    
    # Test getting weights back
    retrieved_weights = custom_scorer.get_weights()
    assert isinstance(retrieved_weights, dict)


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_pagerank_analyzer_creation():
    """Test PageRank analyzer creation."""
    from scribe_rs import PageRankAnalyzer
    
    # Test default analyzer
    analyzer = PageRankAnalyzer()
    assert analyzer is not None
    
    # Test custom configuration
    custom_analyzer = PageRankAnalyzer(
        damping_factor=0.9,
        max_iterations=200,
        tolerance=1e-8
    )
    assert custom_analyzer is not None
    
    # Test getting config back
    config = custom_analyzer.get_config()
    assert isinstance(config, dict)
    assert config["damping_factor"] == 0.9
    assert config["max_iterations"] == 200
    assert config["tolerance"] == 1e-8


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_pattern_matcher_creation():
    """Test pattern matcher creation."""
    from scribe_rs import PatternMatcher
    
    matcher = PatternMatcher()
    assert matcher is not None
    
    # Test adding a rule
    rule = {
        "name": "test_rule",
        "pattern": r"\btest\b", 
        "language": "python",
        "description": "Test pattern"
    }
    
    matcher.add_rule(rule)
    
    # Test getting rules back
    rules = matcher.get_rules()
    assert isinstance(rules, list)
    assert len(rules) >= 1  # Should have at least our test rule


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Scribe-RS bindings not available")
def test_module_metadata():
    """Test module metadata is available.""" 
    import scribe_rs
    
    assert hasattr(scribe_rs, "__version__")
    assert hasattr(scribe_rs, "__author__")  
    assert hasattr(scribe_rs, "__license__")
    assert hasattr(scribe_rs, "__description__")
    
    assert isinstance(scribe_rs.__version__, str)
    assert len(scribe_rs.__version__) > 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
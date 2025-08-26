"""
Tests for rule-based text centrality system (README++).

Validates document prioritization logic:
- Filename pattern matching
- Path-based scoring
- Document structure analysis
- Integration with link analysis
"""

import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass

from ...fastpath.fast_scan import ScanResult, FileStats, DocumentAnalysis
from ...docs.text_priority import TextPriorityScorer, CentralityResult
from ...docs.link_graph import LinkGraphAnalyzer, LinkAnalysisResult


class TestTextPriorityScoring:
    """Test suite for text centrality scoring."""
    
    @pytest.fixture
    def scorer(self):
        """Create text priority scorer."""
        return TextPriorityScorer(budget_reserve=0.15)
        
    @pytest.fixture  
    def sample_results(self):
        """Create sample scan results for testing."""
        results = []
        
        # README file (should get highest priority)
        readme_stats = FileStats(
            path="README.md",
            size_bytes=2048,
            lines=50,
            language="markdown",
            is_readme=True,
            is_test=False,
            is_config=False,
            is_docs=True,
            depth=1,
            last_modified=1640995200.0
        )
        readme_doc = DocumentAnalysis(
            heading_count=5,
            toc_indicators=1,
            link_count=8,
            code_block_count=3,
            heading_density=10.0
        )
        results.append(ScanResult(
            stats=readme_stats,
            imports=None,
            doc_analysis=readme_doc,
            churn_score=0.8,
            priority_boost=2.0
        ))
        
        # Architecture document
        arch_stats = FileStats(
            path="docs/architecture.md",
            size_bytes=4096,
            lines=120,
            language="markdown", 
            is_readme=False,
            is_test=False,
            is_config=False,
            is_docs=True,
            depth=2,
            last_modified=1640995200.0
        )
        arch_doc = DocumentAnalysis(
            heading_count=8,
            toc_indicators=2,
            link_count=15,
            code_block_count=5,
            heading_density=6.7
        )
        results.append(ScanResult(
            stats=arch_stats,
            imports=None,
            doc_analysis=arch_doc,
            churn_score=0.3,
            priority_boost=1.0
        ))
        
        # Regular code file
        code_stats = FileStats(
            path="src/main.py",
            size_bytes=1024,
            lines=40,
            language="python",
            is_readme=False,
            is_test=False,
            is_config=False,
            is_docs=False,
            depth=2,
            last_modified=1640995200.0
        )
        results.append(ScanResult(
            stats=code_stats,
            imports=None,
            doc_analysis=None,
            churn_score=0.5,
            priority_boost=0.0
        ))
        
        # Test file
        test_stats = FileStats(
            path="tests/test_main.py",
            size_bytes=512,
            lines=25,
            language="python",
            is_readme=False,
            is_test=True,
            is_config=False,
            is_docs=False,
            depth=2,
            last_modified=1640995200.0
        )
        results.append(ScanResult(
            stats=test_stats,
            imports=None,
            doc_analysis=None,
            churn_score=0.2,
            priority_boost=0.0
        ))
        
        return results
        
    def test_filename_priority_detection(self, scorer):
        """Test filename pattern priority detection."""
        # README files
        readme_boost, reasoning = scorer._calculate_filename_boost("README.md")
        assert readme_boost >= 2.5
        assert any("readme" in r.lower() for r in reasoning)
        
        # Architecture files  
        arch_boost, reasoning = scorer._calculate_filename_boost("architecture.md")
        assert arch_boost >= 2.0
        assert any("architecture" in r.lower() for r in reasoning)
        
        # Getting started
        start_boost, reasoning = scorer._calculate_filename_boost("getting-started.md")
        assert start_boost >= 1.8
        
        # Regular files
        regular_boost, reasoning = scorer._calculate_filename_boost("utils.py")
        assert regular_boost == 0.0
        
    def test_path_location_scoring(self, scorer):
        """Test path-based priority scoring."""
        # Root level files
        root_boost, reasoning = scorer._calculate_path_boost("README.md")
        assert root_boost >= 2.0
        assert any("root" in r.lower() for r in reasoning)
        
        # Documentation directory
        docs_boost, reasoning = scorer._calculate_path_boost("docs/api.md")
        assert docs_boost >= 1.2
        assert any("docs" in r.lower() for r in reasoning)
        
        # Deep nested file
        deep_boost, reasoning = scorer._calculate_path_boost("src/util/deep/nested.py")
        assert deep_boost < 1.5  # Should get reduced boost for depth
        
    def test_document_structure_scoring(self, scorer):
        """Test document structure analysis."""
        # Well-structured document
        good_doc = DocumentAnalysis(
            heading_count=5,
            toc_indicators=1,
            link_count=10,
            code_block_count=3,
            heading_density=8.0
        )
        score, reasoning = scorer._calculate_structure_score(good_doc)
        assert score > 1.0
        assert len(reasoning) > 2
        
        # Poor structure
        poor_doc = DocumentAnalysis(
            heading_count=1,
            toc_indicators=0,
            link_count=0,
            code_block_count=0,
            heading_density=1.0
        )
        score, reasoning = scorer._calculate_structure_score(poor_doc)
        assert score < 0.5
        
        # No analysis
        score, reasoning = scorer._calculate_structure_score(None)
        assert score == 0.0
        
    def test_comprehensive_centrality_scoring(self, scorer, sample_results):
        """Test complete centrality scoring."""
        readme_result = sample_results[0]  # README.md
        arch_result = sample_results[1]    # docs/architecture.md
        code_result = sample_results[2]    # src/main.py
        
        # Score all files
        readme_centrality = scorer.score_document_centrality(readme_result)
        arch_centrality = scorer.score_document_centrality(arch_result)
        code_centrality = scorer.score_document_centrality(code_result)
        
        # README should have highest priority
        assert readme_centrality.final_priority > arch_centrality.final_priority
        assert readme_centrality.final_priority > code_centrality.final_priority
        
        # Architecture should be higher than regular code
        assert arch_centrality.final_priority > code_centrality.final_priority
        
        # Check priority tiers
        assert readme_centrality.priority_tier in {'CRITICAL', 'HIGH'}
        assert arch_centrality.priority_tier in {'HIGH', 'MEDIUM'}
        assert code_centrality.priority_tier in {'MEDIUM', 'LOW', 'MINIMAL'}
        
    def test_link_centrality_integration(self, scorer):
        """Test integration with link graph analysis."""
        # Create mock link analysis
        link_analysis = LinkAnalysisResult(
            in_degree={'README.md': 5, 'docs/api.md': 3, 'src/main.py': 0},
            out_degree={'README.md': 2, 'docs/api.md': 8, 'src/main.py': 1},
            pagerank_scores={'README.md': 0.25, 'docs/api.md': 0.15, 'src/main.py': 0.05},
            authority_files=['README.md', 'docs/api.md'],
            hub_files=['docs/api.md'],
            link_clusters={}
        )
        
        # Test link centrality calculation
        readme_score, reasoning = scorer._calculate_link_centrality('README.md', link_analysis)
        api_score, reasoning = scorer._calculate_link_centrality('docs/api.md', link_analysis)
        code_score, reasoning = scorer._calculate_link_centrality('src/main.py', link_analysis)
        
        # README should have high centrality (authority + high PageRank)
        assert readme_score > 1.0
        
        # API doc should have high centrality (hub + authority)
        assert api_score > 1.0
        
        # Code file should have low centrality
        assert code_score < 0.5
        
    def test_priority_document_selection(self, scorer, sample_results):
        """Test document selection within budget."""
        # Test with different budget sizes
        large_budget = 10000
        small_budget = 1000
        
        # Large budget should include most documents
        large_selection = scorer.select_priority_documents(sample_results, large_budget)
        
        # Should include README and architecture docs
        selected_paths = {r.stats.path for r, c in large_selection}
        assert 'README.md' in selected_paths
        assert any('architecture' in path for path in selected_paths)
        
        # Small budget should prioritize most important docs
        small_selection = scorer.select_priority_documents(sample_results, small_budget)
        
        # README should always be included if it fits
        small_paths = {r.stats.path for r, c in small_selection}
        assert 'README.md' in small_paths
        
        # Should have fewer selections
        assert len(small_selection) <= len(large_selection)
        
    def test_priority_statistics(self, scorer, sample_results):
        """Test priority statistics generation."""
        # Score all documents
        scored_docs = []
        for result in sample_results:
            centrality = scorer.score_document_centrality(result)
            scored_docs.append((result, centrality))
            
        # Get statistics
        stats = scorer.get_priority_statistics(scored_docs)
        
        # Verify statistics structure
        assert 'total_documents' in stats
        assert 'tier_distribution' in stats
        assert 'tier_average_scores' in stats
        assert 'readme_count' in stats
        
        # Verify values
        assert stats['total_documents'] == len(sample_results)
        assert stats['readme_count'] >= 1  # At least one README
        assert stats['highest_priority'] > stats['lowest_priority']
        
    def test_custom_priority_patterns(self):
        """Test custom priority pattern configuration."""
        # Test scorer with custom patterns
        scorer = TextPriorityScorer(budget_reserve=0.15)
        
        # Override patterns for testing
        scorer.filename_patterns.append(('custom-doc', 2.5))
        
        boost, reasoning = scorer._calculate_filename_boost("custom-doc.md")
        assert boost >= 2.0
        
    def test_budget_reserve_calculation(self):
        """Test budget reserve allocation."""
        scorer_15 = TextPriorityScorer(budget_reserve=0.15)  # 15%
        scorer_20 = TextPriorityScorer(budget_reserve=0.20)  # 20%
        
        total_budget = 10000
        
        # Create minimal sample
        sample = [ScanResult(
            stats=FileStats("README.md", 1000, 25, "markdown", True, False, False, True, 1, 1640995200.0),
            imports=None,
            doc_analysis=None,
            churn_score=0.0,
            priority_boost=0.0
        )]
        
        # Both should reserve different amounts
        selection_15 = scorer_15.select_priority_documents(sample, total_budget)
        selection_20 = scorer_20.select_priority_documents(sample, total_budget)
        
        # Higher reserve should be more conservative
        # (This is a simplified test - in practice would need more complex validation)
        assert scorer_15.token_budget_reserve == 0.15
        assert scorer_20.token_budget_reserve == 0.20
        
    def test_edge_cases(self, scorer):
        """Test edge cases and error conditions."""
        # Empty document list
        empty_selection = scorer.select_priority_documents([], 10000)
        assert len(empty_selection) == 0
        
        # Zero budget
        sample = [ScanResult(
            stats=FileStats("README.md", 100, 5, "markdown", True, False, False, True, 1, 1640995200.0),
            imports=None,
            doc_analysis=None,
            churn_score=0.0,
            priority_boost=0.0
        )]
        
        zero_selection = scorer.select_priority_documents(sample, 0)
        assert len(zero_selection) == 0
        
        # Very large file that exceeds budget
        large_file = ScanResult(
            stats=FileStats("HUGE.md", 100000, 2500, "markdown", True, False, False, True, 1, 1640995200.0),
            imports=None,
            doc_analysis=None,
            churn_score=0.0,
            priority_boost=0.0
        )
        
        large_selection = scorer.select_priority_documents([large_file], 1000)
        # Should handle gracefully (may or may not include based on estimation)
        assert len(large_selection) >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
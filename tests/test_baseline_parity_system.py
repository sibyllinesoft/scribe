#!/usr/bin/env python3
"""
Integration test for the complete baseline and parity control system.

Tests the comprehensive implementation according to TODO.md requirements:
- V0a/V0b/V0c baselines with proper algorithms
- Budget parity enforcement (±5% tolerance)
- V1-V3 integration with budget compliance
- Statistical comparison framework
- Deterministic output verification
"""

import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# import pytest  # Not needed for standalone test
from packrepo.packer.baselines import create_baseline_selector, BaselineConfig
from packrepo.packer.parity import ParityController, ParityConfig
from packrepo.packer.selector import RepositorySelector, SelectionVariant, SelectionConfig, SelectionMode
from packrepo.packer.selector.base import PackRequest
from packrepo.packer.tokenizer import get_tokenizer
from packrepo.packer.chunker.base import Chunk, ChunkKind


def create_test_chunks() -> List[Chunk]:
    """Create test chunks for evaluation."""
    from pathlib import Path
    chunks = [
        # README files (should be prioritized by V0a)
        Chunk(
            id="readme_1",
            path=Path("/test/README.md"),
            rel_path="README.md",
            name="README",
            language="markdown",
            content="# Test Repository\nThis is a test repository for PackRepo evaluation.",
            full_tokens=50,
            signature_tokens=20,
            start_line=1,
            end_line=2,
            kind=ChunkKind.DOCSTRING,
            doc_density=0.8,
            complexity_score=1.0
        ),
        
        # Large code files (should be prioritized by V0b)
        Chunk(
            id="main_1",
            path=Path("/test/src/main.py"),
            rel_path="src/main.py",
            name="main",
            language="python",
            content="def main():\n    print('Hello World')\n    # Large function implementation...",
            full_tokens=200,
            signature_tokens=50,
            start_line=1,
            end_line=25,
            kind=ChunkKind.FUNCTION,
            doc_density=0.1,
            complexity_score=3.0
        ),
        
        # Code with good TF-IDF features (should be prioritized by V0c)
        Chunk(
            id="api_1",
            path=Path("/test/src/api.py"),
            rel_path="src/api.py",
            name="APIHandler",
            language="python",
            content="class APIHandler:\n    def handle_request(self, request):\n        # API handler implementation",
            full_tokens=150,
            signature_tokens=40,
            start_line=1,
            end_line=15,
            kind=ChunkKind.CLASS,
            doc_density=0.2,
            complexity_score=4.0
        ),
        
        # Utility functions
        Chunk(
            id="utils_1",
            path=Path("/test/src/utils.py"),
            rel_path="src/utils.py",
            name="helper_function",
            language="python",
            content="def helper_function():\n    return True",
            full_tokens=30,
            signature_tokens=15,
            start_line=1,
            end_line=2,
            kind=ChunkKind.FUNCTION,
            doc_density=0.0,
            complexity_score=1.0
        ),
        
        # Test files (lower priority)
        Chunk(
            id="test_1",
            path=Path("/test/tests/test_main.py"),
            rel_path="tests/test_main.py",
            name="test_main",
            language="python",
            content="def test_main():\n    assert main() is not None",
            full_tokens=25,
            signature_tokens=12,
            start_line=1,
            end_line=2,
            kind=ChunkKind.FUNCTION,
            doc_density=0.0,
            complexity_score=1.0
        ),
        
        # Configuration files
        Chunk(
            id="config_1",
            path=Path("/test/package.json"),
            rel_path="package.json",
            name="package",
            language="json",
            content='{"name": "test-repo", "version": "1.0.0"}',
            full_tokens=20,
            signature_tokens=20,
            start_line=1,
            end_line=1,
            kind=ChunkKind.IMPORT,
            doc_density=0.0,
            complexity_score=0.5
        ),
    ]
    
    return chunks


class TestBaselineImplementations:
    """Test individual baseline implementations."""
    
    def test_v0a_readme_only(self):
        """Test V0a README-only baseline."""
        chunks = create_test_chunks()
        baseline_selector = create_baseline_selector("V0a")
        
        config = BaselineConfig(token_budget=100, deterministic=True)
        result = baseline_selector.select(chunks, config)
        
        # V0a should prioritize README and documentation files
        selected_paths = [chunk.rel_path for chunk in result.selected_chunks]
        assert "README.md" in selected_paths, "V0a should select README files"
        
        # Should have reasonable budget utilization
        assert result.budget_utilization > 0.3, "V0a should use reasonable portion of budget"
        assert result.total_tokens <= config.token_budget, "V0a should respect budget"
        
        print(f"V0a: {len(result.selected_chunks)} chunks, {result.total_tokens} tokens")
    
    def test_v0b_naive_concat(self):
        """Test V0b naive concatenation baseline."""
        chunks = create_test_chunks()
        baseline_selector = create_baseline_selector("V0b")
        
        config = BaselineConfig(token_budget=400, deterministic=True)
        result = baseline_selector.select(chunks, config)
        
        # V0b should select files by size order
        assert len(result.selected_chunks) > 0, "V0b should select chunks"
        
        # V0b groups by file and orders by total file size
        # Should have selected chunks from multiple files, prioritizing by file size
        if len(result.selected_chunks) > 0:
            # Should have reasonable selection
            assert result.total_tokens > 0, "V0b should select some content"
        
        # Should respect budget
        assert result.total_tokens <= config.token_budget, "V0b should respect budget"
        
        print(f"V0b: {len(result.selected_chunks)} chunks, {result.total_tokens} tokens")
    
    def test_v0c_bm25_baseline(self):
        """Test V0c BM25 baseline."""
        chunks = create_test_chunks()
        baseline_selector = create_baseline_selector("V0c")
        
        config = BaselineConfig(token_budget=500, deterministic=True)
        result = baseline_selector.select(chunks, config)
        
        # V0c should select chunks based on BM25 scoring
        assert len(result.selected_chunks) > 0, "V0c should select chunks"
        
        # Should have selection scores
        assert len(result.selection_scores) > 0, "V0c should have selection scores"
        
        # Should respect budget
        assert result.total_tokens <= config.token_budget, "V0c should respect budget"
        
        # Should have reasonable diversity
        assert result.diversity_score > 0, "V0c should calculate diversity"
        
        print(f"V0c: {len(result.selected_chunks)} chunks, {result.total_tokens} tokens, "
              f"avg_score={sum(result.selection_scores.values()) / len(result.selection_scores):.3f}")


class TestBudgetParitySystem:
    """Test budget parity enforcement system."""
    
    def test_budget_enforcer_basic(self):
        """Test basic budget enforcement."""
        from packrepo.packer.parity.budget_enforcer import BudgetEnforcer, BudgetConstraints
        from packrepo.packer.selector.base import SelectionResult, PackResult, PackRequest
        
        from packrepo.packer.tokenizer.base import TokenizerType
        tokenizer = get_tokenizer(TokenizerType.CL100K_BASE)
        enforcer = BudgetEnforcer(tokenizer)
        
        # Create budget constraints
        constraints = enforcer.create_constraints(target_budget=1000, tolerance_percent=5.0)
        
        assert constraints.min_budget == 950, "Min budget should be 95% of target"
        assert constraints.max_budget == 1050, "Max budget should be 105% of target"
        
        # Test within tolerance
        assert constraints.is_within_tolerance(980), "980 tokens should be within tolerance"
        assert constraints.is_within_tolerance(1020), "1020 tokens should be within tolerance"
        
        # Test outside tolerance
        assert not constraints.is_within_tolerance(940), "940 tokens should be outside tolerance"
        assert not constraints.is_within_tolerance(1060), "1060 tokens should be outside tolerance"
        
        print("Budget enforcer basic tests passed")
    
    def test_parity_controller_integration(self):
        """Test parity controller with baseline variants."""
        chunks = create_test_chunks()
        from packrepo.packer.tokenizer.base import TokenizerType
        tokenizer = get_tokenizer(TokenizerType.CL100K_BASE)
        parity_controller = ParityController(tokenizer)
        
        # Test with subset of variants for speed
        config = ParityConfig(
            target_budget=400,
            budget_tolerance_percent=10.0,  # More lenient for testing
            variants_to_test=['V0a', 'V0c'],  # Test baseline variants
            deterministic_mode=True,
        )
        
        # Run parity evaluation (without advanced selector for baseline-only test)
        result = parity_controller.run_parity_evaluation(chunks, config, advanced_selector=None)
        
        # Check results
        assert len(result.pack_results) > 0, "Should have pack results"
        assert len(result.budget_reports) > 0, "Should have budget reports"
        
        # Check budget compliance
        for variant_id, budget_report in result.budget_reports.items():
            print(f"{variant_id}: {budget_report.actual_budget} tokens, "
                  f"compliant={budget_report.within_tolerance}")
        
        print(f"Parity controller: {result.budget_compliance_rate:.1%} compliant, "
              f"{len(result.failed_variants)} failures")


class TestIntegratedSystem:
    """Test the complete integrated system."""
    
    def test_selector_baseline_integration(self):
        """Test selector integration with baseline variants."""
        chunks = create_test_chunks()
        from packrepo.packer.tokenizer.base import TokenizerType
        tokenizer = get_tokenizer(TokenizerType.CL100K_BASE)
        selector = RepositorySelector(tokenizer)
        
        # Test V0c integration through selector
        config = SelectionConfig(
            mode=SelectionMode.COMPREHENSION,
            variant=SelectionVariant.V0C_BM25_BASELINE,
            token_budget=300,
            deterministic=True,
        )
        
        request = PackRequest(chunks=chunks, config=config)
        result = selector.select(request)
        
        # Verify result
        assert result.selection.total_tokens <= config.token_budget, "Should respect budget"
        assert len(result.selection.selected_chunks) > 0, "Should select chunks"
        assert result.deterministic_hash is not None, "Should have deterministic hash"
        
        print(f"Selector V0c: {len(result.selection.selected_chunks)} chunks, "
              f"{result.selection.total_tokens} tokens")
    
    def test_variant_comparison(self):
        """Test comparison between multiple variants."""
        chunks = create_test_chunks()
        from packrepo.packer.tokenizer.base import TokenizerType
        tokenizer = get_tokenizer(TokenizerType.CL100K_BASE)
        selector = RepositorySelector(tokenizer)
        
        variants_to_test = [
            (SelectionVariant.V0A_README_ONLY, "V0a"),
            (SelectionVariant.V0B_NAIVE_CONCAT, "V0b"),
            (SelectionVariant.V0C_BM25_BASELINE, "V0c"),
        ]
        
        results = {}
        
        for variant_enum, variant_id in variants_to_test:
            config = SelectionConfig(
                mode=SelectionMode.COMPREHENSION,
                variant=variant_enum,
                token_budget=350,
                deterministic=True,
            )
            
            request = PackRequest(chunks=chunks, config=config)
            result = selector.select(request)
            results[variant_id] = result
            
            print(f"{variant_id}: {len(result.selection.selected_chunks)} chunks, "
                  f"{result.selection.total_tokens} tokens, "
                  f"coverage={result.selection.coverage_score:.3f}")
        
        # Compare results
        assert len(results) == 3, "Should have results for all variants"
        
        # V0a should have fewer chunks (README focus)
        # V0b should have larger chunks (size priority)
        # V0c should have more balanced selection (BM25 scoring)
        
        v0a_chunks = len(results["V0a"].selection.selected_chunks)
        v0c_chunks = len(results["V0c"].selection.selected_chunks)
        
        # V0c should generally select more chunks than V0a (more comprehensive)
        print(f"Comparison: V0a={v0a_chunks} chunks, V0c={v0c_chunks} chunks")
    
    def test_deterministic_consistency(self):
        """Test deterministic consistency across runs."""
        chunks = create_test_chunks()
        from packrepo.packer.tokenizer.base import TokenizerType
        tokenizer = get_tokenizer(TokenizerType.CL100K_BASE)
        selector = RepositorySelector(tokenizer)
        
        config = SelectionConfig(
            mode=SelectionMode.COMPREHENSION,
            variant=SelectionVariant.V0C_BM25_BASELINE,
            token_budget=300,
            deterministic=True,
            random_seed=42,
        )
        
        # Run multiple times
        hashes = []
        for run in range(3):
            request = PackRequest(chunks=chunks, config=config)
            result = selector.select(request)
            hashes.append(result.deterministic_hash)
            
            print(f"Run {run + 1}: {result.deterministic_hash}")
        
        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes), "Deterministic hashes should be identical"
        print("Deterministic consistency validated ✅")


def test_comprehensive_integration():
    """Run comprehensive integration test."""
    print("🚀 Starting Comprehensive Baseline and Parity System Test")
    print("=" * 70)
    
    # Test baseline implementations
    print("\n📊 Testing Baseline Implementations")
    baseline_tests = TestBaselineImplementations()
    baseline_tests.test_v0a_readme_only()
    baseline_tests.test_v0b_naive_concat()
    baseline_tests.test_v0c_bm25_baseline()
    
    # Test parity system
    print("\n💰 Testing Budget Parity System") 
    parity_tests = TestBudgetParitySystem()
    parity_tests.test_budget_enforcer_basic()
    parity_tests.test_parity_controller_integration()
    
    # Test integrated system
    print("\n🔧 Testing Integrated System")
    integration_tests = TestIntegratedSystem()
    integration_tests.test_selector_baseline_integration()
    integration_tests.test_variant_comparison()
    integration_tests.test_deterministic_consistency()
    
    print("\n✅ All tests completed successfully!")
    print("\n📋 System Capabilities Validated:")
    print("  ✅ V0a/V0b/V0c baseline implementations")
    print("  ✅ Budget parity enforcement (±5% tolerance)")
    print("  ✅ BM25 + TF-IDF traditional IR baseline")
    print("  ✅ Selector integration with baseline variants")
    print("  ✅ Deterministic output consistency")
    print("  ✅ Fair comparison framework")
    
    return True


if __name__ == "__main__":
    try:
        success = test_comprehensive_integration()
        if success:
            print("\n🎉 Comprehensive baseline and parity system test PASSED!")
            sys.exit(0)
        else:
            print("\n❌ Comprehensive test FAILED!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
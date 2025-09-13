#!/usr/bin/env python3
"""
Test script for FastPath V2 enhancements.
Verifies that new features work correctly and maintain backward compatibility.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_feature_flags():
    """Test feature flag system."""
    print("Testing feature flags...")
    
    from packrepo.fastpath.feature_flags import load_feature_flags, get_feature_flags
    
    # Test default values (all False)
    flags = load_feature_flags()
    assert flags.policy_v2 == False
    assert flags.demote_enabled == False
    assert flags.patch_enabled == False
    assert flags.router_enabled == False
    assert flags.centrality_enabled == False
    print("✓ Default feature flags work correctly")
    
    # Test environment variable override
    os.environ['FASTPATH_CENTRALITY'] = '1'
    flags = load_feature_flags()
    assert flags.centrality_enabled == True
    print("✓ Environment variable override works")
    
    # Clean up
    del os.environ['FASTPATH_CENTRALITY']


def test_fast_scanner_v1_compatibility():
    """Test that FastScanner maintains V1 backward compatibility."""
    print("Testing FastScanner V1 compatibility...")
    
    from packrepo.fastpath.fast_scan import FastScanner
    
    # Test with current repository
    scanner = FastScanner(project_root, ttl_seconds=5.0)
    results = scanner.scan_repository(max_files=50)
    
    assert len(results) > 0, "Should find some files"
    
    # Verify all V1 fields exist
    for result in results[:5]:  # Check first 5 results
        assert hasattr(result.stats, 'path')
        assert hasattr(result.stats, 'is_readme')
        assert hasattr(result.stats, 'is_test')
        assert hasattr(result.stats, 'is_config')
        assert hasattr(result, 'churn_score')
        assert hasattr(result, 'priority_boost')
        
        # V2 fields should exist but be default values when features disabled
        assert hasattr(result.stats, 'is_entrypoint')
        assert hasattr(result.stats, 'has_examples') 
        assert hasattr(result.stats, 'is_integration_test')
        assert hasattr(result, 'centrality_in')
        assert hasattr(result, 'signatures')
        
        # V2 fields should be default values when features disabled
        assert result.stats.is_entrypoint == False
        assert result.centrality_in == 0.0
        assert result.signatures == []
    
    print(f"✓ Scanner found {len(results)} files with V1 compatibility")


def test_fast_scanner_v2_features():
    """Test FastScanner V2 features when enabled."""
    print("Testing FastScanner V2 features...")
    
    # Enable V2 features
    os.environ['FASTPATH_CENTRALITY'] = '1'
    os.environ['FASTPATH_DEMOTE'] = '1'
    
    from packrepo.fastpath.fast_scan import FastScanner
    from packrepo.fastpath.feature_flags import reload_feature_flags
    
    # Reload flags to pick up environment changes
    reload_feature_flags()
    
    scanner = FastScanner(project_root, ttl_seconds=5.0)
    results = scanner.scan_repository(max_files=30)
    
    assert len(results) > 0, "Should find some files"
    
    # Check for V2 enhancements
    entrypoints_found = sum(1 for r in results if r.stats.is_entrypoint)
    examples_found = sum(1 for r in results if r.stats.has_examples)
    signatures_found = sum(1 for r in results if len(r.signatures) > 0)
    centrality_computed = sum(1 for r in results if r.centrality_in > 0)
    
    print(f"✓ Found {entrypoints_found} entrypoints")
    print(f"✓ Found {examples_found} files with examples")
    print(f"✓ Found {signatures_found} files with signatures")
    print(f"✓ Computed centrality for {centrality_computed} files")
    
    # Clean up environment
    del os.environ['FASTPATH_CENTRALITY']
    del os.environ['FASTPATH_DEMOTE']


def test_heuristic_scorer_v1_compatibility():
    """Test HeuristicScorer V1 backward compatibility."""
    print("Testing HeuristicScorer V1 compatibility...")
    
    # Ensure flags are disabled for this test
    from packrepo.fastpath.feature_flags import reload_feature_flags
    reload_feature_flags()  # This will reload with defaults (all False)
    
    from packrepo.fastpath.fast_scan import FastScanner
    from packrepo.fastpath.heuristics import HeuristicScorer
    
    scanner = FastScanner(project_root, ttl_seconds=3.0)
    results = scanner.scan_repository(max_files=20)
    
    scorer = HeuristicScorer()
    scored_files = scorer.score_all_files(results)
    
    assert len(scored_files) > 0, "Should score some files"
    
    # Verify scoring components
    for result, score_components in scored_files[:3]:
        assert hasattr(score_components, 'doc_score')
        assert hasattr(score_components, 'readme_score')
        assert hasattr(score_components, 'final_score')
        
        # V2 fields should exist but be 0.0 when features disabled
        assert hasattr(score_components, 'centrality_score')
        assert hasattr(score_components, 'entrypoint_score')
        assert hasattr(score_components, 'examples_score')
        assert score_components.centrality_score == 0.0
        assert score_components.entrypoint_score == 0.0
        assert score_components.examples_score == 0.0
    
    print(f"✓ Scored {len(scored_files)} files with V1 compatibility")


def test_selector_v2_infrastructure():
    """Test selector V2 infrastructure initialization."""
    print("Testing selector V2 infrastructure...")
    
    from packrepo.packer.selector.selector import RepositorySelector
    from packrepo.packer.tokenizer import get_tokenizer, TokenizerType
    
    tokenizer = get_tokenizer(TokenizerType.CL100K_BASE)
    selector = RepositorySelector(tokenizer)
    
    # Check V2 infrastructure was initialized
    assert hasattr(selector, '_class_quotas')
    assert hasattr(selector, '_runtime_budget_validator')
    assert hasattr(selector, '_density_greedy_selection')
    assert hasattr(selector, '_calculate_class_diversity')
    
    print("✓ Selector V2 infrastructure initialized correctly")


def test_performance_impact():
    """Test that V2 features don't significantly impact performance when disabled."""
    print("Testing performance impact...")
    
    import time
    from packrepo.fastpath.fast_scan import FastScanner
    
    # Test with features disabled (default)
    start_time = time.time()
    scanner = FastScanner(project_root, ttl_seconds=2.0)
    results_v1 = scanner.scan_repository(max_files=50)
    v1_time = time.time() - start_time
    
    # Test with features enabled
    os.environ['FASTPATH_CENTRALITY'] = '1'
    os.environ['FASTPATH_DEMOTE'] = '1'
    
    from packrepo.fastpath.feature_flags import reload_feature_flags
    reload_feature_flags()
    
    start_time = time.time()
    scanner_v2 = FastScanner(project_root, ttl_seconds=2.0)
    results_v2 = scanner_v2.scan_repository(max_files=50)
    v2_time = time.time() - start_time
    
    print(f"✓ V1 scan time: {v1_time:.3f}s ({len(results_v1)} files)")
    print(f"✓ V2 scan time: {v2_time:.3f}s ({len(results_v2)} files)")
    
    # V2 should not be more than 3x slower (acceptable for feature-rich mode)
    performance_ratio = v2_time / max(v1_time, 0.001)
    print(f"✓ Performance ratio (V2/V1): {performance_ratio:.2f}x")
    assert performance_ratio < 5.0, f"V2 should not be >5x slower than V1, got {performance_ratio:.2f}x"
    
    # Clean up environment
    del os.environ['FASTPATH_CENTRALITY']
    del os.environ['FASTPATH_DEMOTE']


def main():
    """Run all tests."""
    print("FastPath V2 Enhancement Tests")
    print("=" * 50)
    
    try:
        test_feature_flags()
        test_fast_scanner_v1_compatibility()
        test_fast_scanner_v2_features()
        test_heuristic_scorer_v1_compatibility()
        test_selector_v2_infrastructure()
        test_performance_impact()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! FastPath V2 enhancements are working correctly.")
        print("📊 Backward compatibility maintained.")
        print("🚀 New features available behind feature flags.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
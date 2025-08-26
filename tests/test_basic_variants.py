#!/usr/bin/env python3
"""Quick test of basic PackRepo variants to ensure they work."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from packrepo.library import RepositoryPacker
from packrepo.packer.selector.base import SelectionVariant


def test_variant(variant: str, name: str):
    """Test a single variant."""
    print(f"\n🧪 Testing {name} ({variant})...")
    
    try:
        packer = RepositoryPacker()
        pack = packer.pack_repository(
            project_root,
            token_budget=10000,
            variant=variant,
            deterministic=True,
            enable_oracles=False  # Skip oracles for quick test
        )
        
        print(f"✅ {name} successful:")
        print(f"   Tokens: {pack.index.actual_tokens}/{pack.index.target_budget}")
        print(f"   Utilization: {pack.index.budget_utilization:.1%}")
        print(f"   Chunks: {len(pack.index.chunks)}")
        return True
        
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return False


def main():
    """Test all variants quickly."""
    print("🚀 Quick Variant Test")
    
    variants_to_test = [
        ("baseline", "V0 Baseline"),
        ("comprehensive", "V1 Comprehensive"), 
        ("coverage_enhanced", "V2 Coverage"),
        ("stability_controlled", "V3 Stability")
    ]
    
    results = {}
    for variant, name in variants_to_test:
        results[name] = test_variant(variant, name)
    
    print(f"\n📊 Results Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} variants working")
    
    if passed == total:
        print("🎉 All variants working!")
        return 0
    else:
        print("⚠️ Some variants need fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
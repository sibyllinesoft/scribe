#!/usr/bin/env python3
"""
Quick V1 validation test

A simple, fast test to validate that the V1 implementation
is working correctly with basic deterministic output.
"""

import sys
import os
from pathlib import Path
import json
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packrepo.library import RepositoryPacker


def test_basic_deterministic():
    """Test basic deterministic packing."""
    print("🧪 Testing basic deterministic packing...")
    
    try:
        packer = RepositoryPacker()
        test_repo_path = project_root
        
        print("  📦 Generating pack...")
        pack = packer.pack_repository(
            test_repo_path,
            token_budget=2000,
            deterministic=True,
            enable_oracles=False  # Skip oracles for quick test
        )
        
        # Validate basic properties
        print(f"  ✓ Pack generated with {len(pack.index.chunks)} chunks")
        print(f"  ✓ Manifest digest: {pack.index.manifest_digest}")
        print(f"  ✓ Budget utilization: {pack.index.budget_info.utilization:.1%}")
        
        # Test deterministic output by generating twice
        print("  📦 Generating second pack...")
        pack2 = packer.pack_repository(
            test_repo_path,
            token_budget=2000,
            deterministic=True,
            enable_oracles=False
        )
        
        # Compare JSON output
        json1 = pack.to_json()
        json2 = pack2.to_json()
        
        hash1 = hashlib.sha256(json1.encode('utf-8')).hexdigest()
        hash2 = hashlib.sha256(json2.encode('utf-8')).hexdigest()
        
        if hash1 == hash2:
            print(f"  ✅ DETERMINISTIC: Both runs produced identical output")
            print(f"     Hash: {hash1[:16]}...")
            return True
        else:
            print(f"  ❌ NON-DETERMINISTIC: Runs produced different output")
            print(f"     Hash 1: {hash1[:16]}...")
            print(f"     Hash 2: {hash2[:16]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ Test failed with error: {str(e)}")
        return False


def test_oracle_basic():
    """Test basic oracle functionality."""
    print("\n🧪 Testing oracle validation...")
    
    try:
        packer = RepositoryPacker()
        test_repo_path = project_root
        
        print("  📦 Generating pack with oracles...")
        pack = packer.pack_repository(
            test_repo_path,
            token_budget=2000,
            deterministic=True,
            enable_oracles=True,
            oracle_categories=['determinism']  # Only test determinism oracle
        )
        
        print(f"  ✅ ORACLE VALIDATION: Pack generated successfully")
        print(f"     Chunks: {len(pack.index.chunks)}")
        print(f"     Manifest digest: {pack.index.manifest_digest}")
        return True
        
    except Exception as e:
        print(f"  ❌ Oracle validation failed: {str(e)}")
        return False


def test_manifest_integrity():
    """Test manifest digest integrity."""
    print("\n🧪 Testing manifest integrity...")
    
    try:
        packer = RepositoryPacker()
        test_repo_path = project_root
        
        pack = packer.pack_repository(
            test_repo_path,
            token_budget=2000,
            deterministic=True,
            enable_oracles=False
        )
        
        # Verify manifest digest matches body
        original_digest = pack.index.manifest_digest
        body_content = pack.body.format_body()
        recalculated_digest = pack.index.generate_manifest_digest(body_content)
        
        if original_digest == recalculated_digest:
            print(f"  ✅ INTEGRITY: Manifest digest matches body content")
            print(f"     Digest: {original_digest[:16]}...")
            return True
        else:
            print(f"  ❌ INTEGRITY FAILED: Digest mismatch")
            print(f"     Original: {original_digest[:16]}...")
            print(f"     Recalculated: {recalculated_digest[:16]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ Integrity test failed: {str(e)}")
        return False


def main():
    """Run quick V1 validation tests."""
    print("🚀 Quick V1 Validation Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_deterministic,
        test_manifest_integrity,
        test_oracle_basic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - V1 implementation is working!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed - implementation needs fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
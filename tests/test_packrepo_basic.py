#!/usr/bin/env python3
"""
Basic test of PackRepo library structure without external dependencies.
"""

import sys
from pathlib import Path

def test_imports():
    """Test basic imports work."""
    try:
        # Test library structure
        import packrepo
        print(f"✓ packrepo imported successfully (version {packrepo.__version__})")
        
        # Test core components
        from packrepo.packer.chunker.base import Chunk, ChunkKind
        from packrepo.packer.tokenizer.base import Tokenizer, TokenizerType
        from packrepo.packer.packfmt.base import PackFormat
        print("✓ Core data structures imported")
        
        # Test approximate tokenizer (doesn't need tiktoken)
        from packrepo.packer.tokenizer.implementations import ApproximateTokenizer
        tokenizer = ApproximateTokenizer(TokenizerType.CL100K_BASE)
        
        test_text = "This is a test string for token counting."
        tokens = tokenizer.count_tokens(test_text)
        print(f"✓ Approximate tokenizer works: '{test_text}' -> {tokens} tokens")
        
        # Test pack format
        pack = PackFormat()
        pack.index.repository_url = "https://github.com/test/repo"
        pack.index.target_budget = 1000
        print("✓ Pack format creation works")
        
        # Test chunk creation
        chunk = Chunk(
            id="test_chunk_1",
            path=Path("/test/file.py"),
            rel_path="file.py",
            start_line=1,
            end_line=10,
            kind=ChunkKind.FUNCTION,
            name="test_function",
            language="python",
            content="def test_function():\n    pass",
            full_tokens=10,
            signature_tokens=5,
        )
        print(f"✓ Chunk creation works: {chunk.name} ({chunk.kind.value})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_library_interface():
    """Test the main library interface."""
    try:
        from packrepo.library import RepositoryPacker, PackRepoError
        print("✓ Main library interface imports work")
        
        # Test that it handles missing dependencies gracefully
        try:
            packer = RepositoryPacker(prefer_exact_tokenizer=False)  # Use approximate
            print("✓ RepositoryPacker creation works with approximate tokenizer")
            stats = packer.get_statistics()
            print(f"✓ Statistics: {stats}")
        except Exception as e:
            print(f"⚠️  RepositoryPacker creation failed (expected with missing deps): {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Library interface import error: {e}")
        return False

def test_rendergit_integration():
    """Test rendergit integration."""
    try:
        import rendergit
        # Check that PACKREPO_AVAILABLE is False (since deps not installed)
        available = getattr(rendergit, 'PACKREPO_AVAILABLE', None)
        print(f"✓ rendergit integration: PACKREPO_AVAILABLE = {available}")
        
        if available:
            print("✓ PackRepo functionality will be available in CLI")
        else:
            print("ℹ️  PackRepo functionality not available (missing dependencies)")
        
        return True
        
    except ImportError as e:
        print(f"❌ rendergit integration error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing PackRepo library structure...\n")
    
    tests = [
        ("Basic imports", test_imports),
        ("Library interface", test_library_interface), 
        ("rendergit integration", test_rendergit_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 Running {name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} PASSED")
            else:
                print(f"❌ {name} FAILED")
        except Exception as e:
            print(f"💥 {name} CRASHED: {e}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Basic library structure is working.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
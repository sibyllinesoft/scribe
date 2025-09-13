#!/usr/bin/env python3
"""Test the main entry point module."""

import sys
import subprocess
import pytest
import pathlib
from unittest.mock import patch


class TestMainEntry:
    """Test the __main__.py entry point."""
    
    def test_main_entry_point(self):
        """Test that python -m scribe works."""
        # Test with --help to avoid needing full setup
        result = subprocess.run([
            sys.executable, "-m", "scribe", "--help"
        ], capture_output=True, text=True)
        
        # Should exit with code 0 for help
        assert result.returncode == 0
        assert "scribe" in result.stdout.lower()
    
    def test_main_entry_with_invalid_args(self):
        """Test that python -m scribe with invalid args returns non-zero."""
        result = subprocess.run([
            sys.executable, "-m", "scribe", "--nonexistent-flag"
        ], capture_output=True, text=True)
        
        # Should exit with non-zero for invalid args
        assert result.returncode != 0
    
    def test_main_module_import(self):
        """Test importing the __main__ module directly."""
        # This should cover the __main__.py file
        try:
            import scribe.__main__
            # If we get here, the module imported successfully
            assert True
        except ImportError:
            pytest.fail("Failed to import scribe.__main__")
    
    def test_main_module_execution_simulation(self):
        """Simulate the __main__ module execution."""
        # This is a more direct test of the __main__ logic
        with patch('sys.argv', ['scribe', '--help']):
            with pytest.raises(SystemExit):
                # Simulate what happens in __main__.py
                import scribe
                scribe.main()
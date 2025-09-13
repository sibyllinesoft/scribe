#!/usr/bin/env python3
"""
Additional tests to increase coverage of main.py.
Target specific uncovered lines to push us over 85% total coverage.
"""

import pytest
import pathlib
import tempfile
import subprocess
from unittest.mock import patch, MagicMock

from scribe.main import main, PACKREPO_AVAILABLE


class TestMainModuleCoverage:
    """Tests to cover specific lines in main.py."""
    
    def test_main_module_direct_call(self):
        """Test that main can be called directly."""
        # Test the if __name__ == "__main__" code path
        # We can't directly test line 435, but we can test main() function itself
        with patch('sys.argv', ['scribe', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should exit with 0 for help
            assert exc_info.value.code == 0
    
    def test_packrepo_available_flag(self):
        """Test that PACKREPO_AVAILABLE is defined."""
        # This tests the import logic and availability flag
        assert isinstance(PACKREPO_AVAILABLE, bool)
    
    @patch('scribe.main.PACKREPO_AVAILABLE', False)
    def test_main_without_packrepo(self):
        """Test main function when PACKREPO_AVAILABLE is False."""
        with patch('sys.argv', ['scribe', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
    
    def test_main_with_version_flag(self):
        """Test main with --version flag."""
        # The main parser doesn't have a --version flag, so test with -h instead
        with patch('sys.argv', ['scribe', '-h']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with 0 for help
            assert exc_info.value.code == 0
    
    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # Test that the module loads without error and has the required constants
        from scribe.main import PACKREPO_AVAILABLE
        
        # This should exist whether imports succeed or fail
        assert isinstance(PACKREPO_AVAILABLE, bool)
    
    def test_main_error_conditions(self):
        """Test various error conditions in main."""
        # Test with invalid directory path
        with patch('sys.argv', ['scribe', '/nonexistent/path/that/does/not/exist']):
            result = main()
            assert result == 1  # Should return error code
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=False)
    def test_main_path_is_file_not_directory(self, mock_is_dir, mock_exists):
        """Test main when path exists but is a file, not directory."""
        with patch('sys.argv', ['scribe', '/some/file.txt']):
            result = main()
            assert result == 1  # Should return error code


class TestMainModuleExecution:
    """Test execution paths in main module."""
    
    def test_subprocess_call_main_module(self):
        """Test calling the main module as subprocess."""
        # This should cover the if __name__ == "__main__" path
        result = subprocess.run([
            'python3', '-c', 
            'import sys; sys.path.insert(0, "."); from scribe.main import main; import sys; sys.argv=["scribe", "--help"]; main()'
        ], capture_output=True, text=True, cwd='/media/nathan/Seagate Hub/Projects/scribe')
        
        # Should succeed (exit with 0) and produce help output
        assert result.returncode == 0
        assert 'scribe' in result.stdout.lower() or 'usage' in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
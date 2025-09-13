#!/usr/bin/env python3
"""
Test module for command-line interface functionality in Scribe.

This module tests all aspects of the CLI including:
- Argument parsing
- Main function integration
- Output file handling
- Error handling
- Different execution modes (traditional, intelligent, editor)
"""

import pytest
import pathlib
import tempfile
import sys
import subprocess
from unittest.mock import patch, MagicMock, call
from argparse import Namespace

# Add the project root to the path so we can import scribe
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scribe import main, FileInfo, RenderDecision


class TestMainFunction:
    """Test the main function and overall CLI behavior."""
    
    @pytest.fixture
    def mock_file_info(self):
        """Create mock FileInfo objects for testing."""
        return [
            FileInfo(
                pathlib.Path("src/main.py"),
                "src/main.py",
                100,
                RenderDecision(True, "ok"),
                "print('hello')",
                5
            ),
            FileInfo(
                pathlib.Path("README.md"),
                "README.md",
                200,
                RenderDecision(True, "ok"),
                "# Test Project",
                4
            )
        ]
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_basic_execution(self, mock_load, mock_collect, mock_git, 
                                  mock_is_dir, mock_exists, mock_mkdir, 
                                  mock_stat, mock_write, mock_file_info):
        """Test basic main function execution."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--output-format", "cxml",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_collect.return_value = mock_file_info
        mock_load.side_effect = lambda x: x  # Return input unchanged
        
        # Create a proper mock stat result with st_size
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1024
        mock_stat.return_value = mock_stat_result
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_collect.assert_called_once()
        mock_write.assert_called_once()
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_with_include_patterns(self, mock_load, mock_collect, mock_git,
                                        mock_is_dir, mock_exists, mock_mkdir,
                                        mock_stat, mock_write, mock_file_info):
        """Test main function with --include flag."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--include", "*.py,*.md",
            "--output-format", "cxml",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_collect.return_value = mock_file_info
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        # Verify that collect_files was called with correct include patterns
        args = mock_collect.call_args[0]
        assert args[2] == ["*.py", "*.md"]  # include_patterns
        assert args[3] == []  # exclude_patterns
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_with_exclude_patterns(self, mock_load, mock_collect, mock_git,
                                        mock_is_dir, mock_exists, mock_mkdir,
                                        mock_stat, mock_write, mock_file_info):
        """Test main function with --exclude flag."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--exclude", "*.test.*,node_modules/**",
            "--output-format", "cxml",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_collect.return_value = mock_file_info
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        # Verify that collect_files was called with correct exclude patterns
        args = mock_collect.call_args[0]
        assert args[2] == []  # include_patterns
        assert args[3] == ["*.test.*", "node_modules/**"]  # exclude_patterns
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_with_both_patterns(self, mock_load, mock_collect, mock_git,
                                     mock_is_dir, mock_exists, mock_mkdir,
                                     mock_stat, mock_write, mock_file_info):
        """Test main function with both --include and --exclude flags."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--include", "*.py,*.js",
            "--exclude", "*.test.*",
            "--output-format", "cxml",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_collect.return_value = mock_file_info
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        # Verify that collect_files was called with correct patterns
        args = mock_collect.call_args[0]
        assert args[2] == ["*.py", "*.js"]  # include_patterns
        assert args[3] == ["*.test.*"]  # exclude_patterns
    
    def test_main_nonexistent_directory(self):
        """Test main function with non-existent directory."""
        test_args = [
            "scribe.py",
            "/nonexistent/directory",
            "--out", "/tmp/test_output.xml"
        ]
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 1  # Should return error code
    
    def test_main_file_instead_of_directory(self):
        """Test main function when path is a file instead of directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            test_args = [
                "scribe.py",
                temp_file.name,
                "--out", "/tmp/test_output.xml"
            ]
            
            with patch('sys.argv', test_args):
                result = main()
            
            assert result == 1  # Should return error code
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files', return_value=[])  # No files found
    def test_main_no_files_to_process(self, mock_collect, mock_git,
                                      mock_is_dir, mock_exists, mock_mkdir,
                                      mock_stat, mock_write):
        """Test main function when no files are found to process."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--out", "/tmp/test_output.xml"
        ]
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 1  # Should return error code
    
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_output_formats(self, mock_load, mock_collect, mock_git,
                                 mock_is_dir, mock_exists, mock_mkdir,
                                 mock_stat, mock_write, mock_file_info):
        """Test main function with different output formats."""
        formats = ["html", "cxml", "repomix"]
        
        mock_collect.return_value = mock_file_info
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        for output_format in formats:
            test_args = [
                "scribe.py",
                ".",
                "--force-traditional",
                "--output-format", output_format,
                "--out", f"/tmp/test_output.{output_format}"
            ]
            
            with patch('sys.argv', test_args):
                result = main()
            
            assert result == 0, f"Failed for format: {output_format}"
    
    def test_main_invalid_output_format(self):
        """Test main function with invalid output format."""
        test_args = [
            "scribe.py",
            ".",
            "--output-format", "invalid_format",
            "--out", "/tmp/test_output.txt"
        ]
        
        with patch('sys.argv', test_args):
            # argparse should exit with code 2 for invalid choice
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 2


class TestRemoteRepositoryHandling:
    """Test handling of remote Git repositories."""
    
    @patch('scribe.main.git_clone')
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('shutil.rmtree')
    def test_main_with_github_url(self, mock_rmtree, mock_mkdir, mock_stat,
                                  mock_write, mock_load, mock_collect,
                                  mock_git_commit, mock_git_clone):
        """Test main function with GitHub URL."""
        test_args = [
            "scribe.py",
            "https://github.com/user/repo.git",
            "--force-traditional",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_files = [
            FileInfo(pathlib.Path("test.py"), "test.py", 100, RenderDecision(True, "ok"), "content", 5)
        ]
        mock_collect.return_value = mock_files
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_git_clone.assert_called_once()
        mock_rmtree.assert_called_once()  # Should cleanup temp directory
    
    @patch('scribe.main.git_clone', side_effect=subprocess.CalledProcessError(1, 'git'))
    def test_main_git_clone_failure(self, mock_git_clone):
        """Test main function when git clone fails."""
        test_args = [
            "scribe.py",
            "https://github.com/user/nonexistent-repo.git",
            "--out", "/tmp/test_output.xml"
        ]
        
        with patch('sys.argv', test_args):
            # Should raise the subprocess error since git clone fails
            with pytest.raises(subprocess.CalledProcessError):
                main()


class TestEditorMode:
    """Test editor mode functionality."""
    
    @patch('scribe_editor.create_bundle_editor')
    @patch('webbrowser.open')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    def test_main_editor_mode(self, mock_is_dir, mock_exists, mock_browser, mock_create_editor):
        """Test main function in editor mode."""
        test_args = [
            "scribe.py",
            ".",
            "--editor",
            "--open"
        ]
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_create_editor.assert_called_once()
        mock_browser.assert_called_once()
    
    def test_main_editor_mode_with_remote_url(self):
        """Test editor mode with remote URL (should fail)."""
        test_args = [
            "scribe.py",
            "https://github.com/user/repo.git",
            "--editor"
        ]
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 1  # Should return error code
    
    @patch('scribe_editor.create_bundle_editor', side_effect=ImportError())
    def test_main_editor_mode_import_error(self, mock_create_editor):
        """Test editor mode when scribe_editor is not available."""
        test_args = [
            "scribe.py",
            ".",
            "--editor"
        ]
        
        with patch('sys.argv', test_args), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            result = main()
        
        assert result == 1  # Should return error code


class TestConfigurationHandling:
    """Test configuration file handling."""
    
    @patch('scribe.main.PACKREPO_AVAILABLE', True)
    @patch('packrepo.fastpath.config_manager.load_config')
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_with_config_file(self, mock_load, mock_collect, mock_git,
                                   mock_is_dir, mock_exists, mock_mkdir,
                                   mock_stat, mock_write, mock_load_config):
        """Test main function with configuration file."""
        # Mock config object
        mock_config = MagicMock()
        mock_config.output_file_path = "/custom/output/path.xml"
        mock_load_config.return_value = mock_config
        
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional"
            # No --out specified, should use config file path
        ]
        
        mock_files = [
            FileInfo(pathlib.Path("test.py"), "test.py", 100, RenderDecision(True, "ok"), "content", 5)
        ]
        mock_collect.return_value = mock_files
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_load_config.assert_called_once()
        # Should write to the path from config file
        mock_write.assert_called_once()


class TestIntelligentMode:
    """Test intelligent selection mode."""
    
    @patch('scribe.main.FASTPATH_AVAILABLE', True)
    @patch('scribe.main.should_use_intelligent_mode', return_value=True)
    @patch('scribe.main.select_files_fastpath')
    @patch('scribe.main.load_file_content')
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    def test_main_intelligent_mode(self, mock_git, mock_is_dir, mock_exists,
                                   mock_mkdir, mock_stat, mock_write,
                                   mock_load, mock_select_fastpath,
                                   mock_should_use_intelligent):
        """Test main function with intelligent mode."""
        test_args = [
            "scribe.py",
            ".",
            "--token-target", "50000",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_files = [
            FileInfo(pathlib.Path("test.py"), "test.py", 100, RenderDecision(True, "ok"), "content", 5)
        ]
        mock_select_fastpath.return_value = (mock_files, None)  # files, diff_content
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_select_fastpath.assert_called_once()
        # Should not call collect_files since intelligent mode was used
        
    @patch('scribe.FASTPATH_AVAILABLE', True)
    @patch('scribe.should_use_intelligent_mode', return_value=True)
    @patch('scribe.main.select_files_fastpath', side_effect=Exception("FastPath error"))
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    def test_main_intelligent_mode_fallback(self, mock_git, mock_is_dir, mock_exists,
                                            mock_mkdir, mock_stat, mock_write,
                                            mock_load, mock_collect, mock_select_fastpath,
                                            mock_should_use_intelligent):
        """Test intelligent mode fallback to traditional when FastPath fails."""
        test_args = [
            "scribe.py",
            ".",
            "--token-target", "50000",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_files = [
            FileInfo(pathlib.Path("test.py"), "test.py", 100, RenderDecision(True, "ok"), "content", 5)
        ]
        mock_collect.return_value = mock_files
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_select_fastpath.assert_called_once()
        mock_collect.assert_called_once()  # Should fallback to traditional


class TestBrowserIntegration:
    """Test browser opening functionality."""
    
    @patch('scribe.main.webbrowser.open')
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_with_open_flag(self, mock_load, mock_collect, mock_git,
                                 mock_is_dir, mock_exists, mock_mkdir,
                                 mock_stat, mock_write, mock_browser):
        """Test main function with --open flag."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--output-format", "html",
            "--open",
            "--out", "/tmp/test_output.html"
        ]
        
        mock_files = [
            FileInfo(pathlib.Path("test.py"), "test.py", 100, RenderDecision(True, "ok"), "content", 5)
        ]
        mock_collect.return_value = mock_files
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_browser.assert_called_once()
        # Verify it was called with file:// URL
        call_args = mock_browser.call_args[0][0]
        assert call_args.startswith("file://")
        assert call_args.endswith("test_output.html")
    
    @patch('webbrowser.open')
    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.stat')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('scribe.main.git_head_commit', return_value='abc123')
    @patch('scribe.main.collect_files')
    @patch('scribe.main.load_file_content')
    def test_main_open_flag_non_html(self, mock_load, mock_collect, mock_git,
                                     mock_is_dir, mock_exists, mock_mkdir,
                                     mock_stat, mock_write, mock_browser):
        """Test that --open flag is ignored for non-HTML formats."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--output-format", "cxml",
            "--open",
            "--out", "/tmp/test_output.xml"
        ]
        
        mock_files = [
            FileInfo(pathlib.Path("test.py"), "test.py", 100, RenderDecision(True, "ok"), "content", 5)
        ]
        mock_collect.return_value = mock_files
        mock_load.side_effect = lambda x: x
        mock_stat.return_value.st_size = 1024
        
        with patch('sys.argv', test_args):
            result = main()
        
        assert result == 0
        mock_browser.assert_not_called()  # Should not open browser for non-HTML


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_main_exception_in_file_processing(self):
        """Test main function when an exception occurs during file processing."""
        test_args = [
            "scribe.py",
            ".",
            "--out", "/tmp/test_output.xml"
        ]
        
        with patch('sys.argv', test_args), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('scribe.main.git_head_commit', side_effect=Exception("Git error")):
            
            # The exception should be allowed to propagate
            with pytest.raises(Exception):
                main()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
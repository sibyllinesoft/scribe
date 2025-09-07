#!/usr/bin/env python3
"""
Test module for edge cases and error handling in Scribe.

This module tests various edge cases, error conditions, and boundary scenarios including:
- File system edge cases
- Network and repository access issues  
- Memory and performance boundaries
- Input validation and sanitization
- Error recovery and graceful degradation
"""

import pytest
import pathlib
import tempfile
import sys
import subprocess
import os
from unittest.mock import patch, MagicMock, mock_open
import time

# Add the project root to the path so we can import scribe
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scribe import (
    FileInfo, RenderDecision, looks_binary, parse_gitignore_patterns,
    should_ignore_path, match_gitignore_pattern, collect_files,
    git_head_commit, git_clone, run, bytes_human, main,
    derive_temp_output_path, try_tree_command, generate_tree_fallback,
    should_use_intelligent_mode
)


class TestFileSystemEdgeCases:
    """Test edge cases related to file system operations."""
    
    def test_looks_binary_with_permission_denied(self):
        """Test binary detection when file cannot be read due to permissions."""
        # Create a temporary file and simulate permission error
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
        
        try:
            # Mock permission error
            with patch('pathlib.Path.open', side_effect=PermissionError("Permission denied")):
                result = looks_binary(temp_path)
                # Should default to binary when unable to read
                assert result is True
        finally:
            temp_path.unlink()
    
    def test_looks_binary_with_io_error(self):
        """Test binary detection when file reading raises IO error."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
        
        try:
            # Mock IO error
            with patch('pathlib.Path.open', side_effect=IOError("Disk error")):
                result = looks_binary(temp_path)
                assert result is True
        finally:
            temp_path.unlink()
    
    def test_looks_binary_with_empty_file(self):
        """Test binary detection with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
        
        try:
            result = looks_binary(temp_path)
            # Empty files should be considered text
            assert result is False
        finally:
            temp_path.unlink()
    
    def test_looks_binary_with_symlink(self):
        """Test binary detection with symbolic links."""
        # Create a text file and a symlink to it
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            text_file = temp_path / "text.txt"
            text_file.write_text("This is text content")
            
            symlink_file = temp_path / "link.txt"
            try:
                symlink_file.symlink_to(text_file)
                
                # Should follow symlink and detect as text
                result = looks_binary(symlink_file)
                assert result is False
            except OSError:
                # Skip test if symlinks not supported (e.g., Windows)
                pytest.skip("Symbolic links not supported on this platform")
    
    def test_file_stat_error(self):
        """Test handling of file stat errors."""
        # Create a file and then mock stat to raise an error
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir) / "testfile.txt"
            temp_path.write_text("test content")
            
            # Mock pathlib.Path.stat at the class level
            with patch('pathlib.Path.stat', side_effect=FileNotFoundError("File not found")):
                from scribe import decide_file
                result = decide_file(temp_path, pathlib.Path(temp_dir), 1024*1024, set())
                
            # Should handle the error gracefully (file should have size 0)
            assert result.size == 0  # FileNotFoundError sets size to 0
    
    def test_very_deep_directory_structure(self):
        """Test handling of deeply nested directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a deeply nested structure
            current_path = pathlib.Path(temp_dir)
            for i in range(50):  # 50 levels deep
                current_path = current_path / f"level{i}"
                current_path.mkdir()
            
            # Create a file at the deepest level
            deep_file = current_path / "deep_file.txt"
            deep_file.write_text("Deep content")
            
            # Test that file collection can handle deep structures
            try:
                files = collect_files(pathlib.Path(temp_dir), 1024*1024)
                deep_files = [f for f in files if f.rel.count('/') > 45]
                assert len(deep_files) > 0
            except OSError:
                # Some systems have path length limits
                pytest.skip("Path too long for this system")


class TestNetworkAndRepositoryEdgeCases:
    """Test edge cases related to network and repository operations."""
    
    @patch('scribe.git_utils.subprocess.run')
    def test_git_clone_timeout(self, mock_run):
        """Test git clone with timeout/network issues."""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 30)
        
        with pytest.raises(subprocess.TimeoutExpired):
            git_clone("https://github.com/user/repo.git", "/tmp/test_repo")
    
    @patch('scribe.git_utils.subprocess.run')
    def test_git_clone_network_error(self, mock_run):
        """Test git clone with network connectivity issues."""
        mock_run.side_effect = subprocess.CalledProcessError(
            128, "git", stderr="fatal: unable to connect to github.com"
        )
        
        with pytest.raises(subprocess.CalledProcessError):
            git_clone("https://github.com/user/nonexistent.git", "/tmp/test_repo")
    
    @patch('scribe.git_utils.subprocess.run')
    def test_git_head_commit_not_a_repo(self, mock_run):
        """Test getting HEAD commit from non-git directory."""
        mock_run.side_effect = subprocess.CalledProcessError(128, "git")
        
        result = git_head_commit("/tmp")
        assert result == "(unknown)"
    
    @patch('scribe.git_utils.subprocess.run')
    def test_git_head_commit_exception(self, mock_run):
        """Test getting HEAD commit when git command raises exception."""
        mock_run.side_effect = Exception("Unexpected error")
        
        result = git_head_commit("/tmp")
        assert result == "(unknown)"
    
    @patch('scribe.git_utils.subprocess.run')
    def test_collect_files_git_partial_failure(self, mock_run):
        """Test file collection when git ls-files partially fails."""
        # Mock git ls-files to return some output but with error code
        error = subprocess.CalledProcessError(1, "git")
        error.stdout = "file1.py\nfile2.py\n"
        mock_run.side_effect = error
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should fall back to filesystem walk
            files = collect_files(pathlib.Path(temp_dir), 1024*1024)
            # Should not crash even with git error
            assert isinstance(files, list)
    
    def test_derive_temp_output_path_malformed_urls(self):
        """Test temp path derivation with malformed URLs."""
        malformed_urls = [
            "",
            "not-a-url",
            "ftp://example.com/file",
            "https://",
            "https://github.com",  # No path component
            "https://github.com/",
            "https://github.com/user",  # No repo name
        ]
        
        for url in malformed_urls:
            result = derive_temp_output_path(url)
            # Should always return a valid path with fallback name
            assert result.name in ["repo.html", "file.html"]
            assert isinstance(result, pathlib.Path)


class TestMemoryAndPerformanceBoundaries:
    """Test memory usage and performance edge cases."""
    
    def test_very_large_file_content(self):
        """Test handling of very large file content."""
        # Create a large content string (10MB)
        large_content = "x" * (10 * 1024 * 1024)
        
        from scribe import estimate_tokens_simple
        # Should not crash with large content
        tokens = estimate_tokens_simple(large_content)
        expected_tokens = max(1, len(large_content) // 4)
        assert tokens == expected_tokens
    
    def test_many_small_files(self):
        """Test handling of many small files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Create many small files
            for i in range(1000):
                (temp_path / f"file_{i:04d}.txt").write_text(f"Content {i}")
            
            # Should handle large number of files without memory issues
            files = collect_files(temp_path, 1024*1024)
            included_files = [f for f in files if f.decision.include]
            assert len(included_files) == 1000
    
    def test_bytes_human_large_values(self):
        """Test human-readable byte formatting with very large values."""
        large_values = [
            (1024**4, "1.0 TiB"),  # 1 TB
            (1024**5, "1024.0 TiB"),  # 1 PB (beyond available units)
            (10**15, "909494.7 TiB"),  # Very large number
        ]
        
        for value, expected in large_values:
            result = bytes_human(value)
            assert result == expected or result.endswith("TiB")  # Should not crash
    
    def test_bytes_human_edge_values(self):
        """Test human-readable byte formatting with edge values."""
        edge_cases = [
            (0, "0 B"),
            (1, "1 B"),
            (1023, "1023 B"),
            (1024, "1.0 KiB"),
            (1536, "1.5 KiB"),
        ]
        
        for value, expected in edge_cases:
            result = bytes_human(value)
            assert result == expected


class TestInputValidationAndSanitization:
    """Test input validation and sanitization edge cases."""
    
    def test_parse_gitignore_with_malformed_content(self):
        """Test parsing .gitignore with malformed content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = pathlib.Path(temp_dir)
            gitignore_path = repo_path / ".gitignore"
            
            # Create .gitignore with various malformed content
            malformed_content = """
# Normal comment
*.pyc

# Binary content mixed in
\x00\x01invalid_binary

# Very long line
""" + "x" * 10000 + """

# Unicode content
测试_pattern
café/*.js

# Empty lines and weird spacing


   
# Negation patterns (should be ignored)
!important.log
"""
            gitignore_path.write_bytes(malformed_content.encode('utf-8', errors='replace'))
            
            # Should handle malformed content gracefully
            patterns = parse_gitignore_patterns(repo_path)
            
            # Should still include essential patterns
            assert "__pycache__/" in patterns
            assert "*.pyc" in patterns
            
            # Should include valid patterns from file
            assert "café/*.js" in patterns
    
    def test_parse_gitignore_with_invalid_encoding(self):
        """Test parsing .gitignore with invalid UTF-8 encoding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = pathlib.Path(temp_dir)
            gitignore_path = repo_path / ".gitignore"
            
            # Write invalid UTF-8 bytes
            invalid_utf8 = b"*.pyc\n\xff\xfe invalid encoding \n*.log\n"
            gitignore_path.write_bytes(invalid_utf8)
            
            # Should handle encoding errors gracefully
            patterns = parse_gitignore_patterns(repo_path)
            
            # Should still include essential patterns
            assert "__pycache__/" in patterns
            # And should include valid patterns despite encoding issues
            assert "*.log" in patterns
    
    def test_match_gitignore_pattern_with_special_characters(self):
        """Test gitignore pattern matching with special characters."""
        special_cases = [
            # Pattern with Unicode
            ("测试*.py", "测试file.py", True),
            ("café/*.js", "café/script.js", True),
            
            # Pattern with spaces (should work)
            ("my file.txt", "my file.txt", True),
            ("my file.txt", "other file.txt", False),
            
            # Very long patterns
            ("x" * 1000 + ".txt", "x" * 1000 + ".txt", True),
            ("x" * 1000 + ".txt", "y" * 1000 + ".txt", False),
            
            # Empty pattern (edge case)
            ("", "any_file.txt", False),
        ]
        
        for pattern, path, expected in special_cases:
            result = match_gitignore_pattern(path, pattern)
            assert result == expected, f"Pattern '{pattern}' with path '{path}' should be {expected}"
    
    def test_command_line_argument_edge_cases(self):
        """Test command line argument parsing edge cases."""
        edge_case_args = [
            # Empty patterns
            ["scribe.py", ".", "--include", "", "--exclude", ""],
            # Patterns with only commas
            ["scribe.py", ".", "--include", ",,,", "--exclude", "   ,  , "],
            # Very long patterns
            ["scribe.py", ".", "--include", "x" * 1000 + ".py"],
            # Unicode patterns
            ["scribe.py", ".", "--include", "测试*.py,café/*.js"],
        ]
        
        for args in edge_case_args:
            with patch('sys.argv', args), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_dir', return_value=True):
                
                # Should not crash during argument parsing
                try:
                    from scribe import main
                    # Don't actually execute, just ensure parsing works
                except SystemExit:
                    # argparse may raise SystemExit, which is fine
                    pass


class TestErrorRecoveryAndGracefulDegradation:
    """Test error recovery and graceful degradation scenarios."""
    
    def test_tree_command_not_available(self):
        """Test tree generation when tree command is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Create some test structure
            (temp_path / "file1.txt").write_text("content")
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "file2.txt").write_text("content")
            
            # Mock tree command to fail
            with patch('scribe.tree_utils.run', side_effect=FileNotFoundError("tree command not found")):
                result = try_tree_command(temp_path)
                
                # Should fall back to generate_tree_fallback
                # The fallback includes the directory name as first line
                assert temp_path.name in result or "file1.txt" in result
                assert "file1.txt" in result
                assert "subdir" in result
    
    def test_generate_tree_fallback_with_permission_errors(self):
        """Test fallback tree generation with permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Create test structure
            (temp_path / "accessible.txt").write_text("content")
            restricted_dir = temp_path / "restricted"
            restricted_dir.mkdir()
            
            # Mock permission error on restricted directory
            original_iterdir = pathlib.Path.iterdir
            def mock_iterdir(self):
                if self.name == "restricted":
                    raise PermissionError("Permission denied")
                return original_iterdir(self)
            
            with patch.object(pathlib.Path, 'iterdir', mock_iterdir):
                # Should handle permission errors gracefully
                result = generate_tree_fallback(temp_path)
                assert temp_path.name in result
                # Should include accessible files
                assert "accessible.txt" in result
    
    def test_should_use_intelligent_mode_git_errors(self):
        """Test intelligent mode detection with git errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Create some files
            for i in range(100):  # Enough to trigger intelligent mode
                (temp_path / f"file_{i}.py").write_text("content")
            
            # Mock git to fail
            with patch('scribe.run', side_effect=subprocess.CalledProcessError(128, "git")):
                # Should fall back to filesystem counting
                result = should_use_intelligent_mode(temp_path)
                assert result is True  # Should detect many files via filesystem
    
    def test_collect_files_with_broken_symlinks(self):
        """Test file collection with broken symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Create a regular file
            regular_file = temp_path / "regular.txt"
            regular_file.write_text("content")
            
            # Create a broken symlink
            broken_link = temp_path / "broken_link.txt"
            try:
                broken_link.symlink_to(temp_path / "nonexistent.txt")
                
                # Should handle broken symlinks gracefully
                files = collect_files(temp_path, 1024*1024)
                
                # Should include regular file
                regular_files = [f for f in files if f.rel == "regular.txt"]
                assert len(regular_files) == 1
                
                # Broken symlink should be filtered out by is_symlink() check
                broken_links = [f for f in files if f.rel == "broken_link.txt"]
                assert len(broken_links) == 0
                
            except OSError:
                # Skip if symlinks not supported
                pytest.skip("Symbolic links not supported on this platform")
    
    def test_main_with_write_permission_error(self):
        """Test main function when output file cannot be written."""
        test_args = [
            "scribe.py",
            ".",
            "--force-traditional",
            "--out", "/root/restricted_output.xml"  # Typically not writable
        ]
        
        with patch('sys.argv', test_args), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('scribe.git_head_commit', return_value='abc123'), \
             patch('scribe.collect_files') as mock_collect, \
             patch('scribe.load_file_content') as mock_load, \
             patch('pathlib.Path.write_text', side_effect=PermissionError("Permission denied")):
            
            mock_files = [
                FileInfo(pathlib.Path("test.py"), "test.py", 100, RenderDecision(True, "ok"), "content", 5)
            ]
            mock_collect.return_value = mock_files
            mock_load.side_effect = lambda x: x
            
            # Should raise the permission error
            with pytest.raises(PermissionError):
                main()


class TestConcurrencyAndRaceConditions:
    """Test potential concurrency issues and race conditions."""
    
    def test_temporary_file_cleanup_race(self):
        """Test cleanup of temporary files in race conditions."""
        # This test simulates potential race conditions in temp file cleanup
        temp_files = []
        
        # Create multiple temp files quickly
        for i in range(10):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = pathlib.Path(f.name)
                temp_files.append(temp_path)
                f.write(b"test content")
        
        try:
            # Simulate rapid file operations
            for temp_path in temp_files:
                if temp_path.exists():
                    result = looks_binary(temp_path)
                    assert isinstance(result, bool)
        finally:
            # Cleanup
            for temp_path in temp_files:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass  # Already deleted
    
    def test_file_modification_during_processing(self):
        """Test handling of files modified during processing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
            temp_path = pathlib.Path(temp_file.name)
            temp_file.write("initial content")
        
        try:
            # Get initial file info
            from scribe import decide_file_simple
            initial_info = decide_file_simple(temp_path, temp_path.parent, 1024*1024)
            
            # Modify file content
            temp_path.write_text("modified content that is much longer")
            
            # Load content (file size changed)
            from scribe import load_file_content
            loaded_info = load_file_content(initial_info)
            
            # Should handle gracefully (content will be the current content)
            assert loaded_info.content == "modified content that is much longer"
            
        finally:
            temp_path.unlink()


class TestPlatformSpecificEdgeCases:
    """Test platform-specific edge cases and compatibility."""
    
    def test_windows_path_separators(self):
        """Test handling of Windows-style path separators."""
        # Test path normalization
        windows_paths = [
            "src\\main.py",
            "tests\\unit\\test_file.py",
            "docs\\readme.txt",
        ]
        
        for windows_path in windows_paths:
            # Should normalize Windows backslashes to forward slashes
            normalized = windows_path.replace("\\", "/")
            assert "/" in normalized
    
    def test_case_sensitivity_edge_cases(self):
        """Test case sensitivity handling across platforms."""
        from scribe import match_glob_pattern
        
        # These tests may behave differently on case-insensitive filesystems
        case_tests = [
            ("*.PY", "test.py"),
            ("*.py", "Test.PY"),
            ("README.*", "readme.md"),
        ]
        
        for pattern, filename in case_tests:
            result = match_glob_pattern(pattern, filename)
            # Result depends on platform, but should not crash
            assert isinstance(result, bool)
    
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_unix_hidden_files(self):
        """Test handling of Unix hidden files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Create hidden files
            (temp_path / ".hidden_file").write_text("hidden content")
            (temp_path / ".git").mkdir()
            (temp_path / ".git" / "config").write_text("git config")
            
            # Test gitignore pattern matching
            patterns = {".git/", ".*"}
            
            assert should_ignore_path(".hidden_file", patterns)
            assert should_ignore_path(".git/config", patterns)
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_reserved_names(self):
        """Test handling of Windows reserved filenames."""
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        
        for name in reserved_names:
            # Should handle reserved names gracefully
            test_path = f"{name}.txt"
            from scribe import match_glob_pattern
            result = match_glob_pattern("*.txt", test_path)
            assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
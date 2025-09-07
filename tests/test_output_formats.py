#!/usr/bin/env python3
"""
Test module for output format generation in Scribe.

This module tests all aspects of output format generation including:
- CXML format generation
- Repomix format generation
- HTML format generation
- File icon detection
- Utility formatting functions
"""

import pytest
import pathlib
import tempfile
import time
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the path so we can import scribe
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scribe import (
    FileInfo, RenderDecision, generate_cxml_text, generate_repomix_text,
    get_file_icon, slugify, build_html, derive_temp_output_path
)


class TestCXMLFormat:
    """Test CXML format generation."""
    
    @pytest.fixture
    def sample_files(self):
        """Create sample FileInfo objects for testing."""
        return [
            FileInfo(
                pathlib.Path("src/main.py"),
                "src/main.py",
                100,
                RenderDecision(True, "ok"),
                "print('Hello, World!')\n",
                5
            ),
            FileInfo(
                pathlib.Path("README.md"),
                "README.md",
                200,
                RenderDecision(True, "ok"),
                "# Test Project\nThis is a test.\n",
                8
            ),
            FileInfo(
                pathlib.Path("package.json"),
                "package.json",
                150,
                RenderDecision(True, "ok"),
                '{"name": "test-project", "version": "1.0.0"}\n',
                10
            )
        ]
    
    def test_generate_cxml_text_basic(self, sample_files):
        """Test basic CXML format generation."""
        repo_url = "https://github.com/test/repo"
        head_commit = "abc123"
        
        result = generate_cxml_text(sample_files, repo_url, head_commit)
        
        # Check document structure
        assert "<documents>" in result
        assert "</documents>" in result
        
        # Check metadata
        assert repo_url in result
        assert head_commit in result
        assert "Files: 3" in result
        assert "Estimated tokens: 23" in result  # 5 + 8 + 10
        
        # Check file content
        assert "src/main.py" in result
        assert "README.md" in result
        assert "package.json" in result
        
        # Check HTML escaping
        assert "print(&#x27;Hello, World!&#x27;)" in result
        assert "&quot;name&quot;: &quot;test-project&quot;" in result
        
        # Check document structure for each file
        assert '<document index="1">' in result
        assert '<document index="2">' in result
        assert '<document index="3">' in result
        assert "<document_content>" in result
        assert "</document_content>" in result
    
    def test_generate_cxml_text_with_diff(self, sample_files):
        """Test CXML format generation with diff content."""
        repo_url = "https://github.com/test/repo"
        head_commit = "abc123"
        diff_content = "diff --git a/test.py b/test.py\n+print('new line')\n-old_line()"
        
        result = generate_cxml_text(sample_files, repo_url, head_commit, diff_content)
        
        # Should contain regular files
        assert "src/main.py" in result
        
        # Should contain diff section
        assert '<document index="diffs">' in result
        assert "Repository Diffs (Relevance Filtered)" in result
        assert "print(&#x27;new line&#x27;)" in result  # HTML escaped
        assert "old_line()" in result
    
    def test_generate_cxml_text_empty_files(self):
        """Test CXML format generation with empty file list."""
        repo_url = "https://github.com/test/repo"
        head_commit = "abc123"
        
        result = generate_cxml_text([], repo_url, head_commit)
        
        assert "<documents>" in result
        assert "</documents>" in result
        assert "Files: 0" in result
        assert "Estimated tokens: 0" in result
    
    def test_generate_cxml_text_file_metadata(self, sample_files):
        """Test CXML format includes file metadata."""
        repo_url = "https://github.com/test/repo"
        head_commit = "abc123"
        
        result = generate_cxml_text(sample_files, repo_url, head_commit)
        
        # Check for file size and token metadata
        assert "Size: 100 B" in result
        assert "Size: 200 B" in result
        assert "Size: 150 B" in result
        assert "Tokens: ~5" in result
        assert "Tokens: ~8" in result
        assert "Tokens: ~10" in result


class TestRepomixFormat:
    """Test Repomix format generation."""
    
    @pytest.fixture
    def sample_files(self):
        """Create sample FileInfo objects for testing."""
        return [
            FileInfo(
                pathlib.Path("src/main.py"),
                "src/main.py",
                100,
                RenderDecision(True, "ok"),
                "print('Hello, World!')\n",
                5
            ),
            FileInfo(
                pathlib.Path("README.md"),
                "README.md",
                200,
                RenderDecision(True, "ok"),
                "# Test Project\nThis is a test.\n",
                8
            )
        ]
    
    def test_generate_repomix_text_basic(self, sample_files):
        """Test basic Repomix format generation."""
        repo_url = "https://github.com/test/repo"
        head_commit = "abc123"
        
        with patch('time.strftime', return_value='2023-01-01 12:00:00 UTC'):
            result = generate_repomix_text(sample_files, repo_url, head_commit)
        
        # Check header information
        assert f"Repository: {repo_url}" in result
        assert f"Commit: {head_commit}" in result
        assert "Generated: 2023-01-01 12:00:00 UTC" in result
        
        # Check file statistics
        assert "Files: 2" in result
        assert "Estimated tokens: 13" in result  # 5 + 8
        
        # Check file sections
        assert "File: src/main.py" in result
        assert "File: README.md" in result
        assert "Size: 100 B" in result
        assert "Size: 200 B" in result
        assert "Tokens: 5" in result
        assert "Tokens: 8" in result
        
        # Check file content (not HTML escaped in repomix format)
        assert "print('Hello, World!')" in result
        assert "# Test Project" in result
        
        # Check separators
        assert "---" in result
    
    def test_generate_repomix_text_with_diff(self, sample_files):
        """Test Repomix format generation with diff content."""
        repo_url = "https://github.com/test/repo"
        head_commit = "abc123"
        diff_content = "diff --git a/test.py b/test.py\n+print('new line')"
        
        result = generate_repomix_text(sample_files, repo_url, head_commit, diff_content)
        
        # Should contain regular files
        assert "File: src/main.py" in result
        
        # Should contain diff section
        assert "File: Repository Diffs (Relevance Filtered)" in result
        assert "Type: Git diffs" in result
        assert diff_content in result
    
    def test_generate_repomix_text_no_token_estimates(self):
        """Test Repomix format with files that have no token estimates."""
        files_without_tokens = [
            FileInfo(
                pathlib.Path("src/main.py"),
                "src/main.py",
                100,
                RenderDecision(True, "ok"),
                "print('Hello, World!')\n",
                None  # No token estimate
            )
        ]
        
        repo_url = "https://github.com/test/repo"
        head_commit = "abc123"
        
        result = generate_repomix_text(files_without_tokens, repo_url, head_commit)
        
        assert "File: src/main.py" in result
        assert "Size: 100 B" in result
        # Should handle missing token estimates gracefully
        assert "Estimated tokens: 0" in result


class TestHTMLFormat:
    """Test HTML format generation."""
    
    @pytest.fixture
    def sample_files(self):
        """Create sample FileInfo objects for testing."""
        return [
            FileInfo(
                pathlib.Path("src/main.py"),
                "src/main.py",
                100,
                RenderDecision(True, "ok"),
                "print('Hello, World!')\n",
                5
            ),
            FileInfo(
                pathlib.Path("README.md"),
                "README.md",
                200,
                RenderDecision(True, "ok"),
                "# Test Project\n",
                4
            )
        ]
    
    def test_build_html_basic(self, sample_files):
        """Test basic HTML format generation."""
        repo_url = "https://github.com/test/repo"
        repo_dir = pathlib.Path("/test/repo")
        head_commit = "abc123"
        
        result = build_html(repo_url, repo_dir, head_commit, sample_files)
        
        # Check HTML structure
        assert "<!DOCTYPE html>" in result
        assert "<html lang=\"en\">" in result
        assert "</html>" in result
        
        # Check header content
        assert "Repository Analysis" in result
        assert repo_url in result
        assert head_commit in result
        
        # Check stats section
        assert "2" in result  # file count
        assert "9" in result  # total tokens (5 + 4)
        
        # Check table of contents
        assert "Table of Contents" in result
        assert "src/main.py" in result
        assert "README.md" in result
        
        # Check file content sections
        assert "print(&#x27;Hello, World!&#x27;)" in result  # HTML escaped
        assert "# Test Project" in result
        
        # Check CSS is included
        assert "<style>" in result
        assert "</style>" in result
        
        # Check JavaScript for icons
        assert "lucide.createIcons()" in result
    
    def test_build_html_with_diff(self, sample_files):
        """Test HTML format generation with diff content."""
        repo_url = "https://github.com/test/repo"
        repo_dir = pathlib.Path("/test/repo")
        head_commit = "abc123"
        diff_content = "diff --git a/test.py b/test.py\n+print('new line')"
        
        result = build_html(repo_url, repo_dir, head_commit, sample_files, diff_content)
        
        # Should contain regular files
        assert "src/main.py" in result
        
        # Should contain diff section
        assert "Repository Diffs" in result
        assert "print(&#x27;new line&#x27;)" in result  # HTML escaped
        assert 'id="diffs"' in result
    
    def test_build_html_empty_files(self):
        """Test HTML format generation with empty file list."""
        repo_url = "https://github.com/test/repo"
        repo_dir = pathlib.Path("/test/repo")
        head_commit = "abc123"
        
        result = build_html(repo_url, repo_dir, head_commit, [])
        
        # Should still generate valid HTML
        assert "<!DOCTYPE html>" in result
        assert "Repository Analysis" in result
        assert "0" in result  # file count should be 0


class TestFileIcons:
    """Test file icon detection functionality."""
    
    def test_get_file_icon_special_files(self):
        """Test icon detection for special files."""
        test_cases = [
            ("README.md", "book-open"),
            ("readme.txt", "book-open"),
            ("LICENSE", "scale"),
            ("Dockerfile", "box"),
            ("docker-compose.yml", "box"),
            ("Makefile", "settings"),
            ("package.json", "package"),
            ("tsconfig.json", "settings"),
            ("requirements.txt", "package"),
            ("pyproject.toml", "package"),
            ("Cargo.toml", "package"),
            ("go.mod", "package"),
        ]
        
        for filename, expected_icon in test_cases:
            result = get_file_icon(filename)
            assert result == expected_icon, f"File {filename} should have icon {expected_icon}, got {result}"
    
    def test_get_file_icon_by_extension(self):
        """Test icon detection by file extension."""
        test_cases = [
            ("script.py", "file-code"),
            ("app.js", "file-code"),
            ("component.tsx", "file-code"),
            ("index.html", "globe"),
            ("style.css", "palette"),
            ("data.json", "braces"),
            ("config.yml", "list"),
            ("docs.md", "file-text"),
            ("notes.txt", "file-text"),
            ("main.rs", "file-code"),
            ("server.go", "file-code"),
            ("App.java", "file-code"),
            ("main.c", "file-code"),
            ("script.sh", "terminal"),
            ("database.sql", "database"),
            ("image.png", "image"),
            ("document.pdf", "file-text"),
            ("archive.zip", "archive"),
            (".env", "key"),
        ]
        
        for filename, expected_icon in test_cases:
            result = get_file_icon(filename)
            assert result == expected_icon, f"File {filename} should have icon {expected_icon}, got {result}"
    
    def test_get_file_icon_config_files(self):
        """Test icon detection for configuration files."""
        config_files = [
            "webpack.config.js",
            "babel.config.json", 
            "prettier.config.ts",
        ]
        
        for filename in config_files:
            result = get_file_icon(filename)
            assert result == "settings", f"Config file {filename} should have settings icon"
    
    def test_get_file_icon_unknown_extension(self):
        """Test icon detection for unknown file types."""
        result = get_file_icon("unknown.xyz")
        assert result == "file"
    
    def test_get_file_icon_no_extension(self):
        """Test icon detection for files without extensions."""
        result = get_file_icon("CHANGELOG")
        assert result == "file"


class TestUtilityFunctions:
    """Test utility functions for output formatting."""
    
    def test_slugify(self):
        """Test string slugification."""
        test_cases = [
            ("hello world", "hello-world"),
            ("file.name.ext", "file-name-ext"),
            ("src/main.py", "src-main-py"),
            ("special!@#$%chars", "special-----chars"),
            ("unicode_café_测试", "unicode_caf-_--"),
            ("already-good", "already-good"),
            ("snake_case", "snake_case"),
            ("", ""),
        ]
        
        for input_str, expected in test_cases:
            result = slugify(input_str)
            assert result == expected, f"slugify('{input_str}') should be '{expected}', got '{result}'"
    
    def test_derive_temp_output_path(self):
        """Test derivation of temporary output paths from repo URLs."""
        test_cases = [
            ("https://github.com/user/repo", "repo.html"),
            ("https://github.com/user/repo.git", "repo.html"),
            ("https://gitlab.com/user/project", "project.html"),
            ("https://bitbucket.org/user/repository.git", "repository.html"),
            ("invalid-url", "repo.html"),  # fallback
        ]
        
        for url, expected_filename in test_cases:
            result = derive_temp_output_path(url)
            assert result.name == expected_filename, f"URL '{url}' should derive filename '{expected_filename}'"
            assert result.parent == pathlib.Path(tempfile.gettempdir())


class TestContentHandling:
    """Test content handling in output formats."""
    
    def test_html_escaping_in_cxml(self):
        """Test that HTML content is properly escaped in CXML format."""
        files_with_html = [
            FileInfo(
                pathlib.Path("test.html"),
                "test.html",
                100,
                RenderDecision(True, "ok"),
                '<script>alert("xss")</script>\n<div class="test">content</div>',
                10
            )
        ]
        
        result = generate_cxml_text(files_with_html, "test_repo", "abc123")
        
        # HTML should be escaped
        assert "&lt;script&gt;" in result
        assert "&lt;div class=&quot;test&quot;&gt;" in result
        assert "<script>" not in result  # Should not contain unescaped HTML
    
    def test_html_escaping_in_html_format(self):
        """Test that HTML content is properly escaped in HTML format."""
        files_with_html = [
            FileInfo(
                pathlib.Path("test.html"),
                "test.html",
                100,
                RenderDecision(True, "ok"),
                '<script>alert("xss")</script>',
                10
            )
        ]
        
        result = build_html("test_repo", pathlib.Path("/test"), "abc123", files_with_html)
        
        # Content inside <pre> tags should be escaped
        assert "&lt;script&gt;" in result
        # But HTML structure should remain intact
        assert "<html" in result
        assert "</html>" in result
    
    def test_no_html_escaping_in_repomix(self):
        """Test that HTML content is NOT escaped in Repomix format."""
        files_with_html = [
            FileInfo(
                pathlib.Path("test.html"),
                "test.html",
                100,
                RenderDecision(True, "ok"),
                '<script>alert("test")</script>',
                10
            )
        ]
        
        result = generate_repomix_text(files_with_html, "test_repo", "abc123")
        
        # HTML should NOT be escaped in repomix format
        assert '<script>alert("test")</script>' in result
        assert "&lt;script&gt;" not in result
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode content in all formats."""
        unicode_content = "Unicode test: 测试 café ñoño 🚀"
        
        files_with_unicode = [
            FileInfo(
                pathlib.Path("unicode.txt"),
                "unicode.txt",
                100,
                RenderDecision(True, "ok"),
                unicode_content,
                10
            )
        ]
        
        # Test CXML format
        cxml_result = generate_cxml_text(files_with_unicode, "test_repo", "abc123")
        assert "测试" in cxml_result
        assert "café" in cxml_result
        assert "🚀" in cxml_result
        
        # Test Repomix format
        repomix_result = generate_repomix_text(files_with_unicode, "test_repo", "abc123")
        assert unicode_content in repomix_result
        
        # Test HTML format
        html_result = build_html("test_repo", pathlib.Path("/test"), "abc123", files_with_unicode)
        assert "测试" in html_result
        assert "café" in html_result
    
    def test_large_content_handling(self):
        """Test handling of large file content."""
        large_content = "x" * 100000  # 100KB of content
        
        large_file = [
            FileInfo(
                pathlib.Path("large.txt"),
                "large.txt",
                100000,
                RenderDecision(True, "ok"),
                large_content,
                25000
            )
        ]
        
        # All formats should handle large content without crashing
        cxml_result = generate_cxml_text(large_file, "test_repo", "abc123")
        assert len(cxml_result) > 100000
        
        repomix_result = generate_repomix_text(large_file, "test_repo", "abc123")
        assert large_content in repomix_result
        
        html_result = build_html("test_repo", pathlib.Path("/test"), "abc123", large_file)
        assert large_content in html_result


class TestErrorHandling:
    """Test error handling in output format generation."""
    
    def test_missing_content_handling(self):
        """Test handling of files with missing content."""
        files_no_content = [
            FileInfo(
                pathlib.Path("missing.txt"),
                "missing.txt",
                100,
                RenderDecision(True, "ok"),
                None,  # No content
                None
            )
        ]
        
        # Should attempt to read content from file path
        with patch('scribe.output_formats.read_text', return_value="fallback content"):
            cxml_result = generate_cxml_text(files_no_content, "test_repo", "abc123")
            assert "fallback content" in cxml_result
    
    def test_read_error_handling(self):
        """Test handling of file read errors."""
        files_no_content = [
            FileInfo(
                pathlib.Path("unreadable.txt"),
                "unreadable.txt",
                100,
                RenderDecision(True, "ok"),
                None,
                None
            )
        ]
        
        # Simulate read error
        with patch('scribe.output_formats.read_text', side_effect=Exception("Permission denied")):
            cxml_result = generate_cxml_text(files_no_content, "test_repo", "abc123")
            assert "Failed to read: Permission denied" in cxml_result
            
            repomix_result = generate_repomix_text(files_no_content, "test_repo", "abc123")
            assert "Failed to read: Permission denied" in repomix_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
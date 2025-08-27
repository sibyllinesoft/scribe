"""
Test suite for repomix compatibility features.

Ensures all new repomix-compatible features work correctly and maintain 
high test coverage (>85%).
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from packrepo.fastpath.pattern_filter import PatternFilter, FilterConfig, create_repomix_compatible_filter
from packrepo.fastpath.config_manager import ConfigManager, ScribeConfig
from packrepo.fastpath.git_integration import GitIntegration, RemoteRepoHandler, GitFileInfo
from packrepo.fastpath.output_formats import (
    JSONFormatter, MarkdownFormatter, PlainFormatter, OutputConfig,
    create_formatter, create_pack_summary, create_directory_structure
)


class TestPatternFilter:
    """Test pattern filtering with .scribeignore/.repomixignore support."""
    
    def setup_method(self):
        """Setup test directory structure."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_scribeignore_priority(self):
        """Test that .scribeignore takes priority over .repomixignore."""
        # Create both ignore files
        scribeignore = self.temp_dir / ".scribeignore"
        scribeignore.write_text("*.scribe\\ntests/")
        
        repomixignore = self.temp_dir / ".repomixignore"
        repomixignore.write_text("*.repomix\\nold_tests/")
        
        # Create test files
        (self.temp_dir / "test.scribe").touch()
        (self.temp_dir / "test.repomix").touch()
        (self.temp_dir / "main.py").touch()
        (self.temp_dir / "tests").mkdir()
        (self.temp_dir / "tests" / "test_main.py").touch()
        (self.temp_dir / "old_tests").mkdir()
        (self.temp_dir / "old_tests" / "test_old.py").touch()
        
        config = FilterConfig(
            include_patterns=["**/*"],
            exclude_patterns=[],
            use_gitignore=False,
            use_default_patterns=False
        )
        
        filter_obj = PatternFilter(config, self.temp_dir)
        
        # Should exclude .scribe files and tests/ (from .scribeignore)
        # Should NOT exclude .repomix files or old_tests/ (from .repomixignore)
        assert not filter_obj.should_include(self.temp_dir / "test.scribe")
        assert filter_obj.should_include(self.temp_dir / "test.repomix")  # Only .scribeignore loaded
        assert filter_obj.should_include(self.temp_dir / "main.py")
        assert not filter_obj.should_include(self.temp_dir / "tests" / "test_main.py")
        assert filter_obj.should_include(self.temp_dir / "old_tests" / "test_old.py")
        
    def test_repomixignore_fallback(self):
        """Test .repomixignore used when .scribeignore doesn't exist."""
        # Only create .repomixignore
        repomixignore = self.temp_dir / ".repomixignore"
        repomixignore.write_text("*.tmp\\nbuild/")
        
        # Create test files
        (self.temp_dir / "test.tmp").touch()
        (self.temp_dir / "main.py").touch()
        (self.temp_dir / "build").mkdir()
        (self.temp_dir / "build" / "output.o").touch()
        
        config = FilterConfig(
            include_patterns=["**/*"],
            exclude_patterns=[],
            use_gitignore=False,
            use_default_patterns=False
        )
        
        filter_obj = PatternFilter(config, self.temp_dir)
        
        # Should exclude based on .repomixignore
        assert not filter_obj.should_include(self.temp_dir / "test.tmp")
        assert filter_obj.should_include(self.temp_dir / "main.py")
        assert not filter_obj.should_include(self.temp_dir / "build" / "output.o")
        
    def test_gitignore_integration(self):
        """Test .gitignore still works with new ignore files."""
        # Create all ignore files
        gitignore = self.temp_dir / ".gitignore"
        gitignore.write_text("*.git\\n__pycache__/")
        
        scribeignore = self.temp_dir / ".scribeignore"
        scribeignore.write_text("*.scribe")
        
        # Create test files
        (self.temp_dir / "test.git").touch()
        (self.temp_dir / "test.scribe").touch()
        (self.temp_dir / "main.py").touch()
        (self.temp_dir / "__pycache__").mkdir()
        (self.temp_dir / "__pycache__" / "main.pyc").touch()
        
        config = FilterConfig(
            include_patterns=["**/*"],
            exclude_patterns=[],
            use_gitignore=True,
            use_default_patterns=False
        )
        
        filter_obj = PatternFilter(config, self.temp_dir)
        
        # Should exclude from both .gitignore and .scribeignore
        assert not filter_obj.should_include(self.temp_dir / "test.git")
        assert not filter_obj.should_include(self.temp_dir / "test.scribe")
        assert filter_obj.should_include(self.temp_dir / "main.py")
        assert not filter_obj.should_include(self.temp_dir / "__pycache__" / "main.pyc")
        
    def test_pattern_priority_order(self):
        """Test pattern priority: custom > .scribeignore > .gitignore > defaults."""
        # Create ignore files
        gitignore = self.temp_dir / ".gitignore" 
        gitignore.write_text("*.git")
        
        scribeignore = self.temp_dir / ".scribeignore"
        scribeignore.write_text("*.scribe")
        
        # Create test files
        (self.temp_dir / "test.git").touch()
        (self.temp_dir / "test.scribe").touch()
        (self.temp_dir / "test.custom").touch()
        (self.temp_dir / "test.pyc").touch()  # Default pattern
        
        config = FilterConfig(
            include_patterns=["**/*"],
            exclude_patterns=["*.custom"],  # Custom pattern
            use_gitignore=True,
            use_default_patterns=True
        )
        
        filter_obj = PatternFilter(config, self.temp_dir)
        
        # All should be excluded based on different priority levels
        assert not filter_obj.should_include(self.temp_dir / "test.custom")  # Custom
        assert not filter_obj.should_include(self.temp_dir / "test.scribe")  # .scribeignore
        assert not filter_obj.should_include(self.temp_dir / "test.git")     # .gitignore
        assert not filter_obj.should_include(self.temp_dir / "test.pyc")     # Default
        
    def test_create_repomix_compatible_filter(self):
        """Test the convenience function works correctly."""
        # Create test structure
        (self.temp_dir / "src").mkdir()
        (self.temp_dir / "src" / "main.py").touch()
        (self.temp_dir / "tests").mkdir()
        (self.temp_dir / "tests" / "test_main.py").touch()
        
        filter_obj = create_repomix_compatible_filter(
            repo_path=self.temp_dir,
            include=["**/*.py"],
            ignore_custom_patterns=["**/tests/**"],
            use_gitignore=False,
            use_default_patterns=False
        )
        
        assert filter_obj.should_include(self.temp_dir / "src" / "main.py")
        assert not filter_obj.should_include(self.temp_dir / "tests" / "test_main.py")


class TestConfigManager:
    """Test configuration management with repomix compatibility."""
    
    def setup_method(self):
        """Setup test directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_scribe_config_priority(self):
        """Test that scribe.config.json takes priority over repomix.config.json."""
        # Create both config files
        scribe_config = {
            "output_style": "json",
            "include": ["**/*.py"],
            "git_sort_by_changes": True
        }
        
        repomix_config = {
            "output": {
                "style": "markdown",
                "filePath": "repomix-output.md"
            },
            "include": ["**/*.js"],
            "ignore": {
                "useGitignore": False
            }
        }
        
        scribe_config_path = self.temp_dir / "scribe.config.json"
        with open(scribe_config_path, 'w') as f:
            json.dump(scribe_config, f)
            
        repomix_config_path = self.temp_dir / "repomix.config.json"
        with open(repomix_config_path, 'w') as f:
            json.dump(repomix_config, f)
            
        manager = ConfigManager(self.temp_dir)
        config = manager.load_config()
        
        # Should use scribe config values
        assert config.output_style == "json"
        assert config.include == ["**/*.py"]
        assert config.git_sort_by_changes == True
        
    def test_repomix_config_fallback(self):
        """Test repomix config used when scribe config doesn't exist."""
        repomix_config = {
            "output": {
                "style": "xml",
                "showLineNumbers": True,
                "fileSummary": False,
                "git": {
                    "sortByChanges": True,
                    "includeDiffs": True
                }
            },
            "include": ["**/*.ts"],
            "ignore": {
                "useGitignore": False,
                "customPatterns": ["**/node_modules/**"]
            },
            "tokenCount": {
                "encoding": "cl100k_base"
            }
        }
        
        repomix_config_path = self.temp_dir / "repomix.config.json"
        with open(repomix_config_path, 'w') as f:
            json.dump(repomix_config, f)
            
        manager = ConfigManager(self.temp_dir)
        config = manager.load_config()
        
        # Should convert repomix config correctly
        assert config.output_style == "xml"
        assert config.output_show_line_numbers == True
        assert config.output_file_summary == False
        assert config.git_sort_by_changes == True
        assert config.git_include_diffs == True
        assert config.include == ["**/*.ts"]
        assert config.ignore_use_gitignore == False
        assert config.ignore_custom_patterns == ["**/node_modules/**"]
        assert config.token_count_encoding == "cl100k_base"
        
    def test_repomix_config_detection(self):
        """Test automatic detection of repomix-style config."""
        manager = ConfigManager(self.temp_dir)
        
        # Repomix-style config
        repomix_style = {
            "output": {"style": "json"},
            "ignore": {"customPatterns": ["*.tmp"]}
        }
        assert manager._is_repomix_style_config(repomix_style) == True
        
        # Scribe-style config
        scribe_style = {
            "output_style": "json", 
            "ignore_custom_patterns": ["*.tmp"]
        }
        assert manager._is_repomix_style_config(scribe_style) == False
        
    def test_custom_config_path(self):
        """Test loading custom configuration file."""
        custom_config = {
            "output_style": "markdown",
            "include": ["**/*.py", "**/*.md"]
        }
        
        custom_path = self.temp_dir / "custom.config.json"
        with open(custom_path, 'w') as f:
            json.dump(custom_config, f)
            
        manager = ConfigManager(self.temp_dir)
        config = manager.load_config(custom_path)
        
        assert config.output_style == "markdown"
        assert config.include == ["**/*.py", "**/*.md"]
        
    def test_default_config(self):
        """Test default configuration when no config files exist."""
        manager = ConfigManager(self.temp_dir)
        config = manager.load_config()
        
        # Should use default values
        assert config.output_style == "plain"
        assert config.include == ["**/*"]
        assert config.ignore_use_gitignore == True
        assert config.ignore_use_default_patterns == True


class TestGitIntegration:
    """Test git integration functionality."""
    
    def setup_method(self):
        """Setup test directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('subprocess.run')
    def test_git_availability_check(self, mock_run):
        """Test git availability detection."""
        # Mock successful git command
        mock_run.return_value = Mock(returncode=0)
        
        git = GitIntegration(self.temp_dir)
        assert git.is_git_repo() == True
        
        # Mock failed git command
        mock_run.return_value = Mock(returncode=1)
        git = GitIntegration(self.temp_dir)
        assert git.is_git_repo() == False
        
    @patch('subprocess.run')
    def test_file_change_frequencies(self, mock_run):
        """Test git log analysis for change frequencies."""
        # Mock git log output
        git_log_output = """abc123|John Doe|1234567890|Initial commit
10\\t5\\tmain.py
20\\t0\\tREADME.md

def456|Jane Smith|1234567900|Add tests
15\\t2\\tmain.py
5\\t0\\ttest_main.py
"""
        
        mock_run.return_value = Mock(returncode=0, stdout=git_log_output)
        
        git = GitIntegration(self.temp_dir)
        frequencies = git.get_file_change_frequencies(max_commits=10)
        
        assert "main.py" in frequencies
        assert "README.md" in frequencies
        assert "test_main.py" in frequencies
        
        main_py_info = frequencies["main.py"]
        assert main_py_info.change_count == 2  # Changed in both commits
        assert main_py_info.author_count == 2  # Two different authors
        assert main_py_info.lines_added == 25  # 10 + 15
        assert main_py_info.lines_deleted == 7  # 5 + 2
        
    @patch('subprocess.run')
    def test_sort_files_by_changes(self, mock_run):
        """Test file sorting by git change frequency."""
        # Mock git log output
        git_log_output = """abc123|John|1234567890|Commit 1
5\\t0\\tfrequent.py
1\\t0\\trare.py

def456|Jane|1234567900|Commit 2  
10\\t0\\tfrequent.py
"""
        
        mock_run.return_value = Mock(returncode=0, stdout=git_log_output)
        
        git = GitIntegration(self.temp_dir)
        files = ["frequent.py", "rare.py", "new.py"]
        sorted_files = git.sort_files_by_changes(files)
        
        # frequent.py should be first (2 changes), then rare.py (1 change), then new.py (0 changes)
        assert sorted_files[0][0] == "frequent.py"
        assert sorted_files[1][0] == "rare.py" 
        assert sorted_files[2][0] == "new.py"
        assert sorted_files[0][1] > sorted_files[1][1] > sorted_files[2][1]


class TestRemoteRepoHandler:
    """Test remote repository handling."""
    
    def test_remote_url_detection(self):
        """Test detection of remote repository URLs."""
        remote_urls = [
            "https://github.com/user/repo.git",
            "git@github.com:user/repo.git", 
            "https://gitlab.com/user/repo",
            "https://bitbucket.org/user/repo.git"
        ]
        
        local_paths = [
            "/local/path/to/repo",
            "./relative/path",
            "~/home/repo"
        ]
        
        for url in remote_urls:
            assert RemoteRepoHandler.is_remote_url(url) == True
            
        for path in local_paths:
            assert RemoteRepoHandler.is_remote_url(path) == False
            
    @patch('subprocess.run')
    @patch('tempfile.mkdtemp')
    def test_repository_cloning(self, mock_mkdtemp, mock_run):
        """Test remote repository cloning."""
        temp_path = "/tmp/test_clone"
        mock_mkdtemp.return_value = temp_path
        mock_run.return_value = Mock(returncode=0)
        
        with patch('pathlib.Path.exists', return_value=True):
            result = RemoteRepoHandler.clone_repository(
                "https://github.com/user/repo.git",
                branch="main", 
                depth=1
            )
            
        assert result == Path(temp_path) / "repo"
        
        # Verify git clone command
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "git" in args
        assert "clone" in args
        assert "--depth" in args
        assert "1" in args
        assert "--branch" in args
        assert "main" in args
        assert "https://github.com/user/repo.git" in args


class TestOutputFormats:
    """Test output formatting functionality."""
    
    def test_json_formatter(self):
        """Test JSON output formatting."""
        config = OutputConfig(style="json", include_file_summary=True)
        formatter = JSONFormatter(config)
        
        # Mock scan results
        mock_scan_result = Mock()
        mock_scan_result.stats.path = "main.py"
        mock_scan_result.stats.language = "python"
        mock_scan_result.stats.size_bytes = 1000
        mock_scan_result.stats.lines = 50
        mock_scan_result.stats.is_readme = False
        mock_scan_result.stats.is_test = False
        mock_scan_result.stats.is_config = False
        mock_scan_result.stats.is_docs = False
        mock_scan_result.priority_boost = 0.5
        mock_scan_result.churn_score = 0.8
        mock_scan_result.imports = None
        mock_scan_result.doc_analysis = None
        
        from packrepo.fastpath.output_formats import PackSummary
        summary = PackSummary(
            total_files=1,
            total_size_bytes=1000,
            total_lines=50,
            total_tokens=200,
            languages={"python": 1},
            file_types={".py": 1},
            generation_time=1234567890
        )
        
        output = formatter.format_output([mock_scan_result], summary)
        
        # Parse and verify JSON structure
        data = json.loads(output)
        assert "metadata" in data
        assert "summary" in data
        assert "files" in data
        assert data["metadata"]["generator"] == "Scribe FastPath"
        assert len(data["files"]) == 1
        assert data["files"][0]["path"] == "main.py"
        assert data["files"][0]["language"] == "python"
        
    def test_markdown_formatter(self):
        """Test Markdown output formatting.""" 
        config = OutputConfig(style="markdown", show_line_numbers=True)
        formatter = MarkdownFormatter(config)
        
        # Mock scan results
        mock_scan_result = Mock()
        mock_scan_result.stats.path = "README.md"
        mock_scan_result.stats.language = "markdown"
        mock_scan_result.stats.size_bytes = 2000
        mock_scan_result.stats.lines = 100
        
        from packrepo.fastpath.output_formats import PackSummary
        summary = PackSummary(
            total_files=1,
            total_size_bytes=2000,
            total_lines=100,
            total_tokens=400,
            languages={"markdown": 1},
            file_types={".md": 1},
            generation_time=1234567890
        )
        
        with patch.object(formatter, '_get_file_content', return_value="# Test\\nContent"):
            output = formatter.format_output([mock_scan_result], summary)
            
        # Verify Markdown structure
        assert "# Repository Pack" in output
        assert "## Summary" in output
        assert "## Files" in output
        assert "### README.md" in output
        assert "**Language**: markdown" in output
        assert "```markdown" in output
        
    def test_create_formatter(self):
        """Test formatter factory function."""
        json_config = OutputConfig(style="json")
        json_formatter = create_formatter(json_config)
        assert isinstance(json_formatter, JSONFormatter)
        
        markdown_config = OutputConfig(style="markdown")
        markdown_formatter = create_formatter(markdown_config)
        assert isinstance(markdown_formatter, MarkdownFormatter)
        
        plain_config = OutputConfig(style="plain")
        plain_formatter = create_formatter(plain_config)
        assert isinstance(plain_formatter, PlainFormatter)


class TestIntegrationScenarios:
    """Integration tests for complete repomix compatibility scenarios."""
    
    def setup_method(self):
        """Setup test repository structure."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create realistic repo structure
        (self.temp_dir / "src").mkdir()
        (self.temp_dir / "src" / "main.py").write_text("print('Hello World')")
        (self.temp_dir / "src" / "utils.py").write_text("def helper(): pass")
        
        (self.temp_dir / "tests").mkdir()
        (self.temp_dir / "tests" / "test_main.py").write_text("def test_main(): pass")
        
        (self.temp_dir / "README.md").write_text("# My Project\\n\\nDescription")
        (self.temp_dir / "requirements.txt").write_text("requests==2.25.1")
        
        # Add ignore files
        (self.temp_dir / ".scribeignore").write_text("*.tmp\\n__pycache__/")
        (self.temp_dir / ".gitignore").write_text("*.pyc\\n.env")
        
    def teardown_method(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_complete_repomix_workflow(self):
        """Test complete workflow with repomix-style configuration."""
        # Create repomix config
        repomix_config = {
            "output": {
                "style": "json",
                "fileSummary": True,
                "showLineNumbers": False
            },
            "include": ["**/*.py", "**/*.md"],
            "ignore": {
                "useGitignore": True,
                "customPatterns": ["**/tests/**"]
            }
        }
        
        config_path = self.temp_dir / "repomix.config.json"
        with open(config_path, 'w') as f:
            json.dump(repomix_config, f)
            
        # Load config
        manager = ConfigManager(self.temp_dir)
        config = manager.load_config()
        
        # Create pattern filter
        pattern_filter = create_repomix_compatible_filter(
            repo_path=self.temp_dir,
            include=config.include,
            ignore_custom_patterns=config.ignore_custom_patterns,
            use_gitignore=config.ignore_use_gitignore,
            use_default_patterns=config.ignore_use_default_patterns
        )
        
        # Test filtering
        assert pattern_filter.should_include(self.temp_dir / "src" / "main.py")
        assert pattern_filter.should_include(self.temp_dir / "README.md")
        assert not pattern_filter.should_include(self.temp_dir / "tests" / "test_main.py")
        assert pattern_filter.should_include(self.temp_dir / "requirements.txt")  # Not Python/MD but not excluded
        
        # Test output formatting
        output_config = OutputConfig(
            style=config.output_style,
            include_file_summary=config.output_file_summary,
            show_line_numbers=config.output_show_line_numbers
        )
        
        formatter = create_formatter(output_config)
        assert isinstance(formatter, JSONFormatter)
        
    def test_scribe_enhancements_over_repomix(self):
        """Test that Scribe enhancements work alongside repomix compatibility."""
        # This test would demonstrate features that Scribe has beyond repomix
        config_manager = ConfigManager(self.temp_dir)
        config = config_manager.load_config()
        
        # Scribe-specific features that repomix doesn't have
        assert hasattr(config, 'git_sort_by_changes')  # Advanced git integration
        assert hasattr(config, 'token_count_encoding')  # Token counting
        
        # Test that we can use both repomix patterns AND scribe enhancements
        filter_obj = create_repomix_compatible_filter(
            repo_path=self.temp_dir,
            include=["**/*"],
            ignore_custom_patterns=[],
            use_gitignore=True,
            use_default_patterns=True,
            max_file_size=50_000_000  # Scribe enhancement: file size limits
        )
        
        stats = filter_obj.get_stats()
        assert "max_file_size_mb" in stats  # Scribe enhancement
        assert stats["gitignore_enabled"] == True  # Repomix compatibility


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "--cov=packrepo.fastpath.pattern_filter",
        "--cov=packrepo.fastpath.config_manager", 
        "--cov=packrepo.fastpath.git_integration",
        "--cov=packrepo.fastpath.output_formats",
        "--cov-report=term-missing",
        "--cov-fail-under=85",
        __file__
    ])
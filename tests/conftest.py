#!/usr/bin/env python3
"""
Pytest configuration file for Scribe tests.

This file contains shared fixtures and configuration for all test modules.
"""

import pytest
import pathlib
import tempfile
import sys
import subprocess
from unittest.mock import MagicMock, patch

# Add the project root to the path so we can import scribe
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scribe import FileInfo, RenderDecision


@pytest.fixture
def temp_repo():
    """Create a temporary repository structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = pathlib.Path(tmpdir)
        
        # Create typical repository structure
        (repo_path / "README.md").write_text("# Test Repository\n\nThis is a test.")
        (repo_path / "LICENSE").write_text("MIT License")
        (repo_path / ".gitignore").write_text("__pycache__/\n*.pyc\nnode_modules/\n")
        
        # Source directory
        src_dir = repo_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('Hello, World!')")
        (src_dir / "utils.py").write_text("def helper():\n    return True")
        (src_dir / "config.json").write_text('{"debug": true}')
        
        # Tests directory
        tests_dir = repo_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test_main():\n    assert True")
        (tests_dir / "__init__.py").write_text("")
        
        # Web assets
        static_dir = repo_path / "static"
        static_dir.mkdir()
        (static_dir / "style.css").write_text("body { margin: 0; }")
        (static_dir / "app.js").write_text("console.log('app loaded');")
        
        # Create some binary files
        (static_dir / "logo.png").write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01')
        
        # Documentation
        docs_dir = repo_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "api.md").write_text("# API Documentation")
        (docs_dir / "tutorial.md").write_text("# Tutorial")
        
        # Package files
        (repo_path / "package.json").write_text('{"name": "test-project", "version": "1.0.0"}')
        (repo_path / "requirements.txt").write_text("requests==2.28.0\npytest==7.1.0")
        
        yield repo_path


@pytest.fixture
def sample_file_infos():
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
            pathlib.Path("src/utils.py"),
            "src/utils.py",
            80,
            RenderDecision(True, "ok"),
            "def helper():\n    return True\n",
            6
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
            pathlib.Path("static/logo.png"),
            "static/logo.png",
            1024,
            RenderDecision(False, "binary"),
            None,
            None
        ),
        FileInfo(
            pathlib.Path("large_file.txt"),
            "large_file.txt",
            1024 * 1024 * 2,  # 2MB
            RenderDecision(False, "too_large"),
            None,
            None
        )
    ]


@pytest.fixture
def mock_git_operations():
    """Mock git operations for testing."""
    with patch('scribe.run') as mock_run, \
         patch('scribe.git_head_commit') as mock_head_commit, \
         patch('scribe.git_clone') as mock_clone:
        
        # Configure git ls-files to return typical output
        mock_run.return_value.stdout = "src/main.py\nsrc/utils.py\nREADME.md\npackage.json"
        mock_run.return_value.returncode = 0
        
        # Configure git head commit
        mock_head_commit.return_value = "abc123def456"
        
        # Configure git clone to succeed
        mock_clone.return_value = None
        
        yield {
            'run': mock_run,
            'head_commit': mock_head_commit,
            'clone': mock_clone
        }


@pytest.fixture
def mock_file_operations():
    """Mock file system operations for testing."""
    with patch('pathlib.Path.write_text') as mock_write, \
         patch('pathlib.Path.stat') as mock_stat, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.is_dir') as mock_is_dir:
        
        # Configure typical successful operations
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_stat.return_value = MagicMock(st_size=1024)
        
        yield {
            'write_text': mock_write,
            'stat': mock_stat,
            'mkdir': mock_mkdir,
            'exists': mock_exists,
            'is_dir': mock_is_dir
        }


@pytest.fixture
def capture_stderr(capsys):
    """Capture stderr output during tests."""
    def _capture():
        captured = capsys.readouterr()
        return captured.err
    return _capture


@pytest.fixture
def temp_output_file():
    """Create a temporary output file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        output_path = pathlib.Path(f.name)
    
    yield output_path
    
    # Cleanup
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def reset_sys_modules():
    """Reset sys.modules to avoid import conflicts between tests."""
    original_modules = sys.modules.copy()
    yield
    # Restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)


# Test markers for different categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may skip in quick runs)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "platform_specific: marks tests that are platform specific"
    )


# Skip network tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    skip_network = pytest.mark.skip(reason="Network tests skipped by default")
    skip_slow = pytest.mark.skip(reason="Slow tests skipped by default")
    
    for item in items:
        if "network" in item.keywords and not config.getoption("--run-network"):
            item.add_marker(skip_network)
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run network-dependent tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="Run slow tests"
    )


# Utility functions for tests
def create_test_file_with_content(parent_dir, filename, content, size_override=None):
    """Create a test file with specific content."""
    file_path = parent_dir / filename
    file_path.write_text(content)
    
    actual_size = len(content.encode('utf-8'))
    test_size = size_override if size_override is not None else actual_size
    
    return FileInfo(
        path=file_path,
        rel=filename,
        size=test_size,
        decision=RenderDecision(True, "ok"),
        content=content,
        token_estimate=max(1, len(content) // 4)
    )


def create_binary_test_file(parent_dir, filename, binary_data):
    """Create a binary test file."""
    file_path = parent_dir / filename
    file_path.write_bytes(binary_data)
    
    return FileInfo(
        path=file_path,
        rel=filename,
        size=len(binary_data),
        decision=RenderDecision(False, "binary"),
        content=None,
        token_estimate=None
    )


# Mock data generators
def generate_mock_codebase(base_path, num_files=10):
    """Generate a mock codebase with specified number of files."""
    files = []
    
    for i in range(num_files):
        filename = f"file_{i:03d}.py"
        content = f"# File {i}\ndef function_{i}():\n    return {i}"
        
        file_path = base_path / filename
        files.append(FileInfo(
            path=file_path,
            rel=filename,
            size=len(content),
            decision=RenderDecision(True, "ok"),
            content=content,
            token_estimate=len(content) // 4
        ))
    
    return files
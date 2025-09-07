#!/usr/bin/env python3
"""Git utilities for repository operations."""

import fnmatch
import os
import pathlib
import re
import subprocess
from typing import List, Set


def run(cmd: List[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def parse_gitignore_patterns(repo_root: pathlib.Path) -> Set[str]:
    """Parse .gitignore files and return a set of normalized patterns."""
    patterns = set()
    
    # Add some essential patterns that should always be ignored even without .gitignore
    essential_patterns = {
        ".git/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "Thumbs.db",
    }
    patterns.update(essential_patterns)
    
    # Parse the main .gitignore file in the repo root
    main_gitignore = repo_root / ".gitignore"
    if main_gitignore.exists():
        try:
            with main_gitignore.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    # Remove negation patterns for simplicity (!)
                    if line.startswith("!"):
                        continue
                    patterns.add(line)
        except Exception:
            # If we can't read the .gitignore file, just skip it
            pass
    
    return patterns


def should_ignore_path(path: str, patterns: Set[str]) -> bool:
    """Check if a path should be ignored based on gitignore patterns."""
    for pattern in patterns:
        if match_gitignore_pattern(path, pattern):
            return True
    return False


def match_gitignore_pattern(path: str, pattern: str) -> bool:
    """Match a path against a gitignore pattern."""
    if pattern.endswith('/'):
        # Directory pattern - matches if any part of the path contains this directory
        dir_name = pattern[:-1]
        path_parts = path.split('/')
        for i, part in enumerate(path_parts):
            if part == dir_name:
                # Found the directory, check if there are more parts after it
                if i < len(path_parts) - 1:  # Not the last part
                    return True
        # Also check if the path ends with this directory name
        return path == dir_name or path.endswith('/' + dir_name)
    else:
        # File pattern - use fnmatch for full path and filename matching
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)


def git_clone(url: str, dst: str) -> None:
    """Clone a git repository."""
    subprocess.run(["git", "clone", url, dst], check=True)


def git_head_commit(repo_dir: str) -> str:
    """Get the HEAD commit hash of a repository."""
    try:
        result = run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
        return result.stdout.strip()[:7]
    except Exception:
        return "(unknown)"
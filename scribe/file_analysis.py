#!/usr/bin/env python3
"""File analysis and decision utilities."""

import os
import pathlib
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Set

# Import from other modules
from .git_utils import should_ignore_path, run, parse_gitignore_patterns
from .glob_patterns import should_include_path

# Binary file extensions that should be skipped
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".ogg", ".flac",
    ".ttf", ".otf", ".eot", ".woff", ".woff2",
    ".so", ".dll", ".dylib", ".class", ".jar", ".exe", ".bin",
}

MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd", ".mkdn"}


@dataclass
class RenderDecision:
    include: bool
    reason: str  # "ok" | "binary" | "too_large" | "ignored"


@dataclass
class FileInfo:
    path: pathlib.Path  # absolute path on disk
    rel: str            # path relative to repo root (slash-separated)
    size: int
    decision: RenderDecision
    content: Optional[str] = None  # File content when loaded
    token_estimate: Optional[int] = None  # Token count estimate


def bytes_human(n: int) -> str:
    """Human-readable bytes: 1 decimal for KiB and above, integer for B."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    i = 0
    while f >= 1024.0 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    if i == 0:
        return f"{int(f)} {units[i]}"
    else:
        return f"{f:.1f} {units[i]}"


def looks_binary(path: pathlib.Path) -> bool:
    ext = path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return True
        # Heuristic: try UTF-8 decode; if it hard-fails, likely binary
        # Handle partial UTF-8 characters at chunk boundaries
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError as e:
            # If the error is at the end of the chunk, it might be a partial UTF-8 character
            # Read a few more bytes to see if we can complete the sequence
            if e.start >= len(chunk) - 4:  # UTF-8 characters are at most 4 bytes
                try:
                    with path.open("rb") as f:
                        f.seek(0)
                        extended_chunk = f.read(8196)  # Read 4 more bytes
                    extended_chunk.decode("utf-8")
                    return False  # Successfully decoded with more bytes - it's text
                except UnicodeDecodeError:
                    return True  # Still can't decode - likely binary
                except Exception:
                    return True  # Any other error - treat as binary
            else:
                return True  # Decode error not at end - likely binary
        return False
    except Exception:
        # If unreadable, treat as binary to be safe
        return True


def decide_file(path: pathlib.Path, repo_root: pathlib.Path, max_bytes: int, ignore_patterns: Set[str], 
                include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> FileInfo:
    rel = str(path.relative_to(repo_root)).replace(os.sep, "/")
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    
    # Check include/exclude patterns first (if provided)
    if include_patterns is not None or exclude_patterns is not None:
        include_patterns = include_patterns or []
        exclude_patterns = exclude_patterns or []
        if not should_include_path(rel, include_patterns, exclude_patterns):
            return FileInfo(path, rel, size, RenderDecision(False, "excluded"))
    
    # Check if the file should be ignored based on gitignore patterns
    if should_ignore_path(rel, ignore_patterns):
        return FileInfo(path, rel, size, RenderDecision(False, "ignored"))
    
    if size > max_bytes:
        return FileInfo(path, rel, size, RenderDecision(False, "too_large"))
    if looks_binary(path):
        return FileInfo(path, rel, size, RenderDecision(False, "binary"))
    return FileInfo(path, rel, size, RenderDecision(True, "ok"))


def decide_file_simple(path: pathlib.Path, repo_root: pathlib.Path, max_bytes: int,
                       include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> FileInfo:
    """Simplified file decision for git-tracked files (no ignore checking needed)."""
    rel = str(path.relative_to(repo_root)).replace(os.sep, "/")
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    
    # Check include/exclude patterns first (if provided)
    if include_patterns is not None or exclude_patterns is not None:
        include_patterns = include_patterns or []
        exclude_patterns = exclude_patterns or []
        if not should_include_path(rel, include_patterns, exclude_patterns):
            return FileInfo(path, rel, size, RenderDecision(False, "excluded"))
    
    if size > max_bytes:
        return FileInfo(path, rel, size, RenderDecision(False, "too_large"))
    if looks_binary(path):
        return FileInfo(path, rel, size, RenderDecision(False, "binary"))
    return FileInfo(path, rel, size, RenderDecision(True, "ok"))

def estimate_tokens_simple(text: str) -> int:
    """Simple token estimation (roughly 4 chars per token for English)."""
    return max(1, len(text) // 4)


def load_file_content(file_info: FileInfo) -> FileInfo:
    """Load content for a file and estimate tokens."""
    from .output_formats import read_text
    
    # Check if file is binary before attempting to load content
    if looks_binary(file_info.path):
        return FileInfo(
            path=file_info.path,
            rel=file_info.rel,
            size=file_info.size,
            decision=RenderDecision(False, "binary"),
            content=None,
            token_estimate=None
        )
    
    try:
        content = read_text(file_info.path)
        token_estimate = estimate_tokens_simple(content)
        return FileInfo(
            path=file_info.path,
            rel=file_info.rel,
            size=file_info.size,
            decision=file_info.decision,
            content=content,
            token_estimate=token_estimate
        )
    except Exception:
        return FileInfo(
            path=file_info.path,
            rel=file_info.rel,
            size=file_info.size,
            decision=RenderDecision(False, "read_error"),
            content=None,
            token_estimate=None
        )


def collect_files(repo_root: pathlib.Path, max_bytes: int, 
                  include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> List[FileInfo]:
    """Collect files from the repository, preferring git ls-files if available."""
    infos: List[FileInfo] = []
    
    # Try to use git ls-files first (respects .gitignore automatically)
    try:
        result = run(["git", "ls-files"], cwd=str(repo_root), check=True)
        git_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        for path in git_files:
            if not path:  # Skip empty lines
                continue
            abs_path = repo_root / path
            if abs_path.exists() and abs_path.is_file() and not abs_path.is_symlink():
                infos.append(decide_file_simple(abs_path, repo_root, max_bytes, include_patterns, exclude_patterns))
        
        print(f"✓ Using git ls-files: found {len(git_files)} tracked files", file=sys.stderr)
        return infos
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to filesystem walk if git is not available or not a git repo
        print("⚠️  Git not available, falling back to filesystem walk", file=sys.stderr)
        
        # Parse gitignore patterns for manual filtering
        ignore_patterns = parse_gitignore_patterns(repo_root)
        
        for p in sorted(repo_root.rglob("*")):
            if p.is_symlink():
                continue
            if p.is_file():
                infos.append(decide_file(p, repo_root, max_bytes, ignore_patterns, include_patterns, exclude_patterns))
        return infos

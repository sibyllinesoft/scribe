"""Scribe - A tool for creating structured documentation from codebases.

This package provides functionality for:
- Glob pattern matching and file filtering
- Git repository utilities and gitignore processing
- File analysis and rendering decisions
- Tree structure generation
- Multiple output format generation (HTML, CXML, Repomix)
"""

# Core classes and functions that should be available at package level
from .file_analysis import (
    RenderDecision, FileInfo, decide_file, decide_file_simple, 
    looks_binary, load_file_content, estimate_tokens_simple, bytes_human,
    collect_files
)
from .glob_patterns import (
    parse_comma_separated_globs, should_include_path, match_glob_pattern, 
    _match_recursive_glob
)
from .git_utils import (
    git_clone, git_head_commit, should_ignore_path, run, 
    parse_gitignore_patterns, match_gitignore_pattern
)
from .tree_utils import generate_tree_fallback, try_tree_command
from .output_formats import (
    build_html, generate_cxml_text, generate_repomix_text, 
    get_file_icon, slugify, derive_temp_output_path, read_text
)
from .fastpath import should_use_intelligent_mode, select_files_fastpath, FASTPATH_AVAILABLE

# Import main function and constants from internal main module
from .main import main, MAX_DEFAULT_BYTES

__version__ = "1.0.0"

__all__ = [
    # File analysis
    "RenderDecision",
    "FileInfo", 
    "decide_file",
    "decide_file_simple",
    "looks_binary",
    "load_file_content",
    "estimate_tokens_simple",
    "bytes_human",
    "collect_files",
    
    # Glob patterns
    "parse_comma_separated_globs",
    "should_include_path",
    "match_glob_pattern",
    "_match_recursive_glob",
    
    # Git utilities
    "git_clone",
    "git_head_commit", 
    "should_ignore_path",
    "run",
    "parse_gitignore_patterns",
    "match_gitignore_pattern",
    
    # Tree utilities
    "generate_tree_fallback",
    "try_tree_command",
    
    # Output formats
    "build_html",
    "generate_cxml_text", 
    "generate_repomix_text",
    "get_file_icon",
    "slugify",
    "derive_temp_output_path",
    "read_text",
    
    # FastPath intelligent selection
    "should_use_intelligent_mode",
    "select_files_fastpath", 
    "FASTPATH_AVAILABLE",
    
    # Main entry point and constants
    "main",
    "MAX_DEFAULT_BYTES",
]
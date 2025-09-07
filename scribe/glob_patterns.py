#!/usr/bin/env python3
"""Glob pattern matching utilities."""

import fnmatch
import re
from typing import List


def parse_comma_separated_globs(pattern_string: str) -> List[str]:
    """Parse comma-separated glob patterns and return a list of patterns."""
    if not pattern_string.strip():
        return []
    
    patterns = []
    for pattern in pattern_string.split(','):
        pattern = pattern.strip()
        if pattern:
            patterns.append(pattern)
    return patterns


def should_include_path(path: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    """Check if a relative path should be included based on include/exclude patterns.
    
    Args:
        path: Relative path of the file (forward slash separated)
        include_patterns: List of glob patterns to include (empty means include all)
        exclude_patterns: List of glob patterns to exclude
    
    Returns:
        True if the file should be included, False otherwise
    
    Logic:
        - If include_patterns is empty, include all files by default
        - If include_patterns is not empty, only include files that match at least one include pattern
        - Always exclude files that match any exclude pattern (exclude takes precedence)
    """
    # First check exclude patterns - they take precedence
    for pattern in exclude_patterns:
        if match_glob_pattern(pattern, path):
            return False
    
    # If no include patterns specified, include by default (unless excluded above)
    if not include_patterns:
        return True
    
    # If include patterns specified, file must match at least one
    for pattern in include_patterns:
        if match_glob_pattern(pattern, path):
            return True
    
    return False


def match_glob_pattern(pattern: str, path: str) -> bool:
    """Match a glob pattern against a relative path.
    
    Supports standard glob patterns:
    - * matches any characters within a single path component
    - ** matches any characters including path separators (recursive)
    - ? matches any single character
    - [seq] matches any character in seq
    - [!seq] matches any character not in seq
    """
    # Handle directory patterns ending with /
    if pattern.endswith('/'):
        pattern_dir = pattern.rstrip('/')
        if path.startswith(pattern_dir + '/'):
            return True
        # Also check if the path itself matches the directory
        if fnmatch.fnmatch(path, pattern_dir):
            return True
        return False
    
    # Handle recursive glob patterns (**)
    if '**' in pattern:
        # Use a custom recursive glob implementation
        return _match_recursive_glob(pattern, path)
    
    # For non-recursive patterns, try full path first
    if fnmatch.fnmatch(path, pattern):
        return True
    
    # Also try matching just the filename for convenience
    filename = path.split('/')[-1]
    if fnmatch.fnmatch(filename, pattern):
        return True
    
    return False


def _match_recursive_glob(pattern: str, path: str) -> bool:
    """Match a recursive glob pattern containing ** against a relative path."""
    # Convert the pattern to a regular expression
    pattern_parts = pattern.split('/')
    regex_parts = []
    
    for part in pattern_parts:
        if part == '**':
            # ** matches zero or more path segments
            regex_parts.append(r'(?:[^/]+/)*(?:[^/]+)?')
        elif '**' in part and part != '**':
            # Handle mixed patterns like '**/*.py' (though this shouldn't happen after split)
            part_regex = re.escape(part).replace(r'\*\*', r'.*').replace(r'\*', r'[^/]*').replace(r'\?', r'[^/]')
            regex_parts.append(part_regex)
        else:
            # Normal glob part
            part_regex = re.escape(part).replace(r'\*', r'[^/]*').replace(r'\?', r'[^/]')
            regex_parts.append(part_regex)
    
    # Join parts with '/' and create full regex
    full_regex = '/'.join(regex_parts)
    
    # Handle special cases
    if pattern.startswith('**/'):
        # **/pattern should match pattern anywhere
        # Remove the leading **/ part and match against path parts
        remaining_pattern = '/'.join(pattern_parts[1:])
        path_parts = path.split('/')
        
        # Try matching the remaining pattern against every possible suffix of the path
        for i in range(len(path_parts)):
            candidate = '/'.join(path_parts[i:])
            if fnmatch.fnmatch(candidate, remaining_pattern):
                return True
        return False
    
    elif pattern.endswith('/**'):
        # pattern/** should match the pattern and everything below
        base_pattern = '/'.join(pattern_parts[:-1])
        return path.startswith(base_pattern) or fnmatch.fnmatch(path, base_pattern)
    
    else:
        # General case with ** in the middle
        # For patterns like 'src/**/*.py', we need to match:
        # src/main.py (** matches zero segments)
        # src/utils/main.py (** matches one segment)  
        # src/deep/nested/main.py (** matches multiple segments)
        
        # Split into prefix, middle, and suffix parts
        double_star_index = pattern_parts.index('**')
        prefix_parts = pattern_parts[:double_star_index]
        suffix_parts = pattern_parts[double_star_index + 1:]
        
        # Build prefix and suffix patterns
        prefix_pattern = '/'.join(prefix_parts) if prefix_parts else ''
        suffix_pattern = '/'.join(suffix_parts) if suffix_parts else ''
        
        # Check if path matches the structure
        path_parts = path.split('/')
        
        # Check prefix match
        if prefix_parts:
            if len(path_parts) < len(prefix_parts):
                return False
            prefix_path = '/'.join(path_parts[:len(prefix_parts)])
            if not fnmatch.fnmatch(prefix_path, prefix_pattern):
                return False
        
        # Check suffix match  
        if suffix_parts:
            if len(path_parts) < len(suffix_parts):
                return False
            suffix_path = '/'.join(path_parts[-len(suffix_parts):])
            if not fnmatch.fnmatch(suffix_path, suffix_pattern):
                return False
        
        # If we get here, the pattern matches
        return True
"""File pattern matching utilities.

This module provides standardized file pattern matching functionality used
across the FastPath system for consistent file filtering and matching behavior.
"""

import fnmatch
import re
from pathlib import Path
from typing import List, Pattern, Optional, Set
from enum import Enum


class MatchMode(Enum):
    """File pattern matching modes."""
    EXACT = "exact"           # Exact path matching
    SUFFIX = "suffix"         # File path ends with pattern  
    GLOB = "glob"            # Glob-style pattern matching with * and ?
    REGEX = "regex"          # Regular expression matching


class FilePatternMatcher:
    """Centralized utility for consistent file pattern matching.
    
    Provides various matching strategies used across the FastPath system
    for entry point matching, file filtering, and path comparisons.
    """
    
    def __init__(self):
        self._compiled_regex_cache: dict[str, Pattern] = {}
    
    def matches(
        self, 
        file_path: str, 
        pattern: str, 
        mode: MatchMode = MatchMode.GLOB
    ) -> bool:
        """Check if file path matches the given pattern.
        
        Args:
            file_path: File path to test
            pattern: Pattern to match against
            mode: Matching mode to use
            
        Returns:
            True if the file path matches the pattern
        """
        # Normalize paths to use forward slashes
        file_path = Path(file_path).as_posix()
        pattern = Path(pattern).as_posix()
        
        if mode == MatchMode.EXACT:
            return self._exact_match(file_path, pattern)
        elif mode == MatchMode.SUFFIX:
            return self._suffix_match(file_path, pattern)
        elif mode == MatchMode.GLOB:
            return self._glob_match(file_path, pattern)
        elif mode == MatchMode.REGEX:
            return self._regex_match(file_path, pattern)
        else:
            raise ValueError(f"Unknown match mode: {mode}")
    
    def matches_any(
        self, 
        file_path: str, 
        patterns: List[str], 
        mode: MatchMode = MatchMode.GLOB
    ) -> bool:
        """Check if file path matches any of the given patterns.
        
        Args:
            file_path: File path to test
            patterns: List of patterns to match against
            mode: Matching mode to use
            
        Returns:
            True if the file path matches any pattern
        """
        return any(self.matches(file_path, pattern, mode) for pattern in patterns)
    
    def filter_files(
        self,
        file_paths: List[str],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        mode: MatchMode = MatchMode.GLOB
    ) -> List[str]:
        """Filter file paths using include/exclude patterns.
        
        Args:
            file_paths: List of file paths to filter
            include_patterns: Patterns that files must match (if specified)
            exclude_patterns: Patterns that files must not match
            mode: Matching mode to use
            
        Returns:
            Filtered list of file paths
        """
        filtered = []
        
        for file_path in file_paths:
            # Check include patterns
            if include_patterns and not self.matches_any(file_path, include_patterns, mode):
                continue
                
            # Check exclude patterns
            if exclude_patterns and self.matches_any(file_path, exclude_patterns, mode):
                continue
                
            filtered.append(file_path)
            
        return filtered
    
    def _exact_match(self, file_path: str, pattern: str) -> bool:
        """Exact path matching."""
        return file_path == pattern
    
    def _suffix_match(self, file_path: str, pattern: str) -> bool:
        """Check if file path ends with pattern."""
        # Handle both exact suffix and path suffix matching
        if file_path.endswith(pattern):
            return True
        
        # Check if pattern matches the end of the path considering path separators
        # e.g., "main.py" should match "src/main.py"
        if '/' not in pattern:
            return file_path.endswith('/' + pattern) or file_path == pattern
            
        return False
    
    def _glob_match(self, file_path: str, pattern: str) -> bool:
        """Glob-style pattern matching with * and ? wildcards."""
        return fnmatch.fnmatch(file_path, pattern)
    
    def _regex_match(self, file_path: str, pattern: str) -> bool:
        """Regular expression pattern matching."""
        if pattern not in self._compiled_regex_cache:
            try:
                self._compiled_regex_cache[pattern] = re.compile(pattern)
            except re.error:
                return False
                
        return bool(self._compiled_regex_cache[pattern].search(file_path))
    
    def get_common_extensions(self) -> Set[str]:
        """Get commonly used file extensions for default filtering."""
        return {
            '.py', '.js', '.ts', '.jsx', '.tsx',
            '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.php', '.rb', '.swift',
            '.kt', '.scala', '.clj', '.hs', '.ml',
            '.r', '.jl', '.dart', '.lua', '.sh',
            '.sql', '.html', '.css', '.scss', '.less',
            '.json', '.yaml', '.yml', '.xml', '.toml',
            '.md', '.rst', '.txt', '.cfg', '.ini'
        }
    
    def get_code_extensions(self) -> Set[str]:
        """Get file extensions that are considered source code."""
        return {
            '.py', '.js', '.ts', '.jsx', '.tsx',
            '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.php', '.rb', '.swift',
            '.kt', '.scala', '.clj', '.hs', '.ml',
            '.r', '.jl', '.dart', '.lua', '.sh',
            '.sql', '.html', '.css', '.scss', '.less'
        }
    
    def is_code_file(self, file_path: str) -> bool:
        """Check if file path appears to be a source code file based on extension."""
        ext = Path(file_path).suffix.lower()
        return ext in self.get_code_extensions()
    
    def create_entry_point_patterns(
        self, 
        function_name: Optional[str] = None, 
        class_name: Optional[str] = None
    ) -> List[str]:
        """Create regex patterns for finding functions/classes in files.
        
        Args:
            function_name: Name of function to find
            class_name: Name of class to find
            
        Returns:
            List of regex patterns that can be used to find the symbols
        """
        patterns = []
        
        if class_name:
            # Match class definitions: "class ClassName:" or "class ClassName("
            patterns.append(rf"class\s+{re.escape(class_name)}\s*[\(:]")
            
        if function_name:
            # Match function definitions: "def function_name("
            patterns.append(rf"def\s+{re.escape(function_name)}\s*\(")
            
        return patterns
    
    def find_symbol_in_file(
        self,
        file_path: str,
        function_name: Optional[str] = None,
        class_name: Optional[str] = None,
        encoding: str = 'utf-8'
    ) -> bool:
        """Check if file contains specified function or class symbols.
        
        Args:
            file_path: Path to file to search
            function_name: Name of function to find
            class_name: Name of class to find 
            encoding: File encoding to use
            
        Returns:
            True if file contains the specified symbols
        """
        if not function_name and not class_name:
            return True  # No symbols specified, consider it a match
            
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            return False
        
        patterns = self.create_entry_point_patterns(function_name, class_name)
        
        for pattern in patterns:
            if not re.search(pattern, content):
                return False
                
        return True
    
    @staticmethod
    def get_default_ignore_patterns() -> List[str]:
        """Get default file patterns that should typically be ignored."""
        return [
            '*.lock', '*.log', '*.tmp', '*.cache', '*.bak',
            '*.pyc', '*.pyo', '*.pyd', '__pycache__/*',
            'node_modules/*', '.git/*', '.svn/*', '.hg/*',
            '.DS_Store', 'Thumbs.db', '*.swp', '*.swo',
            '.pytest_cache/*', '.coverage', '.nyc_output/*',
            'build/*', 'dist/*', 'target/*', 'bin/*', 'obj/*',
            '*.o', '*.so', '*.dll', '*.exe', '*.class',
        ]
    
    @staticmethod
    def get_documentation_patterns() -> List[str]:
        """Get patterns for documentation files."""
        return [
            '*.md', '*.rst', '*.txt',
            'README*', 'CHANGELOG*', 'LICENSE*', 'CONTRIBUTING*',
            'docs/*', 'documentation/*', 'doc/*'
        ]
    
    @staticmethod
    def get_configuration_patterns() -> List[str]:
        """Get patterns for configuration files."""
        return [
            '*.json', '*.yaml', '*.yml', '*.toml', '*.ini', '*.cfg',
            '*.conf', '*.config', '.env*', 'Dockerfile*',
            '*.xml', '*.plist', '*.properties'
        ]
    
    def should_ignore(self, file_path: str, ignore_patterns: List[str]) -> bool:
        """Check if file should be ignored based on ignore patterns.
        
        Args:
            file_path: Path to file to check
            ignore_patterns: List of patterns to ignore
            
        Returns:
            True if file should be ignored
        """
        return self.matches_any(file_path, ignore_patterns)
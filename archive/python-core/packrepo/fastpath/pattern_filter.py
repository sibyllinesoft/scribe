"""
Pattern-based file filtering with .gitignore support.

Provides include/exclude pattern matching compatible with repomix-style filtering.
"""

import os
import re
import fnmatch
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """Configuration for pattern-based file filtering."""
    include_patterns: List[str]
    exclude_patterns: List[str]
    use_gitignore: bool = True
    use_default_patterns: bool = True
    max_file_size: int = 50_000_000  # 50MB like repomix
    

class PatternFilter:
    """
    File pattern filtering with .gitignore support.
    
    Implements priority-based filtering similar to repomix:
    1. Custom exclude patterns (highest priority)
    2. .gitignore patterns (if enabled)
    3. Default exclude patterns (if enabled)
    4. Include patterns (lowest priority)
    """
    
    # Default patterns similar to repomix
    DEFAULT_EXCLUDE_PATTERNS = [
        # Version control
        '.git/**',
        '.svn/**', 
        '.hg/**',
        '.bzr/**',
        
        # Dependencies
        'node_modules/**',
        'vendor/**',
        '.venv/**',
        'venv/**',
        '__pycache__/**',
        '.pytest_cache/**',
        
        # Build outputs
        'build/**',
        'dist/**',
        'target/**',
        'out/**',
        'bin/**',
        'obj/**',
        
        # IDE files
        '.idea/**',
        '.vscode/**',
        '*.swp',
        '*.swo',
        '*~',
        
        # OS files
        '.DS_Store',
        'Thumbs.db',
        'desktop.ini',
        
        # Temporary files
        '*.tmp',
        '*.temp',
        '*.log',
        '*.cache',
        
        # Binary files
        '*.exe',
        '*.dll',
        '*.so',
        '*.dylib',
        '*.o',
        '*.obj',
        '*.pyc',
        '*.pyo',
        '*.class',
        
        # Large media files
        '*.jpg',
        '*.jpeg', 
        '*.png',
        '*.gif',
        '*.bmp',
        '*.tiff',
        '*.ico',
        '*.mp4',
        '*.avi',
        '*.mov',
        '*.wmv',
        '*.flv',
        '*.mp3',
        '*.wav',
        '*.flac',
        '*.zip',
        '*.tar.gz',
        '*.rar',
        '*.7z',
        
        # Documentation that's often auto-generated
        'coverage/**',
        'htmlcov/**',
        '.coverage',
        'junit.xml',
        '.nyc_output/**',
    ]
    
    def __init__(self, config: FilterConfig, repo_path: Path):
        self.config = config
        self.repo_path = repo_path
        
        # Compiled patterns for efficiency
        self._include_regexes: List[re.Pattern] = []
        self._exclude_regexes: List[re.Pattern] = []
        self._gitignore_patterns: List[str] = []
        
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile all patterns into regex for efficient matching."""
        # Include patterns
        for pattern in self.config.include_patterns:
            regex = self._glob_to_regex(pattern)
            self._include_regexes.append(re.compile(regex))
            
        # Custom exclude patterns
        for pattern in self.config.exclude_patterns:
            regex = self._glob_to_regex(pattern)
            self._exclude_regexes.append(re.compile(regex))
            
        # Default exclude patterns (if enabled)
        if self.config.use_default_patterns:
            for pattern in self.DEFAULT_EXCLUDE_PATTERNS:
                regex = self._glob_to_regex(pattern)
                self._exclude_regexes.append(re.compile(regex))
                
        # Load .gitignore patterns (if enabled)
        if self.config.use_gitignore:
            self._load_gitignore_patterns()
            
        # Always load .scribeignore/.repomixignore (with fallback)
        self._load_scribe_ignore_patterns()
            
    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob pattern to regex."""
        # Handle directory patterns
        if pattern.endswith('/'):
            pattern = pattern + '**'
        elif '/' not in pattern:
            # File patterns should match in any directory
            pattern = '**/' + pattern
            
        # Convert glob to regex
        regex_parts = []
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == '*':
                if i + 1 < len(pattern) and pattern[i + 1] == '*':
                    # ** matches any number of directories
                    if i + 2 < len(pattern) and pattern[i + 2] == '/':
                        regex_parts.append(r'(?:[^/]+/)*')
                        i += 3
                    else:
                        regex_parts.append(r'.*')
                        i += 2
                else:
                    # * matches anything except /
                    regex_parts.append(r'[^/]*')
                    i += 1
            elif c == '?':
                regex_parts.append(r'[^/]')
                i += 1
            elif c in r'\.^$+{}[]|()':
                regex_parts.append('\\' + c)
                i += 1
            else:
                regex_parts.append(c)
                i += 1
                
        return '^' + ''.join(regex_parts) + '$'
        
    def _load_gitignore_patterns(self):
        """Load patterns from .gitignore files."""
        gitignore_files = [
            self.repo_path / '.gitignore',
            # Could also check parent directories for global .gitignore
        ]
        
        for gitignore_file in gitignore_files:
            if gitignore_file.exists():
                self._load_ignore_file(gitignore_file)
                    
    def _load_scribe_ignore_patterns(self):
        """Load patterns from .scribeignore or .repomixignore files (with fallback)."""
        # Priority order: .scribeignore -> .repomixignore
        ignore_files = [
            self.repo_path / '.scribeignore',
            self.repo_path / '.repomixignore',
        ]
        
        for ignore_file in ignore_files:
            if ignore_file.exists():
                self._load_ignore_file(ignore_file)
                # Only load the first one found (prioritize .scribeignore)
                break
                
    def _load_ignore_file(self, file_path: Path):
        """Load patterns from a single ignore file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Handle negation patterns (!)
                    if line.startswith('!'):
                        # TODO: Implement negation logic - for now skip
                        continue
                    
                    self._gitignore_patterns.append(line)
                    regex = self._glob_to_regex(line)
                    self._exclude_regexes.append(re.compile(regex))
                    
        except (IOError, UnicodeDecodeError):
            # Skip if can't read ignore file
            pass
                    
    def should_include(self, file_path: Path) -> bool:
        """
        Determine if a file should be included based on patterns.
        
        Priority order:
        1. Custom exclude patterns (highest)
        2. .gitignore patterns
        3. Default exclude patterns  
        4. Include patterns (lowest)
        """
        relative_path = str(file_path.relative_to(self.repo_path))
        
        # Check file size limit
        try:
            if file_path.stat().st_size > self.config.max_file_size:
                return False
        except (OSError, AttributeError):
            return False
            
        # Check custom exclude patterns (highest priority)
        custom_excludes = self.config.exclude_patterns
        for pattern in custom_excludes:
            if self._match_pattern(relative_path, pattern):
                return False
                
        # Check all compiled exclude patterns (includes gitignore and defaults)
        for regex in self._exclude_regexes:
            if regex.match(relative_path):
                return False
                
        # If no include patterns specified, include by default
        if not self.config.include_patterns:
            return True
            
        # Check include patterns
        for regex in self._include_regexes:
            if regex.match(relative_path):
                return True
                
        # If include patterns specified but none match, exclude
        return False
        
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Match a single pattern against a path."""
        regex = self._glob_to_regex(pattern)
        return bool(re.match(regex, path))
        
    def filter_files(self, file_paths: List[Path]) -> List[Path]:
        """Filter a list of file paths based on patterns."""
        return [path for path in file_paths if self.should_include(path)]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        return {
            'include_patterns': len(self.config.include_patterns),
            'exclude_patterns': len(self.config.exclude_patterns),
            'gitignore_patterns': len(self._gitignore_patterns),
            'default_patterns_enabled': self.config.use_default_patterns,
            'gitignore_enabled': self.config.use_gitignore,
            'max_file_size_mb': self.config.max_file_size / (1024 * 1024),
        }


def create_repomix_compatible_filter(
    repo_path: Path,
    include: Optional[List[str]] = None,
    ignore_custom_patterns: Optional[List[str]] = None,
    use_gitignore: bool = True,
    use_default_patterns: bool = True,
    max_file_size: int = 50_000_000
) -> PatternFilter:
    """Create a pattern filter with repomix-compatible defaults."""
    config = FilterConfig(
        include_patterns=include or ['**/*'],
        exclude_patterns=ignore_custom_patterns or [],
        use_gitignore=use_gitignore,
        use_default_patterns=use_default_patterns,
        max_file_size=max_file_size
    )
    
    return PatternFilter(config, repo_path)


# Example usage for testing
if __name__ == "__main__":
    # Test the pattern filtering
    repo_path = Path.cwd()
    
    filter_config = FilterConfig(
        include_patterns=['**/*.py', '**/*.md'],
        exclude_patterns=['**/test_*', '**/tests/**'],
        use_gitignore=True,
        use_default_patterns=True
    )
    
    pattern_filter = PatternFilter(filter_config, repo_path)
    
    # Test with some example paths
    test_paths = [
        repo_path / "main.py",
        repo_path / "test_main.py", 
        repo_path / "README.md",
        repo_path / "node_modules" / "package" / "index.js",
        repo_path / ".git" / "config",
    ]
    
    for path in test_paths:
        if path.exists():
            result = pattern_filter.should_include(path)
            print(f"{path.relative_to(repo_path)}: {'INCLUDE' if result else 'EXCLUDE'}")
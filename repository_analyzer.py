#!/usr/bin/env python3
"""
Repository Analyzer for Intelligent Scribe Defaults

This module analyzes repositories to understand their structure, content, and characteristics,
enabling intelligent configuration generation for optimal scribe performance.
"""

import os
import pathlib
import json
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import subprocess
import re
from collections import defaultdict, Counter


@dataclass
class FileStats:
    """Statistics for individual files."""
    path: str
    size_bytes: int
    extension: str
    is_source_code: bool
    estimated_tokens: int
    language: Optional[str] = None


@dataclass
class RepositoryAnalysis:
    """Complete repository analysis results."""
    total_files: int
    total_size_bytes: int
    
    # Source code analysis
    source_files: List[FileStats]
    total_source_size_bytes: int
    estimated_source_tokens: int
    
    # Language breakdown
    languages: Dict[str, int]  # language -> file count
    language_tokens: Dict[str, int]  # language -> estimated tokens
    
    # File patterns
    excluded_patterns: List[str]
    large_files: List[FileStats]  # Files > 1MB
    binary_files: List[str]
    
    # Repository characteristics  
    is_monorepo: bool
    has_node_modules: bool
    has_python_venv: bool
    has_build_artifacts: bool
    
    # Recommendations
    optimal_token_budget: int
    recommended_algorithm: str
    confidence_score: float  # 0-1, how confident we are in our analysis


class RepositoryAnalyzer:
    """Analyzes repositories to determine optimal scribe configuration."""
    
    # Source code extensions by language
    SOURCE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.kt': 'kotlin',
        '.swift': 'swift',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'objective-c',
        '.mm': 'objective-cpp',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.fish': 'shell',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.dart': 'dart',
        '.lua': 'lua',
        '.nim': 'nim',
        '.zig': 'zig',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.md': 'markdown',
        '.rst': 'rst',
        '.tex': 'latex',
    }
    
    # Binary extensions to exclude
    BINARY_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg', '.ico',
        '.pdf', '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
        '.mp3', '.mp4', '.mov', '.avi', '.mkv', '.wav', '.ogg', '.flac',
        '.ttf', '.otf', '.eot', '.woff', '.woff2',
        '.so', '.dll', '.dylib', '.class', '.jar', '.exe', '.bin',
        '.pyc', '.pyo', '.pyd',
    }
    
    # Patterns to exclude from analysis
    EXCLUDE_PATTERNS = {
        'node_modules', '__pycache__', '.git', '.svn', '.hg',
        'target', 'build', 'dist', 'out', 'bin', 'obj',
        '.idea', '.vscode', '.vs',
        'coverage', '.coverage', 'htmlcov',
        '.tox', '.nox', 'venv', '.venv', 'env', '.env',
        '.pytest_cache', '.mypy_cache',
        'Pods', 'DerivedData',
        '.gradle', '.m2',
    }
    
    def __init__(self, repo_path: pathlib.Path):
        self.repo_path = repo_path.resolve()
        
    def analyze(self) -> RepositoryAnalysis:
        """Perform complete repository analysis."""
        print(f"🔍 Analyzing repository: {self.repo_path}")
        
        # Scan all files
        all_files = self._scan_files()
        print(f"📁 Found {len(all_files)} total files")
        
        # Analyze source code files
        source_files = [f for f in all_files if f.is_source_code]
        print(f"💻 Found {len(source_files)} source code files")
        
        # Calculate statistics
        total_size = sum(f.size_bytes for f in all_files)
        source_size = sum(f.size_bytes for f in source_files)
        source_tokens = sum(f.estimated_tokens for f in source_files)
        
        # Language analysis
        languages = Counter()
        language_tokens = defaultdict(int)
        for f in source_files:
            if f.language:
                languages[f.language] += 1
                language_tokens[f.language] += f.estimated_tokens
        
        # Identify large files (>1MB)
        large_files = [f for f in all_files if f.size_bytes > 1024 * 1024]
        
        # Binary files
        binary_files = [f.path for f in all_files if not f.is_source_code and f.extension in self.BINARY_EXTENSIONS]
        
        # Repository characteristics
        characteristics = self._analyze_characteristics()
        
        # Generate recommendations
        optimal_budget, algorithm, confidence = self._generate_recommendations(
            source_tokens, len(source_files), dict(languages), characteristics
        )
        
        return RepositoryAnalysis(
            total_files=len(all_files),
            total_size_bytes=total_size,
            source_files=source_files,
            total_source_size_bytes=source_size,
            estimated_source_tokens=source_tokens,
            languages=dict(languages),
            language_tokens=dict(language_tokens),
            excluded_patterns=list(self.EXCLUDE_PATTERNS),
            large_files=large_files,
            binary_files=binary_files,
            is_monorepo=characteristics['is_monorepo'],
            has_node_modules=characteristics['has_node_modules'],
            has_python_venv=characteristics['has_python_venv'],
            has_build_artifacts=characteristics['has_build_artifacts'],
            optimal_token_budget=optimal_budget,
            recommended_algorithm=algorithm,
            confidence_score=confidence
        )
    
    def _scan_files(self) -> List[FileStats]:
        """Scan repository and analyze all files."""
        files = []
        
        for root, dirs, filenames in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]
            
            for filename in filenames:
                filepath = pathlib.Path(root) / filename
                relative_path = filepath.relative_to(self.repo_path)
                
                # Skip if file matches exclude patterns
                if self._should_exclude_file(str(relative_path)):
                    continue
                    
                try:
                    stat = filepath.stat()
                    size_bytes = stat.st_size
                    
                    # Skip empty files
                    if size_bytes == 0:
                        continue
                        
                    extension = filepath.suffix.lower()
                    is_source = self._is_source_code_file(filepath, extension)
                    tokens = self._estimate_tokens(filepath, size_bytes) if is_source else 0
                    language = self.SOURCE_EXTENSIONS.get(extension) if is_source else None
                    
                    files.append(FileStats(
                        path=str(relative_path),
                        size_bytes=size_bytes,
                        extension=extension,
                        is_source_code=is_source,
                        estimated_tokens=tokens,
                        language=language
                    ))
                    
                except (OSError, PermissionError):
                    continue  # Skip files we can't access
        
        return files
    
    def _should_exclude_dir(self, dirname: str) -> bool:
        """Check if directory should be excluded from analysis."""
        return dirname in self.EXCLUDE_PATTERNS or dirname.startswith('.')
    
    def _should_exclude_file(self, relative_path: str) -> bool:
        """Check if file should be excluded from analysis."""
        path_parts = relative_path.split('/')
        return any(part in self.EXCLUDE_PATTERNS for part in path_parts)
    
    def _is_source_code_file(self, filepath: pathlib.Path, extension: str) -> bool:
        """Determine if file is source code."""
        # Check extension first
        if extension in self.SOURCE_EXTENSIONS:
            return True
            
        # Check for binary content
        if extension in self.BINARY_EXTENSIONS:
            return False
            
        # For files without clear extensions, check content
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(8192)
                
            # If contains null bytes, likely binary
            if b'\x00' in chunk:
                return False
                
            # Try to decode as UTF-8
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
                
        except (OSError, PermissionError):
            return False
    
    def _estimate_tokens(self, filepath: pathlib.Path, size_bytes: int) -> int:
        """Estimate token count for source file."""
        if size_bytes == 0:
            return 0
            
        # Simple heuristic: ~4 characters per token for most programming languages
        # This can be refined with actual tokenization if needed
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return max(1, len(content) // 4)
        except (OSError, UnicodeDecodeError):
            # Fallback to size-based estimation
            return max(1, size_bytes // 4)
    
    def _analyze_characteristics(self) -> Dict[str, bool]:
        """Analyze repository characteristics."""
        characteristics = {}
        
        # Check for monorepo indicators
        has_multiple_langs = len(list(self.repo_path.rglob("*.py"))) > 0 and len(list(self.repo_path.rglob("*.js"))) > 0
        has_packages_dir = (self.repo_path / "packages").exists() or (self.repo_path / "apps").exists()
        characteristics['is_monorepo'] = has_multiple_langs and has_packages_dir
        
        # Check for Node.js
        characteristics['has_node_modules'] = (self.repo_path / "node_modules").exists()
        
        # Check for Python virtual environments
        venv_paths = ['venv', '.venv', 'env', '.env']
        characteristics['has_python_venv'] = any((self.repo_path / path).exists() for path in venv_paths)
        
        # Check for build artifacts
        build_paths = ['build', 'dist', 'target', 'out', 'bin']
        characteristics['has_build_artifacts'] = any((self.repo_path / path).exists() for path in build_paths)
        
        return characteristics
    
    def _generate_recommendations(self, source_tokens: int, source_files: int, 
                                 languages: Dict[str, int], characteristics: Dict[str, bool]) -> Tuple[int, str, float]:
        """Generate optimal configuration recommendations."""
        confidence = 0.8  # Start with high confidence
        
        # Determine optimal token budget based on actual content
        if source_tokens <= 10000:
            # Small repository - can include everything with margin
            optimal_budget = max(15000, int(source_tokens * 1.5))
            algorithm = "traditional"  # Simple filtering sufficient
        elif source_tokens <= 50000:
            # Medium repository - balance coverage and token limit
            optimal_budget = min(75000, int(source_tokens * 1.2))
            algorithm = "v5_integrated"  # Use intelligent selection
        elif source_tokens <= 200000:
            # Large repository - need intelligent selection
            optimal_budget = 100000
            algorithm = "v5_integrated"
        else:
            # Very large repository - focus on architectural overview
            optimal_budget = 150000
            algorithm = "v5_integrated"
            confidence = 0.6  # Lower confidence for very large repos
        
        # Adjust based on repository characteristics
        if characteristics.get('is_monorepo', False):
            # Monorepos need higher budgets for cross-component understanding
            optimal_budget = int(optimal_budget * 1.3)
            confidence *= 0.9
            
        # Adjust for language diversity
        if len(languages) > 3:
            optimal_budget = int(optimal_budget * 1.2)
            confidence *= 0.85
        
        return optimal_budget, algorithm, confidence
    
    def save_analysis(self, analysis: RepositoryAnalysis, output_path: pathlib.Path) -> None:
        """Save analysis results to JSON file."""
        # Convert dataclass to dict for JSON serialization
        data = {
            'total_files': analysis.total_files,
            'total_size_bytes': analysis.total_size_bytes,
            'source_files_count': len(analysis.source_files),
            'total_source_size_bytes': analysis.total_source_size_bytes,
            'estimated_source_tokens': analysis.estimated_source_tokens,
            'languages': analysis.languages,
            'language_tokens': analysis.language_tokens,
            'excluded_patterns': analysis.excluded_patterns,
            'large_files_count': len(analysis.large_files),
            'binary_files_count': len(analysis.binary_files),
            'is_monorepo': analysis.is_monorepo,
            'has_node_modules': analysis.has_node_modules,
            'has_python_venv': analysis.has_python_venv,
            'has_build_artifacts': analysis.has_build_artifacts,
            'optimal_token_budget': analysis.optimal_token_budget,
            'recommended_algorithm': analysis.recommended_algorithm,
            'confidence_score': analysis.confidence_score,
            'repository_path': str(self.repo_path),
        }
        
        output_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        print(f"💾 Analysis saved to: {output_path}")


def analyze_repository(repo_path: str) -> RepositoryAnalysis:
    """Convenient function to analyze a repository."""
    analyzer = RepositoryAnalyzer(pathlib.Path(repo_path))
    return analyzer.analyze()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python repository_analyzer.py <repository_path>")
        sys.exit(1)
    
    repo_path = pathlib.Path(sys.argv[1])
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    # Analyze repository
    analyzer = RepositoryAnalyzer(repo_path)
    analysis = analyzer.analyze()
    
    # Print summary
    print(f"\n📊 Repository Analysis Summary:")
    print(f"   Total files: {analysis.total_files}")
    print(f"   Source files: {len(analysis.source_files)}")
    print(f"   Estimated tokens: {analysis.estimated_source_tokens:,}")
    print(f"   Languages: {', '.join(analysis.languages.keys())}")
    print(f"   Optimal budget: {analysis.optimal_token_budget:,}")
    print(f"   Recommended algorithm: {analysis.recommended_algorithm}")
    print(f"   Confidence: {analysis.confidence_score:.2%}")
    
    # Save analysis
    output_path = repo_path / "scribe_analysis.json"
    analyzer.save_analysis(analysis, output_path)
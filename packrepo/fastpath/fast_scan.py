"""
FastPath scanning system - Core heuristic-based file analysis.

Performs rapid repository scanning with cheap heuristics:
- File statistics and metadata
- README/ARCHITECTURE/ADR priority detection  
- Import pattern analysis via regex
- Test-code relationship detection
- Document heading density analysis
- Recent change churn detection
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .feature_flags import get_feature_flags

# Common README-like file patterns
README_PATTERNS = {
    "readme", "read_me", "readme.md", "readme.txt", "readme.rst",
    "architecture", "architecture.md", "arch.md",
    "adr", "adrs", "decisions", "decision-records",
    "contributing", "contributing.md",
    "design", "design.md", "design-doc", "design_doc",
    "specification", "spec", "spec.md", "specifications",
    "overview", "overview.md", "getting-started", "getting_started"
}

# Entrypoint detection patterns
ENTRYPOINT_PATTERNS = {
    # Main function patterns (common across languages)
    'main_functions': [
        r'def\s+main\s*\(',  # Python
        r'function\s+main\s*\(',  # JavaScript
        r'func\s+main\s*\(',  # Go
        r'int\s+main\s*\(',  # C/C++
        r'fn\s+main\s*\(',  # Rust
        r'public\s+static\s+void\s+main\s*\(',  # Java
    ],
    # CLI setup patterns
    'cli_patterns': [
        r'argparse\.ArgumentParser',  # Python argparse
        r'click\.',  # Python click
        r'typer\.',  # Python typer
        r'commander\.',  # JavaScript commander
        r'yargs\.',  # JavaScript yargs
        r'cobra\.',  # Go cobra
        r'clap\.',  # Rust clap
    ],
    # Web framework boot patterns
    'web_boot_patterns': [
        r'app\.listen\s*\(',  # Express.js
        r'uvicorn\.run\s*\(',  # FastAPI/Python
        r'flask\.Flask\s*\(',  # Flask
        r'FastAPI\s*\(',  # FastAPI
        r'http\.ListenAndServe\s*\(',  # Go HTTP
        r'actix_web::\w+',  # Rust Actix
    ]
}

# Configuration file patterns
CONFIG_PATTERNS = {
    # Package manifests
    'manifests': {
        'package.json', 'package-lock.json', 'yarn.lock',  # JavaScript/Node.js
        'requirements.txt', 'setup.py', 'pyproject.toml', 'setup.cfg',  # Python
        'Cargo.toml', 'Cargo.lock',  # Rust
        'go.mod', 'go.sum',  # Go
        'pom.xml', 'build.gradle', 'build.gradle.kts',  # Java
        'Gemfile', 'Gemfile.lock',  # Ruby
        'composer.json', 'composer.lock',  # PHP
    },
    # Environment and configuration
    'config_files': {
        '.env', '.env.local', '.env.example',
        'config.json', 'config.yaml', 'config.yml',
        'settings.json', 'settings.yaml',
        'docker-compose.yml', 'docker-compose.yaml',
        'Dockerfile', '.dockerignore',
        'Makefile', 'makefile',
        '.gitignore', '.gitattributes',
        'tsconfig.json', 'tsconfig.build.json',
        'babel.config.js', 'webpack.config.js',
        'jest.config.js', 'vitest.config.ts',
    }
}

# Import regex patterns for different languages
IMPORT_PATTERNS = {
    "python": [
        r'^import\s+(\w+(?:\.\w+)*)',
        r'^from\s+(\w+(?:\.\w+)*)\s+import',
    ],
    "javascript": [
        r'(?:import|require)\s*\(?[\'"]([^\'\"]+)[\'"]',
        r'import\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]',
    ],
    "typescript": [
        r'(?:import|require)\s*\(?[\'"]([^\'\"]+)[\'"]',
        r'import\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]',
    ],
    "java": [
        r'^import\s+([\w.]+);',
        r'^import\s+static\s+([\w.]+);',
    ],
    "cpp": [
        r'#include\s*[<"]([^>"]+)[>"]',
    ],
    "rust": [
        r'^use\s+([\w:]+)(?:::.*)?;',
        r'^extern\s+crate\s+(\w+);',
    ],
    "go": [
        r'^import\s+"([^"]+)"',
        r'^\s*"([^"]+)"',  # In import blocks
    ],
}

# Test file patterns
TEST_PATTERNS = {
    r'test_.*\.py$',
    r'.*_test\.py$', 
    r'.*\.test\.js$',
    r'.*\.spec\.js$',
    r'.*\.test\.ts$',
    r'.*\.spec\.ts$',
    r'.*Test\.java$',
    r'test.*\.cpp$',
    r'.*_test\.cpp$',
    r'.*_test\.go$',
    r'.*_test\.rs$',
}


@dataclass
class FileStats:
    """Basic file statistics and metadata."""
    path: str
    size_bytes: int
    lines: int
    language: Optional[str]
    is_readme: bool
    is_test: bool
    is_config: bool
    is_docs: bool
    depth: int
    last_modified: float
    # New FastPath V2 fields (behind feature flags)
    is_entrypoint: bool = False
    has_examples: bool = False
    is_integration_test: bool = False


@dataclass 
class ImportAnalysis:
    """Import/dependency analysis results."""
    imports: Set[str]
    import_count: int
    unique_modules: int
    relative_imports: int
    external_imports: int


@dataclass
class DocumentAnalysis:
    """Document structure and content analysis."""
    heading_count: int
    toc_indicators: int
    link_count: int
    code_block_count: int
    heading_density: float  # headings per 100 lines


@dataclass
class ScanResult:
    """Complete scan result for a single file."""
    stats: FileStats
    imports: Optional[ImportAnalysis]
    doc_analysis: Optional[DocumentAnalysis]
    churn_score: float
    priority_boost: float
    # New FastPath V2 fields (behind feature flags)
    centrality_in: float = 0.0  # PageRank centrality in adjacency graph
    signatures: List[str] = field(default_factory=list)  # Function/class signatures for demotion


class FastScanner:
    """
    Fast repository scanner using heuristics only.
    
    Designed to complete full repository scan in <2s for typical repos.
    Uses minimal parsing and regex-based analysis for speed.
    """
    
    def __init__(self, repo_path: Path, ttl_seconds: float = 2.0):
        self.repo_path = repo_path
        self.ttl_seconds = ttl_seconds
        self.start_time: Optional[float] = None
        
        # Compile regex patterns once
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self.test_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in TEST_PATTERNS]
        
        self.import_regexes = {}
        for lang, patterns in IMPORT_PATTERNS.items():
            self.import_regexes[lang] = [re.compile(pattern, re.MULTILINE) for pattern in patterns]
            
    def _is_time_exceeded(self) -> bool:
        """Check if TTL has been exceeded."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.ttl_seconds
        
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        
        lang_map = {
            '.py': 'python',
            '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.c': 'cpp', '.h': 'cpp', '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.md': 'markdown', '.rst': 'markdown', '.txt': 'text',
        }
        
        return lang_map.get(suffix)
        
    def _is_readme_file(self, file_path: Path) -> bool:
        """Check if file is README-like documentation."""
        name_lower = file_path.name.lower()
        stem_lower = file_path.stem.lower()
        
        return (name_lower in README_PATTERNS or 
                stem_lower in README_PATTERNS or
                any(pattern in name_lower for pattern in ["readme", "architecture", "design"]))
        
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        name = file_path.name
        return any(regex.match(name) for regex in self.test_regexes)
        
    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is configuration."""
        name_lower = file_path.name.lower()
        
        # Check against comprehensive config patterns
        if name_lower in CONFIG_PATTERNS['manifests'] or name_lower in CONFIG_PATTERNS['config_files']:
            return True
        
        # Legacy pattern matching (keep for backward compatibility)
        config_indicators = {
            'config', 'configuration', 'settings', 'options',
            '.env', 'dockerfile', 'makefile', 'package.json',
            'requirements.txt', 'setup.py', 'pyproject.toml',
            'cargo.toml', 'go.mod', 'pom.xml'
        }
        
        return (any(indicator in name_lower for indicator in config_indicators) or
                file_path.suffix.lower() in {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'})
        
    def _is_docs_file(self, file_path: Path) -> bool:
        """Check if file is documentation."""
        if self._is_readme_file(file_path):
            return True
            
        parts_lower = [p.lower() for p in file_path.parts]
        doc_indicators = {'doc', 'docs', 'documentation', 'manual', 'guide', 'tutorial'}
        
        return (any(indicator in parts_lower for indicator in doc_indicators) or
                file_path.suffix.lower() in {'.md', '.rst', '.txt'})
    
    def _detect_entrypoint(self, content: str, file_path: Path) -> bool:
        """Detect if file contains entrypoint patterns (V2 feature)."""
        flags = get_feature_flags()
        if not flags.centrality_enabled:
            return False
        
        # Check for main functions
        for pattern in ENTRYPOINT_PATTERNS['main_functions']:
            if re.search(pattern, content, re.MULTILINE):
                return True
        
        # Check for CLI setup patterns
        for pattern in ENTRYPOINT_PATTERNS['cli_patterns']:
            if re.search(pattern, content, re.MULTILINE):
                return True
        
        # Check for web framework boot patterns
        for pattern in ENTRYPOINT_PATTERNS['web_boot_patterns']:
            if re.search(pattern, content, re.MULTILINE):
                return True
        
        return False
    
    def _detect_examples(self, content: str, file_path: Path) -> bool:
        """Detect if file contains example code or usage (V2 feature)."""
        flags = get_feature_flags()
        if not flags.centrality_enabled:
            return False
        
        # Check file path for example indicators
        path_lower = str(file_path).lower()
        if any(indicator in path_lower for indicator in ['example', 'examples', 'demo', 'sample', 'tutorial']):
            return True
        
        # Check content for example patterns
        content_lower = content.lower()
        example_indicators = [
            '# example', '## example', '### example',
            '# usage', '## usage', '### usage',
            '# demo', '## demo', '### demo',
            'example:', 'usage:', 'demo:',
            'if __name__ == "__main__":',  # Python main guard
            '>>> ',  # Python doctest
        ]
        
        return any(indicator in content_lower for indicator in example_indicators)
    
    def _classify_test_type(self, file_path: Path, content: str) -> bool:
        """Classify if test is integration test vs unit test (V2 feature)."""
        flags = get_feature_flags()
        if not flags.centrality_enabled:
            return False
        
        if not self._is_test_file(file_path):
            return False
        
        # Integration test indicators
        integration_indicators = [
            'integration', 'e2e', 'end2end', 'functional',
            'database', 'db', 'api', 'http', 'server',
            'docker', 'container', 'selenium', 'playwright'
        ]
        
        path_lower = str(file_path).lower()
        content_lower = content.lower()
        
        # Check file path
        if any(indicator in path_lower for indicator in integration_indicators):
            return True
        
        # Check imports and content
        if any(indicator in content_lower for indicator in integration_indicators):
            return True
        
        return False
    
    def _extract_signatures(self, content: str, language: Optional[str]) -> List[str]:
        """Extract function and class signatures for demotion system (V2 feature)."""
        flags = get_feature_flags()
        if not flags.demote_enabled:
            return []
        
        signatures = []
        
        if language == 'python':
            # Extract Python function and class signatures
            func_pattern = r'^(def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?)\s*:'
            class_pattern = r'^(class\s+\w+(?:\s*\([^)]*\))?)\s*:'
            
            for pattern in [func_pattern, class_pattern]:
                matches = re.findall(pattern, content, re.MULTILINE)
                signatures.extend(matches)
        
        elif language in ['javascript', 'typescript']:
            # Extract JavaScript/TypeScript function signatures
            func_patterns = [
                r'^(function\s+\w+\s*\([^)]*\))',
                r'^(const\s+\w+\s*=\s*\([^)]*\)\s*=>)',
                r'^(export\s+function\s+\w+\s*\([^)]*\))',
                r'^(class\s+\w+(?:\s+extends\s+\w+)?)\s*{',
            ]
            
            for pattern in func_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                signatures.extend(matches)
        
        elif language == 'go':
            # Extract Go function signatures
            func_pattern = r'^(func(?:\s*\([^)]*\))?\s+\w+\s*\([^)]*\)(?:\s+[^{]+)?)\s*{'
            matches = re.findall(func_pattern, content, re.MULTILINE)
            signatures.extend(matches)
        
        elif language == 'rust':
            # Extract Rust function signatures
            func_pattern = r'^((?:pub\s+)?fn\s+\w+(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?)\s*{'
            matches = re.findall(func_pattern, content, re.MULTILINE)
            signatures.extend(matches)
        
        return signatures[:10]  # Limit to first 10 signatures
    
    def _analyze_imports(self, content: str, language: str) -> ImportAnalysis:
        """Analyze import statements in code files."""
        if language not in self.import_regexes:
            return ImportAnalysis(set(), 0, 0, 0, 0)
            
        imports = set()
        relative_count = 0
        external_count = 0
        
        for regex in self.import_regexes[language]:
            matches = regex.findall(content)
            for match in matches:
                imports.add(match)
                
                # Heuristic classification
                if match.startswith('.') or '/' in match:
                    relative_count += 1
                else:
                    external_count += 1
                    
        return ImportAnalysis(
            imports=imports,
            import_count=len(imports),
            unique_modules=len(imports),
            relative_imports=relative_count,
            external_imports=external_count
        )
    
    def _analyze_document(self, content: str) -> DocumentAnalysis:
        """Analyze document structure for markdown/text files."""
        lines = content.split('\n')
        
        # Count headings (# in markdown)
        heading_count = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Look for TOC indicators
        toc_indicators = sum(1 for line in lines 
                           if any(pattern in line.lower() for pattern in 
                                ['table of contents', 'toc', '- [', '* [']))
        
        # Count links
        link_count = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content))
        
        # Count code blocks
        code_block_count = content.count('```') // 2
        
        # Calculate heading density
        total_lines = len(lines)
        heading_density = (heading_count / max(total_lines, 1)) * 100
        
        return DocumentAnalysis(
            heading_count=heading_count,
            toc_indicators=toc_indicators, 
            link_count=link_count,
            code_block_count=code_block_count,
            heading_density=heading_density
        )
    
    def _calculate_churn_score(self, file_path: Path) -> float:
        """Calculate file change frequency (simplified Git log analysis)."""
        # In a real implementation, this would use git log
        # For now, use file modification time as proxy
        try:
            stat = file_path.stat()
            days_since_modified = (time.time() - stat.st_mtime) / (24 * 3600)
            
            # Recent changes get higher churn score
            if days_since_modified < 1:
                return 1.0
            elif days_since_modified < 7:
                return 0.7
            elif days_since_modified < 30:
                return 0.4
            else:
                return 0.1
                
        except (OSError, AttributeError):
            return 0.1
            
    def _calculate_priority_boost(self, stats: FileStats, doc_analysis: Optional[DocumentAnalysis]) -> float:
        """Calculate priority boost based on file type and characteristics."""
        boost = 0.0
        
        # README and documentation files get major boost
        if stats.is_readme:
            boost += 2.0
            
        if stats.is_docs:
            boost += 1.0
            
        # Well-structured documents get boost
        if doc_analysis and doc_analysis.heading_density > 2.0:
            boost += 0.5
            
        # Configuration files get moderate boost
        if stats.is_config:
            boost += 0.3
            
        # Files closer to root get boost
        if stats.depth <= 2:
            boost += 0.2
            
        return boost
        
    def scan_file(self, file_path: Path) -> Optional[ScanResult]:
        """Scan a single file and return analysis result."""
        if self._is_time_exceeded():
            return None
            
        # Use ErrorHandler for file operations
        from .utils.error_handling import ErrorHandler, ErrorContext
        
        handler = ErrorHandler("FastPathScanner")
        context = ErrorContext(
            operation="scan_file",
            component="FastPathScanner",
            file_path=str(file_path)
        )
        
        def read_file_stats():
            stat = file_path.stat()
            # Skip very large files for speed
            if stat.st_size > 1024 * 1024:  # 1MB limit
                return None, None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return stat, content
        
        result = handler.safe_execute(read_file_stats, context)
        if result.is_failure:
            return None
        
        stat_result, content = result.value
        if stat_result is None:  # Large file case
            return None
        
        stat = stat_result
        
        lines = content.count('\n') + 1
        language = self._detect_language(file_path)
        
        # Create file stats with V2 enhancements
        file_stats = FileStats(
            path=str(file_path.relative_to(self.repo_path)),
            size_bytes=stat.st_size,
            lines=lines,
            language=language,
            is_readme=self._is_readme_file(file_path),
            is_test=self._is_test_file(file_path),
            is_config=self._is_config_file(file_path),
            is_docs=self._is_docs_file(file_path),
            depth=len(file_path.relative_to(self.repo_path).parts),
            last_modified=stat.st_mtime,
            # V2 fields (computed only when enabled)
            is_entrypoint=self._detect_entrypoint(content, file_path),
            has_examples=self._detect_examples(content, file_path),
            is_integration_test=self._classify_test_type(file_path, content)
        )
        
        # Analyze imports for code files
        import_analysis = None
        if language and language in self.import_regexes:
            import_analysis = self._analyze_imports(content, language)
            
        # Analyze document structure
        doc_analysis = None
        if file_stats.is_docs or language in {'markdown', 'text'}:
            doc_analysis = self._analyze_document(content)
            
        # Calculate scores
        churn_score = self._calculate_churn_score(file_path)
        priority_boost = self._calculate_priority_boost(file_stats, doc_analysis)
        
        # Extract signatures for demotion system (V2 feature)
        signatures = self._extract_signatures(content, language)
        
        return ScanResult(
            stats=file_stats,
            imports=import_analysis,
            doc_analysis=doc_analysis,
            churn_score=churn_score,
            priority_boost=priority_boost,
            centrality_in=0.0,  # Will be computed later by adjacency graph analysis
            signatures=signatures
        )
            
    def scan_repository(self, max_files: int = 2000) -> List[ScanResult]:
        """
        Scan entire repository with TTL protection.
        
        Returns list of scan results for processable files.
        Automatically stops when TTL exceeded or max_files reached.
        """
        self.start_time = time.time()
        results = []
        file_count = 0
        
        # Common directories to skip
        skip_dirs = {
            '.git', '.svn', '.hg',
            'node_modules', '__pycache__', '.pytest_cache',
            'target', 'build', 'dist', '.venv', 'venv',
            '.idea', '.vscode'
        }
        
        try:
            for root, dirs, files in os.walk(self.repo_path):
                # Skip excluded directories  
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                
                # Check TTL
                if self._is_time_exceeded():
                    break
                    
                for file_name in files:
                    if file_count >= max_files:
                        break
                        
                    file_path = Path(root) / file_name
                    
                    # Skip binary files and common excludes
                    if (file_path.suffix.lower() in {'.pyc', '.o', '.exe', '.dll', '.so', '.dylib'} or
                        file_name.startswith('.') and file_name not in {'.env', '.gitignore'}):
                        continue
                        
                    result = self.scan_file(file_path)
                    if result:
                        results.append(result)
                        file_count += 1
                        
                    # Check TTL periodically
                    if file_count % 100 == 0 and self._is_time_exceeded():
                        break
                        
        except Exception as e:
            # Log error but return partial results
            print(f"Scan interrupted: {e}")
            
        # Build adjacency graph and calculate PageRank centrality (V2 feature)
        if get_feature_flags().centrality_enabled and results:
            results = self._build_adjacency_graph_and_centrality(results)
        
        return results
    
    def _build_adjacency_graph_and_centrality(self, results: List[ScanResult]) -> List[ScanResult]:
        """
        Build adjacency graph from import relationships and calculate PageRank centrality.
        
        V2 feature that enhances file priority scoring with graph centrality measures.
        """
        if not results:
            return results
        
        try:
            # Build adjacency matrix
            file_to_idx = {result.stats.path: idx for idx, result in enumerate(results)}
            n = len(results)
            adjacency_matrix = np.zeros((n, n))
            
            # Fill adjacency matrix based on import relationships
            for i, result in enumerate(results):
                if not result.imports or not result.imports.imports:
                    continue
                
                for import_name in result.imports.imports:
                    # Try to resolve import to actual file paths
                    target_files = self._resolve_import_to_files(import_name, results, result.stats.path)
                    
                    for target_file in target_files:
                        if target_file in file_to_idx:
                            j = file_to_idx[target_file]
                            adjacency_matrix[i][j] = 1.0  # Directed edge from i to j
            
            # Calculate PageRank centrality
            centrality_scores = self._calculate_pagerank(adjacency_matrix)
            
            # Update results with centrality scores
            for i, result in enumerate(results):
                result.centrality_in = centrality_scores[i]
            
            return results
            
        except Exception as e:
            # If centrality calculation fails, return original results
            print(f"Warning: Centrality calculation failed: {e}")
            return results
    
    def _resolve_import_to_files(
        self, 
        import_name: str, 
        all_results: List[ScanResult], 
        importing_file: str
    ) -> List[str]:
        """
        Resolve import name to actual file paths in the repository.
        
        Uses heuristics to map import statements to file paths.
        """
        matched_files = []
        
        # Normalize import name
        import_parts = import_name.lower().replace('.', '/').replace('::', '/').split('/')
        
        for result in all_results:
            file_path = result.stats.path.lower()
            
            # Direct file name match
            if import_parts[-1] in file_path:
                matched_files.append(result.stats.path)
                continue
            
            # Module path match
            if len(import_parts) > 1:
                import_path = '/'.join(import_parts)
                if import_path in file_path:
                    matched_files.append(result.stats.path)
                    continue
            
            # Partial path match
            for part in import_parts:
                if part and part in file_path:
                    # Give lower priority to partial matches
                    if result.stats.path not in matched_files:
                        matched_files.append(result.stats.path)
                    break
        
        # Limit matches to avoid noise
        return matched_files[:5]
    
    def _calculate_pagerank(
        self, 
        adjacency_matrix: np.ndarray, 
        damping_factor: float = 0.85, 
        max_iterations: int = 100, 
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Calculate PageRank centrality scores using power iteration method.
        
        Args:
            adjacency_matrix: Directed adjacency matrix
            damping_factor: PageRank damping factor (default: 0.85)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Array of PageRank centrality scores
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        # Handle case with no edges
        if np.sum(adjacency_matrix) == 0:
            return np.ones(n) / n
        
        # Normalize adjacency matrix (column-stochastic)
        column_sums = np.sum(adjacency_matrix, axis=0)
        # Avoid division by zero
        column_sums[column_sums == 0] = 1.0
        transition_matrix = adjacency_matrix / column_sums
        
        # Initialize PageRank vector
        pagerank = np.ones(n) / n
        
        # Power iteration
        for iteration in range(max_iterations):
            prev_pagerank = pagerank.copy()
            
            # PageRank update formula
            pagerank = (1 - damping_factor) / n + damping_factor * np.dot(transition_matrix, pagerank)
            
            # Check convergence
            if np.linalg.norm(pagerank - prev_pagerank, ord=1) < tolerance:
                break
        
        return pagerank


def create_scanner(repo_path: str | Path, ttl_seconds: float = 2.0) -> FastScanner:
    """Create a FastScanner instance."""
    return FastScanner(Path(repo_path), ttl_seconds)
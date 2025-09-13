"""
Synthetic Repository Generator (Workstream C)

Generates realistic synthetic repositories for scalability testing:
- 10k, 100k, and 10M file repositories
- Realistic dependency graph structures  
- Multiple programming languages
- Configurable clustering and complexity patterns

Enables reproducible benchmarking for FastPath V5 ICSE submission.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import logging

from ..fastpath.centrality import DependencyGraph
from ..fastpath.fast_scan import ScanResult, FileStats, ImportAnalysis

logger = logging.getLogger(__name__)


class RepoScale(Enum):
    """Predefined repository scales for benchmarking."""
    SMALL = "small"      # ~10k files
    MEDIUM = "medium"    # ~100k files  
    LARGE = "large"      # ~10M files


@dataclass
class RepoConfig:
    """Configuration for synthetic repository generation."""
    target_files: int = 10_000
    avg_dependencies_per_file: float = 3.5
    language_distribution: Dict[str, float] = field(default_factory=lambda: {
        'python': 0.4, 'javascript': 0.3, 'java': 0.2, 'go': 0.1
    })
    directory_depth: int = 6
    clustering_factor: float = 0.7  # How clustered files are (0=random, 1=highly clustered)
    powerlaw_exponent: float = 2.1  # For realistic degree distribution
    seed: int = 42  # For reproducible generation


@dataclass  
class SyntheticFile:
    """Represents a synthetic file in the generated repository."""
    path: str
    language: str
    size_bytes: int
    imports: List[str] = field(default_factory=list)
    content_hash: str = ""
    directory_level: int = 0
    

@dataclass
class SyntheticRepository:
    """Complete synthetic repository data."""
    files: List[SyntheticFile]
    dependency_graph: DependencyGraph
    scan_results: List[ScanResult]
    metadata: Dict[str, Any]


class LanguageGenerator:
    """Generates language-specific synthetic files and import patterns."""
    
    # Language-specific file extensions and import patterns
    LANGUAGE_CONFIG = {
        'python': {
            'extensions': ['.py'],
            'import_patterns': ['import', 'from'],
            'avg_lines': 120,
            'std_lines': 50
        },
        'javascript': {
            'extensions': ['.js', '.jsx', '.ts', '.tsx'],
            'import_patterns': ['import', 'require'],
            'avg_lines': 85,
            'std_lines': 40
        },
        'java': {
            'extensions': ['.java'],
            'import_patterns': ['import'],
            'avg_lines': 150,
            'std_lines': 70
        },
        'go': {
            'extensions': ['.go'],
            'import_patterns': ['import'],
            'avg_lines': 95,
            'std_lines': 35
        }
    }
    
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        
    def generate_filename(self, language: str, directory: str, file_id: int) -> str:
        """Generate a realistic filename for the given language."""
        config = self.LANGUAGE_CONFIG.get(language, self.LANGUAGE_CONFIG['python'])
        extension = self.random.choice(config['extensions'])
        
        # Generate meaningful names based on common patterns
        name_patterns = [
            f"module_{file_id}",
            f"handler_{file_id}",
            f"service_{file_id}",
            f"util_{file_id}",
            f"model_{file_id}",
            f"controller_{file_id}",
            f"test_{file_id}",
            f"config_{file_id}"
        ]
        
        base_name = self.random.choice(name_patterns)
        return f"{directory}/{base_name}{extension}"
        
    def estimate_file_size(self, language: str, line_count: int) -> int:
        """Estimate file size in bytes based on language and line count."""
        # Average bytes per line by language
        bytes_per_line = {
            'python': 35,
            'javascript': 40,
            'java': 45,
            'go': 30
        }
        
        bpl = bytes_per_line.get(language, 35)
        return line_count * bpl + self.random.randint(-100, 500)  # Some variation
        
    def generate_import_statement(self, language: str, target_file: str) -> str:
        """Generate a realistic import statement for the target file."""
        config = self.LANGUAGE_CONFIG.get(language, self.LANGUAGE_CONFIG['python'])
        pattern = self.random.choice(config['import_patterns'])
        
        # Convert file path to import name
        if language == 'python':
            # Convert path/to/file.py -> path.to.file  
            import_name = target_file.replace('/', '.').replace('.py', '')
            if pattern == 'import':
                return f"import {import_name}"
            else:
                return f"from {import_name} import something"
                
        elif language in ['javascript', 'typescript']:
            # Use relative imports
            return f"import something from '{target_file}'"
            
        elif language == 'java':
            # Java package imports
            package_name = target_file.replace('/', '.').replace('.java', '')
            return f"import {package_name};"
            
        else:  # go
            return f"import \"{target_file}\""


class DirectoryStructureGenerator:
    """Generates realistic directory structures for synthetic repositories."""
    
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        
    def generate_directory_tree(self, target_files: int, max_depth: int) -> List[str]:
        """Generate a realistic directory tree structure."""
        directories = ["src"]  # Root directory
        
        # Common directory patterns for different scales
        if target_files < 50_000:
            # Small to medium repos
            base_dirs = ["src/main", "src/lib", "src/utils", "src/models", 
                        "src/services", "src/handlers", "src/config", "tests"]
        elif target_files < 500_000:
            # Medium repos  
            base_dirs = ["src/backend", "src/frontend", "src/shared", "src/api",
                        "src/database", "src/middleware", "src/client", "tests",
                        "docs", "scripts", "tools"]
        else:
            # Large repos
            base_dirs = ["services/auth", "services/payment", "services/user",
                        "services/notification", "libraries/common", "libraries/utils",
                        "applications/web", "applications/mobile", "applications/admin",
                        "infrastructure/database", "infrastructure/cache", 
                        "tests/integration", "tests/unit", "docs", "tools", "scripts"]
                        
        directories.extend(base_dirs)
        
        # Generate deeper nested directories
        current_depth = 2
        while current_depth < max_depth and len(directories) < target_files // 20:
            new_dirs = []
            for base_dir in directories:
                if base_dir.count('/') == current_depth - 1:
                    # Add subdirectories
                    subdirs = self._generate_subdirectories(base_dir)
                    new_dirs.extend(subdirs)
                    
            directories.extend(new_dirs[:target_files // 20 - len(directories)])
            current_depth += 1
            
        return sorted(set(directories))
        
    def _generate_subdirectories(self, parent_dir: str) -> List[str]:
        """Generate realistic subdirectories for a parent directory."""
        common_patterns = [
            "models", "views", "controllers", "handlers", "services", "utils",
            "config", "middleware", "auth", "database", "cache", "tests",
            "internal", "external", "common", "shared", "types", "interfaces"
        ]
        
        # Select 1-4 subdirectories randomly
        count = self.random.randint(1, 4)
        selected = self.random.sample(common_patterns, min(count, len(common_patterns)))
        
        return [f"{parent_dir}/{subdir}" for subdir in selected]


class DependencyGraphGenerator:
    """Generates realistic dependency graphs with power-law distributions."""
    
    def __init__(self, config: RepoConfig):
        self.config = config
        self.random = random.Random(config.seed)
        
    def generate_dependencies(self, files: List[SyntheticFile]) -> DependencyGraph:
        """Generate a realistic dependency graph for the given files."""
        graph = DependencyGraph()
        
        # Add all files as nodes
        for file in files:
            graph.add_node(file.path)
            
        # Generate dependencies with clustering and power-law properties
        self._generate_clustered_dependencies(files, graph)
        self._generate_powerlaw_dependencies(files, graph)
        
        # Ensure minimum connectivity
        self._ensure_connectivity(files, graph)
        
        logger.info(f"Generated dependency graph with {len(graph.nodes)} nodes and "
                   f"{sum(len(edges) for edges in graph.forward_edges.values())} edges")
        
        return graph
        
    def _generate_clustered_dependencies(self, files: List[SyntheticFile], graph: DependencyGraph) -> None:
        """Generate dependencies based on directory clustering."""
        # Group files by directory
        directory_groups: Dict[str, List[SyntheticFile]] = {}
        for file in files:
            directory = '/'.join(file.path.split('/')[:-1])
            if directory not in directory_groups:
                directory_groups[directory] = []
            directory_groups[directory].append(file)
            
        # Generate intra-directory dependencies
        for directory, dir_files in directory_groups.items():
            if len(dir_files) < 2:
                continue
                
            # Each file imports some others in the same directory
            for file in dir_files:
                import_count = max(1, int(self.config.avg_dependencies_per_file * self.config.clustering_factor))
                import_count = min(import_count, len(dir_files) - 1)
                
                if import_count > 0:
                    potential_imports = [f for f in dir_files if f.path != file.path and f.language == file.language]
                    if potential_imports:
                        imports = self.random.sample(potential_imports, min(import_count, len(potential_imports)))
                        
                        for imported_file in imports:
                            graph.add_edge(file.path, imported_file.path)
                            file.imports.append(imported_file.path)
                            
    def _generate_powerlaw_dependencies(self, files: List[SyntheticFile], graph: DependencyGraph) -> None:
        """Generate inter-directory dependencies following power-law distribution."""
        # Some files become "hubs" with many dependents (like utility libraries)
        num_hubs = max(1, int(len(files) * 0.05))  # 5% of files are hubs
        
        # Select hub files (prefer files in common/util directories)
        hub_candidates = [f for f in files if any(keyword in f.path.lower() 
                         for keyword in ['util', 'common', 'shared', 'lib'])]
        if len(hub_candidates) < num_hubs:
            hub_candidates.extend(self.random.sample(files, num_hubs - len(hub_candidates)))
            
        hubs = self.random.sample(hub_candidates, min(num_hubs, len(hub_candidates)))
        
        # Generate power-law dependencies to hubs
        for file in files:
            if file in hubs:
                continue
                
            # Probability of importing a hub decreases with existing dependency count
            existing_deps = len(file.imports)
            remaining_deps = max(0, int(self.config.avg_dependencies_per_file * (1.0 - self.config.clustering_factor)) - existing_deps)
            
            if remaining_deps > 0:
                # Power-law selection of hubs to import
                hub_weights = [(1.0 / (i + 1) ** self.config.powerlaw_exponent) for i in range(len(hubs))]
                selected_hubs = self.random.choices(hubs, weights=hub_weights, k=min(remaining_deps, len(hubs)))
                
                for hub in selected_hubs:
                    if hub.path != file.path and hub.language == file.language:
                        graph.add_edge(file.path, hub.path)
                        file.imports.append(hub.path)
                        
    def _ensure_connectivity(self, files: List[SyntheticFile], graph: DependencyGraph) -> None:
        """Ensure minimum connectivity for realistic dependency graph."""
        # Find files with no dependencies and add at least one
        for file in files:
            if len(file.imports) == 0:
                # Add at least one dependency to another file
                potential_deps = [f for f in files if f.path != file.path and f.language == file.language]
                if potential_deps:
                    dep = self.random.choice(potential_deps)
                    graph.add_edge(file.path, dep.path)
                    file.imports.append(dep.path)


class SyntheticRepoGenerator:
    """
    Main synthetic repository generator.
    
    Creates realistic repositories with configurable properties:
    - File count and distribution
    - Dependency patterns
    - Directory structures
    - Language mixing
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)
        
    def generate_repository(self, config: RepoConfig) -> SyntheticRepository:
        """Generate a complete synthetic repository."""
        logger.info(f"Generating synthetic repository with {config.target_files:,} files")
        
        # Initialize generators
        lang_generator = LanguageGenerator(config.seed)
        dir_generator = DirectoryStructureGenerator(config.seed)
        dep_generator = DependencyGraphGenerator(config)
        
        # Generate directory structure
        directories = dir_generator.generate_directory_tree(config.target_files, config.directory_depth)
        
        # Generate files
        files = self._generate_files(config, lang_generator, directories)
        
        # Generate dependencies
        dependency_graph = dep_generator.generate_dependencies(files)
        
        # Convert to ScanResult format
        scan_results = self._convert_to_scan_results(files)
        
        # Generate metadata
        metadata = self._generate_metadata(config, files, dependency_graph)
        
        logger.info(f"Generated repository: {len(files):,} files, "
                   f"{sum(len(edges) for edges in dependency_graph.forward_edges.values()):,} dependencies")
        
        return SyntheticRepository(
            files=files,
            dependency_graph=dependency_graph,
            scan_results=scan_results,
            metadata=metadata
        )
        
    def _generate_files(
        self, 
        config: RepoConfig, 
        lang_generator: LanguageGenerator,
        directories: List[str]
    ) -> List[SyntheticFile]:
        """Generate synthetic files with realistic properties."""
        files = []
        
        # Distribute files across languages
        language_counts = {}
        total_weight = sum(config.language_distribution.values())
        
        for lang, weight in config.language_distribution.items():
            count = int((weight / total_weight) * config.target_files)
            language_counts[lang] = count
            
        # Adjust for rounding errors
        assigned = sum(language_counts.values())
        if assigned < config.target_files:
            # Add remaining files to the most common language
            most_common = max(language_counts.keys(), key=lambda k: language_counts[k])
            language_counts[most_common] += config.target_files - assigned
            
        # Generate files for each language
        file_id = 0
        for language, count in language_counts.items():
            lang_config = LanguageGenerator.LANGUAGE_CONFIG.get(language, LanguageGenerator.LANGUAGE_CONFIG['python'])
            
            for i in range(count):
                # Select random directory
                directory = self.random.choice(directories)
                
                # Generate file
                file_path = lang_generator.generate_filename(language, directory, file_id)
                
                # Generate realistic file size
                line_count = max(10, int(self.random.gauss(lang_config['avg_lines'], lang_config['std_lines'])))
                file_size = lang_generator.estimate_file_size(language, line_count)
                
                # Create content hash
                content = f"synthetic_file_{file_id}_{language}_{line_count}_lines"
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                
                file = SyntheticFile(
                    path=file_path,
                    language=language,
                    size_bytes=file_size,
                    content_hash=content_hash,
                    directory_level=file_path.count('/')
                )
                
                files.append(file)
                file_id += 1
                
        return files
        
    def _convert_to_scan_results(self, files: List[SyntheticFile]) -> List[ScanResult]:
        """Convert synthetic files to ScanResult format for compatibility."""
        scan_results = []
        
        for file in files:
            file_stats = FileStats(
                path=file.path,
                size_bytes=file.size_bytes,
                lines=file.size_bytes // 35,  # Rough estimate
                language=file.language,
                is_readme=False,
                is_test=False, 
                is_config=False,
                is_docs=False,
                depth=file.directory_level,
                last_modified=time.time(),
                is_entrypoint=False,
                has_examples=False,
                is_integration_test=False
            )
            
            import_info = ImportAnalysis(
                imports=set(file.imports),
                import_count=len(file.imports),
                unique_modules=len(set(imp.split('.')[0] for imp in file.imports)),
                relative_imports=len([imp for imp in file.imports if imp.startswith('.')]),
                external_imports=len([imp for imp in file.imports if not imp.startswith('.')])
            )
            
            scan_result = ScanResult(
                stats=file_stats,
                imports=import_info,
                doc_analysis=None,
                churn_score=0.0,
                priority_boost=0.0,
                centrality_in=0.0,
                signatures=[]
            )
            
            scan_results.append(scan_result)
            
        return scan_results
        
    def _generate_metadata(
        self, 
        config: RepoConfig, 
        files: List[SyntheticFile],
        graph: DependencyGraph
    ) -> Dict[str, Any]:
        """Generate repository metadata for analysis."""
        # Language statistics
        lang_stats = {}
        for file in files:
            if file.language not in lang_stats:
                lang_stats[file.language] = {'count': 0, 'total_size': 0}
            lang_stats[file.language]['count'] += 1
            lang_stats[file.language]['total_size'] += file.size_bytes
            
        # Directory statistics
        dir_stats = {}
        for file in files:
            directory = '/'.join(file.path.split('/')[:-1])
            if directory not in dir_stats:
                dir_stats[directory] = 0
            dir_stats[directory] += 1
            
        # Graph statistics  
        graph_stats = graph.get_stats()
        
        return {
            'generation_config': {
                'target_files': config.target_files,
                'avg_dependencies_per_file': config.avg_dependencies_per_file,
                'clustering_factor': config.clustering_factor,
                'seed': config.seed
            },
            'actual_stats': {
                'total_files': len(files),
                'total_size_bytes': sum(f.size_bytes for f in files),
                'average_file_size': sum(f.size_bytes for f in files) / len(files) if files else 0
            },
            'language_distribution': lang_stats,
            'directory_distribution': {
                'total_directories': len(dir_stats),
                'max_files_per_directory': max(dir_stats.values()) if dir_stats else 0,
                'average_files_per_directory': sum(dir_stats.values()) / len(dir_stats) if dir_stats else 0
            },
            'graph_statistics': {
                'total_nodes': graph_stats.total_nodes,
                'total_edges': graph_stats.total_edges,
                'average_in_degree': graph_stats.in_degree_avg,
                'max_in_degree': graph_stats.in_degree_max,
                'graph_density': graph_stats.graph_density
            }
        }


def create_synthetic_repo_generator(seed: int = 42) -> SyntheticRepoGenerator:
    """Create a SyntheticRepoGenerator instance."""
    return SyntheticRepoGenerator(seed)


def get_scale_config(scale: RepoScale) -> RepoConfig:
    """Get predefined configuration for a repository scale."""
    if scale == RepoScale.SMALL:
        return RepoConfig(
            target_files=10_000,
            avg_dependencies_per_file=3.2,
            language_distribution={'python': 0.4, 'javascript': 0.3, 'java': 0.2, 'go': 0.1},
            directory_depth=5,
            clustering_factor=0.7,
            seed=42
        )
    elif scale == RepoScale.MEDIUM:
        return RepoConfig(
            target_files=100_000,
            avg_dependencies_per_file=4.1,
            language_distribution={'python': 0.35, 'javascript': 0.3, 'java': 0.2, 'go': 0.15},
            directory_depth=7,
            clustering_factor=0.65,
            seed=42
        )
    else:  # LARGE
        return RepoConfig(
            target_files=10_000_000,
            avg_dependencies_per_file=2.8,  # Large repos tend to be more modular
            language_distribution={'python': 0.3, 'javascript': 0.25, 'java': 0.25, 'go': 0.2},
            directory_depth=10,
            clustering_factor=0.8,  # More structured organization
            seed=42
        )
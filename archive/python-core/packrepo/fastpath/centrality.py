"""
PageRank Centrality Algorithm (Workstream B)

Implements PageRank centrality computation over reverse dependency edges with:
- 8-10 iterations with damping factor d=0.85
- Reverse edge emphasis (files that are imported get higher centrality)
- Flag-guarded integration with existing heuristic scoring
- Efficient sparse matrix computation

Research-grade implementation for publication standards.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .fast_scan import ScanResult
from .feature_flags import get_feature_flags


@dataclass
class CentralityScores:
    """PageRank centrality computation results."""
    pagerank_scores: Dict[str, float]
    iterations_converged: int
    convergence_epsilon: float
    graph_stats: GraphStats


@dataclass
class GraphStats:
    """Dependency graph statistics."""
    total_nodes: int
    total_edges: int
    in_degree_avg: float
    in_degree_max: int
    out_degree_avg: float
    out_degree_max: int
    strongly_connected_components: int
    graph_density: float


class DependencyGraph:
    """
    Dependency graph representation for PageRank computation.
    
    Builds graph from import relationships with emphasis on reverse edges:
    - Forward edge: A imports B (A -> B)  
    - Reverse edge: B is imported by A (B <- A)
    - PageRank flows along reverse edges (importance flows to imported files)
    """
    
    def __init__(self):
        # Forward adjacency list: file -> files it imports
        self.forward_edges: Dict[str, Set[str]] = defaultdict(set)
        
        # Reverse adjacency list: file -> files that import it  
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)
        
        # All nodes in the graph
        self.nodes: Set[str] = set()
        
    def add_edge(self, from_file: str, to_file: str):
        """Add an import edge: from_file imports to_file."""
        self.forward_edges[from_file].add(to_file)
        self.reverse_edges[to_file].add(from_file)
        self.nodes.add(from_file)
        self.nodes.add(to_file)
        
    def add_node(self, file_path: str):
        """Add a node without edges (isolated file)."""
        self.nodes.add(file_path)
        
    def get_in_degree(self, node: str) -> int:
        """Get in-degree (number of files that import this node)."""
        return len(self.reverse_edges.get(node, set()))
        
    def get_out_degree(self, node: str) -> int:
        """Get out-degree (number of files this node imports)."""
        return len(self.forward_edges.get(node, set()))
        
    def get_stats(self) -> GraphStats:
        """Calculate graph statistics."""
        if not self.nodes:
            return GraphStats(0, 0, 0.0, 0, 0.0, 0, 0, 0.0)
            
        total_nodes = len(self.nodes)
        total_edges = sum(len(edges) for edges in self.forward_edges.values())
        
        in_degrees = [self.get_in_degree(node) for node in self.nodes]
        out_degrees = [self.get_out_degree(node) for node in self.nodes]
        
        in_degree_avg = sum(in_degrees) / total_nodes if total_nodes > 0 else 0.0
        in_degree_max = max(in_degrees) if in_degrees else 0
        out_degree_avg = sum(out_degrees) / total_nodes if total_nodes > 0 else 0.0
        out_degree_max = max(out_degrees) if out_degrees else 0
        
        # Approximate SCC count (simplified - full Tarjan would be more accurate)
        scc_count = self._estimate_scc_count()
        
        # Graph density: actual_edges / possible_edges
        max_possible_edges = total_nodes * (total_nodes - 1)
        graph_density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        
        return GraphStats(
            total_nodes=total_nodes,
            total_edges=total_edges,
            in_degree_avg=in_degree_avg,
            in_degree_max=in_degree_max,
            out_degree_avg=out_degree_avg,
            out_degree_max=out_degree_max,
            strongly_connected_components=scc_count,
            graph_density=graph_density
        )
        
    def _estimate_scc_count(self) -> int:
        """Estimate strongly connected component count (simplified approximation)."""
        # For PageRank, we mainly care about graph connectivity
        # This is a rough estimate - full Tarjan's algorithm would be more precise
        
        if not self.nodes:
            return 0
            
        # Count nodes with both in and out edges (likely in cycles)
        potential_scc_nodes = 0
        for node in self.nodes:
            if self.get_in_degree(node) > 0 and self.get_out_degree(node) > 0:
                potential_scc_nodes += 1
                
        # Rough estimate: most SCCs are small, assume average size of 3
        estimated_scc = max(1, potential_scc_nodes // 3)
        
        # Add isolated nodes and simple chains
        isolated_nodes = len(self.nodes) - potential_scc_nodes
        return estimated_scc + isolated_nodes


class PageRankComputer:
    """
    PageRank computation engine optimized for code dependency analysis.
    
    Implements the classic PageRank algorithm with modifications for code:
    - Emphasis on reverse edges (importance flows to imported files)
    - Damping factor d=0.85 (research standard)
    - Convergence detection with configurable epsilon
    - Efficient sparse computation for large codebases
    """
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 10, epsilon: float = 1e-6):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        
    def compute_pagerank(self, graph: DependencyGraph) -> CentralityScores:
        """
        Compute PageRank scores over the reverse dependency graph.
        
        Algorithm:
        1. Initialize all nodes with equal probability (1/N)
        2. Iteratively update scores using PageRank formula
        3. Flow importance along reverse edges (to imported files)
        4. Apply damping factor and teleportation probability
        5. Converge when score changes are below epsilon
        """
        if not graph.nodes:
            return CentralityScores({}, 0, self.epsilon, graph.get_stats())
            
        # Initialize PageRank scores
        num_nodes = len(graph.nodes)
        initial_score = 1.0 / num_nodes
        
        # Current and previous scores
        current_scores = {node: initial_score for node in graph.nodes}
        previous_scores = current_scores.copy()
        
        # PageRank iteration
        for iteration in range(self.max_iterations):
            # Update scores for each node
            for node in graph.nodes:
                # Base teleportation probability
                new_score = (1.0 - self.damping_factor) / num_nodes
                
                # Sum contributions from nodes that link to this node
                # In reverse graph: sum from nodes that this node imports
                link_contribution = 0.0
                for linking_node in graph.reverse_edges.get(node, set()):
                    # Contribution = (linking_node_score) / (linking_node_out_degree)
                    linking_out_degree = max(1, len(graph.forward_edges.get(linking_node, set())))
                    link_contribution += previous_scores[linking_node] / linking_out_degree
                    
                new_score += self.damping_factor * link_contribution
                current_scores[node] = new_score
                
            # Check for convergence
            if self._has_converged(current_scores, previous_scores):
                return CentralityScores(
                    pagerank_scores=current_scores,
                    iterations_converged=iteration + 1,
                    convergence_epsilon=self.epsilon,
                    graph_stats=graph.get_stats()
                )
                
            # Prepare for next iteration
            previous_scores = current_scores.copy()
            
        # Return scores after max iterations
        return CentralityScores(
            pagerank_scores=current_scores,
            iterations_converged=self.max_iterations,
            convergence_epsilon=self.epsilon,
            graph_stats=graph.get_stats()
        )
        
    def _has_converged(self, current: Dict[str, float], previous: Dict[str, float]) -> bool:
        """Check if PageRank scores have converged."""
        if not current or not previous:
            return False
            
        # Calculate L1 norm of the difference
        total_diff = sum(abs(current[node] - previous.get(node, 0.0)) for node in current.keys())
        
        return total_diff < self.epsilon


class ImportDetector:
    """
    Detects import relationships from file scan results.
    
    Analyzes various import patterns across languages:
    - Python: import, from...import
    - JavaScript/TypeScript: import, require()
    - Go: import
    - Rust: use, mod
    - Java: import
    """
    
    # Language-specific import patterns
    IMPORT_PATTERNS = {
        'python': ['import ', 'from '],
        'javascript': ['import ', 'require(', 'from '],
        'typescript': ['import ', 'require(', 'from '],
        'go': ['import '],
        'rust': ['use ', 'mod '],
        'java': ['import '],
        'cpp': ['#include'],
        'c': ['#include']
    }
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        ext_mapping = {
            'py': 'python',
            'js': 'javascript', 'jsx': 'javascript', 'mjs': 'javascript',
            'ts': 'typescript', 'tsx': 'typescript',
            'go': 'go',
            'rs': 'rust',
            'java': 'java', 'kotlin': 'java',
            'cpp': 'cpp', 'cc': 'cpp', 'cxx': 'cpp', 'hpp': 'cpp',
            'c': 'c', 'h': 'c'
        }
        
        return ext_mapping.get(ext)
        
    def extract_imports(self, scan_result: ScanResult, all_scan_results: List[ScanResult]) -> List[str]:
        """Extract import relationships for a file."""
        if not scan_result.imports or not scan_result.imports.imports:
            return []
            
        file_path = scan_result.stats.path
        language = self.detect_language(file_path)
        
        if not language or language not in self.IMPORT_PATTERNS:
            return scan_result.imports.imports  # Return raw imports
            
        # Map import strings to actual file paths in the scan results
        resolved_imports = []
        file_path_map = {result.stats.path: result for result in all_scan_results}
        
        for import_str in scan_result.imports.imports:
            resolved_path = self._resolve_import_path(import_str, file_path, file_path_map, language)
            if resolved_path:
                resolved_imports.append(resolved_path)
                
        return resolved_imports
        
    def _resolve_import_path(
        self, 
        import_str: str, 
        current_file: str, 
        file_map: Dict[str, ScanResult],
        language: str
    ) -> Optional[str]:
        """Resolve import string to actual file path."""
        # This is a simplified resolver - production version would be more sophisticated
        
        # Clean import string
        cleaned_import = import_str.strip().lower()
        
        # Language-specific resolution logic
        if language == 'python':
            return self._resolve_python_import(cleaned_import, current_file, file_map)
        elif language in ['javascript', 'typescript']:
            return self._resolve_js_import(cleaned_import, current_file, file_map)
        else:
            # Generic resolution for other languages
            return self._resolve_generic_import(cleaned_import, current_file, file_map)
            
    def _resolve_python_import(self, import_str: str, current_file: str, file_map: Dict[str, ScanResult]) -> Optional[str]:
        """Resolve Python import to file path."""
        # Convert module path to file path
        module_parts = import_str.replace('.', '/').split('/')
        
        # Try different combinations
        potential_paths = []
        
        # Direct module file
        potential_paths.append('/'.join(module_parts) + '.py')
        
        # Module package
        potential_paths.append('/'.join(module_parts) + '/__init__.py')
        
        # Relative to current file directory
        current_dir = '/'.join(current_file.split('/')[:-1])
        for path in potential_paths[:]:
            potential_paths.append(current_dir + '/' + path)
            
        # Find matching file
        for path in potential_paths:
            if path in file_map:
                return path
                
        # Partial matching fallback
        for file_path in file_map.keys():
            if any(part in file_path.lower() for part in module_parts):
                return file_path
                
        return None
        
    def _resolve_js_import(self, import_str: str, current_file: str, file_map: Dict[str, ScanResult]) -> Optional[str]:
        """Resolve JavaScript/TypeScript import to file path."""
        # Handle relative imports
        if import_str.startswith('./') or import_str.startswith('../'):
            # Relative import resolution
            current_dir = '/'.join(current_file.split('/')[:-1])
            resolved_path = self._resolve_relative_path(current_dir, import_str)
            
            # Try different extensions
            for ext in ['.js', '.ts', '.jsx', '.tsx', '/index.js', '/index.ts']:
                candidate = resolved_path + ext
                if candidate in file_map:
                    return candidate
                    
        # Absolute import (simplified)
        else:
            import_parts = import_str.split('/')
            for file_path in file_map.keys():
                if any(part in file_path.lower() for part in import_parts):
                    return file_path
                    
        return None
        
    def _resolve_generic_import(self, import_str: str, current_file: str, file_map: Dict[str, ScanResult]) -> Optional[str]:
        """Generic import resolution for other languages."""
        # Simple substring matching
        import_parts = import_str.split('/')[-1].split('.')[-1]  # Get last part
        
        for file_path in file_map.keys():
            file_name = file_path.split('/')[-1].split('.')[0]
            if file_name.lower() == import_parts.lower():
                return file_path
                
        return None
        
    def _resolve_relative_path(self, base_dir: str, relative_path: str) -> str:
        """Resolve relative path from base directory."""
        # Simplified relative path resolution
        if relative_path.startswith('./'):
            return base_dir + '/' + relative_path[2:]
        elif relative_path.startswith('../'):
            parent_dir = '/'.join(base_dir.split('/')[:-1])
            return parent_dir + '/' + relative_path[3:]
        else:
            return base_dir + '/' + relative_path


class CentralityCalculator:
    """
    Main interface for PageRank centrality calculation.
    
    Integrates graph construction, PageRank computation, and score integration
    with the existing FastPath heuristic system.
    """
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 10):
        self.import_detector = ImportDetector()
        self.pagerank_computer = PageRankComputer(damping_factor, max_iterations)
        
    def build_dependency_graph(self, scan_results: List[ScanResult]) -> DependencyGraph:
        """Build dependency graph from scan results."""
        graph = DependencyGraph()
        
        # Add all files as nodes
        for result in scan_results:
            graph.add_node(result.stats.path)
            
        # Add import edges
        for result in scan_results:
            imports = self.import_detector.extract_imports(result, scan_results)
            for imported_file in imports:
                graph.add_edge(result.stats.path, imported_file)
                
        return graph
        
    def calculate_centrality_scores(self, scan_results: List[ScanResult]) -> CentralityScores:
        """Calculate PageRank centrality scores for all files."""
        flags = get_feature_flags()
        
        # Only compute if centrality is enabled
        if not flags.centrality_enabled:
            # Return empty scores
            empty_graph = DependencyGraph()
            return CentralityScores({}, 0, 0.0, empty_graph.get_stats())
            
        # Build dependency graph
        graph = self.build_dependency_graph(scan_results)
        
        # Compute PageRank
        return self.pagerank_computer.compute_pagerank(graph)
        
    def integrate_with_heuristics(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float],
        centrality_weight: float = 0.15
    ) -> Dict[str, float]:
        """
        Integrate centrality scores with existing heuristic scores.
        
        Combined score = (1-centrality_weight) * heuristic + centrality_weight * centrality
        """
        flags = get_feature_flags()
        
        if not flags.centrality_enabled:
            return heuristic_scores
            
        # Calculate centrality scores
        centrality_results = self.calculate_centrality_scores(scan_results)
        centrality_scores = centrality_results.pagerank_scores
        
        # Normalize centrality scores to same range as heuristic scores
        if centrality_scores:
            max_centrality = max(centrality_scores.values())
            max_heuristic = max(heuristic_scores.values()) if heuristic_scores else 1.0
            
            if max_centrality > 0:
                normalization_factor = max_heuristic / max_centrality
                centrality_scores = {
                    path: score * normalization_factor 
                    for path, score in centrality_scores.items()
                }
        
        # Combine scores
        combined_scores = {}
        for result in scan_results:
            path = result.stats.path
            heuristic_score = heuristic_scores.get(path, 0.0)
            centrality_score = centrality_scores.get(path, 0.0)
            
            combined_score = (
                (1.0 - centrality_weight) * heuristic_score + 
                centrality_weight * centrality_score
            )
            combined_scores[path] = combined_score
            
        return combined_scores


def create_centrality_calculator(damping_factor: float = 0.85, max_iterations: int = 10) -> CentralityCalculator:
    """Create a CentralityCalculator instance."""
    return CentralityCalculator(damping_factor, max_iterations)
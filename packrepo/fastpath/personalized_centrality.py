"""
Personalized PageRank for Entry Point Relevance

Extends the standard PageRank centrality system to support entry point bias,
allowing users to specify files/functions that are relevant to their problem
and biasing the importance calculation towards code that's connected to these
entry points.

This enables more targeted repository packing where files are selected based
on their relevance to a specific problem or area of interest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .centrality import CentralityScores, DependencyGraph, PageRankComputer
from .fast_scan import ScanResult
from .base_types import EntryPointSpec
from .utils.entry_points import EntryPointConverter
from .utils.file_patterns import FilePatternMatcher
from .utils.error_handling import ErrorHandler

# For backward compatibility
EntryPoint = EntryPointSpec


@dataclass
class PersonalizedCentralityConfig:
    """Configuration for personalized centrality calculation."""
    entry_points: List[EntryPointSpec]
    personalization_alpha: float = 0.15  # Higher = more bias towards entry points
    damping_factor: float = 0.85
    max_iterations: int = 15
    epsilon: float = 1e-6
    enable_semantic_expansion: bool = True  # Expand to semantically related files
    semantic_similarity_threshold: float = 0.7


class PersonalizedPageRankComputer(PageRankComputer):
    """
    Enhanced PageRank computation with personalized teleportation.
    
    Instead of uniform teleportation probability, biases the random walk
    to restart at specified entry points, making files connected to these
    entry points more central to the computation.
    """
    
    def __init__(self, config: PersonalizedCentralityConfig):
        super().__init__(
            damping_factor=config.damping_factor,
            max_iterations=config.max_iterations,
            epsilon=config.epsilon
        )
        self.config = config
        self.personalization_vector = {}
    
    def compute_personalized_pagerank(
        self,
        graph: DependencyGraph,
        scan_results: List[ScanResult]
    ) -> CentralityScores:
        """
        Compute personalized PageRank with entry point bias.
        
        Args:
            graph: Dependency graph built from scan results
            scan_results: List of scanned files for context
            
        Returns:
            CentralityScores with personalized importance values
        """
        if not graph.nodes:
            return CentralityScores({}, 0, self.epsilon, graph.get_stats())
        
        # Build personalization vector
        self._build_personalization_vector(graph, scan_results)
        
        # Initialize PageRank scores
        num_nodes = len(graph.nodes)
        initial_score = 1.0 / num_nodes
        
        current_scores = {node: initial_score for node in graph.nodes}
        previous_scores = current_scores.copy()
        
        # Personalized PageRank iteration
        for iteration in range(self.max_iterations):
            for node in graph.nodes:
                # Personalized teleportation (biased restart probability)
                teleport_prob = self.personalization_vector.get(node, 0.0)
                new_score = (1.0 - self.damping_factor) * teleport_prob
                
                # Link contribution from incoming edges
                link_contribution = 0.0
                for linking_node in graph.reverse_edges.get(node, set()):
                    linking_out_degree = max(1, len(graph.forward_edges.get(linking_node, set())))
                    link_contribution += previous_scores[linking_node] / linking_out_degree
                
                new_score += self.damping_factor * link_contribution
                current_scores[node] = new_score
            
            # Check convergence
            if self._has_converged(current_scores, previous_scores):
                return CentralityScores(
                    pagerank_scores=current_scores,
                    iterations_converged=iteration + 1,
                    convergence_epsilon=self.epsilon,
                    graph_stats=graph.get_stats()
                )
            
            previous_scores = current_scores.copy()
        
        # Return even if didn't converge
        return CentralityScores(
            pagerank_scores=current_scores,
            iterations_converged=self.max_iterations,
            convergence_epsilon=self.epsilon,
            graph_stats=graph.get_stats()
        )
    
    def _build_personalization_vector(
        self, 
        graph: DependencyGraph, 
        scan_results: List[ScanResult]
    ):
        """Build the personalization vector for biased teleportation."""
        # Create mapping from file paths to scan results
        scan_map = {result.stats.path: result for result in scan_results}
        
        # Initialize with uniform low probability
        base_prob = (1.0 - self.config.personalization_alpha) / len(graph.nodes)
        self.personalization_vector = {node: base_prob for node in graph.nodes}
        
        # Calculate total weight of entry points
        total_entry_weight = sum(ep.weight for ep in self.config.entry_points)
        if total_entry_weight == 0:
            return
        
        # Distribute personalization probability among entry points
        entry_prob_mass = self.config.personalization_alpha
        
        for entry_point in self.config.entry_points:
            # Find matching nodes for this entry point
            matching_nodes = self._find_matching_nodes(entry_point, graph, scan_map)
            
            if matching_nodes:
                # Distribute this entry point's probability among matching nodes
                prob_per_node = (entry_prob_mass * entry_point.weight / total_entry_weight) / len(matching_nodes)
                for node in matching_nodes:
                    self.personalization_vector[node] += prob_per_node
        
        # Normalize to ensure probabilities sum to 1
        total_prob = sum(self.personalization_vector.values())
        if total_prob > 0:
            for node in self.personalization_vector:
                self.personalization_vector[node] /= total_prob
    
    def _find_matching_nodes(
        self, 
        entry_point: EntryPointSpec, 
        graph: DependencyGraph,
        scan_map: Dict[str, ScanResult]
    ) -> Set[str]:
        """Find graph nodes that match the given entry point."""
        matching_nodes = set()
        
        # Direct file path match
        for node in graph.nodes:
            if self._file_matches(node, entry_point.file_path):
                # If no specific function/class specified, include the whole file
                if not entry_point.function_name and not entry_point.class_name:
                    matching_nodes.add(node)
                else:
                    # Check if the file contains the specified function/class
                    if self._contains_symbol(node, entry_point, scan_map):
                        matching_nodes.add(node)
        
        # Semantic expansion if enabled
        if self.config.enable_semantic_expansion and matching_nodes:
            semantic_matches = self._find_semantic_matches(matching_nodes, graph, scan_map)
            matching_nodes.update(semantic_matches)
        
        return matching_nodes
    
    def _file_matches(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches the entry point pattern."""
        matcher = FilePatternMatcher()
        return matcher.matches(file_path, pattern)
    
    def _contains_symbol(
        self, 
        file_path: str, 
        entry_point: EntryPointSpec, 
        scan_map: Dict[str, ScanResult]
    ) -> bool:
        """Check if file contains the specified function or class."""
        scan_result = scan_map.get(file_path)
        if not scan_result:
            return False
        
        # Use the file pattern matcher for symbol detection  
        matcher = FilePatternMatcher()
        return matcher.find_symbol_in_file(
            file_path, 
            function_name=entry_point.function_name,
            class_name=entry_point.class_name
        )
    
    def _find_semantic_matches(
        self, 
        seed_nodes: Set[str], 
        graph: DependencyGraph,
        scan_map: Dict[str, ScanResult]
    ) -> Set[str]:
        """Find semantically related files using graph proximity and content similarity."""
        semantic_matches = set()
        
        # Graph-based expansion: include direct dependencies
        for seed_node in seed_nodes:
            # Add files that import the seed (reverse dependencies)
            semantic_matches.update(graph.reverse_edges.get(seed_node, set()))
            
            # Add files that are imported by the seed (forward dependencies)
            semantic_matches.update(graph.forward_edges.get(seed_node, set()))
        
        # TODO: Content-based semantic similarity could be added here
        # This would require embedding computation and similarity comparison
        
        return semantic_matches


class PersonalizedCentralityCalculator:
    """
    Main interface for personalized centrality calculation.
    
    Extends the standard centrality system to support entry point bias,
    enabling more targeted file selection based on problem relevance.
    """
    
    def __init__(self, config: PersonalizedCentralityConfig):
        self.config = config
        self.pagerank_computer = PersonalizedPageRankComputer(config)
    
    def calculate_personalized_centrality(
        self, 
        scan_results: List[ScanResult]
    ) -> CentralityScores:
        """
        Calculate personalized centrality scores for the given files.
        
        Args:
            scan_results: List of scanned repository files
            
        Returns:
            CentralityScores with personalized importance values
        """
        # Build dependency graph (reuse existing logic)
        from .centrality import CentralityCalculator
        
        # Use existing graph building logic
        standard_calculator = CentralityCalculator()
        graph = standard_calculator.build_dependency_graph(scan_results)
        
        # Compute personalized PageRank
        return self.pagerank_computer.compute_personalized_pagerank(graph, scan_results)
    
    @classmethod
    def create_from_entry_points(
        cls,
        entry_points: List[Union[str, EntryPointSpec]],
        personalization_alpha: float = 0.15,
        **kwargs
    ) -> 'PersonalizedCentralityCalculator':
        """
        Convenient factory method to create calculator from entry point specifications.
        
        Args:
            entry_points: List of file paths or EntryPointSpec objects
            personalization_alpha: Strength of bias towards entry points
            **kwargs: Additional configuration options
            
        Returns:
            Configured PersonalizedCentralityCalculator
        """
        # Use EntryPointConverter to normalize entry points
        processed_entry_points = EntryPointConverter.normalize_entry_points(entry_points)
        
        config = PersonalizedCentralityConfig(
            entry_points=processed_entry_points,
            personalization_alpha=personalization_alpha,
            **kwargs
        )
        
        return cls(config)


# Convenience functions for easy integration

def create_personalized_calculator(
    entry_points: List[Union[str, EntryPointSpec]],
    personalization_alpha: float = 0.15,
    **kwargs
) -> PersonalizedCentralityCalculator:
    """Create a personalized centrality calculator with the given entry points."""
    return PersonalizedCentralityCalculator.create_from_entry_points(
        entry_points=entry_points,
        personalization_alpha=personalization_alpha,
        **kwargs
    )


def create_entry_point(
    file_path: str,
    function_name: Optional[str] = None,
    class_name: Optional[str] = None,
    weight: float = 1.0,
    description: Optional[str] = None
) -> EntryPointSpec:
    """Create an EntryPointSpec object with the given parameters."""
    return EntryPointSpec(
        file_path=file_path,
        function_name=function_name,
        class_name=class_name,
        weight=weight,
        description=description
    )
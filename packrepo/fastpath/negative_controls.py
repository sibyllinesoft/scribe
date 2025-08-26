"""
Negative Controls for FastPath Evaluation

Implements validation controls that should show no improvement or degradation:
- Graph Scramble: Random permutation of dependency graph edges  
- Edge Direction Flip: Reverse all import dependencies
- Random Quota Allocation: Random budget allocation within categories

These controls validate that observed improvements are due to algorithmic changes
rather than evaluation artifacts or statistical flukes.
"""

import random
import logging
from typing import Dict, List, Set, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class GraphScrambleControl:
    """Randomly scramble dependency graph edges while preserving structure."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        
    def scramble_adjacency_matrix(self, adjacency: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Scramble graph edges while preserving node degrees."""
        random.seed(self.seed)
        
        # Extract all edges
        edges = []
        for source, targets in adjacency.items():
            for target in targets:
                edges.append((source, target))
                
        # Extract all nodes
        nodes = set(adjacency.keys())
        for targets in adjacency.values():
            nodes.update(targets)
        nodes = list(nodes)
        
        # Randomly reassign edges to preserve degree distribution
        random.shuffle(edges)
        scrambled_edges = []
        
        for i, (source, target) in enumerate(edges):
            # Assign to random nodes
            new_source = random.choice(nodes)
            new_target = random.choice(nodes)
            
            # Avoid self-loops
            while new_source == new_target:
                new_target = random.choice(nodes)
                
            scrambled_edges.append((new_source, new_target))
            
        # Rebuild adjacency matrix
        scrambled_adjacency = {node: set() for node in nodes}
        for source, target in scrambled_edges:
            scrambled_adjacency[source].add(target)
            
        logger.info(f"Scrambled {len(edges)} edges across {len(nodes)} nodes")
        return scrambled_adjacency
        
    def scramble_centrality_scores(self, centrality: Dict[str, float]) -> Dict[str, float]:
        """Randomly permute centrality scores across nodes."""
        random.seed(self.seed)
        
        nodes = list(centrality.keys())
        scores = list(centrality.values())
        
        # Shuffle scores
        random.shuffle(scores)
        
        # Reassign to nodes
        scrambled_centrality = dict(zip(nodes, scores))
        
        logger.info(f"Scrambled centrality scores for {len(nodes)} nodes")
        return scrambled_centrality


class EdgeDirectionFlipControl:
    """Reverse all dependency graph edges (imports become exporters)."""
    
    def flip_adjacency_matrix(self, adjacency: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Flip all edge directions in the dependency graph."""
        
        # Build reverse adjacency matrix
        flipped_adjacency = {}
        
        # Initialize all nodes
        all_nodes = set(adjacency.keys())
        for targets in adjacency.values():
            all_nodes.update(targets)
            
        for node in all_nodes:
            flipped_adjacency[node] = set()
            
        # Flip edges
        for source, targets in adjacency.items():
            for target in targets:
                # Original: source -> target
                # Flipped: target -> source  
                flipped_adjacency[target].add(source)
                
        logger.info(f"Flipped edges for {len(all_nodes)} nodes")
        return flipped_adjacency
        
    def flip_centrality_interpretation(self, centrality: Dict[str, float]) -> Dict[str, float]:
        """Invert centrality scores (high becomes low, low becomes high)."""
        
        if not centrality:
            return centrality
            
        max_centrality = max(centrality.values())
        min_centrality = min(centrality.values())
        
        # Invert scores linearly
        flipped_centrality = {}
        for node, score in centrality.items():
            flipped_score = max_centrality + min_centrality - score
            flipped_centrality[node] = flipped_score
            
        logger.info(f"Flipped centrality interpretation for {len(centrality)} nodes")
        return flipped_centrality


class RandomQuotaControl:
    """Randomly allocate token budget across file categories."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        
    def randomize_quota_allocation(self, 
                                 total_budget: int,
                                 categories: List[str]) -> Dict[str, int]:
        """Randomly allocate budget across categories."""
        random.seed(self.seed)
        
        # Generate random proportions that sum to 1
        random_weights = [random.random() for _ in categories]
        total_weight = sum(random_weights)
        proportions = [w / total_weight for w in random_weights]
        
        # Allocate budget
        quota_allocation = {}
        remaining_budget = total_budget
        
        for i, category in enumerate(categories[:-1]):
            # Allocate proportion of budget
            allocation = int(proportions[i] * total_budget)
            quota_allocation[category] = allocation
            remaining_budget -= allocation
            
        # Give remainder to last category
        quota_allocation[categories[-1]] = remaining_budget
        
        logger.info(f"Random quota allocation: {quota_allocation}")
        return quota_allocation
        
    def randomize_priority_scores(self, 
                                file_features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Randomly shuffle priority scores while keeping other features."""
        random.seed(self.seed)
        
        # Extract all priority-related features
        priority_features = ['centrality_in', 'is_entrypoint', 'is_config', 
                           'has_examples', 'api_surface_score']
        
        files = list(file_features.keys())
        randomized_features = {}
        
        for file_path in files:
            randomized_features[file_path] = file_features[file_path].copy()
            
        # For each priority feature, randomly permute values across files
        for feature in priority_features:
            if any(feature in features for features in file_features.values()):
                # Collect all values for this feature
                values = []
                for file_path in files:
                    if feature in file_features[file_path]:
                        values.append(file_features[file_path][feature])
                    else:
                        values.append(0.0)  # Default value
                        
                # Shuffle values
                random.shuffle(values)
                
                # Reassign to files
                for file_path, value in zip(files, values):
                    randomized_features[file_path][feature] = value
                    
        logger.info(f"Randomized priority scores for {len(files)} files")
        return randomized_features


class NegativeControlManager:
    """Coordinate negative controls for FastPath evaluation."""
    
    def __init__(self, control_type: str, seed: int = 42):
        self.control_type = control_type
        self.seed = seed
        
        if control_type == 'scramble':
            self.controller = GraphScrambleControl(seed)
        elif control_type == 'flip':
            self.controller = EdgeDirectionFlipControl()
        elif control_type == 'random_quota':
            self.controller = RandomQuotaControl(seed)
        else:
            raise ValueError(f"Unknown control type: {control_type}")
            
        logger.info(f"Initialized negative control: {control_type}")
        
    def apply_control(self, 
                     adjacency: Dict[str, Set[str]] = None,
                     centrality: Dict[str, float] = None,
                     file_features: Dict[str, Dict[str, Any]] = None,
                     total_budget: int = None,
                     categories: List[str] = None) -> Dict[str, Any]:
        """Apply the negative control and return modified data structures."""
        
        result = {}
        
        if self.control_type == 'scramble':
            if adjacency is not None:
                result['adjacency'] = self.controller.scramble_adjacency_matrix(adjacency)
            if centrality is not None:
                result['centrality'] = self.controller.scramble_centrality_scores(centrality)
                
        elif self.control_type == 'flip':
            if adjacency is not None:
                result['adjacency'] = self.controller.flip_adjacency_matrix(adjacency)
            if centrality is not None:
                result['centrality'] = self.controller.flip_centrality_interpretation(centrality)
                
        elif self.control_type == 'random_quota':
            if total_budget is not None and categories is not None:
                result['quota_allocation'] = self.controller.randomize_quota_allocation(
                    total_budget, categories)
            if file_features is not None:
                result['file_features'] = self.controller.randomize_priority_scores(file_features)
                
        logger.info(f"Applied {self.control_type} control")
        return result
        
    def validate_control_effect(self, 
                              baseline_score: float,
                              control_score: float,
                              tolerance: float = 0.05) -> bool:
        """Validate that control shows expected neutral/negative effect."""
        
        improvement = (control_score - baseline_score) / baseline_score
        
        if self.control_type in ['scramble', 'random_quota']:
            # These should show approximately zero improvement
            expected_range = (-tolerance, tolerance)
            valid = expected_range[0] <= improvement <= expected_range[1]
            
        elif self.control_type == 'flip':
            # This should show zero or negative improvement
            valid = improvement <= tolerance
            
        else:
            valid = False
            
        logger.info(f"Control validation ({self.control_type}): "
                   f"improvement={improvement:.3f}, valid={valid}")
        
        return valid
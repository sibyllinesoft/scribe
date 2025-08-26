"""
Heuristic scoring system for FastPath file prioritization.

Implements the core scoring formula:
score = w_doc*doc + w_readme*readme + w_imp*imp_deg + w_path*path_depth^-1 + w_test*test_link + w_churn*churn

Uses only fast, rule-based features without embeddings or heavy analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .fast_scan import ScanResult
from .feature_flags import get_feature_flags


@dataclass
class ScoreComponents:
    """Individual components of the heuristic score."""
    doc_score: float
    readme_score: float
    import_score: float
    path_score: float
    test_link_score: float
    churn_score: float
    final_score: float
    # V2 features (only populated when enabled)
    centrality_score: float = 0.0
    entrypoint_score: float = 0.0
    examples_score: float = 0.0


@dataclass
class HeuristicWeights:
    """Weights for different scoring components."""
    doc: float = 0.3
    readme: float = 0.25
    import_deg: float = 0.15
    path: float = 0.1
    test_link: float = 0.1
    churn: float = 0.1
    # V2 feature weights (only used when enabled)
    centrality: float = 0.0  # PageRank centrality weight
    entrypoint: float = 0.0  # Entrypoint detection weight  
    examples: float = 0.0    # Examples/usage weight
    
    def __post_init__(self):
        # Check if V2 features are enabled and adjust weights accordingly
        flags = get_feature_flags()
        
        if flags.centrality_enabled:
            # Redistribute weights when centrality features are enabled
            self.centrality = 0.15
            self.entrypoint = 0.10
            self.examples = 0.05
            # Reduce other weights proportionally
            reduction_factor = 0.7  # 30% reduction to make room for new features
            self.doc *= reduction_factor
            self.readme *= reduction_factor
            self.import_deg *= reduction_factor
            self.path *= reduction_factor
            self.test_link *= reduction_factor
            self.churn *= reduction_factor
        
        # Normalize weights to sum to 1.0
        total = (self.doc + self.readme + self.import_deg + self.path + 
                self.test_link + self.churn + self.centrality + 
                self.entrypoint + self.examples)
        if total > 0:
            self.doc /= total
            self.readme /= total
            self.import_deg /= total
            self.path /= total
            self.test_link /= total
            self.churn /= total
            self.centrality /= total
            self.entrypoint /= total
            self.examples /= total


class HeuristicScorer:
    """
    Fast heuristic-based file scoring system.
    
    Computes importance scores using only lightweight features:
    - Document type and structure analysis
    - README detection and prioritization
    - Import degree approximation via regex
    - Path depth penalty
    - Test-code relationship detection
    - Change recency (churn) signals
    """
    
    def __init__(self, weights: Optional[HeuristicWeights] = None):
        self.weights = weights or HeuristicWeights()
        
        # Cache for import degree calculations
        self._import_graph: Optional[Dict[str, Set[str]]] = None
        self._degree_cache: Dict[str, float] = {}
        
    def _build_import_graph(self, scan_results: List[ScanResult]) -> Dict[str, Set[str]]:
        """Build lightweight import graph from scan results."""
        graph = {}
        
        for result in scan_results:
            path = result.stats.path
            graph[path] = set()
            
            if result.imports and result.imports.imports:
                # Map import strings to actual file paths (simplified)
                for import_name in result.imports.imports:
                    # Try to find corresponding files
                    for other_result in scan_results:
                        other_path = other_result.stats.path
                        if self._matches_import(import_name, other_path):
                            graph[path].add(other_path)
                            
        return graph
        
    def _matches_import(self, import_name: str, file_path: str) -> bool:
        """Check if an import name matches a file path (heuristic)."""
        # Very simplified import resolution
        import_parts = import_name.lower().replace('.', '/').replace('::', '/')
        path_parts = file_path.lower().replace('\\', '/')
        
        # Direct name match
        if import_parts in path_parts:
            return True
            
        # Module name match
        if import_parts.split('/')[-1] in path_parts:
            return True
            
        return False
        
    def _calculate_doc_score(self, result: ScanResult) -> float:
        """Calculate documentation importance score."""
        score = 0.0
        
        # Base documentation file bonus
        if result.stats.is_docs:
            score += 1.0
            
        # README gets maximum documentation score
        if result.stats.is_readme:
            score += 2.0
            
        # Well-structured documents get bonus
        if result.doc_analysis:
            doc = result.doc_analysis
            
            # Heading structure bonus
            if doc.heading_count > 0:
                score += min(doc.heading_count / 10.0, 0.5)
                
            # TOC indicates well-organized document
            if doc.toc_indicators > 0:
                score += 0.3
                
            # Links indicate reference document
            if doc.link_count > 0:
                score += min(doc.link_count / 20.0, 0.3)
                
            # Code blocks in docs indicate technical documentation
            if doc.code_block_count > 0:
                score += min(doc.code_block_count / 10.0, 0.2)
                
        return min(score, 3.0)  # Cap at 3.0
    
    def _calculate_centrality_score(self, result: ScanResult) -> float:
        """Calculate PageRank centrality score (V2 feature)."""
        flags = get_feature_flags()
        if not flags.centrality_enabled:
            return 0.0
        
        # Use centrality_in score from adjacency graph analysis
        return result.centrality_in
    
    def _calculate_entrypoint_score(self, result: ScanResult) -> float:
        """Calculate entrypoint importance score (V2 feature)."""
        flags = get_feature_flags()
        if not flags.centrality_enabled:
            return 0.0
        
        # Entrypoint files get significant boost
        if result.stats.is_entrypoint:
            return 2.5
        
        return 0.0
    
    def _calculate_examples_score(self, result: ScanResult) -> float:
        """Calculate examples/usage score (V2 feature).""" 
        flags = get_feature_flags()
        if not flags.centrality_enabled:
            return 0.0
        
        # Files with examples get moderate boost
        if result.stats.has_examples:
            return 1.0
        
        return 0.0
    
    def _get_effective_weights(self):
        """Get effective weights based on current feature flag state."""
        flags = get_feature_flags()
        
        # Use V1 weights if V2 features disabled
        if not flags.centrality_enabled:
            return {
                'doc': self.weights.doc / (self.weights.doc + self.weights.readme + self.weights.import_deg + 
                       self.weights.path + self.weights.test_link + self.weights.churn),
                'readme': self.weights.readme / (self.weights.doc + self.weights.readme + self.weights.import_deg + 
                         self.weights.path + self.weights.test_link + self.weights.churn),
                'import_deg': self.weights.import_deg / (self.weights.doc + self.weights.readme + self.weights.import_deg + 
                             self.weights.path + self.weights.test_link + self.weights.churn),
                'path': self.weights.path / (self.weights.doc + self.weights.readme + self.weights.import_deg + 
                       self.weights.path + self.weights.test_link + self.weights.churn),
                'test_link': self.weights.test_link / (self.weights.doc + self.weights.readme + self.weights.import_deg + 
                            self.weights.path + self.weights.test_link + self.weights.churn),
                'churn': self.weights.churn / (self.weights.doc + self.weights.readme + self.weights.import_deg + 
                        self.weights.path + self.weights.test_link + self.weights.churn),
                'centrality': 0.0,
                'entrypoint': 0.0,
                'examples': 0.0
            }
        
        # Use configured weights if V2 features enabled
        return {
            'doc': self.weights.doc,
            'readme': self.weights.readme,
            'import_deg': self.weights.import_deg,
            'path': self.weights.path,
            'test_link': self.weights.test_link,
            'churn': self.weights.churn,
            'centrality': self.weights.centrality,
            'entrypoint': self.weights.entrypoint,
            'examples': self.weights.examples
        }
        
    def _calculate_readme_score(self, result: ScanResult) -> float:
        """Calculate README-specific importance score."""
        if not result.stats.is_readme:
            return 0.0
            
        score = 3.0  # Base README score
        
        # Boost for root-level README
        if result.stats.depth <= 2:
            score += 1.0
            
        # Architecture and design documents get extra boost
        path_lower = result.stats.path.lower()
        if any(keyword in path_lower for keyword in ['architecture', 'design', 'adr']):
            score += 0.5
            
        return score
        
    def _calculate_import_score(self, result: ScanResult, all_results: List[ScanResult]) -> float:
        """Calculate import degree score (simplified graph centrality)."""
        if not result.imports:
            return 0.0
            
        path = result.stats.path
        
        # Use cached value if available
        if path in self._degree_cache:
            return self._degree_cache[path]
            
        # Build import graph lazily
        if self._import_graph is None:
            self._import_graph = self._build_import_graph(all_results)
            
        # Calculate in-degree (how many files import this one)
        in_degree = sum(1 for imports in self._import_graph.values() if path in imports)
        
        # Calculate out-degree (how many files this imports)
        out_degree = len(self._import_graph.get(path, set()))
        
        # Combined degree score (normalized)
        total_files = len(all_results)
        degree_score = (in_degree * 2 + out_degree) / max(total_files, 1)
        
        self._degree_cache[path] = degree_score
        return degree_score
        
    def _calculate_path_score(self, result: ScanResult) -> float:
        """Calculate path depth penalty (files closer to root are more important)."""
        depth = result.stats.depth
        
        # Inverse depth scoring with diminishing returns
        if depth <= 1:
            return 1.0
        elif depth <= 2:
            return 0.8
        elif depth <= 3:
            return 0.6
        else:
            return 1.0 / (depth * 0.5)  # Gentle decay
            
    def _calculate_test_link_score(self, result: ScanResult, all_results: List[ScanResult]) -> float:
        """Calculate test-code relationship score."""
        if result.stats.is_test:
            # Test files get moderate score
            return 0.3
            
        # Check if this code file has corresponding tests
        code_path = result.stats.path
        
        # Look for corresponding test files
        test_count = 0
        for other_result in all_results:
            if not other_result.stats.is_test:
                continue
                
            test_path = other_result.stats.path
            
            # Heuristic test-code matching
            if self._is_test_for_code(test_path, code_path):
                test_count += 1
                
        # Files with tests are more important
        if test_count > 0:
            return min(test_count * 0.2, 1.0)
        else:
            return 0.0
            
    def _is_test_for_code(self, test_path: str, code_path: str) -> bool:
        """Check if test file corresponds to code file (heuristic)."""
        # Extract base names
        test_base = test_path.lower().replace('test_', '').replace('_test', '').replace('.test', '').replace('.spec', '')
        code_base = code_path.lower()
        
        # Remove extensions
        test_stem = test_base.split('.')[-2] if '.' in test_base else test_base
        code_stem = code_base.split('.')[-2] if '.' in code_base else code_base
        
        # Check if stems match
        return test_stem in code_stem or code_stem in test_stem
        
    def _calculate_churn_score(self, result: ScanResult) -> float:
        """Use pre-calculated churn score from scan result."""
        return result.churn_score
        
    def score_file(self, result: ScanResult, all_results: List[ScanResult]) -> ScoreComponents:
        """Calculate complete heuristic score for a file."""
        # Calculate individual components
        doc_score = self._calculate_doc_score(result)
        readme_score = self._calculate_readme_score(result)
        import_score = self._calculate_import_score(result, all_results)
        path_score = self._calculate_path_score(result)
        test_link_score = self._calculate_test_link_score(result, all_results)
        churn_score = self._calculate_churn_score(result)
        
        # Calculate V2 feature scores
        centrality_score = self._calculate_centrality_score(result)
        entrypoint_score = self._calculate_entrypoint_score(result)
        examples_score = self._calculate_examples_score(result)
        
        # Get effective weights based on current feature flag state
        weights = self._get_effective_weights()
        
        # Apply weights and priority boost
        final_score = (
            weights['doc'] * doc_score +
            weights['readme'] * readme_score + 
            weights['import_deg'] * import_score +
            weights['path'] * path_score +
            weights['test_link'] * test_link_score +
            weights['churn'] * churn_score +
            weights['centrality'] * centrality_score +
            weights['entrypoint'] * entrypoint_score +
            weights['examples'] * examples_score +
            result.priority_boost  # Add priority boost from scanner
        )
        
        return ScoreComponents(
            doc_score=doc_score,
            readme_score=readme_score,
            import_score=import_score,
            path_score=path_score,
            test_link_score=test_link_score,
            churn_score=churn_score,
            final_score=final_score,
            centrality_score=centrality_score,
            entrypoint_score=entrypoint_score,
            examples_score=examples_score
        )
        
    def score_all_files(self, scan_results: List[ScanResult]) -> List[Tuple[ScanResult, ScoreComponents]]:
        """Score all files and return sorted by importance."""
        # Reset caches for new batch
        self._import_graph = None
        self._degree_cache.clear()
        
        scored_files = []
        for result in scan_results:
            score_components = self.score_file(result, scan_results)
            scored_files.append((result, score_components))
            
        # Sort by final score (descending)
        scored_files.sort(key=lambda x: x[1].final_score, reverse=True)
        
        return scored_files
        
    def get_top_files(self, scan_results: List[ScanResult], top_k: int = 256) -> List[Tuple[ScanResult, ScoreComponents]]:
        """Get top-K files by heuristic score."""
        scored_files = self.score_all_files(scan_results)
        return scored_files[:top_k]


def create_scorer(weights: Optional[HeuristicWeights] = None) -> HeuristicScorer:
    """Create a HeuristicScorer instance."""
    return HeuristicScorer(weights)
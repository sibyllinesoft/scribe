"""
Utility Calculation System for V3 Demotion Controller

Implements utility calculation protocols for the V3 demotion controller,
providing ΔU/c calculations for chunk selection and demotion decisions.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..chunker.base import Chunk, ChunkKind
from ..selector.base import SelectionConfig

logger = logging.getLogger(__name__)


class V3UtilityCalculator:
    """
    Utility calculator implementing protocols required by V3 demotion controller.
    
    Provides comprehensive utility calculations for:
    - ΔU/c (utility per cost) for chunk selection decisions
    - Coverage contributions for facility location algorithms
    - Diversity contributions for MMR-style algorithms
    """
    
    def __init__(self):
        """Initialize utility calculator."""
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._similarity_cache: Dict[tuple, float] = {}
        
        # Performance tracking
        self._calculations_count = 0
        self._cache_hits = 0
    
    def calculate_utility_per_cost(
        self, 
        chunk: Chunk, 
        mode: str,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> float:
        """
        Calculate ΔU/c (utility per cost) for a chunk in given mode.
        
        This is the core utility calculation for V3 demotion decisions.
        Higher values indicate better utility per token spent.
        
        Args:
            chunk: Chunk to calculate utility for
            mode: Mode to evaluate chunk in ('full', 'signature', 'summary')
            selected_chunks: Currently selected chunks for context
            config: Selection configuration
            
        Returns:
            Utility per cost ratio (higher is better)
        """
        self._calculations_count += 1
        
        # Get token cost for this mode
        cost = self._get_chunk_tokens(chunk, mode)
        if cost <= 0:
            return 0.0
        
        # Calculate base utility components
        relevance_score = self._calculate_relevance_score(chunk, config)
        coverage_score = self._calculate_coverage_score(chunk, selected_chunks, config)
        diversity_score = self._calculate_diversity_score(chunk, selected_chunks, config)
        quality_score = self._calculate_quality_score(chunk, mode, config)
        
        # Mode-specific adjustments
        mode_multiplier = self._get_mode_multiplier(mode)
        
        # Combined utility score
        utility = (
            config.coverage_weight * coverage_score +
            (1 - config.coverage_weight) * diversity_score +
            0.3 * relevance_score +
            0.2 * quality_score
        ) * mode_multiplier
        
        # Calculate utility per cost
        utility_per_cost = utility / cost
        
        logger.debug(
            f"Utility calculation for {chunk.id}[{mode}]: "
            f"utility={utility:.3f}, cost={cost}, ΔU/c={utility_per_cost:.4f}"
        )
        
        return utility_per_cost
    
    def calculate_coverage_contribution(
        self,
        chunk: Chunk,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> float:
        """
        Calculate chunk's contribution to coverage score.
        
        Args:
            chunk: Chunk to evaluate
            selected_chunks: Currently selected chunks
            config: Selection configuration
            
        Returns:
            Coverage contribution score (0-1)
        """
        return self._calculate_coverage_score(chunk, selected_chunks, config)
    
    def calculate_diversity_contribution(
        self,
        chunk: Chunk,
        selected_chunks: List[Chunk], 
        config: SelectionConfig
    ) -> float:
        """
        Calculate chunk's contribution to diversity score.
        
        Args:
            chunk: Chunk to evaluate
            selected_chunks: Currently selected chunks
            config: Selection configuration
            
        Returns:
            Diversity contribution score (0-1)
        """
        return self._calculate_diversity_score(chunk, selected_chunks, config)
    
    def calculate_demotion_impact(
        self,
        chunk: Chunk,
        current_mode: str,
        demoted_mode: str,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> Dict[str, float]:
        """
        Calculate impact of demoting a chunk from current mode to demoted mode.
        
        Args:
            chunk: Chunk being demoted
            current_mode: Current chunk mode
            demoted_mode: Mode after demotion
            selected_chunks: Currently selected chunks
            config: Selection configuration
            
        Returns:
            Dictionary with impact metrics
        """
        current_utility = self.calculate_utility_per_cost(
            chunk, current_mode, selected_chunks, config
        )
        
        demoted_utility = self.calculate_utility_per_cost(
            chunk, demoted_mode, selected_chunks, config
        )
        
        tokens_freed = (
            self._get_chunk_tokens(chunk, current_mode) - 
            self._get_chunk_tokens(chunk, demoted_mode)
        )
        
        utility_loss = current_utility - demoted_utility
        
        return {
            'current_utility': current_utility,
            'demoted_utility': demoted_utility,
            'utility_loss': utility_loss,
            'tokens_freed': tokens_freed,
            'utility_per_token_freed': utility_loss / max(1, tokens_freed),
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get utility calculator performance metrics."""
        cache_hit_rate = self._cache_hits / max(1, self._calculations_count)
        
        return {
            'calculations_count': self._calculations_count,
            'cache_hits': self._cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'feature_cache_size': len(self._feature_cache),
            'similarity_cache_size': len(self._similarity_cache),
        }
    
    def clear_cache(self):
        """Clear utility calculation caches."""
        self._feature_cache.clear()
        self._similarity_cache.clear()
        self._cache_hits = 0
        logger.info("Cleared utility calculator caches")
    
    # Private implementation methods
    
    def _calculate_relevance_score(
        self,
        chunk: Chunk,
        config: SelectionConfig
    ) -> float:
        """Calculate chunk relevance based on intrinsic properties."""
        score = 1.0
        
        # Chunk kind boosts
        kind_boosts = {
            ChunkKind.MAIN: 3.0,
            ChunkKind.CLASS: 2.0,
            ChunkKind.FUNCTION: 1.5,
            ChunkKind.IMPORT: 0.5,
            ChunkKind.TEST: config.boost_tests,
            ChunkKind.DOCSTRING: 1.2,
        }
        score *= kind_boosts.get(chunk.kind, 1.0)
        
        # File type boosts
        path_lower = chunk.rel_path.lower()
        manifest_patterns = ['package.json', 'requirements.txt', 'go.mod', 'cargo.toml', 'pyproject.toml']
        entrypoint_patterns = ['main.', 'index.', '__init__', 'app.']
        
        if any(pattern in path_lower for pattern in manifest_patterns):
            score *= config.boost_manifests
        elif any(pattern in path_lower for pattern in entrypoint_patterns):
            score *= config.boost_entrypoints
        
        # Must-include patterns
        import re
        for pattern in config.must_include_patterns:
            if re.search(pattern, chunk.rel_path, re.IGNORECASE):
                score *= 5.0
                break
        
        # Documentation density boost
        score *= (1.0 + chunk.doc_density * 0.5)
        
        # Complexity penalty
        if chunk.complexity_score > 5.0:
            score *= (1.0 - min(0.5, (chunk.complexity_score - 5.0) / 10.0))
        
        return min(score, 10.0)  # Cap at 10.0 to prevent extreme scores
    
    def _calculate_coverage_score(
        self,
        chunk: Chunk,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> float:
        """Calculate coverage contribution using facility location principles."""
        if not selected_chunks:
            return 1.0  # Maximum coverage gain if nothing selected
        
        chunk_features = self._get_chunk_features(chunk)
        
        # Calculate minimum distance to selected chunks (facility location)
        min_distance = float('inf')
        
        for selected_chunk in selected_chunks:
            selected_features = self._get_chunk_features(selected_chunk)
            distance = self._calculate_feature_distance(chunk_features, selected_features)
            min_distance = min(min_distance, distance)
        
        # Convert distance to coverage score (higher distance = better coverage)
        coverage_score = min(1.0, min_distance / 2.0)  # Normalize to [0, 1]
        
        # File-level coverage bonus
        selected_files = {c.rel_path for c in selected_chunks}
        if chunk.rel_path not in selected_files:
            coverage_score *= 1.2  # Bonus for covering new files
        
        return coverage_score
    
    def _calculate_diversity_score(
        self,
        chunk: Chunk,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> float:
        """Calculate diversity contribution using MMR principles."""
        if not selected_chunks:
            return 1.0  # Maximum diversity if nothing selected
        
        chunk_features = self._get_chunk_features(chunk)
        
        # Calculate maximum similarity to selected chunks
        max_similarity = 0.0
        
        for selected_chunk in selected_chunks:
            selected_features = self._get_chunk_features(selected_chunk)
            similarity = self._calculate_cosine_similarity(chunk_features, selected_features)
            max_similarity = max(max_similarity, similarity)
        
        # Diversity is 1 - maximum similarity
        diversity_score = 1.0 - max_similarity
        
        # Penalize chunks in same file (reduces diversity)
        same_file_chunks = [c for c in selected_chunks if c.rel_path == chunk.rel_path]
        if same_file_chunks:
            diversity_score *= 0.8
        
        return max(0.0, diversity_score)
    
    def _calculate_quality_score(
        self,
        chunk: Chunk,
        mode: str,
        config: SelectionConfig
    ) -> float:
        """Calculate chunk quality score based on mode and properties."""
        base_score = 1.0
        
        # Mode affects quality perception
        if mode == 'full':
            base_score *= 1.0  # Full content is highest quality
        elif mode == 'summary':
            base_score *= 0.8  # Summary is good but not complete
        elif mode == 'signature':
            base_score *= 0.6  # Signature is lowest quality but still useful
        
        # Documentation boosts quality
        if chunk.doc_density > 0.5:
            base_score *= 1.2
        
        # Test links boost quality
        if chunk.test_links > 0:
            base_score *= 1.1
        
        # Complexity affects quality (moderate complexity is good)
        if 2.0 <= chunk.complexity_score <= 8.0:
            base_score *= 1.1  # Good complexity range
        elif chunk.complexity_score > 15.0:
            base_score *= 0.8  # Too complex
        elif chunk.complexity_score < 1.0:
            base_score *= 0.9  # Too simple
        
        return base_score
    
    def _get_mode_multiplier(self, mode: str) -> float:
        """Get mode-specific multiplier for utility calculation."""
        mode_multipliers = {
            'full': 1.0,      # Full content gets base utility
            'summary': 0.85,  # Summary mode gets slight penalty
            'signature': 0.7, # Signature mode gets larger penalty
        }
        return mode_multipliers.get(mode, 1.0)
    
    def _get_chunk_features(self, chunk: Chunk) -> np.ndarray:
        """Get feature vector for chunk (with caching)."""
        cache_key = f"{chunk.id}_{chunk.full_tokens}_{chunk.complexity_score}"
        
        if cache_key in self._feature_cache:
            self._cache_hits += 1
            return self._feature_cache[cache_key]
        
        # Generate features
        features = [
            chunk.full_tokens / 1000.0,         # Normalized token count
            chunk.complexity_score / 10.0,      # Normalized complexity
            chunk.doc_density,                  # Documentation density
            float(chunk.kind == ChunkKind.FUNCTION),  # Is function
            float(chunk.kind == ChunkKind.CLASS),     # Is class
            float(chunk.kind == ChunkKind.IMPORT),    # Is import
            float(chunk.test_links > 0),              # Has tests
            len(chunk.dependencies) / 10.0,          # Normalized dependency count
            float('main' in chunk.rel_path.lower()),  # Is main file
            float('test' in chunk.rel_path.lower()),  # Is test file
        ]
        
        feature_vector = np.array(features, dtype=np.float32)
        self._feature_cache[cache_key] = feature_vector
        
        return feature_vector
    
    def _calculate_feature_distance(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        """Calculate distance between feature vectors."""
        return np.linalg.norm(features_a - features_b)
    
    def _calculate_cosine_similarity(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors (with caching)."""
        # Create cache key from feature hashes
        key_a = hash(features_a.tobytes())
        key_b = hash(features_b.tobytes())
        cache_key = (min(key_a, key_b), max(key_a, key_b))
        
        if cache_key in self._similarity_cache:
            self._cache_hits += 1
            return self._similarity_cache[cache_key]
        
        # Calculate similarity
        dot_product = np.dot(features_a, features_b)
        norm_a = np.linalg.norm(features_a)
        norm_b = np.linalg.norm(features_b)
        
        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_a * norm_b)
        
        # Cache result
        self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _get_chunk_tokens(self, chunk: Chunk, mode: str) -> int:
        """Get token count for chunk in specified mode."""
        if mode == "full":
            return chunk.full_tokens
        elif mode == "signature":
            return chunk.signature_tokens
        elif mode == "summary":
            return chunk.summary_tokens if chunk.summary_tokens else chunk.signature_tokens
        else:
            return chunk.full_tokens
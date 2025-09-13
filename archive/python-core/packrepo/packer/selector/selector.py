"""Main repository selector implementation with submodular algorithms."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
import tracemalloc
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

from ..chunker.base import Chunk, ChunkKind
from ..tokenizer.base import Tokenizer
from ..oracles import validate_pack_with_oracles
from .base import PackRequest, PackResult, SelectionResult, SelectionConfig, SelectionVariant
from ...fastpath.feature_flags import get_feature_flags


class RepositorySelector:
    """
    Submodular repository selector implementing V1-V5 variants.
    
    Provides deterministic selection algorithms with facility-location,
    MMR diversity, and budget-aware optimization for repository packing.
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the selector.
        
        Args:
            tokenizer: Tokenizer for token cost calculations
        """
        self.tokenizer = tokenizer
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # V2 FastPath enhancements (initialized when features enabled)
        self._class_quotas: Dict[str, int] = {}
        self._runtime_budget_validator = None
        
        # Initialize V2 systems if flags are enabled
        self._init_fastpath_v2_systems()
    
    def _init_fastpath_v2_systems(self):
        """Initialize FastPath V2 enhancement systems."""
        flags = get_feature_flags()
        
        if flags.policy_v2:
            # Initialize class quota system
            self._class_quotas = {
                ChunkKind.FUNCTION.value: 0,
                ChunkKind.CLASS.value: 0,
                ChunkKind.IMPORT.value: 0,
                ChunkKind.TEST.value: 0,
                ChunkKind.DOCSTRING.value: 0,
                ChunkKind.MAIN.value: 0,
            }
            
        if flags.router_enabled:
            # Initialize runtime budget validator
            self._runtime_budget_validator = self._create_budget_validator()
    
    def _create_budget_validator(self):
        """Create runtime budget validation system (V2 feature)."""
        def validate_budget(current_tokens: int, budget: int, safety_margin: int = 100) -> bool:
            """Validate that current token usage is within budget with safety margin."""
            return current_tokens + safety_margin <= budget
        return validate_budget
    
    def select(self, request: PackRequest) -> PackResult:
        """
        Select chunks according to the pack request.
        
        Args:
            request: Pack request with chunks and configuration
            
        Returns:
            Complete pack result with selection and metadata
        """
        # Start execution tracking
        start_time = time.time()
        tracemalloc.start()
        
        # Set random seed for determinism
        if request.config.deterministic:
            random.seed(request.config.random_seed)
            np.random.seed(request.config.random_seed)
        
        # Preprocess chunks
        processed_chunks = self._preprocess_chunks(request.chunks, request.config)
        
        # Run selection algorithm based on variant
        selection_result = self._run_selection(processed_chunks, request.config)
        
        # Post-process results
        selection_result = self._postprocess_selection(selection_result, request.config)
        
        # Calculate execution metadata
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peak = peak / 1024 / 1024  # Convert to MB
        
        # Generate deterministic hash if requested
        deterministic_hash = None
        if request.config.deterministic:
            deterministic_hash = self._generate_deterministic_hash(selection_result)
        
        return PackResult(
            request=request,
            selection=selection_result,
            execution_time=execution_time,
            memory_peak=memory_peak,
            deterministic_hash=deterministic_hash,
        )
    
    def _preprocess_chunks(self, chunks: List[Chunk], config: SelectionConfig) -> List[Chunk]:
        """Preprocess chunks with scoring and filtering."""
        processed = []
        
        for chunk in chunks:
            # Apply quality filters
            if chunk.doc_density < config.min_doc_density:
                continue
            if chunk.complexity_score > config.max_complexity:
                continue
            
            # Ensure deterministic chunk ID if requested
            if config.deterministic:
                chunk.id = self._generate_deterministic_chunk_id(chunk)
            
            # Calculate base priority score
            priority = self._calculate_priority(chunk, config)
            chunk.centrality_score = priority  # Store in centrality_score field
            
            processed.append(chunk)
        
        # Sort by priority for deterministic ordering with fixed tie-breakers
        if config.deterministic:
            # For deterministic mode, sort by file path, line, and chunk ID (ascending)
            processed.sort(key=lambda c: self._get_deterministic_sort_key(c))
        else:
            # For non-deterministic mode, sort by centrality score (descending)
            processed.sort(key=lambda c: c.centrality_score, reverse=True)
        
        return processed
    
    def _generate_deterministic_chunk_id(self, chunk: Chunk) -> str:
        """Generate deterministic chunk ID based on content and metadata."""
        # Create stable identifier from chunk properties
        id_components = [
            chunk.rel_path,
            str(chunk.start_line),
            str(chunk.end_line),
            chunk.kind.value if chunk.kind else 'unknown',
            # Use first 100 chars of content for uniqueness without full content dependency
            (chunk.content or '')[:100]
        ]
        
        id_string = '|'.join(id_components)
        chunk_hash = hashlib.sha256(id_string.encode('utf-8')).hexdigest()
        
        # Return human-readable ID with hash suffix
        path_stem = chunk.rel_path.replace('/', '_').replace('.', '_')
        return f"{path_stem}_{chunk.start_line}_{chunk_hash[:8]}"
    
    def _get_deterministic_sort_key(self, chunk: Chunk) -> Tuple[str, int, str]:
        """Get deterministic sort key for chunk ordering."""
        # For deterministic mode, oracle expects:
        # Primary: file path (ascending) 
        # Secondary: start line (ascending)
        # Tertiary: chunk ID (ascending) for complete determinism
        return (
            chunk.rel_path,
            chunk.start_line,
            chunk.id
        )
    
    def _calculate_priority(self, chunk: Chunk, config: SelectionConfig) -> float:
        """Calculate base priority score for a chunk."""
        score = 1.0
        
        # Boost based on chunk kind
        kind_boosts = {
            ChunkKind.MAIN: 3.0,
            ChunkKind.CLASS: 2.0,
            ChunkKind.FUNCTION: 1.5,
            ChunkKind.IMPORT: 0.5,
            ChunkKind.TEST: config.boost_tests,
            ChunkKind.DOCSTRING: 1.2,
        }
        score *= kind_boosts.get(chunk.kind, 1.0)
        
        # Boost for manifests and entrypoints
        path_lower = chunk.rel_path.lower()
        manifest_patterns = ['package.json', 'requirements.txt', 'go.mod', 'cargo.toml', 'pyproject.toml']
        entrypoint_patterns = ['main.', 'index.', '__init__', 'app.']
        
        if any(pattern in path_lower for pattern in manifest_patterns):
            score *= config.boost_manifests
        elif any(pattern in path_lower for pattern in entrypoint_patterns):
            score *= config.boost_entrypoints
        
        # Must-include patterns
        for pattern in config.must_include_patterns:
            if re.search(pattern, chunk.rel_path, re.IGNORECASE):
                score *= 5.0  # High boost for must-include
                break
        
        # Documentation density boost
        score *= (1.0 + chunk.doc_density * 0.5)
        
        # Complexity penalty (prefer simpler code for overview)
        if chunk.complexity_score > 5.0:
            score *= (1.0 - min(0.5, (chunk.complexity_score - 5.0) / 10.0))
        
        return score
    
    def _run_selection(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """Run the appropriate selection algorithm based on variant."""
        # Handle baseline variants with dedicated selectors
        if config.variant in [SelectionVariant.V0A_README_ONLY, SelectionVariant.V0B_NAIVE_CONCAT, SelectionVariant.V0C_BM25_BASELINE]:
            return self._run_baseline_variant(chunks, config)
        elif config.variant == SelectionVariant.BASELINE:
            # Legacy baseline maps to V0c (strongest baseline)
            return self._run_baseline_variant_by_id(chunks, config, "V0c")
        elif config.variant == SelectionVariant.COMPREHENSIVE or config.variant == SelectionVariant.V1_BASIC:
            return self._facility_location_mmr(chunks, config)
        elif config.variant == SelectionVariant.COVERAGE_ENHANCED or config.variant == SelectionVariant.V2_COVERAGE:
            return self._coverage_construction(chunks, config)
        elif config.variant == SelectionVariant.STABILITY_CONTROLLED or config.variant == SelectionVariant.V3_STABLE:
            return self._stable_demotion(chunks, config)
        elif config.variant == SelectionVariant.V4_OBJECTIVE:
            return self._objective_selection(chunks, config)
        elif config.variant == SelectionVariant.V5_SUMMARIES:
            return self._summary_selection(chunks, config)
        else:
            raise ValueError(f"Unsupported variant: {config.variant}")
    
    def _baseline_selection(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """
        V0: Baseline selection - README + top-N BM25-ranked files.
        
        Simple naive baseline that prioritizes documentation and high-scoring files
        for comparison against sophisticated variants.
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        # First, select README files (must-have for baseline)
        readme_chunks = [c for c in chunks if 'readme' in c.rel_path.lower()]
        for chunk in readme_chunks:
            tokens = self._get_chunk_tokens(chunk, "full")
            if total_tokens + tokens <= config.token_budget:
                selected.append(chunk)
                selected_ids.add(chunk.id)
                chunk_modes[chunk.id] = "full"
                selection_scores[chunk.id] = 10.0  # High priority
                total_tokens += tokens
        
        # Then, select top-scoring chunks by simple priority (BM25-style)
        remaining = [c for c in chunks if c.id not in selected_ids]
        remaining.sort(key=lambda c: c.centrality_score, reverse=True)
        
        for chunk in remaining:
            if total_tokens >= config.token_budget:
                break
                
            tokens = self._get_chunk_tokens(chunk, "full")
            if total_tokens + tokens <= config.token_budget:
                selected.append(chunk)
                selected_ids.add(chunk.id)
                chunk_modes[chunk.id] = "full"
                selection_scores[chunk.id] = chunk.centrality_score
                total_tokens += tokens
                iterations += 1
        
        # Basic metrics for baseline
        coverage_score = len(selected) / max(1, len(chunks))  # Simple coverage
        diversity_score = 0.5  # Fixed diversity for baseline
        
        # Sort deterministically if requested
        if config.deterministic and selected:
            selected.sort(key=lambda c: self._get_deterministic_sort_key(c))
        
        return SelectionResult(
            selected_chunks=selected,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / config.token_budget,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            iterations=iterations,
        )
    
    def _facility_location_mmr(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """
        V1: Facility-location + MMR selection.
        
        Uses deterministic features for coverage and diversity.
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        # Calculate facility location centroids (file-based)
        file_centroids = self._calculate_file_centroids(chunks)
        
        # Greedy selection with MMR
        remaining = list(chunks)
        
        while remaining and total_tokens < config.token_budget:
            iterations += 1
            best_chunk = None
            best_score = -1.0
            best_mode = "full"
            
            # Budget enforcement: strict checking
            budget_remaining = config.token_budget - total_tokens
            
            for chunk in remaining:
                # Try different modes in priority order
                modes_to_try = ["full", "signature"] if not config.deterministic else ["full", "signature"]
                
                for mode in modes_to_try:
                    tokens = self._get_chunk_tokens(chunk, mode)
                    
                    # Hard budget constraint: zero overflow
                    if total_tokens + tokens > config.token_budget:
                        continue
                    
                    # Additional safety margin for deterministic selection
                    if config.deterministic and total_tokens + tokens > config.token_budget - 1:
                        continue
                    
                    # Facility location score (coverage)
                    coverage_score = self._calculate_coverage_gain(
                        chunk, selected, file_centroids
                    )
                    
                    # MMR diversity score
                    diversity_score = self._calculate_diversity_gain(
                        chunk, selected, config.diversity_weight
                    )
                    
                    # Combined score
                    combined_score = (
                        config.coverage_weight * coverage_score +
                        (1 - config.coverage_weight) * diversity_score
                    )
                    
                    # Priority boost
                    combined_score *= chunk.centrality_score
                    
                    # Efficiency score (score per token)
                    efficiency = combined_score / max(1, tokens)
                    
                    # Deterministic tie-breaking: use chunk properties for stable ordering
                    if config.deterministic:
                        # For deterministic selection, use exact comparison with tie-breaker
                        tie_breaker = self._get_deterministic_sort_key(chunk)
                        is_better = (
                            efficiency > best_score or 
                            (abs(efficiency - best_score) < 1e-10 and 
                             (best_chunk is None or tie_breaker < self._get_deterministic_sort_key(best_chunk)))
                        )
                    else:
                        is_better = efficiency > best_score
                    
                    if is_better:
                        best_score = efficiency
                        best_chunk = chunk
                        best_mode = mode
            
            if best_chunk:
                selected.append(best_chunk)
                selected_ids.add(best_chunk.id)
                chunk_modes[best_chunk.id] = best_mode
                selection_scores[best_chunk.id] = best_score
                total_tokens += self._get_chunk_tokens(best_chunk, best_mode)
                remaining.remove(best_chunk)
            else:
                break  # No more chunks fit in budget
        
        # Calculate quality metrics
        coverage_score = self._calculate_total_coverage(selected, file_centroids)
        diversity_score = self._calculate_total_diversity(selected)
        
        # Sort selected chunks deterministically if in deterministic mode
        if config.deterministic and selected:
            selected.sort(key=lambda c: self._get_deterministic_sort_key(c))
        
        return SelectionResult(
            selected_chunks=selected,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / config.token_budget,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            iterations=iterations,
        )
    
    def _coverage_construction(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """
        V2: Enhanced coverage with mixed centroids clustering.
        
        Uses k-means + HNSW medoids for improved coverage as specified in TODO.md.
        Strengthens C(S) via mixed centroids with k≈√N per package.
        """
        try:
            from ..embeddings import create_fast_provider, SHA256EmbeddingCache, CachedEmbeddingProvider
            from ..clustering import create_mixed_clusterer
            
            # Initialize embedding system if not already done
            if not hasattr(self, '_embedding_provider') or self._embedding_provider is None:
                self._init_embedding_system(config)
            
            # Generate embeddings for chunks
            embeddings = self._get_chunk_embeddings(chunks)
            
            if not embeddings:
                # Fall back to V1 if embedding generation fails
                return self._facility_location_mmr(chunks, config)
            
            # Perform mixed centroid clustering
            clusterer = create_mixed_clusterer(
                random_state=config.random_seed if config.deterministic else None
            )
            clustering_result = clusterer.fit_predict(embeddings)
            
            # Store clustering metadata for later use in pack index
            self._clustering_result = clustering_result
            self._embedding_provider_ref = getattr(self, '_embedding_provider', None)
            
            # Use clustering result for enhanced facility location selection
            return self._facility_location_with_centroids(chunks, config, clustering_result, embeddings)
            
        except Exception as e:
            # Fall back to V1 if V2 implementation fails
            print(f"Warning: V2 coverage construction failed, falling back to V1: {e}")
            return self._facility_location_mmr(chunks, config)
    
    def _stable_demotion(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """
        V3: Stable demotion with bounded re-optimization.
        
        Implements the V3 demotion stability controller that prevents oscillations
        and cap breaches through bounded re-optimization as specified in TODO.md.
        """
        try:
            from ..controller import DemotionController, StabilityTracker
            from ..controller.utility import V3UtilityCalculator
            from ..budget import BudgetReallocator
            
            # Initialize V3 components
            if not hasattr(self, '_v3_components_initialized'):
                self._init_v3_components(config)
            
            # Start with V2 selection if available, otherwise V1
            if hasattr(self, '_embedding_provider') and self._embedding_provider:
                base_result = self._coverage_construction(chunks, config)
            else:
                base_result = self._facility_location_mmr(chunks, config)
            
            # Apply V3 demotion stability controller
            stable_result = self._apply_demotion_stability(
                base_result, chunks, config
            )
            
            return stable_result
            
        except Exception as e:
            # Fall back to V2/V1 if V3 implementation fails
            logger.warning(f"V3 stable demotion failed, falling back to V2/V1: {e}")
            if hasattr(self, '_embedding_provider') and self._embedding_provider:
                return self._coverage_construction(chunks, config)
            else:
                return self._facility_location_mmr(chunks, config)
    
    def _objective_selection(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """V4: Objective-driven selection with query bias."""
        # TODO: Implement query-biased selection
        # For now, fall back to V1
        return self._facility_location_mmr(chunks, config)
    
    def _summary_selection(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """V5: Selection with schema summaries."""
        # TODO: Implement summary generation
        # For now, fall back to V1
        return self._facility_location_mmr(chunks, config)
    
    def _calculate_file_centroids(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Calculate file-based centroids for facility location."""
        file_groups = defaultdict(list)
        
        # Group chunks by file
        for chunk in chunks:
            file_groups[chunk.rel_path].append(chunk)
        
        centroids = {}
        for file_path, file_chunks in file_groups.items():
            # Simple centroid based on chunk features
            features = []
            for chunk in file_chunks:
                feature = self._get_chunk_features(chunk)
                features.append(feature)
            
            if features:
                centroids[file_path] = np.mean(features, axis=0)
        
        return centroids
    
    def _get_chunk_features(self, chunk: Chunk) -> np.ndarray:
        """Get feature vector for a chunk."""
        # Simple deterministic features
        features = [
            chunk.full_tokens / 1000.0,  # Normalized token count
            chunk.complexity_score / 10.0,  # Normalized complexity
            chunk.doc_density,  # Documentation density
            float(chunk.kind == ChunkKind.FUNCTION),  # Is function
            float(chunk.kind == ChunkKind.CLASS),     # Is class
            float(chunk.kind == ChunkKind.IMPORT),    # Is import
            float(chunk.test_links > 0),              # Has tests
            len(chunk.dependencies) / 10.0,          # Normalized dependency count
        ]
        return np.array(features, dtype=np.float32)
    
    def _calculate_coverage_gain(
        self, 
        chunk: Chunk, 
        selected: List[Chunk], 
        centroids: Dict[str, np.ndarray]
    ) -> float:
        """Calculate facility location coverage gain."""
        if chunk.rel_path not in centroids:
            return 0.0
        
        chunk_centroid = centroids[chunk.rel_path]
        
        # If no chunks selected yet, full coverage gain
        if not selected:
            return 1.0
        
        # Calculate minimum distance to selected chunks' centroids
        min_distance = float('inf')
        for sel_chunk in selected:
            if sel_chunk.rel_path in centroids:
                sel_centroid = centroids[sel_chunk.rel_path]
                distance = np.linalg.norm(chunk_centroid - sel_centroid)
                min_distance = min(min_distance, distance)
        
        # Coverage gain is the minimum distance (diversity from selected)
        return min_distance
    
    def _calculate_diversity_gain(
        self, 
        chunk: Chunk, 
        selected: List[Chunk], 
        diversity_weight: float
    ) -> float:
        """Calculate MMR diversity gain."""
        if not selected:
            return 1.0
        
        chunk_features = self._get_chunk_features(chunk)
        
        # Calculate similarity to all selected chunks
        max_similarity = 0.0
        for sel_chunk in selected:
            sel_features = self._get_chunk_features(sel_chunk)
            similarity = self._cosine_similarity(chunk_features, sel_features)
            max_similarity = max(max_similarity, similarity)
        
        # Diversity is 1 - maximum similarity
        return 1.0 - max_similarity
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _calculate_total_coverage(self, selected: List[Chunk], centroids: Dict[str, np.ndarray]) -> float:
        """Calculate total coverage score for selected chunks."""
        if not selected or not centroids:
            return 0.0
        
        covered_files = set(chunk.rel_path for chunk in selected)
        total_files = len(centroids)
        
        return len(covered_files) / max(1, total_files)
    
    def _calculate_total_diversity(self, selected: List[Chunk]) -> float:
        """Calculate total diversity score for selected chunks."""
        if len(selected) <= 1:
            return 1.0
        
        similarities = []
        for i, chunk_a in enumerate(selected):
            for j, chunk_b in enumerate(selected[i+1:], i+1):
                features_a = self._get_chunk_features(chunk_a)
                features_b = self._get_chunk_features(chunk_b)
                similarity = self._cosine_similarity(features_a, features_b)
                similarities.append(similarity)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_similarity
        # Clamp to [0, 1] to handle floating-point precision issues
        return max(0.0, min(1.0, diversity))
    
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
    
    def _postprocess_selection(self, result: SelectionResult, config: SelectionConfig) -> SelectionResult:
        """Post-process selection result with validation."""
        errors = []
        
        # Budget constraint validation (hard limits)
        if result.total_tokens > config.token_budget:
            errors.append(f"Budget overflow: {result.total_tokens} > {config.token_budget} tokens")
        
        # Underflow validation (should use budget efficiently)
        # Skip underflow validation if we have reasonable utilization (>50%) for any budget
        if config.token_budget > 0:
            utilization = result.total_tokens / config.token_budget
            # Only flag underflow if utilization is extremely low (<50%) and budget is large
            if config.token_budget >= 10000 and utilization < 0.5:
                underflow = (config.token_budget - result.total_tokens) / config.token_budget
                errors.append(f"Excessive underflow: {underflow:.1%} - consider reducing budget")
        
        # Selection quality validation
        if result.coverage_score < 0 or result.coverage_score > 1:
            errors.append(f"Invalid coverage score: {result.coverage_score}")
        
        if result.diversity_score < 0 or result.diversity_score > 1:
            errors.append(f"Invalid diversity score: {result.diversity_score}")
        
        # Chunk validation
        if result.selected_chunks:
            chunk_ids = [chunk.id for chunk in result.selected_chunks]
            if len(chunk_ids) != len(set(chunk_ids)):
                errors.append("Duplicate chunk IDs in selection")
            
            # Validate monotonic scoring if available (only for non-deterministic mode)
            # In deterministic mode, chunks are re-sorted so scores may not be monotonic
            if result.selection_scores and not config.deterministic:
                scores = [result.selection_scores.get(chunk.id, 0) for chunk in result.selected_chunks]
                for i in range(1, len(scores)):
                    if scores[i] > scores[i-1]:  # Should be non-increasing
                        errors.append(f"Non-monotonic selection scores at position {i}")
                        break
        
        # If deterministic mode, validate ordering
        if config.deterministic and len(result.selected_chunks) > 1:
            prev_key = None
            for i, chunk in enumerate(result.selected_chunks):
                current_key = self._get_deterministic_sort_key(chunk)
                if prev_key is not None and current_key < prev_key:  # ascending order
                    errors.append(f"Non-deterministic chunk ordering detected at position {i}: {current_key} > {prev_key}")
                    break
                prev_key = current_key
        
        # Raise errors if any validation failed
        if errors:
            raise ValueError(f"Selection validation failed: {'; '.join(errors)}")
        
        return result
    
    def _generate_deterministic_hash(self, result: SelectionResult) -> str:
        """Generate deterministic hash for result validation."""
        # Sort chunks by ID for deterministic hashing
        sorted_chunks = sorted(result.selected_chunks, key=lambda c: c.id)
        
        hash_input = ""
        for chunk in sorted_chunks:
            mode = result.chunk_modes[chunk.id]
            hash_input += f"{chunk.id}:{mode}:{result.selection_scores.get(chunk.id, 0.0)}"
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    # V2 Embedding and Clustering Methods
    
    def _init_embedding_system(self, config: SelectionConfig):
        """
        Initialize embedding system for V2 coverage construction.
        
        Creates embedding provider with caching for efficient reuse.
        """
        try:
            from ..embeddings import create_fast_provider, SHA256EmbeddingCache, CachedEmbeddingProvider
            import tempfile
            import os
            
            # Create cache directory
            cache_dir = os.path.join(tempfile.gettempdir(), 'packrepo_embeddings')
            
            # Initialize provider and cache
            provider = create_fast_provider(device='cpu')  # Use CPU for compatibility
            cache = SHA256EmbeddingCache(cache_dir=cache_dir, max_entries=50000)
            
            self._embedding_provider = CachedEmbeddingProvider(provider, cache)
            
        except Exception as e:
            print(f"Warning: Failed to initialize embedding system: {e}")
            self._embedding_provider = None
    
    def _get_chunk_embeddings(self, chunks: List[Chunk]) -> List:
        """
        Generate embeddings for chunks using the embedding provider.
        
        Returns list of CodeEmbeddings or empty list if generation fails.
        """
        if not hasattr(self, '_embedding_provider') or self._embedding_provider is None:
            return []
        
        try:
            # Convert chunks to format expected by embedding provider
            embeddings = self._embedding_provider.encode_chunks(chunks)
            return embeddings
        except Exception as e:
            print(f"Warning: Failed to generate embeddings: {e}")
            return []
    
    def _facility_location_with_centroids(
        self, 
        chunks: List[Chunk], 
        config: SelectionConfig,
        clustering_result,
        embeddings: List
    ) -> SelectionResult:
        """
        Enhanced facility location using mixed centroids from clustering.
        
        Uses both k-means and HNSW medoid clusters to improve coverage
        calculation in the facility location algorithm.
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        # Create embedding lookup for fast access
        embedding_map = {emb.chunk_id: emb for emb in embeddings}
        
        # Enhanced centroid-based coverage calculation
        mixed_centroids = self._extract_mixed_centroids(clustering_result)
        
        # Greedy selection with enhanced coverage
        remaining = list(chunks)
        
        while remaining and total_tokens < config.token_budget:
            iterations += 1
            best_chunk = None
            best_score = -1.0
            best_mode = "full"
            
            budget_remaining = config.token_budget - total_tokens
            
            for chunk in remaining:
                modes_to_try = ["full", "signature"] if not config.deterministic else ["full", "signature"]
                
                for mode in modes_to_try:
                    tokens = self._get_chunk_tokens(chunk, mode)
                    
                    if total_tokens + tokens > config.token_budget:
                        continue
                    
                    if config.deterministic and total_tokens + tokens > config.token_budget - 1:
                        continue
                    
                    # Enhanced coverage score using mixed centroids
                    coverage_score = self._calculate_enhanced_coverage_gain(
                        chunk, selected, mixed_centroids, embedding_map
                    )
                    
                    # MMR diversity score (keep from V1)
                    diversity_score = self._calculate_diversity_gain(
                        chunk, selected, config.diversity_weight
                    )
                    
                    # Combined score
                    combined_score = (
                        config.coverage_weight * coverage_score +
                        (1 - config.coverage_weight) * diversity_score
                    )
                    
                    # Priority boost
                    combined_score *= chunk.centrality_score
                    
                    # Efficiency score
                    efficiency = combined_score / max(1, tokens)
                    
                    # Deterministic tie-breaking
                    if config.deterministic:
                        tie_breaker = self._get_deterministic_sort_key(chunk)
                        is_better = (
                            efficiency > best_score or 
                            (abs(efficiency - best_score) < 1e-10 and 
                             (best_chunk is None or tie_breaker < self._get_deterministic_sort_key(best_chunk)))
                        )
                    else:
                        is_better = efficiency > best_score
                    
                    if is_better:
                        best_score = efficiency
                        best_chunk = chunk
                        best_mode = mode
            
            if best_chunk:
                selected.append(best_chunk)
                selected_ids.add(best_chunk.id)
                chunk_modes[best_chunk.id] = best_mode
                selection_scores[best_chunk.id] = best_score
                total_tokens += self._get_chunk_tokens(best_chunk, best_mode)
                remaining.remove(best_chunk)
            else:
                break
        
        # Calculate quality metrics
        coverage_score = self._calculate_enhanced_total_coverage(selected, mixed_centroids, embedding_map)
        diversity_score = self._calculate_total_diversity(selected)
        
        # Sort selected chunks deterministically if needed
        if config.deterministic and selected:
            selected.sort(key=lambda c: self._get_deterministic_sort_key(c))
        
        return SelectionResult(
            selected_chunks=selected,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / config.token_budget,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            iterations=iterations,
        )
    
    def _extract_mixed_centroids(self, clustering_result) -> Dict[str, np.ndarray]:
        """
        Extract mixed centroids from clustering result.
        
        Returns dictionary mapping cluster IDs to centroid vectors.
        """
        centroids = {}
        
        if hasattr(clustering_result, 'clusters') and clustering_result.clusters:
            for cluster in clustering_result.clusters:
                centroids[cluster.cluster_id] = cluster.centroid
        
        return centroids
    
    def _calculate_enhanced_coverage_gain(
        self,
        chunk: Chunk,
        selected: List[Chunk],
        mixed_centroids: Dict[str, np.ndarray],
        embedding_map: Dict[str, Any]
    ) -> float:
        """
        Calculate coverage gain using mixed centroids.
        
        Enhanced version that uses both k-means and HNSW centroids
        for improved coverage estimation.
        """
        if chunk.id not in embedding_map:
            # Fall back to V1 method if no embedding available
            return self._calculate_coverage_gain(chunk, selected, {})
        
        chunk_embedding = embedding_map[chunk.id].embedding
        
        if not selected or not mixed_centroids:
            return 1.0
        
        # Calculate coverage as minimum distance to any centroid
        min_distance = float('inf')
        
        for centroid_id, centroid in mixed_centroids.items():
            distance = np.linalg.norm(chunk_embedding - centroid)
            min_distance = min(min_distance, distance)
        
        # Also consider distance to selected chunks' embeddings
        for sel_chunk in selected:
            if sel_chunk.id in embedding_map:
                sel_embedding = embedding_map[sel_chunk.id].embedding
                distance = np.linalg.norm(chunk_embedding - sel_embedding)
                min_distance = min(min_distance, distance)
        
        # Convert distance to coverage gain
        return min_distance
    
    def _calculate_enhanced_total_coverage(
        self,
        selected: List[Chunk],
        mixed_centroids: Dict[str, np.ndarray],
        embedding_map: Dict[str, Any]
    ) -> float:
        """
        Calculate total coverage score using mixed centroids.
        
        Enhanced version that considers both selected chunks and
        how well they cover the centroid space.
        """
        if not selected or not mixed_centroids:
            return 0.0
        
        # Get selected embeddings
        selected_embeddings = []
        for chunk in selected:
            if chunk.id in embedding_map:
                selected_embeddings.append(embedding_map[chunk.id].embedding)
        
        if not selected_embeddings:
            return 0.0
        
        # Calculate coverage of centroid space
        covered_centroids = 0
        total_centroids = len(mixed_centroids)
        
        for centroid_id, centroid in mixed_centroids.items():
            # Find minimum distance from centroid to any selected embedding
            min_distance = float('inf')
            for sel_emb in selected_embeddings:
                distance = np.linalg.norm(centroid - sel_emb)
                min_distance = min(min_distance, distance)
            
            # Consider centroid "covered" if within reasonable distance
            if min_distance < 1.0:  # Threshold for coverage
                covered_centroids += 1
        
        return covered_centroids / max(total_centroids, 1)
    
    def get_clustering_metadata(self):
        """
        Get V2 clustering metadata for pack index storage.
        
        Returns tuple of (clustering_result, embedding_provider) or (None, None)
        if V2 was not used or failed.
        """
        clustering_result = getattr(self, '_clustering_result', None)
        embedding_provider = getattr(self, '_embedding_provider_ref', None)
        return clustering_result, embedding_provider
    
    # V3 Demotion Stability Methods
    
    def _init_v3_components(self, config: SelectionConfig):
        """
        Initialize V3 demotion stability components.
        
        Sets up the demotion controller, stability tracker, budget reallocator,
        and utility calculator for V3 operations.
        """
        try:
            from ..controller import DemotionController, StabilityTracker
            from ..controller.utility import V3UtilityCalculator
            from ..budget import BudgetReallocator
            
            # Initialize utility calculator
            self._v3_utility_calculator = V3UtilityCalculator()
            
            # Initialize stability tracker
            self._v3_stability_tracker = StabilityTracker(max_history_size=1000)
            
            # Initialize demotion controller
            self._v3_demotion_controller = DemotionController(
                utility_calculator=self._v3_utility_calculator,
                stability_tracker=self._v3_stability_tracker
            )
            
            # Initialize budget reallocator
            self._v3_budget_reallocator = BudgetReallocator(
                utility_calculator=self._v3_utility_calculator
            )
            
            # Track previous selections for oscillation detection
            self._v3_selection_history = []
            
            # Mark as initialized
            self._v3_components_initialized = True
            
            logger.info("V3 demotion stability components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize V3 components: {e}")
            raise
    
    def _apply_demotion_stability(
        self,
        base_result: SelectionResult,
        chunks: List[Chunk],
        config: SelectionConfig
    ) -> SelectionResult:
        """
        Apply V3 demotion stability controller to base selection result.
        
        Implements the core V3 algorithm:
        1. Detect demotion candidates based on utility thresholds
        2. Execute bounded re-optimization with single corrective step
        3. Check oscillation constraints and update ban lists
        4. Return stabilized selection result
        
        Args:
            base_result: Base selection from V1/V2
            chunks: All available chunks
            config: Selection configuration
            
        Returns:
            Stabilized selection result with demotion tracking
        """
        logger.info("Applying V3 demotion stability controller")
        
        # Create chunks lookup for efficient access
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        
        # Step 1: Detect demotion candidates
        demotion_candidates = self._v3_demotion_controller.detect_demotion_candidates(
            base_result, chunks_by_id, config, budget_pressure_threshold=0.95
        )
        
        if not demotion_candidates:
            logger.info("No demotion candidates detected, returning base result")
            self._v3_selection_history.append(base_result)
            return base_result
        
        logger.info(f"Detected {len(demotion_candidates)} demotion candidates")
        
        # Step 2: Execute bounded re-optimization
        stable_result, corrective_actions = self._v3_demotion_controller.execute_bounded_reoptimization(
            demotion_candidates, base_result, chunks_by_id, config, max_corrective_steps=1
        )
        
        # Step 3: Check oscillation constraints
        constraint_satisfied, oscillations = self._v3_demotion_controller.check_oscillation_constraints(
            stable_result, self._v3_selection_history, max_oscillations=1
        )
        
        if not constraint_satisfied:
            logger.warning(f"Oscillation constraints violated with {len(oscillations)} oscillations")
            # Apply additional stabilization if needed
            stable_result = self._apply_additional_stabilization(
                stable_result, oscillations, chunks_by_id, config
            )
        
        # Step 4: Advance epoch and update history
        self._v3_demotion_controller.advance_epoch()
        self._v3_selection_history.append(stable_result)
        
        # Keep only recent history to prevent memory growth
        if len(self._v3_selection_history) > 10:
            self._v3_selection_history = self._v3_selection_history[-10:]
        
        # Step 5: Log demotion analytics
        self._log_v3_analytics(demotion_candidates, corrective_actions, stable_result)
        
        logger.info(
            f"V3 demotion stability applied: {len(demotion_candidates)} demotions, "
            f"{len(corrective_actions)} corrective actions, "
            f"constraint_satisfied={constraint_satisfied}"
        )
        
        return stable_result
    
    def _apply_additional_stabilization(
        self,
        result: SelectionResult,
        oscillations: List,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig
    ) -> SelectionResult:
        """
        Apply additional stabilization measures when oscillation constraints are violated.
        
        This method handles cases where the initial demotion process still results
        in oscillations, applying more aggressive stabilization.
        """
        logger.info(f"Applying additional stabilization for {len(oscillations)} oscillations")
        
        # Create a copy of the result to modify
        stabilized_result = SelectionResult(
            selected_chunks=result.selected_chunks.copy(),
            chunk_modes=result.chunk_modes.copy(),
            selection_scores=result.selection_scores.copy(),
            total_tokens=result.total_tokens,
            budget_utilization=result.budget_utilization,
            coverage_score=result.coverage_score,
            diversity_score=result.diversity_score,
            iterations=result.iterations,
            demoted_chunks=result.demoted_chunks.copy(),
        )
        
        # For each oscillating chunk, apply aggressive demotion
        tokens_freed = 0
        
        for oscillation in oscillations:
            chunk_id = oscillation.chunk_id
            if chunk_id in chunks_by_id and chunk_id in stabilized_result.chunk_modes:
                current_mode = stabilized_result.chunk_modes[chunk_id]
                
                # Force demotion to signature mode to stabilize
                if current_mode == 'full':
                    chunk = chunks_by_id[chunk_id]
                    old_tokens = chunk.full_tokens
                    new_tokens = chunk.signature_tokens
                    tokens_freed += (old_tokens - new_tokens)
                    
                    stabilized_result.chunk_modes[chunk_id] = 'signature'
                    stabilized_result.demoted_chunks[chunk_id] = current_mode
                    stabilized_result.total_tokens -= (old_tokens - new_tokens)
                    
                    # Ban chunk for extended period
                    self._v3_demotion_controller.stability_tracker.add_to_ban_list(
                        chunk_id, 
                        self._v3_demotion_controller._current_epoch + 3
                    )
                    
                    logger.info(f"Applied aggressive stabilization: demoted {chunk_id} to signature")
        
        # Recalculate budget utilization
        if stabilized_result.total_tokens > 0:
            stabilized_result.budget_utilization = stabilized_result.total_tokens / max(1, config.token_budget)
        
        # Use freed tokens for corrective actions if beneficial
        if tokens_freed > 0:
            logger.info(f"Additional stabilization freed {tokens_freed} tokens")
            # Could add corrective selection here if needed
        
        return stabilized_result
    
    def _log_v3_analytics(
        self,
        demotion_candidates: List,
        corrective_actions: List[Dict[str, Any]],
        final_result: SelectionResult
    ):
        """Log V3 analytics for debugging and monitoring."""
        logger.info("V3 Analytics Summary:")
        logger.info(f"  Demotion candidates: {len(demotion_candidates)}")
        logger.info(f"  Corrective actions: {len(corrective_actions)}")
        logger.info(f"  Total demotions: {len(final_result.demoted_chunks)}")
        logger.info(f"  Final token usage: {final_result.total_tokens}")
        logger.info(f"  Budget utilization: {final_result.budget_utilization:.2%}")
        
        # Log demotion breakdown by strategy
        if demotion_candidates:
            from collections import Counter
            strategy_counts = Counter(d.strategy.value for d in demotion_candidates)
            logger.info(f"  Demotion strategies: {dict(strategy_counts)}")
        
        # Log performance metrics
        if hasattr(self, '_v3_demotion_controller'):
            metrics = self._v3_demotion_controller.get_performance_metrics()
            logger.info(f"  Controller metrics: {metrics}")
    
    def get_v3_analytics(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive V3 analytics and optionally export to CSV.
        
        Args:
            output_path: Optional path to export demotions.csv
            
        Returns:
            Dictionary with V3 analytics summary
        """
        if not hasattr(self, '_v3_demotion_controller'):
            return {'error': 'V3 components not initialized'}
        
        analytics = {
            'controller_metrics': self._v3_demotion_controller.get_performance_metrics(),
            'stability_metrics': self._v3_stability_tracker.get_metrics(),
            'utility_metrics': self._v3_utility_calculator.get_performance_metrics(),
            'budget_metrics': self._v3_budget_reallocator.get_performance_metrics(),
            'selection_history_size': len(getattr(self, '_v3_selection_history', [])),
        }
        
        # Export detailed demotion CSV if path provided
        if output_path:
            export_summary = self._v3_demotion_controller.export_demotion_analytics(output_path)
            analytics['export_summary'] = export_summary
        
        return analytics
    
    def clear_v3_history(self):
        """Clear V3 history and caches (useful for testing)."""
        if hasattr(self, '_v3_selection_history'):
            self._v3_selection_history.clear()
        
        if hasattr(self, '_v3_utility_calculator'):
            self._v3_utility_calculator.clear_cache()
        
        logger.info("Cleared V3 history and caches")
    
    # Baseline Variant Integration Methods
    
    def _run_baseline_variant(self, chunks: List[Chunk], config: SelectionConfig) -> SelectionResult:
        """
        Run baseline variant using dedicated baseline selectors.
        
        Maps SelectionVariant enum values to baseline selector implementations.
        """
        # Map variant enum to baseline ID
        variant_mapping = {
            SelectionVariant.V0A_README_ONLY: "V0a",
            SelectionVariant.V0B_NAIVE_CONCAT: "V0b", 
            SelectionVariant.V0C_BM25_BASELINE: "V0c",
        }
        
        baseline_id = variant_mapping.get(config.variant)
        if not baseline_id:
            raise ValueError(f"Unknown baseline variant: {config.variant}")
        
        return self._run_baseline_variant_by_id(chunks, config, baseline_id)
    
    def _run_baseline_variant_by_id(self, chunks: List[Chunk], config: SelectionConfig, baseline_id: str) -> SelectionResult:
        """
        Run baseline variant by ID using baseline selector implementations.
        
        Args:
            chunks: Chunks to select from
            config: Selection configuration
            baseline_id: Baseline ID ('V0a', 'V0b', 'V0c')
            
        Returns:
            Selection result from baseline algorithm
        """
        try:
            from ..baselines import create_baseline_selector
            from ..baselines.base import BaselineConfig
            
            # Create baseline selector
            baseline_selector = create_baseline_selector(baseline_id)
            
            # Convert SelectionConfig to BaselineConfig
            baseline_config = BaselineConfig(
                token_budget=config.token_budget,
                deterministic=config.deterministic,
                random_seed=config.random_seed,
                min_doc_density=config.min_doc_density,
                max_complexity=config.max_complexity,
            )
            
            # Run baseline selection
            result = baseline_selector.select(chunks, baseline_config)
            
            logger.info(f"Baseline {baseline_id} completed: {len(result.selected_chunks)} chunks, {result.total_tokens} tokens")
            
            return result
            
        except ImportError as e:
            logger.error(f"Failed to import baseline selectors: {e}")
            # Fall back to simple baseline selection
            return self._baseline_selection(chunks, config)
        except Exception as e:
            logger.error(f"Baseline {baseline_id} failed: {e}")
            # Fall back to simple baseline selection
            return self._baseline_selection(chunks, config)
    
    def _density_greedy_selection(
        self, 
        chunks: List[Chunk], 
        config: SelectionConfig
    ) -> SelectionResult:
        """
        Density-greedy selection algorithm (V2 FastPath enhancement).
        
        Implements the density-greedy algorithm that prioritizes chunks with
        high information density per token while respecting class quotas.
        """
        flags = get_feature_flags()
        if not flags.policy_v2:
            # Fall back to V1 algorithm if V2 policy not enabled
            return self._facility_location_mmr(chunks, config)
        
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        # Reset class quotas for this selection
        class_counts = defaultdict(int)
        max_per_class = max(1, len(chunks) // 20)  # Limit each class to 5% of total
        
        # Sort chunks by density score (importance per token)
        chunk_densities = []
        for chunk in chunks:
            # Calculate information density
            base_score = chunk.centrality_score if hasattr(chunk, 'centrality_score') else 1.0
            density = base_score / max(1, chunk.full_tokens)
            chunk_densities.append((density, chunk))
        
        # Sort by density descending
        chunk_densities.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy selection with class quotas and budget validation
        for density, chunk in chunk_densities:
            if total_tokens >= config.token_budget:
                break
            
            iterations += 1
            
            # Check class quota constraints
            chunk_class = chunk.kind.value if chunk.kind else 'unknown'
            if class_counts[chunk_class] >= max_per_class:
                continue  # Skip if class quota exceeded
            
            # Try different modes
            best_mode = None
            best_tokens = float('inf')
            
            for mode in ["signature", "full"]:  # Prefer signature mode for density
                tokens = self._get_chunk_tokens(chunk, mode)
                
                # Validate budget with safety margin
                if self._runtime_budget_validator:
                    if not self._runtime_budget_validator(total_tokens + tokens, config.token_budget):
                        continue
                elif total_tokens + tokens > config.token_budget:
                    continue
                
                if tokens < best_tokens:
                    best_tokens = tokens
                    best_mode = mode
            
            if best_mode:
                selected.append(chunk)
                selected_ids.add(chunk.id)
                chunk_modes[chunk.id] = best_mode
                selection_scores[chunk.id] = density
                total_tokens += best_tokens
                class_counts[chunk_class] += 1
        
        # Calculate quality metrics
        coverage_score = len(selected) / max(1, len(chunks))
        diversity_score = self._calculate_class_diversity(selected)
        
        # Sort deterministically if requested
        if config.deterministic and selected:
            selected.sort(key=lambda c: self._get_deterministic_sort_key(c))
        
        return SelectionResult(
            selected_chunks=selected,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / config.token_budget,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            iterations=iterations,
        )
    
    def _calculate_class_diversity(self, chunks: List[Chunk]) -> float:
        """Calculate diversity score based on chunk class distribution."""
        if not chunks:
            return 0.0
        
        class_counts = defaultdict(int)
        for chunk in chunks:
            chunk_class = chunk.kind.value if chunk.kind else 'unknown'
            class_counts[chunk_class] += 1
        
        # Calculate Shannon diversity index
        total = len(chunks)
        diversity = 0.0
        for count in class_counts.values():
            if count > 0:
                p = count / total
                diversity -= p * math.log2(p)
        
        # Normalize by maximum possible diversity (log2 of number of classes)
        max_diversity = math.log2(min(len(class_counts), 6))  # Max 6 chunk classes
        return diversity / max_diversity if max_diversity > 0 else 0.0
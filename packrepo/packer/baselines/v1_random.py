"""V1: Random Baseline Implementation."""

from __future__ import annotations

import logging
import random
from typing import List, Dict, Any

from .base import BaselineSelector, BaselineConfig
from ..chunker.base import Chunk
from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class V1RandomBaseline(BaselineSelector):
    """
    V1: Random Baseline - Pure random selection for lower bound comparison.
    
    Randomly selects files until budget exhausted with uniform random selection
    across all repository files. Provides reproducible results with configurable
    seed for research comparison purposes.
    
    This represents the weakest possible baseline - purely random file selection
    without any semantic understanding or heuristics.
    """
    
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier."""
        return "V1"
    
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        return "Random selection baseline"
    
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select chunks randomly until budget exhausted.
        
        Selection strategy:
        1. Shuffle all chunks with deterministic seed
        2. Select chunks sequentially until budget full
        3. Use signature mode if full content exceeds budget
        4. Ensure reproducible results with config.random_seed
        
        Args:
            chunks: Preprocessed chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with randomly selected chunks
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        # Create a copy for shuffling (don't modify original)
        chunk_pool = chunks.copy()
        
        # Shuffle with deterministic seed for reproducibility
        random.seed(config.random_seed)
        random.shuffle(chunk_pool)
        
        logger.info(f"V1: Starting random selection from {len(chunk_pool)} chunks (seed={config.random_seed})")
        
        # Select chunks randomly until budget exhausted
        for chunk in chunk_pool:
            if total_tokens >= config.token_budget:
                break
                
            iterations += 1
            
            # Random score for tracking (helps with debugging)
            score = random.random()
            
            # Try full mode first
            tokens = self._get_chunk_tokens(chunk, "full")
            
            # Hard budget constraint: zero overflow allowed
            if total_tokens + tokens > config.token_budget:
                # Try signature mode for large files
                signature_tokens = self._get_chunk_tokens(chunk, "signature")
                if total_tokens + signature_tokens <= config.token_budget:
                    selected.append(chunk)
                    selected_ids.add(chunk.id)
                    chunk_modes[chunk.id] = "signature"
                    selection_scores[chunk.id] = score
                    total_tokens += signature_tokens
                    
                    logger.debug(f"V1: Using signature mode for {chunk.rel_path} ({signature_tokens} tokens)")
                else:
                    # Skip if even signature doesn't fit
                    logger.debug(f"V1: Skipping {chunk.rel_path} - too large even in signature mode")
                continue
            
            # Add chunk in full mode
            selected.append(chunk)
            selected_ids.add(chunk.id)
            chunk_modes[chunk.id] = "full"
            selection_scores[chunk.id] = score
            total_tokens += tokens
            
            logger.debug(f"V1: Selected {chunk.rel_path} ({tokens} tokens, score={score:.3f})")
        
        # Calculate simple metrics
        coverage_score = len(selected) / max(1, len(chunks))
        diversity_score = len(set(chunk.rel_path for chunk in selected)) / max(1, len(selected))
        
        # Ensure deterministic ordering if requested (sort by original order)
        if config.deterministic and selected:
            # Sort by chunk ID to maintain deterministic order
            selected.sort(key=lambda c: c.id)
        
        # Create result
        result = SelectionResult(
            selected_chunks=selected,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / config.token_budget,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            iterations=iterations,
        )
        
        logger.info(
            f"V1 completed: {len(selected)} chunks, {total_tokens} tokens, "
            f"{result.budget_utilization:.1%} budget utilization"
        )
        
        return result
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate V1-specific constraints."""
        errors = []
        
        # V1 should have no systematic bias - just check for basic sanity
        if len(result.selected_chunks) == 0 and len(result.selected_chunks) > 0:
            errors.append("V1 selected no chunks despite available budget")
        
        # Check for reasonable diversity (random should hit many files)
        if result.selected_chunks:
            file_count = len(set(chunk.rel_path for chunk in result.selected_chunks))
            chunk_count = len(result.selected_chunks)
            
            # If we have many chunks but very few files, might indicate an issue
            if chunk_count >= 20 and file_count <= 3:
                errors.append(
                    f"V1 selected {chunk_count} chunks from only {file_count} files - "
                    "possible bias in random selection"
                )
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get V1-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        base_metrics.update({
            "selection_strategy": "uniform_random",
            "supports_signature_fallback": True,
            "theoretical_bias": "none",
            "expected_performance": "worst_case_lower_bound",
        })
        return base_metrics
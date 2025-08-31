"""V2: Recency Baseline Implementation."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaselineSelector, BaselineConfig
from ..chunker.base import Chunk
from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class V2RecencyBaseline(BaselineSelector):
    """
    V2: Recency Baseline - Select files by most recent modification time.
    
    Uses git log timestamps to order files by recency, showing temporal bias
    in file selection. This represents the assumption that recently modified
    files are more important or relevant for understanding a repository.
    
    This baseline tests whether recency is a useful signal for code selection,
    as recently modified files might contain the most active development.
    """
    
    def __init__(self, tokenizer=None):
        """Initialize recency baseline with git timestamp caching."""
        super().__init__(tokenizer)
        self._timestamp_cache: Dict[str, float] = {}
    
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier."""
        return "V2"
    
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        return "Recency-based selection baseline"
    
    def _get_file_timestamp(self, file_path: str) -> float:
        """
        Get the last modification timestamp for a file using git log.
        
        Args:
            file_path: Relative path to file
            
        Returns:
            Unix timestamp of last modification (0.0 if unable to determine)
        """
        if file_path in self._timestamp_cache:
            return self._timestamp_cache[file_path]
        
        try:
            # Use git log to get the last modification timestamp
            cmd = [
                "git", "log", "-1", "--format=%ct", "--follow", "--", file_path
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                timestamp = float(result.stdout.strip())
                self._timestamp_cache[file_path] = timestamp
                return timestamp
            else:
                logger.debug(f"V2: Failed to get git timestamp for {file_path}: {result.stderr}")
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            logger.debug(f"V2: Git command failed for {file_path}: {e}")
        
        # Fallback to file system timestamp
        try:
            full_path = Path.cwd() / file_path
            if full_path.exists():
                timestamp = full_path.stat().st_mtime
                self._timestamp_cache[file_path] = timestamp
                return timestamp
        except OSError as e:
            logger.debug(f"V2: Failed to get filesystem timestamp for {file_path}: {e}")
        
        # Ultimate fallback - use current time minus path hash for deterministic ordering
        fallback_timestamp = datetime.now().timestamp() - hash(file_path) % 86400
        self._timestamp_cache[file_path] = fallback_timestamp
        return fallback_timestamp
    
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select chunks ordered by most recent modification time.
        
        Selection strategy:
        1. Get git log timestamp for each file
        2. Sort chunks by timestamp (most recent first)
        3. Select chunks sequentially until budget full
        4. Use signature mode if full content exceeds budget
        
        Args:
            chunks: Preprocessed chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with recency-ordered chunks
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        logger.info(f"V2: Starting recency-based selection from {len(chunks)} chunks")
        
        # Calculate recency scores for all chunks
        chunk_timestamps = []
        for chunk in chunks:
            timestamp = self._get_file_timestamp(chunk.rel_path)
            chunk_timestamps.append((chunk, timestamp))
        
        # Sort by timestamp (most recent first)
        chunk_timestamps.sort(key=lambda x: -x[1])  # Negative for descending order
        
        # Log some timestamp info for debugging
        if chunk_timestamps:
            newest_time = datetime.fromtimestamp(chunk_timestamps[0][1])
            oldest_time = datetime.fromtimestamp(chunk_timestamps[-1][1])
            logger.info(f"V2: File timestamps range from {oldest_time} to {newest_time}")
        
        # Select chunks in recency order until budget exhausted
        for chunk, timestamp in chunk_timestamps:
            if total_tokens >= config.token_budget:
                break
                
            iterations += 1
            
            # Score is normalized timestamp (higher = more recent)
            max_timestamp = chunk_timestamps[0][1] if chunk_timestamps else timestamp
            min_timestamp = chunk_timestamps[-1][1] if len(chunk_timestamps) > 1 else timestamp
            
            if max_timestamp > min_timestamp:
                score = (timestamp - min_timestamp) / (max_timestamp - min_timestamp)
            else:
                score = 1.0  # All files have same timestamp
            
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
                    
                    file_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    logger.debug(f"V2: Using signature mode for {chunk.rel_path} (modified {file_time})")
                else:
                    # Skip if even signature doesn't fit
                    logger.debug(f"V2: Skipping {chunk.rel_path} - too large even in signature mode")
                continue
            
            # Add chunk in full mode
            selected.append(chunk)
            selected_ids.add(chunk.id)
            chunk_modes[chunk.id] = "full"
            selection_scores[chunk.id] = score
            total_tokens += tokens
            
            file_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            logger.debug(f"V2: Selected {chunk.rel_path} (modified {file_time}, {tokens} tokens)")
        
        # Calculate metrics
        coverage_score = len(selected) / max(1, len(chunks))
        diversity_score = len(set(chunk.rel_path for chunk in selected)) / max(1, len(selected))
        
        # Ensure deterministic ordering if requested
        if config.deterministic and selected:
            # Sort by timestamp then by path for deterministic ordering
            selected.sort(key=lambda c: (
                -self._timestamp_cache.get(c.rel_path, 0),  # Most recent first
                c.rel_path,  # Then by path
                c.id  # Finally by chunk ID
            ))
        
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
            f"V2 completed: {len(selected)} chunks, {total_tokens} tokens, "
            f"{result.budget_utilization:.1%} budget utilization"
        )
        
        return result
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate V2-specific constraints."""
        errors = []
        
        # V2 should show temporal bias - check that selection has reasonable recency ordering
        if len(result.selected_chunks) >= 2:
            timestamps = []
            for chunk in result.selected_chunks:
                timestamp = self._timestamp_cache.get(chunk.rel_path)
                if timestamp is not None:
                    timestamps.append(timestamp)
            
            if len(timestamps) >= 2:
                # Check if timestamps are generally in descending order (allowing some flexibility)
                ascending_pairs = 0
                descending_pairs = 0
                
                for i in range(len(timestamps) - 1):
                    if timestamps[i] > timestamps[i + 1]:
                        descending_pairs += 1
                    elif timestamps[i] < timestamps[i + 1]:
                        ascending_pairs += 1
                
                # Expect at least 60% of pairs to be in descending order for recency bias
                if descending_pairs + ascending_pairs > 0:
                    descending_ratio = descending_pairs / (descending_pairs + ascending_pairs)
                    if descending_ratio < 0.6:
                        errors.append(
                            f"V2 selection doesn't show strong recency bias: "
                            f"only {descending_ratio:.1%} of adjacent pairs are in descending time order"
                        )
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get V2-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        base_metrics.update({
            "selection_strategy": "recency_based",
            "supports_signature_fallback": True,
            "timestamp_source": "git_log_with_fallback",
            "cache_size": len(self._timestamp_cache),
            "theoretical_bias": "temporal_recency",
        })
        return base_metrics
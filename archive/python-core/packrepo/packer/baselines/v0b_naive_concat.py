"""V0b: Naive Concatenation Baseline Implementation."""

from __future__ import annotations

import logging
from typing import List, Dict, Any
from pathlib import Path

from .base import BaselineSelector, BaselineConfig
from ..chunker.base import Chunk
from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class V0bNaiveConcatBaseline(BaselineSelector):
    """
    V0b: Naive Concatenation Baseline - File size ordering.
    
    Simple baseline that concatenates files by size order until
    budget is exhausted. This represents a basic "grab the largest
    files first" approach without any semantic understanding.
    
    Selection strategy:
    1. Sort all files by size (largest first)
    2. Include files sequentially until budget exhausted
    3. No semantic analysis or prioritization
    """
    
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier."""
        return "V0b"
    
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        return "Naive concatenation by file size"
    
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select chunks by naive file size ordering.
        
        Strategy:
        1. Group chunks by file
        2. Calculate file sizes (sum of chunk tokens)
        3. Sort files by total size (descending)
        4. Include files sequentially until budget exhausted
        5. Within each file, include all chunks or none
        
        Args:
            chunks: Preprocessed chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with size-ordered selection
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        # Group chunks by file path
        files_by_path = {}
        for chunk in chunks:
            if chunk.rel_path not in files_by_path:
                files_by_path[chunk.rel_path] = []
            files_by_path[chunk.rel_path].append(chunk)
        
        # Calculate file sizes and create file candidates
        file_candidates = []
        for file_path, file_chunks in files_by_path.items():
            # Sort chunks within file deterministically
            file_chunks.sort(key=lambda c: (c.start_line, c.id))
            
            # Calculate total tokens for file
            file_tokens_full = sum(self._get_chunk_tokens(chunk, "full") for chunk in file_chunks)
            file_tokens_signature = sum(self._get_chunk_tokens(chunk, "signature") for chunk in file_chunks)
            
            file_candidates.append({
                'path': file_path,
                'chunks': file_chunks,
                'tokens_full': file_tokens_full,
                'tokens_signature': file_tokens_signature,
                'chunk_count': len(file_chunks)
            })
        
        # Sort files by size (descending) with deterministic tie-breaking
        file_candidates.sort(key=lambda f: (-f['tokens_full'], f['path']))
        
        logger.info(f"V0b: Processing {len(file_candidates)} files")
        
        # Select files sequentially until budget exhausted
        for file_info in file_candidates:
            if total_tokens >= config.token_budget:
                break
                
            iterations += 1
            file_path = file_info['path']
            file_chunks = file_info['chunks']
            
            # Try full mode first
            full_tokens = file_info['tokens_full']
            signature_tokens = file_info['tokens_signature']
            
            selected_mode = None
            selected_tokens = 0
            
            # Check if full file fits
            if total_tokens + full_tokens <= config.token_budget:
                selected_mode = "full"
                selected_tokens = full_tokens
            # Try signature mode
            elif total_tokens + signature_tokens <= config.token_budget:
                selected_mode = "signature"
                selected_tokens = signature_tokens
            else:
                # File is too large even in signature mode
                logger.warning(f"V0b: Skipping {file_path} - too large ({full_tokens} full, {signature_tokens} sig)")
                continue
            
            # Add all chunks from this file
            for chunk in file_chunks:
                selected.append(chunk)
                selected_ids.add(chunk.id)
                chunk_modes[chunk.id] = selected_mode
                selection_scores[chunk.id] = full_tokens  # Score by file size
            
            total_tokens += selected_tokens
            
            logger.debug(
                f"V0b: Selected {file_path} ({selected_mode} mode, "
                f"{selected_tokens} tokens, {len(file_chunks)} chunks)"
            )
        
        # Calculate basic metrics
        coverage_score = len(selected) / max(1, len(chunks))  # Chunk coverage
        file_coverage = len(set(chunk.rel_path for chunk in selected)) / max(1, len(files_by_path))
        diversity_score = file_coverage  # Use file coverage as diversity
        
        # Ensure deterministic ordering if requested
        if config.deterministic and selected:
            # Sort by: file path, then start line, then chunk id
            selected.sort(key=lambda c: (c.rel_path, c.start_line, c.id))
        
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
            f"V0b completed: {len(selected)} chunks from {len(set(chunk.rel_path for chunk in selected))} files, "
            f"{total_tokens} tokens, {result.budget_utilization:.1%} budget utilization"
        )
        
        return result
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate V0b-specific constraints."""
        errors = []
        
        # V0b should maintain file integrity (all chunks from a file or none)
        file_chunk_modes = {}
        for chunk in result.selected_chunks:
            file_path = chunk.rel_path
            chunk_mode = result.chunk_modes.get(chunk.id, "full")
            
            if file_path not in file_chunk_modes:
                file_chunk_modes[file_path] = set()
            file_chunk_modes[file_path].add(chunk_mode)
        
        # Check that all chunks from the same file use the same mode
        mixed_mode_files = []
        for file_path, modes in file_chunk_modes.items():
            if len(modes) > 1:
                mixed_mode_files.append(file_path)
        
        if mixed_mode_files:
            errors.append(f"V0b has mixed modes within files: {mixed_mode_files[:3]}")
        
        # V0b should prefer larger files (check if selection scores reflect file sizes)
        if result.selection_scores:
            scores = list(result.selection_scores.values())
            if len(scores) > 1:
                # Check if scores are generally decreasing (allowing for same-file chunks)
                file_scores = {}
                for chunk in result.selected_chunks:
                    file_path = chunk.rel_path
                    score = result.selection_scores.get(chunk.id, 0.0)
                    if file_path not in file_scores:
                        file_scores[file_path] = score
                
                sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
                for i in range(1, len(sorted_files)):
                    if sorted_files[i][1] > sorted_files[i-1][1]:
                        errors.append("V0b selection scores not in descending order by file size")
                        break
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get V0b-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        base_metrics.update({
            "selection_strategy": "size_ordered_concatenation",
            "file_level_selection": True,
            "maintains_file_integrity": True,
            "supports_signature_fallback": True,
            "sorting_criterion": "descending_file_size",
        })
        return base_metrics
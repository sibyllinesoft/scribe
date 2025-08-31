"""Base class for baseline selection algorithms."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass

from ..chunker.base import Chunk
from ..selector.base import SelectionResult, SelectionConfig
from ..tokenizer.base import Tokenizer


@dataclass
class BaselineConfig:
    """Configuration for baseline selection algorithms."""
    
    # Budget constraints
    token_budget: int = 120000
    allow_overage: float = 0.005  # Max 0.5% under-budget allowed
    
    # Deterministic settings
    random_seed: int = 42
    deterministic: bool = True
    
    # Quality constraints (minimal for baselines)
    min_doc_density: float = 0.0
    max_complexity: float = float('inf')  # No complexity limit for baselines
    
    def to_selection_config(self) -> SelectionConfig:
        """Convert to SelectionConfig for compatibility."""
        from ..selector.base import SelectionConfig, SelectionMode, SelectionVariant
        
        return SelectionConfig(
            mode=SelectionMode.COMPREHENSION,
            variant=SelectionVariant.BASELINE,
            token_budget=self.token_budget,
            allow_overage=self.allow_overage,
            random_seed=self.random_seed,
            deterministic=self.deterministic,
            min_doc_density=self.min_doc_density,
            max_complexity=self.max_complexity,
        )


class BaselineSelector(ABC):
    """
    Abstract base class for baseline selection algorithms.
    
    Provides common functionality and interface for V0a, V0b, V0c baselines
    as specified in TODO.md requirements.
    """
    
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """Initialize baseline selector."""
        self.tokenizer = tokenizer
        self._chunk_cache: Dict[str, int] = {}
    
    @abstractmethod
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier (V0a, V0b, V0c)."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        pass
    
    def select(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select chunks according to baseline algorithm.
        
        Args:
            chunks: Available chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with baseline metrics
        """
        start_time = time.time()
        
        # Preprocess chunks
        processed_chunks = self._preprocess_chunks(chunks, config)
        
        # Run baseline-specific selection
        selection_result = self._select_baseline(processed_chunks, config)
        
        # Post-process with validation
        selection_result = self._postprocess_selection(selection_result, config)
        
        # Add execution time
        execution_time = time.time() - start_time
        selection_result.execution_time = execution_time
        
        return selection_result
    
    @abstractmethod
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """Implement baseline-specific selection algorithm."""
        pass
    
    def _preprocess_chunks(self, chunks: List[Chunk], config: BaselineConfig) -> List[Chunk]:
        """Preprocess chunks with basic filtering and deterministic ordering."""
        processed = []
        
        for chunk in chunks:
            # Apply minimal quality filters
            if chunk.doc_density < config.min_doc_density:
                continue
            if chunk.complexity_score > config.max_complexity:
                continue
                
            # Generate deterministic chunk ID if needed
            if config.deterministic:
                chunk.id = self._generate_deterministic_chunk_id(chunk)
            
            processed.append(chunk)
        
        # Sort deterministically by file path and line number
        if config.deterministic:
            processed.sort(key=lambda c: (c.rel_path, c.start_line, c.id))
        
        return processed
    
    def _generate_deterministic_chunk_id(self, chunk: Chunk) -> str:
        """Generate deterministic chunk ID based on content and metadata."""
        # Create stable identifier from chunk properties
        id_components = [
            chunk.rel_path,
            str(chunk.start_line),
            str(chunk.end_line),
            chunk.kind.value if chunk.kind else 'unknown',
            # Use first 100 chars of content for uniqueness
            (chunk.content or '')[:100]
        ]
        
        id_string = '|'.join(id_components)
        chunk_hash = hashlib.sha256(id_string.encode('utf-8')).hexdigest()
        
        # Return human-readable ID with hash suffix
        path_stem = chunk.rel_path.replace('/', '_').replace('.', '_')
        return f"{path_stem}_{chunk.start_line}_{chunk_hash[:8]}"
    
    def _get_chunk_tokens(self, chunk: Chunk, mode: str = "full") -> int:
        """Get token count for chunk in specified mode."""
        cache_key = f"{chunk.id}:{mode}"
        
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        
        if mode == "full":
            tokens = chunk.full_tokens
        elif mode == "signature":
            tokens = chunk.signature_tokens
        elif mode == "summary":
            tokens = chunk.summary_tokens if chunk.summary_tokens else chunk.signature_tokens
        else:
            tokens = chunk.full_tokens
        
        # Use tokenizer if available for more accurate counting
        if self.tokenizer and chunk.content:
            try:
                tokens = self.tokenizer.count_tokens(chunk.content)
            except Exception:
                # Fall back to chunk's token count
                pass
        
        self._chunk_cache[cache_key] = tokens
        return tokens
    
    def _postprocess_selection(self, result: SelectionResult, config: BaselineConfig) -> SelectionResult:
        """Post-process selection result with validation."""
        errors = []
        
        # Budget constraint validation (hard limits)
        if result.total_tokens > config.token_budget:
            errors.append(f"Budget overflow: {result.total_tokens} > {config.token_budget} tokens")
        
        # Underflow validation (should use budget reasonably)
        if config.token_budget > 0:
            utilization = result.total_tokens / config.token_budget
            # Only flag severe underflow for large budgets
            if config.token_budget >= 10000 and utilization < 0.3:
                underflow = (config.token_budget - result.total_tokens) / config.token_budget
                errors.append(f"Severe underflow: {underflow:.1%} - consider optimizing selection")
        
        # Chunk validation
        if result.selected_chunks:
            chunk_ids = [chunk.id for chunk in result.selected_chunks]
            if len(chunk_ids) != len(set(chunk_ids)):
                errors.append("Duplicate chunk IDs in selection")
        
        # Baseline-specific validation
        baseline_errors = self._validate_baseline_constraints(result, config)
        errors.extend(baseline_errors)
        
        # Raise errors if any validation failed
        if errors:
            raise ValueError(f"Baseline validation failed: {'; '.join(errors)}")
        
        return result
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate baseline-specific constraints. Override in subclasses."""
        return []
    
    def _calculate_simple_coverage(self, selected: List[Chunk], all_chunks: List[Chunk]) -> float:
        """Calculate simple coverage score as ratio of selected to total chunks."""
        if not all_chunks:
            return 0.0
        return len(selected) / len(all_chunks)
    
    def _calculate_file_coverage(self, selected: List[Chunk], all_chunks: List[Chunk]) -> float:
        """Calculate file coverage as ratio of covered files to total files."""
        if not all_chunks:
            return 0.0
            
        selected_files = set(chunk.rel_path for chunk in selected)
        total_files = set(chunk.rel_path for chunk in all_chunks)
        
        return len(selected_files) / len(total_files)
    
    def _generate_deterministic_hash(self, result: SelectionResult) -> str:
        """Generate deterministic hash for result validation."""
        # Sort chunks by ID for deterministic hashing
        sorted_chunks = sorted(result.selected_chunks, key=lambda c: c.id)
        
        hash_input = ""
        for chunk in sorted_chunks:
            mode = result.chunk_modes.get(chunk.id, "full")
            score = result.selection_scores.get(chunk.id, 0.0)
            hash_input += f"{chunk.id}:{mode}:{score}"
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this baseline selector."""
        return {
            "variant_id": self.get_variant_id(),
            "description": self.get_description(),
            "cache_size": len(self._chunk_cache),
            "supports_deterministic": True,
            "supports_budget_enforcement": True,
        }
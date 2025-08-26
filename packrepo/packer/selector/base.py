"""Base classes and data structures for chunk selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set

from ..chunker.base import Chunk


class SelectionMode(Enum):
    """Selection mode for repository packing."""
    
    COMPREHENSION = "comprehension"  # Deterministic, no LLM
    OBJECTIVE = "objective"          # Query-driven with LLM


class SelectionVariant(Enum):
    """Selection algorithm variants."""
    
    # V0 Baseline variants as per TODO.md requirements
    V0A_README_ONLY = "v0a_readme_only"      # V0a: README-only minimal baseline
    V0B_NAIVE_CONCAT = "v0b_naive_concat"    # V0b: Naive concatenation by file size
    V0C_BM25_BASELINE = "v0c_bm25_baseline"  # V0c: BM25 + TF-IDF traditional IR baseline
    
    # Advanced variants
    COMPREHENSIVE = "comprehensive"          # V1: Facility-location + MMR + oracles  
    COVERAGE_ENHANCED = "coverage_enhanced"  # V2: + k-means + HNSW medoids
    STABILITY_CONTROLLED = "stability_controlled"  # V3: + demotion stability controller
    V4_OBJECTIVE = "v4_objective"            # V4: + query-biased selection
    V5_SUMMARIES = "v5_summaries"            # V5: + schema summaries
    
    # Legacy aliases for compatibility
    BASELINE = "v0c_bm25_baseline"           # Default baseline is V0c (strongest)
    V1_BASIC = "comprehensive" 
    V2_COVERAGE = "coverage_enhanced"
    V3_STABLE = "stability_controlled"


@dataclass
class SelectionConfig:
    """Configuration for chunk selection algorithms."""
    
    # Selection algorithm
    mode: SelectionMode = SelectionMode.COMPREHENSION
    variant: SelectionVariant = SelectionVariant.V1_BASIC
    
    # Budget constraints
    token_budget: int = 120000
    allow_overage: float = 0.005  # Max 0.5% under-budget allowed
    
    # Algorithm parameters
    diversity_weight: float = 0.3  # MMR diversity vs relevance
    coverage_weight: float = 0.7   # Coverage vs diversity balance
    demotion_threshold: float = 0.5  # Threshold for demoting full->signature
    
    # Priority boosts
    must_include_patterns: List[str] = field(default_factory=list)  # Regex patterns
    boost_manifests: float = 2.0        # Boost for package.json, requirements.txt, etc.
    boost_entrypoints: float = 1.5      # Boost for main.py, index.ts, etc.
    boost_tests: float = 0.8            # Reduce priority for test files
    
    # Quality constraints
    min_doc_density: float = 0.0        # Minimum documentation density
    max_complexity: float = 10.0        # Maximum complexity score
    
    # Objective mode specific
    objective_query: Optional[str] = None
    rerank_k: int = 64                  # Max items to rerank
    frontier_size: int = 256            # Max frontier size
    
    # Deterministic settings
    random_seed: int = 42
    deterministic: bool = True          # Enable --no-llm deterministic mode


@dataclass 
class PackRequest:
    """Request for repository packing."""
    
    chunks: List[Chunk]
    config: SelectionConfig
    
    # Repository metadata
    repo_path: Optional[str] = None
    commit_hash: Optional[str] = None
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks to select from."""
        return len(self.chunks)
    
    def get_total_tokens(self) -> int:
        """Get total tokens in all chunks (full mode)."""
        return sum(chunk.full_tokens for chunk in self.chunks)
    
    def filter_chunks(self, min_tokens: int = 1) -> List[Chunk]:
        """Filter chunks by minimum token count."""
        return [chunk for chunk in self.chunks if chunk.full_tokens >= min_tokens]


@dataclass
class SelectionResult:
    """Result from chunk selection algorithm."""
    
    selected_chunks: List[Chunk]
    chunk_modes: Dict[str, str]  # chunk_id -> mode (full/summary/signature)
    selection_scores: Dict[str, float]  # chunk_id -> selection score
    
    # Budget tracking
    total_tokens: int = 0
    budget_utilization: float = 0.0
    
    # Algorithm metrics
    coverage_score: float = 0.0
    diversity_score: float = 0.0
    iterations: int = 0
    
    # Demotion tracking
    demoted_chunks: Dict[str, str] = field(default_factory=dict)  # chunk_id -> original_mode
    
    def get_mode_breakdown(self) -> Dict[str, int]:
        """Get count of chunks by mode."""
        breakdown = {"full": 0, "summary": 0, "signature": 0}
        for mode in self.chunk_modes.values():
            breakdown[mode] = breakdown.get(mode, 0) + 1
        return breakdown
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection statistics."""
        return {
            "selected_chunks": len(self.selected_chunks),
            "total_tokens": self.total_tokens,
            "budget_utilization": self.budget_utilization,
            "coverage_score": self.coverage_score,
            "diversity_score": self.diversity_score,
            "iterations": self.iterations,
            "mode_breakdown": self.get_mode_breakdown(),
            "demoted_count": len(self.demoted_chunks),
        }


@dataclass
class PackResult:
    """Complete result from repository packing."""
    
    request: PackRequest
    selection: SelectionResult
    
    # Execution metadata
    execution_time: float = 0.0
    memory_peak: float = 0.0
    
    # Determinism validation
    deterministic_hash: Optional[str] = None
    
    def get_pack_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pack statistics."""
        stats = self.selection.get_selection_statistics()
        stats.update({
            "total_input_chunks": self.request.get_total_chunks(),
            "total_input_tokens": self.request.get_total_tokens(),
            "selection_ratio": len(self.selection.selected_chunks) / max(1, self.request.get_total_chunks()),
            "compression_ratio": self.selection.total_tokens / max(1, self.request.get_total_tokens()),
            "execution_time": self.execution_time,
            "memory_peak": self.memory_peak,
        })
        return stats
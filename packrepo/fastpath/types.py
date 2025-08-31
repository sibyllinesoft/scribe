"""FastPath common types and data structures.

This module contains shared type definitions used across the FastPath system,
preventing circular imports and centralizing type definitions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from .fast_scan import ScanResult
from .base_types import EntryPointSpec


class FastPathVariant(Enum):
    """FastPath system variants for evaluation."""
    V1_BASELINE = "v1_baseline"
    V2_QUOTAS = "v2_quotas"
    V3_CENTRALITY = "v3_centrality"
    V4_DEMOTION = "v4_demotion"
    V5_INTEGRATED = "v5_integrated"


@dataclass 
class DiffPackingOptions:
    """Options for including diffs in the repository pack."""
    enabled: bool = False
    include_staged: bool = True
    include_unstaged: bool = True
    commit_range: Optional[str] = None  # e.g., "HEAD~5..HEAD"
    branch_comparison: Optional[str] = None  # e.g., "main..feature-branch"
    max_commits: int = 50
    relevance_threshold: float = 0.1
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.lock", "*.log", "*.tmp", "*.cache",
        "node_modules/*", ".git/*", "__pycache__/*"
    ])


@dataclass
class ScribeConfig:
    """Configuration for Scribe variant execution."""
    variant: FastPathVariant
    total_budget: int
    
    # Entry point relevance (NEW)
    entry_points: List[EntryPointSpec] = field(default_factory=list)
    personalization_alpha: float = 0.15  # Strength of entry point bias
    
    # Diff packing (NEW)
    diff_options: Optional[DiffPackingOptions] = None
    
    # Feature toggles (override flags for evaluation)
    force_quotas: Optional[bool] = None
    force_centrality: Optional[bool] = None  
    force_demotion: Optional[bool] = None
    force_patch: Optional[bool] = None
    force_bandit: Optional[bool] = None
    
    # Algorithm parameters
    centrality_weight: float = 0.15
    speculation_budget_ratio: float = 0.75
    quota_config_recall_target: float = 0.95
    quota_entry_examples_budget_pct: float = 10.0
    
    # Performance parameters
    max_iterations: int = 10
    convergence_epsilon: float = 1e-6
    
    @classmethod
    def with_entry_points(
        cls,
        entry_points: List[Union[str, EntryPointSpec]],
        variant: FastPathVariant = FastPathVariant.V5_INTEGRATED,
        total_budget: int = 120000,
        personalization_alpha: float = 0.15,
        **kwargs
    ) -> 'ScribeConfig':
        """Create configuration with entry points for personalized relevance."""
        # Import locally to avoid circular imports
        from .utils.entry_points import EntryPointConverter
        processed_entry_points = EntryPointConverter.normalize_entry_points(entry_points)
        
        return cls(
            variant=variant,
            total_budget=total_budget,
            entry_points=processed_entry_points,
            personalization_alpha=personalization_alpha,
            **kwargs
        )
    
    @classmethod
    def with_diffs(
        cls,
        variant: FastPathVariant = FastPathVariant.V5_INTEGRATED,
        total_budget: int = 120000,
        commit_range: Optional[str] = "HEAD~10..HEAD",
        entry_points: Optional[List[Union[str, EntryPointSpec]]] = None,
        **kwargs
    ) -> 'ScribeConfig':
        """Create configuration with diff packing enabled."""
        diff_options = DiffPackingOptions(
            enabled=True,
            commit_range=commit_range,
            **{k: v for k, v in kwargs.items() if k.startswith('diff_') or k in ['max_commits', 'relevance_threshold']}
        )
        
        # Process entry points if provided
        processed_entry_points = []
        if entry_points:
            from .utils.entry_points import EntryPointConverter
            processed_entry_points = EntryPointConverter.normalize_entry_points(entry_points)
        
        # Filter out diff-related kwargs from main config
        filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('diff_') and k not in ['max_commits', 'relevance_threshold']}
        
        return cls(
            variant=variant,
            total_budget=total_budget,
            entry_points=processed_entry_points,
            diff_options=diff_options,
            **filtered_kwargs
        )


@dataclass
class FastPathResult:
    """Result of FastPath selection execution."""
    variant: FastPathVariant
    selected_files: List[ScanResult]
    total_files_considered: int
    budget_used: int
    budget_allocated: int
    
    # Performance metrics
    selection_time_ms: float
    memory_usage_mb: float
    
    # Quality metrics
    heuristic_scores: Dict[str, float]
    final_scores: Dict[str, float]
    coverage_completeness: float
    
    # Detailed breakdown
    stage_timings: Dict[str, float]
    stage_memory: Dict[str, float]
    
    # Entry point and diff features (NEW)
    entry_point_stats: Optional[Dict[str, Any]] = None
    included_diffs: List[Any] = field(default_factory=list)  # List[DiffEntry]
    diff_content: Optional[str] = None
    
    # Variant-specific metrics
    quotas_allocation: Optional[Dict[str, Any]] = None
    centrality_stats: Optional[Dict[str, float]] = None
    demotion_stats: Optional[Dict[str, int]] = None
    patch_stats: Optional[Dict[str, Any]] = None
    routing_decision: Optional[Dict[str, Any]] = None
    
    def get_compression_ratio(self) -> float:
        """Calculate overall compression ratio."""
        return self.budget_used / max(self.budget_allocated, 1)
        
    def get_files_selected_ratio(self) -> float:
        """Calculate ratio of files selected."""
        return len(self.selected_files) / max(self.total_files_considered, 1)
    
    def has_entry_points(self) -> bool:
        """Check if this result used entry point personalization."""
        return self.entry_point_stats is not None
    
    def has_diffs(self) -> bool:
        """Check if this result includes diff content."""
        return len(self.included_diffs) > 0 or self.diff_content is not None

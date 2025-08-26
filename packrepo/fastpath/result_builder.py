"""FastPath result construction utilities.

This module provides builders and utilities to standardize the creation
of FastPathResult objects and eliminate code duplication across variant
implementations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .fast_scan import ScanResult
from .types import FastPathVariant, FastPathResult
from ..packer.tokenizer import estimate_tokens_scan_result


class FastPathResultBuilder:
    """Builder pattern for creating FastPathResult objects with consistent defaults.
    
    This eliminates the boilerplate of creating FastPathResult objects
    throughout the codebase and ensures consistent field handling.
    """
    
    def __init__(self, variant: FastPathVariant):
        """Initialize builder with required variant."""
        self.variant = variant
        self.selected_files: List[ScanResult] = []
        self.total_files_considered: int = 0
        self.budget_used: int = 0
        self.budget_allocated: int = 0
        self.selection_time_ms: float = 0.0
        self.memory_usage_mb: float = 0.0
        self.heuristic_scores: Dict[str, float] = {}
        self.final_scores: Dict[str, float] = {}
        self.coverage_completeness: float = 0.0
        self.stage_timings: Dict[str, float] = {}
        self.stage_memory: Dict[str, float] = {}
        
        # Entry point and diff features (NEW)
        self.entry_point_stats: Optional[Dict[str, Any]] = None
        self.included_diffs: List[Any] = []
        self.diff_content: Optional[str] = None
        
        # Variant-specific fields
        self.quotas_allocation: Optional[Dict[str, Any]] = None
        self.centrality_stats: Optional[Dict[str, float]] = None
        self.demotion_stats: Optional[Dict[str, int]] = None
        self.patch_stats: Optional[Dict[str, Any]] = None
        self.routing_decision: Optional[Dict[str, Any]] = None
    
    def with_selection(
        self, 
        selected_files: List[ScanResult], 
        total_files_considered: int
    ) -> 'FastPathResultBuilder':
        """Set file selection results."""
        self.selected_files = selected_files
        self.total_files_considered = total_files_considered
        
        # Auto-calculate budget used if not set
        if self.budget_used == 0:
            self.budget_used = sum(estimate_tokens_scan_result(f) for f in selected_files)
        
        # Auto-calculate coverage completeness
        if total_files_considered > 0:
            self.coverage_completeness = len(selected_files) / total_files_considered
        
        return self
    
    def with_budget(self, budget_allocated: int, budget_used: int = 0) -> 'FastPathResultBuilder':
        """Set budget allocation information."""
        self.budget_allocated = budget_allocated
        if budget_used > 0:
            self.budget_used = budget_used
        return self
    
    def with_scores(
        self, 
        heuristic_scores: Dict[str, float], 
        final_scores: Optional[Dict[str, float]] = None
    ) -> 'FastPathResultBuilder':
        """Set scoring information."""
        self.heuristic_scores = heuristic_scores
        self.final_scores = final_scores or heuristic_scores
        return self
    
    def with_performance(
        self, 
        selection_time_ms: float = 0.0, 
        memory_usage_mb: float = 0.0
    ) -> 'FastPathResultBuilder':
        """Set performance metrics."""
        self.selection_time_ms = selection_time_ms
        self.memory_usage_mb = memory_usage_mb
        return self
    
    def with_stage_metrics(
        self, 
        stage_timings: Dict[str, float], 
        stage_memory: Optional[Dict[str, float]] = None
    ) -> 'FastPathResultBuilder':
        """Set stage-level performance metrics."""
        self.stage_timings = stage_timings
        self.stage_memory = stage_memory or {}
        return self
    
    def with_quotas_allocation(self, quotas_allocation: Dict[str, Any]) -> 'FastPathResultBuilder':
        """Set quotas allocation results (V2 specific)."""
        self.quotas_allocation = quotas_allocation
        return self
    
    def with_centrality_stats(self, centrality_stats: Dict[str, float]) -> 'FastPathResultBuilder':
        """Set centrality statistics (V3 specific)."""
        self.centrality_stats = centrality_stats
        return self
    
    def with_demotion_stats(self, demotion_stats: Dict[str, int]) -> 'FastPathResultBuilder':
        """Set demotion statistics (V4 specific)."""
        self.demotion_stats = demotion_stats
        return self
    
    def with_patch_stats(self, patch_stats: Dict[str, Any]) -> 'FastPathResultBuilder':
        """Set patch statistics (V5 specific)."""
        self.patch_stats = patch_stats
        return self
    
    def with_routing_decision(self, routing_decision: Dict[str, Any]) -> 'FastPathResultBuilder':
        """Set routing decision information (V5 specific)."""
        self.routing_decision = routing_decision
        return self
    
    def with_entry_point_stats(self, entry_point_stats: Dict[str, Any]) -> 'FastPathResultBuilder':
        """Set entry point statistics (NEW feature)."""
        self.entry_point_stats = entry_point_stats
        return self
    
    def with_diffs(self, included_diffs: List[Any], diff_content: Optional[str] = None) -> 'FastPathResultBuilder':
        """Set diff information (NEW feature)."""
        self.included_diffs = included_diffs
        self.diff_content = diff_content
        return self
    
    def build(self) -> FastPathResult:
        """Build the final FastPathResult object."""
        return FastPathResult(
            variant=self.variant,
            selected_files=self.selected_files,
            total_files_considered=self.total_files_considered,
            budget_used=self.budget_used,
            budget_allocated=self.budget_allocated,
            selection_time_ms=self.selection_time_ms,
            memory_usage_mb=self.memory_usage_mb,
            heuristic_scores=self.heuristic_scores,
            final_scores=self.final_scores,
            coverage_completeness=self.coverage_completeness,
            stage_timings=self.stage_timings,
            stage_memory=self.stage_memory,
            entry_point_stats=self.entry_point_stats,
            included_diffs=self.included_diffs,
            diff_content=self.diff_content,
            quotas_allocation=self.quotas_allocation,
            centrality_stats=self.centrality_stats,
            demotion_stats=self.demotion_stats,
            patch_stats=self.patch_stats,
            routing_decision=self.routing_decision,
        )


def create_result_builder(variant: FastPathVariant) -> FastPathResultBuilder:
    """Factory function to create a result builder for the given variant."""
    return FastPathResultBuilder(variant)


def create_simple_result(
    variant: FastPathVariant,
    selected_files: List[ScanResult],
    total_files_considered: int,
    budget_allocated: int,
    heuristic_scores: Dict[str, float]
) -> FastPathResult:
    """Create a simple FastPathResult with minimal configuration.
    
    This is a convenience function for the most common case where
    you just need basic file selection results without complex metrics.
    """
    return (create_result_builder(variant)
            .with_selection(selected_files, total_files_considered)
            .with_budget(budget_allocated)
            .with_scores(heuristic_scores)
            .build())


def create_timed_result(
    variant: FastPathVariant,
    selected_files: List[ScanResult],
    total_files_considered: int,
    budget_allocated: int,
    heuristic_scores: Dict[str, float],
    selection_time_ms: float,
    memory_usage_mb: float = 0.0
) -> FastPathResult:
    """Create a FastPathResult with timing information.
    
    This is a convenience function for cases where you want to include
    performance metrics in the result.
    """
    return (create_result_builder(variant)
            .with_selection(selected_files, total_files_considered)
            .with_budget(budget_allocated)
            .with_scores(heuristic_scores)
            .with_performance(selection_time_ms, memory_usage_mb)
            .build())


class ResultStandardizer:
    """Utility class to standardize result formatting across variants.
    
    Provides consistent formatting and validation of FastPathResult objects
    to ensure research-grade output quality.
    """
    
    @staticmethod
    def validate_result(result: FastPathResult) -> List[str]:
        """Validate a FastPathResult and return any issues found.
        
        Args:
            result: FastPathResult to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Budget validation
        if result.budget_used > result.budget_allocated:
            issues.append(f"Budget exceeded: {result.budget_used} > {result.budget_allocated}")
        
        # File count consistency
        expected_files = len(result.selected_files)
        if result.total_files_considered < expected_files:
            issues.append(f"Selected more files than considered: {expected_files} > {result.total_files_considered}")
        
        # Coverage completeness validation
        if result.total_files_considered > 0:
            expected_coverage = expected_files / result.total_files_considered
            if abs(result.coverage_completeness - expected_coverage) > 0.01:
                issues.append(f"Coverage completeness mismatch: {result.coverage_completeness} vs expected {expected_coverage}")
        
        # Variant-specific validation
        if result.variant == FastPathVariant.V2_QUOTAS and result.quotas_allocation is None:
            issues.append("V2 result missing quotas allocation data")
        
        if result.variant == FastPathVariant.V3_CENTRALITY and result.centrality_stats is None:
            issues.append("V3 result missing centrality statistics")
        
        if result.variant == FastPathVariant.V4_DEMOTION and result.demotion_stats is None:
            issues.append("V4 result missing demotion statistics")
        
        if result.variant == FastPathVariant.V5_INTEGRATED and result.routing_decision is None:
            issues.append("V5 result missing routing decision")
        
        return issues
    
    @staticmethod
    def normalize_scores(result: FastPathResult) -> FastPathResult:
        """Normalize score dictionaries to ensure consistent key formats.
        
        Args:
            result: FastPathResult to normalize
            
        Returns:
            FastPathResult with normalized scores
        """
        # Ensure all selected files have scores
        selected_paths = {f.stats.path for f in result.selected_files}
        
        normalized_heuristic = {}
        for path, score in result.heuristic_scores.items():
            normalized_path = str(path)
            normalized_heuristic[normalized_path] = float(score)
        
        normalized_final = {}
        for path, score in result.final_scores.items():
            normalized_path = str(path)
            normalized_final[normalized_path] = float(score)
        
        # Create new result with normalized scores
        return FastPathResult(
            variant=result.variant,
            selected_files=result.selected_files,
            total_files_considered=result.total_files_considered,
            budget_used=result.budget_used,
            budget_allocated=result.budget_allocated,
            selection_time_ms=result.selection_time_ms,
            memory_usage_mb=result.memory_usage_mb,
            heuristic_scores=normalized_heuristic,
            final_scores=normalized_final,
            coverage_completeness=result.coverage_completeness,
            stage_timings=result.stage_timings,
            stage_memory=result.stage_memory,
            quotas_allocation=result.quotas_allocation,
            centrality_stats=result.centrality_stats,
            demotion_stats=result.demotion_stats,
            patch_stats=result.patch_stats,
            routing_decision=result.routing_decision,
        )
    
    @staticmethod
    def add_computed_metrics(result: FastPathResult) -> FastPathResult:
        """Add computed metrics to enhance result information.
        
        Args:
            result: FastPathResult to enhance
            
        Returns:
            FastPathResult with additional computed metrics
        """
        # Add file size distribution stats
        if result.selected_files:
            file_sizes = [f.stats.size_bytes for f in result.selected_files]
            file_tokens = [estimate_tokens_scan_result(f) for f in result.selected_files]
            
            size_stats = {
                "total_size_bytes": sum(file_sizes),
                "avg_size_bytes": sum(file_sizes) / len(file_sizes),
                "total_tokens_estimated": sum(file_tokens),
                "avg_tokens_per_file": sum(file_tokens) / len(file_tokens),
            }
            
            # Add to stage memory (repurpose for computed metrics)
            enhanced_stage_memory = dict(result.stage_memory)
            enhanced_stage_memory.update(size_stats)
            
            return FastPathResult(
                variant=result.variant,
                selected_files=result.selected_files,
                total_files_considered=result.total_files_considered,
                budget_used=result.budget_used,
                budget_allocated=result.budget_allocated,
                selection_time_ms=result.selection_time_ms,
                memory_usage_mb=result.memory_usage_mb,
                heuristic_scores=result.heuristic_scores,
                final_scores=result.final_scores,
                coverage_completeness=result.coverage_completeness,
                stage_timings=result.stage_timings,
                stage_memory=enhanced_stage_memory,
                quotas_allocation=result.quotas_allocation,
                centrality_stats=result.centrality_stats,
                demotion_stats=result.demotion_stats,
                patch_stats=result.patch_stats,
                routing_decision=result.routing_decision,
            )
        
        return result
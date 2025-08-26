"""
Integrated FastPath V5 System

Combines all workstreams into a unified, research-grade selection system:
- Workstream A: Quotas + Density-Greedy
- Workstream B: PageRank Centrality  
- Workstream C: Hybrid Demotion
- Workstream D: Two-pass Speculate-Patch
- Workstream E: Router Guard + Thompson Sampling

Provides V1-V5 variant selection with flag-guarded feature combinations
for rigorous paired evaluation and statistical validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from .fast_scan import ScanResult
from .types import FastPathVariant, ScribeConfig, FastPathResult
from .feature_flags import get_feature_flags
from .quotas import QuotaManager, create_quota_manager
from .centrality import CentralityCalculator, create_centrality_calculator
from .demotion import DemotionEngine, create_demotion_engine, FidelityMode
from .patch_system import TwoPassSelector, create_two_pass_selector
from .bandit_router import RouterGuard, create_router_guard, SelectionAlgorithm, ContextVector
from .heuristics import HeuristicScorer, create_scorer
from ..packer.tokenizer import estimate_tokens_scan_result
from .result_builder import create_simple_result, create_result_builder
from .execution_strategy import create_variant_executor, VariantExecutor




class FastPathEngine:
    """
    Main FastPath execution engine that orchestrates all workstreams.
    
    Provides unified interface for executing any FastPath variant with
    comprehensive performance and quality metrics collection.
    """
    
    def __init__(self):
        # Initialize all component systems
        self.heuristic_scorer = create_scorer()
        self.quota_manager = None  # Lazy-loaded with budget
        self.centrality_calculator = create_centrality_calculator()
        self.demotion_engine = create_demotion_engine()
        self.two_pass_selector = create_two_pass_selector()
        self.router_guard = create_router_guard()
        self.variant_executor = create_variant_executor(self)
        
    def execute_variant(
        self, 
        scan_results: List[ScanResult],
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute specified FastPath variant with comprehensive metrics.
        
        Delegates to the strategy-based executor for cleaner separation of concerns.
        """
        return self.variant_executor.execute_variant(scan_results, config, query_hint)
    
    def _compute_heuristic_scores(self, scan_results: List[ScanResult], config: ScribeConfig) -> Dict[str, float]:
        """Compute base heuristic scores for all files."""
        scored_files = self.heuristic_scorer.score_all_files(scan_results)
        return {result.stats.path: components.final_score 
                for result, components in scored_files}
    
    def _execute_v1_baseline(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig
    ) -> FastPathResult:
        """Execute V1 baseline: heuristics-only selection."""
        
        # Simple greedy selection by heuristic score
        scored_files = [(result, heuristic_scores.get(result.stats.path, 0.0)) 
                       for result in scan_results]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        selected_files = []
        budget_used = 0
        
        for result, score in scored_files:
            estimated_tokens = estimate_tokens_scan_result(result)
            if budget_used + estimated_tokens <= config.total_budget:
                selected_files.append(result)
                budget_used += estimated_tokens
        
        return create_simple_result(
            variant=config.variant,
            selected_files=selected_files,
            total_files_considered=len(scan_results),
            budget_allocated=config.total_budget,
            heuristic_scores=heuristic_scores
        )
    
    def _execute_v2_quotas(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig
    ) -> FastPathResult:
        """Execute V2: V1 + Quotas + Density-Greedy."""
        
        # Initialize quota manager  
        quota_manager = create_quota_manager(config.total_budget)
        
        # Apply quotas selection
        selected_files, quotas_allocation = quota_manager.apply_quotas_selection(
            scan_results, heuristic_scores
        )
        
        budget_used = sum(estimate_tokens_scan_result(result) for result in selected_files)
        
        return (create_result_builder(config.variant)
                .with_selection(selected_files, len(scan_results))
                .with_budget(config.total_budget, budget_used)
                .with_scores(heuristic_scores)
                .with_quotas_allocation({cat.value: {
                    'allocated_budget': alloc.allocated_budget,
                    'used_budget': alloc.used_budget, 
                    'file_count': alloc.file_count,
                    'recall_achieved': alloc.recall_achieved
                } for cat, alloc in quotas_allocation.items()})
                .build())
    
    def _execute_v3_centrality(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig
    ) -> FastPathResult:
        """Execute V3: V2 + PageRank Centrality."""
        
        # First apply V2 quotas
        quota_manager = create_quota_manager(config.total_budget)
        
        # Integrate centrality with heuristics
        centrality_scores = self.centrality_calculator.integrate_with_heuristics(
            scan_results, heuristic_scores, config.centrality_weight
        )
        
        # Apply quotas with enhanced scores
        selected_files, quotas_allocation = quota_manager.apply_quotas_selection(
            scan_results, centrality_scores
        )
        
        # Get centrality statistics
        centrality_results = self.centrality_calculator.calculate_centrality_scores(scan_results)
        
        budget_used = sum(estimate_tokens_scan_result(result) for result in selected_files)
        
        return FastPathResult(
            variant=config.variant,
            selected_files=selected_files,
            total_files_considered=len(scan_results),
            budget_used=budget_used,
            budget_allocated=config.total_budget,
            selection_time_ms=0,
            memory_usage_mb=0,
            heuristic_scores=heuristic_scores,
            final_scores=centrality_scores,
            coverage_completeness=len(selected_files) / max(len(scan_results), 1),
            quotas_allocation={cat.value: {
                'allocated_budget': alloc.allocated_budget,
                'used_budget': alloc.used_budget,
                'file_count': alloc.file_count, 
                'recall_achieved': alloc.recall_achieved
            } for cat, alloc in quotas_allocation.items()},
            centrality_stats={
                'iterations_converged': centrality_results.iterations_converged,
                'total_nodes': centrality_results.graph_stats.total_nodes,
                'total_edges': centrality_results.graph_stats.total_edges,
                'graph_density': centrality_results.graph_stats.graph_density
            },
            stage_timings={},
            stage_memory={}
        )
    
    def _execute_v4_demotion(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig
    ) -> FastPathResult:
        """Execute V4: V3 + Hybrid Demotion."""
        
        # First execute V3 to get initial selection
        v3_result = self._execute_v3_centrality(scan_results, heuristic_scores, config)
        
        # Apply demotion to optimize budget usage
        demotion_results, final_budget = self.demotion_engine.batch_demote_files(
            v3_result.selected_files, config.total_budget, v3_result.final_scores
        )
        
        # Map demotion results back to original scan results
        final_selected_files = []
        path_to_scan_result = {sr.stats.path: sr for sr in v3_result.selected_files}
        
        for demote_result in demotion_results:
            if demote_result.original_path in path_to_scan_result:
                final_selected_files.append(path_to_scan_result[demote_result.original_path])
        
        # Calculate demotion statistics
        demotion_stats = {
            'files_demoted': len([dr for dr in demotion_results if dr.fidelity_mode != FidelityMode.FULL]),
            'total_compression_ratio': sum(dr.compression_ratio for dr in demotion_results) / max(len(demotion_results), 1),
            'average_quality_score': sum(dr.quality_score for dr in demotion_results) / max(len(demotion_results), 1)
        }
        
        # Update result with demotion
        v3_result.variant = config.variant
        v3_result.selected_files = final_selected_files
        v3_result.budget_used = final_budget
        v3_result.demotion_stats = demotion_stats
        
        return v3_result
    
    def _execute_v5_integrated(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute V5: Full integration with router and two-pass selection."""
        
        # Stage 1: Router decision
        routing_decision = self.router_guard.route_selection_algorithm(scan_results, query_hint)
        
        # Stage 2: Execute routed algorithm
        if routing_decision.selected_algorithm == SelectionAlgorithm.TWO_PASS_PATCH:
            # Use two-pass system
            final_files, speculation_result, patch_result = self.two_pass_selector.execute_two_pass_selection(
                scan_results, heuristic_scores, config.total_budget, config.speculation_budget_ratio
            )
            
            budget_used = speculation_result.budget_used + patch_result.additional_budget_used
            
            patch_stats = {
                'speculation_files': len(speculation_result.selected_files),
                'patch_files': len(patch_result.patched_files),
                'coverage_improvement': patch_result.coverage_improvement,
                'rules_applied': len(patch_result.patch_rules_applied)
            }
        else:
            # Fall back to V4 execution
            v4_result = self._execute_v4_demotion(scan_results, heuristic_scores, config)
            final_files = v4_result.selected_files
            budget_used = v4_result.budget_used
            patch_stats = None
        
        return FastPathResult(
            variant=config.variant,
            selected_files=final_files,
            total_files_considered=len(scan_results),
            budget_used=budget_used,
            budget_allocated=config.total_budget,
            selection_time_ms=0,
            memory_usage_mb=0,
            heuristic_scores=heuristic_scores,
            final_scores=heuristic_scores,  # Could be enhanced
            coverage_completeness=len(final_files) / max(len(scan_results), 1),
            patch_stats=patch_stats,
            routing_decision={
                'selected_algorithm': routing_decision.selected_algorithm.value,
                'confidence_score': routing_decision.confidence_score,
                'rationale': routing_decision.decision_rationale
            },
            stage_timings={},
            stage_memory={}
        )


def create_fastpath_engine() -> FastPathEngine:
    """Create a FastPath execution engine."""
    return FastPathEngine()


def get_variant_flag_configuration(variant: FastPathVariant) -> Dict[str, bool]:
    """Get required feature flag configuration for a variant."""
    
    configs = {
        FastPathVariant.V1_BASELINE: {
            'FASTPATH_QUOTAS': False,
            'FASTPATH_CENTRALITY': False, 
            'FASTPATH_DEMOTE': False,
            'FASTPATH_PATCH': False,
            'FASTPATH_BANDIT': False
        },
        FastPathVariant.V2_QUOTAS: {
            'FASTPATH_QUOTAS': True,
            'FASTPATH_CENTRALITY': False,
            'FASTPATH_DEMOTE': False, 
            'FASTPATH_PATCH': False,
            'FASTPATH_BANDIT': False
        },
        FastPathVariant.V3_CENTRALITY: {
            'FASTPATH_QUOTAS': True,
            'FASTPATH_CENTRALITY': True,
            'FASTPATH_DEMOTE': False,
            'FASTPATH_PATCH': False,
            'FASTPATH_BANDIT': False
        },
        FastPathVariant.V4_DEMOTION: {
            'FASTPATH_QUOTAS': True,
            'FASTPATH_CENTRALITY': True,
            'FASTPATH_DEMOTE': True,
            'FASTPATH_PATCH': False,
            'FASTPATH_BANDIT': False
        },
        FastPathVariant.V5_INTEGRATED: {
            'FASTPATH_QUOTAS': True,
            'FASTPATH_CENTRALITY': True,
            'FASTPATH_DEMOTE': True,
            'FASTPATH_PATCH': True,
            'FASTPATH_BANDIT': True
        }
    }
    
    return configs.get(variant, configs[FastPathVariant.V1_BASELINE])
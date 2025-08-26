"""FastPath execution strategy pattern.

This module provides a strategy pattern for executing different FastPath variants,
separating the execution logic from the performance monitoring and result management.
This decomposition reduces complexity and improves maintainability.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Protocol
import time
import psutil
import os

from .fast_scan import ScanResult
from .types import FastPathVariant, ScribeConfig, FastPathResult
from .result_builder import create_result_builder


class PerformanceMonitor:
    """Handles performance tracking for FastPath execution.
    
    Separates performance monitoring concerns from execution logic,
    making the code more testable and maintainable.
    """
    
    def __init__(self):
        self.start_time: float = 0.0
        self.start_memory: float = 0.0
        self.stage_timings: Dict[str, float] = {}
        self.stage_memory: Dict[str, float] = {}
        self.process = psutil.Process(os.getpid())
    
    def start_monitoring(self) -> None:
        """Start overall performance monitoring."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.stage_timings.clear()
        self.stage_memory.clear()
    
    def start_stage(self, stage_name: str) -> 'StageTimer':
        """Start monitoring a specific stage."""
        return StageTimer(self, stage_name)
    
    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Record stage timing and memory usage."""
        self.stage_timings[stage_name] = duration_ms
        self.stage_memory[stage_name] = self.process.memory_info().rss / 1024 / 1024
    
    def get_total_metrics(self) -> tuple[float, float]:
        """Get total execution time and memory usage."""
        total_time_ms = (time.perf_counter() - self.start_time) * 1000
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_delta = current_memory - self.start_memory
        return total_time_ms, memory_delta
    
    def get_stage_metrics(self) -> tuple[Dict[str, float], Dict[str, float]]:
        """Get stage-level performance metrics."""
        return self.stage_timings.copy(), self.stage_memory.copy()


class StageTimer:
    """Context manager for timing execution stages."""
    
    def __init__(self, monitor: PerformanceMonitor, stage_name: str):
        self.monitor = monitor
        self.stage_name = stage_name
        self.stage_start: float = 0.0
    
    def __enter__(self):
        self.stage_start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.stage_start) * 1000
        self.monitor.record_stage(self.stage_name, duration_ms)


class VariantExecutionStrategy(ABC):
    """Abstract strategy for executing FastPath variants.
    
    This separates the specific execution logic for each variant
    from the common performance monitoring and error handling.
    """
    
    @abstractmethod
    def execute(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute the specific variant algorithm."""
        pass
    
    @property
    @abstractmethod
    def supported_variant(self) -> FastPathVariant:
        """Return the FastPath variant this strategy supports."""
        pass


class BaselineExecutionStrategy(VariantExecutionStrategy):
    """Strategy for V1 baseline execution."""
    
    def __init__(self, engine):
        self.engine = engine
    
    @property
    def supported_variant(self) -> FastPathVariant:
        return FastPathVariant.V1_BASELINE
    
    def execute(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute V1 baseline algorithm."""
        return self.engine._execute_v1_baseline(scan_results, heuristic_scores, config)


class QuotasExecutionStrategy(VariantExecutionStrategy):
    """Strategy for V2 quotas execution."""
    
    def __init__(self, engine):
        self.engine = engine
    
    @property
    def supported_variant(self) -> FastPathVariant:
        return FastPathVariant.V2_QUOTAS
    
    def execute(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute V2 quotas algorithm."""
        return self.engine._execute_v2_quotas(scan_results, heuristic_scores, config)


class CentralityExecutionStrategy(VariantExecutionStrategy):
    """Strategy for V3 centrality execution."""
    
    def __init__(self, engine):
        self.engine = engine
    
    @property
    def supported_variant(self) -> FastPathVariant:
        return FastPathVariant.V3_CENTRALITY
    
    def execute(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute V3 centrality algorithm."""
        return self.engine._execute_v3_centrality(scan_results, heuristic_scores, config)


class DemotionExecutionStrategy(VariantExecutionStrategy):
    """Strategy for V4 demotion execution."""
    
    def __init__(self, engine):
        self.engine = engine
    
    @property
    def supported_variant(self) -> FastPathVariant:
        return FastPathVariant.V4_DEMOTION
    
    def execute(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute V4 demotion algorithm."""
        return self.engine._execute_v4_demotion(scan_results, heuristic_scores, config)


class IntegratedExecutionStrategy(VariantExecutionStrategy):
    """Strategy for V5 integrated execution."""
    
    def __init__(self, engine):
        self.engine = engine
    
    @property
    def supported_variant(self) -> FastPathVariant:
        return FastPathVariant.V5_INTEGRATED
    
    def execute(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute V5 integrated algorithm."""
        return self.engine._execute_v5_integrated(scan_results, heuristic_scores, config, query_hint)


class VariantExecutor:
    """Coordinates variant execution with performance monitoring and error handling.
    
    This class replaces the complex execute_variant method with a cleaner
    design that separates concerns and is easier to test and maintain.
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.performance_monitor = PerformanceMonitor()
        self.strategies: Dict[FastPathVariant, VariantExecutionStrategy] = {
            FastPathVariant.V1_BASELINE: BaselineExecutionStrategy(engine),
            FastPathVariant.V2_QUOTAS: QuotasExecutionStrategy(engine),
            FastPathVariant.V3_CENTRALITY: CentralityExecutionStrategy(engine),
            FastPathVariant.V4_DEMOTION: DemotionExecutionStrategy(engine),
            FastPathVariant.V5_INTEGRATED: IntegratedExecutionStrategy(engine),
        }
    
    def execute_variant(
        self, 
        scan_results: List[ScanResult],
        config: ScribeConfig,
        query_hint: str = ""
    ) -> FastPathResult:
        """Execute specified FastPath variant with comprehensive metrics.
        
        This is the main entry point that replaces the original complex method.
        """
        self.performance_monitor.start_monitoring()
        
        try:
            # Get the appropriate strategy
            strategy = self.strategies.get(config.variant)
            if strategy is None:
                raise ValueError(f"Unknown variant: {config.variant}")
            
            # Stage 1: Heuristic Scoring (All variants)
            with self.performance_monitor.start_stage('heuristic_scoring'):
                heuristic_scores = self.engine._compute_heuristic_scores(scan_results, config)
            
            # Stage 2: Variant-specific execution
            with self.performance_monitor.start_stage('variant_execution'):
                result = strategy.execute(scan_results, heuristic_scores, config, query_hint)
            
            # Finalize with performance metrics
            return self._finalize_result(result, heuristic_scores)
            
        except Exception as e:
            # Return error result with diagnostic information
            return self._create_error_result(config, scan_results, str(e))
    
    def _finalize_result(self, result: FastPathResult, heuristic_scores: Dict[str, float]) -> FastPathResult:
        """Add performance metrics to the result."""
        total_time_ms, memory_delta_mb = self.performance_monitor.get_total_metrics()
        stage_timings, stage_memory = self.performance_monitor.get_stage_metrics()
        
        # Create enhanced result with performance metrics
        return (create_result_builder(result.variant)
                .with_selection(result.selected_files, result.total_files_considered)
                .with_budget(result.budget_allocated, result.budget_used)
                .with_scores(heuristic_scores, result.final_scores)
                .with_performance(total_time_ms, memory_delta_mb)
                .with_stage_metrics(stage_timings, stage_memory)
                .with_quotas_allocation(result.quotas_allocation)
                .with_centrality_stats(result.centrality_stats)
                .with_demotion_stats(result.demotion_stats)
                .with_patch_stats(result.patch_stats)
                .with_routing_decision(result.routing_decision)
                .build())
    
    def _create_error_result(self, config: ScribeConfig, scan_results: List[ScanResult], error_msg: str) -> FastPathResult:
        """Create a result object for error cases."""
        total_time_ms, _ = self.performance_monitor.get_total_metrics()
        
        return (create_result_builder(config.variant)
                .with_selection([], len(scan_results))
                .with_budget(config.total_budget, 0)
                .with_scores({})
                .with_performance(total_time_ms, 0.0)
                .with_stage_metrics({'error': total_time_ms}, {'error': self.performance_monitor.start_memory})
                .build())


def create_variant_executor(engine) -> VariantExecutor:
    """Factory function to create a variant executor."""
    return VariantExecutor(engine)
"""Main parity controller for coordinating fair baseline comparisons."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .budget_enforcer import BudgetEnforcer, BudgetConstraints, BudgetReport
from .comparison_framework import ComparisonFramework, ComparisonResult
from ..baselines import create_baseline_selector, BaselineSelector
from ..baselines.base import BaselineConfig
from ..selector.base import PackResult, PackRequest
from ..chunker.base import Chunk
from ..tokenizer.base import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class ParityConfig:
    """Configuration for parity-controlled evaluation."""
    
    # Budget settings
    target_budget: int = 120000
    budget_tolerance_percent: float = 5.0
    separate_decode_budget: bool = True
    
    # Variant settings
    variants_to_test: List[str] = None  # Default: ['V0a', 'V0b', 'V0c', 'V1', 'V2', 'V3']
    
    # Execution settings
    deterministic_mode: bool = True
    random_seed: int = 42
    
    # Performance settings
    max_execution_time_sec: float = 300.0  # 5 minutes per variant
    enable_performance_tracking: bool = True
    
    def __post_init__(self):
        """Set default variants if not specified."""
        if self.variants_to_test is None:
            self.variants_to_test = ['V0a', 'V0b', 'V0c', 'V1', 'V2', 'V3']


@dataclass
class ParityEvaluationResult:
    """Complete result from parity-controlled evaluation."""
    
    config: ParityConfig
    
    # Results by variant
    pack_results: Dict[str, PackResult]  # variant_id -> PackResult
    
    # Budget analysis
    budget_reports: Dict[str, BudgetReport]
    budget_compliance_rate: float
    
    # Comparative analysis
    comparison_result: ComparisonResult
    
    # Execution metadata
    execution_times: Dict[str, float]  # variant_id -> seconds
    total_execution_time: float
    failed_variants: List[str]
    
    # Quality metrics
    deterministic_hashes: Dict[str, str]  # variant_id -> hash
    
    def get_compliant_variants(self) -> List[str]:
        """Get list of budget-compliant variants."""
        return [
            variant_id for variant_id, report in self.budget_reports.items()
            if report.within_tolerance
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        successful_variants = [v for v in self.config.variants_to_test if v not in self.failed_variants]
        
        return {
            "total_variants": len(self.config.variants_to_test),
            "successful_variants": len(successful_variants),
            "failed_variants": len(self.failed_variants),
            "success_rate": len(successful_variants) / len(self.config.variants_to_test),
            "budget_compliance_rate": self.budget_compliance_rate,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": sum(self.execution_times.values()) / max(1, len(self.execution_times)),
        }


class ParityController:
    """
    Main controller for parity-controlled baseline and variant comparisons.
    
    Coordinates budget enforcement, baseline selection, and comparative analysis
    to ensure fair evaluation according to TODO.md requirements.
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """Initialize parity controller."""
        self.tokenizer = tokenizer
        self.budget_enforcer = BudgetEnforcer(tokenizer)
        self.comparison_framework = ComparisonFramework()
        
        # Cache for baseline selectors
        self._baseline_cache: Dict[str, BaselineSelector] = {}
        
    def run_parity_evaluation(
        self,
        chunks: List[Chunk],
        config: ParityConfig,
        advanced_selector = None  # For V1-V3 variants
    ) -> ParityEvaluationResult:
        """
        Run complete parity-controlled evaluation across all variants.
        
        Args:
            chunks: Chunks to evaluate on
            config: Parity configuration
            advanced_selector: Selector instance for V1-V3 variants
            
        Returns:
            Complete parity evaluation result
        """
        logger.info(f"Starting parity evaluation with {len(config.variants_to_test)} variants")
        start_time = time.time()
        
        # Initialize results
        pack_results: Dict[str, PackResult] = {}
        execution_times: Dict[str, float] = {}
        failed_variants: List[str] = []
        deterministic_hashes: Dict[str, str] = {}
        
        # Create budget constraints
        budget_constraints = self.budget_enforcer.create_constraints(
            target_budget=config.target_budget,
            tolerance_percent=config.budget_tolerance_percent,
            separate_decode_budget=config.separate_decode_budget,
        )
        
        # Run each variant
        for variant_id in config.variants_to_test:
            logger.info(f"Evaluating {variant_id}...")
            
            try:
                variant_start = time.time()
                
                # Run variant with timeout
                result = self._run_single_variant(
                    variant_id, chunks, config, budget_constraints, advanced_selector
                )
                
                variant_time = time.time() - variant_start
                
                # Check execution time limit
                if variant_time > config.max_execution_time_sec:
                    logger.warning(f"{variant_id} exceeded time limit ({variant_time:.1f}s > {config.max_execution_time_sec}s)")
                
                # Store results
                pack_results[variant_id] = result
                execution_times[variant_id] = variant_time
                
                # Extract deterministic hash if available
                if result.deterministic_hash:
                    deterministic_hashes[variant_id] = result.deterministic_hash
                
                logger.info(f"{variant_id} completed in {variant_time:.2f}s")
                
            except Exception as e:
                logger.error(f"{variant_id} failed: {str(e)}")
                failed_variants.append(variant_id)
        
        # Validate budget parity
        budget_reports = self.budget_enforcer.validate_budget_parity(
            pack_results, budget_constraints
        )
        
        # Calculate compliance rate
        compliant_count = sum(1 for report in budget_reports.values() if report.within_tolerance)
        budget_compliance_rate = compliant_count / max(1, len(budget_reports))
        
        # Run comparative analysis
        comparison_result = self.comparison_framework.compare_variants(
            pack_results, budget_reports
        )
        
        total_time = time.time() - start_time
        
        # Create final result
        result = ParityEvaluationResult(
            config=config,
            pack_results=pack_results,
            budget_reports=budget_reports,
            budget_compliance_rate=budget_compliance_rate,
            comparison_result=comparison_result,
            execution_times=execution_times,
            total_execution_time=total_time,
            failed_variants=failed_variants,
            deterministic_hashes=deterministic_hashes,
        )
        
        # Log summary
        self._log_evaluation_summary(result)
        
        return result
    
    def _run_single_variant(
        self,
        variant_id: str,
        chunks: List[Chunk],
        config: ParityConfig,
        budget_constraints: BudgetConstraints,
        advanced_selector = None
    ) -> PackResult:
        """Run evaluation for a single variant."""
        
        if variant_id in ['V0a', 'V0b', 'V0c']:
            # Use baseline selector
            return self._run_baseline_variant(variant_id, chunks, config, budget_constraints)
        else:
            # Use advanced selector for V1-V3
            return self._run_advanced_variant(variant_id, chunks, config, budget_constraints, advanced_selector)
    
    def _run_baseline_variant(
        self,
        variant_id: str,
        chunks: List[Chunk],
        config: ParityConfig,
        budget_constraints: BudgetConstraints
    ) -> PackResult:
        """Run baseline variant (V0a, V0b, V0c)."""
        
        # Get or create baseline selector
        if variant_id not in self._baseline_cache:
            self._baseline_cache[variant_id] = create_baseline_selector(variant_id)
        
        baseline_selector = self._baseline_cache[variant_id]
        
        # Create baseline configuration
        baseline_config = BaselineConfig(
            token_budget=budget_constraints.target_budget,
            deterministic=config.deterministic_mode,
            random_seed=config.random_seed,
        )
        
        # Run selection
        selection_result = baseline_selector.select(chunks, baseline_config)
        
        # Create pack request for consistency
        pack_request = PackRequest(
            chunks=chunks,
            config=baseline_config.to_selection_config()
        )
        
        # Create pack result
        pack_result = PackResult(
            request=pack_request,
            selection=selection_result,
            execution_time=getattr(selection_result, 'execution_time', 0.0),
            memory_peak=0.0,  # Not tracked in baseline
            deterministic_hash=baseline_selector._generate_deterministic_hash(selection_result) if config.deterministic_mode else None,
        )
        
        return pack_result
    
    def _run_advanced_variant(
        self,
        variant_id: str,
        chunks: List[Chunk],
        config: ParityConfig,
        budget_constraints: BudgetConstraints,
        advanced_selector
    ) -> PackResult:
        """Run advanced variant (V1-V3)."""
        
        if advanced_selector is None:
            raise ValueError(f"Advanced selector required for {variant_id} but not provided")
        
        # Map variant ID to selection variant enum
        from ..selector.base import SelectionVariant, SelectionConfig, SelectionMode
        
        variant_mapping = {
            'V1': SelectionVariant.COMPREHENSIVE,
            'V2': SelectionVariant.COVERAGE_ENHANCED,
            'V3': SelectionVariant.STABILITY_CONTROLLED,
        }
        
        if variant_id not in variant_mapping:
            raise ValueError(f"Unknown advanced variant: {variant_id}")
        
        # Create selection configuration
        selection_config = SelectionConfig(
            mode=SelectionMode.COMPREHENSION,
            variant=variant_mapping[variant_id],
            token_budget=budget_constraints.target_budget,
            deterministic=config.deterministic_mode,
            random_seed=config.random_seed,
        )
        
        # Create pack request
        pack_request = PackRequest(
            chunks=chunks,
            config=selection_config,
        )
        
        # Run selection
        pack_result = advanced_selector.select(pack_request)
        
        return pack_result
    
    def validate_deterministic_consistency(
        self,
        chunks: List[Chunk],
        config: ParityConfig,
        num_runs: int = 3
    ) -> Dict[str, bool]:
        """Validate that variants produce consistent results across multiple runs."""
        logger.info(f"Validating deterministic consistency with {num_runs} runs")
        
        consistency_results = {}
        
        for variant_id in config.variants_to_test:
            logger.info(f"Testing consistency for {variant_id}...")
            
            hashes = []
            
            for run_idx in range(num_runs):
                try:
                    # Create temporary config for this run
                    run_config = ParityConfig(
                        target_budget=config.target_budget,
                        variants_to_test=[variant_id],
                        deterministic_mode=True,
                        random_seed=config.random_seed,  # Same seed for consistency
                    )
                    
                    # Run single evaluation
                    result = self.run_parity_evaluation(chunks, run_config)
                    
                    if variant_id in result.deterministic_hashes:
                        hashes.append(result.deterministic_hashes[variant_id])
                    else:
                        hashes.append(None)
                        
                except Exception as e:
                    logger.error(f"Consistency test run {run_idx + 1} failed for {variant_id}: {e}")
                    hashes.append(None)
            
            # Check consistency
            valid_hashes = [h for h in hashes if h is not None]
            if len(valid_hashes) >= 2:
                all_same = all(h == valid_hashes[0] for h in valid_hashes)
                consistency_results[variant_id] = all_same
                
                if all_same:
                    logger.info(f"{variant_id} is deterministically consistent")
                else:
                    logger.warning(f"{variant_id} is NOT deterministically consistent")
            else:
                consistency_results[variant_id] = False
                logger.warning(f"{variant_id} failed too many consistency runs")
        
        return consistency_results
    
    def export_parity_report(
        self,
        result: ParityEvaluationResult,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export comprehensive parity evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_files = {}
        
        # Export budget report
        budget_report_path = output_dir / "budget_parity_report.json"
        self.budget_enforcer.export_budget_report(budget_report_path)
        report_files["budget_report"] = budget_report_path
        
        # Export comparison report
        comparison_report_path = output_dir / "comparison_analysis.json"
        self.comparison_framework.export_comparison_report(result.comparison_result, comparison_report_path)
        report_files["comparison_report"] = comparison_report_path
        
        # Export performance summary
        performance_path = output_dir / "performance_summary.json"
        import json
        with open(performance_path, 'w') as f:
            json.dump({
                "performance_summary": result.get_performance_summary(),
                "execution_times": result.execution_times,
                "deterministic_hashes": result.deterministic_hashes,
                "failed_variants": result.failed_variants,
            }, f, indent=2)
        report_files["performance_summary"] = performance_path
        
        logger.info(f"Parity evaluation report exported to {output_dir}")
        
        return report_files
    
    def _log_evaluation_summary(self, result: ParityEvaluationResult):
        """Log evaluation summary."""
        perf = result.get_performance_summary()
        
        logger.info("Parity Evaluation Summary:")
        logger.info(f"  Variants: {perf['successful_variants']}/{perf['total_variants']} successful")
        logger.info(f"  Budget compliance: {result.budget_compliance_rate:.1%}")
        logger.info(f"  Total time: {result.total_execution_time:.1f}s")
        
        if result.failed_variants:
            logger.warning(f"  Failed variants: {result.failed_variants}")
        
        # Log comparison highlights
        if result.comparison_result.efficiency_rankings:
            best_variant = result.comparison_result.efficiency_rankings[0]
            logger.info(f"  Best efficiency: {best_variant['variant_id']} ({best_variant['efficiency']:.3f})")
    
    def clear_cache(self):
        """Clear all cached data."""
        self._baseline_cache.clear()
        self.budget_enforcer.clear_reports()
        self.comparison_framework.clear_cache()
        logger.info("Cleared parity controller cache")
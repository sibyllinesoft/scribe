"""Comparison framework for analyzing variant performance at parity."""

from __future__ import annotations

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .budget_enforcer import BudgetReport
from ..selector.base import PackResult

logger = logging.getLogger(__name__)


@dataclass
class VariantMetrics:
    """Comprehensive metrics for a single variant."""
    
    variant_id: str
    
    # Budget metrics
    token_efficiency: float  # QA accuracy per 100k tokens (placeholder)
    budget_utilization: float
    
    # Selection metrics  
    total_chunks: int
    coverage_score: float
    diversity_score: float
    
    # Performance metrics
    execution_time: float
    memory_peak: float
    
    # Quality metrics
    deterministic: bool
    budget_compliant: bool
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metric summary."""
        return {
            "variant_id": self.variant_id,
            "token_efficiency": round(self.token_efficiency, 3),
            "budget_utilization": round(self.budget_utilization, 3),
            "total_chunks": self.total_chunks,
            "coverage_score": round(self.coverage_score, 3),
            "diversity_score": round(self.diversity_score, 3),
            "execution_time": round(self.execution_time, 2),
            "memory_peak": round(self.memory_peak, 1),
            "deterministic": self.deterministic,
            "budget_compliant": self.budget_compliant,
        }


@dataclass
class ComparisonResult:
    """Results from comparative analysis."""
    
    # Variant metrics
    variant_metrics: Dict[str, VariantMetrics]
    
    # Rankings
    efficiency_rankings: List[Dict[str, Any]]  # Sorted by token efficiency
    performance_rankings: List[Dict[str, Any]]  # Sorted by execution time
    
    # Statistical analysis
    statistical_significance: Dict[str, Any]
    
    # Parity validation
    budget_compliance_rate: float
    all_compliant: bool
    
    # Comparative insights
    best_baseline: str  # Best performing baseline (V0a/V0b/V0c)
    best_advanced: str  # Best performing advanced variant (V1/V2/V3)
    
    # Improvement analysis
    baseline_vs_advanced: Dict[str, float]  # Improvement percentages
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comparison summary."""
        return {
            "total_variants": len(self.variant_metrics),
            "budget_compliance_rate": round(self.budget_compliance_rate, 3),
            "all_compliant": self.all_compliant,
            "best_baseline": self.best_baseline,
            "best_advanced": self.best_advanced,
            "efficiency_leader": self.efficiency_rankings[0]["variant_id"] if self.efficiency_rankings else None,
            "performance_leader": self.performance_rankings[0]["variant_id"] if self.performance_rankings else None,
        }


class ComparisonFramework:
    """
    Framework for comparing variant performance at budget parity.
    
    Provides statistical analysis, ranking, and insight generation
    for fair comparison of baseline and advanced variants.
    """
    
    def __init__(self):
        """Initialize comparison framework."""
        self._cached_results: Dict[str, ComparisonResult] = {}
    
    def compare_variants(
        self,
        pack_results: Dict[str, PackResult],
        budget_reports: Dict[str, BudgetReport]
    ) -> ComparisonResult:
        """
        Compare variants with comprehensive analysis.
        
        Args:
            pack_results: Dictionary of variant_id -> PackResult
            budget_reports: Dictionary of variant_id -> BudgetReport
            
        Returns:
            Comprehensive comparison result
        """
        logger.info(f"Comparing {len(pack_results)} variants")
        
        # Extract metrics for each variant
        variant_metrics = {}
        for variant_id, pack_result in pack_results.items():
            budget_report = budget_reports.get(variant_id)
            metrics = self._extract_variant_metrics(variant_id, pack_result, budget_report)
            variant_metrics[variant_id] = metrics
        
        # Create rankings
        efficiency_rankings = self._create_efficiency_rankings(variant_metrics)
        performance_rankings = self._create_performance_rankings(variant_metrics)
        
        # Statistical analysis
        statistical_significance = self._analyze_statistical_significance(variant_metrics)
        
        # Parity validation
        budget_compliance_rate = sum(1 for r in budget_reports.values() if r.within_tolerance) / max(1, len(budget_reports))
        all_compliant = all(r.within_tolerance for r in budget_reports.values())
        
        # Find best variants by category
        best_baseline, best_advanced = self._find_category_leaders(variant_metrics)
        
        # Improvement analysis
        baseline_vs_advanced = self._analyze_improvements(variant_metrics)
        
        # Create comparison result
        result = ComparisonResult(
            variant_metrics=variant_metrics,
            efficiency_rankings=efficiency_rankings,
            performance_rankings=performance_rankings,
            statistical_significance=statistical_significance,
            budget_compliance_rate=budget_compliance_rate,
            all_compliant=all_compliant,
            best_baseline=best_baseline,
            best_advanced=best_advanced,
            baseline_vs_advanced=baseline_vs_advanced,
        )
        
        # Cache result
        cache_key = "_".join(sorted(pack_results.keys()))
        self._cached_results[cache_key] = result
        
        # Log summary
        self._log_comparison_summary(result)
        
        return result
    
    def _extract_variant_metrics(
        self,
        variant_id: str,
        pack_result: PackResult,
        budget_report: Optional[BudgetReport]
    ) -> VariantMetrics:
        """Extract comprehensive metrics from pack result."""
        
        # Calculate token efficiency (placeholder - would use actual QA results)
        # For now, use a simple proxy based on coverage and diversity
        coverage = pack_result.selection.coverage_score
        diversity = pack_result.selection.diversity_score
        token_efficiency = (coverage + diversity) / 2.0  # Simplified metric
        
        # Determine if variant is deterministic
        is_deterministic = pack_result.deterministic_hash is not None
        
        # Budget compliance
        is_budget_compliant = budget_report.within_tolerance if budget_report else False
        
        return VariantMetrics(
            variant_id=variant_id,
            token_efficiency=token_efficiency,
            budget_utilization=pack_result.selection.budget_utilization,
            total_chunks=len(pack_result.selection.selected_chunks),
            coverage_score=pack_result.selection.coverage_score,
            diversity_score=pack_result.selection.diversity_score,
            execution_time=pack_result.execution_time,
            memory_peak=pack_result.memory_peak,
            deterministic=is_deterministic,
            budget_compliant=is_budget_compliant,
        )
    
    def _create_efficiency_rankings(self, variant_metrics: Dict[str, VariantMetrics]) -> List[Dict[str, Any]]:
        """Create efficiency rankings."""
        rankings = []
        
        for variant_id, metrics in variant_metrics.items():
            rankings.append({
                "variant_id": variant_id,
                "efficiency": metrics.token_efficiency,
                "rank": 0,  # Will be filled after sorting
            })
        
        # Sort by efficiency (descending)
        rankings.sort(key=lambda x: x["efficiency"], reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(rankings):
            entry["rank"] = i + 1
        
        return rankings
    
    def _create_performance_rankings(self, variant_metrics: Dict[str, VariantMetrics]) -> List[Dict[str, Any]]:
        """Create performance (execution time) rankings."""
        rankings = []
        
        for variant_id, metrics in variant_metrics.items():
            rankings.append({
                "variant_id": variant_id,
                "execution_time": metrics.execution_time,
                "rank": 0,  # Will be filled after sorting
            })
        
        # Sort by execution time (ascending - faster is better)
        rankings.sort(key=lambda x: x["execution_time"])
        
        # Assign ranks
        for i, entry in enumerate(rankings):
            entry["rank"] = i + 1
        
        return rankings
    
    def _analyze_statistical_significance(self, variant_metrics: Dict[str, VariantMetrics]) -> Dict[str, Any]:
        """Analyze statistical significance of differences."""
        
        # Extract efficiency scores
        efficiencies = [metrics.token_efficiency for metrics in variant_metrics.values()]
        
        if len(efficiencies) < 2:
            return {"error": "Insufficient data for statistical analysis"}
        
        # Basic statistical analysis
        analysis = {
            "efficiency_stats": {
                "mean": statistics.mean(efficiencies),
                "median": statistics.median(efficiencies),
                "stdev": statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0.0,
                "min": min(efficiencies),
                "max": max(efficiencies),
                "range": max(efficiencies) - min(efficiencies),
            }
        }
        
        # Pairwise comparisons (placeholder for more sophisticated analysis)
        baseline_variants = [v for v in variant_metrics.keys() if v.startswith('V0')]
        advanced_variants = [v for v in variant_metrics.keys() if not v.startswith('V0')]
        
        if baseline_variants and advanced_variants:
            baseline_efficiencies = [variant_metrics[v].token_efficiency for v in baseline_variants]
            advanced_efficiencies = [variant_metrics[v].token_efficiency for v in advanced_variants]
            
            analysis["baseline_vs_advanced"] = {
                "baseline_mean": statistics.mean(baseline_efficiencies),
                "advanced_mean": statistics.mean(advanced_efficiencies),
                "improvement": statistics.mean(advanced_efficiencies) - statistics.mean(baseline_efficiencies),
                "improvement_percent": ((statistics.mean(advanced_efficiencies) - statistics.mean(baseline_efficiencies)) / statistics.mean(baseline_efficiencies)) * 100 if statistics.mean(baseline_efficiencies) > 0 else 0.0,
            }
        
        return analysis
    
    def _find_category_leaders(self, variant_metrics: Dict[str, VariantMetrics]) -> Tuple[str, str]:
        """Find best performing variants by category."""
        
        # Separate baselines and advanced variants
        baselines = {v_id: metrics for v_id, metrics in variant_metrics.items() if v_id.startswith('V0')}
        advanced = {v_id: metrics for v_id, metrics in variant_metrics.items() if not v_id.startswith('V0')}
        
        # Find best baseline
        best_baseline = ""
        if baselines:
            best_baseline = max(baselines.keys(), key=lambda v: baselines[v].token_efficiency)
        
        # Find best advanced
        best_advanced = ""
        if advanced:
            best_advanced = max(advanced.keys(), key=lambda v: advanced[v].token_efficiency)
        
        return best_baseline, best_advanced
    
    def _analyze_improvements(self, variant_metrics: Dict[str, VariantMetrics]) -> Dict[str, float]:
        """Analyze improvements between baseline and advanced variants."""
        
        improvements = {}
        
        # Get baseline reference (prefer V0c as strongest baseline)
        baseline_variants = [v for v in variant_metrics.keys() if v.startswith('V0')]
        if not baseline_variants:
            return improvements
        
        # Use V0c as reference if available, otherwise best baseline
        reference_variant = 'V0c' if 'V0c' in baseline_variants else max(
            baseline_variants, key=lambda v: variant_metrics[v].token_efficiency
        )
        reference_efficiency = variant_metrics[reference_variant].token_efficiency
        
        # Calculate improvements for each advanced variant
        advanced_variants = [v for v in variant_metrics.keys() if not v.startswith('V0')]
        for variant_id in advanced_variants:
            advanced_efficiency = variant_metrics[variant_id].token_efficiency
            
            if reference_efficiency > 0:
                improvement_percent = ((advanced_efficiency - reference_efficiency) / reference_efficiency) * 100
                improvements[f"{variant_id}_vs_{reference_variant}"] = improvement_percent
        
        return improvements
    
    def _log_comparison_summary(self, result: ComparisonResult):
        """Log comparison summary."""
        logger.info("Comparison Analysis Summary:")
        logger.info(f"  Variants: {len(result.variant_metrics)}")
        logger.info(f"  Budget compliance: {result.budget_compliance_rate:.1%}")
        
        if result.efficiency_rankings:
            top_variant = result.efficiency_rankings[0]
            logger.info(f"  Efficiency leader: {top_variant['variant_id']} ({top_variant['efficiency']:.3f})")
        
        if result.best_baseline and result.best_advanced:
            logger.info(f"  Best baseline: {result.best_baseline}")
            logger.info(f"  Best advanced: {result.best_advanced}")
        
        # Log significant improvements
        if result.baseline_vs_advanced:
            for comparison, improvement in result.baseline_vs_advanced.items():
                if improvement > 5.0:  # Log improvements > 5%
                    logger.info(f"  {comparison}: +{improvement:.1f}%")
    
    def export_comparison_report(self, result: ComparisonResult, output_path: Path) -> Dict[str, Any]:
        """Export comprehensive comparison report."""
        import json
        from datetime import datetime
        
        # Prepare detailed report
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": result.get_summary(),
            "variant_metrics": {
                variant_id: metrics.get_summary()
                for variant_id, metrics in result.variant_metrics.items()
            },
            "rankings": {
                "efficiency": result.efficiency_rankings,
                "performance": result.performance_rankings,
            },
            "statistical_analysis": result.statistical_significance,
            "improvement_analysis": result.baseline_vs_advanced,
            "category_leaders": {
                "best_baseline": result.best_baseline,
                "best_advanced": result.best_advanced,
            },
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Comparison report exported to {output_path}")
        
        return report_data["summary"]
    
    def get_variant_comparison(
        self,
        variant_a: str,
        variant_b: str,
        result: ComparisonResult
    ) -> Dict[str, Any]:
        """Get detailed comparison between two specific variants."""
        
        if variant_a not in result.variant_metrics or variant_b not in result.variant_metrics:
            return {"error": "One or both variants not found in results"}
        
        metrics_a = result.variant_metrics[variant_a]
        metrics_b = result.variant_metrics[variant_b]
        
        # Calculate differences
        efficiency_diff = metrics_b.token_efficiency - metrics_a.token_efficiency
        efficiency_percent = (efficiency_diff / metrics_a.token_efficiency) * 100 if metrics_a.token_efficiency > 0 else 0.0
        
        time_diff = metrics_b.execution_time - metrics_a.execution_time
        time_percent = (time_diff / metrics_a.execution_time) * 100 if metrics_a.execution_time > 0 else 0.0
        
        return {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "efficiency_comparison": {
                "variant_a": metrics_a.token_efficiency,
                "variant_b": metrics_b.token_efficiency,
                "difference": efficiency_diff,
                "percent_change": efficiency_percent,
                "winner": variant_b if efficiency_diff > 0 else variant_a,
            },
            "performance_comparison": {
                "variant_a": metrics_a.execution_time,
                "variant_b": metrics_b.execution_time,
                "difference": time_diff,
                "percent_change": time_percent,
                "winner": variant_b if time_diff < 0 else variant_a,  # Faster is better
            },
            "quality_comparison": {
                "coverage_diff": metrics_b.coverage_score - metrics_a.coverage_score,
                "diversity_diff": metrics_b.diversity_score - metrics_a.diversity_score,
                "chunk_count_diff": metrics_b.total_chunks - metrics_a.total_chunks,
            }
        }
    
    def clear_cache(self):
        """Clear cached comparison results."""
        self._cached_results.clear()
        logger.info("Cleared comparison framework cache")
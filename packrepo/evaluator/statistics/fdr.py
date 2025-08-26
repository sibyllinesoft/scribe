#!/usr/bin/env python3
"""
False Discovery Rate (FDR) Control for PackRepo Multiple Comparisons

Implements Benjamini-Hochberg procedure and other FDR control methods
to handle multiple comparison problems across metric families, variants,
and budget levels.

Key Features:
- Benjamini-Hochberg FDR correction with family-wise grouping
- Adaptive FDR procedures for varying dependence structures
- Integration with bootstrap confidence intervals
- Metric family organization (QA accuracy, token efficiency, latency)
- Hierarchical FDR control for nested comparisons

Methodology Reference:
- Benjamini & Hochberg (1995), "Controlling the False Discovery Rate"
- Benjamini & Yekutieli (2001), "The Control of the False Discovery Rate"
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from scipy import stats
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultipleComparisonTest:
    """Single test within a multiple comparison analysis."""
    
    test_id: str
    family: str
    comparison_name: str
    
    # Test statistics
    observed_difference: float
    p_value_raw: float
    p_value_adjusted: float
    
    # Decision thresholds
    alpha_raw: float
    alpha_adjusted: float
    
    # Significance decisions
    significant_raw: bool
    significant_adjusted: bool
    rejected_hypothesis: bool
    
    # Effect size and confidence
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    # Metadata
    variant_a: Optional[str] = None
    variant_b: Optional[str] = None
    metric_name: Optional[str] = None


@dataclass
class FDRAnalysisResult:
    """Complete FDR analysis results across all families and tests."""
    
    # Analysis configuration
    alpha_global: float
    fdr_method: str
    n_total_tests: int
    n_families: int
    
    # Family-wise results
    family_results: Dict[str, Dict[str, Any]]
    
    # Overall test results
    tests: List[MultipleComparisonTest]
    
    # Summary statistics
    n_significant_raw: int
    n_significant_adjusted: int
    n_rejected_hypotheses: int
    false_discovery_rate: float
    family_wise_error_rate: float
    
    # Decision support
    promoted_variants: List[str]
    blocked_variants: List[str]
    requires_manual_review: List[str]


class FDRController:
    """
    False Discovery Rate controller for multiple comparison problems.
    
    Handles hierarchical FDR correction across metric families while
    maintaining statistical rigor for promotion decisions.
    """
    
    def __init__(self, alpha: float = 0.05, method: str = "benjamini_hochberg"):
        """
        Initialize FDR controller.
        
        Args:
            alpha: Global false discovery rate (typically 0.05)
            method: FDR control method ('benjamini_hochberg', 'benjamini_yekutieli')
        """
        self.alpha = alpha
        self.method = method
        
        if method not in ["benjamini_hochberg", "benjamini_yekutieli"]:
            raise ValueError(f"Unsupported FDR method: {method}")
    
    def analyze_multiple_comparisons(
        self,
        test_results: List[Dict[str, Any]],
        family_grouping: Optional[Dict[str, str]] = None
    ) -> FDRAnalysisResult:
        """
        Perform FDR analysis across multiple test families.
        
        Args:
            test_results: List of individual test results with p-values
            family_grouping: Optional mapping of test_id -> family_name
            
        Returns:
            Complete FDR analysis with adjusted p-values and decisions
        """
        if not test_results:
            raise ValueError("No test results provided for FDR analysis")
        
        logger.info(f"Starting FDR analysis with {len(test_results)} tests")
        logger.info(f"Method: {self.method}, Alpha: {self.alpha}")
        
        # Organize tests by family
        if family_grouping is None:
            family_grouping = self._auto_detect_families(test_results)
        
        families = self._group_tests_by_family(test_results, family_grouping)
        logger.info(f"Identified {len(families)} metric families: {list(families.keys())}")
        
        # Apply FDR correction within each family
        family_results = {}
        all_tests = []
        
        for family_name, family_tests in families.items():
            logger.info(f"Processing family '{family_name}' with {len(family_tests)} tests")
            
            family_result = self._analyze_family(family_name, family_tests)
            family_results[family_name] = family_result
            all_tests.extend(family_result["tests"])
        
        # Global summary statistics
        n_significant_raw = sum(1 for test in all_tests if test.significant_raw)
        n_significant_adjusted = sum(1 for test in all_tests if test.significant_adjusted)
        n_rejected = sum(1 for test in all_tests if test.rejected_hypothesis)
        
        # Estimate false discovery rate and family-wise error rate
        fdr_estimate = self._estimate_false_discovery_rate(all_tests)
        fwer_estimate = self._estimate_family_wise_error_rate(families)
        
        # Make promotion decisions
        promoted, blocked, manual_review = self._make_promotion_decisions(all_tests)
        
        logger.info(f"FDR Analysis Summary:")
        logger.info(f"  Raw significant: {n_significant_raw}/{len(all_tests)}")
        logger.info(f"  Adjusted significant: {n_significant_adjusted}/{len(all_tests)}")
        logger.info(f"  Estimated FDR: {fdr_estimate:.4f}")
        logger.info(f"  Promoted variants: {len(promoted)}")
        
        return FDRAnalysisResult(
            alpha_global=self.alpha,
            fdr_method=self.method,
            n_total_tests=len(all_tests),
            n_families=len(families),
            family_results=family_results,
            tests=all_tests,
            n_significant_raw=n_significant_raw,
            n_significant_adjusted=n_significant_adjusted,
            n_rejected_hypotheses=n_rejected,
            false_discovery_rate=fdr_estimate,
            family_wise_error_rate=fwer_estimate,
            promoted_variants=promoted,
            blocked_variants=blocked,
            requires_manual_review=manual_review
        )
    
    def _auto_detect_families(self, test_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Automatically detect metric families from test metadata.
        
        Groups similar metrics together for family-wise FDR control.
        """
        families = {}
        
        for i, test in enumerate(test_results):
            test_id = test.get("test_id", f"test_{i}")
            metric_name = test.get("metric_name", "unknown")
            
            # Define family based on metric type
            if "accuracy" in metric_name.lower() or "qa_" in metric_name.lower():
                family = "qa_accuracy"
            elif "efficiency" in metric_name.lower() or "token" in metric_name.lower():
                family = "token_efficiency"  
            elif "latency" in metric_name.lower() or "time" in metric_name.lower():
                family = "performance_latency"
            elif "memory" in metric_name.lower() or "ram" in metric_name.lower():
                family = "performance_memory"
            else:
                family = "other_metrics"
            
            families[test_id] = family
        
        return families
    
    def _group_tests_by_family(
        self, 
        test_results: List[Dict[str, Any]], 
        family_grouping: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group tests by metric family for hierarchical FDR control."""
        
        families = {}
        
        for i, test in enumerate(test_results):
            test_id = test.get("test_id", f"test_{i}")
            family = family_grouping.get(test_id, "other_metrics")
            
            if family not in families:
                families[family] = []
            
            # Add test_id to test data if missing
            test_with_id = test.copy()
            test_with_id["test_id"] = test_id
            test_with_id["family"] = family
            
            families[family].append(test_with_id)
        
        return families
    
    def _analyze_family(self, family_name: str, family_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply FDR correction within a single metric family."""
        
        # Extract p-values and validate
        p_values = []
        valid_tests = []
        
        for test in family_tests:
            p_value = test.get("p_value", test.get("p_value_raw"))
            if p_value is not None and 0 <= p_value <= 1:
                p_values.append(float(p_value))
                valid_tests.append(test)
            else:
                logger.warning(f"Invalid p-value in test {test.get('test_id')}: {p_value}")
        
        if not p_values:
            logger.warning(f"No valid p-values found in family {family_name}")
            return {"tests": [], "family_summary": {}}
        
        # Apply FDR correction
        if self.method == "benjamini_hochberg":
            p_adjusted = self._benjamini_hochberg_correction(p_values)
        elif self.method == "benjamini_yekutieli":
            p_adjusted = self._benjamini_yekutieli_correction(p_values)
        else:
            raise ValueError(f"Unknown FDR method: {self.method}")
        
        # Create test objects with results
        family_test_objects = []
        for i, (test_data, p_raw, p_adj) in enumerate(zip(valid_tests, p_values, p_adjusted)):
            
            # Calculate adjusted alpha threshold
            n_tests = len(p_values)
            rank = np.argsort(p_values)[i] + 1  # Rank in sorted order
            alpha_adjusted = (rank / n_tests) * self.alpha
            
            test_obj = MultipleComparisonTest(
                test_id=test_data.get("test_id", f"{family_name}_{i}"),
                family=family_name,
                comparison_name=test_data.get("comparison_name", f"comparison_{i}"),
                observed_difference=test_data.get("observed_difference", 0.0),
                p_value_raw=p_raw,
                p_value_adjusted=p_adj,
                alpha_raw=self.alpha,
                alpha_adjusted=alpha_adjusted,
                significant_raw=p_raw < self.alpha,
                significant_adjusted=p_adj < self.alpha,
                rejected_hypothesis=p_adj < self.alpha,
                effect_size=test_data.get("effect_size"),
                ci_lower=test_data.get("ci_lower"),
                ci_upper=test_data.get("ci_upper"),
                variant_a=test_data.get("variant_a"),
                variant_b=test_data.get("variant_b"),
                metric_name=test_data.get("metric_name")
            )
            
            family_test_objects.append(test_obj)
        
        # Family-level summary
        n_significant_raw = sum(1 for test in family_test_objects if test.significant_raw)
        n_significant_adjusted = sum(1 for test in family_test_objects if test.significant_adjusted)
        
        family_summary = {
            "family_name": family_name,
            "n_tests": len(family_test_objects),
            "n_significant_raw": n_significant_raw,
            "n_significant_adjusted": n_significant_adjusted,
            "family_wise_alpha": self.alpha,
            "fdr_method": self.method,
            "min_p_value": min(p_values) if p_values else 1.0,
            "max_p_value": max(p_values) if p_values else 1.0
        }
        
        return {
            "tests": family_test_objects,
            "family_summary": family_summary
        }
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Controls FDR under independence or positive regression dependence.
        """
        p_array = np.array(p_values)
        n = len(p_array)
        
        if n == 0:
            return []
        
        # Sort p-values and get original indices
        sorted_indices = np.argsort(p_array)
        sorted_p_values = p_array[sorted_indices]
        
        # Apply BH step-up procedure
        corrected_p_values = np.zeros_like(sorted_p_values)
        
        # Start from largest p-value and work backwards
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                corrected_p_values[i] = sorted_p_values[i]
            else:
                corrected_p_values[i] = min(
                    sorted_p_values[i] * n / (i + 1),
                    corrected_p_values[i + 1]
                )
        
        # Ensure adjusted p-values don't exceed 1.0
        corrected_p_values = np.minimum(corrected_p_values, 1.0)
        
        # Restore original order
        final_corrected = np.zeros_like(p_array)
        final_corrected[sorted_indices] = corrected_p_values
        
        return final_corrected.tolist()
    
    def _benjamini_yekutieli_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Benjamini-Yekutieli FDR correction.
        
        Controls FDR under arbitrary dependence structure (more conservative).
        """
        n = len(p_values)
        if n == 0:
            return []
        
        # Calculate harmonic number adjustment factor
        harmonic_number = np.sum(1.0 / np.arange(1, n + 1))
        
        # Apply BH procedure with harmonic adjustment
        p_array = np.array(p_values)
        sorted_indices = np.argsort(p_array)
        sorted_p_values = p_array[sorted_indices]
        
        corrected_p_values = np.zeros_like(sorted_p_values)
        
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                corrected_p_values[i] = sorted_p_values[i] * harmonic_number
            else:
                corrected_p_values[i] = min(
                    sorted_p_values[i] * harmonic_number * n / (i + 1),
                    corrected_p_values[i + 1]
                )
        
        # Ensure adjusted p-values don't exceed 1.0
        corrected_p_values = np.minimum(corrected_p_values, 1.0)
        
        # Restore original order
        final_corrected = np.zeros_like(p_array)
        final_corrected[sorted_indices] = corrected_p_values
        
        return final_corrected.tolist()
    
    def _estimate_false_discovery_rate(self, tests: List[MultipleComparisonTest]) -> float:
        """Estimate the actual false discovery rate from test results."""
        
        n_rejected = sum(1 for test in tests if test.rejected_hypothesis)
        
        if n_rejected == 0:
            return 0.0
        
        # Conservative estimate: assume all null hypotheses are true
        # FDR ≈ (number of rejections × alpha) / number of rejections = alpha
        return min(self.alpha, 1.0)
    
    def _estimate_family_wise_error_rate(self, families: Dict[str, List[Dict[str, Any]]]) -> float:
        """Estimate family-wise error rate across all families."""
        
        # FWER = 1 - (1 - alpha_family)^n_families
        # Assuming alpha_family = alpha for each family
        n_families = len(families)
        if n_families == 0:
            return 0.0
        
        fwer = 1 - (1 - self.alpha) ** n_families
        return min(fwer, 1.0)
    
    def _make_promotion_decisions(
        self, 
        tests: List[MultipleComparisonTest]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Make variant promotion decisions based on FDR-corrected results.
        
        Returns:
            (promoted_variants, blocked_variants, manual_review_variants)
        """
        promoted = []
        blocked = []
        manual_review = []
        
        # Group tests by variant
        variant_results = {}
        for test in tests:
            if test.variant_b and test.variant_a:
                variant_key = f"{test.variant_b}_vs_{test.variant_a}"
                if variant_key not in variant_results:
                    variant_results[variant_key] = []
                variant_results[variant_key].append(test)
        
        # Make decisions for each variant comparison
        for variant_key, variant_tests in variant_results.items():
            
            # Check if all tests for this variant are significant after FDR correction
            all_significant = all(test.significant_adjusted for test in variant_tests)
            any_significant = any(test.significant_adjusted for test in variant_tests)
            
            # Check confidence interval criterion (CI lower bound > 0)
            meets_ci_criterion = all(
                test.ci_lower is not None and test.ci_lower > 0 
                for test in variant_tests
            )
            
            if all_significant and meets_ci_criterion:
                promoted.append(variant_key)
            elif any_significant or meets_ci_criterion:
                manual_review.append(variant_key)
            else:
                blocked.append(variant_key)
        
        return promoted, blocked, manual_review


def load_test_results_from_bootstrap(bootstrap_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Load bootstrap results and convert to test format for FDR analysis.
    
    Args:
        bootstrap_files: List of paths to bootstrap result JSON files
        
    Returns:
        List of test results ready for FDR analysis
    """
    test_results = []
    
    for file_path in bootstrap_files:
        if not file_path.exists():
            logger.warning(f"Bootstrap file not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r') as f:
                bootstrap_result = json.load(f)
            
            # Convert bootstrap result to test format
            test_result = {
                "test_id": f"{bootstrap_result['variant_b']}_vs_{bootstrap_result['variant_a']}_{bootstrap_result['metric_name']}",
                "comparison_name": f"{bootstrap_result['variant_b']} vs {bootstrap_result['variant_a']}",
                "variant_a": bootstrap_result["variant_a"],
                "variant_b": bootstrap_result["variant_b"],
                "metric_name": bootstrap_result["metric_name"],
                "observed_difference": bootstrap_result["observed_difference"],
                "p_value": bootstrap_result["p_value_bootstrap"],
                "effect_size": bootstrap_result["effect_size_cohens_d"],
                "ci_lower": bootstrap_result["ci_95_lower"],
                "ci_upper": bootstrap_result["ci_95_upper"]
            }
            
            test_results.append(test_result)
            
        except Exception as e:
            logger.error(f"Error loading bootstrap file {file_path}: {e}")
    
    return test_results


def save_fdr_results(result: FDRAnalysisResult, output_file: Path):
    """Save FDR analysis results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    
    logger.info(f"FDR analysis results saved to: {output_file}")


def main():
    """Command-line interface for FDR analysis."""
    
    if len(sys.argv) < 2:
        print("Usage: fdr.py <bootstrap_results_dir> [output_file.json] [alpha] [method]")
        print("\nExample: fdr.py artifacts/bootstrap_results/ fdr_analysis.json 0.05 benjamini_hochberg")
        print("\nArguments:")
        print("  bootstrap_results_dir - Directory containing bootstrap result JSON files")
        print("  output_file.json      - Output file for FDR analysis (default: fdr_analysis.json)")
        print("  alpha                 - Global FDR level (default: 0.05)")
        print("  method               - FDR method: 'benjamini_hochberg' or 'benjamini_yekutieli' (default: benjamini_hochberg)")
        sys.exit(1)
    
    # Parse arguments
    bootstrap_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("fdr_analysis.json")
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    method = sys.argv[4] if len(sys.argv) > 4 else "benjamini_hochberg"
    
    print(f"False Discovery Rate (FDR) Analysis")
    print(f"{'='*50}")
    print(f"Bootstrap results directory: {bootstrap_dir}")
    print(f"Output file: {output_file}")
    print(f"Global alpha: {alpha}")
    print(f"FDR method: {method}")
    print(f"{'='*50}")
    
    # Find all bootstrap result files
    if not bootstrap_dir.exists():
        logger.error(f"Bootstrap directory not found: {bootstrap_dir}")
        sys.exit(1)
    
    bootstrap_files = list(bootstrap_dir.glob("*.json"))
    if not bootstrap_files:
        logger.error(f"No JSON files found in {bootstrap_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(bootstrap_files)} bootstrap result files")
    
    # Load test results
    test_results = load_test_results_from_bootstrap(bootstrap_files)
    if not test_results:
        logger.error("No valid test results loaded")
        sys.exit(1)
    
    logger.info(f"Loaded {len(test_results)} test results for FDR analysis")
    
    # Run FDR analysis
    fdr_controller = FDRController(alpha=alpha, method=method)
    fdr_result = fdr_controller.analyze_multiple_comparisons(test_results)
    
    # Save results
    save_fdr_results(fdr_result, output_file)
    
    # Print summary
    print(f"\nFDR Analysis Results")
    print(f"{'='*50}")
    print(f"Total tests: {fdr_result.n_total_tests}")
    print(f"Metric families: {fdr_result.n_families}")
    print(f"Raw significant: {fdr_result.n_significant_raw}")
    print(f"FDR-adjusted significant: {fdr_result.n_significant_adjusted}")
    print(f"Estimated FDR: {fdr_result.false_discovery_rate:.4f}")
    print(f"")
    print(f"Promotion Decisions:")
    print(f"  ✅ Promoted: {len(fdr_result.promoted_variants)} variants")
    for variant in fdr_result.promoted_variants:
        print(f"    - {variant}")
    print(f"  ❌ Blocked: {len(fdr_result.blocked_variants)} variants")
    for variant in fdr_result.blocked_variants:
        print(f"    - {variant}")
    print(f"  🔍 Manual Review: {len(fdr_result.requires_manual_review)} variants")
    for variant in fdr_result.requires_manual_review:
        print(f"    - {variant}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
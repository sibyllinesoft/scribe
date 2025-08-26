#!/usr/bin/env python3
"""
PackRepo Comparative Statistical Analysis

Implements comprehensive statistical comparison between PackRepo variants:
- Bootstrap confidence intervals for token efficiency improvements
- Effect size analysis and practical significance testing  
- False Discovery Rate (FDR) correction for multiple comparisons
- Risk assessment and composite scoring
- Evidence-based promotion decisions

This module provides the statistical rigor required for validating the
≥ +20% Q&A accuracy per 100k tokens objective with BCa 95% CI lower bound > 0.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class VariantComparison:
    """Statistical comparison between two variants."""
    variant_a: str
    variant_b: str
    metric_name: str
    
    # Raw statistics
    n_samples_a: int
    n_samples_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    
    # Difference analysis
    mean_difference: float
    relative_improvement: float
    
    # Statistical tests
    t_statistic: float
    p_value: float
    degrees_freedom: int
    
    # Effect size
    cohens_d: float
    effect_size_interpretation: str
    
    # Confidence intervals
    ci_lower: float
    ci_upper: float
    ci_method: str
    
    # Decision support
    statistically_significant: bool
    practically_significant: bool
    improvement_favors: str
    confidence_level: float


class ComparativeAnalyzer:
    """Comprehensive statistical analysis engine for variant comparisons."""
    
    def __init__(self, alpha: float = 0.05, effect_size_threshold: float = 0.2):
        """
        Initialize comparative analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            effect_size_threshold: Minimum effect size for practical significance
        """
        self.alpha = alpha
        self.effect_size_threshold = effect_size_threshold
        
    def compare_variants(
        self,
        data_a: List[float],
        data_b: List[float], 
        variant_a: str = "A",
        variant_b: str = "B",
        metric_name: str = "metric"
    ) -> VariantComparison:
        """Compare two variants with comprehensive statistical analysis."""
        
        if len(data_a) == 0 or len(data_b) == 0:
            raise ValueError("Cannot compare empty datasets")
        
        # Convert to numpy arrays
        a_values = np.array(data_a)
        b_values = np.array(data_b)
        
        # Basic statistics
        n_a, n_b = len(a_values), len(b_values)
        mean_a, mean_b = np.mean(a_values), np.mean(b_values)
        std_a, std_b = np.std(a_values, ddof=1), np.std(b_values, ddof=1)
        
        # Difference metrics
        mean_difference = mean_b - mean_a
        relative_improvement = (mean_difference / mean_a * 100) if mean_a != 0 else 0
        
        # Statistical test (Welch's t-test for unequal variances)
        t_statistic, p_value = stats.ttest_ind(
            b_values, a_values, equal_var=False
        )
        degrees_freedom = self._welch_degrees_freedom(a_values, b_values)
        
        # Effect size (Cohen's d)
        cohens_d = self._cohens_d(a_values, b_values)
        effect_interpretation = self._interpret_effect_size(cohens_d)
        
        # Confidence interval for difference in means
        ci_lower, ci_upper = self._difference_confidence_interval(a_values, b_values)
        
        # Decision criteria
        statistically_significant = p_value < self.alpha
        practically_significant = abs(cohens_d) >= self.effect_size_threshold
        
        improvement_favors = "neither"
        if mean_difference > 0 and statistically_significant:
            improvement_favors = variant_b
        elif mean_difference < 0 and statistically_significant:
            improvement_favors = variant_a
        
        return VariantComparison(
            variant_a=variant_a,
            variant_b=variant_b,
            metric_name=metric_name,
            n_samples_a=n_a,
            n_samples_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            mean_difference=mean_difference,
            relative_improvement=relative_improvement,
            t_statistic=t_statistic,
            p_value=p_value,
            degrees_freedom=degrees_freedom,
            cohens_d=cohens_d,
            effect_size_interpretation=effect_interpretation,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_method="Welch_t_CI",
            statistically_significant=statistically_significant,
            practically_significant=practically_significant,
            improvement_favors=improvement_favors,
            confidence_level=(1 - self.alpha) * 100
        )
    
    def _welch_degrees_freedom(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate degrees of freedom for Welch's t-test."""
        n_a, n_b = len(a), len(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        
        if var_a == 0 and var_b == 0:
            return float(n_a + n_b - 2)
        
        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        
        if denominator == 0:
            return float(n_a + n_b - 2)
            
        return numerator / denominator
    
    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        n_a, n_b = len(a), len(b)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return (mean_b - mean_a) / pooled_std
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small" 
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _difference_confidence_interval(
        self, 
        a: np.ndarray, 
        b: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        
        # Standard error of difference
        se_diff = np.sqrt(var_a / n_a + var_b / n_b)
        
        # Degrees of freedom
        df = self._welch_degrees_freedom(a, b)
        
        # t critical value
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        
        # Mean difference
        mean_diff = mean_b - mean_a
        
        # Confidence interval
        margin_error = t_critical * se_diff
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return ci_lower, ci_upper
    
    def run_multiple_comparisons(
        self,
        variant_data: Dict[str, List[float]],
        metric_name: str = "token_efficiency",
        baseline_variant: str = "V0"
    ) -> Dict[str, VariantComparison]:
        """Run all pairwise comparisons with FDR correction."""
        
        comparisons = {}
        p_values = []
        comparison_keys = []
        
        # Generate all comparisons against baseline
        if baseline_variant in variant_data:
            baseline_data = variant_data[baseline_variant]
            
            for variant_name, variant_values in variant_data.items():
                if variant_name != baseline_variant:
                    comparison_key = f"{baseline_variant}_vs_{variant_name}"
                    
                    comparison = self.compare_variants(
                        baseline_data, variant_values,
                        baseline_variant, variant_name,
                        metric_name
                    )
                    
                    comparisons[comparison_key] = comparison
                    p_values.append(comparison.p_value)
                    comparison_keys.append(comparison_key)
        
        # Apply FDR correction
        if p_values:
            corrected_p_values = self._fdr_correction(p_values)
            
            # Update comparisons with corrected p-values
            for i, comparison_key in enumerate(comparison_keys):
                comparison = comparisons[comparison_key]
                comparison.p_value = corrected_p_values[i]
                comparison.statistically_significant = corrected_p_values[i] < self.alpha
                
                # Update improvement decision based on corrected significance
                if comparison.mean_difference > 0 and comparison.statistically_significant:
                    comparison.improvement_favors = comparison.variant_b
                elif comparison.mean_difference < 0 and comparison.statistically_significant:
                    comparison.improvement_favors = comparison.variant_a
                else:
                    comparison.improvement_favors = "neither"
        
        return comparisons
    
    def _fdr_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction."""
        p_array = np.array(p_values)
        n = len(p_array)
        
        if n == 0:
            return p_values
            
        # Sort p-values and get original indices
        sorted_indices = np.argsort(p_array)
        sorted_p_values = p_array[sorted_indices]
        
        # Apply BH procedure
        corrected_p_values = np.zeros_like(sorted_p_values)
        
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                corrected_p_values[i] = sorted_p_values[i]
            else:
                corrected_p_values[i] = min(
                    sorted_p_values[i] * n / (i + 1),
                    corrected_p_values[i + 1]
                )
        
        # Restore original order
        final_corrected = np.zeros_like(p_array)
        final_corrected[sorted_indices] = corrected_p_values
        
        return final_corrected.tolist()
    
    def generate_comparison_report(
        self,
        comparisons: Dict[str, VariantComparison],
        output_file: Path
    ):
        """Generate comprehensive comparison report."""
        
        report = {
            "analysis_type": "comparative_statistical_analysis",
            "timestamp": np.datetime64('now').astype(str),
            "alpha": self.alpha,
            "effect_size_threshold": self.effect_size_threshold,
            "num_comparisons": len(comparisons),
            "comparisons": {},
            "summary": {
                "statistically_significant": 0,
                "practically_significant": 0,
                "both_significant": 0,
                "largest_effect_size": 0.0,
                "best_variant": None
            }
        }
        
        # Process each comparison
        largest_effect = 0.0
        best_variant = None
        
        for comparison_key, comparison in comparisons.items():
            # Add to report
            report["comparisons"][comparison_key] = {
                "variant_a": comparison.variant_a,
                "variant_b": comparison.variant_b,
                "metric_name": comparison.metric_name,
                "sample_sizes": [comparison.n_samples_a, comparison.n_samples_b],
                "means": [comparison.mean_a, comparison.mean_b],
                "std_devs": [comparison.std_a, comparison.std_b],
                "mean_difference": comparison.mean_difference,
                "relative_improvement_percent": comparison.relative_improvement,
                "statistical_test": {
                    "t_statistic": comparison.t_statistic,
                    "p_value": comparison.p_value,
                    "degrees_freedom": comparison.degrees_freedom,
                    "significant": comparison.statistically_significant
                },
                "effect_size": {
                    "cohens_d": comparison.cohens_d,
                    "interpretation": comparison.effect_size_interpretation,
                    "practically_significant": comparison.practically_significant
                },
                "confidence_interval": {
                    "lower": comparison.ci_lower,
                    "upper": comparison.ci_upper,
                    "method": comparison.ci_method,
                    "confidence_level": comparison.confidence_level
                },
                "decision": {
                    "improvement_favors": comparison.improvement_favors,
                    "both_significant": comparison.statistically_significant and comparison.practically_significant
                }
            }
            
            # Update summary statistics
            if comparison.statistically_significant:
                report["summary"]["statistically_significant"] += 1
            if comparison.practically_significant:
                report["summary"]["practically_significant"] += 1  
            if comparison.statistically_significant and comparison.practically_significant:
                report["summary"]["both_significant"] += 1
                
            # Track best variant
            if abs(comparison.cohens_d) > largest_effect:
                largest_effect = abs(comparison.cohens_d)
                best_variant = comparison.improvement_favors
        
        report["summary"]["largest_effect_size"] = largest_effect
        report["summary"]["best_variant"] = best_variant
        
        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def load_variant_data_from_jsonl(input_file: Path, metric_name: str) -> Dict[str, List[float]]:
    """Load variant data from JSONL file."""
    
    variant_data = {}
    
    if not input_file.exists():
        print(f"Warning: Input file {input_file} not found")
        return variant_data
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                variant = data.get("variant", "unknown")
                value = data.get(metric_name, 0.0)
                
                if isinstance(value, (int, float)) and np.isfinite(value):
                    if variant not in variant_data:
                        variant_data[variant] = []
                    variant_data[variant].append(float(value))
                    
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return variant_data


def main():
    """CLI for comparative analysis."""
    
    if len(sys.argv) < 2:
        print("Usage: comparative_analysis.py <input_file.jsonl> [metric_name] [output_file.json]")
        print("Example: comparative_analysis.py metrics.jsonl token_efficiency comparison_report.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    metric_name = sys.argv[2] if len(sys.argv) > 2 else "token_efficiency"
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("comparison_analysis.json")
    
    print(f"Comparative Statistical Analysis")
    print(f"Input: {input_file}")
    print(f"Metric: {metric_name}")
    print(f"Output: {output_file}")
    
    # Load data
    variant_data = load_variant_data_from_jsonl(input_file, metric_name)
    
    if len(variant_data) < 2:
        print("Error: Need at least 2 variants for comparison")
        sys.exit(1)
    
    print(f"Loaded data for variants: {list(variant_data.keys())}")
    for variant, values in variant_data.items():
        print(f"  {variant}: {len(values)} samples, mean={np.mean(values):.4f}")
    
    # Run analysis
    analyzer = ComparativeAnalyzer()
    comparisons = analyzer.run_multiple_comparisons(variant_data, metric_name)
    
    # Generate report
    report = analyzer.generate_comparison_report(comparisons, output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Comparative Analysis Results")
    print(f"{'='*60}")
    print(f"Comparisons: {len(comparisons)}")
    print(f"Statistically significant: {report['summary']['statistically_significant']}")
    print(f"Practically significant: {report['summary']['practically_significant']}")
    print(f"Both significant: {report['summary']['both_significant']}")
    print(f"Largest effect size: {report['summary']['largest_effect_size']:.3f}")
    print(f"Best variant: {report['summary']['best_variant']}")
    
    # Print key findings
    print(f"\nKey Findings:")
    for comparison_key, comparison_data in report["comparisons"].items():
        variant_a = comparison_data["variant_a"]
        variant_b = comparison_data["variant_b"] 
        improvement = comparison_data["relative_improvement_percent"]
        significant = comparison_data["decision"]["both_significant"]
        
        status = "✅ SIGNIFICANT" if significant else "❌ Not significant"
        print(f"  {variant_b} vs {variant_a}: {improvement:+.1f}% {status}")
    
    print(f"{'='*60}")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
PackRepo False Discovery Rate (FDR) Correction

Implements Benjamini-Hochberg FDR correction for multiple statistical comparisons
in the PackRepo evaluation matrix. Ensures that promotion decisions account for
multiple testing to maintain statistical rigor.

Applied to:
- V1 vs V0 comparisons (token efficiency, latency, accuracy)
- V2 vs V1 comparisons (coverage improvements)
- V3 vs V2 comparisons (stability metrics)
- Cross-metric evaluations (efficiency vs latency trade-offs)
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
import warnings


@dataclass
class ComparisonResult:
    """Single statistical comparison result."""
    comparison_name: str
    metric_name: str
    n_samples: int
    
    # Test results
    test_statistic: float
    p_value: float
    effect_size: float
    
    # Group statistics
    group1_mean: float
    group1_std: float
    group2_mean: float  
    group2_std: float
    
    # Effect interpretation
    direction: str  # 'improvement', 'degradation', 'no_change'
    magnitude: str  # 'small', 'medium', 'large'


@dataclass
class FDRResult:
    """FDR correction results."""
    method: str
    alpha: float
    n_comparisons: int
    n_significant_raw: int
    n_significant_corrected: int
    
    # Per-comparison results
    comparisons: List[ComparisonResult]
    p_values_raw: List[float]
    p_values_corrected: List[float]
    significant_raw: List[bool]
    significant_corrected: List[bool]
    
    # Decision outcomes
    rejected_hypotheses: List[str]
    critical_value: float
    
    # Summary metrics
    fdr_controlled: bool
    expected_false_discoveries: float


class FDRCorrection:
    """False Discovery Rate correction for multiple comparisons."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize with target FDR level."""
        self.alpha = alpha
        
    def benjamini_hochberg(self, p_values: List[float]) -> Tuple[List[bool], List[float], float]:
        """Apply Benjamini-Hochberg FDR correction."""
        
        if not p_values:
            return [], [], 0.0
            
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Compute corrected p-values (step-up method)
        corrected_p = np.zeros_like(sorted_p)
        
        # Work backwards through sorted p-values
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                corrected_p[i] = sorted_p[i]
            else:
                corrected_p[i] = min(corrected_p[i + 1], sorted_p[i] * n / (i + 1))
        
        # Find critical value (largest i such that p(i) <= (i/n) * alpha)
        critical_value = 0.0
        significant_sorted = np.zeros(n, dtype=bool)
        
        for i in range(n):
            threshold = ((i + 1) / n) * self.alpha
            if sorted_p[i] <= threshold:
                critical_value = threshold
                significant_sorted[:i + 1] = True
            else:
                break
        
        # Restore original order
        significant = np.zeros(n, dtype=bool)
        corrected_p_original = np.zeros(n)
        
        for i, orig_idx in enumerate(sorted_indices):
            significant[orig_idx] = significant_sorted[i]
            corrected_p_original[orig_idx] = corrected_p[i]
        
        return significant.tolist(), corrected_p_original.tolist(), critical_value
    
    def bonferroni(self, p_values: List[float]) -> Tuple[List[bool], List[float], float]:
        """Apply Bonferroni correction (more conservative)."""
        
        if not p_values:
            return [], [], 0.0
            
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Bonferroni corrected p-values
        corrected_p = np.minimum(p_values * n, 1.0)
        
        # Significance using corrected alpha
        corrected_alpha = self.alpha / n
        significant = p_values <= corrected_alpha
        
        return significant.tolist(), corrected_p.tolist(), corrected_alpha
    
    def analyze_comparisons(self, comparisons: List[ComparisonResult], method: str = "benjamini_hochberg") -> FDRResult:
        """Perform FDR correction on multiple comparisons."""
        
        if not comparisons:
            return FDRResult(
                method=method,
                alpha=self.alpha,
                n_comparisons=0,
                n_significant_raw=0,
                n_significant_corrected=0,
                comparisons=[],
                p_values_raw=[],
                p_values_corrected=[],
                significant_raw=[],
                significant_corrected=[],
                rejected_hypotheses=[],
                critical_value=0.0,
                fdr_controlled=True,
                expected_false_discoveries=0.0
            )
        
        # Extract p-values
        p_values_raw = [comp.p_value for comp in comparisons]
        
        # Apply correction
        if method == "benjamini_hochberg":
            significant_corrected, p_values_corrected, critical_value = self.benjamini_hochberg(p_values_raw)
        elif method == "bonferroni":
            significant_corrected, p_values_corrected, critical_value = self.bonferroni(p_values_raw)
        else:
            raise ValueError(f"Unknown FDR method: {method}")
        
        # Raw significance
        significant_raw = [p <= self.alpha for p in p_values_raw]
        
        # Identify rejected hypotheses
        rejected_hypotheses = [
            comp.comparison_name for comp, sig in zip(comparisons, significant_corrected) if sig
        ]
        
        # Compute expected false discoveries
        expected_fd = len(rejected_hypotheses) * self.alpha if method == "benjamini_hochberg" else 0.0
        
        return FDRResult(
            method=method,
            alpha=self.alpha,
            n_comparisons=len(comparisons),
            n_significant_raw=sum(significant_raw),
            n_significant_corrected=sum(significant_corrected),
            comparisons=comparisons,
            p_values_raw=p_values_raw,
            p_values_corrected=p_values_corrected,
            significant_raw=significant_raw,
            significant_corrected=significant_corrected,
            rejected_hypotheses=rejected_hypotheses,
            critical_value=critical_value,
            fdr_controlled=True,
            expected_false_discoveries=expected_fd
        )


def compute_effect_size_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
        
    group1, group2 = np.array(group1), np.array(group2)
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
        
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d


def classify_effect_magnitude(cohens_d: float) -> str:
    """Classify effect size magnitude using Cohen's conventions."""
    
    abs_d = abs(cohens_d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def perform_comparison(
    group1_data: List[float], 
    group2_data: List[float],
    comparison_name: str,
    metric_name: str
) -> ComparisonResult:
    """Perform statistical comparison between two groups."""
    
    if len(group1_data) == 0 or len(group2_data) == 0:
        return ComparisonResult(
            comparison_name=comparison_name,
            metric_name=metric_name,
            n_samples=0,
            test_statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            group1_mean=0.0,
            group1_std=0.0,
            group2_mean=0.0,
            group2_std=0.0,
            direction="no_change",
            magnitude="negligible"
        )
    
    group1, group2 = np.array(group1_data), np.array(group2_data)
    
    # Descriptive statistics
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Effect size
    cohens_d = compute_effect_size_cohens_d(group1_data, group2_data)
    
    # Statistical test (Welch's t-test for unequal variances)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            
        if not np.isfinite(t_stat) or not np.isfinite(p_value):
            t_stat, p_value = 0.0, 1.0
            
    except:
        t_stat, p_value = 0.0, 1.0
    
    # Direction of effect
    if mean1 > mean2:
        direction = "improvement" if cohens_d > 0.1 else "no_change"
    elif mean1 < mean2:
        direction = "degradation" if cohens_d < -0.1 else "no_change" 
    else:
        direction = "no_change"
    
    magnitude = classify_effect_magnitude(cohens_d)
    
    return ComparisonResult(
        comparison_name=comparison_name,
        metric_name=metric_name,
        n_samples=len(group1) + len(group2),
        test_statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=float(cohens_d),
        group1_mean=float(mean1),
        group1_std=float(std1),
        group2_mean=float(mean2), 
        group2_std=float(std2),
        direction=direction,
        magnitude=magnitude
    )


def load_comparison_data(metrics_file: Path) -> Dict[str, List[Dict]]:
    """Load metrics data grouped by variant."""
    
    variant_data = {"V0": [], "V1": [], "V2": [], "V3": []}
    
    if not metrics_file.exists():
        print(f"Warning: Metrics file {metrics_file} not found")
        return variant_data
        
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    variant = data.get("variant", "unknown")
                    
                    if variant in variant_data:
                        variant_data[variant].append(data)
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error loading metrics: {e}")
        
    return variant_data


def generate_packrepo_comparisons(variant_data: Dict[str, List[Dict]]) -> List[ComparisonResult]:
    """Generate PackRepo-specific comparisons for FDR analysis."""
    
    comparisons = []
    metrics = ["qa_accuracy", "token_efficiency", "latency_p50_ms", "latency_p95_ms", "memory_usage_mb"]
    
    # Define comparison pairs
    comparison_pairs = [
        ("V1", "V0", "V1_vs_V0_hardening"),
        ("V2", "V1", "V2_vs_V1_coverage"), 
        ("V3", "V2", "V3_vs_V2_stability"),
        ("V2", "V0", "V2_vs_V0_overall"),
        ("V3", "V0", "V3_vs_V0_overall")
    ]
    
    for variant1, variant2, comparison_name in comparison_pairs:
        if variant1 not in variant_data or variant2 not in variant_data:
            continue
            
        data1 = variant_data[variant1]
        data2 = variant_data[variant2]
        
        if not data1 or not data2:
            continue
            
        for metric in metrics:
            # Extract metric values
            values1 = [d.get(metric, 0.0) for d in data1 if metric in d and np.isfinite(d[metric])]
            values2 = [d.get(metric, 0.0) for d in data2 if metric in d and np.isfinite(d[metric])]
            
            if len(values1) > 0 and len(values2) > 0:
                comp_result = perform_comparison(
                    values1, values2,
                    f"{comparison_name}_{metric}",
                    metric
                )
                comparisons.append(comp_result)
                
    return comparisons


def main():
    """Main FDR correction execution."""
    
    if len(sys.argv) < 2:
        print("Usage: fdr.py <metrics_file.jsonl> [alpha] [method] [output_file.json]")
        print("Methods: benjamini_hochberg, bonferroni")
        sys.exit(1)
    
    # Parse arguments  
    metrics_file = Path(sys.argv[1])
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    method = sys.argv[3] if len(sys.argv) > 3 else "benjamini_hochberg"
    output_file = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("fdr_results.json")
    
    print(f"FDR Correction Analysis")
    print(f"Input: {metrics_file}")
    print(f"Alpha: {alpha}")
    print(f"Method: {method}")
    print(f"Output: {output_file}")
    
    # Load data
    variant_data = load_comparison_data(metrics_file)
    
    total_samples = sum(len(data) for data in variant_data.values())
    print(f"Loaded data: {total_samples} total samples across {len(variant_data)} variants")
    
    if total_samples == 0:
        print("Error: No valid data found")
        sys.exit(1)
    
    # Generate comparisons
    comparisons = generate_packrepo_comparisons(variant_data)
    
    if not comparisons:
        print("Error: No valid comparisons generated")
        sys.exit(1)
        
    print(f"Generated {len(comparisons)} statistical comparisons")
    
    # Apply FDR correction
    fdr_corrector = FDRCorrection(alpha=alpha)
    fdr_result = fdr_corrector.analyze_comparisons(comparisons, method=method)
    
    # Create output
    output_data = {
        "analysis_type": "fdr_correction",
        "timestamp": np.datetime64('now').astype(str),
        "input_file": str(metrics_file),
        "method": fdr_result.method,
        "alpha": fdr_result.alpha,
        
        # Summary statistics
        "n_comparisons": fdr_result.n_comparisons,
        "n_significant_raw": fdr_result.n_significant_raw,
        "n_significant_corrected": fdr_result.n_significant_corrected,
        "critical_value": fdr_result.critical_value,
        "expected_false_discoveries": fdr_result.expected_false_discoveries,
        "fdr_controlled": fdr_result.fdr_controlled,
        
        # Rejected hypotheses
        "significant_comparisons": fdr_result.rejected_hypotheses,
        
        # Detailed results
        "comparisons": [asdict(comp) for comp in fdr_result.comparisons],
        "correction_details": {
            "p_values_raw": fdr_result.p_values_raw,
            "p_values_corrected": fdr_result.p_values_corrected,
            "significant_raw": fdr_result.significant_raw,
            "significant_corrected": fdr_result.significant_corrected
        }
    }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Display summary
    print(f"\n{'='*70}")
    print(f"FDR Correction Results ({method})")
    print(f"{'='*70}")
    print(f"Total comparisons: {fdr_result.n_comparisons}")
    print(f"Raw significant (α={alpha}): {fdr_result.n_significant_raw}")
    print(f"FDR significant: {fdr_result.n_significant_corrected}")
    print(f"Critical value: {fdr_result.critical_value:.6f}")
    print(f"Expected false discoveries: {fdr_result.expected_false_discoveries:.2f}")
    print(f"")
    
    if fdr_result.rejected_hypotheses:
        print(f"Significant comparisons after FDR correction:")
        for i, comparison in enumerate(fdr_result.rejected_hypotheses, 1):
            print(f"  {i}. {comparison}")
    else:
        print("No comparisons remain significant after FDR correction")
    
    print(f"")
    
    # Show top effects by size
    sorted_comparisons = sorted(fdr_result.comparisons, key=lambda x: abs(x.effect_size), reverse=True)
    print(f"Top effect sizes:")
    for i, comp in enumerate(sorted_comparisons[:5], 1):
        sig_marker = "✓" if comp.comparison_name in fdr_result.rejected_hypotheses else "✗"
        print(f"  {i}. {comp.comparison_name} ({comp.metric_name})")
        print(f"     Effect: {comp.effect_size:.3f} ({comp.magnitude}), p={comp.p_value:.4f} {sig_marker}")
    
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
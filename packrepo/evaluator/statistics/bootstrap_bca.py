#!/usr/bin/env python3
"""
BCa Bootstrap Implementation for PackRepo Token Efficiency Analysis

Implements Bias-Corrected and Accelerated (BCa) bootstrap methodology
with numerical stability and memory efficiency for 10,000+ bootstrap iterations.

Key Features:
- Paired difference analysis for variant comparisons
- BCa acceleration constant calculation for skewness correction
- Memory-efficient bootstrap sampling for large datasets
- Stable confidence interval calculation with proper percentile interpolation
- Integration with acceptance gates for promotion decisions

Methodology Reference:
- Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
- DiCiccio & Efron (1996), "Bootstrap Confidence Intervals"
"""

import json
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import stats
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress numpy warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class BootstrapResult:
    """Results from BCa bootstrap analysis."""
    
    # Input metadata
    variant_a: str
    variant_b: str
    metric_name: str
    n_bootstrap: int
    
    # Sample statistics
    n_pairs: int
    observed_difference: float
    observed_mean_a: float
    observed_mean_b: float
    
    # Bootstrap distribution
    bootstrap_differences: List[float]
    bootstrap_mean: float
    bootstrap_std: float
    bootstrap_skewness: float
    
    # BCa correction factors
    bias_correction: float
    acceleration: float
    
    # Confidence intervals
    ci_95_lower: float
    ci_95_upper: float
    ci_90_lower: float
    ci_90_upper: float
    ci_99_lower: float
    ci_99_upper: float
    
    # Statistical inference
    p_value_bootstrap: float
    effect_size_cohens_d: float
    
    # Decision support
    ci_95_excludes_zero: bool
    meets_acceptance_gate: bool
    practical_significance: bool


@dataclass 
class PairedBootstrapInput:
    """Input data for paired bootstrap analysis."""
    
    values_a: List[float]
    values_b: List[float]
    variant_a: str = "Baseline"
    variant_b: str = "Treatment"
    metric_name: str = "token_efficiency"
    
    def __post_init__(self):
        """Validate paired input data."""
        if len(self.values_a) != len(self.values_b):
            raise ValueError(
                f"Paired samples must have equal length: "
                f"{len(self.values_a)} vs {len(self.values_b)}"
            )
        
        if len(self.values_a) < 3:
            raise ValueError(f"Need at least 3 paired samples, got {len(self.values_a)}")
        
        # Check for finite values
        if not all(np.isfinite(self.values_a)) or not all(np.isfinite(self.values_b)):
            raise ValueError("All input values must be finite")


class BCaBootstrap:
    """
    Bias-Corrected and Accelerated (BCa) Bootstrap implementation.
    
    Provides robust confidence intervals for paired differences with
    correction for bias and skewness in the bootstrap distribution.
    """
    
    def __init__(self, n_bootstrap: int = 10000, random_state: Optional[int] = None):
        """
        Initialize BCa bootstrap engine.
        
        Args:
            n_bootstrap: Number of bootstrap iterations (minimum 1000, recommended 10000+)
            random_state: Random seed for reproducible results
        """
        if n_bootstrap < 1000:
            logger.warning(f"Low bootstrap count ({n_bootstrap}) may produce unstable results")
        
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)
        
    def analyze_paired_differences(self, input_data: PairedBootstrapInput) -> BootstrapResult:
        """
        Perform BCa bootstrap analysis on paired differences.
        
        Args:
            input_data: Paired sample data with metadata
            
        Returns:
            Complete bootstrap analysis results
        """
        logger.info(f"Starting BCa bootstrap analysis: {input_data.variant_b} vs {input_data.variant_a}")
        logger.info(f"Sample size: {len(input_data.values_a)} pairs, Bootstrap iterations: {self.n_bootstrap}")
        
        # Convert to numpy arrays
        a_values = np.array(input_data.values_a, dtype=float)
        b_values = np.array(input_data.values_b, dtype=float)
        n_pairs = len(a_values)
        
        # Calculate paired differences
        differences = b_values - a_values
        observed_difference = np.mean(differences)
        
        logger.info(f"Observed difference: {observed_difference:.6f}")
        
        # Generate bootstrap distribution
        bootstrap_differences = self._bootstrap_paired_differences(differences)
        
        # Calculate BCa correction factors
        bias_correction = self._calculate_bias_correction(bootstrap_differences, observed_difference)
        acceleration = self._calculate_acceleration(differences)
        
        logger.info(f"BCa corrections - Bias: {bias_correction:.6f}, Acceleration: {acceleration:.6f}")
        
        # Calculate confidence intervals
        ci_levels = [0.90, 0.95, 0.99]
        confidence_intervals = {}
        
        for level in ci_levels:
            ci_lower, ci_upper = self._bca_confidence_interval(
                bootstrap_differences, observed_difference, 
                bias_correction, acceleration, level
            )
            confidence_intervals[level] = (ci_lower, ci_upper)
        
        # Bootstrap p-value (two-tailed)
        p_value_bootstrap = self._bootstrap_p_value(bootstrap_differences)
        
        # Effect size calculation
        pooled_std = np.sqrt(np.var(differences, ddof=1))
        effect_size_cohens_d = observed_difference / pooled_std if pooled_std > 0 else 0.0
        
        # Decision criteria
        ci_95_excludes_zero = (confidence_intervals[0.95][0] > 0) or (confidence_intervals[0.95][1] < 0)
        meets_acceptance_gate = confidence_intervals[0.95][0] > 0  # CI lower bound > 0
        practical_significance = abs(effect_size_cohens_d) >= 0.2  # Small effect size threshold
        
        # Bootstrap distribution statistics
        bootstrap_mean = np.mean(bootstrap_differences)
        bootstrap_std = np.std(bootstrap_differences, ddof=1)
        bootstrap_skewness = stats.skew(bootstrap_differences)
        
        logger.info(f"95% CI: [{confidence_intervals[0.95][0]:.6f}, {confidence_intervals[0.95][1]:.6f}]")
        logger.info(f"Acceptance gate: {'✅ PASS' if meets_acceptance_gate else '❌ FAIL'}")
        
        return BootstrapResult(
            variant_a=input_data.variant_a,
            variant_b=input_data.variant_b,
            metric_name=input_data.metric_name,
            n_bootstrap=self.n_bootstrap,
            n_pairs=n_pairs,
            observed_difference=observed_difference,
            observed_mean_a=np.mean(a_values),
            observed_mean_b=np.mean(b_values),
            bootstrap_differences=bootstrap_differences.tolist(),
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            bootstrap_skewness=bootstrap_skewness,
            bias_correction=bias_correction,
            acceleration=acceleration,
            ci_95_lower=confidence_intervals[0.95][0],
            ci_95_upper=confidence_intervals[0.95][1],
            ci_90_lower=confidence_intervals[0.90][0],
            ci_90_upper=confidence_intervals[0.90][1],
            ci_99_lower=confidence_intervals[0.99][0],
            ci_99_upper=confidence_intervals[0.99][1],
            p_value_bootstrap=p_value_bootstrap,
            effect_size_cohens_d=effect_size_cohens_d,
            ci_95_excludes_zero=ci_95_excludes_zero,
            meets_acceptance_gate=meets_acceptance_gate,
            practical_significance=practical_significance
        )
    
    def _bootstrap_paired_differences(self, differences: np.ndarray) -> np.ndarray:
        """
        Generate bootstrap distribution of paired differences.
        
        Uses memory-efficient batch processing for large bootstrap counts.
        """
        n_pairs = len(differences)
        bootstrap_means = np.zeros(self.n_bootstrap)
        
        # Process in batches to manage memory
        batch_size = min(1000, self.n_bootstrap)
        n_batches = (self.n_bootstrap + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.n_bootstrap)
            current_batch_size = end_idx - start_idx
            
            # Generate bootstrap indices
            bootstrap_indices = self.rng.randint(0, n_pairs, size=(current_batch_size, n_pairs))
            
            # Sample and compute means
            for i in range(current_batch_size):
                bootstrap_sample = differences[bootstrap_indices[i]]
                bootstrap_means[start_idx + i] = np.mean(bootstrap_sample)
        
        return bootstrap_means
    
    def _calculate_bias_correction(self, bootstrap_differences: np.ndarray, observed_difference: float) -> float:
        """
        Calculate bias correction factor for BCa interval.
        
        The bias correction accounts for the difference between the bootstrap
        distribution median and the observed statistic.
        """
        # Proportion of bootstrap values less than observed
        prop_less = np.mean(bootstrap_differences < observed_difference)
        
        # Handle edge cases
        if prop_less <= 0:
            prop_less = 1 / (2 * len(bootstrap_differences))
        elif prop_less >= 1:
            prop_less = 1 - 1 / (2 * len(bootstrap_differences))
        
        # Convert to z-score
        bias_correction = stats.norm.ppf(prop_less)
        
        return bias_correction
    
    def _calculate_acceleration(self, differences: np.ndarray) -> float:
        """
        Calculate acceleration constant using jackknife method.
        
        The acceleration constant corrects for skewness in the bootstrap distribution.
        """
        n_pairs = len(differences)
        
        # Jackknife estimates (leave-one-out)
        jackknife_estimates = np.zeros(n_pairs)
        overall_mean = np.mean(differences)
        
        for i in range(n_pairs):
            # Leave out i-th observation
            jackknife_sample = np.concatenate([differences[:i], differences[i+1:]])
            jackknife_estimates[i] = np.mean(jackknife_sample)
        
        # Mean of jackknife estimates
        jackknife_mean = np.mean(jackknife_estimates)
        
        # Calculate acceleration
        centered_estimates = jackknife_mean - jackknife_estimates
        numerator = np.sum(centered_estimates ** 3)
        denominator = 6 * (np.sum(centered_estimates ** 2)) ** 1.5
        
        if abs(denominator) < 1e-12:
            return 0.0
        
        acceleration = numerator / denominator
        
        # Numerical stability check
        if not np.isfinite(acceleration):
            logger.warning("Acceleration calculation resulted in non-finite value, using 0.0")
            return 0.0
        
        return acceleration
    
    def _bca_confidence_interval(
        self, 
        bootstrap_differences: np.ndarray,
        observed_difference: float,
        bias_correction: float,
        acceleration: float,
        confidence_level: float
    ) -> Tuple[float, float]:
        """
        Calculate BCa confidence interval with bias and acceleration corrections.
        """
        alpha = 1 - confidence_level
        
        # Standard normal percentiles
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        
        # BCa adjusted percentiles
        def adjusted_percentile(z):
            numerator = bias_correction + z
            denominator = 1 - acceleration * (bias_correction + z)
            
            # Numerical stability
            if abs(denominator) < 1e-12:
                adjusted_z = bias_correction + z
            else:
                adjusted_z = bias_correction + numerator / denominator
            
            # Convert back to percentile, with bounds checking
            percentile = stats.norm.cdf(adjusted_z)
            percentile = max(0.0001, min(0.9999, percentile))  # Avoid extreme percentiles
            
            return percentile
        
        # Calculate adjusted percentiles
        lower_percentile = adjusted_percentile(z_alpha_2)
        upper_percentile = adjusted_percentile(z_1_alpha_2)
        
        # Extract confidence interval from bootstrap distribution
        ci_lower = np.percentile(bootstrap_differences, lower_percentile * 100)
        ci_upper = np.percentile(bootstrap_differences, upper_percentile * 100)
        
        return ci_lower, ci_upper
    
    def _bootstrap_p_value(self, bootstrap_differences: np.ndarray) -> float:
        """
        Calculate two-tailed bootstrap p-value.
        
        P-value represents the probability of observing a difference
        as extreme or more extreme than observed, assuming null hypothesis.
        """
        # Two-tailed test: proportion of |bootstrap_diff| >= |observed_diff|
        observed_abs = abs(np.mean(bootstrap_differences))
        bootstrap_abs = np.abs(bootstrap_differences)
        
        p_value = np.mean(bootstrap_abs >= observed_abs)
        
        # Ensure minimum p-value based on bootstrap iterations
        min_p_value = 1 / self.n_bootstrap
        p_value = max(p_value, min_p_value)
        
        return p_value


def load_paired_data_from_jsonl(
    input_file: Path, 
    metric_name: str,
    variant_a: str,
    variant_b: str,
    question_id_field: str = "question_id"
) -> Optional[PairedBootstrapInput]:
    """
    Load paired comparison data from JSONL evaluation results.
    
    Args:
        input_file: Path to JSONL file with evaluation results
        metric_name: Metric to analyze (e.g., 'qa_accuracy_per_100k')
        variant_a: Baseline variant name
        variant_b: Treatment variant name
        question_id_field: Field name for question identifier
        
    Returns:
        Paired input data ready for bootstrap analysis
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return None
    
    # Parse JSONL data
    variant_a_data = {}
    variant_b_data = {}
    
    try:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    variant = data.get("variant")
                    question_id = data.get(question_id_field)
                    metric_value = data.get(metric_name)
                    
                    if not all([variant, question_id, metric_value is not None]):
                        continue
                    
                    if not np.isfinite(metric_value):
                        logger.warning(f"Non-finite value at line {line_num}: {metric_value}")
                        continue
                    
                    # Store data by variant and question
                    if variant == variant_a:
                        variant_a_data[question_id] = float(metric_value)
                    elif variant == variant_b:
                        variant_b_data[question_id] = float(metric_value)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading file {input_file}: {e}")
        return None
    
    # Find matching question IDs
    common_questions = set(variant_a_data.keys()) & set(variant_b_data.keys())
    
    if len(common_questions) < 3:
        logger.error(
            f"Insufficient paired samples: {len(common_questions)} "
            f"(need at least 3 for bootstrap)"
        )
        return None
    
    # Create paired lists
    values_a = []
    values_b = []
    
    for question_id in sorted(common_questions):
        values_a.append(variant_a_data[question_id])
        values_b.append(variant_b_data[question_id])
    
    logger.info(
        f"Loaded {len(values_a)} paired samples for {variant_b} vs {variant_a} "
        f"on metric '{metric_name}'"
    )
    
    return PairedBootstrapInput(
        values_a=values_a,
        values_b=values_b,
        variant_a=variant_a,
        variant_b=variant_b,
        metric_name=metric_name
    )


def save_bootstrap_results(result: BootstrapResult, output_file: Path):
    """Save bootstrap analysis results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    
    logger.info(f"Bootstrap results saved to: {output_file}")


def main():
    """Command-line interface for BCa bootstrap analysis."""
    
    if len(sys.argv) < 5:
        print("Usage: bootstrap_bca.py <input_file.jsonl> <metric_name> <variant_a> <variant_b> [output_file.json] [n_bootstrap]")
        print("\nExample: bootstrap_bca.py metrics.jsonl qa_accuracy_per_100k V0 V1 bootstrap_results.json 10000")
        print("\nRequired arguments:")
        print("  input_file.jsonl  - Evaluation results in JSONL format")
        print("  metric_name       - Metric to analyze (e.g., 'qa_accuracy_per_100k')")
        print("  variant_a         - Baseline variant name")
        print("  variant_b         - Treatment variant name")
        print("\nOptional arguments:")
        print("  output_file.json  - Output file for results (default: bootstrap_results.json)")
        print("  n_bootstrap       - Number of bootstrap iterations (default: 10000)")
        sys.exit(1)
    
    # Parse arguments
    input_file = Path(sys.argv[1])
    metric_name = sys.argv[2]
    variant_a = sys.argv[3]
    variant_b = sys.argv[4]
    output_file = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("bootstrap_results.json")
    n_bootstrap = int(sys.argv[6]) if len(sys.argv) > 6 else 10000
    
    print(f"BCa Bootstrap Analysis")
    print(f"{'='*50}")
    print(f"Input file: {input_file}")
    print(f"Metric: {metric_name}")
    print(f"Comparison: {variant_b} vs {variant_a}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Output: {output_file}")
    print(f"{'='*50}")
    
    # Load paired data
    paired_data = load_paired_data_from_jsonl(input_file, metric_name, variant_a, variant_b)
    if paired_data is None:
        logger.error("Failed to load paired data")
        sys.exit(1)
    
    # Run BCa bootstrap analysis
    bootstrap_engine = BCaBootstrap(n_bootstrap=n_bootstrap, random_state=42)
    result = bootstrap_engine.analyze_paired_differences(paired_data)
    
    # Save results
    save_bootstrap_results(result, output_file)
    
    # Print summary
    print(f"\nBootstrap Analysis Results")
    print(f"{'='*50}")
    print(f"Sample size: {result.n_pairs} paired observations")
    print(f"Observed difference: {result.observed_difference:.6f}")
    print(f"95% BCa CI: [{result.ci_95_lower:.6f}, {result.ci_95_upper:.6f}]")
    print(f"Effect size (Cohen's d): {result.effect_size_cohens_d:.3f}")
    print(f"Bootstrap p-value: {result.p_value_bootstrap:.6f}")
    print(f"")
    print(f"Decision Criteria:")
    print(f"  CI excludes zero: {'✅ Yes' if result.ci_95_excludes_zero else '❌ No'}")
    print(f"  Acceptance gate (CI lower > 0): {'✅ PASS' if result.meets_acceptance_gate else '❌ FAIL'}")
    print(f"  Practical significance: {'✅ Yes' if result.practical_significance else '❌ No'}")
    print(f"{'='*50}")
    
    if result.meets_acceptance_gate:
        print("🎉 PROMOTION ELIGIBLE: Confidence interval lower bound > 0")
    else:
        print("⚠️  PROMOTION BLOCKED: Confidence interval includes or is below zero")


if __name__ == "__main__":
    main()
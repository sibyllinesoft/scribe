#!/usr/bin/env python3
"""
BCa Bootstrap Implementation with FDR Control for FastPath Evaluation.

Implements paired difference analysis with:
- Bias-Corrected and Accelerated (BCa) bootstrap confidence intervals
- False Discovery Rate (FDR) control using Benjamini-Hochberg procedure
- Paired evaluation support for baseline vs. variant comparisons
- Conservative statistical validation for promotion decisions

Reference:
- Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
- Benjamini & Hochberg (1995), "Controlling the false discovery rate"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress numerical warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class PairedBootstrapResult:
    """Results from paired BCa bootstrap analysis."""
    
    # Comparison metadata
    baseline_system: str
    experimental_system: str
    metric_name: str
    budget: int
    
    # Sample statistics
    n_pairs: int
    observed_difference: float
    observed_difference_pct: float
    baseline_mean: float
    experimental_mean: float
    
    # Bootstrap results
    n_bootstrap: int
    bootstrap_mean: float
    bootstrap_std: float
    
    # BCa confidence interval
    ci_lower: float
    ci_upper: float
    confidence_level: float
    
    # Statistical significance
    p_value: float
    significant_raw: bool
    significant_fdr: bool
    fdr_adjusted_alpha: float
    
    # Quality metrics
    effective_sample_size: int
    bootstrap_convergence: bool
    bias_correction: float
    acceleration: float


class PairedBootstrapAnalyzer:
    """Paired bootstrap analysis with BCa confidence intervals and FDR control."""
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def load_results(self, baseline_file: Path, experimental_file: Path) -> Tuple[List[Dict], List[Dict]]:
        """Load paired evaluation results from JSONL files."""
        baseline_results = []
        experimental_results = []
        
        # Load baseline results
        with open(baseline_file, 'r') as f:
            for line in f:
                baseline_results.append(json.loads(line.strip()))
                
        # Load experimental results  
        with open(experimental_file, 'r') as f:
            for line in f:
                experimental_results.append(json.loads(line.strip()))
                
        logger.info(f"Loaded {len(baseline_results)} baseline and {len(experimental_results)} experimental results")
        
        return baseline_results, experimental_results
        
    def create_paired_samples(
        self, 
        baseline_results: List[Dict], 
        experimental_results: List[Dict],
        metric: str = "qa_score"
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Create paired samples matching by budget and seed."""
        
        # Index baseline results
        baseline_index = {}
        for result in baseline_results:
            key = (result['budget'], result['seed'])
            baseline_index[key] = result[metric]
            
        # Create pairs
        paired_samples = {}
        for result in experimental_results:
            key = (result['budget'], result['seed'])
            if key in baseline_index:
                paired_samples[key] = (baseline_index[key], result[metric])
                
        logger.info(f"Created {len(paired_samples)} paired samples")
        return paired_samples
        
    def compute_bca_bootstrap(
        self,
        baseline_values: np.ndarray,
        experimental_values: np.ndarray,
        metric_name: str,
        budget: int
    ) -> PairedBootstrapResult:
        """Compute BCa bootstrap confidence interval for paired differences."""
        
        n_pairs = len(baseline_values)
        if n_pairs < 10:
            raise ValueError(f"Insufficient paired samples: {n_pairs} < 10")
            
        # Compute observed difference
        differences = experimental_values - baseline_values
        observed_diff = np.mean(differences)
        observed_diff_pct = (observed_diff / np.mean(baseline_values)) * 100
        
        logger.info(f"Observed difference: {observed_diff:.4f} ({observed_diff_pct:+.2f}%)")
        
        # Bootstrap resampling
        bootstrap_diffs = []
        rng = np.random.RandomState(42)  # Reproducible results
        
        for i in range(self.n_bootstrap):
            # Sample with replacement
            indices = rng.choice(n_pairs, size=n_pairs, replace=True)
            bootstrap_baseline = baseline_values[indices]
            bootstrap_experimental = experimental_values[indices]
            bootstrap_diff = np.mean(bootstrap_experimental - bootstrap_baseline)
            bootstrap_diffs.append(bootstrap_diff)
            
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Bias correction
        n_less = np.sum(bootstrap_diffs < observed_diff)
        bias_correction = stats.norm.ppf(n_less / self.n_bootstrap) if n_less > 0 else 0.0
        
        # Acceleration constant via jackknife
        acceleration = self._compute_acceleration(baseline_values, experimental_values)
        
        # BCa confidence interval
        z_alpha_2 = stats.norm.ppf(self.alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - self.alpha / 2)
        
        alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha_2) / 
                                (1 - acceleration * (bias_correction + z_alpha_2)))
        alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / 
                                (1 - acceleration * (bias_correction + z_1_alpha_2)))
        
        # Handle edge cases
        alpha_1 = np.clip(alpha_1, 0.001, 0.999)
        alpha_2 = np.clip(alpha_2, 0.001, 0.999)
        
        ci_lower = np.percentile(bootstrap_diffs, alpha_1 * 100)
        ci_upper = np.percentile(bootstrap_diffs, alpha_2 * 100)
        
        # P-value (two-tailed test)
        p_value = 2 * min(
            np.mean(bootstrap_diffs <= 0),
            np.mean(bootstrap_diffs >= 0)
        )
        
        # Convergence check
        convergence = self._check_convergence(bootstrap_diffs)
        
        return PairedBootstrapResult(
            baseline_system="baseline",
            experimental_system="experimental", 
            metric_name=metric_name,
            budget=budget,
            n_pairs=n_pairs,
            observed_difference=observed_diff,
            observed_difference_pct=observed_diff_pct,
            baseline_mean=np.mean(baseline_values),
            experimental_mean=np.mean(experimental_values),
            n_bootstrap=self.n_bootstrap,
            bootstrap_mean=np.mean(bootstrap_diffs),
            bootstrap_std=np.std(bootstrap_diffs),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            p_value=p_value,
            significant_raw=p_value < self.alpha,
            significant_fdr=False,  # Set by FDR procedure
            fdr_adjusted_alpha=self.alpha,  # Set by FDR procedure
            effective_sample_size=n_pairs,
            bootstrap_convergence=convergence,
            bias_correction=bias_correction,
            acceleration=acceleration
        )
        
    def _compute_acceleration(self, baseline_values: np.ndarray, experimental_values: np.ndarray) -> float:
        """Compute acceleration constant via jackknife."""
        n = len(baseline_values)
        differences = experimental_values - baseline_values
        
        # Jackknife estimates
        jackknife_means = []
        for i in range(n):
            # Leave-one-out
            jackknife_diffs = np.concatenate([differences[:i], differences[i+1:]])
            jackknife_means.append(np.mean(jackknife_diffs))
            
        jackknife_means = np.array(jackknife_means)
        overall_mean = np.mean(jackknife_means)
        
        # Acceleration constant
        numerator = np.sum((overall_mean - jackknife_means) ** 3)
        denominator = 6 * (np.sum((overall_mean - jackknife_means) ** 2) ** 1.5)
        
        if denominator == 0:
            return 0.0
            
        acceleration = numerator / denominator
        return acceleration
        
    def _check_convergence(self, bootstrap_samples: np.ndarray) -> bool:
        """Check bootstrap convergence using running mean stability."""
        n_samples = len(bootstrap_samples)
        if n_samples < 1000:
            return False
            
        # Check if running mean stabilizes
        running_means = np.cumsum(bootstrap_samples) / np.arange(1, n_samples + 1)
        
        # Compare last 10% vs previous 10%
        n_check = n_samples // 10
        last_segment = running_means[-n_check:]
        prev_segment = running_means[-2*n_check:-n_check]
        
        relative_change = np.abs(np.mean(last_segment) - np.mean(prev_segment)) / np.abs(np.mean(prev_segment))
        
        return relative_change < 0.01  # 1% relative change threshold
        
    def apply_fdr_control(self, results: List[PairedBootstrapResult], fdr_level: float = 0.05) -> List[PairedBootstrapResult]:
        """Apply Benjamini-Hochberg FDR control to multiple comparisons."""
        if not results:
            return results
            
        # Sort by p-value
        sorted_results = sorted(results, key=lambda x: x.p_value)
        n_tests = len(sorted_results)
        
        # Apply Benjamini-Hochberg procedure
        for i, result in enumerate(sorted_results):
            bh_threshold = fdr_level * (i + 1) / n_tests
            result.fdr_adjusted_alpha = bh_threshold
            result.significant_fdr = result.p_value <= bh_threshold
            
        logger.info(f"FDR control: {sum(r.significant_fdr for r in results)}/{n_tests} tests significant at FDR={fdr_level}")
        
        return results
        
    def analyze_paired_results(
        self, 
        baseline_file: Path, 
        experimental_file: Path,
        metric: str = "qa_score",
        fdr_level: float = 0.05
    ) -> List[PairedBootstrapResult]:
        """Complete paired bootstrap analysis with FDR control."""
        
        # Load data
        baseline_results, experimental_results = self.load_results(baseline_file, experimental_file)
        
        # Create paired samples
        paired_samples = self.create_paired_samples(baseline_results, experimental_results, metric)
        
        # Group by budget
        budget_groups = {}
        for (budget, seed), (baseline_val, exp_val) in paired_samples.items():
            if budget not in budget_groups:
                budget_groups[budget] = {'baseline': [], 'experimental': []}
            budget_groups[budget]['baseline'].append(baseline_val)
            budget_groups[budget]['experimental'].append(exp_val)
            
        # Analyze each budget separately
        bootstrap_results = []
        for budget in sorted(budget_groups.keys()):
            baseline_vals = np.array(budget_groups[budget]['baseline'])
            experimental_vals = np.array(budget_groups[budget]['experimental'])
            
            logger.info(f"Analyzing budget {budget} with {len(baseline_vals)} pairs")
            
            result = self.compute_bca_bootstrap(
                baseline_vals, experimental_vals, metric, budget
            )
            bootstrap_results.append(result)
            
        # Apply FDR control across all budgets
        bootstrap_results = self.apply_fdr_control(bootstrap_results, fdr_level)
        
        return bootstrap_results


def main():
    parser = argparse.ArgumentParser(
        description="Paired BCa bootstrap analysis with FDR control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python paired_bootstrap.py --baseline artifacts/baseline.jsonl --exp artifacts/v5.jsonl --bca --iters 10000 --fdr --out artifacts/ci.json
        """
    )
    
    parser.add_argument('--baseline', type=str, required=True,
                        help='Baseline results JSONL file')
    parser.add_argument('--exp', type=str, required=True,
                        help='Experimental results JSONL file')
    parser.add_argument('--metric', type=str, default='qa_score',
                        help='Metric to analyze')
    parser.add_argument('--bca', action='store_true',
                        help='Use BCa bootstrap (vs standard percentile)')
    parser.add_argument('--iters', type=int, default=10000,
                        help='Number of bootstrap iterations')
    parser.add_argument('--fdr', action='store_true',
                        help='Apply FDR control for multiple comparisons')
    parser.add_argument('--fdr-level', type=float, default=0.05,
                        help='FDR control level')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level')
    parser.add_argument('--out', type=str, required=True,
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PairedBootstrapAnalyzer(
        n_bootstrap=args.iters,
        confidence_level=args.confidence
    )
    
    # Run analysis
    results = analyzer.analyze_paired_results(
        baseline_file=Path(args.baseline),
        experimental_file=Path(args.exp),
        metric=args.metric,
        fdr_level=args.fdr_level if args.fdr else None
    )
    
    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'analysis_type': 'paired_bca_bootstrap',
            'parameters': {
                'n_bootstrap': args.iters,
                'confidence_level': args.confidence,
                'fdr_control': args.fdr,
                'fdr_level': args.fdr_level if args.fdr else None,
                'metric': args.metric
            },
            'results': [asdict(result) for result in results]
        }, f, indent=2)
        
    logger.info(f"Saved analysis results to {output_path}")
    
    # Print summary
    print(f"\nPaired Bootstrap Analysis Summary:")
    print(f"Metric: {args.metric}")
    print(f"Confidence Level: {args.confidence}")
    print(f"Bootstrap Iterations: {args.iters}")
    if args.fdr:
        print(f"FDR Control Level: {args.fdr_level}")
    print()
    
    for result in results:
        print(f"Budget {result.budget}:")
        print(f"  Observed difference: {result.observed_difference:+.4f} ({result.observed_difference_pct:+.2f}%)")
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"  P-value: {result.p_value:.4f}")
        if args.fdr:
            print(f"  Significant (FDR): {'Yes' if result.significant_fdr else 'No'}")
        else:
            print(f"  Significant: {'Yes' if result.significant_raw else 'No'}")
        print(f"  Convergence: {'Yes' if result.bootstrap_convergence else 'No'}")
        print()


if __name__ == "__main__":
    main()
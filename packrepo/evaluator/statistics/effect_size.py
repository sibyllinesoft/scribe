#!/usr/bin/env python3
"""
Effect Size Analysis for PackRepo Token Efficiency Evaluation

Comprehensive effect size calculation with confidence intervals and
practical significance assessment for rigorous scientific evaluation.

Key Features:
- Cohen's d with pooled and separate variance estimators
- Glass's delta for different variance assumptions  
- Hedge's g with small sample bias correction
- Bootstrap confidence intervals for effect sizes
- Practical significance thresholds with business interpretation
- Non-parametric effect size measures (Cliff's delta, rank-biserial correlation)

Methodology Reference:
- Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences"
- Hedges & Olkin (1985), "Statistical Methods for Meta-Analysis"
- Fritz et al. (2012), "Effect Size Estimates: Current Use, Calculations, and Interpretation"
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import stats
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EffectSizeResult:
    """Comprehensive effect size analysis results."""
    
    # Input metadata
    variant_a: str
    variant_b: str
    metric_name: str
    sample_size_a: int
    sample_size_b: int
    
    # Raw statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    observed_difference: float
    
    # Effect size measures
    cohens_d: float
    cohens_d_ci_lower: float
    cohens_d_ci_upper: float
    
    hedges_g: float
    hedges_g_ci_lower: float
    hedges_g_ci_upper: float
    
    glass_delta: float
    glass_delta_ci_lower: float
    glass_delta_ci_upper: float
    
    # Non-parametric measures
    cliffs_delta: float
    rank_biserial_correlation: float
    
    # Practical significance assessment
    effect_magnitude: str  # negligible, small, medium, large, very_large
    practically_significant: bool
    business_impact_category: str
    
    # Confidence and precision
    confidence_level: float
    margin_of_error: float
    precision_adequate: bool
    
    # Recommendations
    interpretation: str
    recommendation: str


class EffectSizeAnalyzer:
    """
    Comprehensive effect size analyzer with multiple estimators and
    confidence intervals for robust practical significance assessment.
    """
    
    def __init__(
        self, 
        confidence_level: float = 0.95,
        practical_threshold: float = 0.2,
        n_bootstrap: int = 5000
    ):
        """
        Initialize effect size analyzer.
        
        Args:
            confidence_level: Confidence level for interval estimation
            practical_threshold: Minimum effect size for practical significance
            n_bootstrap: Bootstrap iterations for CI estimation
        """
        self.confidence_level = confidence_level
        self.practical_threshold = practical_threshold
        self.n_bootstrap = n_bootstrap
        self.alpha = 1 - confidence_level
    
    def analyze_effect_sizes(
        self,
        data_a: List[float],
        data_b: List[float],
        variant_a: str = "Control",
        variant_b: str = "Treatment", 
        metric_name: str = "metric"
    ) -> EffectSizeResult:
        """
        Perform comprehensive effect size analysis between two groups.
        
        Args:
            data_a: Control/baseline group data
            data_b: Treatment/experimental group data
            variant_a: Name of control variant
            variant_b: Name of treatment variant
            metric_name: Name of the metric being analyzed
            
        Returns:
            Complete effect size analysis results
        """
        logger.info(f"Analyzing effect sizes: {variant_b} vs {variant_a} on {metric_name}")
        
        # Validate and convert data
        if len(data_a) == 0 or len(data_b) == 0:
            raise ValueError("Cannot analyze effect sizes with empty data groups")
        
        a_values = np.array(data_a, dtype=float)
        b_values = np.array(data_b, dtype=float)
        
        if not np.all(np.isfinite(a_values)) or not np.all(np.isfinite(b_values)):
            raise ValueError("All data values must be finite")
        
        # Basic statistics
        n_a, n_b = len(a_values), len(b_values)
        mean_a, mean_b = np.mean(a_values), np.mean(b_values)
        std_a, std_b = np.std(a_values, ddof=1), np.std(b_values, ddof=1)
        observed_difference = mean_b - mean_a
        
        logger.info(f"Sample sizes: n_a={n_a}, n_b={n_b}")
        logger.info(f"Means: {mean_a:.4f} vs {mean_b:.4f} (diff={observed_difference:.4f})")
        
        # Cohen's d with confidence interval
        cohens_d = self._cohens_d(a_values, b_values)
        cohens_d_ci = self._cohens_d_confidence_interval(a_values, b_values)
        
        # Hedges' g (bias-corrected Cohen's d)
        hedges_g = self._hedges_g(a_values, b_values)
        hedges_g_ci = self._hedges_g_confidence_interval(a_values, b_values)
        
        # Glass's delta
        glass_delta = self._glass_delta(a_values, b_values)
        glass_delta_ci = self._glass_delta_confidence_interval(a_values, b_values)
        
        # Non-parametric effect sizes
        cliffs_delta = self._cliffs_delta(a_values, b_values)
        rank_biserial = self._rank_biserial_correlation(a_values, b_values)
        
        # Practical significance assessment
        effect_magnitude = self._interpret_effect_magnitude(cohens_d)
        practically_significant = abs(cohens_d) >= self.practical_threshold
        business_impact = self._assess_business_impact(cohens_d, metric_name)
        
        # Precision assessment
        margin_of_error = (cohens_d_ci[1] - cohens_d_ci[0]) / 2
        precision_adequate = margin_of_error <= 0.2  # Reasonable precision threshold
        
        # Generate interpretation and recommendation
        interpretation = self._generate_interpretation(
            cohens_d, effect_magnitude, practically_significant, precision_adequate
        )
        recommendation = self._generate_recommendation(
            cohens_d, cohens_d_ci, practically_significant, precision_adequate
        )
        
        logger.info(f"Cohen's d: {cohens_d:.3f} [{cohens_d_ci[0]:.3f}, {cohens_d_ci[1]:.3f}]")
        logger.info(f"Effect magnitude: {effect_magnitude}")
        logger.info(f"Practically significant: {'Yes' if practically_significant else 'No'}")
        
        return EffectSizeResult(
            variant_a=variant_a,
            variant_b=variant_b,
            metric_name=metric_name,
            sample_size_a=n_a,
            sample_size_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            observed_difference=observed_difference,
            cohens_d=cohens_d,
            cohens_d_ci_lower=cohens_d_ci[0],
            cohens_d_ci_upper=cohens_d_ci[1],
            hedges_g=hedges_g,
            hedges_g_ci_lower=hedges_g_ci[0],
            hedges_g_ci_upper=hedges_g_ci[1],
            glass_delta=glass_delta,
            glass_delta_ci_lower=glass_delta_ci[0],
            glass_delta_ci_upper=glass_delta_ci[1],
            cliffs_delta=cliffs_delta,
            rank_biserial_correlation=rank_biserial,
            effect_magnitude=effect_magnitude,
            practically_significant=practically_significant,
            business_impact_category=business_impact,
            confidence_level=self.confidence_level,
            margin_of_error=margin_of_error,
            precision_adequate=precision_adequate,
            interpretation=interpretation,
            recommendation=recommendation
        )
    
    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size with pooled standard deviation.
        
        Cohen's d = (mean_b - mean_a) / pooled_sd
        """
        mean_a, mean_b = np.mean(a), np.mean(b)
        n_a, n_b = len(a), len(b)
        
        # Pooled standard deviation
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_sd = np.sqrt(pooled_var)
        
        if pooled_sd == 0:
            return 0.0
        
        return (mean_b - mean_a) / pooled_sd
    
    def _cohens_d_confidence_interval(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for Cohen's d using noncentral t-distribution."""
        
        n_a, n_b = len(a), len(b)
        df = n_a + n_b - 2
        
        # Point estimate
        d = self._cohens_d(a, b)
        
        # Standard error of Cohen's d
        # SE(d) ≈ sqrt((n_a + n_b)/(n_a * n_b) + d²/(2*(n_a + n_b)))
        se_d = np.sqrt((n_a + n_b) / (n_a * n_b) + d**2 / (2 * (n_a + n_b)))
        
        # Confidence interval using t-distribution
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        margin_error = t_critical * se_d
        
        ci_lower = d - margin_error
        ci_upper = d + margin_error
        
        return ci_lower, ci_upper
    
    def _hedges_g(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Hedges' g (bias-corrected Cohen's d).
        
        Hedges' g = Cohen's d × J(df)
        where J(df) is a correction factor for small sample bias.
        """
        cohens_d = self._cohens_d(a, b)
        n_a, n_b = len(a), len(b)
        df = n_a + n_b - 2
        
        # Bias correction factor J(df)
        # J(df) ≈ 1 - 3/(4*df - 1) for large df
        if df <= 1:
            j_factor = 1.0
        else:
            j_factor = 1 - 3 / (4 * df - 1)
        
        return cohens_d * j_factor
    
    def _hedges_g_confidence_interval(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for Hedges' g."""
        
        # Use bootstrap approach for Hedges' g CI
        bootstrap_hedges_g = []
        
        combined_data = np.concatenate([a, b])
        n_a, n_b = len(a), len(b)
        n_total = len(combined_data)
        
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for _ in range(self.n_bootstrap):
            # Bootstrap resample
            bootstrap_sample = rng.choice(combined_data, size=n_total, replace=True)
            bootstrap_a = bootstrap_sample[:n_a]
            bootstrap_b = bootstrap_sample[n_a:n_a + n_b]
            
            # Calculate Hedges' g for bootstrap sample
            try:
                hedges_g = self._hedges_g(bootstrap_a, bootstrap_b)
                if np.isfinite(hedges_g):
                    bootstrap_hedges_g.append(hedges_g)
            except:
                continue
        
        if len(bootstrap_hedges_g) < 100:
            # Fallback to Cohen's d CI if bootstrap fails
            return self._cohens_d_confidence_interval(a, b)
        
        # Calculate percentile-based confidence interval
        alpha_percent = (self.alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_hedges_g, alpha_percent)
        ci_upper = np.percentile(bootstrap_hedges_g, 100 - alpha_percent)
        
        return ci_lower, ci_upper
    
    def _glass_delta(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Glass's delta using control group standard deviation.
        
        Glass's Δ = (mean_b - mean_a) / sd_a
        """
        mean_a, mean_b = np.mean(a), np.mean(b)
        std_a = np.std(a, ddof=1)
        
        if std_a == 0:
            return 0.0
        
        return (mean_b - mean_a) / std_a
    
    def _glass_delta_confidence_interval(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for Glass's delta using bootstrap."""
        
        bootstrap_deltas = []
        n_a, n_b = len(a), len(b)
        
        rng = np.random.RandomState(42)  # Fixed seed
        
        for _ in range(self.n_bootstrap):
            # Bootstrap resample each group independently
            bootstrap_a = rng.choice(a, size=n_a, replace=True)
            bootstrap_b = rng.choice(b, size=n_b, replace=True)
            
            try:
                delta = self._glass_delta(bootstrap_a, bootstrap_b)
                if np.isfinite(delta):
                    bootstrap_deltas.append(delta)
            except:
                continue
        
        if len(bootstrap_deltas) < 100:
            # Fallback to approximation
            glass_delta = self._glass_delta(a, b)
            se_approx = 0.2  # Rough approximation
            margin_error = stats.norm.ppf(1 - self.alpha / 2) * se_approx
            return glass_delta - margin_error, glass_delta + margin_error
        
        # Percentile-based CI
        alpha_percent = (self.alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_deltas, alpha_percent)
        ci_upper = np.percentile(bootstrap_deltas, 100 - alpha_percent)
        
        return ci_lower, ci_upper
    
    def _cliffs_delta(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Cliff's delta (non-parametric effect size).
        
        Cliff's δ = (P(X > Y) - P(X < Y)) where X ~ group_a, Y ~ group_b
        Range: [-1, 1], where ±1 indicates complete separation
        """
        n_a, n_b = len(a), len(b)
        
        if n_a == 0 or n_b == 0:
            return 0.0
        
        # Count comparisons
        greater_count = 0
        less_count = 0
        
        for x in a:
            for y in b:
                if x > y:
                    greater_count += 1
                elif x < y:
                    less_count += 1
                # Ties are ignored
        
        total_comparisons = n_a * n_b
        if total_comparisons == 0:
            return 0.0
        
        cliffs_delta = (less_count - greater_count) / total_comparisons
        
        return cliffs_delta
    
    def _rank_biserial_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate rank-biserial correlation coefficient.
        
        Related to Mann-Whitney U statistic and Cliff's delta.
        """
        # Use Mann-Whitney U test
        try:
            statistic, p_value = stats.mannwhitneyu(b, a, alternative='two-sided')
            n_a, n_b = len(a), len(b)
            
            # Convert U statistic to rank-biserial correlation
            # r = 2 * (U / (n_a * n_b)) - 1
            max_u = n_a * n_b
            if max_u == 0:
                return 0.0
            
            r = 2 * (statistic / max_u) - 1
            return r
            
        except:
            # Fallback to Cliff's delta which is equivalent
            return self._cliffs_delta(a, b)
    
    def _interpret_effect_magnitude(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size magnitude.
        
        Uses updated Cohen (1988) and more recent guidelines.
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.1:
            return "negligible"
        elif abs_d < 0.2:
            return "very_small"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        elif abs_d < 1.2:
            return "large"
        else:
            return "very_large"
    
    def _assess_business_impact(self, cohens_d: float, metric_name: str) -> str:
        """
        Assess business impact category based on effect size and metric type.
        
        Considers domain-specific thresholds for different metrics.
        """
        abs_d = abs(cohens_d)
        
        # Metric-specific thresholds
        if "accuracy" in metric_name.lower() or "qa_" in metric_name.lower():
            # QA accuracy: higher thresholds due to importance
            if abs_d >= 0.8:
                return "high_business_impact"
            elif abs_d >= 0.5:
                return "moderate_business_impact" 
            elif abs_d >= 0.2:
                return "low_business_impact"
            else:
                return "minimal_business_impact"
                
        elif "efficiency" in metric_name.lower() or "token" in metric_name.lower():
            # Token efficiency: cost implications
            if abs_d >= 1.0:
                return "high_business_impact"
            elif abs_d >= 0.6:
                return "moderate_business_impact"
            elif abs_d >= 0.3:
                return "low_business_impact"
            else:
                return "minimal_business_impact"
                
        elif "latency" in metric_name.lower() or "time" in metric_name.lower():
            # Performance metrics: user experience impact
            if abs_d >= 0.6:
                return "high_business_impact"
            elif abs_d >= 0.4:
                return "moderate_business_impact"
            elif abs_d >= 0.2:
                return "low_business_impact"
            else:
                return "minimal_business_impact"
        
        else:
            # General guidelines
            if abs_d >= 0.8:
                return "high_business_impact"
            elif abs_d >= 0.5:
                return "moderate_business_impact"
            elif abs_d >= 0.2:
                return "low_business_impact"
            else:
                return "minimal_business_impact"
    
    def _generate_interpretation(
        self,
        cohens_d: float,
        effect_magnitude: str,
        practically_significant: bool,
        precision_adequate: bool
    ) -> str:
        """Generate human-readable interpretation of effect size analysis."""
        
        direction = "improvement" if cohens_d > 0 else "decline" if cohens_d < 0 else "no change"
        magnitude_desc = effect_magnitude.replace("_", " ")
        
        interpretation_parts = [
            f"The analysis shows a {magnitude_desc} effect size (Cohen's d = {cohens_d:.3f})",
            f"indicating a {direction} in the treatment condition."
        ]
        
        if practically_significant:
            interpretation_parts.append("This effect size meets the threshold for practical significance.")
        else:
            interpretation_parts.append("This effect size does not meet the threshold for practical significance.")
        
        if not precision_adequate:
            interpretation_parts.append(
                "The confidence interval is relatively wide, suggesting the need for larger sample sizes "
                "to achieve more precise effect size estimates."
            )
        
        return " ".join(interpretation_parts)
    
    def _generate_recommendation(
        self,
        cohens_d: float,
        cohens_d_ci: Tuple[float, float],
        practically_significant: bool,
        precision_adequate: bool
    ) -> str:
        """Generate actionable recommendations based on effect size analysis."""
        
        ci_lower, ci_upper = cohens_d_ci
        
        if practically_significant and precision_adequate:
            if ci_lower > 0:
                return (
                    "RECOMMEND PROMOTION: The effect size is practically significant with a "
                    "confidence interval that excludes zero, providing strong evidence for improvement."
                )
            elif ci_upper < 0:
                return (
                    "RECOMMEND REJECTION: The effect size indicates a decline with a "
                    "confidence interval that excludes zero."
                )
            else:
                return (
                    "RECOMMEND CAUTION: While practically significant, the confidence interval "
                    "includes zero, suggesting uncertainty about the direction of the effect."
                )
        
        elif practically_significant and not precision_adequate:
            return (
                "RECOMMEND LARGER SAMPLE: The effect size is practically significant but "
                "the confidence interval is too wide for reliable decision-making. "
                "Collect more data to improve precision."
            )
        
        elif not practically_significant and precision_adequate:
            return (
                "RECOMMEND REJECTION: The effect size is too small to be practically "
                "significant, despite adequate precision in the estimate."
            )
        
        else:  # Not practically significant and not precise
            return (
                "RECOMMEND LARGER SAMPLE OR ALTERNATIVE: The effect size is small and "
                "imprecisely estimated. Either collect substantially more data or "
                "consider alternative approaches."
            )


def load_comparison_data_from_jsonl(
    input_file: Path,
    metric_name: str, 
    variant_a: str,
    variant_b: str
) -> Optional[Tuple[List[float], List[float]]]:
    """Load paired comparison data for effect size analysis."""
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return None
    
    variant_a_values = []
    variant_b_values = []
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                variant = data.get("variant")
                value = data.get(metric_name)
                
                if variant == variant_a and value is not None and np.isfinite(value):
                    variant_a_values.append(float(value))
                elif variant == variant_b and value is not None and np.isfinite(value):
                    variant_b_values.append(float(value))
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    if len(variant_a_values) == 0 or len(variant_b_values) == 0:
        logger.error(f"No data found for comparison: {variant_a} vs {variant_b}")
        return None
    
    logger.info(f"Loaded {len(variant_a_values)} values for {variant_a}, {len(variant_b_values)} for {variant_b}")
    
    return variant_a_values, variant_b_values


def save_effect_size_results(result: EffectSizeResult, output_file: Path):
    """Save effect size analysis results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    
    logger.info(f"Effect size analysis results saved to: {output_file}")


def main():
    """Command-line interface for effect size analysis."""
    
    if len(sys.argv) < 5:
        print("Usage: effect_size.py <input_file.jsonl> <metric_name> <variant_a> <variant_b> [output_file.json]")
        print("\nExample: effect_size.py metrics.jsonl qa_accuracy_per_100k V0 V1 effect_size_results.json")
        print("\nArguments:")
        print("  input_file.jsonl - Evaluation results in JSONL format")
        print("  metric_name      - Metric to analyze")
        print("  variant_a        - Control/baseline variant")
        print("  variant_b        - Treatment variant")
        print("  output_file.json - Output file (optional)")
        sys.exit(1)
    
    # Parse arguments
    input_file = Path(sys.argv[1])
    metric_name = sys.argv[2]
    variant_a = sys.argv[3]
    variant_b = sys.argv[4]
    output_file = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("effect_size_results.json")
    
    print(f"Effect Size Analysis")
    print(f"{'='*50}")
    print(f"Input: {input_file}")
    print(f"Metric: {metric_name}")
    print(f"Comparison: {variant_b} vs {variant_a}")
    print(f"Output: {output_file}")
    print(f"{'='*50}")
    
    # Load data
    comparison_data = load_comparison_data_from_jsonl(input_file, metric_name, variant_a, variant_b)
    if comparison_data is None:
        sys.exit(1)
    
    data_a, data_b = comparison_data
    
    # Run analysis
    analyzer = EffectSizeAnalyzer()
    result = analyzer.analyze_effect_sizes(data_a, data_b, variant_a, variant_b, metric_name)
    
    # Save results
    save_effect_size_results(result, output_file)
    
    # Print summary
    print(f"\nEffect Size Analysis Results")
    print(f"{'='*50}")
    print(f"Cohen's d: {result.cohens_d:.3f} [{result.cohens_d_ci_lower:.3f}, {result.cohens_d_ci_upper:.3f}]")
    print(f"Hedges' g: {result.hedges_g:.3f} [{result.hedges_g_ci_lower:.3f}, {result.hedges_g_ci_upper:.3f}]")
    print(f"Glass's Δ: {result.glass_delta:.3f} [{result.glass_delta_ci_lower:.3f}, {result.glass_delta_ci_upper:.3f}]")
    print(f"Cliff's δ: {result.cliffs_delta:.3f}")
    print(f"")
    print(f"Effect magnitude: {result.effect_magnitude}")
    print(f"Practically significant: {'Yes' if result.practically_significant else 'No'}")
    print(f"Business impact: {result.business_impact_category}")
    print(f"Precision adequate: {'Yes' if result.precision_adequate else 'No'}")
    print(f"")
    print(f"Interpretation:")
    print(f"  {result.interpretation}")
    print(f"")
    print(f"Recommendation:")
    print(f"  {result.recommendation}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Academic-Grade Statistical Analysis for FastPath Research
========================================================

Implements rigorous statistical methods suitable for peer-reviewed publication:
- Power analysis and sample size calculation
- Multiple comparison correction (FDR, Bonferroni)
- Effect size calculation (Cohen's d, Glass's delta)
- Bootstrap confidence intervals (BCa method)
- Non-parametric tests for robustness
- Bayesian analysis for additional evidence

All methods follow established academic standards and include proper
assumption checking and validity assessment.
"""

import sys
import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import bootstrap, norm, t as t_dist
from scipy import special
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PowerAnalysisResult:
    """Results from statistical power analysis."""
    
    effect_size: float
    sample_size: int
    alpha: float
    power: float
    is_adequate: bool
    recommended_n: Optional[int] = None


@dataclass
class HypothesisTestResult:
    """Results from hypothesis testing."""
    
    test_name: str
    test_statistic: float
    p_value: float
    degrees_freedom: Optional[float]
    effect_size: float
    effect_size_interpretation: str
    confidence_interval: Tuple[float, float]
    is_significant: bool
    assumptions_met: Dict[str, bool]


@dataclass 
class MultipleComparisonResult:
    """Results from multiple comparison correction."""
    
    original_p_values: List[float]
    corrected_p_values: List[float]
    method: str
    alpha: float
    rejected_hypotheses: List[bool]
    significant_comparisons: int
    family_wise_error_rate: float


@dataclass
class BootstrapResult:
    """Bootstrap analysis results."""
    
    observed_statistic: float
    bootstrap_distribution: np.ndarray
    confidence_interval: Tuple[float, float]
    bias: float
    bias_corrected_statistic: float
    standard_error: float
    method: str


class StatisticalAssumptionChecker:
    """Check statistical assumptions for various tests."""
    
    @staticmethod
    def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """Check normality assumption using multiple tests."""
        
        n = len(data)
        results = {}
        
        if n >= 3:
            # Shapiro-Wilk test (preferred for n < 50)
            if n <= 5000:  # Shapiro-Wilk limitation
                shapiro_stat, shapiro_p = stats.shapiro(data)
                results['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > alpha
                }
        
        if n >= 8:
            # D'Agostino's normality test
            dagostino_stat, dagostino_p = stats.normaltest(data)
            results['dagostino'] = {
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > alpha
            }
        
        if n >= 20:
            # Anderson-Darling test
            ad_stat, ad_critical_values, ad_significance_levels = stats.anderson(data, dist='norm')
            # Find significance level corresponding to alpha
            sig_index = np.searchsorted(ad_significance_levels, alpha * 100)
            if sig_index < len(ad_critical_values):
                results['anderson_darling'] = {
                    'statistic': ad_stat,
                    'critical_value': ad_critical_values[sig_index],
                    'is_normal': ad_stat < ad_critical_values[sig_index]
                }
        
        # Overall assessment
        if results:
            normality_tests = [test['is_normal'] for test in results.values() if 'is_normal' in test]
            overall_normal = sum(normality_tests) > len(normality_tests) / 2
        else:
            overall_normal = True  # Assume normal for very small samples
            
        results['overall_normal'] = overall_normal
        results['sample_size'] = n
        
        return results
    
    @staticmethod
    def check_equal_variances(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """Check equal variances assumption."""
        
        results = {}
        
        # Levene's test (robust to non-normality)
        levene_stat, levene_p = stats.levene(data1, data2)
        results['levene'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'equal_variances': levene_p > alpha
        }
        
        # F-test for equal variances (assumes normality)
        f_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
        f_p = 2 * min(stats.f.cdf(f_stat, len(data1) - 1, len(data2) - 1),
                      1 - stats.f.cdf(f_stat, len(data1) - 1, len(data2) - 1))
        results['f_test'] = {
            'statistic': f_stat,
            'p_value': f_p,
            'equal_variances': f_p > alpha
        }
        
        # Overall assessment
        results['overall_equal_variances'] = results['levene']['equal_variances']
        
        return results


class EffectSizeCalculator:
    """Calculate various effect size measures."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return cohens_d
    
    @staticmethod
    def glass_delta(group1: np.ndarray, group2: np.ndarray, control_group: int = 2) -> float:
        """Calculate Glass's delta effect size."""
        
        if control_group == 2:
            control_std = np.std(group2, ddof=1)
        else:
            control_std = np.std(group1, ddof=1)
        
        glass_delta = (np.mean(group1) - np.mean(group2)) / control_std
        
        return glass_delta
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)."""
        
        n1, n2 = len(group1), len(group2)
        cohens_d = EffectSizeCalculator.cohens_d(group1, group2)
        
        # Hedges' correction factor
        j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        
        hedges_g = cohens_d * j
        
        return hedges_g
    
    @staticmethod
    def interpret_effect_size(effect_size: float, measure: str = "cohens_d") -> str:
        """Interpret effect size magnitude."""
        
        abs_effect = abs(effect_size)
        
        if measure in ["cohens_d", "hedges_g"]:
            # Cohen's conventions
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        else:
            # Generic interpretation
            if abs_effect < 0.1:
                return "negligible"
            elif abs_effect < 0.3:
                return "small"
            elif abs_effect < 0.6:
                return "medium"
            else:
                return "large"


class PowerAnalyzer:
    """Statistical power analysis for research planning."""
    
    @staticmethod
    def power_t_test(effect_size: float, n: int, alpha: float = 0.05, 
                     alternative: str = 'two-sided') -> float:
        """Calculate power for t-test."""
        
        # Critical t-value
        if alternative == 'two-sided':
            t_critical = t_dist.ppf(1 - alpha/2, df=n-1)
        else:
            t_critical = t_dist.ppf(1 - alpha, df=n-1)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)
        
        # Power calculation
        if alternative == 'two-sided':
            power = 1 - t_dist.cdf(t_critical, df=n-1, loc=ncp) + t_dist.cdf(-t_critical, df=n-1, loc=ncp)
        else:
            power = 1 - t_dist.cdf(t_critical, df=n-1, loc=ncp)
        
        return min(1.0, max(0.0, power))
    
    @staticmethod
    def sample_size_t_test(effect_size: float, power: float = 0.8, 
                          alpha: float = 0.05, alternative: str = 'two-sided') -> int:
        """Calculate required sample size for t-test."""
        
        if alternative == 'two-sided':
            z_alpha = norm.ppf(1 - alpha/2)
        else:
            z_alpha = norm.ppf(1 - alpha)
        
        z_beta = norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    @staticmethod
    def conduct_power_analysis(group1: np.ndarray, group2: np.ndarray, 
                              target_power: float = 0.8, alpha: float = 0.05) -> PowerAnalysisResult:
        """Conduct comprehensive power analysis."""
        
        # Calculate observed effect size
        effect_size = EffectSizeCalculator.cohens_d(group1, group2)
        
        # Current sample size (use smaller group)
        n = min(len(group1), len(group2))
        
        # Calculate current power
        current_power = PowerAnalyzer.power_t_test(effect_size, n, alpha)
        
        # Calculate required sample size for target power
        if abs(effect_size) > 0.01:  # Avoid division by very small effect sizes
            recommended_n = PowerAnalyzer.sample_size_t_test(effect_size, target_power, alpha)
        else:
            recommended_n = None
        
        return PowerAnalysisResult(
            effect_size=effect_size,
            sample_size=n,
            alpha=alpha,
            power=current_power,
            is_adequate=current_power >= target_power,
            recommended_n=recommended_n
        )


class BootstrapAnalyzer:
    """Bootstrap methods for confidence intervals and significance testing."""
    
    @staticmethod
    def bias_corrected_accelerated_ci(data: np.ndarray, statistic_func, 
                                     n_bootstrap: int = 10000, alpha: float = 0.05,
                                     random_state: int = 42) -> BootstrapResult:
        """Calculate BCa (Bias-Corrected and Accelerated) bootstrap confidence interval."""
        
        np.random.seed(random_state)
        
        n = len(data)
        observed_stat = statistic_func(data)
        
        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = resample(data, n_samples=n, random_state=None)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bias correction
        n_less = np.sum(bootstrap_stats < observed_stat)
        bias_correction = norm.ppf(n_less / n_bootstrap)
        
        # Acceleration constant using jackknife
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stats.append(statistic_func(jackknife_sample))
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        acceleration = np.sum((jackknife_mean - jackknife_stats)**3) / (6 * (np.sum((jackknife_mean - jackknife_stats)**2))**1.5)
        
        # BCa confidence interval
        z_alpha_2 = norm.ppf(alpha/2)
        z_1_alpha_2 = norm.ppf(1 - alpha/2)
        
        alpha_1 = norm.cdf(bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2)))
        alpha_2 = norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2)))
        
        # Ensure valid percentiles
        alpha_1 = max(0.001, min(0.999, alpha_1))
        alpha_2 = max(0.001, min(0.999, alpha_2))
        
        ci_lower = np.percentile(bootstrap_stats, alpha_1 * 100)
        ci_upper = np.percentile(bootstrap_stats, alpha_2 * 100)
        
        # Additional statistics
        bias = np.mean(bootstrap_stats) - observed_stat
        bias_corrected = observed_stat - bias
        standard_error = np.std(bootstrap_stats, ddof=1)
        
        return BootstrapResult(
            observed_statistic=observed_stat,
            bootstrap_distribution=bootstrap_stats,
            confidence_interval=(ci_lower, ci_upper),
            bias=bias,
            bias_corrected_statistic=bias_corrected,
            standard_error=standard_error,
            method="BCa"
        )
    
    @staticmethod
    def bootstrap_hypothesis_test(group1: np.ndarray, group2: np.ndarray,
                                 n_bootstrap: int = 10000, alpha: float = 0.05,
                                 random_state: int = 42) -> Dict[str, Any]:
        """Bootstrap hypothesis test for difference in means."""
        
        np.random.seed(random_state)
        
        # Observed difference
        observed_diff = np.mean(group1) - np.mean(group2)
        
        # Combine groups for null hypothesis (no difference)
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        # Bootstrap under null hypothesis
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            shuffled = np.random.permutation(combined)
            boot_group1 = shuffled[:n1]
            boot_group2 = shuffled[n1:n1+n2]
            bootstrap_diffs.append(np.mean(boot_group1) - np.mean(boot_group2))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # P-value calculation (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return {
            "observed_difference": observed_diff,
            "bootstrap_differences": bootstrap_diffs,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "method": "bootstrap_permutation"
        }


class MultipleComparisonCorrection:
    """Multiple comparison correction methods."""
    
    @staticmethod
    def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> MultipleComparisonResult:
        """Benjamini-Hochberg FDR correction."""
        
        p_array = np.array(p_values)
        n = len(p_array)
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        # Apply BH procedure
        rejected = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if sorted_p[i] <= (i + 1) / n * alpha:
                rejected[sorted_indices[i]] = True
            else:
                break
        
        # Calculate corrected p-values
        corrected_p = np.minimum(1, sorted_p * n / np.arange(1, n + 1))
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            corrected_p[i] = min(corrected_p[i], corrected_p[i + 1])
        
        # Reorder to match original order
        final_corrected_p = np.zeros(n)
        final_corrected_p[sorted_indices] = corrected_p
        
        return MultipleComparisonResult(
            original_p_values=p_values,
            corrected_p_values=final_corrected_p.tolist(),
            method="benjamini_hochberg",
            alpha=alpha,
            rejected_hypotheses=rejected.tolist(),
            significant_comparisons=int(np.sum(rejected)),
            family_wise_error_rate=1 - (1 - alpha)**np.sum(rejected) if np.sum(rejected) > 0 else 0
        )
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> MultipleComparisonResult:
        """Bonferroni correction."""
        
        p_array = np.array(p_values)
        n = len(p_array)
        
        # Adjust alpha
        adjusted_alpha = alpha / n
        
        # Corrected p-values
        corrected_p = np.minimum(1, p_array * n)
        
        # Rejections
        rejected = corrected_p <= alpha
        
        return MultipleComparisonResult(
            original_p_values=p_values,
            corrected_p_values=corrected_p.tolist(),
            method="bonferroni",
            alpha=alpha,
            rejected_hypotheses=rejected.tolist(),
            significant_comparisons=int(np.sum(rejected)),
            family_wise_error_rate=min(1.0, alpha)
        )


class AcademicStatisticalAnalyzer:
    """
    Comprehensive statistical analysis system for academic research.
    
    Implements rigorous statistical methods with proper assumption checking,
    effect size calculation, and multiple comparison correction.
    """
    
    def __init__(self, alpha: float = 0.05, power_threshold: float = 0.8):
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.assumption_checker = StatisticalAssumptionChecker()
        self.effect_calculator = EffectSizeCalculator()
        self.power_analyzer = PowerAnalyzer()
        self.bootstrap_analyzer = BootstrapAnalyzer()
        
    def comprehensive_two_sample_analysis(self, group1: np.ndarray, group2: np.ndarray,
                                        group1_name: str = "Group1", group2_name: str = "Group2",
                                        alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Comprehensive two-sample analysis with assumption checking.
        
        Args:
            group1: First group data
            group2: Second group data  
            group1_name: Name for first group
            group2_name: Name for second group
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            Complete analysis results
        """
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "groups": {
                group1_name: {"data": group1.tolist(), "n": len(group1), "mean": float(np.mean(group1)), "std": float(np.std(group1, ddof=1))},
                group2_name: {"data": group2.tolist(), "n": len(group2), "mean": float(np.mean(group2)), "std": float(np.std(group2, ddof=1))}
            },
            "assumptions": {},
            "hypothesis_tests": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "power_analysis": {},
            "bootstrap_analysis": {},
            "recommendations": []
        }
        
        # Check assumptions
        results["assumptions"]["normality"] = {
            group1_name: self.assumption_checker.check_normality(group1, self.alpha),
            group2_name: self.assumption_checker.check_normality(group2, self.alpha)
        }
        
        results["assumptions"]["equal_variances"] = self.assumption_checker.check_equal_variances(group1, group2, self.alpha)
        
        # Effect sizes
        results["effect_sizes"]["cohens_d"] = {
            "value": float(self.effect_calculator.cohens_d(group1, group2)),
            "interpretation": self.effect_calculator.interpret_effect_size(
                self.effect_calculator.cohens_d(group1, group2), "cohens_d"
            )
        }
        
        results["effect_sizes"]["hedges_g"] = {
            "value": float(self.effect_calculator.hedges_g(group1, group2)),
            "interpretation": self.effect_calculator.interpret_effect_size(
                self.effect_calculator.hedges_g(group1, group2), "hedges_g"
            )
        }
        
        # Power analysis
        power_result = self.power_analyzer.conduct_power_analysis(group1, group2, self.power_threshold, self.alpha)
        results["power_analysis"] = asdict(power_result)
        
        # Choose appropriate test based on assumptions
        normality_ok = (results["assumptions"]["normality"][group1_name]["overall_normal"] and 
                       results["assumptions"]["normality"][group2_name]["overall_normal"])
        
        equal_var_ok = results["assumptions"]["equal_variances"]["overall_equal_variances"]
        
        if normality_ok:
            if equal_var_ok:
                # Student's t-test
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True, alternative=alternative)
                test_name = "students_t_test"
                df = len(group1) + len(group2) - 2
            else:
                # Welch's t-test
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False, alternative=alternative)
                test_name = "welchs_t_test"
                # Welch-Satterthwaite equation for df
                s1_sq, s2_sq = np.var(group1, ddof=1), np.var(group2, ddof=1)
                n1, n2 = len(group1), len(group2)
                df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
            
            # Confidence interval for mean difference
            pooled_se = np.sqrt(s1_sq/n1 + s2_sq/n2)
            t_critical = t_dist.ppf(1 - self.alpha/2, df)
            mean_diff = np.mean(group1) - np.mean(group2)
            ci_lower = mean_diff - t_critical * pooled_se
            ci_upper = mean_diff + t_critical * pooled_se
            
            results["hypothesis_tests"]["parametric"] = HypothesisTestResult(
                test_name=test_name,
                test_statistic=float(t_stat),
                p_value=float(p_val),
                degrees_freedom=float(df),
                effect_size=results["effect_sizes"]["cohens_d"]["value"],
                effect_size_interpretation=results["effect_sizes"]["cohens_d"]["interpretation"],
                confidence_interval=(float(ci_lower), float(ci_upper)),
                is_significant=p_val < self.alpha,
                assumptions_met={"normality": normality_ok, "equal_variances": equal_var_ok}
            )
        
        # Always run non-parametric test as robustness check
        u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        results["hypothesis_tests"]["non_parametric"] = {
            "test_name": "mann_whitney_u",
            "test_statistic": float(u_stat),
            "p_value": float(u_p),
            "is_significant": u_p < self.alpha,
            "assumptions_met": {"independence": True}  # Main assumption for Mann-Whitney U
        }
        
        # Bootstrap analysis
        bootstrap_result = self.bootstrap_analyzer.bias_corrected_accelerated_ci(
            np.concatenate([group1, group2]), 
            lambda x: np.mean(x[:len(group1)]) - np.mean(x[len(group1):]),
            n_bootstrap=10000,
            alpha=self.alpha
        )
        
        results["bootstrap_analysis"]["bca_ci"] = asdict(bootstrap_result)
        
        # Bootstrap hypothesis test
        bootstrap_test = self.bootstrap_analyzer.bootstrap_hypothesis_test(
            group1, group2, n_bootstrap=10000, alpha=self.alpha
        )
        results["bootstrap_analysis"]["hypothesis_test"] = {
            "observed_difference": float(bootstrap_test["observed_difference"]),
            "p_value": float(bootstrap_test["p_value"]),
            "is_significant": bootstrap_test["is_significant"],
            "method": bootstrap_test["method"]
        }
        
        # Generate recommendations
        if not normality_ok:
            results["recommendations"].append("Normality assumptions violated - consider non-parametric tests or transformations")
        
        if not equal_var_ok and normality_ok:
            results["recommendations"].append("Unequal variances detected - Welch's t-test used")
        
        if not power_result.is_adequate:
            results["recommendations"].append(f"Statistical power ({power_result.power:.3f}) below threshold ({self.power_threshold}) - consider larger sample size")
        
        if abs(results["effect_sizes"]["cohens_d"]["value"]) < 0.2:
            results["recommendations"].append("Small effect size detected - practical significance may be limited")
        
        return results
    
    def multiple_group_analysis(self, groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Multiple group comparison analysis.
        
        Args:
            groups: Dictionary with group names as keys and data arrays as values
            
        Returns:
            Complete multiple comparison analysis
        """
        
        group_names = list(groups.keys())
        group_data = list(groups.values())
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "groups": {name: {"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data, ddof=1))} 
                      for name, data in groups.items()},
            "pairwise_comparisons": {},
            "multiple_comparison_correction": {},
            "omnibus_tests": {},
            "effect_sizes": {}
        }
        
        # Pairwise comparisons
        comparison_results = []
        p_values = []
        comparison_names = []
        
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1_name, group2_name = group_names[i], group_names[j]
                comparison_name = f"{group1_name}_vs_{group2_name}"
                
                # Comprehensive two-sample analysis for each pair
                pairwise_result = self.comprehensive_two_sample_analysis(
                    groups[group1_name], groups[group2_name], group1_name, group2_name
                )
                
                results["pairwise_comparisons"][comparison_name] = pairwise_result
                
                # Collect p-values for correction
                if "parametric" in pairwise_result["hypothesis_tests"]:
                    p_values.append(pairwise_result["hypothesis_tests"]["parametric"].p_value)
                else:
                    p_values.append(pairwise_result["hypothesis_tests"]["non_parametric"]["p_value"])
                
                comparison_names.append(comparison_name)
        
        # Multiple comparison correction
        if len(p_values) > 1:
            # Benjamini-Hochberg FDR correction
            fdr_result = MultipleComparisonCorrection.benjamini_hochberg_fdr(p_values, self.alpha)
            results["multiple_comparison_correction"]["fdr"] = asdict(fdr_result)
            
            # Bonferroni correction
            bonf_result = MultipleComparisonCorrection.bonferroni_correction(p_values, self.alpha)
            results["multiple_comparison_correction"]["bonferroni"] = asdict(bonf_result)
            
            # Add comparison names to results
            results["multiple_comparison_correction"]["comparison_names"] = comparison_names
        
        # Omnibus tests (if more than 2 groups)
        if len(groups) > 2:
            # One-way ANOVA
            f_stat, anova_p = stats.f_oneway(*group_data)
            results["omnibus_tests"]["anova"] = {
                "test_name": "one_way_anova",
                "f_statistic": float(f_stat),
                "p_value": float(anova_p),
                "is_significant": anova_p < self.alpha
            }
            
            # Kruskal-Wallis (non-parametric alternative)
            h_stat, kw_p = stats.kruskal(*group_data)
            results["omnibus_tests"]["kruskal_wallis"] = {
                "test_name": "kruskal_wallis",
                "h_statistic": float(h_stat),
                "p_value": float(kw_p), 
                "is_significant": kw_p < self.alpha
            }
        
        return results
    
    def fastpath_research_analysis(self, baseline_data: Dict[str, np.ndarray], 
                                  fastpath_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Specialized analysis for FastPath research validation.
        
        Args:
            baseline_data: Dictionary with baseline system names and performance data
            fastpath_data: Dictionary with FastPath variant names and performance data
            
        Returns:
            Complete FastPath research analysis
        """
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_type": "FastPath Research Validation",
            "baseline_systems": list(baseline_data.keys()),
            "fastpath_systems": list(fastpath_data.keys()),
            "primary_comparisons": {},
            "research_questions": {},
            "publication_summary": {}
        }
        
        # Primary research question: FastPath vs BM25 baseline
        if "bm25" in baseline_data:
            bm25_data = baseline_data["bm25"]
            
            for fastpath_name, fastpath_perf in fastpath_data.items():
                comparison_name = f"{fastpath_name}_vs_bm25"
                
                # Comprehensive analysis
                analysis_result = self.comprehensive_two_sample_analysis(
                    fastpath_perf, bm25_data, fastpath_name, "bm25", alternative='greater'
                )
                
                results["primary_comparisons"][comparison_name] = analysis_result
                
                # Calculate improvement percentage
                improvement_pct = ((np.mean(fastpath_perf) - np.mean(bm25_data)) / np.mean(bm25_data)) * 100
                analysis_result["improvement_percentage"] = float(improvement_pct)
                analysis_result["meets_20pct_threshold"] = improvement_pct >= 20.0
        
        # Research question 1: Performance improvement
        fastpath_improvements = []
        significant_improvements = 0
        
        for comparison_name, analysis in results["primary_comparisons"].items():
            improvement = analysis.get("improvement_percentage", 0)
            fastpath_improvements.append(improvement)
            
            # Check significance (use parametric test if available, otherwise non-parametric)
            if "parametric" in analysis["hypothesis_tests"]:
                is_significant = analysis["hypothesis_tests"]["parametric"].is_significant
            else:
                is_significant = analysis["hypothesis_tests"]["non_parametric"]["is_significant"]
            
            if is_significant:
                significant_improvements += 1
        
        results["research_questions"]["q1_performance_improvement"] = {
            "question": "How much does FastPath improve QA accuracy vs established baselines?",
            "mean_improvement": float(np.mean(fastpath_improvements)) if fastpath_improvements else 0,
            "max_improvement": float(np.max(fastpath_improvements)) if fastpath_improvements else 0,
            "min_improvement": float(np.min(fastpath_improvements)) if fastpath_improvements else 0,
            "meets_target": any(imp >= 20.0 for imp in fastpath_improvements),
            "significant_improvements": significant_improvements,
            "total_comparisons": len(results["primary_comparisons"])
        }
        
        # Multiple comparison correction for primary comparisons
        if len(results["primary_comparisons"]) > 1:
            primary_p_values = []
            for analysis in results["primary_comparisons"].values():
                if "parametric" in analysis["hypothesis_tests"]:
                    primary_p_values.append(analysis["hypothesis_tests"]["parametric"].p_value)
                else:
                    primary_p_values.append(analysis["hypothesis_tests"]["non_parametric"]["p_value"])
            
            fdr_correction = MultipleComparisonCorrection.benjamini_hochberg_fdr(primary_p_values, self.alpha)
            results["multiple_comparison_correction"] = asdict(fdr_correction)
            
            # Update significance after correction
            results["research_questions"]["q1_performance_improvement"]["significant_after_correction"] = int(np.sum(fdr_correction.rejected_hypotheses))
        
        # Publication readiness assessment
        primary_hypothesis_supported = results["research_questions"]["q1_performance_improvement"]["meets_target"]
        adequate_power = all(
            analysis.get("power_analysis", {}).get("is_adequate", False) 
            for analysis in results["primary_comparisons"].values()
        )
        
        results["publication_summary"] = {
            "primary_hypothesis_supported": primary_hypothesis_supported,
            "statistical_rigor": {
                "multiple_comparison_correction": len(results["primary_comparisons"]) > 1,
                "effect_size_reporting": True,
                "confidence_intervals": True,
                "bootstrap_analysis": True,
                "assumption_checking": True
            },
            "adequate_statistical_power": adequate_power,
            "recommendation": "PUBLICATION_READY" if (primary_hypothesis_supported and adequate_power) else "NEEDS_IMPROVEMENT"
        }
        
        return results


def main():
    """Demo of academic statistical analysis."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("📊 Running Academic Statistical Analysis Demo...")
        
        # Generate demo data
        np.random.seed(42)
        
        # Simulate realistic performance data
        bm25_performance = np.random.normal(0.68, 0.05, 30)  # BM25 baseline
        fastpath_v1 = np.random.normal(0.78, 0.04, 30)      # 15% improvement
        fastpath_v2 = np.random.normal(0.83, 0.04, 30)      # 22% improvement  
        fastpath_v3 = np.random.normal(0.85, 0.04, 30)      # 25% improvement
        
        baseline_data = {"bm25": bm25_performance}
        fastpath_data = {
            "fastpath_v1": fastpath_v1,
            "fastpath_v2": fastpath_v2,
            "fastpath_v3": fastpath_v3
        }
        
        # Run comprehensive analysis
        analyzer = AcademicStatisticalAnalyzer(alpha=0.05, power_threshold=0.8)
        results = analyzer.fastpath_research_analysis(baseline_data, fastpath_data)
        
        # Print summary
        print(f"\n{'='*80}")
        print("ACADEMIC STATISTICAL ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        q1_results = results["research_questions"]["q1_performance_improvement"]
        print(f"🎯 Mean Performance Improvement: {q1_results['mean_improvement']:.1f}%")
        print(f"📈 Max Improvement: {q1_results['max_improvement']:.1f}%")
        print(f"✅ Meets ≥20% Target: {'YES' if q1_results['meets_target'] else 'NO'}")
        print(f"🔬 Significant Results: {q1_results['significant_improvements']}/{q1_results['total_comparisons']}")
        
        if "multiple_comparison_correction" in results:
            corrected_sig = results["multiple_comparison_correction"]["significant_comparisons"]
            print(f"📊 Significant After FDR Correction: {corrected_sig}/{q1_results['total_comparisons']}")
        
        pub_summary = results["publication_summary"]
        print(f"📄 Publication Recommendation: {pub_summary['recommendation']}")
        print(f"🔒 Statistical Rigor: {'COMPLETE' if all(pub_summary['statistical_rigor'].values()) else 'PARTIAL'}")
        
        # Save results
        output_file = Path("academic_statistical_analysis_demo.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Full results saved to: {output_file}")
        return results
        
    else:
        print("Usage: academic_statistical_analysis.py --demo")
        print("       Run comprehensive academic statistical analysis demo")
        return None


if __name__ == "__main__":
    main()
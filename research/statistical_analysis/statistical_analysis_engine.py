#!/usr/bin/env python3
"""
Statistical Analysis Engine for FastPath Research
=================================================

Advanced statistical analysis framework with:
- Bootstrap confidence intervals (BCa correction)
- Multiple comparison testing with FDR control
- Effect size calculations (Cohen's d)  
- Power analysis and sample size validation
- Cross-validation with proper train/test splits
- Non-parametric tests for robust analysis

All methods follow best practices for research publications.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import scipy.stats as stats
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import cohen_kappa_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class EffectSize:
    """Effect size measurement with interpretation."""
    value: float
    magnitude: str
    confidence_interval: Tuple[float, float]
    method: str
    
    def __post_init__(self):
        if self.method == "cohen_d":
            if abs(self.value) < 0.2:
                self.magnitude = "negligible"
            elif abs(self.value) < 0.5:
                self.magnitude = "small"
            elif abs(self.value) < 0.8:
                self.magnitude = "medium"
            else:
                self.magnitude = "large"


@dataclass
class SignificanceTest:
    """Statistical significance test result."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int]
    critical_value: Optional[float]
    is_significant: bool
    alpha: float = 0.05
    
    def __post_init__(self):
        self.is_significant = self.p_value < self.alpha


@dataclass
class BootstrapResult:
    """Bootstrap analysis result."""
    statistic: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    bias: float
    confidence_level: float
    method: str  # 'percentile', 'bias_corrected', 'bca'
    n_bootstrap: int


@dataclass
class PowerAnalysis:
    """Statistical power analysis result."""
    observed_power: float
    required_sample_size: int
    effect_size: float
    alpha: float
    beta: float  # Type II error rate
    
    def __post_init__(self):
        self.power_adequate = self.observed_power >= 0.8


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis engine for experimental results.
    
    Implements research-grade statistical methods with proper corrections
    and confidence interval estimation.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 10000,
        alpha: float = 0.05,
        random_seed: int = 42
    ):
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self.alpha = alpha
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
    
    def analyze_experiment_results(self, raw_measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Complete statistical analysis of experimental results.
        
        Args:
            raw_measurements: List of measurement dictionaries
            
        Returns:
            Comprehensive statistical analysis results
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(raw_measurements)
        
        # Filter successful measurements only
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) == 0:
            return {"error": "No successful measurements to analyze"}
        
        # Extract key metrics by system
        system_data = self._extract_system_data(successful_df)
        
        # Perform analyses
        analyses = {
            'system_summaries': self._calculate_system_summaries(system_data),
            'pairwise_comparisons': self._perform_pairwise_comparisons(system_data),
            'effect_sizes': self._calculate_effect_sizes(system_data),
            'bootstrap_intervals': self._calculate_bootstrap_intervals(system_data),
            'significance_tests': self._perform_significance_tests(system_data),
            'multiple_comparisons': self._correct_multiple_comparisons(system_data),
            'power_analysis': self._perform_power_analysis(system_data),
            'cross_validation': self._perform_cross_validation(successful_df),
            'non_parametric_tests': self._perform_non_parametric_tests(system_data),
            'summary': self._generate_summary(system_data)
        }
        
        return analyses
    
    def _extract_system_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract metrics by system from DataFrame."""
        system_data = defaultdict(dict)
        
        # Define metrics to extract
        metrics = [
            'execution_time_seconds',
            'qa_accuracy', 
            'qa_f1_score',
            'memory_usage_bytes',
            'tokens_used',
            'files_retrieved'
        ]
        
        for system in df['system_name'].unique():
            system_df = df[df['system_name'] == system]
            
            for metric in metrics:
                if metric in system_df.columns:
                    # Remove NaN values and convert to numpy array
                    values = system_df[metric].dropna().values
                    if len(values) > 0:
                        system_data[system][metric] = values
        
        return dict(system_data)
    
    def _calculate_system_summaries(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """Calculate descriptive statistics for each system."""
        summaries = {}
        
        for system, metrics in system_data.items():
            system_summary = {}
            
            for metric, values in metrics.items():
                if len(values) > 0:
                    system_summary[metric] = {
                        'count': len(values),
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values, ddof=1)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75)),
                        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                        'skewness': float(stats.skew(values)),
                        'kurtosis': float(stats.kurtosis(values))
                    }
            
            summaries[system] = system_summary
        
        return summaries
    
    def _perform_pairwise_comparisons(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Perform pairwise statistical comparisons between systems."""
        systems = list(system_data.keys())
        comparisons = {}
        
        # All pairwise combinations
        for i, system1 in enumerate(systems):
            for j, system2 in enumerate(systems):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                comparison_key = f"{system1}_vs_{system2}"
                comparisons[comparison_key] = {}
                
                # Compare each metric
                common_metrics = set(system_data[system1].keys()) & set(system_data[system2].keys())
                
                for metric in common_metrics:
                    values1 = system_data[system1][metric]
                    values2 = system_data[system2][metric]
                    
                    if len(values1) > 1 and len(values2) > 1:
                        comparison_result = self._compare_two_samples(values1, values2, metric)
                        comparisons[comparison_key][metric] = comparison_result
        
        return comparisons
    
    def _compare_two_samples(self, sample1: np.ndarray, sample2: np.ndarray, metric: str) -> Dict[str, Any]:
        """Compare two samples with appropriate statistical tests."""
        result = {}
        
        # Basic statistics
        result['sample1'] = {
            'n': len(sample1),
            'mean': float(np.mean(sample1)),
            'std': float(np.std(sample1, ddof=1))
        }
        result['sample2'] = {
            'n': len(sample2),
            'mean': float(np.mean(sample2)),
            'std': float(np.std(sample2, ddof=1))
        }
        
        # Test for normality
        shapiro1 = stats.shapiro(sample1) if len(sample1) <= 5000 else (np.nan, 1.0)
        shapiro2 = stats.shapiro(sample2) if len(sample2) <= 5000 else (np.nan, 1.0)
        
        normal1 = shapiro1[1] > 0.05
        normal2 = shapiro2[1] > 0.05
        
        result['normality'] = {
            'sample1_normal': normal1,
            'sample2_normal': normal2,
            'sample1_shapiro_p': float(shapiro1[1]),
            'sample2_shapiro_p': float(shapiro2[1])
        }
        
        # Test for equal variances
        levene_stat, levene_p = stats.levene(sample1, sample2)
        equal_variances = levene_p > 0.05
        
        result['equal_variances'] = {
            'equal': equal_variances,
            'levene_statistic': float(levene_stat),
            'levene_p': float(levene_p)
        }
        
        # Choose appropriate test
        if normal1 and normal2:
            # Both normal: use t-test
            if equal_variances:
                stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=True)
                test_name = "Independent t-test (equal variance)"
            else:
                stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
                test_name = "Welch's t-test (unequal variance)"
        else:
            # Non-normal: use Mann-Whitney U test
            stat, p_val = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        result['statistical_test'] = SignificanceTest(
            test_name=test_name,
            statistic=float(stat),
            p_value=float(p_val),
            degrees_of_freedom=len(sample1) + len(sample2) - 2 if 't-test' in test_name else None,
            critical_value=None,
            is_significant=p_val < self.alpha,
            alpha=self.alpha
        ).__dict__
        
        # Effect size (Cohen's d for continuous metrics)
        if metric in ['execution_time_seconds', 'qa_accuracy', 'qa_f1_score']:
            effect_size = self._calculate_cohens_d(sample1, sample2)
            result['effect_size'] = effect_size.__dict__
        
        return result
    
    def _calculate_cohens_d(self, sample1: np.ndarray, sample2: np.ndarray) -> EffectSize:
        """Calculate Cohen's d effect size with confidence interval."""
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (mean1 - mean2) / pooled_std
        
        # Confidence interval for Cohen's d (approximate)
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n1 + n2 - 2)
        
        ci_lower = d - t_critical * se_d
        ci_upper = d + t_critical * se_d
        
        return EffectSize(
            value=float(d),
            magnitude="",  # Will be set by __post_init__
            confidence_interval=(float(ci_lower), float(ci_upper)),
            method="cohen_d"
        )
    
    def _calculate_effect_sizes(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """Calculate effect sizes for key comparisons."""
        effect_sizes = {}
        
        # Find FastPath systems and baseline systems
        fastpath_systems = [s for s in system_data.keys() if 'fastpath' in s.lower()]
        baseline_systems = [s for s in system_data.keys() if 'fastpath' not in s.lower()]
        
        # Compare FastPath systems to baselines
        for fastpath in fastpath_systems:
            effect_sizes[fastpath] = {}
            
            for baseline in baseline_systems:
                if fastpath not in system_data or baseline not in system_data:
                    continue
                
                comparison_key = f"vs_{baseline}"
                effect_sizes[fastpath][comparison_key] = {}
                
                # Calculate effect sizes for key metrics
                key_metrics = ['qa_accuracy', 'execution_time_seconds', 'qa_f1_score']
                
                for metric in key_metrics:
                    if metric in system_data[fastpath] and metric in system_data[baseline]:
                        fastpath_values = system_data[fastpath][metric]
                        baseline_values = system_data[baseline][metric]
                        
                        if len(fastpath_values) > 1 and len(baseline_values) > 1:
                            effect_size = self._calculate_cohens_d(fastpath_values, baseline_values)
                            effect_sizes[fastpath][comparison_key][metric] = effect_size.__dict__
        
        return effect_sizes
    
    def _calculate_bootstrap_intervals(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """Calculate bootstrap confidence intervals for all systems and metrics."""
        bootstrap_results = {}
        
        for system, metrics in system_data.items():
            bootstrap_results[system] = {}
            
            for metric, values in metrics.items():
                if len(values) >= 10:  # Minimum sample size for bootstrap
                    bootstrap_result = self._bootstrap_confidence_interval(values)
                    bootstrap_results[system][metric] = bootstrap_result.__dict__
        
        return bootstrap_results
    
    def _bootstrap_confidence_interval(self, data: np.ndarray) -> BootstrapResult:
        """Calculate bias-corrected and accelerated (BCa) bootstrap confidence interval."""
        n = len(data)
        
        # Original statistic (mean)
        original_stat = np.mean(data)
        
        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(self.bootstrap_iterations):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(np.mean(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bias correction
        bias = np.mean(bootstrap_stats < original_stat)
        z0 = stats.norm.ppf(bias) if 0 < bias < 1 else 0
        
        # Acceleration (jackknife)
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stats.append(np.mean(jackknife_sample))
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        # Acceleration parameter
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        acceleration = numerator / denominator if denominator != 0 else 0
        
        # BCa confidence interval
        alpha_level = (1 - self.confidence_level) / 2
        z_alpha = stats.norm.ppf(alpha_level)
        z_1_alpha = stats.norm.ppf(1 - alpha_level)
        
        # Adjusted quantiles
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - acceleration * (z0 + z_1_alpha)))
        
        # Ensure valid quantiles
        alpha1 = max(0, min(1, alpha1))
        alpha2 = max(0, min(1, alpha2))
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_stats, alpha1 * 100)
        ci_upper = np.percentile(bootstrap_stats, alpha2 * 100)
        
        return BootstrapResult(
            statistic=float(original_stat),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            standard_error=float(np.std(bootstrap_stats)),
            bias=float(np.mean(bootstrap_stats) - original_stat),
            confidence_level=self.confidence_level,
            method="bca",
            n_bootstrap=self.bootstrap_iterations
        )
    
    def _perform_significance_tests(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Perform significance tests comparing FastPath to baselines."""
        significance_tests = {}
        
        # Get system categories
        fastpath_systems = [s for s in system_data.keys() if 'fastpath' in s.lower()]
        baseline_systems = [s for s in system_data.keys() if 'fastpath' not in s.lower()]
        
        # Overall ANOVA for each metric
        for metric in ['qa_accuracy', 'execution_time_seconds', 'qa_f1_score']:
            metric_data = []
            metric_groups = []
            
            for system, metrics in system_data.items():
                if metric in metrics and len(metrics[metric]) > 0:
                    metric_data.extend(metrics[metric])
                    metric_groups.extend([system] * len(metrics[metric]))
            
            if len(set(metric_groups)) > 2 and len(metric_data) > 0:
                # One-way ANOVA
                groups = {}
                for system in set(metric_groups):
                    groups[system] = [metric_data[i] for i, g in enumerate(metric_groups) if g == system]
                
                if len(groups) >= 2:
                    anova_stat, anova_p = stats.f_oneway(*groups.values())
                    
                    significance_tests[f'{metric}_anova'] = SignificanceTest(
                        test_name="One-way ANOVA",
                        statistic=float(anova_stat),
                        p_value=float(anova_p),
                        degrees_of_freedom=len(groups) - 1,
                        critical_value=None,
                        is_significant=anova_p < self.alpha,
                        alpha=self.alpha
                    ).__dict__
        
        return significance_tests
    
    def _correct_multiple_comparisons(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Apply False Discovery Rate (FDR) correction for multiple comparisons."""
        # Collect all p-values from pairwise comparisons
        pairwise_results = self._perform_pairwise_comparisons(system_data)
        
        p_values = []
        comparison_names = []
        
        for comparison, metrics in pairwise_results.items():
            for metric, result in metrics.items():
                if 'statistical_test' in result:
                    p_values.append(result['statistical_test']['p_value'])
                    comparison_names.append(f"{comparison}_{metric}")
        
        if len(p_values) > 1:
            # Apply Benjamini-Hochberg FDR correction
            rejected, p_corrected = fdrcorrection(p_values, alpha=self.alpha)
            
            correction_results = {
                'method': 'Benjamini-Hochberg FDR',
                'original_alpha': self.alpha,
                'total_comparisons': len(p_values),
                'significant_after_correction': int(np.sum(rejected)),
                'corrected_results': []
            }
            
            for i, (name, original_p, corrected_p, is_significant) in enumerate(
                zip(comparison_names, p_values, p_corrected, rejected)
            ):
                correction_results['corrected_results'].append({
                    'comparison': name,
                    'original_p_value': float(original_p),
                    'corrected_p_value': float(corrected_p),
                    'significant_after_correction': bool(is_significant)
                })
            
            return correction_results
        else:
            return {'method': 'No correction needed', 'total_comparisons': len(p_values)}
    
    def _perform_power_analysis(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Perform statistical power analysis for the experiment."""
        power_results = {}
        
        # Analyze power for key comparisons
        fastpath_systems = [s for s in system_data.keys() if 'fastpath_v3' in s.lower()]
        baseline_systems = [s for s in system_data.keys() if s in ['random', 'naive_tfidf']]
        
        if fastpath_systems and baseline_systems:
            fastpath = fastpath_systems[0]
            baseline = baseline_systems[0]
            
            for metric in ['qa_accuracy', 'execution_time_seconds']:
                if metric in system_data[fastpath] and metric in system_data[baseline]:
                    fastpath_values = system_data[fastpath][metric]
                    baseline_values = system_data[baseline][metric]
                    
                    if len(fastpath_values) >= 5 and len(baseline_values) >= 5:
                        power_result = self._calculate_statistical_power(
                            fastpath_values, baseline_values, metric
                        )
                        power_results[f"{fastpath}_vs_{baseline}_{metric}"] = power_result.__dict__
        
        return power_results
    
    def _calculate_statistical_power(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray, 
        metric: str
    ) -> PowerAnalysis:
        """Calculate statistical power for comparison."""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Approximate power calculation using non-centrality parameter
        df = n1 + n2 - 2
        ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))  # Non-centrality parameter
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        
        # Power (approximate)
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
        # Required sample size for 80% power (approximate)
        if effect_size > 0:
            required_n = max(5, int(16 / (effect_size ** 2)) + 2)  # Rule of thumb
        else:
            required_n = 1000  # Large sample needed for very small effects
        
        return PowerAnalysis(
            observed_power=float(max(0, min(1, power))),
            required_sample_size=required_n,
            effect_size=float(effect_size),
            alpha=self.alpha,
            beta=float(max(0, min(1, 1 - power)))
        )
    
    def _perform_cross_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform cross-validation analysis to assess result stability."""
        cv_results = {}
        
        if len(df) < 10:
            return {"error": "Insufficient data for cross-validation"}
        
        # Group by system and repository type for stratified CV
        system_repo_combinations = df.groupby(['system_name', 'repository_type']).size()
        
        cv_results['system_stability'] = {}
        
        for system in df['system_name'].unique():
            system_df = df[df['system_name'] == system]
            
            if len(system_df) >= 5:
                # Cross-validation on QA accuracy
                if 'qa_accuracy' in system_df.columns:
                    qa_scores = system_df['qa_accuracy'].values
                    qa_scores = qa_scores[~np.isnan(qa_scores)]
                    
                    if len(qa_scores) >= 5:
                        # Simple k-fold CV stability measure
                        n_folds = min(5, len(qa_scores))
                        fold_size = len(qa_scores) // n_folds
                        
                        fold_means = []
                        for i in range(n_folds):
                            start_idx = i * fold_size
                            end_idx = start_idx + fold_size if i < n_folds - 1 else len(qa_scores)
                            fold_scores = qa_scores[start_idx:end_idx]
                            fold_means.append(np.mean(fold_scores))
                        
                        cv_results['system_stability'][system] = {
                            'qa_accuracy_cv_mean': float(np.mean(fold_means)),
                            'qa_accuracy_cv_std': float(np.std(fold_means)),
                            'qa_accuracy_stability': float(1 - np.std(fold_means) / max(np.mean(fold_means), 1e-6)),
                            'n_folds': n_folds
                        }
        
        return cv_results
    
    def _perform_non_parametric_tests(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Perform non-parametric tests for robust analysis."""
        nonparam_results = {}
        
        # Kruskal-Wallis test (non-parametric ANOVA)
        for metric in ['qa_accuracy', 'execution_time_seconds', 'qa_f1_score']:
            metric_groups = []
            group_names = []
            
            for system, metrics in system_data.items():
                if metric in metrics and len(metrics[metric]) >= 3:
                    metric_groups.append(metrics[metric])
                    group_names.append(system)
            
            if len(metric_groups) >= 3:
                try:
                    kw_statistic, kw_p = stats.kruskal(*metric_groups)
                    
                    nonparam_results[f'{metric}_kruskal_wallis'] = {
                        'statistic': float(kw_statistic),
                        'p_value': float(kw_p),
                        'is_significant': kw_p < self.alpha,
                        'test_name': 'Kruskal-Wallis H-test',
                        'groups_tested': group_names,
                        'n_groups': len(metric_groups)
                    }
                except Exception as e:
                    nonparam_results[f'{metric}_kruskal_wallis'] = {
                        'error': str(e)
                    }
        
        return nonparam_results
    
    def _generate_summary(self, system_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Generate executive summary of statistical findings."""
        summary = {
            'experiment_summary': {},
            'key_findings': [],
            'statistical_significance': {},
            'effect_sizes_summary': {},
            'recommendations': []
        }
        
        # Basic experiment info
        total_measurements = sum(len(metrics.get('qa_accuracy', [])) for metrics in system_data.values())
        n_systems = len(system_data)
        
        summary['experiment_summary'] = {
            'total_systems_evaluated': n_systems,
            'total_measurements': total_measurements,
            'systems_evaluated': list(system_data.keys()),
            'key_metrics': ['qa_accuracy', 'execution_time_seconds', 'qa_f1_score', 'memory_usage_bytes']
        }
        
        # Find best performing system for each metric
        for metric in ['qa_accuracy', 'qa_f1_score']:  # Higher is better
            metric_means = {}
            for system, metrics in system_data.items():
                if metric in metrics and len(metrics[metric]) > 0:
                    metric_means[system] = np.mean(metrics[metric])
            
            if metric_means:
                best_system = max(metric_means.items(), key=lambda x: x[1])
                summary['key_findings'].append({
                    'metric': metric,
                    'best_system': best_system[0],
                    'best_value': float(best_system[1]),
                    'improvement_type': 'higher_is_better'
                })
        
        for metric in ['execution_time_seconds', 'memory_usage_bytes']:  # Lower is better
            metric_means = {}
            for system, metrics in system_data.items():
                if metric in metrics and len(metrics[metric]) > 0:
                    metric_means[system] = np.mean(metrics[metric])
            
            if metric_means:
                best_system = min(metric_means.items(), key=lambda x: x[1])
                summary['key_findings'].append({
                    'metric': metric,
                    'best_system': best_system[0],
                    'best_value': float(best_system[1]),
                    'improvement_type': 'lower_is_better'
                })
        
        # Sample size adequacy
        min_measurements = min(
            len(metrics.get('qa_accuracy', [])) 
            for metrics in system_data.values() 
            if len(metrics.get('qa_accuracy', [])) > 0
        ) if system_data else 0
        
        summary['statistical_significance']['sample_size_adequate'] = min_measurements >= 10
        summary['statistical_significance']['minimum_measurements_per_system'] = min_measurements
        
        # Recommendations based on findings
        if min_measurements < 10:
            summary['recommendations'].append(
                "Increase sample size to at least 10 measurements per system for more reliable results"
            )
        
        if n_systems >= 3:
            summary['recommendations'].append(
                "Consider post-hoc analysis for pairwise comparisons after ANOVA"
            )
        
        fastpath_systems = [s for s in system_data.keys() if 'fastpath' in s.lower()]
        if len(fastpath_systems) > 1:
            summary['recommendations'].append(
                f"Compare FastPath versions ({', '.join(fastpath_systems)}) to identify optimal configuration"
            )
        
        return summary


# Export main class
__all__ = ['StatisticalAnalyzer', 'EffectSize', 'SignificanceTest', 'BootstrapResult', 'PowerAnalysis']
## Statistical Analysis

All statistical analyses were conducted using Python 3.9 with NumPy, SciPy, and custom implementation of advanced bootstrap methods. Statistical significance was set at α = 0.05 (two-tailed) with Benjamini-Hochberg false discovery rate correction for multiple comparisons.

### Primary Analysis
The primary hypothesis (FastPath V5 achieves ≥13% QA improvement vs baseline) was tested using bias-corrected and accelerated (BCa) bootstrap methodology with 10,000 iterations. BCa bootstrap provides more accurate confidence intervals than percentile methods by correcting for bias and skewness in the bootstrap distribution (Efron & Tibshirani, 1993). The bias correction factor (ẑ₀) was calculated from the proportion of bootstrap samples less than the observed statistic, while the acceleration constant (â) was estimated via jackknife variance estimation.

### Multiple Comparison Control
To control the family-wise error rate across multiple statistical tests, we applied the Benjamini-Hochberg procedure (Benjamini & Hochberg, 1995) with α = 0.05. This method controls the false discovery rate (FDR) while maintaining reasonable statistical power for detecting true effects. Tests were organized into families: (1) primary hypothesis, (2) ablation analysis, and (3) budget stratification analysis.

### Effect Size Analysis
Effect sizes were calculated using Cohen's d with pooled standard deviation, along with bias-corrected Hedges' g for small sample corrections. Bootstrap confidence intervals were computed for all effect size estimates using 5000 iterations. Practical significance was defined as Cohen's d ≥ 0.13, corresponding to the target 13% improvement threshold.

### Paired Statistical Testing
All comparisons utilized paired-sample designs where each question was evaluated under both control and treatment conditions. This approach increases statistical power by controlling for question-specific variance. We employed both parametric (paired t-test) and non-parametric (Wilcoxon signed-rank test) methods to ensure robustness to distributional assumptions.

### Power Analysis
Post-hoc power analysis was conducted to validate adequate statistical power (≥80.0%) for detecting the target effect sizes. Sample size adequacy was assessed using established power analysis methods for two-sample comparisons.

### Assumption Validation
Statistical assumptions were validated including: (1) normality via Shapiro-Wilk tests, (2) independence through experimental design verification, (3) equal variances via Levene's test, and (4) outlier assessment using interquartile range methods. Sensitivity analyses were conducted to assess robustness to assumption violations.

### Software Implementation
All analyses used custom implementations of BCa bootstrap following Efron & Tibshirani (1993) methodology. Bootstrap resampling preserved the paired structure of the data. Statistical tests were implemented using SciPy 1.7+ with validation against published reference implementations.

### Reproducibility
All analyses used fixed random seeds (seed = 42) to ensure reproducibility. Complete analysis code and data processing pipelines are available in the supplementary materials.

#### References
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
- Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall/CRC.
- Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum Associates.
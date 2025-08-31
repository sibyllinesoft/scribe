# PackRepo Statistical Analysis Framework

Comprehensive statistical analysis suite for token efficiency validation with rigorous scientific methodology and acceptance gate integration.

## Overview

The PackRepo Statistical Analysis Framework provides the mathematical foundation for making defensible claims about token efficiency improvements with proper academic rigor. It implements state-of-the-art statistical methods to validate the primary KPI of **≥ +20% Q&A accuracy per 100k tokens** with confidence interval lower bounds > 0.

## Key Components

### 1. BCa Bootstrap Engine (`bootstrap_bca.py`)
- **Bias-Corrected and Accelerated (BCa) bootstrap** methodology
- 10,000+ bootstrap iterations for stable confidence intervals
- Paired difference analysis for variant comparisons
- Numerically stable implementation with memory efficiency
- Acceleration constant calculation for skewness correction

```python
from packrepo.evaluator.statistics import BCaBootstrap

bootstrap_engine = BCaBootstrap(n_bootstrap=10000, random_state=42)
result = bootstrap_engine.analyze_paired_differences(paired_data)
print(f"95% CI: [{result.ci_95_lower:.6f}, {result.ci_95_upper:.6f}]")
```

### 2. False Discovery Rate Control (`fdr.py`)
- **Benjamini-Hochberg FDR correction** for multiple comparisons
- Hierarchical FDR control within metric families
- Family-wise error rate protection
- Integration with bootstrap confidence intervals

```python
from packrepo.evaluator.statistics import FDRController

fdr_controller = FDRController(alpha=0.05, method="benjamini_hochberg")
fdr_result = fdr_controller.analyze_multiple_comparisons(test_results)
```

### 3. Effect Size Analysis (`effect_size.py`)
- **Cohen's d** with pooled variance estimator
- **Hedges' g** with small sample bias correction
- **Glass's delta** for different variance assumptions
- Bootstrap confidence intervals for effect sizes
- Practical significance thresholds with business interpretation

```python
from packrepo.evaluator.statistics import EffectSizeAnalyzer

analyzer = EffectSizeAnalyzer()
effect_result = analyzer.analyze_effect_sizes(control_data, treatment_data)
print(f"Cohen's d: {effect_result.cohens_d:.3f} ({effect_result.effect_magnitude})")
```

### 4. Paired Analysis Framework (`paired_analysis.py`)
- **Per-question matched-pair analysis** with bootstrap resampling
- Paired t-test with non-parametric alternatives (Wilcoxon signed-rank)
- Question-level effect heterogeneity analysis
- Multi-run reliability assessment

```python
from packrepo.evaluator.statistics import PairedAnalysisFramework

framework = PairedAnalysisFramework(random_state=42)
paired_result = framework.analyze_paired_comparison(
    evaluation_data, control_variant, treatment_variant, metric_name
)
```

### 5. Statistical Reporting (`reporting.py`)
- **Two-slice analysis**: Focused (objective-like) vs Full (general comprehension)
- Publication-quality statistical summaries
- Cross-slice consistency validation
- Automated report generation in multiple formats (JSON, Markdown, CSV)

```python
from packrepo.evaluator.statistics import StatisticalReporter

reporter = StatisticalReporter()
report = reporter.generate_comprehensive_report(
    evaluation_data, bootstrap_results, fdr_results, effect_size_results, output_dir
)
```

### 6. Acceptance Gates (`acceptance_gates.py`)
- **Multi-criteria promotion decisions** based on statistical evidence
- Primary KPI gates with confidence interval requirements
- Business impact assessment with cost-benefit analysis
- Risk evaluation and remediation recommendations

```python
from packrepo.evaluator.statistics import AcceptanceGateManager

gate_manager = AcceptanceGateManager()
gate_evaluation = gate_manager.evaluate_all_gates(
    bootstrap_results, fdr_results, effect_size_results, statistical_report, metadata
)
print(f"Decision: {gate_evaluation.promotion_decision}")
```

## Statistical Methodology

### Primary KPI Validation
- **Target**: ≥ +20% absolute improvement in Q&A accuracy per 100k tokens
- **Evidence Standard**: BCa 95% CI lower bound > 0
- **Multiple Comparison Control**: FDR correction within metric families
- **Effect Size**: Practical significance assessment with business impact

### Bootstrap Methodology
- **Method**: Bias-Corrected and Accelerated (BCa) bootstrap
- **Iterations**: 10,000 for stable confidence intervals
- **Sampling**: Preserves question-level pairing structure
- **Corrections**: Bias correction and acceleration for skewness

### Two-Slice Analysis
- **Focused Slice**: Objective-like questions requiring specific factual responses
- **Full Slice**: General comprehension questions covering broad understanding
- **Consistency Check**: Cross-slice agreement validation
- **Decision Integration**: Both slices must show consistent evidence

## Usage Examples

### Complete Analysis Pipeline

```python
from packrepo.evaluator.statistics import run_complete_statistical_analysis

# Run complete statistical analysis
summary = run_complete_statistical_analysis(
    evaluation_data_file="qa_results.jsonl",
    control_variant="V0",
    treatment_variants=["V1", "V2", "V3"],
    metric_name="qa_accuracy_per_100k",
    output_dir="statistical_analysis_results",
    bootstrap_iterations=10000,
    fdr_alpha=0.05,
    random_state=42
)

print(f"Final Decision: {summary['final_decision']}")
print(f"Primary KPI Achieved: {summary['primary_kpi_achieved']}")
print(f"Promoted Variants: {summary['promoted_variants']}")
```

### Individual Component Usage

```python
# BCa Bootstrap analysis
from packrepo.evaluator.statistics import BCaBootstrap, load_paired_data_from_jsonl

paired_data = load_paired_data_from_jsonl("results.jsonl", "qa_accuracy", "V0", "V1")
bootstrap_engine = BCaBootstrap(n_bootstrap=10000)
bootstrap_result = bootstrap_engine.analyze_paired_differences(paired_data)

# Effect size analysis  
from packrepo.evaluator.statistics import EffectSizeAnalyzer

analyzer = EffectSizeAnalyzer()
effect_result = analyzer.analyze_effect_sizes(control_data, treatment_data)

# FDR correction
from packrepo.evaluator.statistics import FDRController

fdr_controller = FDRController(alpha=0.05)
fdr_result = fdr_controller.analyze_multiple_comparisons(test_results)
```

## Data Format Requirements

### Evaluation Data (JSONL)
Each line should contain:
```json
{
    "variant": "V1",
    "question_id": "q_001", 
    "qa_accuracy_per_100k": 0.85,
    "question": "What is the primary function of...",
    "question_type": "factual",
    "run_id": 1,
    "seed": 42
}
```

### Required Fields
- `variant`: Variant identifier (e.g., "V0", "V1", "V2")
- `question_id`: Unique question identifier for pairing
- `qa_accuracy_per_100k`: Primary metric value
- Additional fields for slice classification and metadata

## Output Structure

```
statistical_analysis_results/
├── bootstrap_V1_vs_V0.json          # Individual bootstrap results
├── bootstrap_V2_vs_V0.json
├── effect_size_V1_vs_V0.json        # Effect size analyses  
├── effect_size_V2_vs_V0.json
├── paired_analysis_V1_vs_V0.json    # Paired analysis results
├── paired_analysis_V2_vs_V0.json
├── fdr_analysis.json                # FDR correction results
├── acceptance_gate_evaluation.json  # Gate evaluation results
├── analysis_summary.json            # Overall summary
└── statistical_report/              # Comprehensive report
    ├── statistical_report.json
    ├── statistical_report.md
    ├── primary_kpi_results.csv
    ├── focused_slice_comparisons.csv
    └── full_slice_comparisons.csv
```

## Acceptance Gate Criteria

### Safety Gates
- **Data Quality**: Sufficient sample size and completeness
- **Statistical Assumptions**: Key assumptions reasonably satisfied
- **Bootstrap Validity**: Stable bootstrap analysis

### Primary KPI Gates  
- **Primary KPI Achievement**: ≥ +20% token efficiency improvement
- **CI Lower Bound Positive**: 95% confidence interval excludes zero
- **Statistical Significance**: FDR-corrected p-value < 0.05

### Consistency Gates
- **Slice Consistency**: Agreement between Focused and Full slices
- **Multi-run Reliability**: Stable results across evaluation runs (CV ≤ 1.5%)
- **Judge Agreement**: Inter-judge reliability κ ≥ 0.6

### Business Impact Gates
- **Practical Significance**: Effect size indicates meaningful impact
- **Cost-Benefit Ratio**: Benefits justify implementation costs
- **Deployment Risk**: Risk within acceptable bounds

## Command Line Interface

Each module provides a standalone CLI:

```bash
# BCa Bootstrap analysis
python -m packrepo.evaluator.statistics.bootstrap_bca \
    results.jsonl qa_accuracy_per_100k V0 V1 bootstrap_results.json 10000

# FDR correction
python -m packrepo.evaluator.statistics.fdr \
    bootstrap_results/ fdr_analysis.json 0.05 benjamini_hochberg

# Effect size analysis
python -m packrepo.evaluator.statistics.effect_size \
    results.jsonl qa_accuracy_per_100k V0 V1 effect_results.json

# Paired analysis
python -m packrepo.evaluator.statistics.paired_analysis \
    results.jsonl V0 V1 qa_accuracy_per_100k paired_results.json

# Statistical reporting
python -m packrepo.evaluator.statistics.reporting \
    results.jsonl bootstrap_results/ fdr_analysis.json effect_results/ report_output/

# Acceptance gates
python -m packrepo.evaluator.statistics.acceptance_gates \
    bootstrap_results/ fdr_analysis.json effect_results/ report.json metadata.json
```

## Dependencies

Core dependencies:
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `pandas`: Data manipulation (reporting)
- `dataclasses`: Structured data containers

## Validation and Quality Assurance

### Statistical Validation
- Bootstrap confidence intervals contain true parameters (coverage testing)
- FDR correction maintains appropriate error rates  
- Effect size calculations match literature standards
- Statistical power analysis for sample size validation

### Code Quality
- Comprehensive unit tests with >90% coverage
- Integration tests with synthetic data
- Performance benchmarks for large datasets
- Memory efficiency validation for bootstrap iterations

### Reproducibility
- Fixed random seeds for deterministic results
- Version pinning for all dependencies
- Comprehensive logging and audit trails
- Export to standard statistical formats

## Integration with PackRepo

This statistical framework integrates with the broader PackRepo evaluation system:

1. **Evaluation Harness** → generates question-level results
2. **Statistical Framework** → analyzes results with rigorous methodology  
3. **Acceptance Gates** → makes evidence-based promotion decisions
4. **CI/CD Pipeline** → automates promotion workflow based on gates

## Limitations and Assumptions

- Bootstrap confidence intervals assume representative sampling
- FDR correction controls expected proportion of false discoveries
- Two-slice classification relies on heuristic question categorization
- Cross-validation with independent datasets recommended
- Long-term monitoring needed to validate evaluation results

## References

- Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
- Benjamini & Hochberg (1995), "Controlling the False Discovery Rate"
- Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences"
- Hedges & Olkin (1985), "Statistical Methods for Meta-Analysis"

---

For additional documentation and examples, see the individual module docstrings and the `examples/` directory.
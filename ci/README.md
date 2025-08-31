# PackRepo FastPath V2 CI/CD Pipeline

Complete CI/CD automation system for PackRepo FastPath V2 validation, statistical analysis, and promotion decisions.

## 🎯 Overview

The FastPath V2 CI/CD pipeline implements a comprehensive workflow automation system that validates performance improvements through systematic benchmarking, statistical analysis, and automated acceptance gates. The system ensures high-confidence promotion decisions based on rigorous statistical evidence.

### Key Features

- **🔄 Complete Workflow Automation** - B.U.R.T.E methodology (Building → Running → Tracking → Evaluating)
- **📊 Statistical Validation** - BCa bootstrap confidence intervals with 95% confidence
- **⚡ Parallel Execution** - Concurrent benchmark execution for efficiency
- **🚪 Acceptance Gates** - Automated promotion/rejection decisions
- **📈 Performance Monitoring** - Comprehensive latency and quality tracking
- **🔒 Security Integration** - SAST scanning and vulnerability assessment

## 🏗️ Architecture

```
PackRepo FastPath V2 CI/CD Pipeline
├── Building Workflow (B0-B2)      # Environment & Validation
├── Running Workflow (R0-R4)       # Benchmarks & Testing  
├── Tracking Workflow (T1-T2)      # Statistical Analysis
└── Evaluating Workflow (E1)       # Acceptance Gates
```

### Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `pipeline.py` | Main orchestrator | Workflow coordination, gate evaluation |
| `benchmark_runner.py` | Benchmark execution | Flag management, parallel execution |
| `statistical_analysis.py` | Statistical validation | BCa bootstrap, power analysis |
| `run_pipeline.sh` | CLI wrapper | Environment setup, error handling |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment support
- `run_fastpath_benchmarks.py` in project root
- `requirements.txt` with dependencies

### Basic Execution

```bash
# Run complete pipeline
./ci/run_pipeline.sh

# Run specific variants
./ci/run_pipeline.sh --variants "baseline,V1,V3"

# Run with verbose output
./ci/run_pipeline.sh --verbose

# Dry run to see execution plan
./ci/run_pipeline.sh --dry-run
```

### CI Integration

The pipeline is designed for GitHub Actions but works in any CI environment:

```yaml
# .github/workflows/fastpath-v2-pipeline.yml
- name: Run FastPath V2 Pipeline
  run: ./ci/run_pipeline.sh --parallel --verbose
```

## 📋 Workflow Details

### Building Workflow (B0-B2)

**Purpose:** Environment setup and validation

```bash
# B0: Environment Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel && pip install -r requirements.txt

# B1: Static Analysis  
ruff check . --output-format=json
mypy --strict packrepo --json-report artifacts/mypy
bandit -r packrepo -f json -o artifacts/bandit.json

# B2: Hermetic Boot & Golden Smokes
pytest -q tests -k "smoke or basic" --maxfail=1 --disable-warnings
```

**Outputs:**
- `artifacts/boot_env.json` - Environment manifest
- `artifacts/bandit.json` - Security analysis
- `artifacts/smoke_tests.json` - Smoke test results

### Running Workflow (R0-R4)

**Purpose:** Systematic benchmark execution with flag isolation

#### R0: Baseline Benchmark
```bash
FASTPATH_POLICY_V2=0 python run_fastpath_benchmarks.py \
  --budgets 50k,120k,200k --seed 1337 --output artifacts/baseline.json
```

#### R1: Experimental Variants (V1-V5)

Each variant runs with specific flag combinations:

| Variant | Flags | Purpose |
|---------|-------|---------|
| V1 | `FASTPATH_POLICY_V2=1`, `FASTPATH_AGGRESSIVE_CACHING=1` | Aggressive caching |
| V2 | `FASTPATH_POLICY_V2=1`, `FASTPATH_PARALLEL_PROCESSING=1` | Parallel processing |
| V3 | `FASTPATH_POLICY_V2=1`, `FASTPATH_MEMORY_OPTIMIZATION=1` | Memory optimization |
| V4 | `FASTPATH_POLICY_V2=1`, `FASTPATH_HYBRID_MODE=1` | Hybrid approach |
| V5 | All flags combined | Maximum optimization |

#### R2-R4: Advanced Testing
- **R2:** Property/metamorphic/mutation/fuzz testing
- **R3:** Differential comparison against known-good
- **R4:** Shadow traffic invariant testing (10k requests)

**Outputs:**
- `artifacts/{variant}_benchmark.json` - Individual results
- `artifacts/benchmark_comparison.json` - Comparative analysis
- `artifacts/advanced_testing.json` - Advanced test results

### Tracking Workflow (T1-T2)

**Purpose:** Statistical analysis and confidence interval computation

#### T1: BCa Bootstrap Analysis
Implements Bias-Corrected and accelerated (BCa) bootstrap for robust confidence intervals:

```python
# BCa Bootstrap Process
1. Bootstrap resampling (10,000 samples)
2. Bias correction computation
3. Acceleration parameter (jackknife)
4. Adjusted percentile confidence interval
```

**Key Metrics:**
- Point estimate improvement
- 95% confidence interval bounds
- Statistical significance (CI excludes 0)
- Statistical power analysis

#### T2: Risk Assessment
Normalizes decision features for risk evaluation:

- Performance risk (latency regression)
- Quality risk (mutation score deficiency)  
- Coverage risk (property test gaps)
- Stability risk (differential testing)

**Outputs:**
- `artifacts/bca_analysis.json` - Bootstrap results
- `artifacts/risk_assessment.json` - Risk factors
- `artifacts/statistical_analysis.json` - Complete analysis

### Evaluating Workflow (E1)

**Purpose:** Automated acceptance gate evaluation

#### Acceptance Criteria

| Gate | Threshold | Purpose |
|------|-----------|---------|
| **QA/100k Improvement** | ≥+13% | Quality assurance improvement |
| **Performance Regression** | ≤10% | Latency impact control |
| **Mutation Score** | ≥80% | Test quality validation |
| **Property Coverage** | ≥70% | Property test coverage |
| **Statistical Significance** | BCa CI > 0 | Confidence in improvement |
| **Security** | 0 critical issues | Vulnerability control |

#### Decision Logic

```python
promote = (
    qa_improvement >= 0.13 and
    performance_regression <= 0.10 and
    mutation_score >= 0.80 and
    property_coverage >= 0.70 and
    bca_ci_lower > 0 and
    security_issues == 0
)
```

**Outputs:**
- `artifacts/pipeline_report.json` - Final decision
- `artifacts/gates.json` - Gate-by-gate results

## 📊 Statistical Methodology

### BCa Bootstrap Confidence Intervals

The pipeline uses BCa (Bias-Corrected and accelerated) bootstrap for robust statistical inference:

**Advantages:**
- ✅ Non-parametric (no distribution assumptions)
- ✅ Bias correction for more accurate intervals
- ✅ Acceleration adjustment for skewness
- ✅ Higher-order accurate (second-order correct)

**Implementation:**
```python
# Bias correction (z₀)
z0 = Φ⁻¹(#{θ* < θ̂} / B)

# Acceleration (â) via jackknife
â = Σ(θ̄ - θᵢ)³ / (6[Σ(θ̄ - θᵢ)²]³/²)

# BCa percentiles
α₁ = Φ(z₀ + (z₀ + z_{α/2})/(1 - â(z₀ + z_{α/2})))
α₂ = Φ(z₀ + (z₀ + z_{1-α/2})/(1 - â(z₀ + z_{1-α/2})))
```

### Power Analysis

Statistical power ensures adequate sample sizes for reliable detection:

**Target Power:** ≥80%
**Effect Size:** Cohen's d with interpretation
- Small: 0.2 ≤ |d| < 0.5
- Medium: 0.5 ≤ |d| < 0.8  
- Large: |d| ≥ 0.8

### Multiple Comparison Correction

Bonferroni correction for family-wise error rate:
- **Adjusted α:** 0.05 / number_of_variants
- **Purpose:** Control Type I error inflation

## 🔧 Configuration

### Environment Variables

```bash
# Execution control
CI=true                          # CI environment flag
GITHUB_ACTIONS=true             # GitHub Actions flag
ARTIFACTS_DIR=artifacts         # Output directory
PIPELINE_TIMEOUT=7200           # 2-hour timeout

# FastPath V2 feature flags  
FASTPATH_POLICY_V2=1            # Enable V2 policy
FASTPATH_VARIANT=V1             # Specific variant
FASTPATH_AGGRESSIVE_CACHING=1   # Caching optimization
FASTPATH_PARALLEL_PROCESSING=1  # Parallel execution
FASTPATH_MEMORY_OPTIMIZATION=1  # Memory optimization
FASTPATH_HYBRID_MODE=1          # Hybrid approach
FASTPATH_ADAPTIVE_THRESHOLD=0.85 # Adaptive threshold
```

### Acceptance Thresholds

```python
thresholds = {
    "qa_improvement": 0.13,        # +13% QA/100k improvement
    "performance_regression": 0.10, # ≤10% latency regression
    "mutation_score": 0.80,        # ≥80% mutation coverage
    "property_coverage": 0.70,     # ≥70% property coverage
    "baseline_qa": 0.7230,         # Current QA/100k baseline
    "baseline_latency": 896,       # Baseline p95 latency (ms)
}
```

## 📈 Monitoring & Observability

### Key Performance Indicators

| Metric | Target | Measurement |
|--------|--------|-------------|
| **QA/100k** | 0.8170 (+13%) | Quality assurance per 100k operations |
| **Latency P95** | ≤986ms (≤10% regression) | 95th percentile response time |
| **Mutation Score** | ≥80% | Test quality coverage |
| **Statistical Power** | ≥80% | Ability to detect true effects |

### Artifact Structure

```
artifacts/
├── pipeline_report.json         # Final promotion decision
├── benchmark_suite_summary.json # Complete benchmark results
├── statistical_analysis.json    # Statistical validation
├── {variant}_benchmark.json     # Individual variant results
├── bca_analysis.json            # Bootstrap confidence intervals
├── risk_assessment.json         # Risk factor analysis
├── bandit.json                  # Security scan results
├── smoke_tests.json             # Basic functionality tests
└── pipeline.log                 # Execution logs
```

### Integration Points

```bash
# Manual execution
./ci/run_pipeline.sh --variants "baseline,V1,V2"

# GitHub Actions
uses: ./.github/workflows/fastpath-v2-pipeline.yml

# CI integration
python ci/pipeline.py --verbose --project-root .

# Statistical analysis only
python ci/statistical_analysis.py \
  --input artifacts/benchmark_suite_summary.json \
  --output artifacts/stats.json
```

## 🚨 Troubleshooting

### Common Issues

#### Pipeline Timeout
```bash
# Increase timeout
./ci/run_pipeline.sh --timeout 10800  # 3 hours

# Run sequentially to reduce resource contention
./ci/run_pipeline.sh --sequential
```

#### Benchmark Failures
```bash
# Check individual variant logs
cat artifacts/{variant}_result.json

# Run single variant for debugging
python ci/benchmark_runner.py --variants V1 --verbose
```

#### Statistical Analysis Issues
```bash
# Check sample sizes and power
python ci/statistical_analysis.py \
  --input artifacts/benchmark_suite_summary.json \
  --bootstrap-samples 1000 \
  --verbose
```

#### Environment Issues
```bash
# Validate environment
./ci/run_pipeline.sh --dry-run

# Manual environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - promotion approved |
| 1 | Failure - acceptance criteria not met |
| 124 | Timeout - pipeline exceeded time limit |
| 130 | Interrupted - user cancellation |

### Debugging

```bash
# Full verbose execution
./ci/run_pipeline.sh --verbose --dry-run

# Component-level debugging
python ci/pipeline.py --verbose
python ci/benchmark_runner.py --verbose --variants "V1"
python ci/statistical_analysis.py --verbose --input results.json

# Log analysis
tail -f artifacts/pipeline.log
grep "ERROR\|WARN" artifacts/pipeline.log
```

## 🔄 Development Workflow

### Local Development
```bash
# Quick validation
./ci/run_pipeline.sh --skip-building --variants "V1,V2"

# Development with mock data  
./ci/run_pipeline.sh --dry-run --verbose

# Statistical analysis only
python ci/statistical_analysis.py \
  --input existing_results.json \
  --confidence-level 0.90
```

### CI Integration
```yaml
# Minimal CI validation
- run: ./ci/run_pipeline.sh --variants "baseline,V1"

# Full production validation
- run: ./ci/run_pipeline.sh --parallel --verbose
  timeout-minutes: 120
```

### Extension Points

1. **Custom Variants**: Add to `benchmark_runner.py` flag configurations
2. **Additional Gates**: Extend `pipeline.py` evaluating workflow  
3. **New Metrics**: Modify `_extract_metrics()` in benchmark runner
4. **Statistical Tests**: Add methods to `StatisticalAnalyzer`

## 📚 References

- [BCa Bootstrap Theory](https://doi.org/10.1214/aos/1176345338)
- [Statistical Power Analysis](https://www.jstor.org/stable/2281868)
- [Multiple Comparison Procedures](https://doi.org/10.2307/2346101)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Version:** 2.0.0  
**Last Updated:** 2024  
**Maintainer:** PackRepo Development Team
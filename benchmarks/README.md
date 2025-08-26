# FastPath Evaluation Framework

This directory contains the complete evaluation framework for FastPath V1-V5 variants as specified in `TODO.md`. The system provides rigorous paired evaluation with statistical validation and negative controls.

## Overview

The evaluation framework implements the run matrix from TODO.md with:
- **Baseline + V1→V5 evaluation** with progressive feature enablement
- **Paired bootstrap analysis** with BCa 95% confidence intervals  
- **FDR control** for multiple comparison correction
- **Negative controls** for validation (scramble, flip, random_quota)
- **Paper synchronization** for automatic LaTeX updates

## Components

### Core Evaluation Scripts

- **`run.py`** - Main evaluation runner for all variants
- **`paired_bootstrap.py`** - BCa bootstrap with FDR control
- **`collect.py`** - Artifact consolidation system
- **`score.py`** - Statistical analysis framework

### Usage Examples

```bash
# 1. Run baseline evaluation
FASTPATH_POLICY_V2=0 python benchmarks/run.py \
  --system baseline --budgets 50k,120k,200k \
  --paired --seeds 100 --emit artifacts/baseline.jsonl

# 2. Run FastPath V5 evaluation  
FASTPATH_POLICY_V2=1 FASTPATH_CENTRALITY=1 FASTPATH_DEMOTE=1 \
FASTPATH_PATCH=1 FASTPATH_ROUTER=1 python benchmarks/run.py \
  --system v5 --budgets 50k,120k,200k \
  --paired --seeds 100 --emit artifacts/v5.jsonl

# 3. Run negative control (graph scramble)
FASTPATH_NEGCTRL=scramble python benchmarks/run.py \
  --system scramble --budgets 50k,120k,200k \
  --paired --seeds 50 --emit artifacts/ctrl_scramble.jsonl

# 4. Consolidate results
python benchmarks/collect.py \
  --glob "artifacts/*.jsonl" --out artifacts/collected.json

# 5. Compute paired bootstrap CIs with FDR
python benchmarks/paired_bootstrap.py \
  --baseline artifacts/baseline.jsonl \
  --exp artifacts/v5.jsonl --bca --iters 10000 --fdr \
  --out artifacts/ci.json

# 6. Statistical analysis
python benchmarks/score.py \
  --in artifacts/collected.json \
  --bootstrap artifacts/ci.json \
  --out artifacts/analysis.json \
  --promotion-decision
```

## Evaluation Run Matrix

| System | Feature Flags | Expected Gain | Validation |
|--------|---------------|---------------|------------|
| Baseline | All off | 0% (reference) | Stability check |
| V1 | `POLICY_V2=1` | +8-15% QA/100k | Quotas + greedy |
| V2 | V1 + `CENTRALITY=1` | +3-6% over V1 | PageRank centrality |
| V3 | V2 + `DEMOTE=1` | +4-8% over V2 | Hybrid demotion |
| V4 | V3 + `PATCH=1` | +3-7% over V3 | Two-pass patch |
| V5 | V4 + `ROUTER=1` | +2-4% over V4 | Router + bandit |

## Negative Controls

| Control | Environment Variable | Expected Result | Purpose |
|---------|---------------------|-----------------|---------|
| Graph Scramble | `FASTPATH_NEGCTRL=scramble` | ≈ 0% change | Validate centrality causality |
| Edge Flip | `FASTPATH_NEGCTRL=flip` | ≤ 0% change | Confirm directionality matters |
| Random Quota | `FASTPATH_NEGCTRL=random_quota` | ≤ 0% change | Validate greedy allocation |

## Statistical Validation

The framework ensures research-grade statistical rigor:

- **Paired Evaluation**: Consistent seeds/repos/questions across variants
- **BCa Bootstrap**: 10,000 iterations with bias correction and acceleration
- **FDR Control**: Benjamini-Hochberg procedure for multiple comparisons
- **Effect Sizes**: Cohen's d with 95% confidence intervals
- **Promotion Gates**: Conservative criteria for production deployment

### Acceptance Criteria (from TODO.md)

- **Quality**: QA/100k tokens **≥ +13%** vs baseline (BCa 95% CI lower bound > 0)
- **Category Targets**: Usage ≥ 70/100, Config ≥ 65/100, no regression > 5 points
- **Statistical**: FDR-corrected significance with conservative effect sizes
- **Reproducibility**: Hermetic boot transcript with cryptographic signing

## Integration with Paper

The framework automatically updates LaTeX papers:

```bash
# Generate paper tables and figures
python paper/update_tables.py \
  --ci artifacts/ci.json \
  --ir artifacts/ir.jsonl \
  --neg artifacts/ctrl_*.jsonl \
  --out paper/tables.tex

python paper/update_figures.py \
  --metrics artifacts --out paper/figures/

python paper/patch_tex.py \
  --tex paper/fastpath.tex \
  --tables paper/tables.tex \
  --figdir paper/figures
```

## File Structure

```
benchmarks/
├── __init__.py                    # Package initialization
├── README.md                     # This documentation
├── run.py                        # Main evaluation runner
├── paired_bootstrap.py           # BCa bootstrap + FDR
├── collect.py                    # Artifact consolidation  
└── score.py                      # Statistical analysis

paper/
├── __init__.py                   # Package initialization
├── update_tables.py              # Generate LaTeX tables
├── update_figures.py             # Generate publication figures
└── patch_tex.py                  # Update LaTeX documents

scripts/
└── sign_boot.py                  # Boot transcript signing
```

## Validation Workflow

The complete evaluation follows this workflow:

```bash
# 1. Environment setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run evaluation matrix  
./run_evaluation_matrix.sh  # Runs all variants + controls

# 3. Statistical analysis
python benchmarks/collect.py --glob "artifacts/*.jsonl" --out artifacts/collected.json
python benchmarks/paired_bootstrap.py --baseline artifacts/baseline.jsonl --exp artifacts/v5.jsonl --bca --iters 10000 --fdr --out artifacts/ci.json
python benchmarks/score.py --in artifacts/collected.json --bootstrap artifacts/ci.json --out artifacts/analysis.json --promotion-decision

# 4. Paper synchronization
python paper/update_tables.py --ci artifacts/ci.json --neg artifacts/ctrl_*.jsonl --out paper/tables.tex
python paper/update_figures.py --metrics artifacts --out paper/figures/
python paper/patch_tex.py --tex paper/fastpath.tex --tables paper/tables.tex --figdir paper/figures

# 5. Boot transcript signing
python scripts/sign_boot.py --in artifacts/smoke.json --out artifacts/boot_transcript.json
```

## Quality Assurance

The evaluation framework enforces research-grade quality:

- **Reproducibility**: Deterministic with fixed seeds, signed boot transcripts
- **Statistical Rigor**: Conservative BCa bootstrap, FDR correction, effect sizes
- **Negative Controls**: Validation that improvements are algorithmic, not artifacts
- **Paired Analysis**: Eliminates confounding variables through consistent test cases
- **Promotion Gates**: Conservative criteria prevent false positives in production

This system ensures that FastPath improvements are both statistically significant and practically meaningful, supporting confident deployment decisions with full research reproducibility.
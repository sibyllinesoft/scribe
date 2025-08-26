# PackRepo Acceptance Gates System

A comprehensive quality assurance framework implementing the acceptance gate requirements from `TODO.md`. This system provides evidence-based promotion decisions with full transparency, risk assessment, and automated workflows.

## Overview

The acceptance gate system implements a multi-layered quality validation framework:

### 🎯 Core Components

1. **Acceptance Gates Engine** (`scripts/acceptance_gates.py`)
   - Validates all TODO.md acceptance criteria
   - Provides comprehensive evidence collection
   - Generates detailed pass/fail determinations

2. **Gatekeeper Decision System** (`scripts/gatekeeper.py`)
   - Multi-criteria decision analysis
   - Risk assessment and mitigation strategies
   - Routes decisions: PROMOTE | AGENT_REFINE | MANUAL_QA

3. **Pipeline Orchestrator** (`scripts/run_acceptance_pipeline.py`)
   - End-to-end workflow automation
   - Audit trail generation
   - Next action execution

4. **Setup Validator** (`scripts/validate_acceptance_setup.py`)
   - Pre-flight system validation
   - Configuration verification
   - Dependency checking

## 🏗️ Acceptance Gates (from TODO.md)

### Spin-up Gate
- ✅ Clean checkout → container build → dataset fetch
- ✅ Readiness verification → golden smoke tests
- ✅ Signed boot transcript generation

### Static Analysis Gate
- ✅ Zero high/critical SAST issues
- ✅ Clean type checking (mypy/pyright)
- ✅ License policy compliance
- ✅ API surface diff acknowledgment

### Dynamic Testing Gate
- ✅ Mutation score ≥ T_mut (default: 0.80)
- ✅ Property/metamorphic coverage ≥ T_prop (default: 0.70)
- ✅ Fuzz testing ≥ FUZZ_MIN minutes with 0 medium+ crashes

### Budget Parities Gate
- ✅ Selection budgets within ±5% tolerance
- ✅ Decode budgets logged and tracked
- ✅ Tokenizer version pinned and verified

### Primary KPI Gate
- ✅ V1/V2/V3 beat V0c baseline
- ✅ BCa 95% CI lower bound > 0 on QA acc/100k
- ✅ Statistical significance validation

### Stability Gate
- ✅ 3-run reproducibility verification
- ✅ Judge agreement κ ≥ 0.6
- ✅ Flakiness rate < 1%

### Performance Gate
- ✅ P50 latency ≤ +30% vs baseline
- ✅ P95 latency ≤ +50% vs baseline
- ✅ Memory usage ≤ 8GB limit

## 🚀 Quick Start

### 1. Validate Setup
```bash
python scripts/validate_acceptance_setup.py
```

### 2. Run Complete Pipeline
```bash
python scripts/run_acceptance_pipeline.py V2
```

### 3. View Results
```bash
# Check decision
cat artifacts/metrics/gatekeeper_decision.json

# View comprehensive report
cat artifacts/pipeline_results.json
```

## 📋 Detailed Usage

### Individual Component Usage

#### Acceptance Gates Only
```bash
python scripts/acceptance_gates.py V2 scripts/gates.yaml artifacts/acceptance_results.json
```

#### Gatekeeper Only
```bash
python scripts/gatekeeper.py artifacts/metrics V2 scripts/gates.yaml artifacts/decision.json
```

#### Integration Test
```bash
python scripts/test_acceptance_integration.py
```

### Configuration

The system is configured via `scripts/gates.yaml`:

```yaml
gates:
  mutation_score:
    threshold: 0.80
    direction: "min"
    weight: 0.15
    critical: true
    
  sast_high_critical:
    threshold: 0
    direction: "max"
    weight: 0.20
    critical: true

promotion_rules:
  V2:
    required_gates:
      - sast_high_critical
      - mutation_score
      - primary_kpi
    additional_criteria:
      - "CI lower bound > 0 vs V1"

risk_assessment:
  low_risk_threshold: 0.3
  medium_risk_threshold: 0.7
  high_risk_threshold: 1.0
```

## 🎯 Decision Matrix

The gatekeeper implements a sophisticated decision matrix:

### PROMOTE
- ✅ All critical gates passed
- ✅ Composite score ≥ 85%
- ✅ CI-backed wins demonstrated (V2/V3)
- ✅ Risk level: LOW or MEDIUM

### AGENT_REFINE
- 🔧 Failed gates with concrete remediation paths
- 🔧 Specific obligations and thresholds provided
- 🔧 Automated retry capability

### MANUAL_QA
- 👥 High complexity requiring human judgment
- 👥 Risk level: HIGH
- 👥 Critical failures > 2
- 👥 Edge cases detected (judge κ < 0.6, oscillations > 1)

## 📊 Risk Assessment

Multi-dimensional risk scoring across:

### Security Risk (40% weight)
- SAST high/critical issue count
- Vulnerability severity assessment
- Security policy compliance

### Quality Risk (30% weight) 
- Mutation test score vs threshold
- Property-based test coverage
- Test reliability metrics

### Performance Risk (20% weight)
- Latency overhead vs baselines
- Memory usage vs limits  
- Throughput degradation

### Stability Risk (10% weight)
- Deterministic run consistency
- Budget control violations
- Oscillation patterns (V3)

## 📁 Directory Structure

```
artifacts/
├── metrics/
│   ├── acceptance_gate_results.json    # Gate evaluation results
│   ├── gatekeeper_decision.json        # Final promotion decision
│   ├── qa_acc_ci.json                  # Bootstrap CI analysis
│   └── all_metrics.jsonl               # Aggregated metrics
├── reports/
│   ├── evidence_report.json            # Audit trail evidence
│   ├── pipeline_summary.json           # Execution summary
│   └── audit_trail.json                # Compliance documentation
├── boot_transcript.json                # Hermetic boot verification
├── pipeline_results.json               # Complete pipeline results
└── deployment_readiness.json           # Promotion checklist (PROMOTE)
    refinement_tracking.json             # Obligations (AGENT_REFINE)
    manual_qa_package.json               # Review package (MANUAL_QA)
```

## 🔧 Integration with CI/CD

### GitHub Actions Integration

```yaml
- name: Run PackRepo Acceptance Gates
  run: |
    python scripts/run_acceptance_pipeline.py ${{ matrix.variant }}
    
- name: Upload Results
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: acceptance-gate-results
    path: artifacts/
    
- name: Check Decision
  run: |
    DECISION=$(jq -r '.decision' artifacts/metrics/gatekeeper_decision.json)
    echo "::set-output name=decision::$DECISION"
    case $DECISION in
      "PROMOTE") echo "✅ Promotion approved" ;;
      "AGENT_REFINE") echo "🔄 Refinement required" && exit 1 ;;
      "MANUAL_QA") echo "👥 Manual review needed" && exit 2 ;;
    esac
```

### Exit Codes
- `0`: PROMOTE - Ready for production
- `1`: AGENT_REFINE - Automated refinement required
- `2`: MANUAL_QA - Human review required  
- `3`: Pipeline failure - System error

## 📈 Monitoring & Observability

### Key Metrics Tracked
- Gate pass/fail rates over time
- Composite quality scores by variant
- Risk level distribution
- Decision type frequency
- Pipeline execution times

### Evidence Collection
- Complete audit trail with checksums
- Statistical validation artifacts
- Performance benchmark data
- Security scan results
- Test coverage reports

### Compliance Reporting
- Traceability to TODO.md requirements
- Evidence of gate evaluation completeness
- Risk mitigation documentation
- Decision rationale preservation

## 🛠️ Troubleshooting

### Common Issues

#### "Acceptance gate results missing"
```bash
# Check if acceptance gates ran successfully
python scripts/acceptance_gates.py V2 --debug

# Verify test data exists
ls -la artifacts/
```

#### "CI lower bound not positive"
- Ensure statistical analysis completed: `artifacts/metrics/qa_acc_ci.json`
- Check QA accuracy data: `artifacts/metrics/qa_accuracy_summary.json`
- Verify bootstrap analysis: `python scripts/bootstrap_bca.py`

#### "High risk score"
- Review risk breakdown: `artifacts/metrics/gatekeeper_decision.json`
- Address security issues: `artifacts/sast_results.json`
- Improve test coverage: mutation score, property coverage

#### "Budget overruns detected"
- Check budget analysis: `artifacts/budget_analysis.json`
- Verify tokenizer configuration: `artifacts/tokenizer_info.json`
- Review selection logic for budget compliance

## 🧪 Testing

### Unit Tests
```bash
# Test individual components
python -m pytest tests/test_acceptance_gates.py
python -m pytest tests/test_gatekeeper.py
```

### Integration Tests
```bash
# Full system integration test
python scripts/test_acceptance_integration.py

# Component validation
python scripts/validate_acceptance_setup.py
```

### Mock Data Testing
The integration test creates comprehensive mock data to validate:
- Gate evaluation logic
- Decision matrix correctness
- Pipeline orchestration
- Report generation
- Error handling

## 📚 API Reference

### AcceptanceGateEngine

Main class for gate evaluation:

```python
from scripts.acceptance_gates import AcceptanceGateEngine

engine = AcceptanceGateEngine(config_path="scripts/gates.yaml")
results = engine.evaluate_all_gates(variant="V2")
composite_score = engine.compute_composite_score()
```

### GatekeeperDecision

Decision engine for promotion routing:

```python
from scripts.gatekeeper import GatekeeperDecision

gatekeeper = GatekeeperDecision()
gatekeeper.load_acceptance_gate_results(acceptance_file)
decision = gatekeeper.make_decision(metrics, variant="V2")
```

### AcceptancePipeline

Full pipeline orchestration:

```python
from scripts.run_acceptance_pipeline import AcceptancePipeline, PipelineConfig

config = PipelineConfig(variant="V2", artifacts_dir=Path("./artifacts"))
pipeline = AcceptancePipeline(config)
results = pipeline.run_pipeline()
```

## 🤝 Contributing

### Adding New Gates

1. Extend `AcceptanceGateEngine.evaluate_*_gate()` methods
2. Update `scripts/gates.yaml` configuration
3. Add validation in `validate_acceptance_setup.py`
4. Update documentation and tests

### Extending Decision Logic

1. Modify `GatekeeperDecision._make_promotion_decision()`
2. Update risk assessment weights
3. Add new decision criteria to gates configuration
4. Test with integration suite

## 📄 License

This acceptance gate system is part of PackRepo and follows the same license terms.

---

## 🎉 Success Criteria

A successful acceptance gate system provides:

✅ **Comprehensive Coverage**: All TODO.md requirements implemented  
✅ **Evidence-Based Decisions**: Statistical validation with CI-backed wins  
✅ **Risk-Aware Routing**: Multi-dimensional risk assessment  
✅ **Full Automation**: End-to-end pipeline with minimal human intervention  
✅ **Complete Transparency**: Full audit trail and compliance documentation  
✅ **Robust Error Handling**: Graceful degradation and clear error reporting  
✅ **Integration Ready**: CI/CD compatible with proper exit codes and artifacts  

The system enables confident, data-driven promotion decisions while maintaining the highest quality standards for PackRepo's production readiness.
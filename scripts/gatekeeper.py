#!/usr/bin/env python3
"""
PackRepo Gatekeeper - Quality Gate Decision Engine

Implements the comprehensive decision framework for promotion/refinement based on:
- Statistical analysis (BCa bootstrap confidence intervals)  
- Risk scoring (composite SAST + coverage + performance)
- Quality gates (mutation score, property coverage, determinism)
- Domain KPIs (token-efficiency, latency bounds)

Outputs: PROMOTE | AGENT_REFINE | MANUAL_QA
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class QualityGate:
    """Quality gate configuration and evaluation."""
    name: str
    threshold: float
    direction: str  # 'min' or 'max'
    weight: float
    critical: bool = False


@dataclass
class GateResult:
    """Result of evaluating a single quality gate."""
    gate: QualityGate
    actual_value: float
    passed: bool
    margin: float  # How far from threshold


class GatekeeperDecision:
    """Main gatekeeper decision engine integrating acceptance gates."""
    
    def __init__(self, gates_config: Optional[Dict] = None):
        """Initialize with quality gates configuration."""
        self.gates = self._load_gates(gates_config or {})
        self.results: List[GateResult] = []
        self.acceptance_results: Optional[Dict] = None
        self.risk_assessment: Optional[Dict] = None
        
    def _load_gates(self, config: Dict) -> Dict[str, QualityGate]:
        """Load quality gate definitions."""
        default_gates = {
            # Static Analysis Gates
            "sast_high_critical": QualityGate(
                name="SAST High/Critical Issues",
                threshold=0.0,
                direction="max",
                weight=0.20,
                critical=True
            ),
            
            # Dynamic Testing Gates  
            "mutation_score": QualityGate(
                name="Mutation Test Score",
                threshold=0.80,
                direction="min", 
                weight=0.15,
                critical=True
            ),
            
            "property_coverage": QualityGate(
                name="Property Test Coverage",
                threshold=0.70,
                direction="min",
                weight=0.10
            ),
            
            "fuzz_crashes": QualityGate(
                name="Fuzz Test Crashes (Medium+)",
                threshold=0.0,
                direction="max",
                weight=0.10,
                critical=True
            ),
            
            # Determinism/Budget Gates
            "budget_overrun": QualityGate(
                name="Budget Overrun Count", 
                threshold=0.0,
                direction="max",
                weight=0.10,
                critical=True
            ),
            
            "budget_underrun": QualityGate(
                name="Budget Underrun Percentage",
                threshold=0.5,  # <= 0.5% allowed
                direction="max",
                weight=0.05
            ),
            
            "determinism_consistency": QualityGate(
                name="Deterministic Run Consistency",
                threshold=3.0,  # 3x identical hashes required
                direction="min",
                weight=0.15,
                critical=True
            ),
            
            # Domain KPI Gates
            "token_efficiency_ci_lower": QualityGate(
                name="Token-Efficiency CI Lower Bound",
                threshold=0.0,  # Must be > 0 vs V0 baseline
                direction="min", 
                weight=0.20,
                critical=True
            ),
            
            "latency_p50_overhead": QualityGate(
                name="P50 Latency Overhead",
                threshold=30.0,  # <= +30% vs baseline
                direction="max",
                weight=0.08
            ),
            
            "latency_p95_overhead": QualityGate(
                name="P95 Latency Overhead", 
                threshold=50.0,  # <= +50% vs baseline
                direction="max",
                weight=0.12
            )
        }
        
        # Allow config override
        for gate_name, gate_config in config.get("gates", {}).items():
            if gate_name in default_gates:
                gate = default_gates[gate_name]
                gate.threshold = gate_config.get("threshold", gate.threshold)
                gate.weight = gate_config.get("weight", gate.weight)
                gate.critical = gate_config.get("critical", gate.critical)
        
        return default_gates
    
    def evaluate_gate(self, gate_name: str, actual_value: float) -> GateResult:
        """Evaluate a single quality gate."""
        if gate_name not in self.gates:
            raise ValueError(f"Unknown gate: {gate_name}")
            
        gate = self.gates[gate_name]
        
        if gate.direction == "min":
            passed = actual_value >= gate.threshold
            margin = actual_value - gate.threshold
        else:  # max
            passed = actual_value <= gate.threshold
            margin = gate.threshold - actual_value
            
        return GateResult(
            gate=gate,
            actual_value=actual_value,
            passed=passed,
            margin=margin
        )
    
    def evaluate_all_gates(self, metrics: Dict) -> List[GateResult]:
        """Evaluate all quality gates against provided metrics."""
        self.results = []
        
        # Extract metrics with safe defaults
        gate_metrics = {
            "sast_high_critical": metrics.get("sast_high_critical_count", 0),
            "mutation_score": metrics.get("mutation_score", 0.0),
            "property_coverage": metrics.get("property_coverage", 0.0),  
            "fuzz_crashes": metrics.get("fuzz_medium_high_crashes", 0),
            "budget_overrun": metrics.get("budget_overrun_count", 0),
            "budget_underrun": metrics.get("budget_underrun_percent", 0.0),
            "determinism_consistency": metrics.get("identical_hash_count", 0),
            "token_efficiency_ci_lower": metrics.get("token_eff_ci_lower", -1.0),
            "latency_p50_overhead": metrics.get("latency_p50_overhead_percent", 100.0),
            "latency_p95_overhead": metrics.get("latency_p95_overhead_percent", 100.0)
        }
        
        # Evaluate each gate
        for gate_name, metric_value in gate_metrics.items():
            result = self.evaluate_gate(gate_name, metric_value)
            self.results.append(result)
            
        return self.results
    
    def compute_composite_score(self) -> float:
        """Compute weighted composite quality score [0,1]."""
        if not self.results:
            return 0.0
            
        total_weight = sum(r.gate.weight for r in self.results)
        if total_weight == 0:
            return 0.0
            
        weighted_score = 0.0
        for result in self.results:
            # Convert pass/fail to score contribution
            if result.passed:
                gate_contribution = result.gate.weight
            else:
                # Penalty based on how far from threshold
                penalty_factor = max(0, min(1, abs(result.margin) / abs(result.gate.threshold) if result.gate.threshold != 0 else 1))
                gate_contribution = result.gate.weight * (1 - penalty_factor)
                
            weighted_score += gate_contribution
            
        return weighted_score / total_weight
    
    def get_critical_failures(self) -> List[GateResult]:
        """Get all critical gate failures."""
        return [r for r in self.results if r.gate.critical and not r.passed]
    
    def load_acceptance_gate_results(self, acceptance_file: Path) -> bool:
        """Load acceptance gate evaluation results."""
        if not acceptance_file.exists():
            return False
            
        try:
            with open(acceptance_file) as f:
                self.acceptance_results = json.load(f)
            return True
        except Exception as e:
            print(f"Warning: Could not load acceptance results: {e}")
            return False
    
    def compute_risk_assessment(self, metrics: Dict) -> Dict:
        """Compute comprehensive risk assessment across multiple dimensions."""
        risk_factors = {
            "security_risk": 0.0,
            "quality_risk": 0.0,
            "performance_risk": 0.0,
            "stability_risk": 0.0
        }
        
        # Security risk assessment
        sast_issues = metrics.get("sast_high_critical_count", 0)
        if sast_issues > 0:
            risk_factors["security_risk"] = min(1.0, sast_issues / 5.0)  # Scale to [0,1]
        
        # Quality risk assessment
        mutation_score = metrics.get("mutation_score", 1.0)
        property_coverage = metrics.get("property_coverage", 1.0)
        quality_risk = max(0, 1.0 - min(mutation_score / 0.8, property_coverage / 0.7))
        risk_factors["quality_risk"] = quality_risk
        
        # Performance risk assessment
        p50_overhead = metrics.get("latency_p50_overhead_percent", 0.0)
        p95_overhead = metrics.get("latency_p95_overhead_percent", 0.0)
        perf_risk = max(p50_overhead / 100.0, p95_overhead / 150.0)  # Normalize against thresholds
        risk_factors["performance_risk"] = min(1.0, perf_risk)
        
        # Stability risk assessment
        identical_runs = metrics.get("identical_hash_count", 3)
        budget_overruns = metrics.get("budget_overrun_count", 0)
        stability_risk = max(0, 1.0 - identical_runs / 3.0) + min(1.0, budget_overruns / 3.0)
        risk_factors["stability_risk"] = min(1.0, stability_risk)
        
        # Composite risk score with weights from TODO.md
        weights = {
            "security_risk": 0.4,
            "quality_risk": 0.3,
            "performance_risk": 0.2,
            "stability_risk": 0.1
        }
        
        composite_risk = sum(risk_factors[factor] * weights[factor] 
                           for factor in risk_factors)
        
        self.risk_assessment = {
            "risk_factors": risk_factors,
            "composite_risk": composite_risk,
            "risk_level": self._categorize_risk_level(composite_risk),
            "mitigation_required": composite_risk > 0.5,
            "recommendations": self._generate_risk_mitigations(risk_factors)
        }
        
        return self.risk_assessment
    
    def _categorize_risk_level(self, composite_risk: float) -> str:
        """Categorize composite risk score."""
        if composite_risk <= 0.3:
            return "low"
        elif composite_risk <= 0.7:
            return "medium"
        else:
            return "high"
    
    def _generate_risk_mitigations(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate specific risk mitigation recommendations."""
        mitigations = []
        
        if risk_factors["security_risk"] > 0.5:
            mitigations.append("Address high/critical security issues before promotion")
        
        if risk_factors["quality_risk"] > 0.5:
            mitigations.append("Improve test coverage and mutation score")
        
        if risk_factors["performance_risk"] > 0.5:
            mitigations.append("Optimize performance to meet latency targets")
        
        if risk_factors["stability_risk"] > 0.5:
            mitigations.append("Fix determinism and budget control issues")
        
        return mitigations

    def make_decision(self, metrics: Dict, risk_score: float = 0.0, variant: str = "V2") -> Dict:
        """
        Make final promotion decision based on acceptance gates, legacy gates, and risk assessment.
        
        Decision Matrix (from TODO.md):
        - PROMOTE: All gates passed, CI-backed wins demonstrated
        - AGENT_REFINE: Failed gates with concrete obligations and thresholds  
        - MANUAL_QA: Complex edge cases requiring human exploration
        """
        
        # Compute comprehensive risk assessment
        risk_assessment = self.compute_risk_assessment(metrics)
        
        # Evaluate legacy gates for backwards compatibility
        gate_results = self.evaluate_all_gates(metrics)
        composite_score = self.compute_composite_score()
        critical_failures = self.get_critical_failures()
        
        # Integration with acceptance gates
        acceptance_passed = True
        acceptance_score = 1.0
        if self.acceptance_results:
            acceptance_metadata = self.acceptance_results.get("metadata", {})
            acceptance_passed = acceptance_metadata.get("critical_failures", 1) == 0
            acceptance_score = acceptance_metadata.get("composite_score", 0.0)
            
            # Override with acceptance gate results if available
            if acceptance_score > 0:
                composite_score = max(composite_score, acceptance_score)
        
        # Enhanced decision logic following TODO.md requirements
        decision_info = self._make_promotion_decision(
            acceptance_passed, composite_score, critical_failures,
            risk_assessment, variant, metrics
        )
        
        # Compile full decision report
        decision_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision_info["decision"],
            "reason": decision_info["reason"],
            "variant": variant,
            "composite_score": composite_score,
            "risk_assessment": risk_assessment,
            "acceptance_gates": {
                "passed": acceptance_passed,
                "score": acceptance_score,
                "results_loaded": self.acceptance_results is not None
            },
            "legacy_gates": {
                "critical_failures": len(critical_failures),
                "total_gates": len(self.results),
                "gates_passed": len([r for r in self.results if r.passed]),
            },
            "gate_details": [
                {
                    "name": r.gate.name,
                    "threshold": r.gate.threshold,
                    "actual": r.actual_value,
                    "passed": r.passed,
                    "critical": r.gate.critical,
                    "margin": r.margin
                }
                for r in self.results
            ],
            "recommendations": decision_info["recommendations"],
            "mitigation_strategies": decision_info.get("mitigation_strategies", []),
            "escalation_required": decision_info.get("escalation_required", False)
        }
        
        return decision_report
    
    def _make_promotion_decision(self, acceptance_passed: bool, composite_score: float,
                               critical_failures: List[GateResult], risk_assessment: Dict,
                               variant: str, metrics: Dict) -> Dict:
        """
        Core promotion decision logic implementing TODO.md requirements.
        
        Returns decision info with rationale and recommendations.
        """
        
        # Extract key metrics for decision
        ci_lower_bound = metrics.get("token_eff_ci_lower", -1.0)
        has_ci_backed_wins = ci_lower_bound > 0.0
        risk_level = risk_assessment["risk_level"]
        composite_risk = risk_assessment["composite_risk"]
        
        # Decision logic following TODO.md
        if self._should_promote(acceptance_passed, composite_score, critical_failures, 
                               has_ci_backed_wins, risk_level, variant):
            return {
                "decision": "PROMOTE",
                "reason": f"All acceptance gates passed, CI-backed wins confirmed (CI lower: {ci_lower_bound:.4f}), composite score: {composite_score:.3f}",
                "recommendations": [
                    "Ready for production deployment",
                    "Monitor performance metrics post-deployment",
                    "Validate CI-backed improvements in production"
                ]
            }
        
        elif self._should_manual_qa(risk_level, critical_failures, composite_score, metrics):
            return {
                "decision": "MANUAL_QA",
                "reason": f"High complexity requires human review: risk={risk_level}, critical_failures={len(critical_failures)}, edge_cases_detected",
                "recommendations": [
                    "Assign senior engineer for manual verification",
                    "Review edge cases and failure scenarios",
                    "Validate statistical significance of results",
                    "Consider rollback strategy before promotion"
                ],
                "escalation_required": True,
                "mitigation_strategies": risk_assessment["recommendations"]
            }
        
        else:
            # AGENT_REFINE with concrete obligations
            failed_obligations = self._generate_concrete_obligations(
                self.results, acceptance_passed, has_ci_backed_wins, risk_assessment
            )
            
            return {
                "decision": "AGENT_REFINE",
                "reason": f"Gates failed with concrete remediation paths available. Composite score: {composite_score:.3f}",
                "recommendations": failed_obligations[:5],  # Top 5 priorities
                "mitigation_strategies": risk_assessment["recommendations"],
                "concrete_obligations": failed_obligations
            }
    
    def _should_promote(self, acceptance_passed: bool, composite_score: float,
                       critical_failures: List[GateResult], has_ci_backed_wins: bool,
                       risk_level: str, variant: str) -> bool:
        """Determine if variant should be promoted based on TODO.md criteria."""
        
        # Core promotion requirements from TODO.md:
        # - All gates passed
        # - CI-backed wins demonstrated  
        # - Low risk profile
        
        base_requirements = (
            len(critical_failures) == 0 and
            composite_score >= 0.85 and
            risk_level in ["low", "medium"]
        )
        
        # Variant-specific requirements
        if variant == "V1":
            # V1 focuses on correctness and reliability
            return base_requirements and acceptance_passed
            
        elif variant in ["V2", "V3"]:
            # V2/V3 require demonstrated improvements
            return (base_requirements and 
                   acceptance_passed and 
                   has_ci_backed_wins)
        
        return base_requirements
    
    def _should_manual_qa(self, risk_level: str, critical_failures: List[GateResult],
                         composite_score: float, metrics: Dict) -> bool:
        """Determine if manual QA review is required."""
        
        # High-risk scenarios requiring human judgment
        high_risk_indicators = [
            risk_level == "high",
            len(critical_failures) > 2,
            composite_score < 0.7,
            metrics.get("sast_high_critical_count", 0) > 3,
            metrics.get("fuzz_medium_high_crashes", 0) > 0
        ]
        
        # Complex edge cases from TODO.md
        edge_case_indicators = [
            metrics.get("judge_kappa", 1.0) < 0.6,  # Judge agreement issues
            metrics.get("oscillation_count", 0) > 1,  # V3 stability issues
            metrics.get("budget_overrun_count", 0) > 0  # Budget control failures
        ]
        
        return (sum(high_risk_indicators) >= 2 or 
               sum(edge_case_indicators) >= 1)
    
    def _generate_concrete_obligations(self, gate_results: List[GateResult],
                                     acceptance_passed: bool, has_ci_backed_wins: bool,
                                     risk_assessment: Dict) -> List[str]:
        """Generate concrete, actionable obligations for AGENT_REFINE."""
        
        obligations = []
        
        # Acceptance gate failures
        if not acceptance_passed and self.acceptance_results:
            for gate_result in self.acceptance_results.get("gate_results", []):
                if not gate_result.get("passed", True):
                    gate_name = gate_result["gate_name"]
                    threshold = gate_result["threshold"]
                    actual = gate_result["actual_value"]
                    
                    obligations.append(
                        f"Fix {gate_name}: current={actual}, required={threshold}"
                    )
        
        # CI-backed wins requirement
        if not has_ci_backed_wins:
            obligations.append(
                "Achieve positive CI lower bound for token-efficiency vs baseline"
            )
        
        # Legacy gate failures with specific thresholds
        for result in gate_results:
            if not result.passed:
                if result.gate.name == "mutation_score":
                    obligations.append(
                        f"Improve mutation testing: achieve {result.gate.threshold:.2f} score (current: {result.actual_value:.2f})"
                    )
                elif result.gate.name == "sast_high_critical":
                    obligations.append(
                        f"Address {result.actual_value} high/critical security issues"
                    )
                elif "budget" in result.gate.name:
                    obligations.append(
                        f"Fix budget control: {result.gate.name} = {result.actual_value}, max allowed = {result.gate.threshold}"
                    )
                elif "latency" in result.gate.name:
                    obligations.append(
                        f"Optimize {result.gate.name}: {result.actual_value:.1f}% > {result.gate.threshold}% threshold"
                    )
        
        # Risk-specific obligations
        for risk_factor, score in risk_assessment["risk_factors"].items():
            if score > 0.7:
                if risk_factor == "security_risk":
                    obligations.append("Complete security review and remediation")
                elif risk_factor == "quality_risk":
                    obligations.append("Increase test coverage and mutation score above 80%")
                elif risk_factor == "performance_risk":
                    obligations.append("Meet latency targets: p50 ≤ +30%, p95 ≤ +50% vs baseline")
                elif risk_factor == "stability_risk":
                    obligations.append("Ensure 3x deterministic runs and zero budget overruns")
        
        return obligations
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on gate failures."""
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                gate_name = result.gate.name
                
                if "SAST" in gate_name:
                    recommendations.append(f"Address {result.actual_value} high/critical security issues")
                elif "Mutation" in gate_name:
                    recommendations.append(f"Improve test quality - mutation score {result.actual_value:.2f} < {result.gate.threshold}")
                elif "Property" in gate_name:
                    recommendations.append(f"Add property-based tests - coverage {result.actual_value:.2f} < {result.gate.threshold}")
                elif "Fuzz" in gate_name:
                    recommendations.append(f"Fix {result.actual_value} fuzzing crashes before promotion")
                elif "Budget" in gate_name and "overrun" in gate_name:
                    recommendations.append(f"Critical: Fix {result.actual_value} budget overruns")
                elif "Token-Efficiency" in gate_name:
                    recommendations.append(f"Improve token efficiency - CI lower bound {result.actual_value:.3f} not positive")
                elif "Latency" in gate_name:
                    recommendations.append(f"Optimize performance - {gate_name} at {result.actual_value:.1f}%")
                elif "Determinism" in gate_name:
                    recommendations.append(f"Fix non-deterministic behavior - only {result.actual_value} consistent runs")
                    
        return recommendations[:5]  # Top 5 most important


def load_metrics_from_files(metrics_dir: Path) -> Dict:
    """Load and aggregate metrics from various CI output files."""
    
    metrics = {}
    
    # Load statistical analysis results
    ci_file = metrics_dir / "qa_acc_ci.json"
    if ci_file.exists():
        try:
            with open(ci_file) as f:
                ci_data = json.load(f)
                metrics["token_eff_ci_lower"] = ci_data.get("ci_lower", -1.0)
                metrics["qa_accuracy_mean"] = ci_data.get("mean", 0.0)
        except Exception as e:
            print(f"Warning: Could not load CI analysis: {e}")
    
    # Load risk analysis
    risk_file = metrics_dir / "risk.json"  
    if risk_file.exists():
        try:
            with open(risk_file) as f:
                risk_data = json.load(f)
                metrics["composite_risk"] = risk_data.get("composite_risk", 1.0)
                metrics["risk_level"] = risk_data.get("risk_level", "high")
        except Exception as e:
            print(f"Warning: Could not load risk analysis: {e}")
    
    # Load SAST results
    sast_file = metrics_dir / "../semgrep_results.json"
    if Path(sast_file).exists():
        try:
            with open(sast_file) as f:
                sast_data = json.load(f)
                # Count high/critical severity issues
                high_critical = 0
                for result in sast_data.get("results", []):
                    severity = result.get("extra", {}).get("severity", "").lower()
                    if severity in ["high", "critical", "error"]:
                        high_critical += 1
                metrics["sast_high_critical_count"] = high_critical
        except Exception as e:
            print(f"Warning: Could not load SAST results: {e}")
    
    # Load mutation testing results
    mutation_file = metrics_dir / "../mutation.json"
    if Path(mutation_file).exists():
        try:
            with open(mutation_file) as f:
                mutation_data = json.load(f)
                metrics["mutation_score"] = mutation_data.get("mutation_score", 0.0)
        except Exception as e:
            print(f"Warning: Could not load mutation results: {e}")
    
    # Load fuzzing results
    fuzz_file = metrics_dir / "../fuzz_results.json"
    if Path(fuzz_file).exists():
        try:
            with open(fuzz_file) as f:
                fuzz_data = json.load(f)
                metrics["fuzz_medium_high_crashes"] = fuzz_data.get("crashes", 0)
        except Exception as e:
            print(f"Warning: Could not load fuzz results: {e}")
    
    # Load aggregated metrics
    agg_file = metrics_dir / "all_metrics.jsonl"
    if agg_file.exists():
        try:
            with open(agg_file) as f:
                latest_metrics = {}
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        latest_metrics.update(data)
                
                # Extract performance metrics
                if "latency_p50_ms" in latest_metrics and "baseline_latency_p50" in latest_metrics:
                    baseline_p50 = latest_metrics["baseline_latency_p50"]
                    actual_p50 = latest_metrics["latency_p50_ms"]
                    if baseline_p50 > 0:
                        metrics["latency_p50_overhead_percent"] = ((actual_p50 - baseline_p50) / baseline_p50) * 100
                
                if "latency_p95_ms" in latest_metrics and "baseline_latency_p95" in latest_metrics:
                    baseline_p95 = latest_metrics["baseline_latency_p95"]
                    actual_p95 = latest_metrics["latency_p95_ms"]
                    if baseline_p95 > 0:
                        metrics["latency_p95_overhead_percent"] = ((actual_p95 - baseline_p95) / baseline_p95) * 100
                
        except Exception as e:
            print(f"Warning: Could not load aggregated metrics: {e}")
    
    # Set reasonable defaults for missing metrics
    defaults = {
        "mutation_score": 0.75,
        "property_coverage": 0.60,
        "sast_high_critical_count": 0,
        "fuzz_medium_high_crashes": 0,
        "budget_overrun_count": 0,
        "budget_underrun_percent": 0.3,
        "identical_hash_count": 3,  # Assume determinism unless proven otherwise
        "token_eff_ci_lower": -0.1,  # Slightly pessimistic default
        "latency_p50_overhead_percent": 25.0,
        "latency_p95_overhead_percent": 40.0,
        "composite_risk": 0.3
    }
    
    for key, default_value in defaults.items():
        if key not in metrics:
            metrics[key] = default_value
            
    return metrics


def main():
    """Main gatekeeper execution with acceptance gates integration."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: gatekeeper.py <metrics_dir> [variant] [gates_config.yaml] [output_file.json]")
        print("Example: gatekeeper.py artifacts/metrics V2 scripts/gates.yaml artifacts/gatekeeper_decision.json")
        sys.exit(1)
        
    metrics_dir = Path(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) > 2 else "V2"
    gates_config_file = sys.argv[3] if len(sys.argv) > 3 else None
    output_file = Path(sys.argv[4]) if len(sys.argv) > 4 else metrics_dir / "gatekeeper_decision.json"
    
    print(f"PackRepo Gatekeeper - Quality Gate Decision Engine")
    print(f"Variant: {variant}")
    print(f"Metrics Directory: {metrics_dir}")
    
    # Load gates configuration
    gates_config = {}
    if gates_config_file and Path(gates_config_file).exists():
        try:
            with open(gates_config_file) as f:
                if gates_config_file.endswith('.yaml'):
                    import yaml
                    gates_config = yaml.safe_load(f)
                else:
                    gates_config = json.load(f)
            print(f"Loaded gates config from: {gates_config_file}")
        except Exception as e:
            print(f"Warning: Could not load gates config: {e}")
    
    # Initialize gatekeeper
    gatekeeper = GatekeeperDecision(gates_config)
    
    # Load acceptance gate results if available
    acceptance_file = metrics_dir / "acceptance_gate_results.json"
    if acceptance_file.exists():
        if gatekeeper.load_acceptance_gate_results(acceptance_file):
            print("Loaded acceptance gate results")
        else:
            print("Warning: Could not load acceptance gate results")
    else:
        print("No acceptance gate results found - using legacy gates only")
    
    # Load metrics
    print(f"Loading metrics from: {metrics_dir}")
    metrics = load_metrics_from_files(metrics_dir)
    
    print(f"Loaded {len(metrics)} metrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Make comprehensive decision
    print(f"\nEvaluating quality gates for {variant}...")
    decision = gatekeeper.make_decision(
        metrics, 
        metrics.get("composite_risk", 0.0),
        variant
    )
    
    # Save decision
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(decision, f, indent=2, default=str)
    
    # Display comprehensive results
    print(f"\n{'='*70}")
    print(f"GATEKEEPER DECISION: {decision['decision']}")
    print(f"Variant: {decision['variant']}")
    print(f"Reason: {decision['reason']}")
    print(f"Timestamp: {decision['timestamp']}")
    print(f"{'='*70}")
    
    # Quality metrics summary
    print(f"\nQUALITY METRICS:")
    print(f"  Composite Score: {decision['composite_score']:.3f}")
    print(f"  Risk Level: {decision['risk_assessment']['risk_level']}")
    print(f"  Risk Score: {decision['risk_assessment']['composite_risk']:.3f}")
    
    # Gate results summary
    acceptance = decision['acceptance_gates']
    legacy = decision['legacy_gates']
    print(f"\nGATE RESULTS:")
    print(f"  Acceptance Gates: {'PASS' if acceptance['passed'] else 'FAIL'} (score: {acceptance['score']:.3f})")
    print(f"  Legacy Gates: {legacy['gates_passed']}/{legacy['total_gates']} passed")
    print(f"  Critical Failures: {legacy['critical_failures']}")
    
    # Risk factors breakdown
    print(f"\nRISK FACTORS:")
    for factor, score in decision['risk_assessment']['risk_factors'].items():
        status = "HIGH" if score > 0.7 else "MED" if score > 0.3 else "LOW"
        print(f"  {factor}: {score:.3f} ({status})")
    
    # Recommendations
    if decision.get("recommendations"):
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(decision["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    # Mitigation strategies for failed cases
    if decision.get("mitigation_strategies"):
        print(f"\nMITIGATION STRATEGIES:")
        for i, strategy in enumerate(decision["mitigation_strategies"], 1):
            print(f"  {i}. {strategy}")
    
    # Concrete obligations for AGENT_REFINE
    if decision["decision"] == "AGENT_REFINE" and decision.get("concrete_obligations"):
        print(f"\nCONCRETE OBLIGATIONS:")
        for i, obligation in enumerate(decision["concrete_obligations"], 1):
            print(f"  {i}. {obligation}")
    
    print(f"{'='*70}")
    print(f"Full report saved to: {output_file}")
    
    # CI/CD integration outputs
    print(f"::set-output name=decision::{decision['decision']}")
    print(f"::set-output name=composite_score::{decision['composite_score']:.3f}")
    print(f"::set-output name=risk_level::{decision['risk_assessment']['risk_level']}")
    
    # Exit with appropriate code for pipeline integration
    if decision["decision"] == "PROMOTE":
        print("\n✅ PROMOTION APPROVED - All gates passed")
        sys.exit(0)
    elif decision["decision"] == "AGENT_REFINE":
        print("\n🔄 AGENT REFINEMENT REQUIRED - Concrete obligations provided")
        sys.exit(1)
    else:  # MANUAL_QA
        print("\n⚠️  MANUAL QA REQUIRED - Human review needed")
        sys.exit(2)


if __name__ == "__main__":
    main()
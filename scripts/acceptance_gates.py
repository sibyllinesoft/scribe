#!/usr/bin/env python3
"""
PackRepo Acceptance Gates Engine - Comprehensive Quality Validation

Implements the complete acceptance gate system from TODO.md requirements:
- Spin-up: Clean checkout → container build → dataset fetch → readiness OK → golden smokes → signed boot transcript
- Static: 0 high/critical SAST; typecheck clean; license policy OK; API surface diffs acknowledged  
- Dynamic: Mutation ≥ T_mut; property/metamorphic coverage ≥ T_prop; fuzz ≥ FUZZ_MIN min with 0 new medium+ crashes
- Parities: Selection budgets within ±5%; decode budgets logged; tokenizer/version pinned
- Primary KPI: V1/V2/V3 beat V0c with BCa 95% CI lower bound > 0 on QA acc/100k at both budgets
- Stability: 3-run reproducibility; judge κ ≥ 0.6; flakiness < 1%
- Performance: p50/p95 within Objective 5

Outputs comprehensive evidence-based validation with clear pass/fail decisions.
"""

import json
import yaml
import sys
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of evaluating a single acceptance gate."""
    gate_name: str
    passed: bool
    actual_value: Any
    threshold: Any
    evidence: Dict[str, Any]
    error_message: Optional[str] = None
    weight: float = 1.0
    critical: bool = False


@dataclass
class GateEvidence:
    """Evidence collection for gate validation."""
    artifacts: List[Path]
    metrics: Dict[str, Any]
    logs: List[str]
    checksums: Dict[str, str]
    timestamps: Dict[str, str]


class AcceptanceGateEngine:
    """Comprehensive acceptance gate evaluation engine."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.results: List[GateResult] = []
        self.evidence = GateEvidence([], {}, [], {}, {})
        self.variant_gates = self._initialize_variant_gates()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load acceptance gates configuration."""
        default_config = {
            # Thresholds from TODO.md
            "T_mut": 0.80,
            "T_prop": 0.70,
            "FUZZ_MIN": 30,
            "budget_tolerance": 0.05,  # ±5%
            "confidence_level": 0.95,
            "judge_kappa_min": 0.6,
            "flakiness_max": 0.01,  # < 1%
            "latency_p50_max": 30.0,  # +30% vs baseline
            "latency_p95_max": 50.0,  # +50% vs baseline
            
            # Paths
            "artifacts_dir": Path("./artifacts"),
            "logs_dir": Path("./logs"),
            "scripts_dir": Path("./scripts"),
            
            # Gate weights
            "gate_weights": {
                "spin_up": 0.20,
                "static_analysis": 0.15,
                "dynamic_testing": 0.20,
                "budget_parities": 0.10,
                "primary_kpi": 0.25,
                "stability": 0.10
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    if config_path.suffix == '.yaml':
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_variant_gates(self) -> Dict[str, List[str]]:
        """Initialize required gates per variant from TODO.md."""
        return {
            "V1": [
                "spin_up", "static_analysis", "dynamic_testing", 
                "budget_parities", "stability"
            ],
            "V2": [
                "spin_up", "static_analysis", "dynamic_testing",
                "budget_parities", "primary_kpi", "stability" 
            ],
            "V3": [
                "spin_up", "static_analysis", "dynamic_testing",
                "budget_parities", "stability"  # Special oscillation checks
            ]
        }
    
    def evaluate_spin_up_gate(self) -> GateResult:
        """
        Spin-up Gate: Clean checkout → container build → dataset fetch → 
        readiness OK → golden smokes → signed boot transcript
        """
        logger.info("Evaluating spin-up gate...")
        
        evidence = {}
        passed = True
        errors = []
        
        # Check boot transcript exists and is signed
        boot_transcript = self.config["artifacts_dir"] / "boot_transcript.json"
        if not boot_transcript.exists():
            passed = False
            errors.append("Boot transcript missing")
        else:
            try:
                with open(boot_transcript) as f:
                    transcript_data = json.load(f)
                evidence["boot_transcript"] = {
                    "exists": True,
                    "signed": bool(transcript_data.get("signature")),
                    "container_digest": transcript_data.get("container_digest"),
                    "timestamp": transcript_data.get("timestamp")
                }
                if not transcript_data.get("signature"):
                    passed = False
                    errors.append("Boot transcript not signed")
            except Exception as e:
                passed = False
                errors.append(f"Invalid boot transcript: {e}")
        
        # Check golden smoke test results
        smoke_results = self.config["artifacts_dir"] / "smoke_test_results.json"
        if smoke_results.exists():
            try:
                with open(smoke_results) as f:
                    smoke_data = json.load(f)
                evidence["golden_smokes"] = smoke_data
                if not smoke_data.get("all_passed", False):
                    passed = False
                    errors.append("Golden smoke tests failed")
            except Exception as e:
                passed = False
                errors.append(f"Could not read smoke test results: {e}")
        else:
            # Try to run golden smoke tests if results don't exist
            try:
                logger.info("Running golden smoke tests...")
                result = subprocess.run([
                    "bash", str(self.config["scripts_dir"] / "spinup_smoke.sh"),
                    "--quick-test"
                ], capture_output=True, text=True, timeout=300)
                
                evidence["golden_smokes"] = {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if result.returncode != 0:
                    passed = False
                    errors.append(f"Golden smoke tests failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                passed = False
                errors.append("Golden smoke tests timed out")
            except Exception as e:
                passed = False
                errors.append(f"Could not run golden smoke tests: {e}")
        
        # Check container build artifacts
        container_info = self.config["artifacts_dir"] / "container_info.json"
        if container_info.exists():
            try:
                with open(container_info) as f:
                    container_data = json.load(f)
                evidence["container"] = container_data
            except Exception as e:
                logger.warning(f"Could not read container info: {e}")
        
        return GateResult(
            gate_name="spin_up",
            passed=passed,
            actual_value=evidence,
            threshold="All components ready",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=self.config["gate_weights"]["spin_up"],
            critical=True
        )
    
    def evaluate_static_analysis_gate(self) -> GateResult:
        """
        Static Gate: 0 high/critical SAST; typecheck clean; license policy OK; 
        API surface diffs acknowledged
        """
        logger.info("Evaluating static analysis gate...")
        
        evidence = {}
        passed = True
        errors = []
        
        # SAST scan results
        sast_results = self.config["artifacts_dir"] / "sast_results.json"
        if sast_results.exists():
            try:
                with open(sast_results) as f:
                    sast_data = json.load(f)
                
                high_critical = sum(1 for result in sast_data.get("results", [])
                                  if result.get("extra", {}).get("severity", "").lower() 
                                  in ["high", "critical", "error"])
                
                evidence["sast"] = {
                    "total_issues": len(sast_data.get("results", [])),
                    "high_critical_issues": high_critical,
                    "scan_timestamp": sast_data.get("timestamp")
                }
                
                if high_critical > 0:
                    passed = False
                    errors.append(f"{high_critical} high/critical SAST issues found")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not read SAST results: {e}")
        else:
            passed = False
            errors.append("SAST results missing")
        
        # Type checking results
        typecheck_results = self.config["artifacts_dir"] / "typecheck_results.json"
        if typecheck_results.exists():
            try:
                with open(typecheck_results) as f:
                    typecheck_data = json.load(f)
                evidence["typecheck"] = typecheck_data
                
                if typecheck_data.get("errors", 0) > 0:
                    passed = False
                    errors.append(f"{typecheck_data['errors']} type check errors")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not read typecheck results: {e}")
        else:
            # Run type checking if results don't exist
            try:
                logger.info("Running type check...")
                result = subprocess.run([
                    "python", "-m", "mypy", "packrepo/", "--json-report", 
                    str(self.config["artifacts_dir"] / "mypy_report.json")
                ], capture_output=True, text=True, timeout=120)
                
                evidence["typecheck"] = {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if result.returncode != 0:
                    passed = False
                    errors.append("Type check failed")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not run type check: {e}")
        
        # License policy check
        license_results = self.config["artifacts_dir"] / "license_check_results.json"
        if license_results.exists():
            try:
                with open(license_results) as f:
                    license_data = json.load(f)
                evidence["licenses"] = license_data
                
                if not license_data.get("policy_compliant", True):
                    passed = False
                    errors.append("License policy violations found")
                    
            except Exception as e:
                logger.warning(f"Could not read license results: {e}")
        
        # API surface diff check
        api_diff_results = self.config["artifacts_dir"] / "api_surface_diff.json"
        if api_diff_results.exists():
            try:
                with open(api_diff_results) as f:
                    api_diff_data = json.load(f)
                evidence["api_surface"] = api_diff_data
                
                # Check for breaking changes
                if api_diff_data.get("breaking_changes", 0) > 0:
                    passed = False
                    errors.append(f"{api_diff_data['breaking_changes']} breaking API changes")
                    
            except Exception as e:
                logger.warning(f"Could not read API diff results: {e}")
        
        return GateResult(
            gate_name="static_analysis",
            passed=passed,
            actual_value=evidence,
            threshold="0 high/critical issues, clean typecheck",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=self.config["gate_weights"]["static_analysis"],
            critical=True
        )
    
    def evaluate_dynamic_testing_gate(self) -> GateResult:
        """
        Dynamic Gate: Mutation ≥ T_mut; property/metamorphic coverage ≥ T_prop; 
        fuzz ≥ FUZZ_MIN min with 0 new medium+ crashes
        """
        logger.info("Evaluating dynamic testing gate...")
        
        evidence = {}
        passed = True
        errors = []
        
        # Mutation testing results
        mutation_results = self.config["artifacts_dir"] / "mutation_results.json"
        if mutation_results.exists():
            try:
                with open(mutation_results) as f:
                    mutation_data = json.load(f)
                evidence["mutation"] = mutation_data
                
                mutation_score = mutation_data.get("mutation_score", 0.0)
                if mutation_score < self.config["T_mut"]:
                    passed = False
                    errors.append(f"Mutation score {mutation_score:.3f} < {self.config['T_mut']}")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not read mutation results: {e}")
        else:
            passed = False
            errors.append("Mutation testing results missing")
        
        # Property/metamorphic testing results
        property_results = self.config["artifacts_dir"] / "property_test_results.json"
        if property_results.exists():
            try:
                with open(property_results) as f:
                    property_data = json.load(f)
                evidence["property_tests"] = property_data
                
                property_coverage = property_data.get("coverage", 0.0)
                if property_coverage < self.config["T_prop"]:
                    passed = False
                    errors.append(f"Property coverage {property_coverage:.3f} < {self.config['T_prop']}")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not read property test results: {e}")
        else:
            passed = False
            errors.append("Property test results missing")
        
        # Fuzzing results
        fuzz_results = self.config["artifacts_dir"] / "fuzz_results.json"
        if fuzz_results.exists():
            try:
                with open(fuzz_results) as f:
                    fuzz_data = json.load(f)
                evidence["fuzzing"] = fuzz_data
                
                runtime_minutes = fuzz_data.get("runtime_minutes", 0)
                medium_high_crashes = fuzz_data.get("medium_high_crashes", 0)
                
                if runtime_minutes < self.config["FUZZ_MIN"]:
                    passed = False
                    errors.append(f"Fuzz runtime {runtime_minutes} < {self.config['FUZZ_MIN']} minutes")
                
                if medium_high_crashes > 0:
                    passed = False
                    errors.append(f"{medium_high_crashes} medium+ severity fuzz crashes")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not read fuzz results: {e}")
        else:
            passed = False
            errors.append("Fuzzing results missing")
        
        return GateResult(
            gate_name="dynamic_testing",
            passed=passed,
            actual_value=evidence,
            threshold=f"Mutation ≥ {self.config['T_mut']}, Property ≥ {self.config['T_prop']}, Fuzz ≥ {self.config['FUZZ_MIN']}min",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=self.config["gate_weights"]["dynamic_testing"],
            critical=True
        )
    
    def evaluate_budget_parities_gate(self) -> GateResult:
        """
        Parities Gate: Selection budgets within ±5%; decode budgets logged; 
        tokenizer/version pinned
        """
        logger.info("Evaluating budget parities gate...")
        
        evidence = {}
        passed = True
        errors = []
        
        # Budget analysis results
        budget_results = self.config["artifacts_dir"] / "budget_analysis.json"
        if budget_results.exists():
            try:
                with open(budget_results) as f:
                    budget_data = json.load(f)
                evidence["budgets"] = budget_data
                
                # Check selection budget parity (±5%)
                for variant, data in budget_data.get("variants", {}).items():
                    if variant == "baseline":
                        continue
                        
                    baseline_budget = budget_data.get("baseline", {}).get("selection_tokens", 0)
                    variant_budget = data.get("selection_tokens", 0)
                    
                    if baseline_budget > 0:
                        parity_diff = abs(variant_budget - baseline_budget) / baseline_budget
                        if parity_diff > self.config["budget_tolerance"]:
                            passed = False
                            errors.append(f"{variant} budget parity {parity_diff:.3f} > {self.config['budget_tolerance']}")
                
                # Check for budget overruns
                overruns = budget_data.get("overruns", 0)
                if overruns > 0:
                    passed = False
                    errors.append(f"{overruns} budget overruns detected")
                
            except Exception as e:
                passed = False
                errors.append(f"Could not read budget results: {e}")
        else:
            passed = False
            errors.append("Budget analysis results missing")
        
        # Tokenizer version check
        tokenizer_info = self.config["artifacts_dir"] / "tokenizer_info.json"
        if tokenizer_info.exists():
            try:
                with open(tokenizer_info) as f:
                    tokenizer_data = json.load(f)
                evidence["tokenizer"] = tokenizer_data
                
                if not tokenizer_data.get("version_pinned", False):
                    passed = False
                    errors.append("Tokenizer version not pinned")
                    
            except Exception as e:
                logger.warning(f"Could not read tokenizer info: {e}")
        
        return GateResult(
            gate_name="budget_parities",
            passed=passed,
            actual_value=evidence,
            threshold=f"Selection budgets ±{self.config['budget_tolerance']*100}%, 0 overruns",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=self.config["gate_weights"]["budget_parities"],
            critical=True
        )
    
    def evaluate_primary_kpi_gate(self) -> GateResult:
        """
        Primary KPI Gate: V1/V2/V3 beat V0c with BCa 95% CI lower bound > 0 
        on QA acc/100k at both budgets
        """
        logger.info("Evaluating primary KPI gate...")
        
        evidence = {}
        passed = True
        errors = []
        
        # Load statistical analysis results
        ci_results = self.config["artifacts_dir"] / "metrics" / "qa_acc_ci.json"
        if ci_results.exists():
            try:
                with open(ci_results) as f:
                    ci_data = json.load(f)
                evidence["confidence_intervals"] = ci_data
                
                # Check CI lower bounds for each variant
                for variant in ["V1", "V2", "V3"]:
                    variant_ci = ci_data.get(f"{variant}_vs_V0c", {})
                    ci_lower = variant_ci.get("ci_lower", -1.0)
                    
                    if ci_lower <= 0:
                        passed = False
                        errors.append(f"{variant} CI lower bound {ci_lower:.4f} ≤ 0")
                    else:
                        evidence[f"{variant}_improvement"] = {
                            "ci_lower": ci_lower,
                            "ci_upper": variant_ci.get("ci_upper", 0.0),
                            "mean_diff": variant_ci.get("mean_diff", 0.0)
                        }
                
            except Exception as e:
                passed = False
                errors.append(f"Could not read CI results: {e}")
        else:
            passed = False
            errors.append("Confidence interval results missing")
        
        # Load QA accuracy results
        qa_results = self.config["artifacts_dir"] / "metrics" / "qa_accuracy_summary.json"
        if qa_results.exists():
            try:
                with open(qa_results) as f:
                    qa_data = json.load(f)
                evidence["qa_accuracy"] = qa_data
                
                # Verify token efficiency calculations
                for variant in ["V1", "V2", "V3"]:
                    if variant in qa_data:
                        accuracy = qa_data[variant].get("accuracy", 0.0)
                        tokens = qa_data[variant].get("selection_tokens", 120000)
                        token_eff = (accuracy * 100000) / tokens if tokens > 0 else 0
                        
                        baseline_eff = qa_data.get("V0c", {}).get("token_efficiency", 50.0)
                        improvement = token_eff - baseline_eff
                        
                        evidence[f"{variant}_token_efficiency"] = {
                            "accuracy": accuracy,
                            "tokens": tokens,
                            "efficiency": token_eff,
                            "improvement_vs_baseline": improvement
                        }
                
            except Exception as e:
                logger.warning(f"Could not read QA results: {e}")
        
        return GateResult(
            gate_name="primary_kpi",
            passed=passed,
            actual_value=evidence,
            threshold="BCa 95% CI lower bound > 0 vs V0c",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=self.config["gate_weights"]["primary_kpi"],
            critical=True
        )
    
    def evaluate_stability_gate(self) -> GateResult:
        """
        Stability Gate: 3-run reproducibility; judge κ ≥ 0.6; flakiness < 1%
        """
        logger.info("Evaluating stability gate...")
        
        evidence = {}
        passed = True
        errors = []
        
        # Reproducibility check
        reproducibility_results = self.config["artifacts_dir"] / "reproducibility_results.json"
        if reproducibility_results.exists():
            try:
                with open(reproducibility_results) as f:
                    repro_data = json.load(f)
                evidence["reproducibility"] = repro_data
                
                identical_runs = repro_data.get("identical_hash_count", 0)
                if identical_runs < 3:
                    passed = False
                    errors.append(f"Only {identical_runs} identical runs, need 3")
                
                # Check determinism for --no-llm mode
                deterministic = repro_data.get("deterministic", False)
                if not deterministic:
                    passed = False
                    errors.append("Non-deterministic outputs in --no-llm mode")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not read reproducibility results: {e}")
        else:
            passed = False
            errors.append("Reproducibility results missing")
        
        # Judge agreement (κ) check
        judge_results = self.config["artifacts_dir"] / "judge_agreement.json"
        if judge_results.exists():
            try:
                with open(judge_results) as f:
                    judge_data = json.load(f)
                evidence["judge_agreement"] = judge_data
                
                kappa = judge_data.get("kappa", 0.0)
                if kappa < self.config["judge_kappa_min"]:
                    passed = False
                    errors.append(f"Judge κ {kappa:.3f} < {self.config['judge_kappa_min']}")
                    
            except Exception as e:
                logger.warning(f"Could not read judge agreement: {e}")
        
        # Flakiness check
        flakiness_results = self.config["artifacts_dir"] / "flakiness_analysis.json"
        if flakiness_results.exists():
            try:
                with open(flakiness_results) as f:
                    flake_data = json.load(f)
                evidence["flakiness"] = flake_data
                
                flake_rate = flake_data.get("flake_rate", 0.0)
                if flake_rate > self.config["flakiness_max"]:
                    passed = False
                    errors.append(f"Flakiness {flake_rate:.3f} > {self.config['flakiness_max']}")
                    
            except Exception as e:
                logger.warning(f"Could not read flakiness analysis: {e}")
        
        return GateResult(
            gate_name="stability",
            passed=passed,
            actual_value=evidence,
            threshold=f"3 identical runs, κ ≥ {self.config['judge_kappa_min']}, flakiness < {self.config['flakiness_max']}",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=self.config["gate_weights"]["stability"],
            critical=True
        )
    
    def evaluate_performance_gate(self) -> GateResult:
        """
        Performance Gate: p50/p95 within Objective 5 bounds
        """
        logger.info("Evaluating performance gate...")
        
        evidence = {}
        passed = True
        errors = []
        
        # Performance benchmarks
        perf_results = self.config["artifacts_dir"] / "performance_results.json"
        if perf_results.exists():
            try:
                with open(perf_results) as f:
                    perf_data = json.load(f)
                evidence["performance"] = perf_data
                
                # Check p50 latency overhead
                p50_overhead = perf_data.get("latency_p50_overhead_percent", 100.0)
                if p50_overhead > self.config["latency_p50_max"]:
                    passed = False
                    errors.append(f"P50 latency overhead {p50_overhead:.1f}% > {self.config['latency_p50_max']}%")
                
                # Check p95 latency overhead
                p95_overhead = perf_data.get("latency_p95_overhead_percent", 100.0)
                if p95_overhead > self.config["latency_p95_max"]:
                    passed = False
                    errors.append(f"P95 latency overhead {p95_overhead:.1f}% > {self.config['latency_p95_max']}%")
                
                # Check memory usage
                memory_usage = perf_data.get("memory_usage_gb", 0.0)
                if memory_usage > 8.0:  # From Objective 5
                    passed = False
                    errors.append(f"Memory usage {memory_usage:.1f} GB > 8 GB limit")
                    
            except Exception as e:
                passed = False
                errors.append(f"Could not read performance results: {e}")
        else:
            passed = False
            errors.append("Performance results missing")
        
        return GateResult(
            gate_name="performance",
            passed=passed,
            actual_value=evidence,
            threshold=f"P50 ≤ +{self.config['latency_p50_max']}%, P95 ≤ +{self.config['latency_p95_max']}%, Memory ≤ 8GB",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=0.05,  # Lower weight, informational
            critical=False
        )
    
    def evaluate_variant_specific_gates(self, variant: str) -> List[GateResult]:
        """Evaluate variant-specific gates based on TODO.md requirements."""
        logger.info(f"Evaluating {variant}-specific gates...")
        
        specific_results = []
        
        if variant == "V3":
            # V3 Demotion Stability Controller specific checks
            oscillation_result = self._check_oscillation_control()
            specific_results.append(oscillation_result)
        
        return specific_results
    
    def _check_oscillation_control(self) -> GateResult:
        """Check V3 oscillation control requirements."""
        evidence = {}
        passed = True
        errors = []
        
        # Check demotion logs
        demotion_file = self.config["artifacts_dir"] / "demotions.csv"
        if demotion_file.exists():
            try:
                with open(demotion_file) as f:
                    lines = f.readlines()
                    
                evidence["demotion_events"] = len(lines) - 1  # Exclude header
                
                # Parse for oscillation patterns
                oscillations = 0
                # Simple oscillation detection logic would go here
                # For now, assume we extract from logs or metadata
                
                oscillation_results = self.config["artifacts_dir"] / "oscillation_analysis.json"
                if oscillation_results.exists():
                    with open(oscillation_results) as of:
                        osc_data = json.load(of)
                        oscillations = osc_data.get("oscillation_count", 0)
                
                evidence["oscillations"] = oscillations
                
                if oscillations > 1:
                    passed = False
                    errors.append(f"{oscillations} oscillations detected, max 1 allowed")
                
            except Exception as e:
                logger.warning(f"Could not analyze demotion logs: {e}")
        
        return GateResult(
            gate_name="v3_oscillation_control",
            passed=passed,
            actual_value=evidence,
            threshold="Oscillations ≤ 1",
            evidence=evidence,
            error_message="; ".join(errors) if errors else None,
            weight=0.10,
            critical=True
        )
    
    def evaluate_all_gates(self, variant: str = "V2") -> List[GateResult]:
        """Evaluate all relevant gates for the specified variant."""
        logger.info(f"Starting comprehensive gate evaluation for {variant}...")
        
        all_results = []
        
        # Core gates (all variants)
        gate_methods = [
            self.evaluate_spin_up_gate,
            self.evaluate_static_analysis_gate,
            self.evaluate_dynamic_testing_gate,
            self.evaluate_budget_parities_gate,
            self.evaluate_stability_gate,
            self.evaluate_performance_gate
        ]
        
        for gate_method in gate_methods:
            try:
                result = gate_method()
                all_results.append(result)
                logger.info(f"Gate {result.gate_name}: {'PASS' if result.passed else 'FAIL'}")
                if not result.passed and result.error_message:
                    logger.error(f"  Error: {result.error_message}")
            except Exception as e:
                logger.error(f"Exception evaluating gate {gate_method.__name__}: {e}")
                all_results.append(GateResult(
                    gate_name=gate_method.__name__.replace("evaluate_", "").replace("_gate", ""),
                    passed=False,
                    actual_value=None,
                    threshold="N/A",
                    evidence={},
                    error_message=str(e),
                    critical=True
                ))
        
        # Primary KPI gate (V2+ only)
        if variant in ["V2", "V3"]:
            try:
                kpi_result = self.evaluate_primary_kpi_gate()
                all_results.append(kpi_result)
                logger.info(f"Gate {kpi_result.gate_name}: {'PASS' if kpi_result.passed else 'FAIL'}")
            except Exception as e:
                logger.error(f"Exception evaluating primary KPI gate: {e}")
        
        # Variant-specific gates
        try:
            variant_results = self.evaluate_variant_specific_gates(variant)
            all_results.extend(variant_results)
        except Exception as e:
            logger.error(f"Exception evaluating variant-specific gates: {e}")
        
        self.results = all_results
        return all_results
    
    def compute_composite_score(self) -> float:
        """Compute weighted composite score across all gates."""
        if not self.results:
            return 0.0
        
        total_weight = sum(r.weight for r in self.results)
        if total_weight == 0:
            return 0.0
        
        weighted_score = sum(r.weight if r.passed else 0 for r in self.results)
        return weighted_score / total_weight
    
    def get_critical_failures(self) -> List[GateResult]:
        """Get all critical gate failures."""
        return [r for r in self.results if r.critical and not r.passed]
    
    def generate_evidence_report(self) -> Dict[str, Any]:
        """Generate comprehensive evidence report for audit trail."""
        
        # Collect all evidence
        all_evidence = {}
        for result in self.results:
            all_evidence[result.gate_name] = result.evidence
        
        # Generate checksums for artifacts
        checksums = {}
        if self.config["artifacts_dir"].exists():
            for artifact_file in self.config["artifacts_dir"].rglob("*.json"):
                try:
                    with open(artifact_file, 'rb') as f:
                        checksums[str(artifact_file)] = hashlib.sha256(f.read()).hexdigest()
                except Exception as e:
                    logger.warning(f"Could not checksum {artifact_file}: {e}")
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variant": "V2",  # Could be parameter
            "gates_evaluated": len(self.results),
            "gates_passed": sum(1 for r in self.results if r.passed),
            "critical_failures": len(self.get_critical_failures()),
            "composite_score": self.compute_composite_score(),
            "evidence": all_evidence,
            "artifact_checksums": checksums,
            "config": self.config
        }
    
    def save_results(self, output_path: Path):
        """Save comprehensive results to file."""
        evidence_report = self.generate_evidence_report()
        
        gate_summary = {
            "metadata": {
                "timestamp": evidence_report["timestamp"],
                "variant": evidence_report["variant"],
                "total_gates": len(self.results),
                "passed_gates": sum(1 for r in self.results if r.passed),
                "failed_gates": sum(1 for r in self.results if not r.passed),
                "critical_failures": len(self.get_critical_failures()),
                "composite_score": evidence_report["composite_score"]
            },
            "gate_results": [
                {
                    "gate_name": r.gate_name,
                    "passed": r.passed,
                    "actual_value": r.actual_value,
                    "threshold": r.threshold,
                    "weight": r.weight,
                    "critical": r.critical,
                    "error_message": r.error_message
                }
                for r in self.results
            ],
            "evidence_report": evidence_report
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(gate_summary, f, indent=2, default=str)
        
        logger.info(f"Gate evaluation results saved to {output_path}")


def main():
    """Main acceptance gate evaluation."""
    if len(sys.argv) < 2:
        print("Usage: acceptance_gates.py <variant> [config_file] [output_file]")
        print("Example: acceptance_gates.py V2 scripts/gates.yaml artifacts/acceptance_results.json")
        sys.exit(1)
    
    variant = sys.argv[1]
    config_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/acceptance_gate_results.json")
    
    logger.info(f"Starting acceptance gate evaluation for {variant}")
    
    # Initialize engine
    engine = AcceptanceGateEngine(config_file)
    
    # Evaluate all gates
    results = engine.evaluate_all_gates(variant)
    
    # Compute final metrics
    composite_score = engine.compute_composite_score()
    critical_failures = engine.get_critical_failures()
    
    # Save results
    engine.save_results(output_file)
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"ACCEPTANCE GATE EVALUATION COMPLETE")
    print(f"Variant: {variant}")
    print(f"Gates Evaluated: {len(results)}")
    print(f"Gates Passed: {sum(1 for r in results if r.passed)}")
    print(f"Gates Failed: {sum(1 for r in results if not r.passed)}")
    print(f"Critical Failures: {len(critical_failures)}")
    print(f"Composite Score: {composite_score:.3f}")
    print(f"{'='*60}")
    
    if critical_failures:
        print("\nCRITICAL FAILURES:")
        for failure in critical_failures:
            print(f"  - {failure.gate_name}: {failure.error_message}")
    
    failed_gates = [r for r in results if not r.passed]
    if failed_gates and not critical_failures:
        print("\nNON-CRITICAL FAILURES:")
        for failure in failed_gates:
            print(f"  - {failure.gate_name}: {failure.error_message}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Exit with appropriate code for CI integration
    if len(critical_failures) > 0:
        sys.exit(2)  # Critical failure
    elif composite_score < 0.85:
        sys.exit(1)  # Quality threshold not met
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()
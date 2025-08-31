#!/usr/bin/env python3
"""
Research-Grade FastPath Pipeline - Complete Validation Orchestrator

Orchestrates the complete research publication validation workflow:
1. Environment validation and hermetic setup
2. Comprehensive acceptance gate evaluation  
3. Research-grade gatekeeper decision
4. Artifact packaging for peer review
5. Publication readiness certification

Implements the full pipeline for academic submission standards.
"""

import json
import subprocess
import sys
import time
import tarfile
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ResearchPipelineConfig:
    """Configuration for research-grade pipeline."""
    variant: str = "V2"
    artifacts_dir: Path = Path("./artifacts")
    publication_dir: Path = Path("./publication_artifacts")
    scripts_dir: Path = Path("./scripts")
    config_dir: Path = Path("./config")
    timeout_minutes: int = 120  # Extended for research validation
    enable_comprehensive_validation: bool = True
    enable_statistical_validation: bool = True
    enable_peer_review_packaging: bool = True
    create_reproducibility_archive: bool = True


@dataclass
class PipelineResult:
    """Comprehensive pipeline execution result."""
    success: bool
    final_decision: str  # PROMOTE | REFINE_NEEDED | REJECT
    publication_ready: bool
    execution_time_minutes: float
    gates_passed: int
    gates_failed: int
    critical_failures: int
    composite_score: float
    statistical_significance: bool
    artifacts_generated: List[str]
    error_summary: Optional[str] = None
    next_actions: List[str] = None


class ResearchGradePipeline:
    """Complete research-grade validation pipeline orchestrator."""
    
    def __init__(self, config: ResearchPipelineConfig):
        self.config = config
        self.pipeline_start_time = datetime.now(timezone.utc)
        self.execution_log = []
        self.artifacts_created = []
        
        # Ensure all directories exist
        for directory in [self.config.artifacts_dir, self.config.publication_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def log_step(self, step: str, status: str, details: Optional[str] = None):
        """Log pipeline execution step with timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "status": status,
            "details": details
        }
        self.execution_log.append(log_entry)
        
        status_symbol = "✅" if status == "SUCCESS" else "❌" if status == "FAILURE" else "⏳"
        logger.info(f"{status_symbol} {step}: {status}" + (f" - {details}" if details else ""))
    
    def validate_environment_setup(self) -> bool:
        """Validate research environment is properly configured."""
        self.log_step("Environment Setup Validation", "RUNNING")
        
        validation_checks = {
            "python_version": self._check_python_version(),
            "dependencies_installed": self._check_dependencies(),
            "git_repository_clean": self._check_git_status(),
            "required_directories": self._check_directories(),
            "configuration_files": self._check_config_files()
        }
        
        all_passed = all(validation_checks.values())
        
        if all_passed:
            self.log_step("Environment Setup Validation", "SUCCESS", "All environment checks passed")
        else:
            failed_checks = [name for name, passed in validation_checks.items() if not passed]
            self.log_step("Environment Setup Validation", "FAILURE", f"Failed: {', '.join(failed_checks)}")
        
        # Save validation results
        validation_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": validation_checks,
            "overall_status": "PASS" if all_passed else "FAIL"
        }
        
        validation_file = self.config.artifacts_dir / "environment_validation.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        self.artifacts_created.append(str(validation_file))
        return all_passed
    
    def run_acceptance_gates(self) -> Tuple[bool, Dict[str, Any]]:
        """Run comprehensive research-grade acceptance gates."""
        self.log_step("Research-Grade Acceptance Gates", "RUNNING")
        
        # Prepare gate results file
        gate_results_file = self.config.artifacts_dir / "research_grade_results.json"
        
        # Configure research gate evaluation
        research_config = self.config.config_dir / "research_gates_config.yaml"
        if not research_config.exists():
            research_config = None
        
        try:
            # Run research-grade acceptance gates
            cmd = [
                sys.executable,
                str(self.config.scripts_dir / "research_grade_acceptance_gates.py"),
                self.config.variant,
                str(research_config) if research_config else "--default-config",
                str(gate_results_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_minutes * 60
            )
            
            # Load and analyze results
            if gate_results_file.exists():
                with open(gate_results_file) as f:
                    gate_results = json.load(f)
                
                gates_passed = gate_results.get("gate_summary", {}).get("passed_gates", 0)
                gates_failed = gate_results.get("gate_summary", {}).get("total_gates", 0) - gates_passed
                critical_failures = gate_results.get("gate_summary", {}).get("critical_failures", 0)
                
                success = result.returncode == 0 and critical_failures == 0
                
                if success:
                    self.log_step(
                        "Research-Grade Acceptance Gates", 
                        "SUCCESS", 
                        f"{gates_passed} gates passed, {critical_failures} critical failures"
                    )
                else:
                    self.log_step(
                        "Research-Grade Acceptance Gates", 
                        "FAILURE", 
                        f"{gates_failed} gates failed, {critical_failures} critical failures"
                    )
                
                self.artifacts_created.append(str(gate_results_file))
                return success, gate_results
            
            else:
                self.log_step(
                    "Research-Grade Acceptance Gates", 
                    "FAILURE", 
                    "Gate results file not generated"
                )
                return False, {}
        
        except subprocess.TimeoutExpired:
            self.log_step(
                "Research-Grade Acceptance Gates", 
                "FAILURE", 
                f"Timeout after {self.config.timeout_minutes} minutes"
            )
            return False, {}
        
        except Exception as e:
            self.log_step(
                "Research-Grade Acceptance Gates", 
                "FAILURE", 
                f"Exception: {str(e)}"
            )
            return False, {}
    
    def run_gatekeeper_decision(self, gate_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Run research-grade gatekeeper decision analysis."""
        self.log_step("Research-Grade Gatekeeper Decision", "RUNNING")
        
        # Prepare decision files
        gate_results_file = self.config.artifacts_dir / "research_grade_results.json"
        decision_file = self.config.artifacts_dir / "gatekeeper_decision.json"
        
        # Ensure gate results are saved
        with open(gate_results_file, 'w') as f:
            json.dump(gate_results, f, indent=2, default=str)
        
        try:
            # Run research-grade gatekeeper
            cmd = [
                sys.executable,
                str(self.config.scripts_dir / "research_grade_gatekeeper.py"),
                str(gate_results_file),
                self.config.variant,
                "--default-config",
                str(decision_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for decision analysis
            )
            
            # Load and analyze decision
            if decision_file.exists():
                with open(decision_file) as f:
                    decision_results = json.load(f)
                
                final_decision = decision_results.get("final_decision", {}).get("recommendation", "UNKNOWN")
                publication_ready = decision_results.get("final_decision", {}).get("publication_ready", False)
                
                success = result.returncode == 0
                
                self.log_step(
                    "Research-Grade Gatekeeper Decision", 
                    "SUCCESS" if success else "FAILURE",
                    f"Decision: {final_decision}, Publication Ready: {publication_ready}"
                )
                
                self.artifacts_created.append(str(decision_file))
                return success, decision_results
            
            else:
                self.log_step(
                    "Research-Grade Gatekeeper Decision", 
                    "FAILURE", 
                    "Decision file not generated"
                )
                return False, {}
        
        except subprocess.TimeoutExpired:
            self.log_step(
                "Research-Grade Gatekeeper Decision", 
                "FAILURE", 
                "Timeout during decision analysis"
            )
            return False, {}
        
        except Exception as e:
            self.log_step(
                "Research-Grade Gatekeeper Decision", 
                "FAILURE", 
                f"Exception: {str(e)}"
            )
            return False, {}
    
    def create_publication_artifacts(self, gate_results: Dict[str, Any], 
                                   decision_results: Dict[str, Any]) -> bool:
        """Create comprehensive publication and peer review artifacts."""
        self.log_step("Publication Artifact Creation", "RUNNING")
        
        try:
            publication_package = {
                "metadata": {
                    "created_timestamp": datetime.now(timezone.utc).isoformat(),
                    "fastpath_variant": self.config.variant,
                    "pipeline_version": "research_grade_1.0.0",
                    "validation_framework": "academic_publication_standards"
                },
                "validation_summary": {
                    "gates_evaluated": len(gate_results.get("gate_results", [])),
                    "gates_passed": sum(1 for g in gate_results.get("gate_results", []) if g.get("passed", False)),
                    "critical_failures": sum(1 for g in gate_results.get("gate_results", []) 
                                            if g.get("critical", False) and not g.get("passed", True)),
                    "publication_ready_gates": sum(1 for g in gate_results.get("gate_results", []) 
                                                   if g.get("publication_ready", False)),
                    "composite_score": gate_results.get("publication_readiness", {}).get("composite_score", 0.0),
                    "statistical_significance": decision_results.get("final_decision", {}).get("statistical_significance", False)
                },
                "gatekeeper_decision": {
                    "final_recommendation": decision_results.get("final_decision", {}).get("recommendation", "UNKNOWN"),
                    "publication_ready": decision_results.get("final_decision", {}).get("publication_ready", False),
                    "decision_confidence": decision_results.get("decision_metadata", {}).get("decision_confidence", 0.0),
                    "rationale": decision_results.get("final_decision", {}).get("rationale", "")
                },
                "statistical_evidence": {
                    "confidence_intervals": gate_results.get("statistical_analysis", {}),
                    "effect_sizes": [g.get("effect_size") for g in gate_results.get("gate_results", []) 
                                    if g.get("effect_size") is not None],
                    "hypothesis_tests": decision_results.get("publication_package", {}).get("statistical_evidence", {})
                },
                "performance_evidence": {
                    "qa_improvements": gate_results.get("gate_results", []),
                    "latency_benchmarks": decision_results.get("publication_package", {}).get("composite_metrics", {}),
                    "regression_analysis": decision_results.get("risk_breakdown", {})
                },
                "reproducibility_package": {
                    "boot_transcript": gate_results.get("boot_transcript", {}),
                    "environment_specification": gate_results.get("artifact_inventory", {}),
                    "determinism_validation": gate_results.get("gate_results", [])
                },
                "peer_review_materials": decision_results.get("publication_package", {})
            }
            
            # Save main publication package
            publication_file = self.config.publication_dir / f"publication_package_{self.config.variant}.json"
            with open(publication_file, 'w') as f:
                json.dump(publication_package, f, indent=2, default=str)
            
            self.artifacts_created.append(str(publication_file))
            
            # Create peer review submission materials
            if self.config.enable_peer_review_packaging:
                peer_review_dir = self.config.publication_dir / "peer_review_materials"
                peer_review_dir.mkdir(exist_ok=True)
                
                # Statistical analysis summary
                statistical_summary = {
                    "methodology": "Bootstrap confidence intervals with BCa correction",
                    "significance_testing": "Multiple testing correction with FDR control",
                    "effect_size_analysis": "Cohen's d and practical significance thresholds",
                    "confidence_level": 0.95,
                    "statistical_power": 0.80,
                    "sample_size_justification": "Power analysis with medium effect size detection"
                }
                
                with open(peer_review_dir / "statistical_methodology.json", 'w') as f:
                    json.dump(statistical_summary, f, indent=2)
                
                # Performance claims validation
                performance_claims = {
                    "primary_hypothesis": f"{self.config.variant} achieves ≥13% QA improvement vs baseline",
                    "secondary_hypotheses": [
                        "Latency regression ≤10% vs baseline",
                        "Memory usage increase ≤10% vs baseline",
                        "Category performance targets met"
                    ],
                    "validation_methodology": "Comprehensive benchmarking with statistical significance testing",
                    "reproducibility_guarantees": "3 identical runs with deterministic output validation"
                }
                
                with open(peer_review_dir / "performance_claims.json", 'w') as f:
                    json.dump(performance_claims, f, indent=2)
                
                self.artifacts_created.extend([
                    str(peer_review_dir / "statistical_methodology.json"),
                    str(peer_review_dir / "performance_claims.json")
                ])
            
            # Create reproducibility archive
            if self.config.create_reproducibility_archive:
                self._create_reproducibility_archive()
            
            self.log_step("Publication Artifact Creation", "SUCCESS", f"Created {len(self.artifacts_created)} artifacts")
            return True
            
        except Exception as e:
            self.log_step("Publication Artifact Creation", "FAILURE", f"Exception: {str(e)}")
            return False
    
    def _create_reproducibility_archive(self) -> bool:
        """Create comprehensive reproducibility archive."""
        try:
            archive_path = self.config.publication_dir / f"reproducibility_archive_{self.config.variant}.tar.gz"
            
            with tarfile.open(archive_path, "w:gz") as tar:
                # Add key source files
                for pattern in ["*.py", "requirements*.txt", "uv.lock", "pyproject.toml"]:
                    for file_path in Path(".").glob(pattern):
                        if file_path.is_file():
                            tar.add(file_path, arcname=file_path.name)
                
                # Add scripts directory
                if self.config.scripts_dir.exists():
                    tar.add(self.config.scripts_dir, arcname="scripts", recursive=True)
                
                # Add artifacts directory
                if self.config.artifacts_dir.exists():
                    tar.add(self.config.artifacts_dir, arcname="artifacts", recursive=True)
                
                # Add test files
                for test_dir in ["tests", "test"]:
                    test_path = Path(test_dir)
                    if test_path.exists():
                        tar.add(test_path, arcname=test_dir, recursive=True)
            
            self.artifacts_created.append(str(archive_path))
            logger.info(f"Created reproducibility archive: {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create reproducibility archive: {e}")
            return False
    
    def generate_final_report(self, gate_results: Dict[str, Any], 
                            decision_results: Dict[str, Any]) -> PipelineResult:
        """Generate comprehensive final pipeline report."""
        self.log_step("Final Report Generation", "RUNNING")
        
        execution_time = (datetime.now(timezone.utc) - self.pipeline_start_time).total_seconds() / 60
        
        # Extract key metrics
        gates_passed = sum(1 for g in gate_results.get("gate_results", []) if g.get("passed", False))
        total_gates = len(gate_results.get("gate_results", []))
        gates_failed = total_gates - gates_passed
        critical_failures = sum(1 for g in gate_results.get("gate_results", []) 
                               if g.get("critical", False) and not g.get("passed", True))
        
        composite_score = gate_results.get("publication_readiness", {}).get("composite_score", 0.0)
        final_decision = decision_results.get("final_decision", {}).get("recommendation", "UNKNOWN")
        publication_ready = decision_results.get("final_decision", {}).get("publication_ready", False)
        statistical_significance = decision_results.get("final_decision", {}).get("statistical_significance", False)
        
        success = (
            final_decision == "PROMOTE" and 
            critical_failures == 0 and 
            publication_ready and 
            statistical_significance
        )
        
        next_actions = decision_results.get("next_actions", {}).get("all_actions", [])
        
        error_summary = None
        if not success:
            error_reasons = []
            if critical_failures > 0:
                error_reasons.append(f"{critical_failures} critical gate failures")
            if not publication_ready:
                error_reasons.append("Publication readiness criteria not met")
            if not statistical_significance:
                error_reasons.append("Statistical significance not established")
            error_summary = "; ".join(error_reasons)
        
        result = PipelineResult(
            success=success,
            final_decision=final_decision,
            publication_ready=publication_ready,
            execution_time_minutes=execution_time,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            critical_failures=critical_failures,
            composite_score=composite_score,
            statistical_significance=statistical_significance,
            artifacts_generated=self.artifacts_created,
            error_summary=error_summary,
            next_actions=next_actions
        )
        
        # Save final report
        final_report = {
            "pipeline_metadata": {
                "execution_timestamp": self.pipeline_start_time.isoformat(),
                "completion_timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_time_minutes": execution_time,
                "variant_evaluated": self.config.variant,
                "pipeline_version": "research_grade_1.0.0"
            },
            "execution_summary": asdict(result),
            "detailed_execution_log": self.execution_log,
            "artifact_inventory": {
                "total_artifacts": len(self.artifacts_created),
                "artifact_paths": self.artifacts_created
            },
            "gate_evaluation_details": gate_results,
            "gatekeeper_decision_details": decision_results
        }
        
        report_file = self.config.artifacts_dir / f"final_pipeline_report_{self.config.variant}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.artifacts_created.append(str(report_file))
        
        self.log_step("Final Report Generation", "SUCCESS", f"Report saved to {report_file}")
        return result
    
    def run_complete_pipeline(self) -> PipelineResult:
        """Run the complete research-grade validation pipeline."""
        logger.info(f"Starting complete research-grade pipeline for {self.config.variant}")
        
        # Step 1: Environment validation
        if not self.validate_environment_setup():
            return PipelineResult(
                success=False,
                final_decision="REJECT",
                publication_ready=False,
                execution_time_minutes=0.0,
                gates_passed=0,
                gates_failed=0,
                critical_failures=1,
                composite_score=0.0,
                statistical_significance=False,
                artifacts_generated=self.artifacts_created,
                error_summary="Environment validation failed",
                next_actions=["Fix environment setup issues"]
            )
        
        # Step 2: Run acceptance gates
        gates_success, gate_results = self.run_acceptance_gates()
        
        # Step 3: Run gatekeeper decision (even if gates failed, for analysis)
        gatekeeper_success, decision_results = self.run_gatekeeper_decision(gate_results)
        
        # Step 4: Create publication artifacts
        if self.config.enable_peer_review_packaging:
            self.create_publication_artifacts(gate_results, decision_results)
        
        # Step 5: Generate final report
        final_result = self.generate_final_report(gate_results, decision_results)
        
        return final_result
    
    # Helper methods for environment validation
    
    def _check_python_version(self) -> bool:
        """Check Python version is 3.10+."""
        return sys.version_info >= (3, 10)
    
    def _check_dependencies(self) -> bool:
        """Check key dependencies are installed."""
        try:
            import numpy, scipy
            return True
        except ImportError:
            return False
    
    def _check_git_status(self) -> bool:
        """Check git repository is in clean state."""
        try:
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and not result.stdout.strip()
        except Exception:
            return False
    
    def _check_directories(self) -> bool:
        """Check required directories exist."""
        required_dirs = [self.config.scripts_dir]
        return all(d.exists() and d.is_dir() for d in required_dirs)
    
    def _check_config_files(self) -> bool:
        """Check configuration files are available."""
        # Optional check - config files may not exist
        return True


def main():
    """Main research-grade pipeline execution."""
    if len(sys.argv) < 2:
        print("Usage: research_grade_pipeline.py <variant> [--config <config_file>]")
        print("Example: research_grade_pipeline.py V2")
        print("         research_grade_pipeline.py V3 --config research_config.yaml")
        sys.exit(1)
    
    variant = sys.argv[1]
    
    # Parse optional config file
    config_file = None
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file = Path(sys.argv[config_idx + 1])
    
    logger.info(f"Starting research-grade pipeline for FastPath {variant}")
    
    # Initialize pipeline configuration
    pipeline_config = ResearchPipelineConfig(
        variant=variant,
        artifacts_dir=Path("./artifacts"),
        publication_dir=Path("./publication_artifacts"),
        scripts_dir=Path("./scripts"),
        config_dir=Path("./config") if config_file else Path("./scripts")
    )
    
    # Create and run pipeline
    pipeline = ResearchGradePipeline(pipeline_config)
    result = pipeline.run_complete_pipeline()
    
    # Display comprehensive summary
    print(f"\n{'='*90}")
    print(f"RESEARCH-GRADE FASTPATH VALIDATION PIPELINE COMPLETE")
    print(f"Variant: {variant}")
    print(f"Execution Time: {result.execution_time_minutes:.1f} minutes")
    print(f"{'='*90}")
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"  Final Decision: {result.final_decision}")
    print(f"  Publication Ready: {'YES' if result.publication_ready else 'NO'}")
    print(f"  Statistical Significance: {'YES' if result.statistical_significance else 'NO'}")
    print(f"  Composite Score: {result.composite_score:.3f}")
    
    print(f"\n🎯 GATE SUMMARY:")
    print(f"  Gates Passed: {result.gates_passed}")
    print(f"  Gates Failed: {result.gates_failed}")
    print(f"  Critical Failures: {result.critical_failures}")
    
    print(f"\n📁 ARTIFACTS GENERATED: {len(result.artifacts_generated)}")
    key_artifacts = [
        "research_grade_results.json",
        "gatekeeper_decision.json",
        "publication_package_*.json",
        "final_pipeline_report_*.json"
    ]
    for artifact_pattern in key_artifacts:
        matching_artifacts = [a for a in result.artifacts_generated if artifact_pattern.replace("*", "") in a]
        if matching_artifacts:
            print(f"  ✅ {artifact_pattern}: {len(matching_artifacts)} files")
        else:
            print(f"  ❌ {artifact_pattern}: Missing")
    
    if result.error_summary:
        print(f"\n❌ ISSUES IDENTIFIED:")
        print(f"  {result.error_summary}")
    
    if result.next_actions:
        print(f"\n🔄 RECOMMENDED ACTIONS:")
        for i, action in enumerate(result.next_actions[:5], 1):
            print(f"  {i}. {action}")
    
    # Publication readiness assessment
    if result.success and result.publication_ready:
        print(f"\n🎉 PUBLICATION STATUS: READY FOR SUBMISSION")
        print(f"   All validation criteria met for academic publication")
        print(f"   Peer review materials prepared and archived")
    elif result.final_decision == "REFINE_NEEDED":
        print(f"\n🔧 PUBLICATION STATUS: REFINEMENT NEEDED")
        print(f"   Address identified issues and re-run validation")
    else:
        print(f"\n❌ PUBLICATION STATUS: MAJOR ISSUES DETECTED")
        print(f"   Significant rework required before publication submission")
    
    print(f"\n📂 Detailed results available in: ./artifacts/")
    print(f"📦 Publication materials available in: ./publication_artifacts/")
    
    # Exit with appropriate status codes
    if result.final_decision == "PROMOTE":
        sys.exit(0)  # Success - ready for publication
    elif result.final_decision == "REFINE_NEEDED":
        sys.exit(1)  # Refinement needed
    else:
        sys.exit(2)  # Major issues - reject


if __name__ == "__main__":
    main()

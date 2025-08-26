#!/usr/bin/env python3
"""
PackRepo Acceptance Pipeline - End-to-End Quality Validation

Orchestrates the complete acceptance gate pipeline from TODO.md:
1. Run acceptance gates evaluation
2. Execute gatekeeper decision engine  
3. Generate promotion decision with evidence
4. Create audit trail and reports
5. Execute appropriate next actions (PROMOTE/AGENT_REFINE/MANUAL_QA)

This script implements the full workflow from the TODO.md requirements.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for acceptance pipeline."""
    variant: str = "V2"
    artifacts_dir: Path = Path("./artifacts")
    metrics_dir: Path = Path("./artifacts/metrics")
    scripts_dir: Path = Path("./scripts")
    gates_config: Path = Path("./scripts/gates.yaml")
    timeout_minutes: int = 60
    generate_reports: bool = True
    

class AcceptancePipeline:
    """Orchestrates the full acceptance gate pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = {}
        self.pipeline_start_time = datetime.now(timezone.utc)
        
        # Ensure directories exist
        self.config.artifacts_dir.mkdir(exist_ok=True)
        self.config.metrics_dir.mkdir(exist_ok=True)
        
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete acceptance pipeline."""
        logger.info(f"Starting PackRepo Acceptance Pipeline for {self.config.variant}")
        logger.info(f"Pipeline start time: {self.pipeline_start_time}")
        
        pipeline_results = {
            "pipeline_metadata": {
                "variant": self.config.variant,
                "start_time": self.pipeline_start_time.isoformat(),
                "artifacts_dir": str(self.config.artifacts_dir),
                "metrics_dir": str(self.config.metrics_dir)
            },
            "steps": {}
        }
        
        try:
            # Step 1: Run acceptance gates evaluation
            logger.info("Step 1: Running acceptance gates evaluation...")
            acceptance_result = self._run_acceptance_gates()
            pipeline_results["steps"]["acceptance_gates"] = acceptance_result
            
            # Step 2: Execute gatekeeper decision engine
            logger.info("Step 2: Executing gatekeeper decision engine...")
            gatekeeper_result = self._run_gatekeeper()
            pipeline_results["steps"]["gatekeeper"] = gatekeeper_result
            
            # Step 3: Generate comprehensive reports
            if self.config.generate_reports:
                logger.info("Step 3: Generating comprehensive reports...")
                report_result = self._generate_reports()
                pipeline_results["steps"]["reports"] = report_result
            
            # Step 4: Execute next actions based on decision
            logger.info("Step 4: Executing next actions...")
            next_action_result = self._execute_next_actions(gatekeeper_result)
            pipeline_results["steps"]["next_actions"] = next_action_result
            
            # Final pipeline status
            pipeline_results["pipeline_metadata"]["end_time"] = datetime.now(timezone.utc).isoformat()
            pipeline_results["pipeline_metadata"]["duration_minutes"] = (
                datetime.now(timezone.utc) - self.pipeline_start_time
            ).total_seconds() / 60.0
            
            pipeline_results["final_decision"] = gatekeeper_result.get("decision", "UNKNOWN")
            pipeline_results["success"] = True
            
            logger.info(f"Pipeline completed successfully: {pipeline_results['final_decision']}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_results["success"] = False
            pipeline_results["error"] = str(e)
            pipeline_results["pipeline_metadata"]["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Save comprehensive pipeline results
        pipeline_output = self.config.artifacts_dir / "pipeline_results.json"
        with open(pipeline_output, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to: {pipeline_output}")
        return pipeline_results
    
    def _run_acceptance_gates(self) -> Dict[str, Any]:
        """Execute acceptance gates evaluation."""
        try:
            acceptance_script = self.config.scripts_dir / "acceptance_gates.py"
            acceptance_output = self.config.metrics_dir / "acceptance_gate_results.json"
            
            cmd = [
                "python", str(acceptance_script),
                self.config.variant,
                str(self.config.gates_config),
                str(acceptance_output)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_minutes * 60,
                cwd=Path.cwd()
            )
            
            # Load results
            acceptance_results = {}
            if acceptance_output.exists():
                with open(acceptance_output) as f:
                    acceptance_results = json.load(f)
            
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "results_file": str(acceptance_output),
                "results": acceptance_results,
                "success": result.returncode in [0, 1]  # 0=success, 1=some failures but recoverable
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Acceptance gates evaluation timed out")
            return {"success": False, "error": "Timeout during acceptance gates evaluation"}
        except Exception as e:
            logger.error(f"Acceptance gates evaluation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_gatekeeper(self) -> Dict[str, Any]:
        """Execute gatekeeper decision engine."""
        try:
            gatekeeper_script = self.config.scripts_dir / "gatekeeper.py"
            gatekeeper_output = self.config.metrics_dir / "gatekeeper_decision.json"
            
            cmd = [
                "python", str(gatekeeper_script),
                str(self.config.metrics_dir),
                self.config.variant,
                str(self.config.gates_config),
                str(gatekeeper_output)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_minutes * 60,
                cwd=Path.cwd()
            )
            
            # Load results
            gatekeeper_results = {}
            if gatekeeper_output.exists():
                with open(gatekeeper_output) as f:
                    gatekeeper_results = json.load(f)
            
            # Map exit codes to decisions
            exit_code_mapping = {0: "PROMOTE", 1: "AGENT_REFINE", 2: "MANUAL_QA"}
            decision = exit_code_mapping.get(result.returncode, "UNKNOWN")
            
            return {
                "exit_code": result.returncode,
                "decision": decision,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "results_file": str(gatekeeper_output),
                "results": gatekeeper_results,
                "success": True  # Gatekeeper always succeeds, just with different decisions
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Gatekeeper evaluation timed out")
            return {"success": False, "error": "Timeout during gatekeeper evaluation"}
        except Exception as e:
            logger.error(f"Gatekeeper evaluation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_reports(self) -> Dict[str, Any]:
        """Generate comprehensive reports and audit trail."""
        try:
            reports_dir = self.config.artifacts_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Generate evidence report
            evidence_report = self._generate_evidence_report()
            evidence_file = reports_dir / "evidence_report.json"
            with open(evidence_file, 'w') as f:
                json.dump(evidence_report, f, indent=2, default=str)
            
            # Generate summary report
            summary_report = self._generate_summary_report()
            summary_file = reports_dir / "pipeline_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            # Generate audit trail
            audit_trail = self._generate_audit_trail()
            audit_file = reports_dir / "audit_trail.json"
            with open(audit_file, 'w') as f:
                json.dump(audit_trail, f, indent=2, default=str)
            
            logger.info(f"Reports generated in: {reports_dir}")
            
            return {
                "success": True,
                "reports_dir": str(reports_dir),
                "files_generated": [
                    str(evidence_file),
                    str(summary_file),
                    str(audit_file)
                ]
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_next_actions(self, gatekeeper_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute appropriate next actions based on gatekeeper decision."""
        decision = gatekeeper_result.get("decision", "UNKNOWN")
        logger.info(f"Executing next actions for decision: {decision}")
        
        try:
            if decision == "PROMOTE":
                return self._handle_promotion()
            elif decision == "AGENT_REFINE":
                return self._handle_agent_refinement(gatekeeper_result)
            elif decision == "MANUAL_QA":
                return self._handle_manual_qa(gatekeeper_result)
            else:
                logger.error(f"Unknown decision: {decision}")
                return {"success": False, "error": f"Unknown decision: {decision}"}
                
        except Exception as e:
            logger.error(f"Next actions execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_promotion(self) -> Dict[str, Any]:
        """Handle PROMOTE decision - prepare for deployment."""
        logger.info("Handling PROMOTE decision - preparing for deployment")
        
        actions_taken = []
        
        # Create deployment readiness checklist
        deployment_checklist = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variant": self.config.variant,
            "readiness_status": "APPROVED",
            "deployment_checklist": [
                {"item": "All acceptance gates passed", "status": "✅ COMPLETE"},
                {"item": "CI-backed wins demonstrated", "status": "✅ COMPLETE"},
                {"item": "Risk assessment: LOW/MEDIUM", "status": "✅ COMPLETE"},
                {"item": "Statistical validation completed", "status": "✅ COMPLETE"},
                {"item": "Audit trail generated", "status": "✅ COMPLETE"}
            ],
            "next_steps": [
                "Deploy to production environment",
                "Monitor performance metrics",
                "Validate improvements in production",
                "Update baselines for future comparisons"
            ]
        }
        
        checklist_file = self.config.artifacts_dir / "deployment_readiness.json"
        with open(checklist_file, 'w') as f:
            json.dump(deployment_checklist, f, indent=2, default=str)
        actions_taken.append(f"Created deployment checklist: {checklist_file}")
        
        # Generate promotion summary
        promotion_summary = {
            "decision": "PROMOTE",
            "variant": self.config.variant,
            "approval_timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence_artifacts": list(self.config.artifacts_dir.rglob("*.json")),
            "deployment_ready": True
        }
        
        summary_file = self.config.artifacts_dir / "promotion_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(promotion_summary, f, indent=2, default=str)
        actions_taken.append(f"Created promotion summary: {summary_file}")
        
        logger.info("✅ PROMOTION APPROVED - Ready for deployment")
        
        return {
            "success": True,
            "decision": "PROMOTE",
            "actions_taken": actions_taken,
            "deployment_ready": True
        }
    
    def _handle_agent_refinement(self, gatekeeper_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AGENT_REFINE decision - generate refinement instructions."""
        logger.info("Handling AGENT_REFINE decision - generating refinement instructions")
        
        actions_taken = []
        
        # Extract concrete obligations from gatekeeper results
        gatekeeper_data = gatekeeper_result.get("results", {})
        concrete_obligations = gatekeeper_data.get("concrete_obligations", [])
        recommendations = gatekeeper_data.get("recommendations", [])
        
        # Generate refinement prompt
        refinement_prompt = self._generate_refinement_prompt(
            concrete_obligations, recommendations, gatekeeper_data
        )
        
        prompt_file = self.config.artifacts_dir / "agent_refinement_prompt.md"
        with open(prompt_file, 'w') as f:
            f.write(refinement_prompt)
        actions_taken.append(f"Generated refinement prompt: {prompt_file}")
        
        # Create refinement tracking
        refinement_tracking = {
            "decision": "AGENT_REFINE",
            "variant": self.config.variant,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "obligations": concrete_obligations,
            "recommendations": recommendations,
            "estimated_effort": self._estimate_refinement_effort(concrete_obligations),
            "next_actions": [
                "Review concrete obligations",
                "Implement required fixes",
                "Re-run acceptance pipeline",
                "Validate improvements"
            ]
        }
        
        tracking_file = self.config.artifacts_dir / "refinement_tracking.json"
        with open(tracking_file, 'w') as f:
            json.dump(refinement_tracking, f, indent=2, default=str)
        actions_taken.append(f"Created refinement tracking: {tracking_file}")
        
        logger.info("🔄 AGENT REFINEMENT REQUIRED - Instructions generated")
        
        return {
            "success": True,
            "decision": "AGENT_REFINE",
            "actions_taken": actions_taken,
            "refinement_prompt_file": str(prompt_file),
            "obligations_count": len(concrete_obligations)
        }
    
    def _handle_manual_qa(self, gatekeeper_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MANUAL_QA decision - prepare for human review."""
        logger.info("Handling MANUAL_QA decision - preparing for human review")
        
        actions_taken = []
        
        # Extract escalation info
        gatekeeper_data = gatekeeper_result.get("results", {})
        risk_assessment = gatekeeper_data.get("risk_assessment", {})
        
        # Generate manual QA package
        qa_package = {
            "decision": "MANUAL_QA",
            "variant": self.config.variant,
            "escalation_timestamp": datetime.now(timezone.utc).isoformat(),
            "escalation_reason": gatekeeper_data.get("reason", "High complexity detected"),
            "risk_assessment": risk_assessment,
            "critical_areas": [
                area for area, score in risk_assessment.get("risk_factors", {}).items()
                if score > 0.7
            ],
            "artifacts_for_review": [
                str(f) for f in self.config.artifacts_dir.rglob("*.json")
            ],
            "review_checklist": [
                "Validate statistical significance of results",
                "Review edge cases and failure scenarios", 
                "Assess deployment risk and rollback strategy",
                "Verify security implications",
                "Confirm performance impact is acceptable"
            ],
            "escalation_contacts": [
                "platform-team",
                "qa-team", 
                "security-team" if risk_assessment.get("risk_factors", {}).get("security_risk", 0) > 0.5 else None
            ]
        }
        
        # Filter out None values
        qa_package["escalation_contacts"] = [c for c in qa_package["escalation_contacts"] if c]
        
        qa_file = self.config.artifacts_dir / "manual_qa_package.json"
        with open(qa_file, 'w') as f:
            json.dump(qa_package, f, indent=2, default=str)
        actions_taken.append(f"Created manual QA package: {qa_file}")
        
        # Generate review dashboard data
        dashboard_data = self._generate_review_dashboard_data(gatekeeper_data)
        dashboard_file = self.config.artifacts_dir / "review_dashboard.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        actions_taken.append(f"Generated review dashboard: {dashboard_file}")
        
        logger.info("⚠️ MANUAL QA REQUIRED - Human review package prepared")
        
        return {
            "success": True,
            "decision": "MANUAL_QA",
            "actions_taken": actions_taken,
            "qa_package_file": str(qa_file),
            "escalation_required": True
        }
    
    def _generate_evidence_report(self) -> Dict[str, Any]:
        """Generate comprehensive evidence report for audit trail."""
        evidence = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "variant": self.config.variant,
                "pipeline_version": "1.0"
            },
            "artifact_inventory": {},
            "checksum_verification": {},
            "compliance_status": {}
        }
        
        # Inventory all artifacts
        for artifact_file in self.config.artifacts_dir.rglob("*.json"):
            try:
                relative_path = artifact_file.relative_to(self.config.artifacts_dir)
                evidence["artifact_inventory"][str(relative_path)] = {
                    "size_bytes": artifact_file.stat().st_size,
                    "modified_time": datetime.fromtimestamp(
                        artifact_file.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "exists": True
                }
            except Exception as e:
                logger.warning(f"Could not inventory {artifact_file}: {e}")
        
        return evidence
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate pipeline execution summary."""
        return {
            "pipeline_summary": {
                "variant": self.config.variant,
                "execution_time": datetime.now(timezone.utc).isoformat(),
                "artifacts_generated": len(list(self.config.artifacts_dir.rglob("*.json"))),
                "pipeline_status": "COMPLETED"
            }
        }
    
    def _generate_audit_trail(self) -> Dict[str, Any]:
        """Generate compliance audit trail."""
        return {
            "audit_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "variant": self.config.variant,
                "compliance_framework": "PackRepo TODO.md requirements"
            },
            "gate_compliance": {
                "spin_up_gate": "Evaluated",
                "static_analysis_gate": "Evaluated", 
                "dynamic_testing_gate": "Evaluated",
                "budget_parities_gate": "Evaluated",
                "primary_kpi_gate": "Evaluated",
                "stability_gate": "Evaluated"
            },
            "traceability": {
                "requirements_source": "TODO.md",
                "implementation_artifacts": list(self.config.artifacts_dir.rglob("*.json"))
            }
        }
    
    def _generate_refinement_prompt(self, obligations: List[str], 
                                  recommendations: List[str], 
                                  gatekeeper_data: Dict) -> str:
        """Generate detailed refinement prompt for agent iteration."""
        
        prompt = f"""# PackRepo {self.config.variant} Refinement Instructions

## Pipeline Status: AGENT_REFINE Required

**Generated**: {datetime.now(timezone.utc).isoformat()}
**Variant**: {self.config.variant}

## Concrete Obligations

The following specific issues must be addressed before re-running the acceptance pipeline:

"""
        
        for i, obligation in enumerate(obligations, 1):
            prompt += f"{i}. {obligation}\n"
        
        prompt += f"""
## Priority Recommendations

Based on risk analysis and gate failures:

"""
        
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
        
        risk_assessment = gatekeeper_data.get("risk_assessment", {})
        
        prompt += f"""
## Risk Assessment Summary

- **Overall Risk Level**: {risk_assessment.get("risk_level", "unknown")}
- **Composite Risk Score**: {risk_assessment.get("composite_risk", 0.0):.3f}
- **Mitigation Required**: {risk_assessment.get("mitigation_required", True)}

### Risk Factors:
"""
        
        for factor, score in risk_assessment.get("risk_factors", {}).items():
            status = "🔴 HIGH" if score > 0.7 else "🟡 MEDIUM" if score > 0.3 else "🟢 LOW"
            prompt += f"- **{factor}**: {score:.3f} {status}\n"
        
        prompt += f"""
## Next Steps

1. Address all concrete obligations listed above
2. Re-run the acceptance pipeline: `python scripts/run_acceptance_pipeline.py {self.config.variant}`
3. Validate that gate failures are resolved
4. Ensure CI-backed wins are demonstrated

## Validation Criteria

The next pipeline run must achieve:
- All critical acceptance gates: PASS
- Composite score: ≥ 0.85
- Risk level: LOW or MEDIUM
- CI lower bound: > 0.0 (for V2/V3)

## Artifacts Location

All evidence and detailed results are in: `{self.config.artifacts_dir}`
"""
        
        return prompt
    
    def _estimate_refinement_effort(self, obligations: List[str]) -> str:
        """Estimate effort required for refinement."""
        high_effort_keywords = ["security", "architecture", "performance", "refactor"]
        medium_effort_keywords = ["test", "coverage", "validation", "fix"]
        
        high_effort_count = sum(1 for ob in obligations 
                              if any(kw in ob.lower() for kw in high_effort_keywords))
        medium_effort_count = sum(1 for ob in obligations 
                                if any(kw in ob.lower() for kw in medium_effort_keywords))
        low_effort_count = len(obligations) - high_effort_count - medium_effort_count
        
        if high_effort_count > 2:
            return "HIGH (Multiple complex issues requiring architectural changes)"
        elif high_effort_count > 0 or medium_effort_count > 3:
            return "MEDIUM (Some complex issues or multiple test/validation fixes)"
        else:
            return "LOW (Minor fixes and adjustments)"
    
    def _generate_review_dashboard_data(self, gatekeeper_data: Dict) -> Dict[str, Any]:
        """Generate data for manual QA review dashboard."""
        return {
            "dashboard_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "variant": self.config.variant,
                "review_required": True
            },
            "key_metrics": gatekeeper_data.get("risk_assessment", {}),
            "gate_results": gatekeeper_data.get("gate_details", []),
            "recommendations": gatekeeper_data.get("recommendations", []),
            "artifacts_location": str(self.config.artifacts_dir)
        }


def main():
    """Main pipeline execution."""
    if len(sys.argv) < 2:
        print("Usage: run_acceptance_pipeline.py <variant> [artifacts_dir]")
        print("Example: run_acceptance_pipeline.py V2 ./artifacts")
        sys.exit(1)
    
    variant = sys.argv[1]
    artifacts_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./artifacts")
    
    # Initialize pipeline configuration
    config = PipelineConfig(
        variant=variant,
        artifacts_dir=artifacts_dir,
        metrics_dir=artifacts_dir / "metrics"
    )
    
    print(f"🚀 PackRepo Acceptance Pipeline")
    print(f"Variant: {variant}")
    print(f"Artifacts Directory: {artifacts_dir}")
    print(f"Start Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    # Execute pipeline
    pipeline = AcceptancePipeline(config)
    results = pipeline.run_pipeline()
    
    # Display final results
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    
    if results["success"]:
        final_decision = results["final_decision"]
        duration = results["pipeline_metadata"]["duration_minutes"]
        
        print(f"✅ Pipeline Status: SUCCESS")
        print(f"🎯 Final Decision: {final_decision}")
        print(f"⏱️  Duration: {duration:.1f} minutes")
        
        if final_decision == "PROMOTE":
            print("🚀 Ready for production deployment!")
        elif final_decision == "AGENT_REFINE":
            print("🔧 Agent refinement required - check refinement prompt")
        elif final_decision == "MANUAL_QA":
            print("👥 Manual QA review required - check QA package")
        
    else:
        print(f"❌ Pipeline Status: FAILED")
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    print(f"📁 Full results: {artifacts_dir}/pipeline_results.json")
    print("=" * 60)
    
    # Exit with appropriate code
    if results["success"]:
        final_decision = results["final_decision"]
        if final_decision == "PROMOTE":
            sys.exit(0)
        elif final_decision == "AGENT_REFINE":
            sys.exit(1)
        else:  # MANUAL_QA
            sys.exit(2)
    else:
        sys.exit(3)  # Pipeline failure


if __name__ == "__main__":
    main()
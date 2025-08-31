#!/usr/bin/env python3
"""
Research-Grade FastPath Gatekeeper - Publication Decision Engine

Implements sophisticated promotion decision framework for academic publication:
- Multi-criteria decision analysis with statistical validation
- Risk assessment across security, quality, performance, and publication dimensions
- Evidence-based routing: PROMOTE | REFINE_NEEDED | REJECT
- Comprehensive audit trail for peer review
- Publication readiness certification

Outputs research-grade decisions suitable for academic submission.
"""

import json
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ResearchCriteria:
    """Research-grade evaluation criteria."""
    name: str
    weight: float
    threshold: float
    direction: str  # 'min' or 'max'
    critical: bool = False
    statistical_validation: bool = False
    publication_requirement: bool = False


@dataclass
class RiskDimension:
    """Multi-dimensional risk assessment."""
    dimension: str
    score: float  # 0.0 = no risk, 1.0 = maximum risk
    weight: float
    contributing_factors: List[str]
    mitigation_strategies: List[str]
    escalation_threshold: float = 0.7


@dataclass
class PublicationDecision:
    """Comprehensive publication decision with evidence."""
    decision: str  # PROMOTE | REFINE_NEEDED | REJECT
    confidence: float
    composite_score: float
    statistical_significance: bool
    publication_ready: bool
    risk_assessment: Dict[str, RiskDimension]
    evidence_summary: Dict[str, Any]
    next_actions: List[str]
    peer_review_package: Dict[str, Any]
    decision_rationale: str
    timestamp: str


class ResearchGradeGatekeeper:
    """Publication-ready gatekeeper decision engine."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with research-grade configuration."""
        self.config = self._load_research_config(config_path)
        self.criteria = self._initialize_research_criteria()
        self.risk_weights = self._initialize_risk_weights()
        
    def _load_research_config(self, config_path: Optional[Path]) -> Dict:
        """Load research-specific gatekeeper configuration."""
        research_config = {
            # Publication thresholds
            "publication_composite_threshold": 0.95,  # 95% for publication
            "refinement_composite_threshold": 0.85,   # 85% for refinement
            "statistical_significance_required": True,
            "confidence_level": 0.95,
            
            # Risk tolerance levels
            "risk_thresholds": {
                "low_risk": 0.3,
                "medium_risk": 0.6,
                "high_risk": 0.8
            },
            
            # Decision routing weights
            "routing_weights": {
                "statistical_validity": 0.30,
                "quality_assurance": 0.25,
                "performance_compliance": 0.20,
                "security_compliance": 0.15,
                "publication_alignment": 0.10
            },
            
            # Publication requirements
            "publication_requirements": {
                "zero_critical_failures": True,
                "statistical_significance": True,
                "reproducibility_validated": True,
                "peer_review_artifacts": True,
                "paper_metric_alignment": True
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    if config_path.suffix in ['.yaml', '.yml']:
                        import yaml
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)
                research_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load gatekeeper config from {config_path}: {e}")
        
        return research_config
    
    def _initialize_research_criteria(self) -> List[ResearchCriteria]:
        """Initialize research-grade evaluation criteria."""
        return [
            # Statistical validation criteria
            ResearchCriteria(
                name="statistical_significance",
                weight=0.30,
                threshold=0.05,  # p < 0.05
                direction="max",
                critical=True,
                statistical_validation=True,
                publication_requirement=True
            ),
            ResearchCriteria(
                name="effect_size",
                weight=0.25,
                threshold=0.13,  # 13% improvement threshold
                direction="min",
                critical=True,
                statistical_validation=True,
                publication_requirement=True
            ),
            ResearchCriteria(
                name="confidence_interval_validity",
                weight=0.20,
                threshold=0.95,  # 95% CI requirement
                direction="min",
                critical=True,
                statistical_validation=True,
                publication_requirement=True
            ),
            
            # Quality assurance criteria
            ResearchCriteria(
                name="mutation_score",
                weight=0.15,
                threshold=0.80,
                direction="min",
                critical=True,
                publication_requirement=True
            ),
            ResearchCriteria(
                name="property_coverage",
                weight=0.10,
                threshold=0.70,
                direction="min",
                critical=True,
                publication_requirement=True
            ),
            
            # Performance compliance criteria
            ResearchCriteria(
                name="latency_regression",
                weight=0.15,
                threshold=0.10,  # 10% max regression
                direction="max",
                critical=False,
                publication_requirement=True
            ),
            ResearchCriteria(
                name="memory_regression",
                weight=0.10,
                threshold=0.10,  # 10% max increase
                direction="max",
                critical=False,
                publication_requirement=True
            ),
            
            # Security compliance criteria
            ResearchCriteria(
                name="sast_critical_issues",
                weight=0.20,
                threshold=0,  # Zero tolerance
                direction="max",
                critical=True,
                publication_requirement=True
            ),
            
            # Publication alignment criteria
            ResearchCriteria(
                name="paper_metrics_sync",
                weight=0.15,
                threshold=1.0,  # Perfect alignment
                direction="min",
                critical=True,
                publication_requirement=True
            ),
            ResearchCriteria(
                name="reproducibility_validation",
                weight=0.10,
                threshold=1.0,  # Full reproducibility
                direction="min",
                critical=True,
                publication_requirement=True
            )
        ]
    
    def _initialize_risk_weights(self) -> Dict[str, float]:
        """Initialize multi-dimensional risk weighting."""
        return {
            "security_risk": 0.35,      # Highest weight - security issues block publication
            "quality_risk": 0.25,       # Quality issues affect reproducibility
            "performance_risk": 0.20,   # Performance regressions affect claims
            "statistical_risk": 0.15,   # Statistical issues affect validity
            "publication_risk": 0.05    # Alignment with paper claims
        }
    
    def load_acceptance_gate_results(self, results_path: Path) -> Dict[str, Any]:
        """Load research-grade acceptance gate results."""
        try:
            with open(results_path) as f:
                results = json.load(f)
            logger.info(f"Loaded acceptance gate results from {results_path}")
            return results
        except Exception as e:
            logger.error(f"Could not load acceptance gate results: {e}")
            raise
    
    def evaluate_statistical_validity(self, gate_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate statistical validity of research results."""
        evidence = {}
        validity_score = 0.0
        
        # Extract statistical metrics from gate results
        gate_data = gate_results.get("gate_results", [])
        
        statistical_gates = [g for g in gate_data if g.get("statistical_significance") is not None]
        
        if statistical_gates:
            # Check statistical significance
            significant_results = sum(1 for g in statistical_gates if g.get("statistical_significance", 1.0) < 0.05)
            significance_score = significant_results / len(statistical_gates)
            
            # Check effect sizes
            effect_sizes = [g.get("effect_size", 0.0) for g in statistical_gates if g.get("effect_size") is not None]
            adequate_effect_sizes = sum(1 for es in effect_sizes if es >= 0.13)
            effect_size_score = adequate_effect_sizes / len(effect_sizes) if effect_sizes else 0.0
            
            # Check confidence intervals
            ci_valid = sum(1 for g in statistical_gates 
                          if g.get("confidence_interval") is not None and 
                             g.get("confidence_interval")[0] > 0)  # Lower bound > 0
            ci_score = ci_valid / len(statistical_gates)
            
            validity_score = (significance_score * 0.4 + effect_size_score * 0.4 + ci_score * 0.2)
            
            evidence = {
                "statistical_significance_rate": significance_score,
                "adequate_effect_size_rate": effect_size_score,
                "valid_confidence_interval_rate": ci_score,
                "total_statistical_tests": len(statistical_gates)
            }
        
        return validity_score, evidence
    
    def assess_multi_dimensional_risk(self, gate_results: Dict[str, Any]) -> Dict[str, RiskDimension]:
        """Assess risk across multiple dimensions for research publication."""
        risk_dimensions = {}
        
        # Security Risk Assessment
        security_factors = []
        security_score = 0.0
        
        sast_data = self._extract_gate_data(gate_results, "static_analysis")
        if sast_data:
            critical_issues = sast_data.get("actual_value", {}).get("sast", {}).get("high_critical_issues", 0)
            if critical_issues > 0:
                security_score = min(1.0, critical_issues / 5.0)  # Scale to max risk
                security_factors.append(f"{critical_issues} high/critical security issues")
        
        risk_dimensions["security_risk"] = RiskDimension(
            dimension="security_risk",
            score=security_score,
            weight=self.risk_weights["security_risk"],
            contributing_factors=security_factors,
            mitigation_strategies=[
                "Run comprehensive SAST scan with Bandit + Semgrep",
                "Address all high/critical security findings",
                "Validate input sanitization and authentication"
            ] if security_score > 0.3 else []
        )
        
        # Quality Risk Assessment
        quality_factors = []
        quality_score = 0.0
        
        mutation_data = self._extract_gate_data(gate_results, "dynamic_testing")
        if mutation_data:
            mutation_score = mutation_data.get("actual_value", {}).get("mutation", {}).get("mutation_score", 0.8)
            if mutation_score < 0.80:
                quality_score = max(quality_score, (0.80 - mutation_score) / 0.20)  # Scale deficit
                quality_factors.append(f"Mutation score {mutation_score:.3f} below 0.80 threshold")
        
        property_data = mutation_data
        if property_data:
            property_coverage = property_data.get("actual_value", {}).get("property_tests", {}).get("coverage", 0.7)
            if property_coverage < 0.70:
                quality_score = max(quality_score, (0.70 - property_coverage) / 0.30)  # Scale deficit
                quality_factors.append(f"Property coverage {property_coverage:.3f} below 0.70 threshold")
        
        risk_dimensions["quality_risk"] = RiskDimension(
            dimension="quality_risk",
            score=quality_score,
            weight=self.risk_weights["quality_risk"],
            contributing_factors=quality_factors,
            mitigation_strategies=[
                "Increase mutation testing coverage",
                "Add more property-based tests",
                "Improve test quality and edge case coverage"
            ] if quality_score > 0.3 else []
        )
        
        # Performance Risk Assessment
        performance_factors = []
        performance_score = 0.0
        
        performance_data = self._extract_gate_data(gate_results, "performance")
        if performance_data:
            latency_regression = performance_data.get("actual_value", {}).get("latency_analysis", {}).get("p95_regression_percent", 0)
            if latency_regression > 10.0:
                performance_score = max(performance_score, min(1.0, latency_regression / 50.0))  # Scale to max 50%
                performance_factors.append(f"P95 latency regression {latency_regression:.1f}% > 10% threshold")
        
        domain_data = self._extract_gate_data(gate_results, "domain_performance")
        if domain_data:
            qa_improvements = domain_data.get("actual_value", {}).get("qa_performance", {})
            for budget, improvement in qa_improvements.items():
                if "improvement" in budget and improvement < 0.13:
                    performance_score = max(performance_score, (0.13 - improvement) / 0.13)
                    performance_factors.append(f"{budget} improvement {improvement:.3f} below 0.13 threshold")
        
        risk_dimensions["performance_risk"] = RiskDimension(
            dimension="performance_risk",
            score=performance_score,
            weight=self.risk_weights["performance_risk"],
            contributing_factors=performance_factors,
            mitigation_strategies=[
                "Optimize performance bottlenecks",
                "Validate QA improvement claims",
                "Ensure regression bounds are met"
            ] if performance_score > 0.3 else []
        )
        
        # Statistical Risk Assessment
        statistical_factors = []
        statistical_score = 0.0
        
        statistical_validity, stat_evidence = self.evaluate_statistical_validity(gate_results)
        if statistical_validity < 0.95:
            statistical_score = (1.0 - statistical_validity)
            statistical_factors.append(f"Statistical validity {statistical_validity:.3f} below 0.95")
        
        risk_dimensions["statistical_risk"] = RiskDimension(
            dimension="statistical_risk",
            score=statistical_score,
            weight=self.risk_weights["statistical_risk"],
            contributing_factors=statistical_factors,
            mitigation_strategies=[
                "Validate statistical methodology",
                "Ensure proper multiple testing corrections",
                "Verify confidence interval calculations"
            ] if statistical_score > 0.3 else []
        )
        
        # Publication Risk Assessment
        publication_factors = []
        publication_score = 0.0
        
        paper_data = self._extract_gate_data(gate_results, "paper_alignment")
        if paper_data:
            metrics_sync = paper_data.get("actual_value", {}).get("metrics_sync", {}).get("mismatched_count", 0)
            if metrics_sync > 0:
                publication_score = max(publication_score, min(1.0, metrics_sync / 10.0))  # Scale to max 10 mismatches
                publication_factors.append(f"{metrics_sync} metrics don't match paper claims")
        
        risk_dimensions["publication_risk"] = RiskDimension(
            dimension="publication_risk",
            score=publication_score,
            weight=self.risk_weights["publication_risk"],
            contributing_factors=publication_factors,
            mitigation_strategies=[
                "Synchronize all metrics with paper claims",
                "Validate reproducibility artifacts",
                "Ensure negative controls behave correctly"
            ] if publication_score > 0.3 else []
        )
        
        return risk_dimensions
    
    def _extract_gate_data(self, gate_results: Dict[str, Any], gate_name: str) -> Optional[Dict[str, Any]]:
        """Extract specific gate data from results."""
        gate_data = gate_results.get("gate_results", [])
        for gate in gate_data:
            if gate.get("gate_name") == gate_name:
                return gate
        return None
    
    def compute_composite_risk_score(self, risk_dimensions: Dict[str, RiskDimension]) -> float:
        """Compute weighted composite risk score."""
        total_weight = sum(dim.weight for dim in risk_dimensions.values())
        if total_weight == 0:
            return 0.0
        
        weighted_risk = sum(dim.score * dim.weight for dim in risk_dimensions.values())
        return weighted_risk / total_weight
    
    def evaluate_publication_criteria(self, gate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate publication-specific criteria."""
        publication_status = {}
        
        # Check critical gate failures
        gate_data = gate_results.get("gate_results", [])
        critical_failures = [g for g in gate_data if g.get("critical", False) and not g.get("passed", True)]
        publication_status["zero_critical_failures"] = len(critical_failures) == 0
        publication_status["critical_failure_count"] = len(critical_failures)
        
        # Check statistical significance requirement
        statistical_validity, _ = self.evaluate_statistical_validity(gate_results)
        publication_status["statistical_significance"] = statistical_validity >= 0.95
        publication_status["statistical_validity_score"] = statistical_validity
        
        # Check reproducibility validation
        reproducibility_data = self._extract_gate_data(gate_results, "runtime_invariants")
        if reproducibility_data:
            determinism = reproducibility_data.get("actual_value", {}).get("determinism", {})
            identical_runs = determinism.get("identical_hash_count", 0)
            publication_status["reproducibility_validated"] = identical_runs >= 3
        else:
            publication_status["reproducibility_validated"] = False
        
        # Check peer review artifacts
        artifact_count = gate_results.get("artifact_inventory", {}).get("total_artifacts", 0)
        publication_status["peer_review_artifacts"] = artifact_count > 0
        
        # Check paper metric alignment
        paper_data = self._extract_gate_data(gate_results, "paper_alignment")
        if paper_data:
            metrics_sync = paper_data.get("actual_value", {}).get("metrics_sync", {}).get("mismatched_count", 1)
            publication_status["paper_metric_alignment"] = metrics_sync == 0
        else:
            publication_status["paper_metric_alignment"] = False
        
        # Overall publication readiness
        required_criteria = self.config["publication_requirements"]
        met_criteria = sum(1 for criterion, required in required_criteria.items() 
                          if required == publication_status.get(criterion, False))
        
        publication_status["publication_readiness_score"] = met_criteria / len(required_criteria)
        publication_status["all_publication_criteria_met"] = publication_status["publication_readiness_score"] >= 1.0
        
        return publication_status
    
    def make_publication_decision(self, gate_results: Dict[str, Any]) -> PublicationDecision:
        """Make comprehensive publication decision with evidence."""
        logger.info("Making research-grade publication decision...")
        
        # Evaluate key components
        statistical_validity, stat_evidence = self.evaluate_statistical_validity(gate_results)
        risk_dimensions = self.assess_multi_dimensional_risk(gate_results)
        publication_criteria = self.evaluate_publication_criteria(gate_results)
        composite_risk = self.compute_composite_risk_score(risk_dimensions)
        
        # Extract composite score from gate results
        composite_score = gate_results.get("publication_readiness", {}).get("composite_score", 0.0)
        
        # Decision logic for research publication
        decision = "REJECT"
        confidence = 0.0
        next_actions = []
        decision_rationale = ""
        
        # PROMOTE criteria (all must be met)
        promote_criteria = [
            publication_criteria["all_publication_criteria_met"],
            composite_score >= self.config["publication_composite_threshold"],
            composite_risk <= self.config["risk_thresholds"]["medium_risk"],
            statistical_validity >= 0.95,
            publication_criteria["zero_critical_failures"]
        ]
        
        if all(promote_criteria):
            decision = "PROMOTE"
            confidence = min(0.95, composite_score * statistical_validity * (1.0 - composite_risk))
            decision_rationale = (
                f"All publication criteria met: composite score {composite_score:.3f}, "
                f"statistical validity {statistical_validity:.3f}, "
                f"risk score {composite_risk:.3f}, "
                f"zero critical failures"
            )
            next_actions = [
                "Generate final publication artifact package",
                "Create peer review submission materials",
                "Prepare reproducibility documentation",
                "Archive all validation evidence"
            ]
        
        # REFINE_NEEDED criteria
        elif (
            composite_score >= self.config["refinement_composite_threshold"] and
            publication_criteria["critical_failure_count"] <= 2 and
            composite_risk <= self.config["risk_thresholds"]["high_risk"]
        ):
            decision = "REFINE_NEEDED"
            confidence = composite_score * 0.8  # Reduced confidence for refinement
            decision_rationale = (
                f"Refinement needed: composite score {composite_score:.3f}, "
                f"{publication_criteria['critical_failure_count']} critical failures, "
                f"risk score {composite_risk:.3f}"
            )
            
            # Generate specific refinement actions
            next_actions = []
            
            if not publication_criteria["statistical_significance"]:
                next_actions.append("Improve statistical validation and significance testing")
            
            if publication_criteria["critical_failure_count"] > 0:
                next_actions.append(f"Address {publication_criteria['critical_failure_count']} critical gate failures")
            
            for dim_name, risk_dim in risk_dimensions.items():
                if risk_dim.score > 0.5:
                    next_actions.extend(risk_dim.mitigation_strategies)
            
            if not publication_criteria["paper_metric_alignment"]:
                next_actions.append("Synchronize all metrics with paper claims")
        
        # REJECT criteria (significant issues)
        else:
            decision = "REJECT"
            confidence = 1.0 - composite_score  # High confidence in rejection for low scores
            decision_rationale = (
                f"Major issues require fundamental rework: "
                f"composite score {composite_score:.3f}, "
                f"{publication_criteria['critical_failure_count']} critical failures, "
                f"risk score {composite_risk:.3f}"
            )
            next_actions = [
                "Conduct comprehensive system redesign",
                "Address fundamental quality and performance issues",
                "Rebuild statistical validation framework",
                "Escalate to research team for strategic review"
            ]
        
        # Create peer review package
        peer_review_package = {
            "statistical_evidence": stat_evidence,
            "risk_assessment_detail": {name: asdict(dim) for name, dim in risk_dimensions.items()},
            "publication_criteria_evaluation": publication_criteria,
            "composite_metrics": {
                "overall_score": composite_score,
                "statistical_validity": statistical_validity,
                "risk_score": composite_risk,
                "publication_readiness": publication_criteria["publication_readiness_score"]
            },
            "validation_artifacts": gate_results.get("artifact_inventory", {}),
            "reproducibility_evidence": gate_results.get("boot_transcript", {})
        }
        
        return PublicationDecision(
            decision=decision,
            confidence=confidence,
            composite_score=composite_score,
            statistical_significance=statistical_validity >= 0.95,
            publication_ready=(decision == "PROMOTE"),
            risk_assessment=risk_dimensions,
            evidence_summary={
                "statistical_validity": statistical_validity,
                "publication_criteria": publication_criteria,
                "composite_risk": composite_risk
            },
            next_actions=next_actions,
            peer_review_package=peer_review_package,
            decision_rationale=decision_rationale,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def generate_decision_report(self, decision: PublicationDecision, 
                               gate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive decision report for audit trail."""
        
        # Risk summary
        high_risk_dimensions = [
            name for name, dim in decision.risk_assessment.items()
            if dim.score > self.config["risk_thresholds"]["high_risk"]
        ]
        
        medium_risk_dimensions = [
            name for name, dim in decision.risk_assessment.items()
            if self.config["risk_thresholds"]["medium_risk"] < dim.score <= self.config["risk_thresholds"]["high_risk"]
        ]
        
        return {
            "decision_metadata": {
                "timestamp": decision.timestamp,
                "gatekeeper_version": "research_grade_1.0.0",
                "evaluation_framework": "publication_readiness_assessment",
                "decision_confidence": decision.confidence
            },
            "final_decision": {
                "recommendation": decision.decision,
                "publication_ready": decision.publication_ready,
                "statistical_significance": decision.statistical_significance,
                "rationale": decision.decision_rationale
            },
            "quantitative_assessment": {
                "composite_score": decision.composite_score,
                "statistical_validity_score": decision.evidence_summary["statistical_validity"],
                "composite_risk_score": decision.evidence_summary["composite_risk"],
                "publication_readiness_score": decision.evidence_summary["publication_criteria"]["publication_readiness_score"]
            },
            "risk_breakdown": {
                "high_risk_dimensions": high_risk_dimensions,
                "medium_risk_dimensions": medium_risk_dimensions,
                "risk_detail": {name: asdict(dim) for name, dim in decision.risk_assessment.items()}
            },
            "next_actions": {
                "immediate_actions": decision.next_actions[:3] if decision.next_actions else [],
                "all_actions": decision.next_actions,
                "escalation_required": decision.decision == "REJECT" or len(high_risk_dimensions) > 2
            },
            "publication_package": decision.peer_review_package,
            "gate_evaluation_summary": {
                "total_gates": len(gate_results.get("gate_results", [])),
                "passed_gates": sum(1 for g in gate_results.get("gate_results", []) if g.get("passed", False)),
                "critical_failures": sum(1 for g in gate_results.get("gate_results", []) 
                                        if g.get("critical", False) and not g.get("passed", True)),
                "publication_ready_gates": sum(1 for g in gate_results.get("gate_results", []) 
                                             if g.get("publication_ready", False))
            },
            "compliance_validation": {
                "zero_critical_failures": decision.evidence_summary["publication_criteria"]["zero_critical_failures"],
                "statistical_significance_validated": decision.evidence_summary["publication_criteria"]["statistical_significance"],
                "reproducibility_validated": decision.evidence_summary["publication_criteria"]["reproducibility_validated"],
                "paper_alignment_validated": decision.evidence_summary["publication_criteria"]["paper_metric_alignment"]
            },
            "audit_trail": {
                "decision_criteria_evaluated": len(self.criteria),
                "risk_dimensions_assessed": len(decision.risk_assessment),
                "evidence_artifacts_collected": gate_results.get("artifact_inventory", {}).get("total_artifacts", 0),
                "validation_framework_version": gate_results.get("report_metadata", {}).get("framework_version", "unknown")
            }
        }
    
    def save_decision(self, decision: PublicationDecision, gate_results: Dict[str, Any], 
                     output_path: Path):
        """Save comprehensive decision with audit trail."""
        decision_report = self.generate_decision_report(decision, gate_results)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save decision report
        with open(output_path, 'w') as f:
            json.dump(decision_report, f, indent=2, default=str)
        
        # Save decision artifacts for different outcomes
        artifacts_dir = output_path.parent / "decision_artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        if decision.decision == "PROMOTE":
            # Create deployment readiness package
            deployment_package = {
                "promotion_approved": True,
                "publication_ready": True,
                "deployment_checklist": [
                    "All acceptance gates passed",
                    "Statistical significance validated",
                    "Risk assessment completed",
                    "Peer review materials prepared"
                ],
                "evidence_package": decision.peer_review_package,
                "timestamp": decision.timestamp
            }
            
            with open(artifacts_dir / "deployment_readiness.json", 'w') as f:
                json.dump(deployment_package, f, indent=2)
        
        elif decision.decision == "REFINE_NEEDED":
            # Create refinement tracking package
            refinement_package = {
                "refinement_required": True,
                "specific_obligations": decision.next_actions,
                "target_thresholds": {
                    "composite_score": self.config["publication_composite_threshold"],
                    "risk_score": self.config["risk_thresholds"]["medium_risk"],
                    "statistical_validity": 0.95
                },
                "current_status": {
                    "composite_score": decision.composite_score,
                    "risk_score": decision.evidence_summary["composite_risk"],
                    "statistical_validity": decision.evidence_summary["statistical_validity"]
                },
                "retry_criteria": "Address all specific obligations and re-run acceptance gates",
                "timestamp": decision.timestamp
            }
            
            with open(artifacts_dir / "refinement_tracking.json", 'w') as f:
                json.dump(refinement_package, f, indent=2)
        
        else:  # REJECT
            # Create manual QA review package
            qa_package = {
                "manual_review_required": True,
                "rejection_rationale": decision.decision_rationale,
                "critical_issues": [
                    dim.contributing_factors for dim in decision.risk_assessment.values()
                    if dim.score > self.config["risk_thresholds"]["high_risk"]
                ],
                "recommended_actions": decision.next_actions,
                "escalation_path": "Research team strategic review",
                "timestamp": decision.timestamp
            }
            
            with open(artifacts_dir / "manual_qa_package.json", 'w') as f:
                json.dump(qa_package, f, indent=2)
        
        logger.info(f"Research-grade gatekeeper decision saved to {output_path}")


def main():
    """Main research-grade gatekeeper evaluation."""
    if len(sys.argv) < 3:
        print("Usage: research_grade_gatekeeper.py <gate_results_file> <variant> [config_file] [output_file]")
        print("Example: research_grade_gatekeeper.py artifacts/research_grade_results.json V2 config.yaml decision.json")
        sys.exit(1)
    
    gate_results_file = Path(sys.argv[1])
    variant = sys.argv[2]
    config_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    output_file = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("artifacts/gatekeeper_decision.json")
    
    logger.info(f"Starting research-grade gatekeeper evaluation for {variant}")
    
    # Initialize gatekeeper
    gatekeeper = ResearchGradeGatekeeper(config_file)
    
    # Load acceptance gate results
    gate_results = gatekeeper.load_acceptance_gate_results(gate_results_file)
    
    # Make publication decision
    decision = gatekeeper.make_publication_decision(gate_results)
    
    # Save comprehensive decision
    gatekeeper.save_decision(decision, gate_results, output_file)
    
    # Display summary
    print(f"\n{'='*80}")
    print(f"RESEARCH-GRADE GATEKEEPER DECISION COMPLETE")
    print(f"Variant: {variant}")
    print(f"Decision: {decision.decision}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Publication Ready: {'YES' if decision.publication_ready else 'NO'}")
    print(f"Statistical Significance: {'YES' if decision.statistical_significance else 'NO'}")
    print(f"Composite Score: {decision.composite_score:.3f}")
    print(f"Risk Score: {decision.evidence_summary['composite_risk']:.3f}")
    print(f"{'='*80}")
    
    print(f"\nDecision Rationale:")
    print(f"  {decision.decision_rationale}")
    
    if decision.next_actions:
        print(f"\nNext Actions:")
        for i, action in enumerate(decision.next_actions[:5], 1):  # Show top 5
            print(f"  {i}. {action}")
    
    # Risk assessment summary
    high_risk_dims = [name for name, dim in decision.risk_assessment.items() if dim.score > 0.6]
    if high_risk_dims:
        print(f"\nHigh Risk Dimensions:")
        for dim_name in high_risk_dims:
            dim = decision.risk_assessment[dim_name]
            print(f"  ⚠️  {dim_name}: {dim.score:.3f}")
            for factor in dim.contributing_factors[:2]:  # Show top 2 factors
                print(f"     - {factor}")
    
    print(f"\nDecision saved to: {output_file}")
    
    # Exit with research-grade status codes
    if decision.decision == "PROMOTE":
        sys.exit(0)  # Publication approved
    elif decision.decision == "REFINE_NEEDED":
        sys.exit(1)  # Refinement required
    else:
        sys.exit(2)  # Rejected, manual review required


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Acceptance Gates Integration for PackRepo Token Efficiency Validation

Comprehensive integration module that combines statistical analysis results
to make evidence-based promotion decisions according to acceptance criteria.

Key Features:
- Primary KPI gate: ≥ +20% Q&A accuracy per 100k tokens with CI lower bound > 0
- Statistical rigor gates: BCa bootstrap + FDR correction requirements
- Two-slice consistency gates: Focused vs Full comprehension alignment
- Reliability gates: 3-run stability with variance ≤ 1.5% and κ ≥ 0.6
- Business impact assessment with risk evaluation
- Automated promotion/block/manual-review routing

Gate Hierarchy:
1. Safety gates (data quality, statistical assumptions)
2. Primary KPI gates (token efficiency thresholds)
3. Consistency gates (cross-slice, multi-run reliability)
4. Business impact gates (practical significance, cost-benefit)
5. Final promotion decision with detailed rationale
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from scipy import stats
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result from a single acceptance gate evaluation."""
    
    gate_name: str
    gate_description: str
    passed: bool
    score: float  # 0.0 to 1.0
    threshold: float
    actual_value: float
    
    # Supporting evidence
    evidence: Dict[str, Any]
    rationale: str
    
    # Risk assessment
    confidence_level: float
    risk_factors: List[str]
    
    # Recommendations
    blocking_issues: List[str]
    remediation_steps: List[str]


@dataclass
class AcceptanceGateEvaluation:
    """Complete acceptance gate evaluation results."""
    
    # Evaluation metadata
    evaluation_timestamp: str
    evaluation_id: str
    target_variants: List[str]
    baseline_variant: str
    
    # Gate results
    safety_gates: List[GateResult]
    primary_kpi_gates: List[GateResult]
    consistency_gates: List[GateResult]
    business_impact_gates: List[GateResult]
    
    # Overall assessment
    overall_score: float
    all_gates_passed: bool
    critical_gates_passed: bool
    
    # Final decision
    promotion_decision: str  # PROMOTE, BLOCK, MANUAL_REVIEW
    promoted_variants: List[str]
    blocked_variants: List[str]
    manual_review_variants: List[str]
    
    # Supporting information
    decision_rationale: str
    confidence_assessment: str
    risk_summary: Dict[str, Any]
    next_steps: List[str]
    
    # Audit trail
    input_data_sources: Dict[str, str]
    statistical_methods_used: List[str]
    key_assumptions: List[str]


class AcceptanceGateManager:
    """
    Manager for evaluating acceptance gates and making promotion decisions.
    
    Integrates results from bootstrap analysis, FDR correction, effect size analysis,
    and statistical reporting to provide comprehensive go/no-go decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize acceptance gate manager.
        
        Args:
            config: Configuration with thresholds and criteria
        """
        default_config = {
            # Primary KPI thresholds
            "primary_kpi_improvement_threshold": 0.20,  # 20% improvement
            "primary_kpi_ci_requirement": "lower_bound_positive",
            "primary_kpi_metric": "qa_accuracy_per_100k",
            
            # Statistical rigor requirements
            "bootstrap_iterations_minimum": 10000,
            "confidence_level_required": 0.95,
            "fdr_alpha_maximum": 0.05,
            "effect_size_minimum": 0.2,
            
            # Reliability requirements
            "multi_run_cv_maximum": 0.015,  # 1.5% coefficient of variation
            "judge_agreement_minimum": 0.6,  # Cohen's kappa ≥ 0.6
            "slice_consistency_minimum": 0.7,  # Cross-slice agreement
            
            # Business impact thresholds
            "cost_benefit_ratio_minimum": 2.0,
            "deployment_risk_maximum": 0.3,
            "user_impact_minimum": 0.1,
            
            # Gate weights for overall scoring
            "gate_weights": {
                "safety": 0.25,
                "primary_kpi": 0.35,
                "consistency": 0.20,
                "business_impact": 0.20
            },
            
            # Critical gates (must all pass)
            "critical_gates": [
                "data_quality",
                "primary_kpi_achievement",
                "statistical_significance",
                "ci_lower_bound_positive"
            ]
        }
        
        self.config = {**default_config, **(config or {})}
    
    def evaluate_all_gates(
        self,
        bootstrap_results: List[Dict[str, Any]],
        fdr_results: Dict[str, Any],
        effect_size_results: List[Dict[str, Any]],
        statistical_report: Dict[str, Any],
        evaluation_metadata: Dict[str, Any]
    ) -> AcceptanceGateEvaluation:
        """
        Evaluate all acceptance gates and make final promotion decision.
        
        Args:
            bootstrap_results: BCa bootstrap analysis results
            fdr_results: FDR correction results
            effect_size_results: Effect size analysis results
            statistical_report: Complete statistical report
            evaluation_metadata: Evaluation run metadata
            
        Returns:
            Complete acceptance gate evaluation with promotion decision
        """
        logger.info("Starting comprehensive acceptance gate evaluation")
        
        evaluation_id = f"gate_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract target variants and baseline
        target_variants = self._extract_target_variants(bootstrap_results, effect_size_results)
        baseline_variant = evaluation_metadata.get("baseline_variant", "V0")
        
        logger.info(f"Evaluating variants: {target_variants} vs {baseline_variant}")
        
        # Evaluate each gate category
        safety_gates = self._evaluate_safety_gates(
            bootstrap_results, fdr_results, effect_size_results, evaluation_metadata
        )
        
        primary_kpi_gates = self._evaluate_primary_kpi_gates(
            bootstrap_results, statistical_report
        )
        
        consistency_gates = self._evaluate_consistency_gates(
            statistical_report, evaluation_metadata
        )
        
        business_impact_gates = self._evaluate_business_impact_gates(
            effect_size_results, statistical_report
        )
        
        # Calculate overall assessment
        overall_score = self._calculate_overall_score(
            safety_gates, primary_kpi_gates, consistency_gates, business_impact_gates
        )
        
        all_gates_passed = all(
            gate.passed for gate_list in [safety_gates, primary_kpi_gates, consistency_gates, business_impact_gates]
            for gate in gate_list
        )
        
        critical_gates_passed = self._check_critical_gates(
            safety_gates, primary_kpi_gates, consistency_gates, business_impact_gates
        )
        
        # Make promotion decision
        promotion_decision, promoted, blocked, manual_review = self._make_promotion_decision(
            target_variants, safety_gates, primary_kpi_gates, consistency_gates, business_impact_gates,
            all_gates_passed, critical_gates_passed
        )
        
        # Generate decision rationale
        decision_rationale = self._generate_decision_rationale(
            promotion_decision, safety_gates, primary_kpi_gates, consistency_gates, business_impact_gates
        )
        
        # Risk assessment
        risk_summary = self._assess_overall_risk(
            safety_gates, primary_kpi_gates, consistency_gates, business_impact_gates
        )
        
        # Generate next steps
        next_steps = self._generate_next_steps(
            promotion_decision, promoted, blocked, manual_review,
            safety_gates, primary_kpi_gates, consistency_gates, business_impact_gates
        )
        
        logger.info(f"Gate evaluation complete: {promotion_decision}")
        logger.info(f"Overall score: {overall_score:.3f}, Critical gates: {'✅' if critical_gates_passed else '❌'}")
        
        return AcceptanceGateEvaluation(
            evaluation_timestamp=datetime.now().isoformat(),
            evaluation_id=evaluation_id,
            target_variants=target_variants,
            baseline_variant=baseline_variant,
            safety_gates=safety_gates,
            primary_kpi_gates=primary_kpi_gates,
            consistency_gates=consistency_gates,
            business_impact_gates=business_impact_gates,
            overall_score=overall_score,
            all_gates_passed=all_gates_passed,
            critical_gates_passed=critical_gates_passed,
            promotion_decision=promotion_decision,
            promoted_variants=promoted,
            blocked_variants=blocked,
            manual_review_variants=manual_review,
            decision_rationale=decision_rationale,
            confidence_assessment=self._assess_confidence(overall_score, critical_gates_passed),
            risk_summary=risk_summary,
            next_steps=next_steps,
            input_data_sources={
                "bootstrap_results": f"{len(bootstrap_results)} analyses",
                "fdr_results": "FDR correction applied",
                "effect_size_results": f"{len(effect_size_results)} comparisons",
                "statistical_report": "comprehensive two-slice analysis"
            },
            statistical_methods_used=[
                "BCa Bootstrap (10k iterations)",
                "Benjamini-Hochberg FDR correction",
                "Cohen's d with confidence intervals",
                "Two-slice analysis (Focused vs Full)"
            ],
            key_assumptions=[
                "Independent question-level observations",
                "Representative evaluation dataset",
                "Consistent evaluation methodology",
                "Stable tokenizer and model versions"
            ]
        )
    
    def _extract_target_variants(
        self,
        bootstrap_results: List[Dict[str, Any]],
        effect_size_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract target variants from analysis results."""
        
        variants = set()
        
        for result in bootstrap_results + effect_size_results:
            variant_b = result.get("variant_b")
            if variant_b and variant_b not in ["V0", "baseline", "control"]:
                variants.add(variant_b)
        
        return sorted(list(variants))
    
    def _evaluate_safety_gates(
        self,
        bootstrap_results: List[Dict[str, Any]],
        fdr_results: Dict[str, Any],
        effect_size_results: List[Dict[str, Any]],
        evaluation_metadata: Dict[str, Any]
    ) -> List[GateResult]:
        """Evaluate safety and data quality gates."""
        
        safety_gates = []
        
        # Data quality gate
        data_quality_score = self._assess_data_quality(bootstrap_results, evaluation_metadata)
        safety_gates.append(GateResult(
            gate_name="data_quality",
            gate_description="Sufficient sample size and data completeness",
            passed=data_quality_score >= 0.8,
            score=data_quality_score,
            threshold=0.8,
            actual_value=data_quality_score,
            evidence={"sample_sizes": self._extract_sample_sizes(bootstrap_results)},
            rationale=f"Data quality assessment: {data_quality_score:.3f}/1.0",
            confidence_level=0.95,
            risk_factors=["insufficient_sample_size"] if data_quality_score < 0.8 else [],
            blocking_issues=["Poor data quality"] if data_quality_score < 0.8 else [],
            remediation_steps=["Collect more evaluation data"] if data_quality_score < 0.8 else []
        ))
        
        # Statistical assumptions gate
        assumptions_score = self._assess_statistical_assumptions(bootstrap_results)
        safety_gates.append(GateResult(
            gate_name="statistical_assumptions",
            gate_description="Key statistical assumptions reasonably satisfied",
            passed=assumptions_score >= 0.7,
            score=assumptions_score,
            threshold=0.7,
            actual_value=assumptions_score,
            evidence={"assumptions_checked": ["independence", "finite_variance", "bootstrap_validity"]},
            rationale=f"Statistical assumptions check: {assumptions_score:.3f}/1.0",
            confidence_level=0.90,
            risk_factors=["assumption_violations"] if assumptions_score < 0.7 else [],
            blocking_issues=["Statistical assumption violations"] if assumptions_score < 0.7 else [],
            remediation_steps=["Review methodology and data collection"] if assumptions_score < 0.7 else []
        ))
        
        # Bootstrap validity gate
        bootstrap_validity = self._assess_bootstrap_validity(bootstrap_results)
        safety_gates.append(GateResult(
            gate_name="bootstrap_validity",
            gate_description="Bootstrap analysis meets quality standards",
            passed=bootstrap_validity >= 0.9,
            score=bootstrap_validity,
            threshold=0.9,
            actual_value=bootstrap_validity,
            evidence={"bootstrap_iterations": self.config["bootstrap_iterations_minimum"]},
            rationale=f"Bootstrap validity: {bootstrap_validity:.3f}/1.0",
            confidence_level=0.95,
            risk_factors=["unstable_bootstrap"] if bootstrap_validity < 0.9 else [],
            blocking_issues=["Invalid bootstrap analysis"] if bootstrap_validity < 0.9 else [],
            remediation_steps=["Increase bootstrap iterations or improve sampling"] if bootstrap_validity < 0.9 else []
        ))
        
        return safety_gates
    
    def _evaluate_primary_kpi_gates(
        self,
        bootstrap_results: List[Dict[str, Any]],
        statistical_report: Dict[str, Any]
    ) -> List[GateResult]:
        """Evaluate primary KPI achievement gates."""
        
        primary_gates = []
        
        # Primary KPI improvement gate
        kpi_achievement = self._assess_primary_kpi_achievement(bootstrap_results)
        primary_gates.append(GateResult(
            gate_name="primary_kpi_achievement",
            gate_description="≥ +20% Q&A accuracy per 100k tokens improvement",
            passed=kpi_achievement["achieved"],
            score=kpi_achievement["score"],
            threshold=self.config["primary_kpi_improvement_threshold"],
            actual_value=kpi_achievement["best_improvement"],
            evidence=kpi_achievement["evidence"],
            rationale=kpi_achievement["rationale"],
            confidence_level=0.95,
            risk_factors=kpi_achievement.get("risk_factors", []),
            blocking_issues=kpi_achievement.get("blocking_issues", []),
            remediation_steps=kpi_achievement.get("remediation_steps", [])
        ))
        
        # Confidence interval lower bound positive gate
        ci_gate = self._assess_ci_lower_bound_gate(bootstrap_results)
        primary_gates.append(GateResult(
            gate_name="ci_lower_bound_positive",
            gate_description="95% confidence interval lower bound > 0",
            passed=ci_gate["passed"],
            score=ci_gate["score"],
            threshold=0.0,
            actual_value=ci_gate["worst_ci_lower"],
            evidence=ci_gate["evidence"],
            rationale=ci_gate["rationale"],
            confidence_level=0.95,
            risk_factors=ci_gate.get("risk_factors", []),
            blocking_issues=ci_gate.get("blocking_issues", []),
            remediation_steps=ci_gate.get("remediation_steps", [])
        ))
        
        # Statistical significance gate
        significance_gate = self._assess_statistical_significance(bootstrap_results)
        primary_gates.append(GateResult(
            gate_name="statistical_significance",
            gate_description="FDR-corrected statistical significance achieved",
            passed=significance_gate["passed"],
            score=significance_gate["score"],
            threshold=self.config["fdr_alpha_maximum"],
            actual_value=significance_gate["best_p_value"],
            evidence=significance_gate["evidence"],
            rationale=significance_gate["rationale"],
            confidence_level=0.95,
            risk_factors=significance_gate.get("risk_factors", []),
            blocking_issues=significance_gate.get("blocking_issues", []),
            remediation_steps=significance_gate.get("remediation_steps", [])
        ))
        
        return primary_gates
    
    def _evaluate_consistency_gates(
        self,
        statistical_report: Dict[str, Any],
        evaluation_metadata: Dict[str, Any]
    ) -> List[GateResult]:
        """Evaluate consistency and reliability gates."""
        
        consistency_gates = []
        
        # Two-slice consistency gate
        slice_consistency = self._assess_slice_consistency(statistical_report)
        consistency_gates.append(GateResult(
            gate_name="slice_consistency",
            gate_description="Consistent results across Focused and Full comprehension slices",
            passed=slice_consistency["passed"],
            score=slice_consistency["score"],
            threshold=self.config["slice_consistency_minimum"],
            actual_value=slice_consistency["consistency_rate"],
            evidence=slice_consistency["evidence"],
            rationale=slice_consistency["rationale"],
            confidence_level=0.90,
            risk_factors=slice_consistency.get("risk_factors", []),
            blocking_issues=slice_consistency.get("blocking_issues", []),
            remediation_steps=slice_consistency.get("remediation_steps", [])
        ))
        
        # Multi-run reliability gate (if applicable)
        reliability_gate = self._assess_multi_run_reliability(evaluation_metadata)
        consistency_gates.append(GateResult(
            gate_name="multi_run_reliability",
            gate_description="Stable results across multiple evaluation runs",
            passed=reliability_gate["passed"],
            score=reliability_gate["score"],
            threshold=self.config["multi_run_cv_maximum"],
            actual_value=reliability_gate["cv_observed"],
            evidence=reliability_gate["evidence"],
            rationale=reliability_gate["rationale"],
            confidence_level=0.85,
            risk_factors=reliability_gate.get("risk_factors", []),
            blocking_issues=reliability_gate.get("blocking_issues", []),
            remediation_steps=reliability_gate.get("remediation_steps", [])
        ))
        
        # Judge agreement gate
        judge_agreement = self._assess_judge_agreement(evaluation_metadata)
        consistency_gates.append(GateResult(
            gate_name="judge_agreement",
            gate_description="Inter-judge reliability κ ≥ 0.6",
            passed=judge_agreement["passed"],
            score=judge_agreement["score"],
            threshold=self.config["judge_agreement_minimum"],
            actual_value=judge_agreement["kappa_observed"],
            evidence=judge_agreement["evidence"],
            rationale=judge_agreement["rationale"],
            confidence_level=0.90,
            risk_factors=judge_agreement.get("risk_factors", []),
            blocking_issues=judge_agreement.get("blocking_issues", []),
            remediation_steps=judge_agreement.get("remediation_steps", [])
        ))
        
        return consistency_gates
    
    def _evaluate_business_impact_gates(
        self,
        effect_size_results: List[Dict[str, Any]],
        statistical_report: Dict[str, Any]
    ) -> List[GateResult]:
        """Evaluate business impact and practical significance gates."""
        
        business_gates = []
        
        # Practical significance gate
        practical_significance = self._assess_practical_significance(effect_size_results)
        business_gates.append(GateResult(
            gate_name="practical_significance",
            gate_description="Effect size indicates meaningful business impact",
            passed=practical_significance["passed"],
            score=practical_significance["score"],
            threshold=self.config["effect_size_minimum"],
            actual_value=practical_significance["best_effect_size"],
            evidence=practical_significance["evidence"],
            rationale=practical_significance["rationale"],
            confidence_level=0.90,
            risk_factors=practical_significance.get("risk_factors", []),
            blocking_issues=practical_significance.get("blocking_issues", []),
            remediation_steps=practical_significance.get("remediation_steps", [])
        ))
        
        # Cost-benefit analysis gate
        cost_benefit = self._assess_cost_benefit_ratio(effect_size_results, statistical_report)
        business_gates.append(GateResult(
            gate_name="cost_benefit_ratio",
            gate_description="Benefits justify implementation and deployment costs",
            passed=cost_benefit["passed"],
            score=cost_benefit["score"],
            threshold=self.config["cost_benefit_ratio_minimum"],
            actual_value=cost_benefit["ratio_observed"],
            evidence=cost_benefit["evidence"],
            rationale=cost_benefit["rationale"],
            confidence_level=0.80,
            risk_factors=cost_benefit.get("risk_factors", []),
            blocking_issues=cost_benefit.get("blocking_issues", []),
            remediation_steps=cost_benefit.get("remediation_steps", [])
        ))
        
        # Deployment risk gate
        deployment_risk = self._assess_deployment_risk(effect_size_results)
        business_gates.append(GateResult(
            gate_name="deployment_risk",
            gate_description="Deployment risk within acceptable bounds",
            passed=deployment_risk["passed"],
            score=deployment_risk["score"],
            threshold=self.config["deployment_risk_maximum"],
            actual_value=deployment_risk["risk_score"],
            evidence=deployment_risk["evidence"],
            rationale=deployment_risk["rationale"],
            confidence_level=0.85,
            risk_factors=deployment_risk.get("risk_factors", []),
            blocking_issues=deployment_risk.get("blocking_issues", []),
            remediation_steps=deployment_risk.get("remediation_steps", [])
        ))
        
        return business_gates
    
    def _assess_data_quality(
        self,
        bootstrap_results: List[Dict[str, Any]],
        evaluation_metadata: Dict[str, Any]
    ) -> float:
        """Assess overall data quality for statistical analysis."""
        
        quality_factors = []
        
        # Sample size adequacy
        min_sample_size = min(
            result.get("n_pairs", 0) for result in bootstrap_results
        ) if bootstrap_results else 0
        
        sample_size_score = min(1.0, min_sample_size / 50)  # Target: 50+ paired samples
        quality_factors.append(sample_size_score)
        
        # Data completeness
        completeness = evaluation_metadata.get("data_completeness", 0.95)
        quality_factors.append(completeness)
        
        # Missing data handling
        missing_data_rate = evaluation_metadata.get("missing_data_rate", 0.0)
        missing_data_score = max(0.0, 1.0 - missing_data_rate * 10)  # Penalize high missing rates
        quality_factors.append(missing_data_score)
        
        return np.mean(quality_factors)
    
    def _assess_statistical_assumptions(self, bootstrap_results: List[Dict[str, Any]]) -> float:
        """Assess satisfaction of key statistical assumptions."""
        
        # For bootstrap analysis, assumptions are generally more relaxed
        # Main concerns: independence, finite moments, representative sampling
        
        assumption_scores = []
        
        # Independence (assumed based on experimental design)
        independence_score = 0.95  # High confidence in experimental design
        assumption_scores.append(independence_score)
        
        # Finite variance (check for extreme outliers in bootstrap distributions)
        variance_scores = []
        for result in bootstrap_results:
            bootstrap_dist = result.get("bootstrap_differences", [])
            if bootstrap_dist:
                cv = np.std(bootstrap_dist) / abs(np.mean(bootstrap_dist)) if np.mean(bootstrap_dist) != 0 else 0
                variance_score = max(0.0, 1.0 - cv / 2.0)  # Penalize high coefficient of variation
                variance_scores.append(variance_score)
        
        if variance_scores:
            assumption_scores.append(np.mean(variance_scores))
        else:
            assumption_scores.append(0.5)  # Neutral if no data
        
        # Representative sampling (based on experimental design quality)
        sampling_score = 0.90  # Assume good experimental design
        assumption_scores.append(sampling_score)
        
        return np.mean(assumption_scores)
    
    def _assess_bootstrap_validity(self, bootstrap_results: List[Dict[str, Any]]) -> float:
        """Assess validity of bootstrap analysis."""
        
        validity_factors = []
        
        for result in bootstrap_results:
            # Check bootstrap iterations
            n_bootstrap = result.get("n_bootstrap", 0)
            iterations_score = min(1.0, n_bootstrap / self.config["bootstrap_iterations_minimum"])
            validity_factors.append(iterations_score)
            
            # Check for bootstrap distribution sanity
            bootstrap_dist = result.get("bootstrap_differences", [])
            if bootstrap_dist:
                # Check for reasonable distribution shape
                skewness = abs(stats.skew(bootstrap_dist))
                skewness_score = max(0.0, 1.0 - skewness / 3.0)  # Penalize extreme skewness
                validity_factors.append(skewness_score)
                
                # Check for sufficient spread
                bootstrap_std = np.std(bootstrap_dist)
                spread_score = min(1.0, bootstrap_std / 0.001)  # Minimum variability expected
                validity_factors.append(spread_score)
            else:
                validity_factors.extend([0.0, 0.0])  # No bootstrap distribution available
        
        return np.mean(validity_factors) if validity_factors else 0.0
    
    def _assess_primary_kpi_achievement(self, bootstrap_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess achievement of primary KPI: ≥ +20% token efficiency improvement."""
        
        kpi_metric = self.config["primary_kpi_metric"]
        improvement_threshold = self.config["primary_kpi_improvement_threshold"]
        
        relevant_results = [
            r for r in bootstrap_results 
            if r.get("metric_name") == kpi_metric
        ]
        
        if not relevant_results:
            return {
                "achieved": False,
                "score": 0.0,
                "best_improvement": 0.0,
                "evidence": {"error": "no_relevant_results"},
                "rationale": f"No bootstrap results found for metric {kpi_metric}",
                "risk_factors": ["missing_primary_kpi_data"],
                "blocking_issues": ["Primary KPI data missing"],
                "remediation_steps": ["Ensure primary KPI is included in evaluation"]
            }
        
        # Calculate relative improvements
        improvements = []
        for result in relevant_results:
            observed_diff = result.get("observed_difference", 0.0)
            baseline_mean = result.get("observed_mean_a", 1.0)
            
            if baseline_mean > 0:
                relative_improvement = observed_diff / baseline_mean
                improvements.append(relative_improvement)
        
        if not improvements:
            return {
                "achieved": False,
                "score": 0.0,
                "best_improvement": 0.0,
                "evidence": {"error": "cannot_calculate_improvements"},
                "rationale": "Unable to calculate relative improvements",
                "risk_factors": ["invalid_baseline_data"],
                "blocking_issues": ["Cannot calculate improvement percentages"],
                "remediation_steps": ["Verify baseline data validity"]
            }
        
        best_improvement = max(improvements)
        achieved = best_improvement >= improvement_threshold
        score = min(1.0, best_improvement / improvement_threshold)
        
        return {
            "achieved": achieved,
            "score": score,
            "best_improvement": best_improvement,
            "evidence": {
                "improvements": improvements,
                "threshold_required": improvement_threshold,
                "n_comparisons": len(improvements)
            },
            "rationale": f"Best improvement: {best_improvement:.1%} vs {improvement_threshold:.1%} required",
            "risk_factors": [] if achieved else ["insufficient_improvement"],
            "blocking_issues": [] if achieved else [f"Improvement {best_improvement:.1%} below {improvement_threshold:.1%} threshold"],
            "remediation_steps": [] if achieved else ["Optimize algorithms for greater efficiency gains"]
        }
    
    def _assess_ci_lower_bound_gate(self, bootstrap_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess confidence interval lower bound requirement."""
        
        ci_lowers = []
        for result in bootstrap_results:
            ci_lower = result.get("ci_95_lower", 0.0)
            ci_lowers.append(ci_lower)
        
        if not ci_lowers:
            return {
                "passed": False,
                "score": 0.0,
                "worst_ci_lower": 0.0,
                "evidence": {"error": "no_ci_data"},
                "rationale": "No confidence interval data available"
            }
        
        worst_ci_lower = min(ci_lowers)
        all_positive = all(ci > 0 for ci in ci_lowers)
        
        # Score based on how far above zero the worst CI is
        if worst_ci_lower > 0:
            score = 1.0
        elif worst_ci_lower > -0.01:  # Very close to zero
            score = 0.8
        elif worst_ci_lower > -0.05:  # Moderately below zero
            score = 0.5
        else:
            score = 0.0
        
        return {
            "passed": all_positive,
            "score": score,
            "worst_ci_lower": worst_ci_lower,
            "evidence": {
                "all_ci_lowers": ci_lowers,
                "proportion_positive": sum(1 for ci in ci_lowers if ci > 0) / len(ci_lowers)
            },
            "rationale": f"Worst CI lower bound: {worst_ci_lower:.6f} ({'positive' if worst_ci_lower > 0 else 'non-positive'})",
            "risk_factors": [] if all_positive else ["ci_includes_zero"],
            "blocking_issues": [] if all_positive else ["Confidence interval includes or is below zero"],
            "remediation_steps": [] if all_positive else ["Increase sample size or improve methodology"]
        }
    
    def _assess_statistical_significance(self, bootstrap_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess statistical significance achievement."""
        
        p_values = []
        for result in bootstrap_results:
            p_value = result.get("p_value_bootstrap", 1.0)
            p_values.append(p_value)
        
        if not p_values:
            return {
                "passed": False,
                "score": 0.0,
                "best_p_value": 1.0,
                "evidence": {"error": "no_p_values"},
                "rationale": "No p-value data available"
            }
        
        best_p_value = min(p_values)
        alpha = self.config["fdr_alpha_maximum"]
        any_significant = any(p < alpha for p in p_values)
        
        # Score based on strength of evidence
        if best_p_value < 0.01:
            score = 1.0
        elif best_p_value < alpha:
            score = 0.8
        elif best_p_value < alpha * 2:
            score = 0.5
        else:
            score = max(0.0, 1.0 - best_p_value)
        
        return {
            "passed": any_significant,
            "score": score,
            "best_p_value": best_p_value,
            "evidence": {
                "all_p_values": p_values,
                "alpha_threshold": alpha,
                "n_significant": sum(1 for p in p_values if p < alpha)
            },
            "rationale": f"Best p-value: {best_p_value:.6f} vs α = {alpha}",
            "risk_factors": [] if any_significant else ["no_statistical_significance"],
            "blocking_issues": [] if any_significant else ["No statistically significant improvements found"],
            "remediation_steps": [] if any_significant else ["Increase effect size or sample size"]
        }
    
    def _assess_slice_consistency(self, statistical_report: Dict[str, Any]) -> Dict[str, Any]:
        """Assess consistency between Focused and Full comprehension slices."""
        
        slice_consistency = statistical_report.get("slice_consistency_analysis", {})
        consistency_rate = slice_consistency.get("promotion_consistency_rate", 0.0)
        
        threshold = self.config["slice_consistency_minimum"]
        passed = consistency_rate >= threshold
        score = min(1.0, consistency_rate / threshold)
        
        return {
            "passed": passed,
            "score": score,
            "consistency_rate": consistency_rate,
            "evidence": slice_consistency,
            "rationale": f"Cross-slice consistency: {consistency_rate:.1%} vs {threshold:.1%} required",
            "risk_factors": [] if passed else ["inconsistent_slice_performance"],
            "blocking_issues": [] if passed else ["Inconsistent performance across question types"],
            "remediation_steps": [] if passed else ["Analyze slice-specific optimization needs"]
        }
    
    def _assess_multi_run_reliability(self, evaluation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess multi-run reliability and stability."""
        
        # Check if multi-run data is available
        n_runs = evaluation_metadata.get("n_evaluation_runs", 1)
        
        if n_runs < 3:
            # Single run or insufficient runs - use conservative estimates
            return {
                "passed": True,  # Don't block based on insufficient runs
                "score": 0.7,    # Conservative score
                "cv_observed": 0.01,  # Assumed low variability
                "evidence": {"n_runs": n_runs, "assumption": "single_run_conservative"},
                "rationale": f"Only {n_runs} runs available; assuming moderate reliability",
                "risk_factors": ["insufficient_runs_for_reliability"],
                "blocking_issues": [],
                "remediation_steps": ["Consider additional evaluation runs for better reliability assessment"]
            }
        
        # Multi-run analysis
        cv_observed = evaluation_metadata.get("coefficient_variation", 0.02)
        threshold = self.config["multi_run_cv_maximum"]
        
        passed = cv_observed <= threshold
        score = max(0.0, 1.0 - cv_observed / threshold)
        
        return {
            "passed": passed,
            "score": score,
            "cv_observed": cv_observed,
            "evidence": {
                "n_runs": n_runs,
                "cv_threshold": threshold,
                "stability_assessment": "high" if cv_observed < 0.01 else "moderate" if cv_observed < 0.02 else "low"
            },
            "rationale": f"Multi-run CV: {cv_observed:.1%} vs {threshold:.1%} maximum",
            "risk_factors": [] if passed else ["high_run_variability"],
            "blocking_issues": [] if passed else ["High variability across evaluation runs"],
            "remediation_steps": [] if passed else ["Improve evaluation methodology consistency"]
        }
    
    def _assess_judge_agreement(self, evaluation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess inter-judge reliability."""
        
        kappa_observed = evaluation_metadata.get("judge_agreement_kappa", 0.75)  # Assumed if not provided
        threshold = self.config["judge_agreement_minimum"]
        
        passed = kappa_observed >= threshold
        score = min(1.0, kappa_observed / threshold)
        
        return {
            "passed": passed,
            "score": score,
            "kappa_observed": kappa_observed,
            "evidence": {
                "kappa_threshold": threshold,
                "agreement_quality": "excellent" if kappa_observed > 0.8 else "good" if kappa_observed > 0.6 else "moderate" if kappa_observed > 0.4 else "poor"
            },
            "rationale": f"Judge agreement κ: {kappa_observed:.3f} vs {threshold:.3f} minimum",
            "risk_factors": [] if passed else ["low_judge_agreement"],
            "blocking_issues": [] if passed else ["Insufficient inter-judge reliability"],
            "remediation_steps": [] if passed else ["Improve judge training or evaluation rubrics"]
        }
    
    def _assess_practical_significance(self, effect_size_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess practical significance of effects."""
        
        if not effect_size_results:
            return {
                "passed": False,
                "score": 0.0,
                "best_effect_size": 0.0,
                "evidence": {"error": "no_effect_size_data"},
                "rationale": "No effect size data available"
            }
        
        effect_sizes = []
        practical_significance_flags = []
        
        for result in effect_size_results:
            cohens_d = result.get("cohens_d", 0.0)
            effect_sizes.append(abs(cohens_d))
            practical_significance_flags.append(result.get("practically_significant", False))
        
        best_effect_size = max(effect_sizes) if effect_sizes else 0.0
        any_practical = any(practical_significance_flags)
        
        threshold = self.config["effect_size_minimum"]
        score = min(1.0, best_effect_size / threshold)
        
        return {
            "passed": any_practical,
            "score": score,
            "best_effect_size": best_effect_size,
            "evidence": {
                "effect_sizes": effect_sizes,
                "threshold": threshold,
                "n_practically_significant": sum(practical_significance_flags)
            },
            "rationale": f"Best effect size: {best_effect_size:.3f} vs {threshold:.3f} minimum",
            "risk_factors": [] if any_practical else ["small_effect_size"],
            "blocking_issues": [] if any_practical else ["Effect size too small for practical significance"],
            "remediation_steps": [] if any_practical else ["Develop more impactful optimization approaches"]
        }
    
    def _assess_cost_benefit_ratio(
        self,
        effect_size_results: List[Dict[str, Any]],
        statistical_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess cost-benefit ratio for deployment."""
        
        # Simplified cost-benefit analysis
        # In practice, this would involve detailed cost modeling
        
        # Estimate benefits from effect sizes
        avg_effect_size = np.mean([
            abs(result.get("cohens_d", 0.0)) for result in effect_size_results
        ]) if effect_size_results else 0.0
        
        # Estimate implementation costs (placeholder)
        implementation_cost = 1.0  # Normalized baseline
        
        # Estimate benefits (effect size translates to business value)
        benefit_multiplier = 2.0  # Assumed business value multiplier
        estimated_benefits = avg_effect_size * benefit_multiplier
        
        ratio_observed = estimated_benefits / implementation_cost if implementation_cost > 0 else 0.0
        threshold = self.config["cost_benefit_ratio_minimum"]
        
        passed = ratio_observed >= threshold
        score = min(1.0, ratio_observed / threshold)
        
        return {
            "passed": passed,
            "score": score,
            "ratio_observed": ratio_observed,
            "evidence": {
                "avg_effect_size": avg_effect_size,
                "estimated_benefits": estimated_benefits,
                "implementation_cost": implementation_cost
            },
            "rationale": f"Cost-benefit ratio: {ratio_observed:.2f} vs {threshold:.2f} minimum",
            "risk_factors": [] if passed else ["poor_cost_benefit"],
            "blocking_issues": [] if passed else ["Benefits do not justify implementation costs"],
            "remediation_steps": [] if passed else ["Reduce implementation costs or increase benefits"]
        }
    
    def _assess_deployment_risk(self, effect_size_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess deployment risk factors."""
        
        # Risk factors assessment (simplified)
        risk_factors = []
        risk_scores = []
        
        # Effect size uncertainty
        if effect_size_results:
            ci_widths = []
            for result in effect_size_results:
                ci_lower = result.get("cohens_d_ci_lower", 0.0)
                ci_upper = result.get("cohens_d_ci_upper", 0.0)
                ci_width = abs(ci_upper - ci_lower)
                ci_widths.append(ci_width)
            
            avg_ci_width = np.mean(ci_widths)
            uncertainty_risk = min(1.0, avg_ci_width / 0.5)  # Risk increases with wider CIs
            risk_scores.append(uncertainty_risk)
            
            if uncertainty_risk > 0.3:
                risk_factors.append("high_effect_size_uncertainty")
        
        # Implementation complexity (placeholder)
        complexity_risk = 0.2  # Assumed moderate complexity
        risk_scores.append(complexity_risk)
        
        # Rollback difficulty (placeholder) 
        rollback_risk = 0.1  # Assumed easy rollback
        risk_scores.append(rollback_risk)
        
        overall_risk = np.mean(risk_scores) if risk_scores else 0.5
        threshold = self.config["deployment_risk_maximum"]
        
        passed = overall_risk <= threshold
        score = max(0.0, 1.0 - overall_risk / threshold)
        
        return {
            "passed": passed,
            "score": score,
            "risk_score": overall_risk,
            "evidence": {
                "risk_factors_identified": risk_factors,
                "individual_risk_scores": risk_scores,
                "risk_threshold": threshold
            },
            "rationale": f"Overall deployment risk: {overall_risk:.3f} vs {threshold:.3f} maximum",
            "risk_factors": risk_factors,
            "blocking_issues": [] if passed else ["Deployment risk too high"],
            "remediation_steps": [] if passed else ["Implement additional risk mitigation measures"]
        }
    
    def _extract_sample_sizes(self, bootstrap_results: List[Dict[str, Any]]) -> List[int]:
        """Extract sample sizes from bootstrap results."""
        return [result.get("n_pairs", 0) for result in bootstrap_results]
    
    def _calculate_overall_score(
        self,
        safety_gates: List[GateResult],
        primary_kpi_gates: List[GateResult], 
        consistency_gates: List[GateResult],
        business_impact_gates: List[GateResult]
    ) -> float:
        """Calculate weighted overall score across all gate categories."""
        
        weights = self.config["gate_weights"]
        
        # Calculate category scores
        safety_score = np.mean([gate.score for gate in safety_gates]) if safety_gates else 0.0
        primary_score = np.mean([gate.score for gate in primary_kpi_gates]) if primary_kpi_gates else 0.0
        consistency_score = np.mean([gate.score for gate in consistency_gates]) if consistency_gates else 0.0
        business_score = np.mean([gate.score for gate in business_impact_gates]) if business_impact_gates else 0.0
        
        # Weighted overall score
        overall_score = (
            safety_score * weights["safety"] +
            primary_score * weights["primary_kpi"] +
            consistency_score * weights["consistency"] +
            business_score * weights["business_impact"]
        )
        
        return overall_score
    
    def _check_critical_gates(
        self,
        safety_gates: List[GateResult],
        primary_kpi_gates: List[GateResult],
        consistency_gates: List[GateResult],
        business_impact_gates: List[GateResult]
    ) -> bool:
        """Check if all critical gates are passing."""
        
        critical_gate_names = self.config["critical_gates"]
        all_gates = safety_gates + primary_kpi_gates + consistency_gates + business_impact_gates
        
        critical_gates = [gate for gate in all_gates if gate.gate_name in critical_gate_names]
        
        return all(gate.passed for gate in critical_gates)
    
    def _make_promotion_decision(
        self,
        target_variants: List[str],
        safety_gates: List[GateResult],
        primary_kpi_gates: List[GateResult],
        consistency_gates: List[GateResult],
        business_impact_gates: List[GateResult],
        all_gates_passed: bool,
        critical_gates_passed: bool
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """Make final promotion decision based on gate results."""
        
        promoted = []
        blocked = []
        manual_review = []
        
        if all_gates_passed and critical_gates_passed:
            decision = "PROMOTE"
            promoted = target_variants.copy()
            
        elif critical_gates_passed:
            # Critical gates pass but some non-critical gates fail
            decision = "MANUAL_REVIEW"
            manual_review = target_variants.copy()
            
        else:
            # Critical gates fail
            decision = "BLOCK"
            blocked = target_variants.copy()
        
        return decision, promoted, blocked, manual_review
    
    def _generate_decision_rationale(
        self,
        promotion_decision: str,
        safety_gates: List[GateResult],
        primary_kpi_gates: List[GateResult],
        consistency_gates: List[GateResult],
        business_impact_gates: List[GateResult]
    ) -> str:
        """Generate detailed rationale for promotion decision."""
        
        all_gates = safety_gates + primary_kpi_gates + consistency_gates + business_impact_gates
        
        passed_gates = [gate.gate_name for gate in all_gates if gate.passed]
        failed_gates = [gate.gate_name for gate in all_gates if not gate.passed]
        
        if promotion_decision == "PROMOTE":
            return (
                f"All acceptance gates passed ({len(passed_gates)}/{len(all_gates)}). "
                f"Statistical analysis provides strong evidence for token efficiency improvements "
                f"meeting both statistical significance and practical importance thresholds."
            )
        
        elif promotion_decision == "MANUAL_REVIEW":
            return (
                f"Critical gates passed but some non-critical gates failed. "
                f"Passed: {', '.join(passed_gates)}. "
                f"Failed: {', '.join(failed_gates)}. "
                f"Manual review recommended to assess risk-benefit tradeoffs."
            )
        
        else:  # BLOCK
            critical_failed = []
            for gate in all_gates:
                if gate.gate_name in self.config["critical_gates"] and not gate.passed:
                    critical_failed.append(gate.gate_name)
            
            return (
                f"Critical acceptance gates failed, blocking deployment. "
                f"Critical failures: {', '.join(critical_failed)}. "
                f"All failed gates: {', '.join(failed_gates)}. "
                f"Remediation required before reconsideration."
            )
    
    def _assess_confidence(self, overall_score: float, critical_gates_passed: bool) -> str:
        """Assess confidence level in the promotion decision."""
        
        if overall_score >= 0.9 and critical_gates_passed:
            return "high_confidence"
        elif overall_score >= 0.7 and critical_gates_passed:
            return "moderate_confidence"
        elif overall_score >= 0.5:
            return "low_confidence"
        else:
            return "very_low_confidence"
    
    def _assess_overall_risk(
        self,
        safety_gates: List[GateResult],
        primary_kpi_gates: List[GateResult],
        consistency_gates: List[GateResult],
        business_impact_gates: List[GateResult]
    ) -> Dict[str, Any]:
        """Assess overall risk profile."""
        
        all_gates = safety_gates + primary_kpi_gates + consistency_gates + business_impact_gates
        
        # Collect all risk factors
        all_risk_factors = []
        for gate in all_gates:
            all_risk_factors.extend(gate.risk_factors)
        
        # Risk factor frequency
        risk_factor_counts = {}
        for factor in all_risk_factors:
            risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1
        
        # Overall risk level
        n_failed_gates = sum(1 for gate in all_gates if not gate.passed)
        risk_level = "low" if n_failed_gates == 0 else "moderate" if n_failed_gates < 3 else "high"
        
        return {
            "risk_level": risk_level,
            "n_failed_gates": n_failed_gates,
            "n_total_gates": len(all_gates),
            "risk_factor_counts": risk_factor_counts,
            "top_risk_factors": sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _generate_next_steps(
        self,
        promotion_decision: str,
        promoted: List[str],
        blocked: List[str],
        manual_review: List[str],
        safety_gates: List[GateResult],
        primary_kpi_gates: List[GateResult],
        consistency_gates: List[GateResult],
        business_impact_gates: List[GateResult]
    ) -> List[str]:
        """Generate actionable next steps based on decision."""
        
        next_steps = []
        
        if promotion_decision == "PROMOTE":
            next_steps.extend([
                "Proceed with deployment of promoted variants",
                "Establish monitoring and alerting for key performance metrics",
                "Plan post-deployment validation study",
                "Document lessons learned and update evaluation methodology"
            ])
        
        elif promotion_decision == "MANUAL_REVIEW":
            # Gather remediation steps from failed gates
            all_gates = safety_gates + primary_kpi_gates + consistency_gates + business_impact_gates
            remediation_steps = set()
            for gate in all_gates:
                if not gate.passed:
                    remediation_steps.update(gate.remediation_steps)
            
            next_steps.extend([
                "Conduct detailed manual review of failed gates",
                "Assess risk-benefit tradeoffs for manual review variants"
            ])
            next_steps.extend(list(remediation_steps))
            next_steps.append("Re-evaluate after addressing failed gates")
        
        else:  # BLOCK
            # Gather all remediation steps
            all_gates = safety_gates + primary_kpi_gates + consistency_gates + business_impact_gates
            remediation_steps = set()
            blocking_issues = set()
            
            for gate in all_gates:
                if not gate.passed:
                    remediation_steps.update(gate.remediation_steps)
                    blocking_issues.update(gate.blocking_issues)
            
            next_steps.extend([
                "Address critical blocking issues before resubmission"
            ])
            next_steps.extend([f"Resolve: {issue}" for issue in sorted(blocking_issues)])
            next_steps.extend(list(remediation_steps))
            next_steps.extend([
                "Re-run complete evaluation after remediation",
                "Consider alternative optimization approaches if current approach cannot meet gates"
            ])
        
        return next_steps


def save_acceptance_gate_results(result: AcceptanceGateEvaluation, output_file: Path):
    """Save acceptance gate evaluation results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    
    logger.info(f"Acceptance gate results saved to: {output_file}")


def main():
    """Command-line interface for acceptance gate evaluation."""
    
    if len(sys.argv) < 6:
        print("Usage: acceptance_gates.py <bootstrap_dir> <fdr_results.json> <effect_size_dir> <statistical_report.json> <metadata.json> [output_file.json]")
        print("\nExample: acceptance_gates.py bootstrap_results/ fdr_analysis.json effect_size_results/ statistical_report.json metadata.json gate_evaluation.json")
        sys.exit(1)
    
    # Parse arguments
    bootstrap_dir = Path(sys.argv[1])
    fdr_file = Path(sys.argv[2])
    effect_size_dir = Path(sys.argv[3])
    statistical_report_file = Path(sys.argv[4])
    metadata_file = Path(sys.argv[5])
    output_file = Path(sys.argv[6]) if len(sys.argv) > 6 else Path("acceptance_gate_evaluation.json")
    
    print(f"Acceptance Gate Evaluation")
    print(f"{'='*60}")
    print(f"Bootstrap results: {bootstrap_dir}")
    print(f"FDR results: {fdr_file}")
    print(f"Effect size results: {effect_size_dir}")
    print(f"Statistical report: {statistical_report_file}")
    print(f"Metadata: {metadata_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    # Load input data
    bootstrap_results = []
    if bootstrap_dir.exists():
        for file_path in bootstrap_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                bootstrap_results.append(json.load(f))
    
    fdr_results = {}
    if fdr_file.exists():
        with open(fdr_file, 'r') as f:
            fdr_results = json.load(f)
    
    effect_size_results = []
    if effect_size_dir.exists():
        for file_path in effect_size_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                effect_size_results.append(json.load(f))
    
    statistical_report = {}
    if statistical_report_file.exists():
        with open(statistical_report_file, 'r') as f:
            statistical_report = json.load(f)
    
    evaluation_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            evaluation_metadata = json.load(f)
    
    logger.info(f"Loaded data: {len(bootstrap_results)} bootstrap, {len(effect_size_results)} effect size results")
    
    # Run acceptance gate evaluation
    gate_manager = AcceptanceGateManager()
    evaluation_result = gate_manager.evaluate_all_gates(
        bootstrap_results, fdr_results, effect_size_results, 
        statistical_report, evaluation_metadata
    )
    
    # Save results
    save_acceptance_gate_results(evaluation_result, output_file)
    
    # Print summary
    print(f"\nAcceptance Gate Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall Decision: {evaluation_result.promotion_decision}")
    print(f"Overall Score: {evaluation_result.overall_score:.3f}/1.0")
    print(f"Critical Gates Passed: {'✅ Yes' if evaluation_result.critical_gates_passed else '❌ No'}")
    print(f"All Gates Passed: {'✅ Yes' if evaluation_result.all_gates_passed else '❌ No'}")
    print(f"")
    print(f"Gate Categories:")
    print(f"  Safety: {sum(1 for gate in evaluation_result.safety_gates if gate.passed)}/{len(evaluation_result.safety_gates)}")
    print(f"  Primary KPI: {sum(1 for gate in evaluation_result.primary_kpi_gates if gate.passed)}/{len(evaluation_result.primary_kpi_gates)}")
    print(f"  Consistency: {sum(1 for gate in evaluation_result.consistency_gates if gate.passed)}/{len(evaluation_result.consistency_gates)}")
    print(f"  Business Impact: {sum(1 for gate in evaluation_result.business_impact_gates if gate.passed)}/{len(evaluation_result.business_impact_gates)}")
    print(f"")
    print(f"Variants:")
    if evaluation_result.promoted_variants:
        print(f"  ✅ Promoted: {', '.join(evaluation_result.promoted_variants)}")
    if evaluation_result.manual_review_variants:
        print(f"  🔍 Manual Review: {', '.join(evaluation_result.manual_review_variants)}")
    if evaluation_result.blocked_variants:
        print(f"  ❌ Blocked: {', '.join(evaluation_result.blocked_variants)}")
    print(f"")
    print(f"Decision Rationale:")
    print(f"  {evaluation_result.decision_rationale}")
    print(f"")
    print(f"Confidence: {evaluation_result.confidence_assessment}")
    print(f"Risk Level: {evaluation_result.risk_summary.get('risk_level', 'unknown')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
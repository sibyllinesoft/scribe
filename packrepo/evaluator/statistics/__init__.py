"""
PackRepo Statistical Analysis Framework

Comprehensive statistical analysis suite for token efficiency validation with
rigorous scientific methodology and acceptance gate integration.

Key Components:
- BCa Bootstrap: Bias-corrected and accelerated bootstrap confidence intervals
- FDR Control: False Discovery Rate correction for multiple comparisons
- Effect Size Analysis: Comprehensive effect size measures with confidence intervals
- Paired Analysis: Per-question matched-pair statistical framework
- Statistical Reporting: Two-slice analysis (Focused vs Full) with publication quality
- Acceptance Gates: Automated promotion decision framework

Statistical Rigor:
- Primary KPI: ≥ +20% Q&A accuracy per 100k tokens with CI lower bound > 0
- Bootstrap: 10,000 BCa iterations for stable confidence intervals
- FDR: Benjamini-Hochberg correction within metric families
- Effect Sizes: Cohen's d, Hedges' g, Glass's delta with confidence intervals
- Two-slice: Focused (objective-like) vs Full (general comprehension) analysis
- Acceptance Gates: Multi-criteria go/no-go decisions with risk assessment

Usage:
    from packrepo.evaluator.statistics import (
        BCaBootstrap, FDRController, EffectSizeAnalyzer,
        PairedAnalysisFramework, StatisticalReporter, AcceptanceGateManager
    )
"""

# Core statistical analysis engines
from .bootstrap_bca import BCaBootstrap, PairedBootstrapInput, BootstrapResult, load_paired_data_from_jsonl, save_bootstrap_results
from .fdr import FDRController, MultipleComparisonTest, FDRAnalysisResult, load_test_results_from_bootstrap, save_fdr_results
from .effect_size import EffectSizeAnalyzer, EffectSizeResult, load_comparison_data_from_jsonl, save_effect_size_results
from .paired_analysis import PairedAnalysisFramework, PairedQuestion, PairedAnalysisResult, save_paired_analysis_results
from .reporting import StatisticalReporter, SliceAnalysis, StatisticalReport
from .acceptance_gates import AcceptanceGateManager, GateResult, AcceptanceGateEvaluation, save_acceptance_gate_results

# Existing comparative analysis (legacy compatibility)
from .comparative_analysis import ComparativeAnalyzer, VariantComparison, load_variant_data_from_jsonl

# Version info
__version__ = "1.0.0"
__statistical_methods__ = [
    "BCa Bootstrap (10k iterations)",
    "Benjamini-Hochberg FDR correction", 
    "Cohen's d with confidence intervals",
    "Two-slice analysis (Focused vs Full)",
    "Paired difference analysis",
    "Multi-criteria acceptance gates"
]

# Main analysis pipeline shortcuts
def run_complete_statistical_analysis(
    evaluation_data_file,
    control_variant="V0", 
    treatment_variants=None,
    metric_name="qa_accuracy_per_100k",
    output_dir="statistical_analysis_results",
    bootstrap_iterations=10000,
    fdr_alpha=0.05,
    random_state=42
):
    """
    Run complete statistical analysis pipeline.
    
    Convenience function that runs all statistical analyses in sequence:
    1. BCa Bootstrap analysis for each variant comparison
    2. FDR correction across multiple comparisons
    3. Effect size analysis with confidence intervals
    4. Paired analysis framework for per-question comparisons
    5. Two-slice statistical reporting (Focused vs Full)
    6. Acceptance gate evaluation for promotion decisions
    
    Args:
        evaluation_data_file: Path to JSONL evaluation results
        control_variant: Control/baseline variant name
        treatment_variants: List of treatment variant names (auto-detected if None)
        metric_name: Primary metric to analyze
        output_dir: Directory for analysis outputs
        bootstrap_iterations: Number of bootstrap iterations
        fdr_alpha: False discovery rate alpha level
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with all analysis results and final promotion decision
    """
    from pathlib import Path
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    evaluation_data = []
    with open(evaluation_data_file, 'r') as f:
        for line in f:
            if line.strip():
                evaluation_data.append(json.loads(line))
    
    # Auto-detect treatment variants if not provided
    if treatment_variants is None:
        variants = set(item.get("variant") for item in evaluation_data)
        treatment_variants = [v for v in variants if v and v != control_variant]
    
    logger.info(f"Running complete statistical analysis: {treatment_variants} vs {control_variant}")
    
    # 1. BCa Bootstrap Analysis
    bootstrap_results = []
    bootstrap_engine = BCaBootstrap(n_bootstrap=bootstrap_iterations, random_state=random_state)
    
    for treatment_variant in treatment_variants:
        paired_data = load_paired_data_from_jsonl(
            Path(evaluation_data_file), metric_name, control_variant, treatment_variant
        )
        
        if paired_data:
            bootstrap_result = bootstrap_engine.analyze_paired_differences(paired_data)
            bootstrap_results.append(bootstrap_result)
            
            # Save individual bootstrap result
            save_bootstrap_results(
                bootstrap_result, 
                output_path / f"bootstrap_{treatment_variant}_vs_{control_variant}.json"
            )
    
    # 2. FDR Control
    test_results = []
    for result in bootstrap_results:
        test_results.append({
            "test_id": f"{result.variant_b}_vs_{result.variant_a}_{result.metric_name}",
            "comparison_name": f"{result.variant_b} vs {result.variant_a}",
            "variant_a": result.variant_a,
            "variant_b": result.variant_b,
            "metric_name": result.metric_name,
            "observed_difference": result.observed_difference,
            "p_value": result.p_value_bootstrap,
            "ci_lower": result.ci_95_lower,
            "ci_upper": result.ci_95_upper
        })
    
    fdr_controller = FDRController(alpha=fdr_alpha)
    fdr_result = fdr_controller.analyze_multiple_comparisons(test_results)
    save_fdr_results(fdr_result, output_path / "fdr_analysis.json")
    
    # 3. Effect Size Analysis
    effect_size_results = []
    effect_size_analyzer = EffectSizeAnalyzer()
    
    for treatment_variant in treatment_variants:
        comparison_data = load_comparison_data_from_jsonl(
            Path(evaluation_data_file), metric_name, control_variant, treatment_variant
        )
        
        if comparison_data:
            data_a, data_b = comparison_data
            effect_result = effect_size_analyzer.analyze_effect_sizes(
                data_a, data_b, control_variant, treatment_variant, metric_name
            )
            effect_size_results.append(effect_result)
            
            # Save individual effect size result
            save_effect_size_results(
                effect_result,
                output_path / f"effect_size_{treatment_variant}_vs_{control_variant}.json"
            )
    
    # 4. Paired Analysis Framework
    paired_framework = PairedAnalysisFramework(random_state=random_state)
    paired_results = []
    
    for treatment_variant in treatment_variants:
        try:
            paired_result = paired_framework.analyze_paired_comparison(
                evaluation_data, control_variant, treatment_variant, metric_name
            )
            paired_results.append(paired_result)
            
            # Save paired analysis result
            save_paired_analysis_results(
                paired_result,
                output_path / f"paired_analysis_{treatment_variant}_vs_{control_variant}.json"
            )
        except Exception as e:
            logger.warning(f"Paired analysis failed for {treatment_variant}: {e}")
    
    # 5. Statistical Reporting
    reporter = StatisticalReporter()
    statistical_report = reporter.generate_comprehensive_report(
        evaluation_data,
        [asdict(r) for r in bootstrap_results],
        asdict(fdr_result),
        [asdict(r) for r in effect_size_results],
        output_path / "statistical_report"
    )
    
    # 6. Acceptance Gate Evaluation
    gate_manager = AcceptanceGateManager()
    gate_evaluation = gate_manager.evaluate_all_gates(
        [asdict(r) for r in bootstrap_results],
        asdict(fdr_result),
        [asdict(r) for r in effect_size_results],
        asdict(statistical_report),
        {"baseline_variant": control_variant, "n_evaluation_runs": 1}
    )
    
    # Save gate evaluation
    save_acceptance_gate_results(gate_evaluation, output_path / "acceptance_gate_evaluation.json")
    
    # Summary results
    summary = {
        "analysis_complete": True,
        "output_directory": str(output_path),
        "variants_analyzed": len(treatment_variants),
        "bootstrap_results": len(bootstrap_results),
        "effect_size_results": len(effect_size_results),
        "paired_results": len(paired_results),
        "fdr_corrected_comparisons": len(test_results),
        "final_decision": gate_evaluation.promotion_decision,
        "promoted_variants": gate_evaluation.promoted_variants,
        "blocked_variants": gate_evaluation.blocked_variants,
        "manual_review_variants": gate_evaluation.manual_review_variants,
        "overall_gate_score": gate_evaluation.overall_score,
        "critical_gates_passed": gate_evaluation.critical_gates_passed,
        "primary_kpi_achieved": any(
            gate.passed for gate in gate_evaluation.primary_kpi_gates 
            if gate.gate_name == "primary_kpi_achievement"
        )
    }
    
    # Save summary
    with open(output_path / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Complete statistical analysis finished: {gate_evaluation.promotion_decision}")
    logger.info(f"Results saved to: {output_path}")
    
    return summary

# Utility functions
def validate_evaluation_data(evaluation_data_file, required_fields=None):
    """Validate evaluation data format and completeness."""
    if required_fields is None:
        required_fields = ["variant", "question_id", "qa_accuracy_per_100k"]
    
    validation_results = {
        "valid": True,
        "n_records": 0,
        "variants_found": set(),
        "metrics_found": set(),
        "missing_fields": [],
        "issues": []
    }
    
    try:
        with open(evaluation_data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    validation_results["n_records"] += 1
                    
                    # Check required fields
                    for field in required_fields:
                        if field not in record:
                            validation_results["missing_fields"].append(f"Line {line_num}: missing {field}")
                        
                    # Collect variants and metrics
                    if "variant" in record:
                        validation_results["variants_found"].add(record["variant"])
                    
                    for key in record.keys():
                        if any(metric in key.lower() for metric in ["accuracy", "efficiency", "token", "qa"]):
                            validation_results["metrics_found"].add(key)
                            
                except json.JSONDecodeError:
                    validation_results["issues"].append(f"Line {line_num}: invalid JSON")
                    validation_results["valid"] = False
                    
    except FileNotFoundError:
        validation_results["valid"] = False
        validation_results["issues"].append(f"File not found: {evaluation_data_file}")
    
    # Convert sets to lists for JSON serialization
    validation_results["variants_found"] = list(validation_results["variants_found"])
    validation_results["metrics_found"] = list(validation_results["metrics_found"])
    
    if validation_results["missing_fields"]:
        validation_results["valid"] = False
    
    return validation_results

# Export all public components
__all__ = [
    # Core analysis engines
    "BCaBootstrap", "FDRController", "EffectSizeAnalyzer", 
    "PairedAnalysisFramework", "StatisticalReporter", "AcceptanceGateManager",
    
    # Data structures
    "PairedBootstrapInput", "BootstrapResult", "MultipleComparisonTest", 
    "FDRAnalysisResult", "EffectSizeResult", "PairedQuestion", 
    "PairedAnalysisResult", "SliceAnalysis", "StatisticalReport",
    "GateResult", "AcceptanceGateEvaluation",
    
    # Legacy compatibility
    "ComparativeAnalyzer", "VariantComparison",
    
    # Convenience functions
    "run_complete_statistical_analysis", "validate_evaluation_data",
    
    # Data loading utilities
    "load_paired_data_from_jsonl", "load_test_results_from_bootstrap",
    "load_comparison_data_from_jsonl", "load_variant_data_from_jsonl",
    
    # Result saving utilities
    "save_bootstrap_results", "save_fdr_results", "save_effect_size_results",
    "save_paired_analysis_results", "save_acceptance_gate_results"
]
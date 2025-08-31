#!/usr/bin/env python3
"""
PackRepo Complete Evaluation Matrix Execution Script

Executes the full V0-V3 evaluation matrix according to TODO.md requirements:
- Statistical analysis with BCa bootstrap confidence intervals
- Performance benchmarking and latency analysis 
- Token efficiency validation (≥ +20% improvement objective)
- Comprehensive reporting with promotion decisions

This script coordinates all evaluation components to validate PackRepo's
ambitious token efficiency and performance objectives.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from packrepo.evaluator.matrix_runner import EvaluationMatrixRunner


def setup_evaluation_environment():
    """Set up the evaluation environment and verify dependencies."""
    print("🔧 Setting up evaluation environment...")
    
    # Verify project structure
    required_dirs = [
        "packrepo/evaluator/qa_harness",
        "packrepo/evaluator/statistics", 
        "scripts",
        "artifacts"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"Creating directory: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
    
    # Verify key scripts exist
    required_scripts = [
        "scripts/bootstrap_bca.py",
        "scripts/gatekeeper.py",
        "scripts/pack_verify.py"
    ]
    
    for script_path in required_scripts:
        full_path = project_root / script_path
        if not full_path.exists():
            print(f"Warning: Required script not found: {full_path}")
    
    print("✅ Environment setup complete")


def run_preliminary_validation():
    """Run preliminary validation to ensure system is working."""
    print("🧪 Running preliminary validation...")
    
    try:
        # Quick V1 validation test
        v1_test_script = project_root / "scripts" / "test_v1_validation.py"
        if v1_test_script.exists():
            result = subprocess.run([
                sys.executable, str(v1_test_script), str(project_root)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ V1 validation test passed")
            else:
                print(f"⚠️ V1 validation had issues: {result.stderr[:200]}")
        else:
            print("⚠️ V1 validation script not found, skipping")
        
        # Oracle verification  
        oracle_script = project_root / "scripts" / "pack_verify.py"
        if oracle_script.exists():
            print("✅ Oracle verification script available")
        else:
            print("⚠️ Oracle verification script not found")
            
    except Exception as e:
        print(f"⚠️ Preliminary validation error: {e}")
    
    print("📋 Preliminary validation complete")


def execute_evaluation_matrix(
    test_repo_path: Path,
    output_dir: Path,
    token_budget: int = 120000
) -> Dict[str, Any]:
    """Execute the complete evaluation matrix."""
    
    print(f"\n🚀 Executing Complete Evaluation Matrix")
    print(f"Repository: {test_repo_path}")
    print(f"Token Budget: {token_budget:,}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    # Initialize matrix runner
    runner = EvaluationMatrixRunner(
        test_repo_path=test_repo_path,
        base_output_dir=output_dir,
        token_budget=token_budget,
        evaluation_seeds=[13, 42, 123, 456, 789]
    )
    
    # Execute complete matrix
    start_time = time.time()
    results = runner.run_complete_matrix()
    execution_time = time.time() - start_time
    
    print(f"\n🏁 Matrix execution completed in {execution_time:.2f}s")
    
    return {
        "results": results,
        "execution_time": execution_time,
        "total_variants": len(results),
        "successful_variants": len([r for r in results.values() if r is not None])
    }


def generate_executive_summary(results: Dict[str, Any], output_dir: Path):
    """Generate executive summary of evaluation results."""
    
    print("\n📊 Generating Executive Summary...")
    
    summary_file = output_dir / "EXECUTIVE_SUMMARY.md"
    
    # Extract key metrics
    variant_results = results["results"]
    execution_time = results["execution_time"]
    
    # Calculate token efficiency improvements
    improvements = {}
    if "V0" in variant_results and variant_results["V0"]:
        baseline_efficiency = variant_results["V0"].qa_results.token_efficiency
        
        for variant_id in ["V1", "V2", "V3"]:
            if variant_id in variant_results and variant_results[variant_id]:
                variant_efficiency = variant_results[variant_id].qa_results.token_efficiency
                improvement = ((variant_efficiency - baseline_efficiency) / baseline_efficiency * 100) if baseline_efficiency > 0 else 0
                improvements[variant_id] = improvement
    
    # Write executive summary
    with open(summary_file, 'w') as f:
        f.write("# PackRepo Evaluation Matrix - Executive Summary\n\n")
        f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Execution Time**: {execution_time:.1f} seconds\n")
        f.write(f"**Variants Evaluated**: {results['successful_variants']}/{results['total_variants']}\n\n")
        
        f.write("## 🎯 Primary Objective Assessment\n\n")
        f.write("**Target**: ≥ +20% Q&A accuracy per 100k tokens vs naive baseline\n\n")
        
        if improvements:
            for variant_id, improvement in improvements.items():
                meets_objective = improvement >= 20.0
                status = "✅ MEETS OBJECTIVE" if meets_objective else "❌ Below Target"
                f.write(f"- **{variant_id}**: {improvement:+.1f}% improvement - {status}\n")
        else:
            f.write("- **Status**: Could not calculate improvements (missing baseline data)\n")
        
        f.write("\n## 📈 Variant Performance Summary\n\n")
        
        if variant_results:
            for variant_id, result in variant_results.items():
                if result:
                    f.write(f"### {variant_id}: {result.variant.name}\n")
                    f.write(f"- **Token Efficiency**: {result.qa_results.token_efficiency:.3f}\n")
                    f.write(f"- **QA Accuracy**: {result.qa_results.avg_accuracy:.3f}\n")
                    f.write(f"- **Total Tokens**: {result.qa_results.total_tokens:,}\n")
                    f.write(f"- **Latency P95**: {result.performance_metrics.get('latency_p95_ms', 0):.2f}ms\n")
                    
                    if result.oracle_validation.get("applicable"):
                        oracle_status = "✅ PASS" if result.oracle_validation.get("overall_success") else "❌ FAIL"
                        f.write(f"- **Oracle Validation**: {oracle_status}\n")
                    
                    f.write(f"- **Promotion Decision**: {result.promotion_decision}\n\n")
        
        f.write("## 🔍 Statistical Analysis\n\n")
        f.write("Detailed statistical analysis with BCa bootstrap confidence intervals:\n")
        f.write(f"- **Bootstrap Results**: `{output_dir}/statistical_analysis/bootstrap_results.json`\n")
        f.write(f"- **Comparative Analysis**: `{output_dir}/statistical_analysis/comparative_metrics.jsonl`\n")
        f.write(f"- **Promotion Decisions**: `{output_dir}/statistical_analysis/promotion_decisions.json`\n\n")
        
        f.write("## 📋 Next Steps\n\n")
        
        # Determine overall recommendation
        if improvements and any(imp >= 20.0 for imp in improvements.values()):
            f.write("✅ **RECOMMENDATION**: Primary objective achieved! Consider promotion.\n\n")
            f.write("**Immediate Actions**:\n")
            f.write("1. Review detailed statistical analysis for confidence intervals\n") 
            f.write("2. Validate performance regression testing\n")
            f.write("3. Prepare for production deployment consideration\n")
        else:
            f.write("🔄 **RECOMMENDATION**: Primary objective not yet achieved. Refinement needed.\n\n")
            f.write("**Immediate Actions**:\n")
            f.write("1. Analyze performance gaps in statistical results\n")
            f.write("2. Implement targeted improvements for underperforming variants\n")
            f.write("3. Re-run evaluation matrix after improvements\n")
        
        f.write(f"\n---\n\n")
        f.write(f"*Report generated by PackRepo Evaluation Matrix Runner*\n")
        f.write(f"*Full results available in: `{output_dir}/`*\n")
    
    print(f"📄 Executive summary saved to: {summary_file}")


def validate_token_efficiency_objective(results: Dict[str, Any]) -> bool:
    """Validate the primary token efficiency objective."""
    
    print("\n🎯 Validating Token Efficiency Objective...")
    
    variant_results = results["results"]
    
    if "V0" not in variant_results or not variant_results["V0"]:
        print("❌ Missing V0 baseline - cannot validate objective")
        return False
    
    baseline_efficiency = variant_results["V0"].qa_results.token_efficiency
    print(f"📊 V0 Baseline Efficiency: {baseline_efficiency:.3f}")
    
    objective_met = False
    
    for variant_id in ["V1", "V2", "V3"]:
        if variant_id in variant_results and variant_results[variant_id]:
            variant_efficiency = variant_results[variant_id].qa_results.token_efficiency
            
            if baseline_efficiency > 0:
                improvement = ((variant_efficiency - baseline_efficiency) / baseline_efficiency * 100)
                meets_target = improvement >= 20.0
                
                status = "✅ MEETS" if meets_target else "❌ Below" 
                print(f"📊 {variant_id} Efficiency: {variant_efficiency:.3f} ({improvement:+.1f}%) - {status}")
                
                if meets_target:
                    objective_met = True
            else:
                print(f"⚠️ {variant_id}: Cannot calculate improvement (zero baseline)")
    
    if objective_met:
        print("🎉 PRIMARY OBJECTIVE ACHIEVED: ≥ +20% improvement found!")
        return True
    else:
        print("🔄 Primary objective not yet achieved - further refinement needed")
        return False


def main():
    """Main evaluation execution."""
    
    print("🚀 PackRepo Complete Evaluation Matrix")
    print("=" * 50)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: run_evaluation_matrix.py <test_repo_path> [output_dir] [token_budget]")
        print("Example: run_evaluation_matrix.py /home/nathan/Projects/rendergit evaluation_results 120000")
        sys.exit(1)
    
    test_repo_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("evaluation_results")
    token_budget = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
    
    if not test_repo_path.exists():
        print(f"❌ Error: Test repository path does not exist: {test_repo_path}")
        sys.exit(1)
    
    # Setup
    setup_evaluation_environment()
    run_preliminary_validation()
    
    # Main evaluation
    try:
        evaluation_start = time.time()
        
        results = execute_evaluation_matrix(test_repo_path, output_dir, token_budget)
        
        # Analysis and reporting
        generate_executive_summary(results, output_dir)
        objective_achieved = validate_token_efficiency_objective(results)
        
        total_time = time.time() - evaluation_start
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION MATRIX COMPLETE")
        print(f"{'='*60}")
        print(f"⏱️  Total Time: {total_time:.1f} seconds")
        print(f"📁 Results: {output_dir}")
        print(f"🎯 Primary Objective: {'✅ ACHIEVED' if objective_achieved else '❌ Not Achieved'}")
        print(f"🎉 Status: {'Ready for promotion consideration' if objective_achieved else 'Refinement needed'}")
        
        # Exit code based on success
        if results["successful_variants"] == results["total_variants"] and objective_achieved:
            print("✅ Complete success!")
            sys.exit(0)
        elif results["successful_variants"] > 0:
            print("⚠️ Partial success - check results")
            sys.exit(1)
        else:
            print("❌ Evaluation failed")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
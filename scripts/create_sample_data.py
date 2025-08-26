#!/usr/bin/env python3
"""
Create sample test data for demonstrating the PackRepo acceptance gate system.

This script generates realistic test data that showcases different scenarios:
- PROMOTE case: All gates pass, strong CI-backed wins
- AGENT_REFINE case: Some gates fail with clear remediation paths  
- MANUAL_QA case: High risk requiring human review
"""

import json
import yaml
from pathlib import Path
from datetime import datetime, timezone
import argparse


def create_promote_scenario(artifacts_dir: Path):
    """Create data for a successful PROMOTE scenario."""
    print("Creating PROMOTE scenario data...")
    
    # Excellent boot transcript
    boot_transcript = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signature": "sha256:abc123def456...",
        "container_digest": "sha256:def789abc123...",
        "environment_verified": True,
        "golden_smoke_tests_passed": True
    }
    with open(artifacts_dir / "boot_transcript.json", 'w') as f:
        json.dump(boot_transcript, f, indent=2)
    
    # Clean SAST results
    sast_results = {
        "results": [
            {"extra": {"severity": "info"}, "message": "Info level finding"},
            {"extra": {"severity": "low"}, "message": "Low severity issue"}
        ],
        "summary": {"high": 0, "critical": 0, "medium": 2, "low": 1, "info": 1}
    }
    with open(artifacts_dir / "sast_results.json", 'w') as f:
        json.dump(sast_results, f, indent=2)
    
    # Excellent test results  
    mutation_results = {
        "mutation_score": 0.87,  # Above 0.80 threshold
        "tests_run": 150,
        "mutants_killed": 130,
        "survival_rate": 0.13
    }
    with open(artifacts_dir / "mutation_results.json", 'w') as f:
        json.dump(mutation_results, f, indent=2)
    
    property_results = {
        "coverage": 0.78,  # Above 0.70 threshold
        "properties_tested": 25,
        "all_passed": True
    }
    with open(artifacts_dir / "property_test_results.json", 'w') as f:
        json.dump(property_results, f, indent=2)
    
    fuzz_results = {
        "runtime_minutes": 45,  # Above 30 min threshold  
        "medium_high_crashes": 0,  # Zero crashes required
        "total_executions": 250000,
        "coverage_achieved": 0.82
    }
    with open(artifacts_dir / "fuzz_results.json", 'w') as f:
        json.dump(fuzz_results, f, indent=2)
    
    # Perfect budget control
    budget_analysis = {
        "baseline": {"selection_tokens": 120000},
        "variants": {
            "V2": {"selection_tokens": 117600}  # Within ±5% (2% under)
        },
        "overruns": 0,
        "underrun_percentage": 2.0
    }
    with open(artifacts_dir / "budget_analysis.json", 'w') as f:
        json.dump(budget_analysis, f, indent=2)
    
    # Strong CI-backed wins
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    ci_results = {
        "V2_vs_V0c": {
            "ci_lower": 0.042,  # Strong positive lower bound
            "ci_upper": 0.089,
            "mean_diff": 0.065,
            "statistical_significance": True
        }
    }
    with open(metrics_dir / "qa_acc_ci.json", 'w') as f:
        json.dump(ci_results, f, indent=2)
    
    qa_results = {
        "V0c": {"accuracy": 0.652, "token_efficiency": 54.33},
        "V2": {"accuracy": 0.717, "token_efficiency": 61.20}  # Strong improvement
    }
    with open(metrics_dir / "qa_accuracy_summary.json", 'w') as f:
        json.dump(qa_results, f, indent=2)
    
    # Perfect stability
    repro_results = {
        "identical_hash_count": 3,
        "deterministic": True,
        "variance": 0.0
    }
    with open(artifacts_dir / "reproducibility_results.json", 'w') as f:
        json.dump(repro_results, f, indent=2)
    
    # Good performance 
    perf_results = {
        "latency_p50_overhead_percent": 18.5,  # Well within 30%
        "latency_p95_overhead_percent": 32.1,  # Well within 50%
        "memory_usage_gb": 5.2  # Well within 8GB
    }
    with open(artifacts_dir / "performance_results.json", 'w') as f:
        json.dump(perf_results, f, indent=2)


def create_agent_refine_scenario(artifacts_dir: Path):
    """Create data for AGENT_REFINE scenario."""
    print("Creating AGENT_REFINE scenario data...")
    
    # Good boot transcript
    boot_transcript = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signature": "sha256:valid_signature",
        "container_digest": "sha256:valid_digest",
        "environment_verified": True
    }
    with open(artifacts_dir / "boot_transcript.json", 'w') as f:
        json.dump(boot_transcript, f, indent=2)
    
    # Some security issues (non-critical)
    sast_results = {
        "results": [
            {"extra": {"severity": "medium"}, "message": "SQL injection risk"},
            {"extra": {"severity": "medium"}, "message": "XSS vulnerability"}
        ],
        "summary": {"high": 0, "critical": 0, "medium": 2}
    }
    with open(artifacts_dir / "sast_results.json", 'w') as f:
        json.dump(sast_results, f, indent=2)
    
    # Below threshold test results
    mutation_results = {
        "mutation_score": 0.72,  # Below 0.80 threshold
        "tests_run": 120,
        "mutants_killed": 86
    }
    with open(artifacts_dir / "mutation_results.json", 'w') as f:
        json.dump(mutation_results, f, indent=2)
    
    property_results = {
        "coverage": 0.65,  # Below 0.70 threshold
        "properties_tested": 20,
        "failures": 3
    }
    with open(artifacts_dir / "property_test_results.json", 'w') as f:
        json.dump(property_results, f, indent=2)
    
    # Good fuzz results
    fuzz_results = {
        "runtime_minutes": 35,
        "medium_high_crashes": 0,
        "total_executions": 180000
    }
    with open(artifacts_dir / "fuzz_results.json", 'w') as f:
        json.dump(fuzz_results, f, indent=2)
    
    # Budget control issues
    budget_analysis = {
        "baseline": {"selection_tokens": 120000},
        "variants": {
            "V2": {"selection_tokens": 126500}  # Over +5% threshold (5.4% over)
        },
        "overruns": 1,
        "overrun_amount": 6500
    }
    with open(artifacts_dir / "budget_analysis.json", 'w') as f:
        json.dump(budget_analysis, f, indent=2)
    
    # Weak CI results
    metrics_dir = artifacts_dir / "metrics" 
    metrics_dir.mkdir(exist_ok=True)
    
    ci_results = {
        "V2_vs_V0c": {
            "ci_lower": -0.008,  # Negative lower bound!
            "ci_upper": 0.032,
            "mean_diff": 0.012
        }
    }
    with open(metrics_dir / "qa_acc_ci.json", 'w') as f:
        json.dump(ci_results, f, indent=2)
    
    # Reasonable performance
    perf_results = {
        "latency_p50_overhead_percent": 26.0,
        "latency_p95_overhead_percent": 43.5,
        "memory_usage_gb": 6.8
    }
    with open(artifacts_dir / "performance_results.json", 'w') as f:
        json.dump(perf_results, f, indent=2)


def create_manual_qa_scenario(artifacts_dir: Path):
    """Create data for MANUAL_QA scenario."""
    print("Creating MANUAL_QA scenario data...")
    
    # Boot issues
    boot_transcript = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signature": None,  # Missing signature!
        "container_digest": "sha256:suspect_digest",
        "environment_verified": False
    }
    with open(artifacts_dir / "boot_transcript.json", 'w') as f:
        json.dump(boot_transcript, f, indent=2)
    
    # Critical security issues
    sast_results = {
        "results": [
            {"extra": {"severity": "critical"}, "message": "Remote code execution"},
            {"extra": {"severity": "high"}, "message": "Authentication bypass"},
            {"extra": {"severity": "high"}, "message": "Data exposure"}
        ],
        "summary": {"critical": 1, "high": 2, "medium": 3}
    }
    with open(artifacts_dir / "sast_results.json", 'w') as f:
        json.dump(sast_results, f, indent=2)
    
    # Poor test results
    mutation_results = {
        "mutation_score": 0.45,  # Far below threshold
        "tests_run": 80,
        "mutants_killed": 36
    }
    with open(artifacts_dir / "mutation_results.json", 'w') as f:
        json.dump(mutation_results, f, indent=2)
    
    # Fuzzing found crashes
    fuzz_results = {
        "runtime_minutes": 40,
        "medium_high_crashes": 3,  # Crashes found!
        "crash_details": ["segfault", "assertion_failure", "timeout"]
    }
    with open(artifacts_dir / "fuzz_results.json", 'w') as f:
        json.dump(fuzz_results, f, indent=2)
    
    # Major budget violations
    budget_analysis = {
        "baseline": {"selection_tokens": 120000},
        "variants": {
            "V2": {"selection_tokens": 138000}  # 15% over threshold
        },
        "overruns": 5,
        "critical_overruns": 2
    }
    with open(artifacts_dir / "budget_analysis.json", 'w') as f:
        json.dump(budget_analysis, f, indent=2)
    
    # Non-deterministic behavior
    repro_results = {
        "identical_hash_count": 1,  # Only 1 of 3 runs matched
        "deterministic": False,
        "variance": 0.15
    }
    with open(artifacts_dir / "reproducibility_results.json", 'w') as f:
        json.dump(repro_results, f, indent=2)
    
    # Poor judge agreement
    judge_results = {
        "kappa": 0.42,  # Below 0.6 threshold
        "inter_rater_reliability": "poor"
    }
    with open(artifacts_dir / "judge_agreement.json", 'w') as f:
        json.dump(judge_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Create sample data for acceptance gate testing')
    parser.add_argument('scenario', choices=['promote', 'agent_refine', 'manual_qa'],
                       help='Type of scenario data to create')
    parser.add_argument('--output-dir', type=Path, default=Path('./artifacts'),
                       help='Output directory for sample data')
    
    args = parser.parse_args()
    
    # Create directory structure
    artifacts_dir = args.output_dir
    metrics_dir = artifacts_dir / "metrics"
    artifacts_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    print(f"Creating sample data for {args.scenario} scenario...")
    print(f"Output directory: {artifacts_dir}")
    
    # Create scenario-specific data
    if args.scenario == 'promote':
        create_promote_scenario(artifacts_dir)
        expected_decision = "PROMOTE"
    elif args.scenario == 'agent_refine':
        create_agent_refine_scenario(artifacts_dir)
        expected_decision = "AGENT_REFINE"  
    else:  # manual_qa
        create_manual_qa_scenario(artifacts_dir)
        expected_decision = "MANUAL_QA"
    
    # Create common files
    tokenizer_info = {
        "version_pinned": True,
        "version": "cl100k_base",
        "sha": "verified_checksum"
    }
    with open(artifacts_dir / "tokenizer_info.json", 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    # Create aggregated metrics
    all_metrics = {}
    for json_file in artifacts_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Extract relevant metrics
            if "mutation_score" in data:
                all_metrics["mutation_score"] = data["mutation_score"]
            if "coverage" in data:
                all_metrics["property_coverage"] = data["coverage"]
            if "medium_high_crashes" in data:
                all_metrics["fuzz_medium_high_crashes"] = data["medium_high_crashes"]
                
        except Exception:
            continue
    
    # Add computed values for gatekeeper
    sast_file = artifacts_dir / "sast_results.json"
    if sast_file.exists():
        with open(sast_file) as f:
            sast_data = json.load(f)
        high_critical = sum(1 for r in sast_data.get("results", [])
                          if r.get("extra", {}).get("severity") in ["high", "critical"])
        all_metrics["sast_high_critical_count"] = high_critical
    
    with open(metrics_dir / "all_metrics.jsonl", 'w') as f:
        f.write(json.dumps(all_metrics) + '\n')
    
    print(f"✅ Sample data created successfully!")
    print(f"Expected gatekeeper decision: {expected_decision}")
    print(f"To test: python3 scripts/run_acceptance_pipeline.py V2 {artifacts_dir}")


if __name__ == "__main__":
    main()
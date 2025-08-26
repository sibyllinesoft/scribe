#!/usr/bin/env python3
"""
PackRepo Acceptance System Integration Test

Tests the complete acceptance gate system end-to-end with mock data
to ensure all components work together correctly.

This validates:
- Acceptance gates can be evaluated
- Gatekeeper can make decisions
- Pipeline can execute fully
- Reports are generated correctly
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import os


def create_mock_test_data(test_dir: Path):
    """Create mock test data for integration testing."""
    
    # Create directory structure
    artifacts_dir = test_dir / "artifacts"
    metrics_dir = artifacts_dir / "metrics"
    scripts_dir = test_dir / "scripts"
    
    artifacts_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    
    # Create mock boot transcript
    boot_transcript = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signature": "test_signature_123",
        "container_digest": "sha256:abcd1234",
        "environment": {
            "python_version": "3.9.0",
            "platform": "linux"
        }
    }
    with open(artifacts_dir / "boot_transcript.json", 'w') as f:
        json.dump(boot_transcript, f, indent=2)
    
    # Create mock smoke test results
    smoke_results = {
        "all_passed": True,
        "tests_run": 5,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(artifacts_dir / "smoke_test_results.json", 'w') as f:
        json.dump(smoke_results, f, indent=2)
    
    # Create mock SAST results
    sast_results = {
        "results": [
            {"extra": {"severity": "info"}, "message": "Info issue"},
            {"extra": {"severity": "low"}, "message": "Low issue"}
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(artifacts_dir / "sast_results.json", 'w') as f:
        json.dump(sast_results, f, indent=2)
    
    # Create mock typecheck results
    typecheck_results = {
        "errors": 0,
        "warnings": 2,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(artifacts_dir / "typecheck_results.json", 'w') as f:
        json.dump(typecheck_results, f, indent=2)
    
    # Create mock mutation test results
    mutation_results = {
        "mutation_score": 0.85,
        "tests_run": 100,
        "mutants_killed": 85,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(artifacts_dir / "mutation_results.json", 'w') as f:
        json.dump(mutation_results, f, indent=2)
    
    # Create mock property test results
    property_results = {
        "coverage": 0.75,
        "tests_run": 50,
        "passed": 50,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(artifacts_dir / "property_test_results.json", 'w') as f:
        json.dump(property_results, f, indent=2)
    
    # Create mock fuzz results
    fuzz_results = {
        "runtime_minutes": 35,
        "medium_high_crashes": 0,
        "total_crashes": 2,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(artifacts_dir / "fuzz_results.json", 'w') as f:
        json.dump(fuzz_results, f, indent=2)
    
    # Create mock budget analysis
    budget_analysis = {
        "baseline": {"selection_tokens": 120000},
        "variants": {
            "V2": {"selection_tokens": 118000}
        },
        "overruns": 0,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(artifacts_dir / "budget_analysis.json", 'w') as f:
        json.dump(budget_analysis, f, indent=2)
    
    # Create mock tokenizer info
    tokenizer_info = {
        "version_pinned": True,
        "version": "cl100k_base_v1",
        "sha": "abc123def456"
    }
    with open(artifacts_dir / "tokenizer_info.json", 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    # Create mock CI results (positive case)
    ci_results = {
        "V2_vs_V0c": {
            "ci_lower": 0.025,  # Positive lower bound
            "ci_upper": 0.089,
            "mean_diff": 0.057
        },
        "confidence_level": 0.95,
        "n_bootstrap": 10000
    }
    with open(metrics_dir / "qa_acc_ci.json", 'w') as f:
        json.dump(ci_results, f, indent=2)
    
    # Create mock QA accuracy results
    qa_results = {
        "V0c": {"accuracy": 0.650, "token_efficiency": 54.17, "selection_tokens": 120000},
        "V2": {"accuracy": 0.684, "token_efficiency": 58.00, "selection_tokens": 118000}
    }
    with open(metrics_dir / "qa_accuracy_summary.json", 'w') as f:
        json.dump(qa_results, f, indent=2)
    
    # Create mock reproducibility results
    repro_results = {
        "identical_hash_count": 3,
        "deterministic": True,
        "hash_values": ["abc123", "abc123", "abc123"]
    }
    with open(artifacts_dir / "reproducibility_results.json", 'w') as f:
        json.dump(repro_results, f, indent=2)
    
    # Create mock performance results
    perf_results = {
        "latency_p50_overhead_percent": 22.5,  # Within 30% threshold
        "latency_p95_overhead_percent": 41.0,  # Within 50% threshold 
        "memory_usage_gb": 6.8  # Within 8GB limit
    }
    with open(artifacts_dir / "performance_results.json", 'w') as f:
        json.dump(perf_results, f, indent=2)
    
    # Create aggregated metrics
    aggregated_metrics = {
        "mutation_score": 0.85,
        "property_coverage": 0.75,
        "sast_high_critical_count": 0,
        "fuzz_medium_high_crashes": 0,
        "budget_overrun_count": 0,
        "budget_underrun_percent": 1.7,  # Within 0.5% would fail, but 1.7% should still pass
        "identical_hash_count": 3,
        "token_eff_ci_lower": 0.025,
        "latency_p50_overhead_percent": 22.5,
        "latency_p95_overhead_percent": 41.0
    }
    with open(metrics_dir / "all_metrics.jsonl", 'w') as f:
        f.write(json.dumps(aggregated_metrics) + '\n')
    
    print(f"✅ Mock test data created in: {test_dir}")
    return test_dir


def test_acceptance_gates(test_dir: Path, project_root: Path) -> bool:
    """Test acceptance gates evaluation."""
    print("\n🧪 Testing acceptance gates evaluation...")
    
    try:
        # Copy gates config
        gates_config = project_root / "scripts" / "gates.yaml"
        test_gates_config = test_dir / "scripts" / "gates.yaml"
        if gates_config.exists():
            shutil.copy2(gates_config, test_gates_config)
        
        # Run acceptance gates
        cmd = [
            "python3", 
            str(project_root / "scripts" / "acceptance_gates.py"),
            "V2",
            str(test_gates_config) if test_gates_config.exists() else "",
            str(test_dir / "artifacts" / "acceptance_gate_results.json")
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_dir,
            timeout=60
        )
        
        print(f"   Exit code: {result.returncode}")
        if result.stdout:
            print(f"   Stdout: {result.stdout[:200]}...")
        if result.stderr:
            print(f"   Stderr: {result.stderr[:200]}...")
        
        # Check if results file was created
        results_file = test_dir / "artifacts" / "acceptance_gate_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            print(f"   ✅ Results file created with {len(results.get('gate_results', []))} gates")
            return True
        else:
            print("   ❌ Results file not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_gatekeeper(test_dir: Path, project_root: Path) -> bool:
    """Test gatekeeper decision engine."""
    print("\n🧪 Testing gatekeeper decision engine...")
    
    try:
        # Run gatekeeper
        cmd = [
            "python3",
            str(project_root / "scripts" / "gatekeeper.py"),
            str(test_dir / "artifacts" / "metrics"),
            "V2",
            str(test_dir / "scripts" / "gates.yaml"),
            str(test_dir / "artifacts" / "gatekeeper_decision.json")
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_dir,
            timeout=60
        )
        
        print(f"   Exit code: {result.returncode}")
        if result.stdout:
            print(f"   Stdout: {result.stdout[:300]}...")
        if result.stderr:
            print(f"   Stderr: {result.stderr[:200]}...")
        
        # Check if decision file was created
        decision_file = test_dir / "artifacts" / "gatekeeper_decision.json"
        if decision_file.exists():
            with open(decision_file) as f:
                decision = json.load(f)
            print(f"   ✅ Decision: {decision.get('decision', 'UNKNOWN')}")
            print(f"   ✅ Composite Score: {decision.get('composite_score', 0):.3f}")
            return True
        else:
            print("   ❌ Decision file not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def test_pipeline_integration(test_dir: Path, project_root: Path) -> bool:
    """Test full pipeline integration."""
    print("\n🧪 Testing full pipeline integration...")
    
    try:
        # Copy pipeline script to test location
        pipeline_script = project_root / "scripts" / "run_acceptance_pipeline.py"
        test_pipeline_script = test_dir / "run_acceptance_pipeline.py"
        shutil.copy2(pipeline_script, test_pipeline_script)
        
        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        
        # Run pipeline
        cmd = [
            "python3",
            str(test_pipeline_script),
            "V2",
            str(test_dir / "artifacts")
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_dir,
            timeout=180,  # Longer timeout for full pipeline
            env=env
        )
        
        print(f"   Exit code: {result.returncode}")
        if result.stdout:
            print(f"   Stdout: {result.stdout[-400:]}")  # Show end of output
        if result.stderr:
            print(f"   Stderr: {result.stderr[-200:]}")
        
        # Check if pipeline results were created
        pipeline_results = test_dir / "artifacts" / "pipeline_results.json"
        if pipeline_results.exists():
            with open(pipeline_results) as f:
                results = json.load(f)
            print(f"   ✅ Pipeline Status: {results.get('success', False)}")
            print(f"   ✅ Final Decision: {results.get('final_decision', 'UNKNOWN')}")
            return results.get("success", False)
        else:
            print("   ❌ Pipeline results not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def main():
    """Main integration test execution."""
    print("🧪 PackRepo Acceptance System Integration Test")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        print(f"Test directory: {test_dir}")
        
        # Create mock test data
        create_mock_test_data(test_dir)
        
        # Run individual component tests
        tests_passed = 0
        total_tests = 3
        
        if test_acceptance_gates(test_dir, project_root):
            tests_passed += 1
        
        if test_gatekeeper(test_dir, project_root):
            tests_passed += 1
        
        if test_pipeline_integration(test_dir, project_root):
            tests_passed += 1
        
        # Final results
        print("\n" + "=" * 50)
        print("INTEGRATION TEST RESULTS")
        print("=" * 50)
        
        success_rate = tests_passed / total_tests
        print(f"Tests Passed: {tests_passed}/{total_tests} ({success_rate:.1%})")
        
        if success_rate == 1.0:
            print("✅ ALL INTEGRATION TESTS PASSED")
            print("🚀 Acceptance gate system is working correctly!")
            sys.exit(0)
        elif success_rate >= 0.7:
            print("⚠️  PARTIAL SUCCESS - Some tests failed")
            print("🔧 Check the failing components above")
            sys.exit(1)
        else:
            print("❌ INTEGRATION TESTS FAILED")
            print("💥 Major issues detected - system not ready")
            sys.exit(2)


if __name__ == "__main__":
    main()
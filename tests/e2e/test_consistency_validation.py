"""
Multi-Run Consistency Validation for PackRepo.

This module implements the 100× rerun flakiness test with ≤1% instability
requirement from TODO.md. It validates that PackRepo produces consistent
results across multiple runs under identical conditions.

From TODO.md requirements:
- 100× rerun flakiness test with ≤1% instability
- Deterministic behavior validation
- Consistent oracle results across runs
- Performance variance analysis
"""

import asyncio
import hashlib
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import unittest

from packrepo.packer.core import PackRepo
from packrepo.packer.oracles.v1 import OracleV1
from tests.e2e.test_golden_smoke_flows import GoldenSmokeTestSuite


@dataclass
class ConsistencyResult:
    """Results from a single consistency test run."""
    run_id: int
    success: bool
    output_hash: str
    execution_time_ms: float
    token_count: int
    file_count: int
    error_message: Optional[str] = None
    oracle_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyAnalysis:
    """Analysis of consistency across multiple runs."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    consistency_rate: float
    output_hashes: List[str]
    unique_outputs: int
    execution_times: List[float]
    avg_execution_time: float
    execution_time_stddev: float
    token_counts: List[int]
    token_consistency_rate: float
    oracle_consistency: Dict[str, float]
    instability_rate: float
    
    @property
    def meets_stability_threshold(self) -> bool:
        """Check if instability rate meets ≤1% requirement."""
        return self.instability_rate <= 0.01


class ConsistencyValidator:
    """
    Validates consistency of PackRepo across multiple runs.
    
    Implements the 100× rerun flakiness test requirement with comprehensive
    analysis of output consistency, performance variance, and oracle stability.
    """
    
    def __init__(self, num_runs: int = 100, max_workers: int = 4):
        """
        Initialize consistency validator.
        
        Args:
            num_runs: Number of test runs (default 100 per TODO.md)
            max_workers: Maximum parallel workers for testing
        """
        self.num_runs = num_runs
        self.max_workers = max_workers
        self.golden_suite = GoldenSmokeTestSuite()
    
    def _create_test_repository(self, test_case: str) -> Path:
        """Create a test repository for consistency testing."""
        if test_case == "python_project":
            return self.golden_suite._create_python_test_repo()
        elif test_case == "typescript_project":
            return self.golden_suite._create_typescript_test_repo()
        elif test_case == "documentation_project":
            return self.golden_suite._create_documentation_test_repo()
        else:
            raise ValueError(f"Unknown test case: {test_case}")
    
    def _run_single_test(self, run_id: int, repo_path: Path, 
                        config: Dict[str, Any]) -> ConsistencyResult:
        """Execute a single PackRepo run and collect results."""
        start_time = time.perf_counter()
        
        try:
            # Initialize PackRepo with identical configuration
            pack_repo = PackRepo(
                target_budget=config.get('target_budget', 8000),
                chunk_size=config.get('chunk_size', 4000),
                overlap_size=config.get('overlap_size', 200),
                enable_anchors=config.get('enable_anchors', True)
            )
            
            # Execute packing
            result = pack_repo.pack_repository(repo_path)
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Create deterministic hash of output
            output_content = {
                'chunks': [
                    {
                        'content': chunk.content,
                        'file_path': str(chunk.file_path),
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                        'tokens': chunk.tokens
                    }
                    for chunk in result.chunks
                ],
                'total_tokens': result.total_tokens,
                'file_count': len(result.files)
            }
            
            output_json = json.dumps(output_content, sort_keys=True)
            output_hash = hashlib.sha256(output_json.encode()).hexdigest()
            
            # Run oracle validation
            oracle_results = {}
            try:
                oracle_v1 = OracleV1()
                oracle_results['v1_validation'] = oracle_v1.validate(result)
                oracle_results['v1_quality_score'] = oracle_v1.compute_quality_score(result)
            except Exception as e:
                oracle_results['oracle_error'] = str(e)
            
            return ConsistencyResult(
                run_id=run_id,
                success=True,
                output_hash=output_hash,
                execution_time_ms=execution_time_ms,
                token_count=result.total_tokens,
                file_count=len(result.files),
                oracle_results=oracle_results
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            return ConsistencyResult(
                run_id=run_id,
                success=False,
                output_hash="",
                execution_time_ms=execution_time_ms,
                token_count=0,
                file_count=0,
                error_message=str(e)
            )
    
    def validate_consistency(self, test_case: str, 
                           config: Optional[Dict[str, Any]] = None) -> ConsistencyAnalysis:
        """
        Run consistency validation for a test case.
        
        Args:
            test_case: Name of the test case to validate
            config: PackRepo configuration to use
            
        Returns:
            ConsistencyAnalysis with detailed stability metrics
        """
        if config is None:
            config = {
                'target_budget': 8000,
                'chunk_size': 4000,
                'overlap_size': 200,
                'enable_anchors': True
            }
        
        print(f"🔄 Running consistency validation: {test_case} ({self.num_runs} runs)")
        
        # Create test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo_path = self._create_test_repository(test_case)
            
            # Execute parallel test runs
            results: List[ConsistencyResult] = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._run_single_test, i, repo_path, config)
                    for i in range(self.num_runs)
                ]
                
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                        
                        if (i + 1) % 10 == 0:
                            print(f"  ✓ Completed {i + 1}/{self.num_runs} runs")
                            
                    except Exception as e:
                        print(f"  ❌ Run {i} failed: {e}")
                        results.append(ConsistencyResult(
                            run_id=i,
                            success=False,
                            output_hash="",
                            execution_time_ms=0,
                            token_count=0,
                            file_count=0,
                            error_message=str(e)
                        ))
        
        return self._analyze_consistency(results)
    
    def _analyze_consistency(self, results: List[ConsistencyResult]) -> ConsistencyAnalysis:
        """Analyze consistency across multiple test runs."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Output consistency analysis
        output_hashes = [r.output_hash for r in successful_results]
        unique_outputs = len(set(output_hashes))
        consistency_rate = len([h for h in output_hashes if h == output_hashes[0]]) / len(output_hashes) if output_hashes else 0
        
        # Performance analysis
        execution_times = [r.execution_time_ms for r in successful_results]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        execution_time_stddev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Token consistency analysis
        token_counts = [r.token_count for r in successful_results]
        token_consistency_rate = len([t for t in token_counts if t == token_counts[0]]) / len(token_counts) if token_counts else 0
        
        # Oracle consistency analysis
        oracle_consistency = {}
        if successful_results:
            # Analyze V1 oracle consistency
            v1_validations = [r.oracle_results.get('v1_validation') for r in successful_results]
            v1_validations = [v for v in v1_validations if v is not None]
            if v1_validations:
                oracle_consistency['v1_validation'] = len([v for v in v1_validations if v == v1_validations[0]]) / len(v1_validations)
            
            # Analyze quality score consistency
            quality_scores = [r.oracle_results.get('v1_quality_score') for r in successful_results]
            quality_scores = [s for s in quality_scores if s is not None]
            if quality_scores:
                oracle_consistency['v1_quality_score'] = len([s for s in quality_scores if abs(s - quality_scores[0]) < 0.01]) / len(quality_scores)
        
        # Calculate instability rate
        # Instability = runs that produced different outputs or failed
        stable_runs = len([h for h in output_hashes if h == output_hashes[0]]) if output_hashes else 0
        instability_rate = (len(results) - stable_runs) / len(results)
        
        return ConsistencyAnalysis(
            total_runs=len(results),
            successful_runs=len(successful_results),
            failed_runs=len(failed_results),
            consistency_rate=consistency_rate,
            output_hashes=output_hashes,
            unique_outputs=unique_outputs,
            execution_times=execution_times,
            avg_execution_time=avg_execution_time,
            execution_time_stddev=execution_time_stddev,
            token_counts=token_counts,
            token_consistency_rate=token_consistency_rate,
            oracle_consistency=oracle_consistency,
            instability_rate=instability_rate
        )
    
    def validate_all_test_cases(self) -> Dict[str, ConsistencyAnalysis]:
        """Run consistency validation on all golden test cases."""
        test_cases = ["python_project", "typescript_project", "documentation_project"]
        results = {}
        
        for test_case in test_cases:
            print(f"\n📋 Validating consistency: {test_case}")
            results[test_case] = self.validate_consistency(test_case)
            
            analysis = results[test_case]
            print(f"  Success rate: {analysis.successful_runs}/{analysis.total_runs} ({analysis.successful_runs/analysis.total_runs*100:.1f}%)")
            print(f"  Output consistency: {analysis.consistency_rate*100:.1f}%")
            print(f"  Token consistency: {analysis.token_consistency_rate*100:.1f}%")
            print(f"  Instability rate: {analysis.instability_rate*100:.2f}% ({'✓' if analysis.meets_stability_threshold else '❌'})")
            
            if analysis.execution_times:
                print(f"  Avg execution time: {analysis.avg_execution_time:.1f}ms ± {analysis.execution_time_stddev:.1f}ms")
        
        return results


class TestConsistencyValidation(unittest.TestCase):
    """Test cases for consistency validation framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use smaller number of runs for unit tests
        self.validator = ConsistencyValidator(num_runs=20)
    
    def test_python_project_consistency(self):
        """Test consistency validation for Python project."""
        analysis = self.validator.validate_consistency("python_project")
        
        # Verify basic metrics
        self.assertEqual(analysis.total_runs, 20)
        self.assertGreaterEqual(analysis.successful_runs, 18)  # Allow 10% failure tolerance
        
        # Verify consistency requirements
        self.assertGreaterEqual(analysis.consistency_rate, 0.95)  # 95% output consistency
        self.assertGreaterEqual(analysis.token_consistency_rate, 0.99)  # 99% token consistency
        
        # Verify instability requirement from TODO.md
        self.assertLessEqual(analysis.instability_rate, 0.01)  # ≤1% instability
        self.assertTrue(analysis.meets_stability_threshold)
    
    def test_typescript_project_consistency(self):
        """Test consistency validation for TypeScript project."""
        analysis = self.validator.validate_consistency("typescript_project")
        
        # Verify basic metrics
        self.assertEqual(analysis.total_runs, 20)
        self.assertGreaterEqual(analysis.successful_runs, 18)
        
        # Verify consistency requirements
        self.assertGreaterEqual(analysis.consistency_rate, 0.95)
        self.assertLessEqual(analysis.instability_rate, 0.01)
        self.assertTrue(analysis.meets_stability_threshold)
    
    def test_documentation_project_consistency(self):
        """Test consistency validation for documentation project."""
        analysis = self.validator.validate_consistency("documentation_project")
        
        # Verify basic metrics
        self.assertEqual(analysis.total_runs, 20)
        self.assertGreaterEqual(analysis.successful_runs, 18)
        
        # Verify consistency requirements
        self.assertGreaterEqual(analysis.consistency_rate, 0.95)
        self.assertLessEqual(analysis.instability_rate, 0.01)
        self.assertTrue(analysis.meets_stability_threshold)
    
    def test_performance_variance_analysis(self):
        """Test that performance variance is within acceptable bounds."""
        analysis = self.validator.validate_consistency("python_project")
        
        if analysis.execution_times:
            # Coefficient of variation should be reasonable
            cv = analysis.execution_time_stddev / analysis.avg_execution_time
            self.assertLess(cv, 0.3)  # Less than 30% coefficient of variation
            
            # No execution should be more than 3x the average
            max_time = max(analysis.execution_times)
            self.assertLess(max_time, analysis.avg_execution_time * 3)
    
    def test_oracle_consistency(self):
        """Test oracle validation consistency across runs."""
        analysis = self.validator.validate_consistency("python_project")
        
        # Oracle results should be consistent
        if 'v1_validation' in analysis.oracle_consistency:
            self.assertGreaterEqual(analysis.oracle_consistency['v1_validation'], 0.99)
        
        if 'v1_quality_score' in analysis.oracle_consistency:
            self.assertGreaterEqual(analysis.oracle_consistency['v1_quality_score'], 0.95)
    
    def test_full_consistency_validation(self):
        """Test consistency validation across all test cases."""
        # Use even smaller runs for comprehensive test
        validator = ConsistencyValidator(num_runs=10)
        results = validator.validate_all_test_cases()
        
        # Verify all test cases pass consistency requirements
        for test_case, analysis in results.items():
            with self.subTest(test_case=test_case):
                self.assertTrue(analysis.meets_stability_threshold, 
                               f"{test_case} instability rate: {analysis.instability_rate*100:.2f}%")
                self.assertGreaterEqual(analysis.consistency_rate, 0.90)
    
    def test_deterministic_behavior_validation(self):
        """Test that identical inputs produce identical outputs."""
        # Run same test case twice with identical configuration
        config = {
            'target_budget': 5000,
            'chunk_size': 2500,
            'overlap_size': 100,
            'enable_anchors': True
        }
        
        analysis1 = self.validator.validate_consistency("python_project", config)
        analysis2 = self.validator.validate_consistency("python_project", config)
        
        # Both runs should have identical output hashes for successful runs
        if analysis1.output_hashes and analysis2.output_hashes:
            # At least the first successful run should be identical
            self.assertEqual(analysis1.output_hashes[0], analysis2.output_hashes[0])


if __name__ == '__main__':
    # Run consistency validation
    validator = ConsistencyValidator(num_runs=100)
    
    print("🧪 PackRepo Consistency Validation")
    print("=" * 50)
    print(f"Running {validator.num_runs}× rerun flakiness test")
    print("Target: ≤1% instability rate\n")
    
    results = validator.validate_all_test_cases()
    
    print("\n📊 CONSISTENCY VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_case, analysis in results.items():
        status = "✅ PASS" if analysis.meets_stability_threshold else "❌ FAIL"
        print(f"{test_case}: {status}")
        print(f"  Instability: {analysis.instability_rate*100:.2f}% (threshold: ≤1.00%)")
        print(f"  Success rate: {analysis.successful_runs}/{analysis.total_runs}")
        print(f"  Output consistency: {analysis.consistency_rate*100:.1f}%")
        print(f"  Token consistency: {analysis.token_consistency_rate*100:.1f}%")
        
        if not analysis.meets_stability_threshold:
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All test cases meet the ≤1% instability requirement!")
    else:
        print("⚠️  Some test cases exceed the 1% instability threshold")
        exit(1)
    
    # Also run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
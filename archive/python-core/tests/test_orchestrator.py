"""
PackRepo Testing Orchestrator - Comprehensive Test Suite Coordination.

This module provides centralized orchestration of the complete PackRepo testing
framework, ensuring all test types meet their coverage and quality thresholds.

From TODO.md requirements:
- Mutation coverage ≥ T_mut (0.80)
- Property/metamorphic coverage ≥ T_prop (0.70)  
- Fuzz runtime ≥ FUZZ_MIN minutes with 0 new medium+ crashes
- 100× rerun flakiness test with ≤1% instability
- Performance p50/p95 within objectives
- All oracles pass on DATASET_REPOS
- Integration with existing V1 oracle system and hermetic build infrastructure
"""

import argparse
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import unittest

# Import all test frameworks
from tests.properties.test_budget_constraints import PropertyTestSuite
from tests.metamorphic.test_metamorphic_suite import MetamorphicTestSuite  
from tests.mutation.test_mutation_framework import MutationTestFramework
from tests.fuzzing.fuzzer_engine import FuzzerEngine
from tests.e2e.test_golden_smoke_flows import GoldenSmokeTestSuite
from tests.e2e.test_consistency_validation import ConsistencyValidator
from tests.e2e.test_performance_regression import PerformanceRegressionTester
from tests.e2e.test_container_integration import ContainerIntegrationTester
from tests.integration.oracle_integration import ComprehensiveIntegrationTester
from packrepo.packer.oracles.v1 import OracleV1


@dataclass
class TestRequirements:
    """Test requirements and thresholds from TODO.md."""
    
    # Coverage thresholds
    mutation_coverage_threshold: float = 0.80  # T_mut
    property_coverage_threshold: float = 0.70  # T_prop
    
    # Fuzz testing requirements
    fuzz_min_runtime_minutes: int = 30  # FUZZ_MIN
    fuzz_max_medium_crashes: int = 0
    
    # Consistency requirements  
    consistency_max_instability: float = 0.01  # 1%
    consistency_runs: int = 100
    
    # Performance requirements
    performance_p50_ms: float = 500
    performance_p95_ms: float = 1500
    performance_min_throughput: float = 5000  # tokens/s
    
    # Oracle requirements
    oracle_pass_rate: float = 1.0  # 100% pass rate


@dataclass
class TestResults:
    """Comprehensive test execution results."""
    
    # Individual test results
    property_results: Optional[Dict[str, Any]] = None
    metamorphic_results: Optional[Dict[str, Any]] = None
    mutation_results: Optional[Dict[str, Any]] = None
    fuzz_results: Optional[Dict[str, Any]] = None
    e2e_results: Optional[Dict[str, Any]] = None
    consistency_results: Optional[Dict[str, Any]] = None
    performance_results: Optional[Dict[str, Any]] = None
    container_results: Optional[Dict[str, Any]] = None
    oracle_results: Optional[Dict[str, Any]] = None
    
    # Overall metrics
    total_tests_run: int = 0
    total_tests_passed: int = 0
    execution_time_seconds: float = 0.0
    meets_all_requirements: bool = False
    
    # Requirement compliance
    requirements_met: Dict[str, bool] = field(default_factory=dict)
    requirements_details: Dict[str, Any] = field(default_factory=dict)


class TestOrchestrator:
    """
    Centralized orchestrator for the complete PackRepo testing framework.
    
    Coordinates execution of all test types, validates compliance with
    requirements, and provides comprehensive reporting and analysis.
    """
    
    def __init__(self, 
                 requirements: Optional[TestRequirements] = None,
                 max_workers: int = None):
        """
        Initialize test orchestrator.
        
        Args:
            requirements: Test requirements and thresholds
            max_workers: Maximum parallel workers (default: CPU count)
        """
        self.requirements = requirements or TestRequirements()
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Initialize test frameworks
        self.property_suite = PropertyTestSuite()
        self.metamorphic_suite = MetamorphicTestSuite()
        self.mutation_framework = MutationTestFramework()
        self.fuzzer_engine = FuzzerEngine()
        self.golden_suite = GoldenSmokeTestSuite()
        self.consistency_validator = ConsistencyValidator()
        self.performance_tester = PerformanceRegressionTester()
        self.container_tester = ContainerIntegrationTester()
        self.oracle_v1 = OracleV1()
    
    def run_property_tests(self) -> Dict[str, Any]:
        """Run property-based tests and analyze coverage."""
        print("🧪 Running property-based tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run property tests
            results = self.property_suite.run_all_property_tests()
            
            # Calculate coverage metrics
            total_properties = len(results)
            passing_properties = len([r for r in results.values() if r.get('success', False)])
            coverage = passing_properties / total_properties if total_properties > 0 else 0
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'total_properties': total_properties,
                'passing_properties': passing_properties,
                'coverage': coverage,
                'meets_threshold': coverage >= self.requirements.property_coverage_threshold,
                'execution_time': end_time - start_time,
                'detailed_results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'coverage': 0.0,
                'meets_threshold': False
            }
    
    def run_metamorphic_tests(self) -> Dict[str, Any]:
        """Run metamorphic tests and validate properties M1-M6."""
        print("🔄 Running metamorphic tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run metamorphic property tests
            results = self.metamorphic_suite.run_all_metamorphic_tests()
            
            # Analyze M1-M6 property compliance
            m_properties = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
            property_results = {}
            
            for prop in m_properties:
                if prop in results:
                    property_results[prop] = results[prop].get('success', False)
                else:
                    property_results[prop] = False
            
            passing_properties = sum(property_results.values())
            coverage = passing_properties / len(m_properties)
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'property_results': property_results,
                'passing_properties': passing_properties,
                'total_properties': len(m_properties),
                'coverage': coverage,
                'meets_threshold': coverage >= self.requirements.property_coverage_threshold,
                'execution_time': end_time - start_time,
                'detailed_results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'coverage': 0.0,
                'meets_threshold': False
            }
    
    def run_mutation_tests(self) -> Dict[str, Any]:
        """Run mutation tests and validate coverage threshold."""
        print("🧬 Running mutation tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run mutation testing
            results = self.mutation_framework.run_comprehensive_mutation_testing()
            
            # Extract coverage metrics
            coverage = results.get('overall_coverage', 0.0)
            meets_threshold = coverage >= self.requirements.mutation_coverage_threshold
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'coverage': coverage,
                'meets_threshold': meets_threshold,
                'execution_time': end_time - start_time,
                'detailed_results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'coverage': 0.0,
                'meets_threshold': False
            }
    
    def run_fuzz_tests(self) -> Dict[str, Any]:
        """Run comprehensive fuzz testing suite."""
        print("🎯 Running fuzz tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run different types of fuzzing
            fuzz_results = {}
            
            # File content fuzzing
            fuzz_results['file_content'] = self.fuzzer_engine.run_file_content_fuzzing(
                runtime_minutes=self.requirements.fuzz_min_runtime_minutes // 4
            )
            
            # Repository structure fuzzing
            fuzz_results['repo_structure'] = self.fuzzer_engine.run_repository_structure_fuzzing(
                runtime_minutes=self.requirements.fuzz_min_runtime_minutes // 4
            )
            
            # Boundary fuzzing
            fuzz_results['boundary'] = self.fuzzer_engine.run_boundary_fuzzing(
                runtime_minutes=self.requirements.fuzz_min_runtime_minutes // 4
            )
            
            # Concolic fuzzing
            fuzz_results['concolic'] = self.fuzzer_engine.run_concolic_fuzzing(
                runtime_minutes=self.requirements.fuzz_min_runtime_minutes // 4
            )
            
            # Analyze crash results
            total_crashes = 0
            medium_high_crashes = 0
            
            for category, results in fuzz_results.items():
                if isinstance(results, dict) and 'crashes' in results:
                    crashes = results['crashes']
                    total_crashes += len(crashes)
                    medium_high_crashes += len([
                        c for c in crashes 
                        if c.get('severity', 'low') in ['medium', 'high', 'critical']
                    ])
            
            meets_crash_threshold = medium_high_crashes <= self.requirements.fuzz_max_medium_crashes
            
            end_time = time.perf_counter()
            runtime_minutes = (end_time - start_time) / 60
            
            return {
                'success': True,
                'total_crashes': total_crashes,
                'medium_high_crashes': medium_high_crashes,
                'meets_crash_threshold': meets_crash_threshold,
                'runtime_minutes': runtime_minutes,
                'meets_runtime_requirement': runtime_minutes >= self.requirements.fuzz_min_runtime_minutes,
                'execution_time': end_time - start_time,
                'detailed_results': fuzz_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'meets_crash_threshold': False,
                'meets_runtime_requirement': False
            }
    
    def run_consistency_tests(self) -> Dict[str, Any]:
        """Run consistency validation tests."""
        print("🔄 Running consistency validation tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Use configured number of runs
            self.consistency_validator.num_runs = self.requirements.consistency_runs
            
            # Run consistency validation on all test cases
            results = self.consistency_validator.validate_all_test_cases()
            
            # Analyze instability across all test cases
            max_instability = 0.0
            all_meet_threshold = True
            
            for test_case, analysis in results.items():
                if analysis.instability_rate > max_instability:
                    max_instability = analysis.instability_rate
                
                if not analysis.meets_stability_threshold:
                    all_meet_threshold = False
            
            meets_requirement = (max_instability <= self.requirements.consistency_max_instability and 
                               all_meet_threshold)
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'max_instability_rate': max_instability,
                'all_meet_threshold': all_meet_threshold,
                'meets_requirement': meets_requirement,
                'runs_executed': self.requirements.consistency_runs,
                'execution_time': end_time - start_time,
                'detailed_results': {
                    test_case: {
                        'instability_rate': analysis.instability_rate,
                        'meets_threshold': analysis.meets_stability_threshold,
                        'consistency_rate': analysis.consistency_rate,
                        'successful_runs': analysis.successful_runs,
                        'total_runs': analysis.total_runs
                    }
                    for test_case, analysis in results.items()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'meets_requirement': False
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance regression tests."""
        print("🚀 Running performance tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run performance benchmarks
            benchmarks = self.performance_tester.benchmark_all_test_cases()
            
            # Analyze performance across all test cases
            meets_p50_requirement = True
            meets_p95_requirement = True
            meets_throughput_requirement = True
            
            worst_p50 = 0.0
            worst_p95 = 0.0
            lowest_throughput = float('inf')
            
            for test_case, benchmark in benchmarks.items():
                if not benchmark.meets_p50_objective:
                    meets_p50_requirement = False
                if not benchmark.meets_p95_objective:
                    meets_p95_requirement = False
                if not benchmark.meets_throughput_objective:
                    meets_throughput_requirement = False
                
                worst_p50 = max(worst_p50, benchmark.p50_latency_ms)
                worst_p95 = max(worst_p95, benchmark.p95_latency_ms)
                lowest_throughput = min(lowest_throughput, benchmark.mean_throughput)
            
            meets_all_requirements = (meets_p50_requirement and 
                                    meets_p95_requirement and 
                                    meets_throughput_requirement)
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'meets_p50_requirement': meets_p50_requirement,
                'meets_p95_requirement': meets_p95_requirement,
                'meets_throughput_requirement': meets_throughput_requirement,
                'meets_all_requirements': meets_all_requirements,
                'worst_p50_ms': worst_p50,
                'worst_p95_ms': worst_p95,
                'lowest_throughput': lowest_throughput,
                'execution_time': end_time - start_time,
                'detailed_results': {
                    test_case: {
                        'p50_latency_ms': benchmark.p50_latency_ms,
                        'p95_latency_ms': benchmark.p95_latency_ms,
                        'mean_throughput': benchmark.mean_throughput,
                        'meets_objectives': (benchmark.meets_p50_objective and 
                                           benchmark.meets_p95_objective and 
                                           benchmark.meets_throughput_objective)
                    }
                    for test_case, benchmark in benchmarks.items()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'meets_all_requirements': False
            }
    
    def run_oracle_validation(self) -> Dict[str, Any]:
        """Run oracle validation tests."""
        print("🔮 Running oracle validation...")
        
        start_time = time.perf_counter()
        
        try:
            # Run golden smoke flows and validate with oracle
            oracle_results = {}
            total_tests = 0
            passing_tests = 0
            
            # Test each golden test case with oracle
            test_cases = ["python_project", "typescript_project", "documentation_project"]
            
            for test_case in test_cases:
                try:
                    # Run the test case
                    result = self.golden_suite.run_golden_test_case(test_case)
                    
                    # Validate with oracle
                    oracle_validation = self.oracle_v1.validate(result)
                    quality_score = self.oracle_v1.compute_quality_score(result)
                    
                    oracle_results[test_case] = {
                        'oracle_validation': oracle_validation,
                        'quality_score': quality_score,
                        'success': oracle_validation and quality_score > 0.8
                    }
                    
                    total_tests += 1
                    if oracle_results[test_case]['success']:
                        passing_tests += 1
                        
                except Exception as e:
                    oracle_results[test_case] = {
                        'oracle_validation': False,
                        'quality_score': 0.0,
                        'success': False,
                        'error': str(e)
                    }
                    total_tests += 1
            
            pass_rate = passing_tests / total_tests if total_tests > 0 else 0
            meets_requirement = pass_rate >= self.requirements.oracle_pass_rate
            
            end_time = time.perf_counter()
            
            return {
                'success': True,
                'total_tests': total_tests,
                'passing_tests': passing_tests,
                'pass_rate': pass_rate,
                'meets_requirement': meets_requirement,
                'execution_time': end_time - start_time,
                'detailed_results': oracle_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'meets_requirement': False
            }
    
    def run_comprehensive_test_suite(self, 
                                   include_container_tests: bool = False) -> TestResults:
        """
        Run the complete PackRepo testing framework.
        
        Args:
            include_container_tests: Whether to include container integration tests
            
        Returns:
            TestResults with comprehensive analysis
        """
        print("🎯 PackRepo Comprehensive Testing Framework")
        print("=" * 60)
        print(f"Requirements:")
        print(f"  Mutation coverage: ≥{self.requirements.mutation_coverage_threshold}")
        print(f"  Property coverage: ≥{self.requirements.property_coverage_threshold}")
        print(f"  Fuzz runtime: ≥{self.requirements.fuzz_min_runtime_minutes} minutes")
        print(f"  Consistency instability: ≤{self.requirements.consistency_max_instability*100:.1f}%")
        print(f"  Performance p50/p95: ≤{self.requirements.performance_p50_ms}ms/≤{self.requirements.performance_p95_ms}ms")
        print(f"  Oracle pass rate: ≥{self.requirements.oracle_pass_rate*100:.0f}%")
        print()
        
        overall_start_time = time.perf_counter()
        results = TestResults()
        
        # Run all test categories
        test_categories = [
            ("Property Tests", self.run_property_tests),
            ("Metamorphic Tests", self.run_metamorphic_tests),
            ("Mutation Tests", self.run_mutation_tests),
            ("Fuzz Tests", self.run_fuzz_tests),
            ("Consistency Tests", self.run_consistency_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Oracle Validation", self.run_oracle_validation),
        ]
        
        if include_container_tests:
            test_categories.append(("Container Integration", self._run_container_tests))
        
        # Execute tests with parallel execution where safe
        for category_name, test_func in test_categories:
            print(f"\n🔄 Executing: {category_name}")
            print("-" * 40)
            
            try:
                category_results = test_func()
                
                # Store results by category
                if "property" in category_name.lower():
                    results.property_results = category_results
                elif "metamorphic" in category_name.lower():
                    results.metamorphic_results = category_results
                elif "mutation" in category_name.lower():
                    results.mutation_results = category_results
                elif "fuzz" in category_name.lower():
                    results.fuzz_results = category_results
                elif "consistency" in category_name.lower():
                    results.consistency_results = category_results
                elif "performance" in category_name.lower():
                    results.performance_results = category_results
                elif "oracle" in category_name.lower():
                    results.oracle_results = category_results
                elif "container" in category_name.lower():
                    results.container_results = category_results
                
                # Update overall metrics
                if category_results.get('success', False):
                    results.total_tests_passed += 1
                results.total_tests_run += 1
                
                # Print category summary
                if category_results.get('success', False):
                    print(f"  ✅ {category_name}: PASSED")
                else:
                    print(f"  ❌ {category_name}: FAILED")
                    if 'error' in category_results:
                        print(f"     Error: {category_results['error']}")
                        
            except Exception as e:
                print(f"  ❌ {category_name}: EXCEPTION - {e}")
                results.total_tests_run += 1
        
        # Analyze requirement compliance
        results.requirements_met = self._analyze_requirement_compliance(results)
        results.meets_all_requirements = all(results.requirements_met.values())
        
        overall_end_time = time.perf_counter()
        results.execution_time_seconds = overall_end_time - overall_start_time
        
        return results
    
    def _run_container_tests(self) -> Dict[str, Any]:
        """Run container integration tests."""
        print("🐳 Running container integration tests...")
        
        start_time = time.perf_counter()
        
        try:
            source_dir = Path(__file__).parent.parent
            container_results = self.container_tester.run_all_integration_tests(source_dir)
            
            total_suites = len(container_results)
            successful_suites = 0
            total_tests = 0
            successful_tests = 0
            
            for suite_name, suite_results in container_results.items():
                suite_success_count = len([r for r in suite_results if r.success])
                suite_total = len(suite_results)
                
                total_tests += suite_total
                successful_tests += suite_success_count
                
                if suite_success_count == suite_total:
                    successful_suites += 1
            
            success_rate = successful_suites / total_suites if total_suites > 0 else 0
            
            end_time = time.perf_counter()
            
            return {
                'success': success_rate >= 0.8,  # 80% success rate required
                'total_suites': total_suites,
                'successful_suites': successful_suites,
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'execution_time': end_time - start_time,
                'detailed_results': container_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_requirement_compliance(self, results: TestResults) -> Dict[str, bool]:
        """Analyze compliance with all requirements."""
        compliance = {}
        
        # Property-based testing compliance
        if results.property_results:
            compliance['property_coverage'] = results.property_results.get('meets_threshold', False)
        
        # Metamorphic testing compliance  
        if results.metamorphic_results:
            compliance['metamorphic_coverage'] = results.metamorphic_results.get('meets_threshold', False)
        
        # Mutation testing compliance
        if results.mutation_results:
            compliance['mutation_coverage'] = results.mutation_results.get('meets_threshold', False)
        
        # Fuzz testing compliance
        if results.fuzz_results:
            compliance['fuzz_crashes'] = results.fuzz_results.get('meets_crash_threshold', False)
            compliance['fuzz_runtime'] = results.fuzz_results.get('meets_runtime_requirement', False)
        
        # Consistency testing compliance
        if results.consistency_results:
            compliance['consistency_stability'] = results.consistency_results.get('meets_requirement', False)
        
        # Performance testing compliance
        if results.performance_results:
            compliance['performance_objectives'] = results.performance_results.get('meets_all_requirements', False)
        
        # Oracle validation compliance
        if results.oracle_results:
            compliance['oracle_validation'] = results.oracle_results.get('meets_requirement', False)
        
        # Container testing compliance (if run)
        if results.container_results:
            compliance['container_integration'] = results.container_results.get('success', False)
        
        return compliance
    
    def generate_test_report(self, results: TestResults, 
                           output_file: Optional[Path] = None) -> str:
        """Generate comprehensive test report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PACKREPO COMPREHENSIVE TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Status: {'✅ PASS' if results.meets_all_requirements else '❌ FAIL'}")
        report_lines.append(f"Tests Run: {results.total_tests_passed}/{results.total_tests_run}")
        report_lines.append(f"Execution Time: {results.execution_time_seconds:.1f}s")
        report_lines.append("")
        
        # Requirement Compliance
        report_lines.append("REQUIREMENT COMPLIANCE")
        report_lines.append("-" * 40)
        for requirement, met in results.requirements_met.items():
            status = "✅ PASS" if met else "❌ FAIL"
            report_lines.append(f"{requirement}: {status}")
        report_lines.append("")
        
        # Detailed Results
        test_sections = [
            ("Property-Based Testing", results.property_results),
            ("Metamorphic Testing", results.metamorphic_results),
            ("Mutation Testing", results.mutation_results),
            ("Fuzz Testing", results.fuzz_results),
            ("Consistency Validation", results.consistency_results),
            ("Performance Testing", results.performance_results),
            ("Oracle Validation", results.oracle_results),
            ("Container Integration", results.container_results)
        ]
        
        for section_name, section_results in test_sections:
            if section_results is None:
                continue
                
            report_lines.append(f"{section_name.upper()}")
            report_lines.append("-" * 40)
            
            if section_results.get('success', False):
                report_lines.append("✅ PASSED")
            else:
                report_lines.append("❌ FAILED")
                if 'error' in section_results:
                    report_lines.append(f"Error: {section_results['error']}")
            
            # Add key metrics
            if 'coverage' in section_results:
                report_lines.append(f"Coverage: {section_results['coverage']*100:.1f}%")
            if 'execution_time' in section_results:
                report_lines.append(f"Execution Time: {section_results['execution_time']:.1f}s")
                
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Test report saved to: {output_file}")
        
        return report_text


def main():
    """Main entry point for test orchestrator."""
    parser = argparse.ArgumentParser(description="PackRepo Comprehensive Test Orchestrator")
    parser.add_argument("--include-containers", action="store_true", 
                       help="Include container integration tests (requires Docker)")
    parser.add_argument("--output-report", type=str, 
                       help="Output file for test report")
    parser.add_argument("--mutation-threshold", type=float, default=0.80,
                       help="Mutation coverage threshold (default: 0.80)")
    parser.add_argument("--property-threshold", type=float, default=0.70,
                       help="Property coverage threshold (default: 0.70)")
    parser.add_argument("--fuzz-runtime", type=int, default=30,
                       help="Minimum fuzz runtime in minutes (default: 30)")
    parser.add_argument("--consistency-runs", type=int, default=100,
                       help="Number of consistency validation runs (default: 100)")
    
    args = parser.parse_args()
    
    # Create custom requirements if specified
    requirements = TestRequirements(
        mutation_coverage_threshold=args.mutation_threshold,
        property_coverage_threshold=args.property_threshold,
        fuzz_min_runtime_minutes=args.fuzz_runtime,
        consistency_runs=args.consistency_runs
    )
    
    # Initialize orchestrator
    orchestrator = TestOrchestrator(requirements=requirements)
    
    # Run comprehensive test suite
    try:
        results = orchestrator.run_comprehensive_test_suite(
            include_container_tests=args.include_containers
        )
        
        # Generate and display report
        report_file = Path(args.output_report) if args.output_report else None
        report = orchestrator.generate_test_report(results, report_file)
        print(report)
        
        # Exit with appropriate code
        if results.meets_all_requirements:
            print("\n🎉 All requirements met! PackRepo testing framework complete.")
            sys.exit(0)
        else:
            print("\n⚠️  Some requirements not met. See report for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Testing failed with exception: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
"""
Performance Regression Testing for PackRepo.

This module implements comprehensive performance regression testing with
p50/p95 latency monitoring, throughput analysis, and memory profiling.

From TODO.md requirements:
- Performance p50/p95 within objectives
- Regression detection across versions
- Benchmarking against baseline performance
- Memory and CPU utilization monitoring
"""

import gc
import json
import psutil
import statistics
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import unittest

from packrepo.packer.core import PackRepo
from tests.e2e.test_golden_smoke_flows import GoldenSmokeTestSuite


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a single run."""
    run_id: int
    execution_time_ms: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    tokens_per_second: float
    files_processed: int
    total_tokens: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results and analysis."""
    test_case: str
    total_runs: int
    successful_runs: int
    
    # Execution time statistics (ms)
    execution_times: List[float]
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Throughput statistics
    tokens_per_second: List[float]
    mean_throughput: float
    p95_throughput: float
    
    # Resource utilization
    memory_usage_mb: List[float]
    peak_memory_mb: float
    mean_memory_mb: float
    cpu_utilization: List[float]
    mean_cpu_percent: float
    
    # Performance objectives
    meets_p50_objective: bool
    meets_p95_objective: bool
    meets_throughput_objective: bool
    meets_memory_objective: bool


class PerformanceObjectives:
    """Performance objectives and thresholds."""
    
    # Latency objectives (milliseconds)
    P50_LATENCY_TARGET_MS = 500  # p50 < 500ms
    P95_LATENCY_TARGET_MS = 1500  # p95 < 1500ms
    P99_LATENCY_TARGET_MS = 3000  # p99 < 3000ms
    
    # Throughput objectives
    MIN_TOKENS_PER_SECOND = 5000  # Minimum processing speed
    MIN_FILES_PER_SECOND = 10  # Minimum file processing rate
    
    # Resource objectives
    MAX_MEMORY_MB = 512  # Maximum memory usage
    MAX_CPU_PERCENT = 80  # Maximum CPU utilization
    
    # Regression thresholds (percentage increase)
    LATENCY_REGRESSION_THRESHOLD = 0.20  # 20% increase
    THROUGHPUT_REGRESSION_THRESHOLD = 0.15  # 15% decrease
    MEMORY_REGRESSION_THRESHOLD = 0.25  # 25% increase


class PerformanceRegressionTester:
    """
    Tests for performance regressions in PackRepo.
    
    Implements comprehensive benchmarking with statistical analysis,
    resource monitoring, and regression detection capabilities.
    """
    
    def __init__(self, num_runs: int = 50, warmup_runs: int = 5):
        """
        Initialize performance regression tester.
        
        Args:
            num_runs: Number of benchmark runs per test case
            warmup_runs: Number of warmup runs to exclude from statistics
        """
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.golden_suite = GoldenSmokeTestSuite()
        self.objectives = PerformanceObjectives()
    
    def _create_test_repository(self, test_case: str) -> Path:
        """Create a test repository for performance testing."""
        if test_case == "python_project":
            return self.golden_suite._create_python_test_repo()
        elif test_case == "typescript_project":
            return self.golden_suite._create_typescript_test_repo()
        elif test_case == "documentation_project":
            return self.golden_suite._create_documentation_test_repo()
        else:
            raise ValueError(f"Unknown test case: {test_case}")
    
    def _run_performance_test(self, run_id: int, repo_path: Path,
                            config: Dict[str, Any]) -> PerformanceMetrics:
        """Execute a single performance test run with detailed monitoring."""
        
        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean garbage before test
        
        process = psutil.Process()
        cpu_times_start = process.cpu_times()
        start_time = time.perf_counter()
        
        try:
            # Initialize PackRepo
            pack_repo = PackRepo(
                target_budget=config.get('target_budget', 8000),
                chunk_size=config.get('chunk_size', 4000),
                overlap_size=config.get('overlap_size', 200),
                enable_anchors=config.get('enable_anchors', True)
            )
            
            # Execute packing with monitoring
            result = pack_repo.pack_repository(repo_path)
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Collect memory statistics
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_peak_mb = peak_memory / 1024 / 1024
            memory_current_mb = current_memory / 1024 / 1024
            
            # Calculate CPU utilization
            cpu_times_end = process.cpu_times()
            cpu_time_used = (cpu_times_end.user + cpu_times_end.system) - \
                          (cpu_times_start.user + cpu_times_start.system)
            cpu_percent = (cpu_time_used / (end_time - start_time)) * 100
            
            # Calculate throughput
            tokens_per_second = result.total_tokens / (execution_time_ms / 1000)
            
            return PerformanceMetrics(
                run_id=run_id,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=memory_peak_mb,
                memory_current_mb=memory_current_mb,
                cpu_percent=cpu_percent,
                tokens_per_second=tokens_per_second,
                files_processed=len(result.files),
                total_tokens=result.total_tokens,
                success=True
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            tracemalloc.stop()
            
            return PerformanceMetrics(
                run_id=run_id,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=0,
                memory_current_mb=0,
                cpu_percent=0,
                tokens_per_second=0,
                files_processed=0,
                total_tokens=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_test_case(self, test_case: str,
                          config: Optional[Dict[str, Any]] = None) -> PerformanceBenchmark:
        """
        Run comprehensive performance benchmark for a test case.
        
        Args:
            test_case: Name of the test case to benchmark
            config: PackRepo configuration to use
            
        Returns:
            PerformanceBenchmark with detailed performance analysis
        """
        if config is None:
            config = {
                'target_budget': 8000,
                'chunk_size': 4000,
                'overlap_size': 200,
                'enable_anchors': True
            }
        
        print(f"🚀 Performance benchmarking: {test_case}")
        print(f"   Runs: {self.num_runs} (+ {self.warmup_runs} warmup)")
        
        # Create test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = self._create_test_repository(test_case)
            
            # Execute warmup runs
            print("   🔥 Warmup runs...")
            for i in range(self.warmup_runs):
                self._run_performance_test(i, repo_path, config)
            
            # Execute benchmark runs
            print("   📊 Benchmark runs...")
            metrics: List[PerformanceMetrics] = []
            
            for i in range(self.num_runs):
                result = self._run_performance_test(i, repo_path, config)
                metrics.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"     ✓ Completed {i + 1}/{self.num_runs} runs")
        
        return self._analyze_performance(test_case, metrics)
    
    def _analyze_performance(self, test_case: str,
                           metrics: List[PerformanceMetrics]) -> PerformanceBenchmark:
        """Analyze performance metrics and generate benchmark report."""
        successful_metrics = [m for m in metrics if m.success]
        
        if not successful_metrics:
            raise ValueError(f"No successful runs for {test_case}")
        
        # Execution time analysis
        execution_times = [m.execution_time_ms for m in successful_metrics]
        execution_times.sort()
        
        p50_latency = self._percentile(execution_times, 50)
        p95_latency = self._percentile(execution_times, 95)
        p99_latency = self._percentile(execution_times, 99)
        
        # Throughput analysis
        throughput_values = [m.tokens_per_second for m in successful_metrics]
        mean_throughput = statistics.mean(throughput_values)
        p95_throughput = self._percentile(sorted(throughput_values, reverse=True), 95)
        
        # Memory analysis
        memory_values = [m.memory_peak_mb for m in successful_metrics]
        peak_memory = max(memory_values)
        mean_memory = statistics.mean(memory_values)
        
        # CPU analysis
        cpu_values = [m.cpu_percent for m in successful_metrics]
        mean_cpu = statistics.mean(cpu_values)
        
        # Check objectives
        meets_p50 = p50_latency <= self.objectives.P50_LATENCY_TARGET_MS
        meets_p95 = p95_latency <= self.objectives.P95_LATENCY_TARGET_MS
        meets_throughput = mean_throughput >= self.objectives.MIN_TOKENS_PER_SECOND
        meets_memory = peak_memory <= self.objectives.MAX_MEMORY_MB
        
        return PerformanceBenchmark(
            test_case=test_case,
            total_runs=len(metrics),
            successful_runs=len(successful_metrics),
            execution_times=execution_times,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            mean_latency_ms=statistics.mean(execution_times),
            min_latency_ms=min(execution_times),
            max_latency_ms=max(execution_times),
            tokens_per_second=throughput_values,
            mean_throughput=mean_throughput,
            p95_throughput=p95_throughput,
            memory_usage_mb=memory_values,
            peak_memory_mb=peak_memory,
            mean_memory_mb=mean_memory,
            cpu_utilization=cpu_values,
            mean_cpu_percent=mean_cpu,
            meets_p50_objective=meets_p50,
            meets_p95_objective=meets_p95,
            meets_throughput_objective=meets_throughput,
            meets_memory_objective=meets_memory
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0
        
        k = (len(data) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        
        if f == len(data) - 1:
            return data[f]
        
        return data[f] * (1 - c) + data[f + 1] * c
    
    def detect_regression(self, baseline: PerformanceBenchmark,
                         current: PerformanceBenchmark) -> Dict[str, Any]:
        """
        Detect performance regressions between baseline and current benchmarks.
        
        Args:
            baseline: Baseline performance benchmark
            current: Current performance benchmark
            
        Returns:
            Dictionary with regression analysis results
        """
        regression_analysis = {
            'test_case': current.test_case,
            'regressions_detected': [],
            'improvements_detected': [],
            'overall_status': 'PASS'
        }
        
        # Latency regression analysis
        p50_change = (current.p50_latency_ms - baseline.p50_latency_ms) / baseline.p50_latency_ms
        p95_change = (current.p95_latency_ms - baseline.p95_latency_ms) / baseline.p95_latency_ms
        
        if p50_change > self.objectives.LATENCY_REGRESSION_THRESHOLD:
            regression_analysis['regressions_detected'].append({
                'metric': 'p50_latency',
                'baseline': baseline.p50_latency_ms,
                'current': current.p50_latency_ms,
                'change_percent': p50_change * 100,
                'threshold_percent': self.objectives.LATENCY_REGRESSION_THRESHOLD * 100
            })
            regression_analysis['overall_status'] = 'REGRESSION'
        
        if p95_change > self.objectives.LATENCY_REGRESSION_THRESHOLD:
            regression_analysis['regressions_detected'].append({
                'metric': 'p95_latency',
                'baseline': baseline.p95_latency_ms,
                'current': current.p95_latency_ms,
                'change_percent': p95_change * 100,
                'threshold_percent': self.objectives.LATENCY_REGRESSION_THRESHOLD * 100
            })
            regression_analysis['overall_status'] = 'REGRESSION'
        
        # Throughput regression analysis
        throughput_change = (baseline.mean_throughput - current.mean_throughput) / baseline.mean_throughput
        
        if throughput_change > self.objectives.THROUGHPUT_REGRESSION_THRESHOLD:
            regression_analysis['regressions_detected'].append({
                'metric': 'throughput',
                'baseline': baseline.mean_throughput,
                'current': current.mean_throughput,
                'change_percent': -throughput_change * 100,  # Negative because it's a decrease
                'threshold_percent': self.objectives.THROUGHPUT_REGRESSION_THRESHOLD * 100
            })
            regression_analysis['overall_status'] = 'REGRESSION'
        
        # Memory regression analysis
        memory_change = (current.peak_memory_mb - baseline.peak_memory_mb) / baseline.peak_memory_mb
        
        if memory_change > self.objectives.MEMORY_REGRESSION_THRESHOLD:
            regression_analysis['regressions_detected'].append({
                'metric': 'memory_usage',
                'baseline': baseline.peak_memory_mb,
                'current': current.peak_memory_mb,
                'change_percent': memory_change * 100,
                'threshold_percent': self.objectives.MEMORY_REGRESSION_THRESHOLD * 100
            })
            regression_analysis['overall_status'] = 'REGRESSION'
        
        # Detect improvements (negative regressions)
        if p50_change < -0.05:  # 5% improvement
            regression_analysis['improvements_detected'].append({
                'metric': 'p50_latency',
                'improvement_percent': -p50_change * 100
            })
        
        if throughput_change < -0.05:  # 5% improvement
            regression_analysis['improvements_detected'].append({
                'metric': 'throughput',
                'improvement_percent': -throughput_change * 100
            })
        
        return regression_analysis
    
    def benchmark_all_test_cases(self) -> Dict[str, PerformanceBenchmark]:
        """Run performance benchmarks on all test cases."""
        test_cases = ["python_project", "typescript_project", "documentation_project"]
        results = {}
        
        for test_case in test_cases:
            print(f"\n📊 Benchmarking: {test_case}")
            results[test_case] = self.benchmark_test_case(test_case)
            
            benchmark = results[test_case]
            print(f"  ✅ p50: {benchmark.p50_latency_ms:.1f}ms ({'✓' if benchmark.meets_p50_objective else '❌'})")
            print(f"  ✅ p95: {benchmark.p95_latency_ms:.1f}ms ({'✓' if benchmark.meets_p95_objective else '❌'})")
            print(f"  🚀 Throughput: {benchmark.mean_throughput:.0f} tokens/s ({'✓' if benchmark.meets_throughput_objective else '❌'})")
            print(f"  💾 Memory: {benchmark.peak_memory_mb:.1f}MB ({'✓' if benchmark.meets_memory_objective else '❌'})")
        
        return results
    
    def save_baseline(self, benchmarks: Dict[str, PerformanceBenchmark],
                     baseline_file: Path):
        """Save performance benchmarks as baseline for future comparisons."""
        baseline_data = {}
        
        for test_case, benchmark in benchmarks.items():
            baseline_data[test_case] = {
                'p50_latency_ms': benchmark.p50_latency_ms,
                'p95_latency_ms': benchmark.p95_latency_ms,
                'p99_latency_ms': benchmark.p99_latency_ms,
                'mean_throughput': benchmark.mean_throughput,
                'peak_memory_mb': benchmark.peak_memory_mb,
                'mean_memory_mb': benchmark.mean_memory_mb,
                'mean_cpu_percent': benchmark.mean_cpu_percent
            }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        print(f"📄 Baseline saved to: {baseline_file}")
    
    def load_baseline(self, baseline_file: Path) -> Dict[str, Dict[str, float]]:
        """Load baseline performance data from file."""
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
        
        with open(baseline_file, 'r') as f:
            return json.load(f)


class TestPerformanceRegression(unittest.TestCase):
    """Test cases for performance regression testing framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use smaller number of runs for unit tests
        self.tester = PerformanceRegressionTester(num_runs=10, warmup_runs=2)
    
    def test_python_project_performance(self):
        """Test performance benchmarking for Python project."""
        benchmark = self.tester.benchmark_test_case("python_project")
        
        # Verify basic metrics
        self.assertEqual(benchmark.total_runs, 10)
        self.assertGreaterEqual(benchmark.successful_runs, 8)  # Allow some failures
        
        # Verify performance objectives
        self.assertLessEqual(benchmark.p50_latency_ms, self.tester.objectives.P50_LATENCY_TARGET_MS)
        self.assertLessEqual(benchmark.p95_latency_ms, self.tester.objectives.P95_LATENCY_TARGET_MS)
        self.assertGreaterEqual(benchmark.mean_throughput, self.tester.objectives.MIN_TOKENS_PER_SECOND)
        self.assertLessEqual(benchmark.peak_memory_mb, self.tester.objectives.MAX_MEMORY_MB)
        
        # Verify all objectives are met
        self.assertTrue(benchmark.meets_p50_objective)
        self.assertTrue(benchmark.meets_p95_objective)
        self.assertTrue(benchmark.meets_throughput_objective)
        self.assertTrue(benchmark.meets_memory_objective)
    
    def test_typescript_project_performance(self):
        """Test performance benchmarking for TypeScript project."""
        benchmark = self.tester.benchmark_test_case("typescript_project")
        
        # Verify performance objectives
        self.assertTrue(benchmark.meets_p50_objective)
        self.assertTrue(benchmark.meets_p95_objective)
        self.assertTrue(benchmark.meets_throughput_objective)
    
    def test_documentation_project_performance(self):
        """Test performance benchmarking for documentation project."""
        benchmark = self.tester.benchmark_test_case("documentation_project")
        
        # Verify performance objectives
        self.assertTrue(benchmark.meets_p50_objective)
        self.assertTrue(benchmark.meets_p95_objective)
        self.assertTrue(benchmark.meets_throughput_objective)
    
    def test_percentile_calculation(self):
        """Test percentile calculation accuracy."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        self.assertAlmostEqual(self.tester._percentile(data, 50), 5.5, places=1)
        self.assertAlmostEqual(self.tester._percentile(data, 95), 9.55, places=1)
        self.assertEqual(self.tester._percentile(data, 100), 10)
    
    def test_regression_detection(self):
        """Test performance regression detection."""
        # Create baseline benchmark
        baseline = PerformanceBenchmark(
            test_case="test",
            total_runs=10,
            successful_runs=10,
            execution_times=[],
            p50_latency_ms=100,
            p95_latency_ms=200,
            p99_latency_ms=300,
            mean_latency_ms=110,
            min_latency_ms=80,
            max_latency_ms=150,
            tokens_per_second=[],
            mean_throughput=10000,
            p95_throughput=9000,
            memory_usage_mb=[],
            peak_memory_mb=100,
            mean_memory_mb=80,
            cpu_utilization=[],
            mean_cpu_percent=50,
            meets_p50_objective=True,
            meets_p95_objective=True,
            meets_throughput_objective=True,
            meets_memory_objective=True
        )
        
        # Create regressed benchmark (25% slower)
        regressed = PerformanceBenchmark(
            test_case="test",
            total_runs=10,
            successful_runs=10,
            execution_times=[],
            p50_latency_ms=125,  # 25% slower
            p95_latency_ms=250,  # 25% slower
            p99_latency_ms=375,
            mean_latency_ms=137.5,
            min_latency_ms=100,
            max_latency_ms=187.5,
            tokens_per_second=[],
            mean_throughput=8000,  # 20% slower
            p95_throughput=7200,
            memory_usage_mb=[],
            peak_memory_mb=130,  # 30% more memory
            mean_memory_mb=104,
            cpu_utilization=[],
            mean_cpu_percent=65,
            meets_p50_objective=True,
            meets_p95_objective=True,
            meets_throughput_objective=True,
            meets_memory_objective=True
        )
        
        analysis = self.tester.detect_regression(baseline, regressed)
        
        # Should detect latency regression
        self.assertEqual(analysis['overall_status'], 'REGRESSION')
        self.assertGreater(len(analysis['regressions_detected']), 0)
        
        # Check specific regressions
        regression_metrics = [r['metric'] for r in analysis['regressions_detected']]
        self.assertIn('p50_latency', regression_metrics)
        self.assertIn('p95_latency', regression_metrics)
        self.assertIn('throughput', regression_metrics)
        self.assertIn('memory_usage', regression_metrics)


if __name__ == '__main__':
    # Run performance benchmarking
    tester = PerformanceRegressionTester(num_runs=50)
    
    print("🚀 PackRepo Performance Benchmarking")
    print("=" * 50)
    print("Performance Objectives:")
    print(f"  p50 latency: ≤{tester.objectives.P50_LATENCY_TARGET_MS}ms")
    print(f"  p95 latency: ≤{tester.objectives.P95_LATENCY_TARGET_MS}ms")
    print(f"  Throughput: ≥{tester.objectives.MIN_TOKENS_PER_SECOND} tokens/s")
    print(f"  Memory: ≤{tester.objectives.MAX_MEMORY_MB}MB")
    print()
    
    benchmarks = tester.benchmark_all_test_cases()
    
    print("\n📊 PERFORMANCE BENCHMARK RESULTS")
    print("=" * 50)
    
    all_objectives_met = True
    for test_case, benchmark in benchmarks.items():
        objectives_met = (benchmark.meets_p50_objective and 
                         benchmark.meets_p95_objective and 
                         benchmark.meets_throughput_objective and 
                         benchmark.meets_memory_objective)
        
        status = "✅ PASS" if objectives_met else "❌ FAIL"
        print(f"{test_case}: {status}")
        print(f"  p50: {benchmark.p50_latency_ms:.1f}ms (target: ≤{tester.objectives.P50_LATENCY_TARGET_MS}ms)")
        print(f"  p95: {benchmark.p95_latency_ms:.1f}ms (target: ≤{tester.objectives.P95_LATENCY_TARGET_MS}ms)")
        print(f"  p99: {benchmark.p99_latency_ms:.1f}ms")
        print(f"  Throughput: {benchmark.mean_throughput:.0f} tokens/s (target: ≥{tester.objectives.MIN_TOKENS_PER_SECOND})")
        print(f"  Memory: {benchmark.peak_memory_mb:.1f}MB (target: ≤{tester.objectives.MAX_MEMORY_MB}MB)")
        print(f"  CPU: {benchmark.mean_cpu_percent:.1f}%")
        print()
        
        if not objectives_met:
            all_objectives_met = False
    
    if all_objectives_met:
        print("🎉 All test cases meet performance objectives!")
    else:
        print("⚠️  Some test cases do not meet performance objectives")
    
    # Save baseline for future regression testing
    baseline_file = Path("performance_baseline.json")
    tester.save_baseline(benchmarks, baseline_file)
    
    # Also run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
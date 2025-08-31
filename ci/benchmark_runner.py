#!/usr/bin/env python3
"""
Benchmark execution wrapper and flag management system for PackRepo FastPath V2.
Handles systematic execution of benchmark variants with proper flag isolation.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfiguration:
    """Configuration for a benchmark run."""
    variant: str
    flags: Dict[str, str]
    budgets: List[str]
    seed: int
    iterations: int = 3
    timeout: int = 1800  # 30 minutes
    
@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    variant: str
    configuration: BenchmarkConfiguration
    success: bool
    duration: float
    metrics: Dict[str, float]
    raw_output: str
    error_output: str
    artifacts: Dict[str, Any]

class BenchmarkRunner:
    """Manages systematic execution of FastPath V2 benchmarks."""
    
    def __init__(self, project_root: str = ".", artifacts_dir: str = "artifacts"):
        self.project_root = Path(project_root).resolve()
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Ensure benchmark script exists
        self.benchmark_script = self.project_root / "run_fastpath_benchmarks.py"
        if not self.benchmark_script.exists():
            raise FileNotFoundError(f"Benchmark script not found: {self.benchmark_script}")
        
        # Flag combinations for V1-V5 variants
        self.flag_configurations = {
            "baseline": BenchmarkConfiguration(
                variant="baseline",
                flags={"FASTPATH_POLICY_V2": "0"},
                budgets=["50k", "120k", "200k"],
                seed=1337
            ),
            "V1": BenchmarkConfiguration(
                variant="V1",
                flags={
                    "FASTPATH_POLICY_V2": "1",
                    "FASTPATH_VARIANT": "V1",
                    "FASTPATH_AGGRESSIVE_CACHING": "1"
                },
                budgets=["50k", "120k", "200k"],
                seed=1337
            ),
            "V2": BenchmarkConfiguration(
                variant="V2", 
                flags={
                    "FASTPATH_POLICY_V2": "1",
                    "FASTPATH_VARIANT": "V2",
                    "FASTPATH_PARALLEL_PROCESSING": "1",
                    "FASTPATH_BATCH_SIZE": "64"
                },
                budgets=["50k", "120k", "200k"],
                seed=1337
            ),
            "V3": BenchmarkConfiguration(
                variant="V3",
                flags={
                    "FASTPATH_POLICY_V2": "1", 
                    "FASTPATH_VARIANT": "V3",
                    "FASTPATH_MEMORY_OPTIMIZATION": "1",
                    "FASTPATH_COMPRESSION": "1"
                },
                budgets=["50k", "120k", "200k"],
                seed=1337
            ),
            "V4": BenchmarkConfiguration(
                variant="V4",
                flags={
                    "FASTPATH_POLICY_V2": "1",
                    "FASTPATH_VARIANT": "V4", 
                    "FASTPATH_HYBRID_MODE": "1",
                    "FASTPATH_ADAPTIVE_THRESHOLD": "0.85"
                },
                budgets=["50k", "120k", "200k"],
                seed=1337
            ),
            "V5": BenchmarkConfiguration(
                variant="V5",
                flags={
                    "FASTPATH_POLICY_V2": "1",
                    "FASTPATH_VARIANT": "V5",
                    "FASTPATH_AGGRESSIVE_CACHING": "1",
                    "FASTPATH_PARALLEL_PROCESSING": "1", 
                    "FASTPATH_MEMORY_OPTIMIZATION": "1",
                    "FASTPATH_ADAPTIVE_THRESHOLD": "0.90"
                },
                budgets=["50k", "120k", "200k"],
                seed=1337
            )
        }
    
    def run_single_benchmark(self, config: BenchmarkConfiguration) -> BenchmarkResult:
        """Execute a single benchmark configuration."""
        logger.info(f"Running benchmark for variant: {config.variant}")
        
        start_time = time.time()
        result = BenchmarkResult(
            variant=config.variant,
            configuration=config,
            success=False,
            duration=0,
            metrics={},
            raw_output="",
            error_output="",
            artifacts={}
        )
        
        try:
            # Create isolated output file
            output_file = self.artifacts_dir / f"{config.variant.lower()}_benchmark.json"
            
            # Build command
            budgets_str = ",".join(config.budgets)
            cmd = [
                sys.executable,
                str(self.benchmark_script),
                "--budgets", budgets_str,
                "--seed", str(config.seed),
                "--output", str(output_file)
            ]
            
            # Set up environment with flags
            env = os.environ.copy()
            env.update(config.flags)
            
            # Log the configuration
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Environment flags: {config.flags}")
            
            # Execute benchmark
            process = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=config.timeout
            )
            
            result.duration = time.time() - start_time
            result.raw_output = process.stdout
            result.error_output = process.stderr
            
            if process.returncode == 0:
                result.success = True
                logger.info(f"Benchmark {config.variant} completed successfully in {result.duration:.2f}s")
                
                # Load results if output file exists
                if output_file.exists():
                    with open(output_file) as f:
                        benchmark_data = json.load(f)
                    
                    result.artifacts["raw_data"] = benchmark_data
                    result.metrics = self._extract_metrics(benchmark_data, config.variant)
                else:
                    logger.warning(f"Output file not created: {output_file}")
                    
            else:
                logger.error(f"Benchmark {config.variant} failed with exit code {process.returncode}")
                logger.error(f"STDERR: {process.stderr}")
                
        except subprocess.TimeoutExpired:
            result.duration = time.time() - start_time
            result.error_output = f"Benchmark timed out after {config.timeout} seconds"
            logger.error(f"Benchmark {config.variant} timed out")
            
        except Exception as e:
            result.duration = time.time() - start_time
            result.error_output = str(e)
            logger.error(f"Benchmark {config.variant} failed with exception: {e}")
        
        # Save individual result
        result_file = self.artifacts_dir / f"{config.variant.lower()}_result.json"
        with open(result_file, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        return result
    
    def _extract_metrics(self, benchmark_data: Dict[str, Any], variant: str) -> Dict[str, float]:
        """Extract key metrics from benchmark data."""
        metrics = {}
        
        try:
            # Standard metrics expected from run_fastpath_benchmarks.py
            if "qa_per_100k" in benchmark_data:
                metrics["qa_100k"] = float(benchmark_data["qa_per_100k"])
            
            if "latency_p95" in benchmark_data:
                metrics["latency_p95"] = float(benchmark_data["latency_p95"])
            
            if "latency_mean" in benchmark_data:
                metrics["latency_mean"] = float(benchmark_data["latency_mean"])
            
            if "memory_peak_mb" in benchmark_data:
                metrics["memory_peak"] = float(benchmark_data["memory_peak_mb"])
            
            if "throughput_ops_per_sec" in benchmark_data:
                metrics["throughput"] = float(benchmark_data["throughput_ops_per_sec"])
            
            # Compute efficiency metrics
            if "total_requests" in benchmark_data and "total_duration" in benchmark_data:
                metrics["requests_per_second"] = benchmark_data["total_requests"] / benchmark_data["total_duration"]
            
            # Budget-specific metrics
            for budget in ["50k", "120k", "200k"]:
                budget_key = f"budget_{budget}"
                if budget_key in benchmark_data:
                    metrics[f"qa_100k_{budget}"] = benchmark_data[budget_key].get("qa_per_100k", 0)
                    metrics[f"latency_{budget}"] = benchmark_data[budget_key].get("latency_p95", 0)
            
            logger.info(f"Extracted {len(metrics)} metrics for {variant}")
            
        except Exception as e:
            logger.warning(f"Failed to extract metrics for {variant}: {e}")
        
        return metrics
    
    def run_parallel_benchmarks(self, variants: Optional[List[str]] = None, 
                               max_workers: int = 3) -> Dict[str, BenchmarkResult]:
        """Run multiple benchmark variants in parallel."""
        if variants is None:
            variants = list(self.flag_configurations.keys())
        
        logger.info(f"Running benchmarks for variants: {variants}")
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all benchmark jobs
            future_to_variant = {}
            for variant in variants:
                if variant in self.flag_configurations:
                    config = self.flag_configurations[variant]
                    future = executor.submit(self.run_single_benchmark, config)
                    future_to_variant[future] = variant
                else:
                    logger.error(f"Unknown variant: {variant}")
            
            # Collect results as they complete
            for future in as_completed(future_to_variant):
                variant = future_to_variant[future]
                try:
                    result = future.result()
                    results[variant] = result
                    
                    status = "✅" if result.success else "❌"
                    logger.info(f"{status} {variant}: {result.duration:.1f}s")
                    
                except Exception as e:
                    logger.error(f"Failed to get result for {variant}: {e}")
                    # Create a failed result
                    results[variant] = BenchmarkResult(
                        variant=variant,
                        configuration=self.flag_configurations[variant],
                        success=False,
                        duration=0,
                        metrics={},
                        raw_output="",
                        error_output=str(e),
                        artifacts={}
                    )
        
        return results
    
    def run_sequential_benchmarks(self, variants: Optional[List[str]] = None) -> Dict[str, BenchmarkResult]:
        """Run benchmark variants sequentially (more stable but slower)."""
        if variants is None:
            variants = list(self.flag_configurations.keys())
        
        logger.info(f"Running benchmarks sequentially for variants: {variants}")
        results = {}
        
        for variant in variants:
            if variant in self.flag_configurations:
                config = self.flag_configurations[variant]
                result = self.run_single_benchmark(config)
                results[variant] = result
                
                status = "✅" if result.success else "❌"
                logger.info(f"{status} {variant}: {result.duration:.1f}s")
            else:
                logger.error(f"Unknown variant: {variant}")
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        logger.info("Generating benchmark comparison report")
        
        # Get baseline metrics
        baseline_metrics = results.get("baseline", BenchmarkResult("baseline", None, False, 0, {}, "", "", {})).metrics
        baseline_qa = baseline_metrics.get("qa_100k", 0.7230)  # Default baseline
        baseline_latency = baseline_metrics.get("latency_p95", 896)  # Default baseline
        
        report = {
            "timestamp": time.time(),
            "baseline": {
                "qa_100k": baseline_qa,
                "latency_p95": baseline_latency
            },
            "variants": {},
            "comparisons": {},
            "summary": {
                "total_variants": len([r for r in results.values() if r.success]),
                "successful_runs": len([r for r in results.values() if r.success]),
                "failed_runs": len([r for r in results.values() if not r.success])
            },
            "recommendations": []
        }
        
        # Process each variant
        best_qa_improvement = 0
        best_variant = None
        
        for variant, result in results.items():
            if not result.success or variant == "baseline":
                continue
            
            variant_qa = result.metrics.get("qa_100k", 0)
            variant_latency = result.metrics.get("latency_p95", 0)
            
            if variant_qa == 0:
                continue
            
            # Calculate improvements/regressions
            qa_improvement = (variant_qa - baseline_qa) / baseline_qa
            latency_regression = (variant_latency - baseline_latency) / baseline_latency if baseline_latency > 0 else 0
            
            # Track best variant
            if qa_improvement > best_qa_improvement:
                best_qa_improvement = qa_improvement
                best_variant = variant
            
            variant_data = {
                "metrics": result.metrics,
                "qa_improvement": qa_improvement,
                "latency_regression": latency_regression,
                "duration": result.duration,
                "success": result.success
            }
            
            report["variants"][variant] = variant_data
            
            # Compare against thresholds
            report["comparisons"][variant] = {
                "meets_qa_threshold": qa_improvement >= 0.13,  # +13% required
                "meets_latency_threshold": latency_regression <= 0.10,  # ≤10% regression
                "overall_acceptable": qa_improvement >= 0.13 and latency_regression <= 0.10
            }
        
        # Generate recommendations
        acceptable_variants = [
            v for v, comp in report["comparisons"].items() 
            if comp["overall_acceptable"]
        ]
        
        if acceptable_variants:
            report["recommendations"].append(f"✅ Variants meeting acceptance criteria: {', '.join(acceptable_variants)}")
        else:
            report["recommendations"].append("❌ No variants meet both QA (+13%) and latency (≤10%) criteria")
        
        if best_variant:
            report["recommendations"].append(f"🏆 Best QA improvement: {best_variant} (+{best_qa_improvement:.1%})")
        
        # Save report
        report_file = self.artifacts_dir / "benchmark_comparison.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {report_file}")
        return report
    
    def run_complete_benchmark_suite(self, parallel: bool = True) -> Dict[str, Any]:
        """Run the complete benchmark suite and generate report."""
        logger.info("🚀 Starting complete FastPath V2 benchmark suite")
        suite_start = time.time()
        
        # Run benchmarks
        if parallel:
            results = self.run_parallel_benchmarks()
        else:
            results = self.run_sequential_benchmarks()
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report(results)
        
        # Create summary
        suite_duration = time.time() - suite_start
        summary = {
            "suite_duration": suite_duration,
            "results": {variant: asdict(result) for variant, result in results.items()},
            "comparison": comparison_report,
            "execution_summary": {
                "total_time_minutes": suite_duration / 60,
                "variants_executed": len(results),
                "successful_variants": len([r for r in results.values() if r.success]),
                "parallel_execution": parallel
            }
        }
        
        # Save complete summary
        summary_file = self.artifacts_dir / "benchmark_suite_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"🎯 Benchmark suite completed in {suite_duration/60:.1f} minutes")
        logger.info(f"📊 Results saved to {summary_file}")
        
        return summary

def main():
    """Main entry point for benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PackRepo FastPath V2 Benchmark Runner")
    parser.add_argument("--variants", nargs="+", 
                       choices=["baseline", "V1", "V2", "V3", "V4", "V5"],
                       help="Variants to run (default: all)")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run benchmarks in parallel")
    parser.add_argument("--sequential", action="store_true",
                       help="Run benchmarks sequentially")
    parser.add_argument("--max-workers", type=int, default=3,
                       help="Maximum parallel workers")
    parser.add_argument("--project-root", default=".", 
                       help="Project root directory")
    parser.add_argument("--artifacts-dir", default="artifacts",
                       help="Artifacts output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Override parallel setting if sequential requested
    parallel = not args.sequential and args.parallel
    
    try:
        # Create runner
        runner = BenchmarkRunner(
            project_root=args.project_root,
            artifacts_dir=args.artifacts_dir
        )
        
        # Run benchmark suite
        summary = runner.run_complete_benchmark_suite(parallel=parallel)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 FASTPATH V2 BENCHMARK SUITE RESULTS")
        print("="*60)
        print(f"⏱️  Total Duration: {summary['execution_summary']['total_time_minutes']:.1f} minutes")
        print(f"🔄 Variants Executed: {summary['execution_summary']['variants_executed']}")
        print(f"✅ Successful Runs: {summary['execution_summary']['successful_variants']}")
        print(f"🔀 Execution Mode: {'Parallel' if parallel else 'Sequential'}")
        
        if summary["comparison"]["recommendations"]:
            print("\n📋 Recommendations:")
            for rec in summary["comparison"]["recommendations"]:
                print(f"   {rec}")
        
        print(f"\n📁 Detailed results: {args.artifacts_dir}/benchmark_suite_summary.json")
        
        # Exit code based on success
        success_count = summary["execution_summary"]["successful_variants"]
        total_count = summary["execution_summary"]["variants_executed"]
        
        if success_count == total_count and success_count > 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
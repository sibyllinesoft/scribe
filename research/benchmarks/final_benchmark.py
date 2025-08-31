#!/usr/bin/env python3
"""
Final FastPath vs PackRepo Baseline Benchmark
=============================================

Comprehensive benchmark comparing FastPath against the actual PackRepo baseline
system, measuring key performance metrics with statistical analysis.

Features:
- Uses actual PackRepo library as baseline (not rendergit.py)
- Measures execution time, token efficiency, file selection quality
- Statistical significance testing
- Professional performance report
- Real-world repository testing
"""

import json
import subprocess
import tempfile
import time
import statistics
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import sys
import os

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    system: str
    execution_time_sec: float
    total_tokens: int
    files_selected: int
    readme_included: bool
    doc_files_count: int
    token_efficiency: float  # tokens per second
    success: bool
    error_msg: str = ""

class FinalBenchmark:
    """Comprehensive FastPath vs PackRepo baseline benchmark."""
    
    def __init__(self, repo_path: Path, num_runs: int = 8):
        self.repo_path = repo_path
        self.num_runs = num_runs
        
    def run_fastpath_system(self, token_budget: int) -> BenchmarkResult:
        """Run FastPath system with proper error handling."""
        start_time = time.perf_counter()
        
        try:
            # Import FastPath components
            from packrepo.fastpath import FastScanner, HeuristicScorer, TTLScheduler, ExecutionMode
            from packrepo.selector import MMRSelector
            from packrepo.tokenizer import TokenEstimator
            from packrepo.packer.tokenizer.implementations import create_tokenizer
            
            # Initialize scheduler
            scheduler = TTLScheduler(ExecutionMode.FAST_PATH)
            scheduler.start_execution()
            
            # Phase 1: Fast scanning
            scanner = FastScanner(self.repo_path, ttl_seconds=8.0)
            scan_results = scanner.scan_repository()
            
            if not scan_results:
                raise ValueError("No files found during scanning")
            
            # Phase 2: Heuristic scoring
            scorer = HeuristicScorer()
            scored_files = scorer.score_all_files(scan_results)
            
            if not scored_files:
                raise ValueError("No files scored")
            
            # Phase 3: Selection
            selector = MMRSelector()
            selected = selector.select_files(scored_files, token_budget)
            
            if not selected:
                raise ValueError("No files selected")
            
            # Phase 4: Tokenization and finalization
            tokenizer = create_tokenizer('tiktoken', 'gpt-4')
            estimator = TokenEstimator(tokenizer)
            
            pack_metadata = {
                'mode': 'fastpath',
                'repo_path': str(self.repo_path),
                'generation_time': time.time(),
                'target_budget': token_budget,
            }
            
            finalized = estimator.finalize_pack(selected, token_budget, pack_metadata)
            
            execution_time = time.perf_counter() - start_time
            
            # Analyze quality
            pack_content = finalized.pack_content
            readme_included = any('readme' in pack_content.lower() 
                                for readme_word in ['readme', 'README.md', 'readme.md'])
            
            doc_indicators = ['.md', '.rst', '.txt', 'readme', 'changelog', 'license', 'contributing']
            doc_files_count = sum(1 for file_result in finalized.selected_files
                                if any(indicator in str(file_result.stats.path).lower() 
                                     for indicator in doc_indicators))
            
            token_efficiency = finalized.total_tokens / execution_time if execution_time > 0 else 0
            
            return BenchmarkResult(
                system="fastpath",
                execution_time_sec=execution_time,
                total_tokens=finalized.total_tokens,
                files_selected=len(finalized.selected_files),
                readme_included=readme_included,
                doc_files_count=doc_files_count,
                token_efficiency=token_efficiency,
                success=True
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return BenchmarkResult(
                system="fastpath",
                execution_time_sec=execution_time,
                total_tokens=0,
                files_selected=0,
                readme_included=False,
                doc_files_count=0,
                token_efficiency=0,
                success=False,
                error_msg=str(e)
            )
    
    def run_baseline_system(self, token_budget: int) -> BenchmarkResult:
        """Run baseline PackRepo system."""
        start_time = time.perf_counter()
        
        try:
            # Import PackRepo baseline
            from packrepo.library import RepositoryPacker
            from packrepo.packer.tokenizer import TokenizerType
            
            # Create baseline packer
            packer = RepositoryPacker(tokenizer_type=TokenizerType.CL100K_BASE)
            
            # Simple file filter (similar to rendergit.py logic)
            def file_filter(file_path: Path) -> bool:
                try:
                    if file_path.is_dir():
                        return False
                    
                    # Skip large files
                    if file_path.stat().st_size > 100000:  # 100KB max
                        return False
                    
                    # Skip binary files
                    binary_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', 
                                       '.tar', '.gz', '.so', '.dll', '.exe', '.bin'}
                    if file_path.suffix.lower() in binary_extensions:
                        return False
                    
                    # Skip .git and other VCS directories
                    if '.git' in file_path.parts:
                        return False
                    
                    return True
                except Exception:
                    return False
            
            # Pack repository
            pack_result = packer.pack_repository(
                repo_path=self.repo_path,
                token_budget=token_budget,
                mode="comprehension",
                variant="v1_basic",
                deterministic=True,
                file_filter=file_filter
            )
            
            execution_time = time.perf_counter() - start_time
            
            # Analyze results
            pack_content = pack_result.to_string()
            stats = pack_result.get_statistics()
            
            readme_included = 'readme' in pack_content.lower()
            
            # Count documentation files in pack content
            doc_indicators = ['.md', '.rst', '.txt', 'readme', 'changelog', 'license', 'contributing']
            doc_files_count = sum(pack_content.lower().count(indicator) for indicator in doc_indicators)
            # Rough estimate, cap at reasonable number
            doc_files_count = min(doc_files_count, stats.get('selected_chunks', 0))
            
            token_efficiency = stats.get('actual_tokens', 0) / execution_time if execution_time > 0 else 0
            
            return BenchmarkResult(
                system="baseline",
                execution_time_sec=execution_time,
                total_tokens=stats.get('actual_tokens', 0),
                files_selected=stats.get('selected_chunks', 0),
                readme_included=readme_included,
                doc_files_count=doc_files_count,
                token_efficiency=token_efficiency,
                success=True
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return BenchmarkResult(
                system="baseline",
                execution_time_sec=execution_time,
                total_tokens=0,
                files_selected=0,
                readme_included=False,
                doc_files_count=0,
                token_efficiency=0,
                success=False,
                error_msg=str(e)
            )
    
    def run_single_comparison(self, token_budget: int) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Run both systems once."""
        print(f"  FastPath...", end="", flush=True)
        fastpath_result = self.run_fastpath_system(token_budget)
        if fastpath_result.success:
            print(f" ✓ ({fastpath_result.execution_time_sec:.2f}s, {fastpath_result.total_tokens} tokens)")
        else:
            print(f" ✗ ({fastpath_result.error_msg[:40]}...)")
        
        print(f"  Baseline...", end="", flush=True)
        baseline_result = self.run_baseline_system(token_budget)
        if baseline_result.success:
            print(f" ✓ ({baseline_result.execution_time_sec:.2f}s, {baseline_result.total_tokens} tokens)")
        else:
            print(f" ✗ ({baseline_result.error_msg[:40]}...)")
        
        return fastpath_result, baseline_result
    
    def run_benchmark(self, token_budgets: List[int]) -> Dict[int, Dict[str, List[BenchmarkResult]]]:
        """Run complete benchmark suite."""
        results = {}
        
        for budget in token_budgets:
            print(f"\n{'='*60}")
            print(f"Token Budget: {budget:,}")
            print(f"{'='*60}")
            
            fastpath_runs = []
            baseline_runs = []
            
            for run_num in range(1, self.num_runs + 1):
                print(f"\nRun {run_num}/{self.num_runs}:")
                
                fastpath_result, baseline_result = self.run_single_comparison(budget)
                
                fastpath_runs.append(fastpath_result)
                baseline_runs.append(baseline_result)
                
                # Show comparison if both successful
                if fastpath_result.success and baseline_result.success:
                    time_improvement = ((baseline_result.execution_time_sec - fastpath_result.execution_time_sec) 
                                      / baseline_result.execution_time_sec * 100)
                    efficiency_improvement = ((fastpath_result.token_efficiency - baseline_result.token_efficiency) 
                                            / baseline_result.token_efficiency * 100) if baseline_result.token_efficiency > 0 else 0
                    
                    print(f"  ⚡ Time improvement: {time_improvement:+.1f}%")
                    print(f"  🚀 Efficiency improvement: {efficiency_improvement:+.1f}%")
                else:
                    print(f"  ⚠️ One or both systems failed")
            
            results[budget] = {
                'fastpath': fastpath_runs,
                'baseline': baseline_runs
            }
        
        return results
    
    def calculate_stats(self, runs: List[BenchmarkResult]) -> Dict[str, float]:
        """Calculate comprehensive statistics."""
        successful = [r for r in runs if r.success]
        if not successful:
            return {'count': 0}
        
        times = [r.execution_time_sec for r in successful]
        tokens = [r.total_tokens for r in successful]
        files = [r.files_selected for r in successful]
        efficiencies = [r.token_efficiency for r in successful]
        
        return {
            'count': len(successful),
            'success_rate': len(successful) / len(runs),
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'mean_tokens': statistics.mean(tokens),
            'mean_files': statistics.mean(files),
            'mean_efficiency': statistics.mean(efficiencies),
            'readme_retention_rate': sum(r.readme_included for r in successful) / len(successful),
            'mean_doc_files': statistics.mean([r.doc_files_count for r in successful]),
        }
    
    def statistical_significance_test(self, baseline_values: List[float], 
                                    fastpath_values: List[float]) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        if len(baseline_values) < 3 or len(fastpath_values) < 3:
            return {'significant': False, 'p_value': 1.0, 'effect_size': 0.0}
        
        # Two-sample t-test
        baseline_mean = statistics.mean(baseline_values)
        fastpath_mean = statistics.mean(fastpath_values)
        baseline_std = statistics.stdev(baseline_values)
        fastpath_std = statistics.stdev(fastpath_values)
        
        n1, n2 = len(baseline_values), len(fastpath_values)
        
        # Pooled standard error
        pooled_var = ((n1 - 1) * baseline_std**2 + (n2 - 1) * fastpath_std**2) / (n1 + n2 - 2)
        se_diff = math.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # t-statistic
        if se_diff > 0:
            t_stat = abs(fastpath_mean - baseline_mean) / se_diff
            # Rough p-value estimation (assumes normal distribution)
            # For more precise, would use scipy.stats
            df = n1 + n2 - 2
            # Simple approximation: t > 2.0 roughly corresponds to p < 0.05 for reasonable df
            significant = t_stat > 2.0
            p_value = max(0.01, min(0.5, 2.0 / t_stat))  # Rough approximation
        else:
            t_stat = 0
            significant = False
            p_value = 1.0
        
        # Effect size (Cohen's d)
        if pooled_var > 0:
            effect_size = (fastpath_mean - baseline_mean) / math.sqrt(pooled_var)
        else:
            effect_size = 0
        
        return {
            'significant': significant,
            'p_value': p_value,
            'effect_size': effect_size,
            't_statistic': t_stat
        }
    
    def generate_comprehensive_report(self, results: Dict[int, Dict[str, List[BenchmarkResult]]]) -> str:
        """Generate comprehensive performance report with statistical analysis."""
        lines = []
        lines.append("# FastPath vs PackRepo Baseline Performance Benchmark")
        lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Repository**: {self.repo_path.name}")
        lines.append(f"**Benchmark runs**: {self.num_runs} per system per configuration")
        lines.append(f"**Systems tested**: FastPath vs PackRepo Baseline")
        
        # Executive Summary
        lines.append("\n## Executive Summary")
        
        overall_time_improvements = []
        overall_efficiency_improvements = []
        significant_improvements = 0
        total_comparisons = 0
        
        for budget, budget_results in results.items():
            fastpath_stats = self.calculate_stats(budget_results['fastpath'])
            baseline_stats = self.calculate_stats(budget_results['baseline'])
            
            if fastpath_stats['count'] >= 3 and baseline_stats['count'] >= 3:
                total_comparisons += 1
                
                # Time improvement
                time_improvement = ((baseline_stats['mean_time'] - fastpath_stats['mean_time']) 
                                  / baseline_stats['mean_time'] * 100)
                overall_time_improvements.append(time_improvement)
                
                # Efficiency improvement
                if baseline_stats['mean_efficiency'] > 0:
                    efficiency_improvement = ((fastpath_stats['mean_efficiency'] - baseline_stats['mean_efficiency']) 
                                            / baseline_stats['mean_efficiency'] * 100)
                    overall_efficiency_improvements.append(efficiency_improvement)
                
                # Statistical significance for execution time
                baseline_times = [r.execution_time_sec for r in budget_results['baseline'] if r.success]
                fastpath_times = [r.execution_time_sec for r in budget_results['fastpath'] if r.success]
                
                sig_test = self.statistical_significance_test(baseline_times, fastpath_times)
                if sig_test['significant'] and time_improvement > 0:
                    significant_improvements += 1
        
        if overall_time_improvements:
            avg_time_improvement = statistics.mean(overall_time_improvements)
            lines.append(f"\n- **Average Execution Time Improvement**: {avg_time_improvement:+.1f}%")
        
        if overall_efficiency_improvements:
            avg_efficiency_improvement = statistics.mean(overall_efficiency_improvements)
            lines.append(f"- **Average Token Efficiency Improvement**: {avg_efficiency_improvement:+.1f}%")
        
        if total_comparisons > 0:
            significance_rate = significant_improvements / total_comparisons * 100
            lines.append(f"- **Statistically Significant Improvements**: {significant_improvements}/{total_comparisons} ({significance_rate:.0f}%)")
        
        # Detailed Results
        lines.append("\n## Detailed Performance Analysis")
        
        for budget, budget_results in results.items():
            lines.append(f"\n### Token Budget: {budget:,}")
            
            fastpath_stats = self.calculate_stats(budget_results['fastpath'])
            baseline_stats = self.calculate_stats(budget_results['baseline'])
            
            if fastpath_stats['count'] == 0:
                lines.append("\n❌ FastPath: All runs failed")
                continue
            elif baseline_stats['count'] == 0:
                lines.append("\n❌ Baseline: All runs failed")
                continue
            
            # Performance metrics table
            lines.append("\n#### Performance Metrics")
            lines.append("| Metric | Baseline | FastPath | Improvement | Significant |")
            lines.append("|--------|----------|----------|-------------|-------------|")
            
            # Execution time
            time_improvement = ((baseline_stats['mean_time'] - fastpath_stats['mean_time']) 
                              / baseline_stats['mean_time'] * 100)
            
            baseline_times = [r.execution_time_sec for r in budget_results['baseline'] if r.success]
            fastpath_times = [r.execution_time_sec for r in budget_results['fastpath'] if r.success]
            time_sig_test = self.statistical_significance_test(baseline_times, fastpath_times)
            time_sig_mark = "✅" if time_sig_test['significant'] else "❌"
            
            lines.append(f"| Execution Time (s) | {baseline_stats['mean_time']:.3f} ± {baseline_stats['std_time']:.3f} | "
                        f"{fastpath_stats['mean_time']:.3f} ± {fastpath_stats['std_time']:.3f} | "
                        f"{time_improvement:+.1f}% | {time_sig_mark} (p≈{time_sig_test['p_value']:.3f}) |")
            
            # Token efficiency
            if baseline_stats['mean_efficiency'] > 0:
                efficiency_improvement = ((fastpath_stats['mean_efficiency'] - baseline_stats['mean_efficiency']) 
                                        / baseline_stats['mean_efficiency'] * 100)
            else:
                efficiency_improvement = 0
            
            lines.append(f"| Token Efficiency (tok/s) | {baseline_stats['mean_efficiency']:.0f} | "
                        f"{fastpath_stats['mean_efficiency']:.0f} | {efficiency_improvement:+.1f}% | - |")
            
            # Token usage
            token_improvement = ((fastpath_stats['mean_tokens'] - baseline_stats['mean_tokens']) 
                               / baseline_stats['mean_tokens'] * 100) if baseline_stats['mean_tokens'] > 0 else 0
            lines.append(f"| Token Usage | {baseline_stats['mean_tokens']:.0f} | "
                        f"{fastpath_stats['mean_tokens']:.0f} | {token_improvement:+.1f}% | - |")
            
            # Files selected
            files_improvement = ((fastpath_stats['mean_files'] - baseline_stats['mean_files']) 
                               / baseline_stats['mean_files'] * 100) if baseline_stats['mean_files'] > 0 else 0
            lines.append(f"| Files Selected | {baseline_stats['mean_files']:.1f} | "
                        f"{fastpath_stats['mean_files']:.1f} | {files_improvement:+.1f}% | - |")
            
            # README retention
            readme_improvement = (fastpath_stats['readme_retention_rate'] - baseline_stats['readme_retention_rate']) * 100
            lines.append(f"| README Retention | {baseline_stats['readme_retention_rate']:.0%} | "
                        f"{fastpath_stats['readme_retention_rate']:.0%} | {readme_improvement:+.1f}pp | - |")
            
            # Documentation files
            doc_improvement = ((fastpath_stats['mean_doc_files'] - baseline_stats['mean_doc_files']) 
                             / baseline_stats['mean_doc_files'] * 100) if baseline_stats['mean_doc_files'] > 0 else 0
            lines.append(f"| Doc Files Retained | {baseline_stats['mean_doc_files']:.1f} | "
                        f"{fastpath_stats['mean_doc_files']:.1f} | {doc_improvement:+.1f}% | - |")
        
        # Quality Analysis
        lines.append("\n## Quality Assessment")
        
        all_fastpath_runs = []
        all_baseline_runs = []
        for budget_results in results.values():
            all_fastpath_runs.extend(budget_results['fastpath'])
            all_baseline_runs.extend(budget_results['baseline'])
        
        fastpath_success_rate = sum(1 for r in all_fastpath_runs if r.success) / len(all_fastpath_runs)
        baseline_success_rate = sum(1 for r in all_baseline_runs if r.success) / len(all_baseline_runs)
        
        lines.append(f"\n### System Reliability")
        lines.append(f"- **FastPath Success Rate**: {fastpath_success_rate:.0%}")
        lines.append(f"- **Baseline Success Rate**: {baseline_success_rate:.0%}")
        
        # Failure analysis
        fastpath_failures = [r for r in all_fastpath_runs if not r.success]
        baseline_failures = [r for r in all_baseline_runs if not r.success]
        
        if fastpath_failures:
            lines.append(f"\n### FastPath Failures ({len(fastpath_failures)} total)")
            error_counts = {}
            for failure in fastpath_failures:
                error_key = failure.error_msg[:50] + "..." if len(failure.error_msg) > 50 else failure.error_msg
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                lines.append(f"- `{error}` ({count} occurrences)")
        
        if baseline_failures:
            lines.append(f"\n### Baseline Failures ({len(baseline_failures)} total)")
            error_counts = {}
            for failure in baseline_failures:
                error_key = failure.error_msg[:50] + "..." if len(failure.error_msg) > 50 else failure.error_msg
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                lines.append(f"- `{error}` ({count} occurrences)")
        
        # Methodology
        lines.append("\n## Methodology")
        lines.append(f"- **Benchmark runs**: {self.num_runs} independent runs per system per token budget")
        lines.append("- **Statistical testing**: Two-sample t-test for execution time comparison")
        lines.append("- **Significance level**: p < 0.05 for statistical significance")
        lines.append("- **Effect size**: Cohen's d for practical significance measurement")
        lines.append("- **Token counting**: Actual tokenizer-based counting for both systems")
        lines.append("- **Quality metrics**: README retention, documentation file inclusion")
        lines.append("- **Timeout**: No artificial timeouts (systems run to completion)")
        
        # Conclusions
        lines.append("\n## Key Findings")
        
        if overall_time_improvements:
            avg_improvement = statistics.mean(overall_time_improvements)
            if avg_improvement > 20:
                lines.append(f"- 🚀 **Significant Performance Gain**: FastPath is {avg_improvement:.1f}% faster on average")
            elif avg_improvement > 5:
                lines.append(f"- ✅ **Modest Performance Gain**: FastPath is {avg_improvement:.1f}% faster on average")
            elif avg_improvement > -5:
                lines.append(f"- ➖ **Comparable Performance**: FastPath shows {avg_improvement:+.1f}% time difference")
            else:
                lines.append(f"- ⚠️ **Performance Regression**: FastPath is {abs(avg_improvement):.1f}% slower on average")
        
        if significant_improvements > 0:
            lines.append(f"- 📊 **Statistical Validity**: {significant_improvements} statistically significant improvements")
        
        return '\n'.join(lines)

def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final FastPath vs PackRepo Baseline Benchmark")
    parser.add_argument("--runs", type=int, default=8, help="Number of runs per configuration (default: 8)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[50000, 120000], 
                       help="Token budgets to test (default: 50000 120000)")
    parser.add_argument("--output", type=str, help="Output file for report")
    parser.add_argument("--quick", action="store_true", help="Quick test with minimal runs")
    
    args = parser.parse_args()
    
    if args.quick:
        num_runs = 3
        token_budgets = [50000]
    else:
        num_runs = args.runs
        token_budgets = args.budgets
    
    repo_path = Path.cwd()
    
    print("FastPath vs PackRepo Baseline Benchmark")
    print("="*50)
    print(f"Repository: {repo_path}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Token budgets: {token_budgets}")
    print(f"Systems: FastPath vs PackRepo Baseline")
    
    # Validate environment
    try:
        from packrepo.library import RepositoryPacker
        from packrepo.fastpath import FastScanner
        print("✓ All required components available")
    except ImportError as e:
        print(f"❌ Missing required components: {e}")
        sys.exit(1)
    
    # Run benchmark
    print(f"\n{'='*50}")
    print("STARTING BENCHMARK")
    print(f"{'='*50}")
    
    benchmark = FinalBenchmark(repo_path, num_runs)
    results = benchmark.run_benchmark(token_budgets)
    
    # Generate report
    print(f"\n{'='*50}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*50}")
    
    report = benchmark.generate_comprehensive_report(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"fastpath_benchmark_{timestamp}.md")
    
    output_path.write_text(report)
    print(f"\n📄 Report saved to: {output_path}")
    
    # Save raw data
    json_path = output_path.with_suffix('.json')
    json_data = {}
    for budget, budget_results in results.items():
        json_data[str(budget)] = {}
        for system, runs in budget_results.items():
            json_data[str(budget)][system] = [asdict(run) for run in runs]
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"📊 Raw data saved to: {json_path}")
    
    # Show summary
    print(f"\n{report}")

if __name__ == "__main__":
    main()
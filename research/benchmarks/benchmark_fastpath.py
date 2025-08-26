#!/usr/bin/env python3
"""
Comprehensive Real-World Benchmarking System for FastPath vs Baseline PackRepo

This system provides rigorous empirical validation of FastPath performance claims
with statistical significance testing, real repository validation, and comprehensive
performance analysis across multiple dimensions.

Key Features:
- Real-world repository testing with different sizes/languages
- Statistical rigor with confidence intervals and significance testing
- Comprehensive performance metrics (latency, memory, token efficiency)
- Quality assessment and retention analysis
- Automated visualization and reporting
- Production-ready benchmark suite
"""

import gc
import json
import logging
import multiprocessing
import os
import subprocess
import statistics
import tempfile
import time
import tracemalloc
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RepositorySpec:
    """Specification for a test repository."""
    name: str
    url: str
    language: str
    category: str  # small, medium, large
    expected_files: int
    expected_loc: int
    clone_depth: Optional[int] = 1  # Shallow clone by default
    description: str = ""


@dataclass 
class BenchmarkMetrics:
    """Comprehensive performance metrics for a single benchmark run."""
    run_id: int
    system_name: str  # fastpath, baseline, extended
    repo_name: str
    token_budget: int
    
    # Performance metrics
    execution_time_ms: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    
    # Output quality metrics
    total_tokens_used: int
    total_files_selected: int
    readme_retained: bool
    doc_files_count: int
    code_files_count: int
    
    # Token efficiency (key metric)
    tokens_per_second: float
    timestamp: str
    qa_score: Optional[float] = None  # Simulated QA accuracy
    token_efficiency: Optional[float] = None  # QA score per 100k tokens
    
    # Metadata
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results for performance comparison."""
    metric_name: str
    baseline_mean: float
    treatment_mean: float
    improvement_percent: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    significant: bool
    power: Optional[float] = None


@dataclass
class BenchmarkResults:
    """Complete benchmark results for a repository/configuration."""
    repo_name: str
    token_budget: int
    baseline_metrics: List[BenchmarkMetrics]
    fastpath_metrics: List[BenchmarkMetrics]
    extended_metrics: List[BenchmarkMetrics]
    
    # Statistical comparisons
    statistical_analyses: Dict[str, StatisticalAnalysis]
    
    # Summary statistics
    summary_stats: Dict[str, Dict[str, float]]


class RealWorldRepositories:
    """Real-world repository specifications for benchmarking."""
    
    SMALL_REPOS = [
        RepositorySpec(
            name="flask-minimal",
            url="https://github.com/pallets/flask.git",
            language="python",
            category="small",
            expected_files=50,
            expected_loc=5000,
            description="Flask web framework - small Python project"
        ),
        RepositorySpec(
            name="express-minimal", 
            url="https://github.com/expressjs/express.git",
            language="javascript",
            category="small",
            expected_files=60,
            expected_loc=8000,
            description="Express.js web framework - small JavaScript project"
        ),
    ]
    
    MEDIUM_REPOS = [
        RepositorySpec(
            name="requests",
            url="https://github.com/psf/requests.git", 
            language="python",
            category="medium",
            expected_files=200,
            expected_loc=25000,
            description="Python HTTP library - medium complexity"
        ),
        RepositorySpec(
            name="axios",
            url="https://github.com/axios/axios.git",
            language="javascript", 
            category="medium",
            expected_files=150,
            expected_loc=15000,
            description="Promise-based HTTP client - medium JavaScript project"
        ),
        RepositorySpec(
            name="serde",
            url="https://github.com/serde-rs/serde.git",
            language="rust",
            category="medium", 
            expected_files=300,
            expected_loc=30000,
            description="Rust serialization framework - medium complexity"
        ),
    ]
    
    LARGE_REPOS = [
        RepositorySpec(
            name="django",
            url="https://github.com/django/django.git",
            language="python",
            category="large",
            expected_files=2000,
            expected_loc=200000,
            clone_depth=1,  # Shallow clone for large repos
            description="Django web framework - large Python project"
        ),
        RepositorySpec(
            name="typescript",
            url="https://github.com/microsoft/TypeScript.git",
            language="typescript",
            category="large",
            expected_files=1500,
            expected_loc=150000,
            clone_depth=1,
            description="TypeScript compiler - large TypeScript project"
        ),
    ]
    
    @classmethod
    def get_all_repos(cls) -> List[RepositorySpec]:
        """Get all repository specifications."""
        return cls.SMALL_REPOS + cls.MEDIUM_REPOS + cls.LARGE_REPOS
    
    @classmethod
    def get_repos_by_category(cls, category: str) -> List[RepositorySpec]:
        """Get repositories by size category."""
        all_repos = cls.get_all_repos()
        return [repo for repo in all_repos if repo.category == category]


class RepositoryManager:
    """Manages cloning and preparing test repositories."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def clone_repository(self, spec: RepositorySpec) -> Path:
        """Clone a repository for testing."""
        repo_dir = self.base_dir / spec.name
        
        if repo_dir.exists():
            logger.info(f"Repository {spec.name} already exists, skipping clone")
            return repo_dir
            
        logger.info(f"Cloning {spec.name} from {spec.url}")
        
        cmd = ["git", "clone"]
        if spec.clone_depth:
            cmd.extend(["--depth", str(spec.clone_depth)])
        cmd.extend([spec.url, str(repo_dir)])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully cloned {spec.name}")
            return repo_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {spec.name}: {e.stderr}")
            raise
    
    def prepare_all_repositories(self, specs: List[RepositorySpec]) -> Dict[str, Path]:
        """Clone all specified repositories."""
        repos = {}
        
        for spec in specs:
            try:
                repo_path = self.clone_repository(spec)
                repos[spec.name] = repo_path
            except Exception as e:
                logger.error(f"Failed to prepare {spec.name}: {e}")
                continue
                
        return repos


class FastPathBenchmarker:
    """Benchmarks FastPath performance against baseline."""
    
    def __init__(self, num_runs: int = 10, warmup_runs: int = 3):
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        
    def _run_baseline_packrepo(self, repo_path: Path, token_budget: int, 
                              run_id: int) -> BenchmarkMetrics:
        """Run baseline PackRepo system."""
        
        # Start monitoring
        tracemalloc.start()
        gc.collect()
        process = psutil.Process()
        start_time = time.perf_counter()
        
        try:
            # Import and run baseline PackRepo
            from packrepo import pack_repository
            
            result = pack_repository(
                repo_path, 
                target_budget=token_budget,
                chunk_size=min(4000, token_budget // 4),
                overlap_size=200
            )
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Collect memory stats
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_peak_mb = peak_memory / 1024 / 1024
            memory_current_mb = current_memory / 1024 / 1024
            
            # Calculate CPU usage
            cpu_percent = process.cpu_percent()
            
            # Analyze output quality
            readme_retained = any('readme' in chunk.metadata.get('file_path', '').lower() 
                                for chunk in result.chunks)
            doc_files = sum(1 for chunk in result.chunks 
                           if any(ext in chunk.metadata.get('file_path', '').lower() 
                                 for ext in ['.md', '.rst', '.txt', 'readme', 'changelog']))
            code_files = len(result.chunks) - doc_files
            
            return BenchmarkMetrics(
                run_id=run_id,
                system_name="baseline",
                repo_name=repo_path.name,
                token_budget=token_budget,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=memory_peak_mb,
                memory_current_mb=memory_current_mb,
                cpu_percent=cpu_percent,
                total_tokens_used=result.total_tokens,
                total_files_selected=len(result.chunks),
                readme_retained=readme_retained,
                doc_files_count=doc_files,
                code_files_count=code_files,
                tokens_per_second=result.total_tokens / (execution_time_ms / 1000),
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            tracemalloc.stop()
            
            return BenchmarkMetrics(
                run_id=run_id,
                system_name="baseline",
                repo_name=repo_path.name,
                token_budget=token_budget,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=0,
                memory_current_mb=0,
                cpu_percent=0,
                total_tokens_used=0,
                total_files_selected=0,
                readme_retained=False,
                doc_files_count=0,
                code_files_count=0,
                tokens_per_second=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def _run_fastpath_packrepo(self, repo_path: Path, token_budget: int,
                              run_id: int, mode: str = "fast") -> BenchmarkMetrics:
        """Run FastPath PackRepo system."""
        
        # Start monitoring
        tracemalloc.start()
        gc.collect()
        process = psutil.Process()
        start_time = time.perf_counter()
        
        try:
            # Run FastPath CLI
            cmd = [
                "python", "-m", "packrepo.cli.fastpack",
                str(repo_path),
                "--budget", str(token_budget),
                "--mode", mode,
                "--output", "-"  # stdout
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, 
                                   cwd=Path(__file__).parent)
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Collect memory stats  
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_peak_mb = peak_memory / 1024 / 1024
            memory_current_mb = current_memory / 1024 / 1024
            
            # Calculate CPU usage
            cpu_percent = process.cpu_percent()
            
            # Parse output to get metrics
            output = result.stdout
            total_tokens = len(output.split()) * 1.3  # Rough token estimate
            
            # Analyze quality
            readme_retained = 'readme' in output.lower()
            doc_sections = output.lower().count('.md') + output.lower().count('readme') + output.lower().count('changelog')
            
            return BenchmarkMetrics(
                run_id=run_id,
                system_name=mode,
                repo_name=repo_path.name,
                token_budget=token_budget,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=memory_peak_mb,
                memory_current_mb=memory_current_mb,
                cpu_percent=cpu_percent,
                total_tokens_used=int(total_tokens),
                total_files_selected=output.count('```'),  # Rough file count
                readme_retained=readme_retained,
                doc_files_count=doc_sections,
                code_files_count=max(0, output.count('```') - doc_sections),
                tokens_per_second=int(total_tokens) / (execution_time_ms / 1000),
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            tracemalloc.stop()
            
            return BenchmarkMetrics(
                run_id=run_id,
                system_name=mode,
                repo_name=repo_path.name,
                token_budget=token_budget,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=0,
                memory_current_mb=0,
                cpu_percent=0,
                total_tokens_used=0,
                total_files_selected=0,
                readme_retained=False,
                doc_files_count=0,
                code_files_count=0,
                tokens_per_second=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def benchmark_repository(self, repo_path: Path, token_budgets: List[int]) -> Dict[int, BenchmarkResults]:
        """Run comprehensive benchmark on a single repository."""
        results = {}
        
        for budget in token_budgets:
            logger.info(f"Benchmarking {repo_path.name} with {budget} token budget")
            
            # Run warmup
            logger.info(f"  Running {self.warmup_runs} warmup runs...")
            for i in range(self.warmup_runs):
                self._run_baseline_packrepo(repo_path, budget, i)
            
            # Collect metrics for all systems
            baseline_metrics = []
            fastpath_metrics = []
            extended_metrics = []
            
            logger.info(f"  Running {self.num_runs} benchmark runs...")
            
            for run_id in range(self.num_runs):
                # Baseline
                baseline_result = self._run_baseline_packrepo(repo_path, budget, run_id)
                baseline_metrics.append(baseline_result)
                
                # FastPath
                fastpath_result = self._run_fastpath_packrepo(repo_path, budget, run_id, "fast")
                fastpath_metrics.append(fastpath_result)
                
                # Extended
                extended_result = self._run_fastpath_packrepo(repo_path, budget, run_id, "extended")
                extended_metrics.append(extended_result)
                
                if (run_id + 1) % 5 == 0:
                    logger.info(f"    Completed {run_id + 1}/{self.num_runs} runs")
            
            # Perform statistical analysis
            statistical_analyses = self._perform_statistical_analysis(
                baseline_metrics, fastpath_metrics, extended_metrics
            )
            
            # Generate summary statistics
            summary_stats = self._generate_summary_statistics(
                baseline_metrics, fastpath_metrics, extended_metrics
            )
            
            results[budget] = BenchmarkResults(
                repo_name=repo_path.name,
                token_budget=budget,
                baseline_metrics=baseline_metrics,
                fastpath_metrics=fastpath_metrics,
                extended_metrics=extended_metrics,
                statistical_analyses=statistical_analyses,
                summary_stats=summary_stats
            )
            
        return results
    
    def _perform_statistical_analysis(self, baseline: List[BenchmarkMetrics],
                                     fastpath: List[BenchmarkMetrics],
                                     extended: List[BenchmarkMetrics]) -> Dict[str, StatisticalAnalysis]:
        """Perform statistical significance testing."""
        analyses = {}
        
        # Filter successful runs only
        baseline_success = [m for m in baseline if m.success]
        fastpath_success = [m for m in fastpath if m.success]
        extended_success = [m for m in extended if m.success]
        
        if not (baseline_success and fastpath_success):
            return analyses
        
        metrics_to_analyze = [
            ('execution_time_ms', lambda m: m.execution_time_ms),
            ('memory_peak_mb', lambda m: m.memory_peak_mb),
            ('tokens_per_second', lambda m: m.tokens_per_second),
            ('total_files_selected', lambda m: m.total_files_selected)
        ]
        
        for metric_name, extractor in metrics_to_analyze:
            baseline_values = [extractor(m) for m in baseline_success]
            fastpath_values = [extractor(m) for m in fastpath_success]
            
            if len(baseline_values) < 3 or len(fastpath_values) < 3:
                continue
                
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(fastpath_values, baseline_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) +
                                 (len(fastpath_values) - 1) * np.var(fastpath_values, ddof=1)) /
                                (len(baseline_values) + len(fastpath_values) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(fastpath_values) - np.mean(baseline_values)) / pooled_std
            else:
                cohens_d = 0
                
            # Confidence interval for difference in means
            se_diff = pooled_std * np.sqrt(1/len(baseline_values) + 1/len(fastpath_values))
            df = len(baseline_values) + len(fastpath_values) - 2
            t_critical = stats.t.ppf(0.975, df)
            diff = np.mean(fastpath_values) - np.mean(baseline_values)
            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff
            
            # Calculate improvement percentage
            baseline_mean = np.mean(baseline_values)
            fastpath_mean = np.mean(fastpath_values)
            
            if metric_name in ['execution_time_ms', 'memory_peak_mb']:
                # Lower is better
                improvement_percent = (baseline_mean - fastpath_mean) / baseline_mean * 100
            else:
                # Higher is better
                improvement_percent = (fastpath_mean - baseline_mean) / baseline_mean * 100
            
            analyses[f"fastpath_vs_baseline_{metric_name}"] = StatisticalAnalysis(
                metric_name=metric_name,
                baseline_mean=baseline_mean,
                treatment_mean=fastpath_mean,
                improvement_percent=improvement_percent,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                effect_size=cohens_d,
                significant=p_value < 0.05
            )
        
        return analyses
    
    def _generate_summary_statistics(self, baseline: List[BenchmarkMetrics],
                                   fastpath: List[BenchmarkMetrics], 
                                   extended: List[BenchmarkMetrics]) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics for all systems."""
        
        def compute_stats(metrics: List[BenchmarkMetrics], metric_name: str) -> Dict[str, float]:
            values = [getattr(m, metric_name) for m in metrics if m.success]
            if not values:
                return {'mean': 0, 'median': 0, 'p95': 0, 'std': 0, 'min': 0, 'max': 0}
            
            return {
                'mean': np.mean(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        stats_dict = {}
        
        for system_name, metrics in [('baseline', baseline), ('fastpath', fastpath), ('extended', extended)]:
            stats_dict[system_name] = {}
            
            for metric in ['execution_time_ms', 'memory_peak_mb', 'tokens_per_second', 'total_files_selected']:
                stats_dict[system_name][metric] = compute_stats(metrics, metric)
        
        return stats_dict


class BenchmarkReporter:
    """Generates comprehensive benchmark reports with visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_report(self, all_results: Dict[str, Dict[int, BenchmarkResults]],
                                    repo_specs: List[RepositorySpec]) -> None:
        """Generate comprehensive benchmark report."""
        
        # Generate text report
        report_path = self.output_dir / "fastpath_benchmark_report.md"
        self._generate_text_report(all_results, repo_specs, report_path)
        
        # Generate JSON data
        json_path = self.output_dir / "benchmark_results.json"
        self._save_json_results(all_results, json_path)
        
        # Generate visualizations
        self._generate_visualizations(all_results)
        
        logger.info(f"Comprehensive benchmark report saved to {self.output_dir}")
    
    def _generate_text_report(self, all_results: Dict[str, Dict[int, BenchmarkResults]],
                            repo_specs: List[RepositorySpec], output_path: Path) -> None:
        """Generate detailed text report."""
        
        with open(output_path, 'w') as f:
            f.write("# FastPath Performance Benchmark Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Overall performance summary
            total_improvements = 0
            significant_improvements = 0
            
            for repo_name, budget_results in all_results.items():
                for budget, results in budget_results.items():
                    for analysis_name, analysis in results.statistical_analyses.items():
                        if 'fastpath_vs_baseline' in analysis_name:
                            total_improvements += 1
                            if analysis.significant and analysis.improvement_percent > 0:
                                significant_improvements += 1
            
            success_rate = (significant_improvements / total_improvements * 100) if total_improvements > 0 else 0
            
            f.write(f"- **Total Performance Tests**: {total_improvements}\n")
            f.write(f"- **Statistically Significant Improvements**: {significant_improvements}\n")
            f.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")
            
            # Repository-by-repository analysis
            f.write("## Repository Performance Analysis\n\n")
            
            for repo_name, budget_results in all_results.items():
                repo_spec = next((spec for spec in repo_specs if spec.name == repo_name), None)
                
                f.write(f"### {repo_name}\n\n")
                if repo_spec:
                    f.write(f"- **Language**: {repo_spec.language}\n")
                    f.write(f"- **Category**: {repo_spec.category}\n")
                    f.write(f"- **Description**: {repo_spec.description}\n\n")
                
                for budget, results in budget_results.items():
                    f.write(f"#### Token Budget: {budget:,}\n\n")
                    
                    # Performance metrics table
                    f.write("| System | Latency (ms) | Memory (MB) | Tokens/sec | Files Selected | README Retained |\n")
                    f.write("|--------|--------------|-------------|------------|----------------|----------------|\n")
                    
                    for system_name in ['baseline', 'fastpath', 'extended']:
                        if system_name in results.summary_stats:
                            stats = results.summary_stats[system_name]
                            
                            # Get README retention rate
                            if system_name == 'baseline':
                                metrics = results.baseline_metrics
                            elif system_name == 'fastpath':
                                metrics = results.fastpath_metrics
                            else:
                                metrics = results.extended_metrics
                            
                            readme_rate = sum(1 for m in metrics if m.readme_retained) / len(metrics) * 100
                            
                            f.write(f"| {system_name} | {stats['execution_time_ms']['mean']:.0f} | ")
                            f.write(f"{stats['memory_peak_mb']['mean']:.1f} | ")
                            f.write(f"{stats['tokens_per_second']['mean']:.0f} | ")
                            f.write(f"{stats['total_files_selected']['mean']:.0f} | ")
                            f.write(f"{readme_rate:.0f}% |\n")
                    
                    f.write("\n")
                    
                    # Statistical significance results
                    f.write("**Statistical Analysis:**\n\n")
                    
                    for analysis_name, analysis in results.statistical_analyses.items():
                        if 'fastpath_vs_baseline' in analysis_name:
                            metric_display = analysis.metric_name.replace('_', ' ').title()
                            significance = "✅" if analysis.significant else "❌"
                            
                            f.write(f"- **{metric_display}**: {analysis.improvement_percent:+.1f}% ")
                            f.write(f"(p={analysis.p_value:.4f}) {significance}\n")
                    
                    f.write("\n")
                    
            f.write("## Methodology\n\n")
            f.write("- **Benchmark Runs**: Multiple runs per configuration for statistical validity\n")
            f.write("- **Statistical Testing**: Two-sample t-tests with 95% confidence intervals\n")
            f.write("- **Effect Size**: Cohen's d calculation for practical significance\n")
            f.write("- **Real Repositories**: Testing on actual open-source projects\n")
            f.write("- **Multiple Budgets**: Testing across different token budget constraints\n")
    
    def _save_json_results(self, all_results: Dict[str, Dict[int, BenchmarkResults]],
                          output_path: Path) -> None:
        """Save results in JSON format for further analysis."""
        
        # Convert to JSON-serializable format
        json_results = {}
        
        for repo_name, budget_results in all_results.items():
            json_results[repo_name] = {}
            
            for budget, results in budget_results.items():
                json_results[repo_name][str(budget)] = {
                    'summary_stats': results.summary_stats,
                    'statistical_analyses': {
                        name: asdict(analysis) 
                        for name, analysis in results.statistical_analyses.items()
                    },
                    'raw_metrics': {
                        'baseline': [asdict(m) for m in results.baseline_metrics],
                        'fastpath': [asdict(m) for m in results.fastpath_metrics],
                        'extended': [asdict(m) for m in results.extended_metrics]
                    }
                }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _generate_visualizations(self, all_results: Dict[str, Dict[int, BenchmarkResults]]) -> None:
        """Generate performance visualization charts."""
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Performance comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FastPath vs Baseline Performance Comparison', fontsize=16)
        
        # Collect data for plotting
        metrics_data = {
            'execution_time_ms': {'baseline': [], 'fastpath': [], 'extended': []},
            'memory_peak_mb': {'baseline': [], 'fastpath': [], 'extended': []},
            'tokens_per_second': {'baseline': [], 'fastpath': [], 'extended': []},
            'total_files_selected': {'baseline': [], 'fastpath': [], 'extended': []}
        }
        
        for repo_name, budget_results in all_results.items():
            for budget, results in budget_results.items():
                for system in ['baseline', 'fastpath', 'extended']:
                    if system in results.summary_stats:
                        for metric in metrics_data.keys():
                            if metric in results.summary_stats[system]:
                                metrics_data[metric][system].append(
                                    results.summary_stats[system][metric]['mean']
                                )
        
        # Plot comparisons
        metrics_info = [
            ('execution_time_ms', 'Execution Time (ms)', axes[0,0]),
            ('memory_peak_mb', 'Peak Memory (MB)', axes[0,1]),
            ('tokens_per_second', 'Tokens per Second', axes[1,0]),
            ('total_files_selected', 'Files Selected', axes[1,1])
        ]
        
        for metric, title, ax in metrics_info:
            data_to_plot = []
            labels = []
            
            for system in ['baseline', 'fastpath', 'extended']:
                if metrics_data[metric][system]:
                    data_to_plot.append(metrics_data[metric][system])
                    labels.append(system.title())
            
            if data_to_plot:
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Improvement heatmap
        self._generate_improvement_heatmap(all_results)
    
    def _generate_improvement_heatmap(self, all_results: Dict[str, Dict[int, BenchmarkResults]]) -> None:
        """Generate heatmap showing improvement percentages."""
        
        # Collect improvement data
        repos = list(all_results.keys())
        budgets = []
        for budget_results in all_results.values():
            budgets.extend(budget_results.keys())
        budgets = sorted(list(set(budgets)))
        
        improvement_matrix = np.zeros((len(repos), len(budgets)))
        
        for i, repo in enumerate(repos):
            for j, budget in enumerate(budgets):
                if budget in all_results[repo]:
                    results = all_results[repo][budget]
                    # Get execution time improvement as representative metric
                    exec_time_analysis = results.statistical_analyses.get(
                        'fastpath_vs_baseline_execution_time_ms'
                    )
                    if exec_time_analysis:
                        improvement_matrix[i, j] = exec_time_analysis.improvement_percent
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(improvement_matrix, 
                   xticklabels=[f"{b//1000}k" for b in budgets],
                   yticklabels=repos,
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn',
                   center=0,
                   cbar_kws={'label': 'Latency Improvement (%)'})
        
        plt.title('FastPath Latency Improvement Over Baseline')
        plt.xlabel('Token Budget')
        plt.ylabel('Repository')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


class ComprehensiveBenchmarkSuite:
    """Main benchmark suite orchestrator."""
    
    def __init__(self, output_dir: Path, num_runs: int = 15):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.repo_manager = RepositoryManager(self.output_dir / "repositories")
        self.benchmarker = FastPathBenchmarker(num_runs=num_runs, warmup_runs=3)
        self.reporter = BenchmarkReporter(self.output_dir / "reports")
        
    def run_comprehensive_benchmark(self, categories: List[str] = None, 
                                  token_budgets: List[int] = None) -> None:
        """Run comprehensive benchmark suite."""
        
        if categories is None:
            categories = ['small', 'medium']  # Skip large by default for time
            
        if token_budgets is None:
            token_budgets = [50000, 120000, 200000]
        
        logger.info("Starting comprehensive FastPath benchmark suite")
        logger.info(f"Categories: {categories}")
        logger.info(f"Token budgets: {token_budgets}")
        
        # Get repository specifications
        all_repo_specs = RealWorldRepositories.get_all_repos()
        target_specs = [spec for spec in all_repo_specs if spec.category in categories]
        
        logger.info(f"Target repositories: {[spec.name for spec in target_specs]}")
        
        # Clone repositories
        logger.info("Preparing repositories...")
        repo_paths = self.repo_manager.prepare_all_repositories(target_specs)
        
        # Run benchmarks
        all_results = {}
        
        for spec in target_specs:
            if spec.name not in repo_paths:
                logger.warning(f"Skipping {spec.name} - repository not available")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"BENCHMARKING: {spec.name} ({spec.language}, {spec.category})")
            logger.info(f"{'='*60}")
            
            repo_path = repo_paths[spec.name]
            
            try:
                results = self.benchmarker.benchmark_repository(repo_path, token_budgets)
                all_results[spec.name] = results
                
                logger.info(f"Completed benchmarking {spec.name}")
                
            except Exception as e:
                logger.error(f"Failed to benchmark {spec.name}: {e}")
                continue
        
        # Generate reports
        logger.info("\nGenerating comprehensive reports...")
        self.reporter.generate_comprehensive_report(all_results, target_specs)
        
        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK SUITE COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point for benchmark suite."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive FastPath vs Baseline Performance Benchmark"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path("fastpath_benchmarks"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=['small', 'medium', 'large'],
        default=['small', 'medium'],
        help="Repository size categories to test"
    )
    
    parser.add_argument(
        '--budgets',
        nargs='+',
        type=int,
        default=[50000, 120000, 200000],
        help="Token budgets to test"
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=15,
        help="Number of benchmark runs per configuration"
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Quick benchmark with fewer runs and smaller repos"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        categories = ['small']
        budgets = [120000]
        runs = 5
    else:
        categories = args.categories
        budgets = args.budgets  
        runs = args.runs
    
    # Initialize and run benchmark suite
    suite = ComprehensiveBenchmarkSuite(args.output_dir, num_runs=runs)
    suite.run_comprehensive_benchmark(categories=categories, token_budgets=budgets)


if __name__ == '__main__':
    main()
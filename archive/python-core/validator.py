#!/usr/bin/env python3
"""
Validation System for Intelligent Scribe Defaults

This module tests the intelligent default system across all repositories,
ensuring 100% success rate and validating optimal token utilization.
"""

import os
import pathlib
import json
import time
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from repository_analyzer import RepositoryAnalyzer, RepositoryAnalysis
from config_generator import ConfigGenerator, ScribeConfig
from smart_filter import SmartFilter


@dataclass
class ValidationResult:
    """Result of validating scribe on a single repository."""
    repo_name: str
    repo_path: str
    success: bool
    
    # Configuration metrics
    predicted_tokens: int
    actual_tokens: Optional[int]
    token_utilization: Optional[float]  # actual / predicted
    
    # Performance metrics
    analysis_time_seconds: float
    config_generation_time_seconds: float
    scribe_execution_time_seconds: Optional[float]
    
    # Quality metrics
    source_files_found: int
    source_files_included: int
    inclusion_rate: float
    confidence_score: float
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Detailed results
    repository_analysis: Optional[RepositoryAnalysis] = None
    generated_config: Optional[ScribeConfig] = None


@dataclass
class ValidationSummary:
    """Overall validation results across all repositories."""
    total_repositories: int
    successful_repositories: int
    success_rate: float
    
    # Token utilization statistics
    avg_token_utilization: float
    median_token_utilization: float
    token_utilization_std: float
    
    # Performance statistics
    avg_analysis_time: float
    avg_config_time: float
    avg_execution_time: float
    
    # Quality statistics
    avg_inclusion_rate: float
    avg_confidence_score: float
    
    # Detailed results
    results: List[ValidationResult]
    failed_repos: List[str]
    
    # Recommendations
    recommendations: List[str]


class ScribeValidator:
    """Validates the intelligent default system across multiple repositories."""
    
    def __init__(self, projects_root: pathlib.Path = pathlib.Path("/home/nathan/Projects")):
        self.projects_root = projects_root
        self.analyzer = RepositoryAnalyzer
        self.config_generator = ConfigGenerator()
        self.smart_filter = SmartFilter()
    
    def discover_repositories(self, exclude_patterns: List[str] = None) -> List[pathlib.Path]:
        """Discover all valid repositories in the projects directory."""
        if exclude_patterns is None:
            exclude_patterns = [
                "rave",  # Exclude the problematic rave directory
                ".git",
                "__pycache__",
                "node_modules",
                "venv",
                ".venv",
                "gitlab-complete"  # Exclude GitLab installations
            ]
        
        repositories = []
        
        for item in self.projects_root.iterdir():
            if not item.is_dir():
                continue
                
            # Skip if matches exclude patterns
            if any(pattern in item.name.lower() for pattern in exclude_patterns):
                continue
                
            # Skip hidden directories
            if item.name.startswith('.'):
                continue
                
            # Check if it's a git repository or contains source code
            if (item / ".git").exists() or self._has_source_code(item):
                repositories.append(item)
        
        return sorted(repositories)
    
    def _has_source_code(self, repo_path: pathlib.Path) -> bool:
        """Quick check if directory contains source code files."""
        source_extensions = {'.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp'}
        
        for root, dirs, files in os.walk(repo_path):
            # Don't go too deep for performance
            depth = len(pathlib.Path(root).relative_to(repo_path).parts)
            if depth > 3:
                continue
                
            # Skip obvious build/dependency directories
            dirs[:] = [d for d in dirs if d not in {'node_modules', '__pycache__', '.git', 'venv', 'target'}]
            
            for filename in files:
                if pathlib.Path(filename).suffix.lower() in source_extensions:
                    return True
        
        return False
    
    def validate_single_repository(self, repo_path: pathlib.Path) -> ValidationResult:
        """Validate intelligent defaults on a single repository."""
        repo_name = repo_path.name
        start_time = time.time()
        
        try:
            print(f"🔍 Validating {repo_name}...")
            
            # Phase 1: Repository Analysis
            analysis_start = time.time()
            analyzer = RepositoryAnalyzer(repo_path)
            analysis = analyzer.analyze()
            analysis_time = time.time() - analysis_start
            
            # Phase 2: Configuration Generation
            config_start = time.time()
            config = self.config_generator.generate_config(repo_path, analysis)
            config_time = time.time() - config_start
            
            # Phase 3: Simulate Scribe Execution (dry run)
            execution_start = time.time()
            actual_tokens, scribe_success = self._simulate_scribe_execution(repo_path, config, analysis)
            execution_time = time.time() - execution_start
            
            # Calculate metrics
            token_utilization = actual_tokens / config.token_budget if actual_tokens else None
            inclusion_rate = len(analysis.source_files) / analysis.total_files if analysis.total_files > 0 else 0
            
            return ValidationResult(
                repo_name=repo_name,
                repo_path=str(repo_path),
                success=scribe_success,
                predicted_tokens=config.token_budget,
                actual_tokens=actual_tokens,
                token_utilization=token_utilization,
                analysis_time_seconds=analysis_time,
                config_generation_time_seconds=config_time,
                scribe_execution_time_seconds=execution_time,
                source_files_found=analysis.total_files,
                source_files_included=len(analysis.source_files),
                inclusion_rate=inclusion_rate,
                confidence_score=config.confidence_score,
                repository_analysis=analysis,
                generated_config=config
            )
            
        except Exception as e:
            return ValidationResult(
                repo_name=repo_name,
                repo_path=str(repo_path),
                success=False,
                predicted_tokens=0,
                actual_tokens=None,
                token_utilization=None,
                analysis_time_seconds=time.time() - start_time,
                config_generation_time_seconds=0,
                scribe_execution_time_seconds=None,
                source_files_found=0,
                source_files_included=0,
                inclusion_rate=0,
                confidence_score=0,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
    
    def _simulate_scribe_execution(self, repo_path: pathlib.Path, config: ScribeConfig, 
                                 analysis: RepositoryAnalysis) -> Tuple[Optional[int], bool]:
        """Simulate scribe execution to validate configuration."""
        try:
            # Apply smart filtering to get actual files that would be included
            all_files = []
            for root, dirs, filenames in os.walk(repo_path):
                # Apply directory exclusions
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in ['node_modules', '__pycache__', '.git'])]
                
                for filename in filenames:
                    file_path = pathlib.Path(root) / filename
                    all_files.append(file_path)
            
            # Filter using smart filter
            included_files, filter_stats = self.smart_filter.filter_files(all_files, repo_path)
            
            # Calculate actual tokens for included files
            actual_tokens = 0
            for file_path in included_files:
                try:
                    # Apply file size limit
                    if file_path.stat().st_size > config.max_file_size:
                        continue
                        
                    # Estimate tokens
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        tokens = max(1, len(content) // 4)  # Simple estimation
                        actual_tokens += tokens
                        
                    # Stop if we exceed budget (simulate scribe behavior)
                    if actual_tokens > config.token_budget:
                        actual_tokens = config.token_budget
                        break
                        
                except (OSError, UnicodeDecodeError, PermissionError):
                    continue
            
            return actual_tokens, True
            
        except Exception:
            return None, False
    
    def validate_all_repositories(self, max_workers: int = 4, limit: Optional[int] = None) -> ValidationSummary:
        """Validate intelligent defaults across all discovered repositories."""
        print("🚀 Starting comprehensive repository validation...")
        
        # Discover repositories
        repositories = self.discover_repositories()
        if limit:
            repositories = repositories[:limit]
        
        print(f"📁 Found {len(repositories)} repositories to validate")
        
        results = []
        failed_repos = []
        
        # Validate repositories in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_repo = {
                executor.submit(self.validate_single_repository, repo): repo.name 
                for repo in repositories
            }
            
            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        print(f"✅ {repo_name}: {result.token_utilization:.1%} utilization")
                    else:
                        print(f"❌ {repo_name}: {result.error_message}")
                        failed_repos.append(repo_name)
                        
                except Exception as e:
                    print(f"💥 {repo_name}: Validation failed with exception: {e}")
                    failed_repos.append(repo_name)
        
        # Calculate summary statistics
        return self._calculate_summary(results, failed_repos)
    
    def _calculate_summary(self, results: List[ValidationResult], 
                         failed_repos: List[str]) -> ValidationSummary:
        """Calculate validation summary statistics."""
        total_repos = len(results)
        successful_results = [r for r in results if r.success]
        success_count = len(successful_results)
        success_rate = success_count / total_repos if total_repos > 0 else 0
        
        # Token utilization statistics
        utilizations = [r.token_utilization for r in successful_results if r.token_utilization is not None]
        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
        median_utilization = sorted(utilizations)[len(utilizations)//2] if utilizations else 0
        
        # Calculate standard deviation
        if len(utilizations) > 1:
            mean = avg_utilization
            variance = sum((x - mean) ** 2 for x in utilizations) / len(utilizations)
            std_deviation = variance ** 0.5
        else:
            std_deviation = 0
        
        # Performance statistics
        analysis_times = [r.analysis_time_seconds for r in results]
        config_times = [r.config_generation_time_seconds for r in results]
        exec_times = [r.scribe_execution_time_seconds for r in results if r.scribe_execution_time_seconds is not None]
        
        avg_analysis_time = sum(analysis_times) / len(analysis_times) if analysis_times else 0
        avg_config_time = sum(config_times) / len(config_times) if config_times else 0
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        # Quality statistics
        inclusion_rates = [r.inclusion_rate for r in successful_results]
        confidence_scores = [r.confidence_score for r in successful_results]
        
        avg_inclusion_rate = sum(inclusion_rates) / len(inclusion_rates) if inclusion_rates else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(successful_results, failed_repos, 
                                                       avg_utilization, success_rate)
        
        return ValidationSummary(
            total_repositories=total_repos,
            successful_repositories=success_count,
            success_rate=success_rate,
            avg_token_utilization=avg_utilization,
            median_token_utilization=median_utilization,
            token_utilization_std=std_deviation,
            avg_analysis_time=avg_analysis_time,
            avg_config_time=avg_config_time,
            avg_execution_time=avg_exec_time,
            avg_inclusion_rate=avg_inclusion_rate,
            avg_confidence_score=avg_confidence,
            results=results,
            failed_repos=failed_repos,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, successful_results: List[ValidationResult],
                                failed_repos: List[str], avg_utilization: float,
                                success_rate: float) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Success rate recommendations
        if success_rate < 0.95:
            recommendations.append(f"Success rate is {success_rate:.1%}. Review failed repositories: {', '.join(failed_repos[:5])}")
        
        # Token utilization recommendations
        if avg_utilization < 0.5:
            recommendations.append("Low token utilization detected. Consider increasing inclusion thresholds or reducing token budgets")
        elif avg_utilization > 0.9:
            recommendations.append("High token utilization detected. Consider more aggressive filtering or increasing budgets")
        
        # Repository-specific recommendations
        over_budget = [r for r in successful_results if r.token_utilization and r.token_utilization > 1.0]
        if over_budget:
            recommendations.append(f"{len(over_budget)} repositories exceeded token budget. Review: {', '.join([r.repo_name for r in over_budget[:3]])}")
        
        # Performance recommendations
        slow_repos = [r for r in successful_results if r.analysis_time_seconds > 30]
        if slow_repos:
            recommendations.append(f"{len(slow_repos)} repositories had slow analysis (>30s). Consider optimization for large repositories")
        
        return recommendations
    
    def save_validation_report(self, summary: ValidationSummary, output_path: pathlib.Path) -> None:
        """Save comprehensive validation report."""
        # Create detailed report
        report = {
            'summary': {
                'total_repositories': summary.total_repositories,
                'successful_repositories': summary.successful_repositories,
                'success_rate': summary.success_rate,
                'avg_token_utilization': summary.avg_token_utilization,
                'median_token_utilization': summary.median_token_utilization,
                'token_utilization_std': summary.token_utilization_std,
                'avg_analysis_time': summary.avg_analysis_time,
                'avg_config_time': summary.avg_config_time,
                'avg_execution_time': summary.avg_execution_time,
                'avg_inclusion_rate': summary.avg_inclusion_rate,
                'avg_confidence_score': summary.avg_confidence_score,
            },
            'failed_repositories': summary.failed_repos,
            'recommendations': summary.recommendations,
            'detailed_results': []
        }
        
        # Add detailed results (without full analysis objects to keep size manageable)
        for result in summary.results:
            detailed_result = {
                'repo_name': result.repo_name,
                'repo_path': result.repo_path,
                'success': result.success,
                'predicted_tokens': result.predicted_tokens,
                'actual_tokens': result.actual_tokens,
                'token_utilization': result.token_utilization,
                'analysis_time_seconds': result.analysis_time_seconds,
                'config_generation_time_seconds': result.config_generation_time_seconds,
                'source_files_found': result.source_files_found,
                'source_files_included': result.source_files_included,
                'inclusion_rate': result.inclusion_rate,
                'confidence_score': result.confidence_score,
                'error_message': result.error_message,
            }
            
            if result.repository_analysis:
                detailed_result['repository_summary'] = {
                    'languages': result.repository_analysis.languages,
                    'estimated_source_tokens': result.repository_analysis.estimated_source_tokens,
                    'is_monorepo': result.repository_analysis.is_monorepo,
                    'recommended_algorithm': result.repository_analysis.recommended_algorithm,
                }
                
            report['detailed_results'].append(detailed_result)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
        
        print(f"📊 Validation report saved to: {output_path}")
    
    def print_validation_summary(self, summary: ValidationSummary) -> None:
        """Print a human-readable validation summary."""
        print(f"\n{'='*60}")
        print(f"🎯 SCRIBE INTELLIGENT DEFAULTS VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"📊 Overall Results:")
        print(f"   Total Repositories: {summary.total_repositories}")
        print(f"   Successful: {summary.successful_repositories}")
        print(f"   Success Rate: {summary.success_rate:.1%}")
        
        print(f"\n💰 Token Utilization:")
        print(f"   Average: {summary.avg_token_utilization:.1%}")
        print(f"   Median: {summary.median_token_utilization:.1%}")
        print(f"   Std Deviation: {summary.token_utilization_std:.1%}")
        
        print(f"\n⚡ Performance:")
        print(f"   Avg Analysis Time: {summary.avg_analysis_time:.2f}s")
        print(f"   Avg Config Time: {summary.avg_config_time:.2f}s")
        print(f"   Avg Execution Time: {summary.avg_execution_time:.2f}s")
        
        print(f"\n📈 Quality Metrics:")
        print(f"   Avg Inclusion Rate: {summary.avg_inclusion_rate:.1%}")
        print(f"   Avg Confidence: {summary.avg_confidence_score:.1%}")
        
        if summary.failed_repos:
            print(f"\n❌ Failed Repositories ({len(summary.failed_repos)}):")
            for repo in summary.failed_repos[:10]:  # Show first 10
                print(f"   - {repo}")
            if len(summary.failed_repos) > 10:
                print(f"   ... and {len(summary.failed_repos) - 10} more")
        
        if summary.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(summary.recommendations, 1):
                print(f"   {i}. {rec}")
        
        print(f"\n{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Scribe Intelligent Defaults")
    parser.add_argument("--limit", type=int, help="Limit number of repositories to test")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="validation_report.json", help="Output report file")
    parser.add_argument("--projects-root", type=str, default="/home/nathan/Projects", help="Projects root directory")
    
    args = parser.parse_args()
    
    # Run validation
    validator = ScribeValidator(pathlib.Path(args.projects_root))
    summary = validator.validate_all_repositories(max_workers=args.workers, limit=args.limit)
    
    # Print and save results
    validator.print_validation_summary(summary)
    validator.save_validation_report(summary, pathlib.Path(args.output))
    
    # Exit with appropriate code
    if summary.success_rate >= 0.95:
        print(f"\n🎉 SUCCESS: {summary.success_rate:.1%} success rate achieved!")
        return 0
    else:
        print(f"\n⚠️  WARNING: Only {summary.success_rate:.1%} success rate achieved")
        return 1


if __name__ == "__main__":
    exit(main())
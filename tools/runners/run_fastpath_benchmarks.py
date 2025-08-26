#!/usr/bin/env python3
"""
FastPath Performance Benchmarking - Quick Start Script

This script provides easy access to the comprehensive FastPath benchmarking system
with predefined configurations for different use cases.

Usage Examples:
  # Quick validation (5-10 minutes)
  python run_fastpath_benchmarks.py --quick

  # Comprehensive benchmark (30-60 minutes) 
  python run_fastpath_benchmarks.py --comprehensive

  # Production-level validation (2-4 hours)
  python run_fastpath_benchmarks.py --production

  # Custom configuration
  python run_fastpath_benchmarks.py --categories small medium --budgets 120000 200000 --runs 10
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Import our integrated benchmark system
from integrated_benchmark_runner import (
    IntegratedBenchmarkRunner,
    IntegratedBenchmarkConfig,
    create_quick_benchmark_config,
    create_comprehensive_benchmark_config,
    create_production_benchmark_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print FastPath benchmarking banner."""
    print("\n" + "="*80)
    print("🚀 FASTPATH PERFORMANCE BENCHMARK SUITE")
    print("   Comprehensive validation of FastPath vs Baseline PackRepo")
    print("="*80)


def print_configuration_summary(config: IntegratedBenchmarkConfig):
    """Print a summary of the benchmark configuration."""
    print(f"\n📋 BENCHMARK CONFIGURATION:")
    print(f"   Categories: {config.categories}")
    print(f"   Token Budgets: {config.token_budgets}")
    print(f"   Performance Runs: {config.num_performance_runs}")
    print(f"   QA Evaluation: {'Enabled' if config.enable_qa_evaluation else 'Disabled'}")
    print(f"   Output Directory: {config.output_dir}")
    
    # Estimate runtime
    num_repos = len(config.categories) * 2  # Rough estimate
    num_configs = len(config.token_budgets)
    estimated_minutes = num_repos * num_configs * (config.num_performance_runs * 0.5 + 5)
    
    if config.enable_qa_evaluation:
        estimated_minutes *= 1.5  # QA adds overhead
    
    print(f"   Estimated Runtime: {estimated_minutes:.0f} minutes\n")


def create_custom_config(args) -> IntegratedBenchmarkConfig:
    """Create custom configuration from command line arguments."""
    
    config = IntegratedBenchmarkConfig()
    
    # Basic configuration
    if args.categories:
        config.categories = args.categories
    else:
        config.categories = ['small']  # Safe default
    
    if args.budgets:
        config.token_budgets = args.budgets
    else:
        config.token_budgets = [120000]  # Safe default
    
    if args.runs:
        config.num_performance_runs = args.runs
    else:
        config.num_performance_runs = 5  # Safe default
    
    # Output directory
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = Path(f"fastpath_benchmarks_{timestamp}")
    
    # Optional flags
    config.enable_qa_evaluation = not args.no_qa
    config.save_raw_data = args.save_raw_data
    config.generate_visualizations = not args.no_viz
    
    return config


async def run_benchmark_with_error_handling(config: IntegratedBenchmarkConfig):
    """Run benchmark with comprehensive error handling."""
    
    try:
        print("\n🔥 Initializing benchmark runner...")
        runner = IntegratedBenchmarkRunner(config)
        
        print("🚀 Starting comprehensive benchmark suite...")
        results = await runner.run_integrated_benchmark()
        
        if not results:
            print("❌ No benchmark results generated")
            return False
        
        # Print final summary
        print("\n" + "="*80)
        print("✅ BENCHMARK SUITE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        successful_improvements = sum(1 for r in results 
                                    if r.overall_improvement_percent and r.overall_improvement_percent > 0)
        meets_claims = sum(1 for r in results if r.meets_10x_claim)
        
        print(f"📊 RESULTS SUMMARY:")
        print(f"   Total Tests: {len(results)}")
        print(f"   Successful Improvements: {successful_improvements}/{len(results)} ({successful_improvements/len(results)*100:.1f}%)")
        print(f"   Meets Performance Claims: {meets_claims}/{len(results)} ({meets_claims/len(results)*100:.1f}%)")
        
        if results:
            avg_improvement = sum(r.overall_improvement_percent or 0 for r in results) / len(results)
            print(f"   Average Improvement: {avg_improvement:+.1f}%")
        
        print(f"\n📁 Results saved to: {config.output_dir}")
        print(f"   - comprehensive_benchmark_report.md")
        print(f"   - raw_benchmark_data.json")
        print(f"   - comprehensive_performance_analysis.png")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        logger.exception("Benchmark execution failed")
        return False


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="FastPath Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (5-10 minutes)
  python run_fastpath_benchmarks.py --quick

  # Comprehensive benchmark (30-60 minutes)
  python run_fastpath_benchmarks.py --comprehensive

  # Production-level validation (2-4 hours)
  python run_fastpath_benchmarks.py --production

  # Custom configuration
  python run_fastpath_benchmarks.py --categories small medium --budgets 120000 --runs 10

Configuration Presets:
  --quick:         Small repos, 1 budget, 5 runs (fast validation)
  --comprehensive: Small+medium repos, 3 budgets, 15 runs (thorough testing)
  --production:    All repo sizes, 5 budgets, 20 runs (complete validation)
        """
    )
    
    # Preset configurations
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        '--quick',
        action='store_true',
        help='Quick benchmark (small repos, 1 budget, 5 runs)'
    )
    preset_group.add_argument(
        '--comprehensive', 
        action='store_true',
        help='Comprehensive benchmark (small+medium repos, 3 budgets, 15 runs)'
    )
    preset_group.add_argument(
        '--production',
        action='store_true', 
        help='Production benchmark (all repos, 5 budgets, 20 runs)'
    )
    
    # Custom configuration options
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=['small', 'medium', 'large'],
        help='Repository size categories to test'
    )
    
    parser.add_argument(
        '--budgets',
        nargs='+',
        type=int,
        help='Token budgets to test (e.g., 50000 120000 200000)'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        help='Number of performance runs per configuration'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results'
    )
    
    # Feature flags
    parser.add_argument(
        '--no-qa',
        action='store_true',
        help='Disable QA evaluation (performance benchmarking only)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization generation'
    )
    
    parser.add_argument(
        '--save-raw-data',
        action='store_true',
        default=True,
        help='Save raw benchmark data (default: True)'
    )
    
    # Information options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit (don\'t run benchmarks)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Determine configuration
    if args.quick:
        config = create_quick_benchmark_config()
        print("🏃‍♂️ Using QUICK benchmark configuration")
    elif args.comprehensive:
        config = create_comprehensive_benchmark_config()
        print("🔬 Using COMPREHENSIVE benchmark configuration")
    elif args.production:
        config = create_production_benchmark_config()
        print("🏭 Using PRODUCTION benchmark configuration")
    else:
        config = create_custom_config(args)
        print("⚙️  Using CUSTOM benchmark configuration")
    
    # Apply output directory override
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    # Show configuration
    print_configuration_summary(config)
    
    # Dry run option
    if args.dry_run:
        print("🔍 DRY RUN: Configuration displayed, exiting without running benchmarks")
        return
    
    # Confirm execution for long-running benchmarks
    if not args.quick:
        estimated_minutes = len(config.categories) * len(config.token_budgets) * config.num_performance_runs * 0.5
        if estimated_minutes > 30:
            response = input(f"\n⚠️  This benchmark is estimated to take {estimated_minutes:.0f} minutes. Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Benchmark cancelled by user")
                return
    
    # Run benchmark
    print(f"\n⏱️  Starting benchmark at {datetime.now().strftime('%H:%M:%S')}")
    success = asyncio.run(run_benchmark_with_error_handling(config))
    
    if success:
        print("\n🎉 Benchmark suite completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Benchmark suite failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
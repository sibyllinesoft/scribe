#!/usr/bin/env python3
"""
Unified CLI for running scribe benchmarks.

Usage:
    ./run.py agent-efficiency --quick
    ./run.py swebench --max-tasks 5
    ./run.py --list
    ./run.py --analyze
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common.results import list_benchmarks, list_results, load_results


BENCHMARKS = {
    "agent-efficiency": {
        "description": "Measures token efficiency when AI agents understand code dependencies",
        "scripts": {
            "statistical": "statistical_benchmark.py",
            "claude-code": "claude_code_benchmark.py",
            "real-agent": "real_agent_benchmark.py",
        },
        "default_script": "statistical",
    },
    "swebench": {
        "description": "Compares SWE-bench task success rates with and without scribe",
        "scripts": {
            "benchmark": "benchmark.py",
        },
        "default_script": "benchmark",
    },
}


def run_benchmark(name: str, script: str, args: list[str]) -> int:
    """Run a benchmark script.

    Args:
        name: Benchmark name.
        script: Script name within the benchmark.
        args: Additional arguments to pass to the script.

    Returns:
        Exit code from the script.
    """
    benchmarks_dir = Path(__file__).parent
    benchmark_dir = benchmarks_dir / name

    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        return 1

    script_path = benchmark_dir / script
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return 1

    print(f"Running: {name}/{script}")
    print(f"Args: {' '.join(args) if args else '(none)'}")
    print("=" * 60)
    print()

    result = subprocess.run(
        [sys.executable, str(script_path)] + args,
        cwd=benchmark_dir,
    )

    return result.returncode


def list_all_benchmarks() -> None:
    """List all available benchmarks."""
    print("Available Benchmarks")
    print("=" * 60)
    print()

    for name, info in BENCHMARKS.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    Scripts: {', '.join(info['scripts'].keys())}")
        print()


def analyze_results() -> None:
    """Analyze results across all benchmarks."""
    print("Benchmark Results Summary")
    print("=" * 60)
    print()

    available = list_benchmarks()

    if not available:
        print("No benchmark results found.")
        return

    for bench_name in available:
        results = list_results(bench_name)
        if results:
            print(f"{bench_name}:")
            print(f"  Results: {len(results)} files")

            # Show most recent result
            latest = results[0]
            print(f"  Latest: {latest['filename']}")

            # Try to load and show summary
            data = load_results(bench_name)
            if data and "summary" in data:
                summary = data["summary"]
                if "mean_savings_pct" in summary:
                    print(f"  Mean savings: {summary['mean_savings_pct']:.1f}%")
            elif data and "overall" in data:
                overall = data["overall"]
                if "ratio_mean" in overall:
                    print(f"  Context ratio: {overall['ratio_mean']:.1f}x")
                if "resolve_rate_scribe" in overall:
                    print(f"  Resolve rate (scribe): {overall['resolve_rate_scribe']:.1%}")
                    print(f"  Resolve rate (standard): {overall['resolve_rate_standard']:.1%}")

            print()


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for scribe benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ./run.py agent-efficiency --quick
    ./run.py agent-efficiency statistical --iterations 5
    ./run.py swebench --max-tasks 10
    ./run.py --list
    ./run.py --analyze
        """,
    )

    parser.add_argument(
        "benchmark",
        nargs="?",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to run",
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Specific script to run within the benchmark",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available benchmarks",
    )
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze results across all benchmarks",
    )

    args, remaining = parser.parse_known_args()

    if args.list:
        list_all_benchmarks()
        return 0

    if args.analyze:
        analyze_results()
        return 0

    if not args.benchmark:
        parser.print_help()
        return 1

    bench_info = BENCHMARKS[args.benchmark]

    # Determine which script to run
    if args.script:
        if args.script in bench_info["scripts"]:
            script = bench_info["scripts"][args.script]
        else:
            # Treat as a script filename
            script = args.script
    else:
        script = bench_info["scripts"][bench_info["default_script"]]

    return run_benchmark(args.benchmark, script, remaining)


if __name__ == "__main__":
    sys.exit(main())

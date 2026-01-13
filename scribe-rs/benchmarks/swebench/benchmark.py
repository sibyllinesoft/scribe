#!/usr/bin/env python3
"""
SWE-bench Benchmark

Compares success rates and token usage when solving SWE-bench tasks
with and without scribe.

Usage:
    ./benchmark.py                          # Run with defaults
    ./benchmark.py --max-tasks 10           # Limit number of tasks
    ./benchmark.py --mode scribe            # Only run scribe mode
    ./benchmark.py --mode standard          # Only run standard mode
    ./benchmark.py --quick                  # Quick test: 3 tasks
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from common.results import get_results_dir

# Handle both direct execution and module import
try:
    from .runner import TaskRunner, run_task_batch, check_opencode_installed
    from .evaluation import analyze_results, generate_report, save_benchmark_results
except ImportError:
    from runner import TaskRunner, run_task_batch, check_opencode_installed
    from evaluation import analyze_results, generate_report, save_benchmark_results


def load_swebench_tasks(dataset: str = "princeton-nlp/SWE-bench_Lite", split: str = "test") -> list:
    """Load SWE-bench tasks from Hugging Face.

    Args:
        dataset: Dataset name on Hugging Face.
        split: Dataset split to use.

    Returns:
        List of task dicts.
    """
    if not HAS_DATASETS:
        raise ImportError(
            "datasets package required: pip install datasets\n"
            "Also need: pip install swebench"
        )

    print(f"Loading dataset: {dataset}")
    ds = load_dataset(dataset, split=split)

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item.get("instance_id", ""),
            "repo": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
            "problem_statement": item.get("problem_statement", ""),
            "hints_text": item.get("hints_text", ""),
            "created_at": item.get("created_at", ""),
            "patch": item.get("patch", ""),  # Gold patch for reference
            "test_patch": item.get("test_patch", ""),
            "version": item.get("version", ""),
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def run_benchmark(
    max_tasks: int = 50,
    mode: str = "both",
    model: str = "anthropic/claude-sonnet-4-20250514",
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    use_docker: bool = True,
    quick: bool = False,
    task_timeout_s: int = 600,
    skip: int = 0,
    task_ids: list = None,
) -> dict:
    """Run the SWE-bench benchmark using OpenCode.

    Args:
        max_tasks: Maximum number of tasks to run.
        mode: "scribe", "standard", or "both".
        model: Model to use (format: provider/model, e.g., openrouter/z-ai/glm-4.7).
        dataset: SWE-bench dataset name.
        use_docker: Whether to use Docker for isolation.
        quick: Quick mode with minimal tasks.

    Returns:
        Dict with results and analysis.
    """
    if quick:
        max_tasks = 3
        print("Quick mode: running 3 tasks only")

    # OpenCode handles API keys via its own config
    # Just check that opencode is available
    if not check_opencode_installed():
        print("Error: OpenCode not installed")
        print("Install with: curl -fsSL https://opencode.ai/install | bash")
        sys.exit(1)

    # Load tasks
    tasks = load_swebench_tasks(dataset)

    # Filter by specific task IDs if provided
    if task_ids:
        task_id_set = set(task_ids)
        tasks = [t for t in tasks if t.get("instance_id") in task_id_set]
        print(f"Filtered to {len(tasks)} specific tasks: {task_ids}")
    else:
        # Apply skip and limit only if not filtering by task IDs
        if skip > 0:
            tasks = tasks[skip:]
            print(f"Skipped first {skip} tasks")

        if max_tasks and max_tasks < len(tasks):
            tasks = tasks[:max_tasks]
            print(f"Limited to {max_tasks} tasks")

    print()
    print("=" * 70)
    print("SWE-bench Benchmark (using OpenCode)")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Tasks: {len(tasks)}")
    print(f"Mode: {mode}")
    print(f"Model: {model}")
    print(f"Docker: {'enabled' if use_docker else 'disabled'}")
    print()

    # Run benchmark
    results = run_task_batch(
        tasks=tasks,
        mode=mode,
        model=model,
        use_docker=use_docker,
        task_timeout_s=task_timeout_s,
    )

    # Analyze and save
    metadata = {
        "dataset": dataset,
        "n_tasks": len(tasks),
        "mode": mode,
        "model": model,
        "use_docker": use_docker,
    }

    results_path, report_path = save_benchmark_results(results, metadata)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    # Print summary
    analysis = analyze_results(results)

    if "scribe" in analysis and analysis["scribe"]["n_tasks"] > 0:
        scribe = analysis["scribe"]
        print(f"\nScribe Mode:")
        print(f"  Resolve rate: {scribe['resolve_rate']:.1%} ({scribe['n_resolved']}/{scribe['n_tasks']})")
        print(f"  Mean tokens: {scribe['tokens_mean']:,.0f}")
        print(f"  Mean tool calls: {scribe['tool_calls_mean']:.1f}")
        print(f"  Mean scribe calls: {scribe['scribe_calls_mean']:.1f}")

    if "standard" in analysis and analysis["standard"]["n_tasks"] > 0:
        standard = analysis["standard"]
        print(f"\nStandard Mode:")
        print(f"  Resolve rate: {standard['resolve_rate']:.1%} ({standard['n_resolved']}/{standard['n_tasks']})")
        print(f"  Mean tokens: {standard['tokens_mean']:,.0f}")
        print(f"  Mean tool calls: {standard['tool_calls_mean']:.1f}")

    if "comparison" in analysis:
        comp = analysis["comparison"]
        print(f"\nComparison:")
        print(f"  Resolve rate difference: {comp['resolve_rate_diff']:+.1%}")
        print(f"  Token ratio (scribe/standard): {comp['tokens_mean_ratio']:.2f}x")

    print()
    print(f"Results saved to: {results_path}")
    print(f"Report saved to: {report_path}")

    return {
        "results": results,
        "analysis": analysis,
        "results_path": str(results_path),
        "report_path": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Benchmark: Compare scribe vs standard approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--max-tasks", "-n",
        type=int,
        default=50,
        help="Maximum number of tasks to run (default: 50)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["scribe-context", "scribe-tool", "standard", "both", "all", "scribe"],
        default="both",
        help="Which mode(s) to run: standard, scribe-context (pre-fetched), scribe-tool (agent uses scribe), both (standard+scribe-context), all (all three). Legacy 'scribe' = 'scribe-context'. (default: both)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-20250514",
        help="Model in provider/model format (e.g., openrouter/z-ai/glm-4.7, anthropic/claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset to use",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker isolation (for testing only)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run only 3 tasks for testing",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="*",
        help="Specific task IDs to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per task in seconds (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of tasks to skip from the start",
    )

    args = parser.parse_args()

    run_benchmark(
        max_tasks=args.max_tasks,
        mode=args.mode,
        model=args.model,
        dataset=args.dataset,
        use_docker=not args.no_docker,
        quick=args.quick,
        task_timeout_s=args.timeout,
        skip=args.skip,
        task_ids=args.task_ids,
    )


if __name__ == "__main__":
    main()

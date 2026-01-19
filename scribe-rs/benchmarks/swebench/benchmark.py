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
import time
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
    from .runner import TaskRunner, run_task_batch, check_opencode_installed, get_scribe_git_info
    from .evaluation import analyze_results, generate_report, save_benchmark_results
except ImportError:
    from runner import TaskRunner, run_task_batch, check_opencode_installed, get_scribe_git_info
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


def load_tasks_from_config(config_path: str = None) -> tuple[list, list[str]]:
    """Load tasks from benchmark_config.json.

    Supports multiple datasets and returns all tasks.

    Args:
        config_path: Path to config file (defaults to benchmark_config.json in same dir)

    Returns:
        Tuple of (tasks list, list of dataset names used)
    """
    if config_path is None:
        config_path = Path(__file__).parent / "benchmark_config.json"

    with open(config_path) as f:
        config = json.load(f)

    all_tasks = []
    datasets_used = []

    for ds_config in config.get("datasets", []):
        dataset_name = ds_config["name"]
        task_ids = [t["id"] for t in ds_config.get("tasks", [])]

        if not task_ids:
            continue

        datasets_used.append(dataset_name)

        # Load the full dataset and filter to our tasks
        full_tasks = load_swebench_tasks(dataset_name)
        task_id_set = set(task_ids)
        filtered = [t for t in full_tasks if t.get("instance_id") in task_id_set]

        print(f"  Selected {len(filtered)}/{len(task_ids)} tasks from {dataset_name}")
        all_tasks.extend(filtered)

    return all_tasks, datasets_used


def run_benchmark(
    max_tasks: int = 50,
    mode: str = "both",
    model: str = "sonnet",
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    use_docker: bool = True,
    quick: bool = False,
    task_timeout_s: int = 600,
    skip: int = 0,
    task_ids: list = None,
    parallel_workers: int = 1,
    runs_per_task: int = 1,
    delay_between_runs: int = 30,
    use_config: bool = False,
    config_path: str = None,
    context_tokens: int = 4000,
) -> dict:
    """Run the SWE-bench benchmark using Claude Code.

    Args:
        max_tasks: Maximum number of tasks to run.
        mode: "scribe", "standard", or "both".
        model: Model to use (e.g., "sonnet", "opus", "claude-sonnet-4-5-20250929").
        dataset: SWE-bench dataset name.
        use_docker: Whether to use Docker for isolation.
        quick: Quick mode with minimal tasks.
        parallel_workers: Number of tasks to run in parallel.
        runs_per_task: Number of times to run each task.
        delay_between_runs: Delay in seconds between runs for rate limiting.

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

    # Load tasks - either from config or from dataset directly
    datasets_used = [dataset]
    if use_config:
        print("Loading tasks from config file...")
        tasks, datasets_used = load_tasks_from_config(config_path)
        if not tasks:
            print("Error: No tasks loaded from config")
            sys.exit(1)
    elif task_ids:
        # Filter by specific task IDs
        tasks = load_swebench_tasks(dataset)
        task_id_set = set(task_ids)
        tasks = [t for t in tasks if t.get("instance_id") in task_id_set]
        print(f"Filtered to {len(tasks)} specific tasks: {task_ids}")
    else:
        # Load from dataset with skip and limit
        tasks = load_swebench_tasks(dataset)
        if skip > 0:
            tasks = tasks[skip:]
            print(f"Skipped first {skip} tasks")

        if max_tasks and max_tasks < len(tasks):
            tasks = tasks[:max_tasks]
            print(f"Limited to {max_tasks} tasks")

    # Capture scribe git info for tracking
    scribe_git_info = get_scribe_git_info()

    print()
    print("=" * 70)
    print("SWE-bench Benchmark (using Claude Code)")
    print("=" * 70)
    print(f"Dataset(s): {', '.join(datasets_used)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Mode: {mode}")
    print(f"Model: {model}")
    if scribe_git_info.get("commit"):
        scribe_version = scribe_git_info["commit"]
        if scribe_git_info.get("dirty"):
            scribe_version += " (dirty)"
        print(f"Scribe commit: {scribe_version}")
    print(f"Docker: {'enabled' if use_docker else 'disabled'}")
    print(f"Runs per task: {runs_per_task}")
    print(f"Delay between runs: {delay_between_runs}s")
    print(f"Timeout per task: {task_timeout_s}s")
    print(f"Context tokens: {context_tokens}")
    print()

    # Run benchmark - multiple runs per task
    all_results = []
    for run_num in range(1, runs_per_task + 1):
        if runs_per_task > 1:
            print(f"\n{'=' * 70}")
            print(f"RUN {run_num}/{runs_per_task}")
            print(f"{'=' * 70}\n")

        results = run_task_batch(
            tasks=tasks,
            mode=mode,
            model=model,
            use_docker=use_docker,
            task_timeout_s=task_timeout_s,
            parallel_workers=parallel_workers,
            context_tokens=context_tokens,
        )

        # Tag results with run number
        for r in results:
            r.run_number = run_num

        all_results.extend(results)

        # Delay between runs (but not after the last run)
        if run_num < runs_per_task and delay_between_runs > 0:
            print(f"\nWaiting {delay_between_runs}s before next run (rate limiting)...")
            time.sleep(delay_between_runs)

    # Use all results for analysis
    results = all_results

    # Analyze and save
    # Extract model_resolved from results if available (most specific model ID)
    model_resolved = None
    for r in results:
        if hasattr(r, 'model_resolved') and r.model_resolved:
            model_resolved = r.model_resolved
            break

    metadata = {
        "dataset": ", ".join(datasets_used),
        "n_tasks": len(tasks),
        "mode": mode,
        "model": model,
        "model_resolved": model_resolved,  # Actual model ID (e.g., "claude-sonnet-4-5-20250929")
        "scribe_commit": scribe_git_info.get("commit"),
        "scribe_commit_full": scribe_git_info.get("commit_full"),
        "scribe_branch": scribe_git_info.get("branch"),
        "scribe_dirty": scribe_git_info.get("dirty"),
        "use_docker": use_docker,
        "runs_per_task": runs_per_task,
        "delay_between_runs": delay_between_runs,
        "context_tokens": context_tokens,
        "task_timeout_s": task_timeout_s,
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
        default="sonnet",
        help="Model to use (e.g., 'sonnet', 'opus', 'claude-sonnet-4-5-20250929')",
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
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of times to run each task (default: 1)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="Delay in seconds between task runs for rate limiting (default: 30)",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Load tasks from benchmark_config.json instead of dataset",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config file (default: benchmark_config.json in same dir)",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=4000,
        help="Token budget for scribe-context mode (default: 4000). Range 1000-16000 recommended.",
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
        parallel_workers=args.parallel,
        runs_per_task=args.runs,
        delay_between_runs=args.delay,
        use_config=args.config,
        config_path=args.config_path,
        context_tokens=args.context_tokens,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run selected SWE-bench tasks for scribe-tool vs standard comparison.

This script runs the selected tasks with both modes, 3 times each.
Results are saved incrementally after each run.
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "swebench"))

from swebench.runner import TaskRunner, get_scribe_git_info

# Selected tasks for benchmarking
SELECTED_TASKS = [
    {
        "id": "vuejs__core-11739",
        "language": "typescript",
        "description": "SSR hydration errors with v-bind in CSS",
        "instance_id": "vuejs__core-11739",
        "repo": "vuejs/core",
        "problem_statement": """
SSR hydration errors with v-bind in CSS

When using v-bind() in CSS with SSR, hydration errors occur. The styles computed
on the server side don't match the client side, causing hydration mismatches.

Steps to reproduce:
1. Use v-bind() in a component's <style> section
2. Enable SSR
3. Hydrate the component on the client

Expected: Smooth hydration without errors
Actual: Hydration mismatch warnings/errors
"""
    },
    {
        "id": "tokio-rs__tokio-4384",
        "language": "rust",
        "description": "UdpSocket not marked as UnwindSafe",
        "instance_id": "tokio-rs__tokio-4384",
        "repo": "tokio-rs/tokio",
        "problem_statement": """
UdpSocket is not marked as UnwindSafe

The tokio::net::UdpSocket type does not implement UnwindSafe or RefUnwindSafe,
even though it should be safe to unwind through code using it.

This prevents using UdpSocket in contexts that require unwind safety, like
catch_unwind or certain testing frameworks.

Expected: UdpSocket should implement UnwindSafe and RefUnwindSafe
Actual: These traits are not implemented
"""
    },
    {
        "id": "pytest-dev__pytest-5413",
        "language": "python",
        "description": "Pytest fixture scope ordering issue",
        "instance_id": "pytest-dev__pytest-5413",
        "repo": "pytest-dev/pytest",
        "problem_statement": """
Fixture scope ordering is incorrect in some cases

When using fixtures with different scopes, the ordering of fixture setup
and teardown can be incorrect. Specifically, session-scoped fixtures may
be torn down before function-scoped fixtures that depend on them.

This violates the expected fixture lifecycle where higher-scoped fixtures
should outlive lower-scoped ones.

Steps to reproduce:
1. Create a session-scoped fixture
2. Create a function-scoped fixture that depends on it
3. Run multiple tests using both fixtures
4. Observe incorrect teardown ordering in some cases
"""
    }
]

RUNS_PER_MODE = 3
MODES = ["standard", "scribe-tool"]


def run_benchmark():
    """Run the benchmark and save results."""
    results_dir = Path(__file__).parent / "results" / "swebench"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"benchmark_selected_{timestamp}.json"
    log_file = results_dir / f"benchmark_selected_{timestamp}.log"

    # Get scribe git info
    scribe_info = get_scribe_git_info()

    # Initialize results structure
    all_results = {
        "timestamp": timestamp,
        "scribe_git": scribe_info,
        "tasks": [t["id"] for t in SELECTED_TASKS],
        "modes": MODES,
        "runs_per_mode": RUNS_PER_MODE,
        "results": []
    }

    def log(msg):
        """Log to both stdout and file."""
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")

    def save_results():
        """Save current results to file."""
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    log(f"Starting benchmark: {len(SELECTED_TASKS)} tasks × {len(MODES)} modes × {RUNS_PER_MODE} runs")
    log(f"Results file: {results_file}")
    log(f"Scribe git: {scribe_info.get('commit', 'unknown')}")

    runner = TaskRunner(
        model="sonnet",
        use_docker=True,
        task_timeout_s=2400,  # 40 minutes per task
        context_tokens=4000,
    )
    runner.verbose = True

    total_runs = len(SELECTED_TASKS) * len(MODES) * RUNS_PER_MODE
    completed = 0

    for task in SELECTED_TASKS:
        for mode in MODES:
            for run_num in range(1, RUNS_PER_MODE + 1):
                completed += 1
                log(f"\n{'='*60}")
                log(f"[{completed}/{total_runs}] {task['id']} - {mode} - run {run_num}/{RUNS_PER_MODE}")
                log(f"{'='*60}")

                start_time = time.time()
                try:
                    result = runner.run_task(task, mode=mode)
                    result.run_number = run_num

                    # Convert to dict for JSON serialization
                    result_dict = {
                        "task_id": result.task_id,
                        "mode": result.mode,
                        "model": result.model,
                        "run_number": run_num,
                        "timestamp": result.timestamp,
                        "resolved": result.resolved,
                        "success": result.success,
                        "duration_s": result.duration_s,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "total_tokens": result.total_tokens,
                        "num_tool_calls": result.num_tool_calls,
                        "scribe_calls": result.scribe_calls,
                        "hook_denies": result.hook_denies,
                        "hook_warnings": result.hook_warnings,
                        "scribe_commands": result.scribe_commands,
                        "patch_length": len(result.patch) if result.patch else 0,
                        "error": result.error,
                        "metrics_reliable": result.metrics_reliable,
                        "model_resolved": result.model_resolved,
                        "context_tokens": result.context_tokens,
                        "repo_code_bytes": result.repo_code_bytes,
                    }

                    all_results["results"].append(result_dict)

                    log(f"  Result: {'RESOLVED' if result.resolved else 'FAILED'}")
                    log(f"  Duration: {result.duration_s:.1f}s")
                    log(f"  Tokens: {result.total_tokens:,}")
                    log(f"  Hook denies: {result.hook_denies}, scribe cmds: {result.scribe_commands}")

                except Exception as e:
                    log(f"  ERROR: {e}")
                    all_results["results"].append({
                        "task_id": task["id"],
                        "mode": mode,
                        "run_number": run_num,
                        "error": str(e),
                        "success": False,
                        "resolved": False
                    })

                # Save after each run
                save_results()
                elapsed = time.time() - start_time
                log(f"  Total time for this run: {elapsed:.1f}s")

    log(f"\n{'='*60}")
    log("BENCHMARK COMPLETE")
    log(f"{'='*60}")

    # Summary
    resolved_by_mode = {}
    tokens_by_mode = {}
    for r in all_results["results"]:
        mode = r["mode"]
        if mode not in resolved_by_mode:
            resolved_by_mode[mode] = {"resolved": 0, "total": 0}
            tokens_by_mode[mode] = []
        resolved_by_mode[mode]["total"] += 1
        if r.get("resolved"):
            resolved_by_mode[mode]["resolved"] += 1
        if r.get("total_tokens"):
            tokens_by_mode[mode].append(r["total_tokens"])

    log("\nSummary by mode:")
    for mode in MODES:
        if mode in resolved_by_mode:
            stats = resolved_by_mode[mode]
            rate = stats["resolved"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_tokens = sum(tokens_by_mode[mode]) / len(tokens_by_mode[mode]) if tokens_by_mode[mode] else 0
            log(f"  {mode}: {stats['resolved']}/{stats['total']} resolved ({rate:.0f}%), avg {avg_tokens:,.0f} tokens")

    save_results()
    log(f"\nResults saved to: {results_file}")
    return results_file


if __name__ == "__main__":
    run_benchmark()

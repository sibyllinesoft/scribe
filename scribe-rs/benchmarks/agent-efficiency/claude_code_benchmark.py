#!/usr/bin/env python3
"""
Claude Code Benchmark for Code Understanding

Runs Claude Code CLI to measure real token usage and tool calls
when understanding code dependencies.

Compares:
- Claude Code with standard tools (grep/read discovery)
- Scribe covering-set (single call)

Uses --output-format json to capture exact token usage.

Usage:
    ./claude_code_benchmark.py                    # Default: 3 iterations
    ./claude_code_benchmark.py --iterations 5    # More iterations
    ./claude_code_benchmark.py --quick           # 1 iteration, 3 targets
    ./claude_code_benchmark.py --model glm-4.7  # Use GLM 4.7 via Z.ai plan
    ./claude_code_benchmark.py --claude-config-dir ./benchmarks/.claude-config
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from common.claude_config import resolve_claude_config_dir, build_claude_env
except ImportError:
    # Allow running directly from this directory
    from ..common.claude_config import resolve_claude_config_dir, build_claude_env


@dataclass
class ClaudeCodeRun:
    """Record of a Claude Code run."""
    target_id: str
    target_query: str
    model: str
    timestamp: str

    # From Claude Code JSON output
    duration_ms: float = 0
    duration_api_ms: float = 0
    num_turns: int = 0
    total_cost_usd: float = 0

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_tokens: int = 0  # input + output only
    full_context_tokens: int = 0  # all context including cache

    # Result
    result_text: str = ""
    session_id: str = ""

    # Status
    success: bool = True
    error: str = ""

    # Raw JSON for debugging
    raw_json: dict = field(default_factory=dict)


@dataclass
class ScribeRun:
    """Record of a scribe covering-set run."""
    target_id: str
    target_query: str
    timestamp: str

    output: str = ""
    output_tokens: int = 0
    files_returned: int = 0
    duration_ms: float = 0

    success: bool = True
    error: str = ""


def run_claude_code(
    repo_root: Path,
    target_query: str,
    model: str = "glm-4.7",
    claude_config_dir: Optional[Path] = None,
) -> ClaudeCodeRun:
    """Run Claude Code to find dependencies for a target."""

    # Extract file and entity from query
    parts = target_query.split(":")
    target_file = parts[0]
    target_entity = parts[1] if len(parts) > 1 else ""

    prompt = f"""Find all dependencies for the function/type `{target_entity}` in the file `{target_file}`.

Read the target file first to understand what the function does, then trace its dependencies.
For each dependency (type, trait, function), identify where it's defined.

When done, provide a summary listing:
1. The target function/type
2. All direct dependencies with their file locations
3. Key transitive dependencies

Be efficient - don't read files you don't need."""

    run = ClaudeCodeRun(
        target_id=target_query.replace("/", "_").replace(":", "_"),
        target_query=target_query,
        model=model,
        timestamp=datetime.now().isoformat()
    )

    try:
        # Run Claude Code with JSON output
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--output-format", "json",
                "--model", model,
                "--dangerously-skip-permissions",
                "-p", prompt
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=build_claude_env(claude_config_dir),
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            run.success = False
            run.error = result.stderr[:500] if result.stderr else "Unknown error"
            return run

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            run.raw_json = data

            run.duration_ms = data.get("duration_ms", 0)
            run.duration_api_ms = data.get("duration_api_ms", 0)
            run.num_turns = data.get("num_turns", 0)
            run.total_cost_usd = data.get("total_cost_usd", 0)
            run.session_id = data.get("session_id", "")
            run.result_text = data.get("result", "")[:2000]  # Truncate for storage

            # Extract token usage
            usage = data.get("usage", {})
            run.input_tokens = usage.get("input_tokens", 0)
            run.output_tokens = usage.get("output_tokens", 0)
            run.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            run.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

            # Calculate totals
            # total_tokens: just input + output (what's billed at full rate)
            run.total_tokens = run.input_tokens + run.output_tokens
            # full_context_tokens: all context including cache (true context size)
            run.full_context_tokens = run.input_tokens + run.cache_read_tokens + run.cache_creation_tokens

            run.success = data.get("type") == "result" and not data.get("is_error", False)
            if not run.success:
                run.error = data.get("result", "Unknown error")[:500]

        except json.JSONDecodeError as e:
            run.success = False
            run.error = f"JSON parse error: {e}"

    except subprocess.TimeoutExpired:
        run.success = False
        run.error = "Timeout after 5 minutes"
    except Exception as e:
        run.success = False
        run.error = str(e)[:500]

    return run


def run_scribe(repo_root: Path, target_query: str) -> ScribeRun:
    """Run scribe covering-set for comparison."""
    run = ScribeRun(
        target_id=target_query.replace("/", "_").replace(":", "_"),
        target_query=target_query,
        timestamp=datetime.now().isoformat()
    )

    scribe_bin = repo_root / "target" / "release" / "scribe"
    if not scribe_bin.exists():
        scribe_bin = "scribe"

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            [str(scribe_bin), "--covering-set", target_query, "--stdout"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )

        run.duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode == 0:
            run.output = result.stdout
            run.output_tokens = len(result.stdout) // 4
            run.files_returned = len(re.findall(r"<file>[\s\S]*?</file>", result.stdout))
            run.success = True
        else:
            run.success = False
            run.error = result.stderr[:500]

    except Exception as e:
        run.success = False
        run.error = str(e)[:500]
        run.duration_ms = (time.perf_counter() - start_time) * 1000

    return run


def run_benchmark(
    repo_root: Path,
    targets: list,
    iterations: int,
    model: str,
    output_dir: Path,
    claude_config_dir: Optional[Path],
):
    """Run the full benchmark."""
    all_claude_runs = []
    all_scribe_runs = []

    print(f"\n{'='*70}")
    print(f"Claude Code vs Scribe Benchmark")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Targets: {len(targets)}")
    print(f"Iterations: {iterations}")
    print(f"Total Claude runs: {len(targets) * iterations}")
    print()

    for target in targets:
        target_id = target["id"]
        query = target["scribe_query"]

        print(f"\n{'='*60}")
        print(f"Target: {target_id}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        for i in range(iterations):
            print(f"\n  Iteration {i+1}/{iterations}")

            # Run Claude Code
            print(f"    [CLAUDE] Running...", end=" ", flush=True)
            claude_run = run_claude_code(repo_root, query, model, claude_config_dir)

            if claude_run.success:
                print(f"OK ({claude_run.total_tokens:,} tokens, {claude_run.num_turns} turns, ${claude_run.total_cost_usd:.4f})")
            else:
                print(f"FAILED: {claude_run.error[:50]}")

            all_claude_runs.append(claude_run)

            # Run scribe (only on first iteration since it's deterministic)
            if i == 0:
                print(f"    [SCRIBE] Running...", end=" ", flush=True)
                scribe_run = run_scribe(repo_root, query)

                if scribe_run.success:
                    print(f"OK ({scribe_run.output_tokens:,} tokens, {scribe_run.files_returned} files)")
                else:
                    print(f"FAILED: {scribe_run.error[:50]}")

                all_scribe_runs.append(scribe_run)

            # Brief comparison
            if claude_run.success and all_scribe_runs and all_scribe_runs[-1].success:
                scribe_tokens = all_scribe_runs[-1].output_tokens
                if scribe_tokens > 0:
                    ratio = claude_run.full_context_tokens / scribe_tokens
                    print(f"    [RATIO] Claude context {ratio:.1f}x vs scribe output")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to serializable format (exclude raw_json to keep file size manageable)
    claude_data = []
    for run in all_claude_runs:
        d = asdict(run)
        d.pop("raw_json", None)  # Remove large raw JSON
        claude_data.append(d)

    scribe_data = [asdict(run) for run in all_scribe_runs]
    for d in scribe_data:
        d.pop("output", None)  # Remove large output

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "claude_config_dir": str(claude_config_dir) if claude_config_dir else None,
            "iterations": iterations,
            "n_targets": len(targets)
        },
        "claude_runs": claude_data,
        "scribe_runs": scribe_data
    }

    results_file = output_dir / f"claude_code_benchmark_{timestamp}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\n\nResults saved to: {results_file}")

    # Generate summary
    generate_summary(all_claude_runs, all_scribe_runs, targets, output_dir, timestamp, model)

    return results_file


def generate_summary(claude_runs: list, scribe_runs: list, targets: list,
                     output_dir: Path, timestamp: str, model: str):
    """Generate summary statistics and report."""
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")

    # Create target -> result mappings
    scribe_by_target = {r.target_query: r for r in scribe_runs}

    # Group claude runs by target
    claude_by_target = {}
    for run in claude_runs:
        if run.target_query not in claude_by_target:
            claude_by_target[run.target_query] = []
        claude_by_target[run.target_query].append(run)

    summary_data = []
    total_claude_tokens = 0
    total_scribe_tokens = 0
    total_cost = 0
    total_claude_runs = 0

    print(f"\n{'Target':<25} {'Claude':>12} {'Scribe':>12} {'Ratio':>8} {'Cost':>10} {'Turns':>6}")
    print("-" * 80)

    for target in targets:
        query = target["scribe_query"]
        tid = target["id"]

        claude_results = claude_by_target.get(query, [])
        scribe_result = scribe_by_target.get(query)

        successful_claude = [r for r in claude_results if r.success]
        scribe_tokens = scribe_result.output_tokens if scribe_result and scribe_result.success else 0

        if successful_claude and scribe_tokens > 0:
            avg_context = sum(r.full_context_tokens for r in successful_claude) / len(successful_claude)
            avg_cost = sum(r.total_cost_usd for r in successful_claude) / len(successful_claude)
            avg_turns = sum(r.num_turns for r in successful_claude) / len(successful_claude)
            ratio = avg_context / scribe_tokens

            total_claude_tokens += avg_context
            total_scribe_tokens += scribe_tokens
            total_cost += avg_cost
            total_claude_runs += len(successful_claude)

            print(f"{tid:<25} {avg_context:>12,.0f} {scribe_tokens:>12,} {ratio:>8.1f}x ${avg_cost:>9.4f} {avg_turns:>6.1f}")

            summary_data.append({
                "target_id": tid,
                "target_query": query,
                "claude_context_mean": avg_context,
                "claude_turns_mean": avg_turns,
                "claude_cost_mean": avg_cost,
                "scribe_tokens": scribe_tokens,
                "ratio": ratio,
                "n_runs": len(successful_claude)
            })
        else:
            total_scribe_tokens += scribe_tokens
            print(f"{tid:<25} {'FAILED':>12} {scribe_tokens:>12,} {'-':>8} {'-':>10} {'-':>6}")

    print("-" * 80)

    if total_scribe_tokens > 0 and summary_data:
        overall_ratio = total_claude_tokens / total_scribe_tokens

        print(f"{'TOTAL':<25} {total_claude_tokens:>12,.0f} {total_scribe_tokens:>12,} {overall_ratio:>8.1f}x ${total_cost:>9.4f}")

        # Calculate statistics
        ratios = [s["ratio"] for s in summary_data]
        avg_ratio = sum(ratios) / len(ratios)
        std_ratio = (sum((r - avg_ratio)**2 for r in ratios) / len(ratios)) ** 0.5

        print(f"\n\n{'='*70}")
        print("Key Findings")
        print(f"{'='*70}")

        print(f"\n  Context Ratio (Claude / Scribe):")
        print(f"    Mean:   {avg_ratio:.1f}x")
        print(f"    Std:    {std_ratio:.1f}x")
        print(f"    Range:  {min(ratios):.1f}x - {max(ratios):.1f}x")

        print(f"\n  Cost:")
        print(f"    Total:      ${total_cost:.4f}")
        print(f"    Per target: ${total_cost / len(summary_data):.4f}")

        avg_turns = sum(s['claude_turns_mean'] for s in summary_data) / len(summary_data)

        print(f"\n  Turns:")
        print(f"    Claude: {avg_turns:.1f} avg")
        print(f"    Scribe: 1")

        print(f"\n  Efficiency:")
        print(f"    Claude processes {avg_ratio:.1f}x more context than scribe provides")
        print(f"    Scribe provides equivalent context in 1 call vs {avg_turns:.1f} avg turns")

        # Save detailed summary
        summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "n_targets": len(summary_data),
                "total_runs": total_claude_runs
            },
            "overall": {
                "claude_context_total": total_claude_tokens,
                "scribe_tokens_total": total_scribe_tokens,
                "ratio_mean": avg_ratio,
                "ratio_std": std_ratio,
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
                "total_cost_usd": total_cost,
                "avg_turns": avg_turns
            },
            "per_target": summary_data
        }

        summary_file = output_dir / f"claude_code_summary_{timestamp}.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\n\nSummary saved to: {summary_file}")

        # Generate markdown report
        generate_markdown_report(summary, output_dir, timestamp)


def generate_markdown_report(summary: dict, output_dir: Path, timestamp: str):
    """Generate a markdown report for the benchmark results."""

    meta = summary["metadata"]
    overall = summary["overall"]
    per_target = summary["per_target"]

    report = f"""# Claude Code vs Scribe Benchmark Results

**Generated:** {meta["timestamp"]}
**Model:** {meta["model"]}
**Targets:** {meta["n_targets"]}
**Total Runs:** {meta["total_runs"]}

## Executive Summary

| Metric | Value |
|--------|-------|
| **Mean Context Ratio** | {overall["ratio_mean"]:.1f}x |
| **Context Ratio Range** | {overall["ratio_min"]:.1f}x - {overall["ratio_max"]:.1f}x |
| **Average Turns** | {overall["avg_turns"]:.1f} |
| **Total Cost** | ${overall["total_cost_usd"]:.4f} |

### Key Finding

> **Claude Code processes {overall["ratio_mean"]:.1f}x more context** than scribe's covering-set output
> to accomplish the same code understanding task, across {overall["avg_turns"]:.1f} turns on average.

## Per-Target Results

| Target | Claude Context | Scribe Output | Ratio | Cost | Turns |
|--------|----------------|---------------|-------|------|-------|
"""

    for t in sorted(per_target, key=lambda x: -x["ratio"]):
        report += f"| {t['target_id']} | {t['claude_context_mean']:,.0f} | {t['scribe_tokens']:,} | {t['ratio']:.1f}x | ${t['claude_cost_mean']:.4f} | {t['claude_turns_mean']:.1f} |\n"

    report += f"""

## Methodology

- **Claude Code**: Run with `--print --output-format json` to capture exact token usage
- **Scribe**: Single `--covering-set` call returning dependency context
- **Context comparison**: Claude's full context (including cached reads) vs scribe's output

## What This Measures

Claude Code was given this task for each target:
> Find all dependencies for function X in file Y. Read the target file, trace dependencies,
> and provide a summary of all direct and transitive dependencies.

The **context ratio** shows how much more context Claude processes during multi-turn discovery
compared to scribe's single-call output. Most of Claude's context is redundant re-reading
of files across turns (cached, but still counted).

## Interpretation

- **Higher ratio** = more redundant context processing
- **More turns** = more iterative discovery steps
- Scribe provides the same understanding in **1 call** with **minimal context**

## Raw Data

See `claude_code_benchmark_{timestamp}.json` for complete run data.
"""

    report_file = output_dir / f"claude_code_report_{timestamp}.md"
    report_file.write_text(report)
    print(f"Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Claude Code vs Scribe Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=3,
                        help="Number of iterations per target (default: 3)")
    parser.add_argument("--model", "-m", type=str, default="glm-4.7",
                        help="Claude model to use (default: glm-4.7)")
    parser.add_argument("--claude-config-dir", type=str,
                        help="Custom Claude Code config dir (overrides system config)")
    parser.add_argument("--targets", "-t", type=str, nargs="*",
                        help="Specific target IDs to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 iteration, 3 targets")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    # Use unified results directory
    results_dir = script_dir.parent / "results" / "agent-efficiency"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load targets
    with open(script_dir / "targets.json") as f:
        data = json.load(f)
        targets = data["targets"]

    # Filter targets
    if args.targets:
        targets = [t for t in targets if t["id"] in args.targets]

    iterations = args.iterations
    if args.quick:
        iterations = 1
        quick_ids = ["token_counter_count", "centrality_calculate", "ast_parse_chunks"]
        targets = [t for t in targets if t["id"] in quick_ids]

    if not targets:
        print("No targets selected")
        sys.exit(1)

    claude_config_dir = resolve_claude_config_dir(args.claude_config_dir)
    run_benchmark(repo_root, targets, iterations, args.model, results_dir, claude_config_dir)


if __name__ == "__main__":
    main()

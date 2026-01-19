"""SWE-bench result evaluation and analysis."""

import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.statistics import mean, std_dev, confidence_interval_95
from common.results import save_results, save_report
from common.reporting import (
    generate_markdown_table,
    generate_comparison_section,
    generate_report_header,
    format_tokens,
)


@dataclass
class ModeStats:
    """Statistics for a single mode (scribe or standard)."""
    mode: str
    n_tasks: int
    n_resolved: int
    resolve_rate: float

    # Token stats
    tokens_mean: float
    tokens_std: float
    tokens_total: int

    # Tool stats
    tool_calls_mean: float
    scribe_calls_mean: float

    # Time stats
    duration_mean_s: float
    duration_total_s: float

    # Efficiency (tokens per resolved task)
    tokens_per_resolved: float


@dataclass
class ComparisonStats:
    """Comparison statistics between scribe and standard modes."""
    resolve_rate_diff: float  # scribe - standard
    resolve_rate_ratio: float  # scribe / standard

    tokens_mean_diff: float
    tokens_mean_ratio: float

    tool_calls_diff: float

    efficiency_diff: float  # tokens per resolved: scribe - standard


def compute_mode_stats(results: list, mode: str) -> ModeStats:
    """Compute statistics for results of a single mode."""
    mode_results = [r for r in results if r.mode == mode and r.success]

    if not mode_results:
        return ModeStats(
            mode=mode,
            n_tasks=0,
            n_resolved=0,
            resolve_rate=0,
            tokens_mean=0,
            tokens_std=0,
            tokens_total=0,
            tool_calls_mean=0,
            scribe_calls_mean=0,
            duration_mean_s=0,
            duration_total_s=0,
            tokens_per_resolved=0,
        )

    n_tasks = len(mode_results)
    n_resolved = sum(1 for r in mode_results if r.resolved)
    resolve_rate = n_resolved / n_tasks if n_tasks > 0 else 0

    tokens = [r.total_tokens for r in mode_results]
    tool_calls = [r.num_tool_calls for r in mode_results]
    scribe_calls = [r.scribe_calls for r in mode_results]
    durations = [r.duration_s for r in mode_results]

    tokens_total = sum(tokens)
    resolved_tokens = sum(r.total_tokens for r in mode_results if r.resolved)
    tokens_per_resolved = resolved_tokens / n_resolved if n_resolved > 0 else 0

    return ModeStats(
        mode=mode,
        n_tasks=n_tasks,
        n_resolved=n_resolved,
        resolve_rate=resolve_rate,
        tokens_mean=mean(tokens),
        tokens_std=std_dev(tokens),
        tokens_total=tokens_total,
        tool_calls_mean=mean(tool_calls),
        scribe_calls_mean=mean(scribe_calls),
        duration_mean_s=mean(durations),
        duration_total_s=sum(durations),
        tokens_per_resolved=tokens_per_resolved,
    )


def compute_comparison(scribe_stats: ModeStats, standard_stats: ModeStats) -> ComparisonStats:
    """Compute comparison between scribe and standard modes."""

    def safe_ratio(a: float, b: float) -> float:
        return a / b if b != 0 else 0

    return ComparisonStats(
        resolve_rate_diff=scribe_stats.resolve_rate - standard_stats.resolve_rate,
        resolve_rate_ratio=safe_ratio(scribe_stats.resolve_rate, standard_stats.resolve_rate),
        tokens_mean_diff=scribe_stats.tokens_mean - standard_stats.tokens_mean,
        tokens_mean_ratio=safe_ratio(scribe_stats.tokens_mean, standard_stats.tokens_mean),
        tool_calls_diff=scribe_stats.tool_calls_mean - standard_stats.tool_calls_mean,
        efficiency_diff=scribe_stats.tokens_per_resolved - standard_stats.tokens_per_resolved,
    )


def analyze_results(results: list) -> dict:
    """Analyze benchmark results and return summary dict."""
    # Check which modes are present in results
    modes_present = set(r.mode if hasattr(r, 'mode') else r.get('mode') for r in results)

    # Compute stats for each mode
    standard_stats = compute_mode_stats(results, "standard")

    # Handle both old "scribe" mode and new scribe-context/scribe-tool modes
    if "scribe-context" in modes_present:
        scribe_context_stats = compute_mode_stats(results, "scribe-context")
    else:
        scribe_context_stats = compute_mode_stats(results, "scribe")  # fallback for legacy

    scribe_tool_stats = compute_mode_stats(results, "scribe-tool")

    # For backwards compatibility, use scribe-context for "scribe" comparison
    comparison = compute_comparison(scribe_context_stats, standard_stats)

    return {
        "scribe": asdict(scribe_context_stats),  # backwards compatible key
        "scribe-context": asdict(scribe_context_stats),
        "scribe-tool": asdict(scribe_tool_stats),
        "standard": asdict(standard_stats),
        "comparison": asdict(comparison),
    }


def generate_report(results: list, metadata: dict) -> str:
    """Generate markdown report from results."""
    analysis = analyze_results(results)
    scribe = analysis["scribe"]
    standard = analysis["standard"]
    comparison = analysis["comparison"]

    lines = []

    # Header
    lines.append("# SWE-bench Benchmark Results")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")

    # Model info - show both the argument and resolved model if different
    model_arg = metadata.get('model', 'unknown')
    model_resolved = metadata.get('model_resolved')
    if model_resolved and model_resolved != model_arg:
        lines.append(f"**Model:** {model_arg} (`{model_resolved}`)")
    else:
        lines.append(f"**Model:** {model_arg}")

    lines.append(f"**Tasks:** {metadata.get('n_tasks', 0)}")
    lines.append(f"**Dataset:** {metadata.get('dataset', 'unknown')}")

    # Scribe version tracking
    scribe_commit = metadata.get('scribe_commit')
    if scribe_commit:
        scribe_version = scribe_commit
        if metadata.get('scribe_dirty'):
            scribe_version += " (dirty)"
        if metadata.get('scribe_branch') and metadata.get('scribe_branch') != 'main':
            scribe_version += f" [{metadata.get('scribe_branch')}]"
        lines.append(f"**Scribe Version:** {scribe_version}")

    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    resolve_diff_pct = comparison["resolve_rate_diff"] * 100
    if resolve_diff_pct > 0:
        lines.append(f"> **Scribe improves resolve rate by {resolve_diff_pct:.1f} percentage points**")
    elif resolve_diff_pct < 0:
        lines.append(f"> **Standard approach has {-resolve_diff_pct:.1f} percentage point higher resolve rate**")
    else:
        lines.append("> **Both approaches have equal resolve rates**")

    tokens_diff_pct = (comparison["tokens_mean_ratio"] - 1) * 100
    if tokens_diff_pct > 0:
        lines.append(f"> Scribe uses {tokens_diff_pct:.1f}% more tokens on average")
    else:
        lines.append(f"> Scribe uses {-tokens_diff_pct:.1f}% fewer tokens on average")

    lines.append("")

    # Key Metrics Table
    lines.append("## Key Metrics")
    lines.append("")

    headers = ["Metric", "Scribe", "Standard", "Difference"]
    rows = [
        [
            "Resolve Rate",
            f"{scribe['resolve_rate']:.1%}",
            f"{standard['resolve_rate']:.1%}",
            f"{comparison['resolve_rate_diff']:+.1%}",
        ],
        [
            "Tasks Resolved",
            f"{scribe['n_resolved']}/{scribe['n_tasks']}",
            f"{standard['n_resolved']}/{standard['n_tasks']}",
            f"{scribe['n_resolved'] - standard['n_resolved']:+d}",
        ],
        [
            "Mean Tokens",
            format_tokens(int(scribe["tokens_mean"])),
            format_tokens(int(standard["tokens_mean"])),
            f"{comparison['tokens_mean_ratio']:.2f}x",
        ],
        [
            "Mean Tool Calls",
            f"{scribe['tool_calls_mean']:.1f}",
            f"{standard['tool_calls_mean']:.1f}",
            f"{comparison['tool_calls_diff']:+.1f}",
        ],
        [
            "Mean Scribe Calls",
            f"{scribe['scribe_calls_mean']:.1f}",
            "N/A",
            "",
        ],
        [
            "Tokens per Resolved",
            format_tokens(int(scribe["tokens_per_resolved"])) if scribe["n_resolved"] > 0 else "N/A",
            format_tokens(int(standard["tokens_per_resolved"])) if standard["n_resolved"] > 0 else "N/A",
            f"{comparison['efficiency_diff']:+,.0f}" if scribe["n_resolved"] > 0 and standard["n_resolved"] > 0 else "N/A",
        ],
    ]

    lines.append(generate_markdown_table(headers, rows, ["left", "right", "right", "right"]))
    lines.append("")

    # Per-Task Results
    lines.append("## Per-Task Results")
    lines.append("")

    # Group results by task_id
    by_task = {}
    for r in results:
        if r.task_id not in by_task:
            by_task[r.task_id] = {}
        by_task[r.task_id][r.mode] = r

    headers = ["Task ID", "Scribe", "Standard", "Scribe Tokens", "Std Tokens"]
    rows = []

    for task_id in sorted(by_task.keys()):
        task_results = by_task[task_id]
        scribe_r = task_results.get("scribe")
        std_r = task_results.get("standard")

        scribe_status = "PASS" if scribe_r and scribe_r.resolved else "FAIL" if scribe_r else "-"
        std_status = "PASS" if std_r and std_r.resolved else "FAIL" if std_r else "-"
        scribe_tokens = format_tokens(scribe_r.total_tokens) if scribe_r else "-"
        std_tokens = format_tokens(std_r.total_tokens) if std_r else "-"

        # Truncate long task IDs
        display_id = task_id[:40] + "..." if len(task_id) > 40 else task_id

        rows.append([display_id, scribe_status, std_status, scribe_tokens, std_tokens])

    lines.append(generate_markdown_table(headers, rows, ["left", "center", "center", "right", "right"]))
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Scribe Mode")
    lines.append("Agent has access to standard tools (bash, read, write, search, edit) plus:")
    lines.append("- `scribe`: Get function/class and all dependencies in one call")
    lines.append("")
    lines.append("### Standard Mode")
    lines.append("Agent has access to basic tools only:")
    lines.append("- `bash`: Execute shell commands")
    lines.append("- `read_file`: Read file contents")
    lines.append("- `write_file`: Write file contents")
    lines.append("- `search_files`: Search for patterns")
    lines.append("- `edit_file`: Edit file by replacing text")
    lines.append("")
    lines.append("### Evaluation")
    lines.append("Tasks are evaluated using SWE-bench's test harness in isolated Docker containers.")
    lines.append("A task is considered 'resolved' when the agent's patch passes all relevant tests.")
    lines.append("")

    return "\n".join(lines)


def save_benchmark_results(
    results: list,
    metadata: dict,
    benchmark_name: str = "swebench",
) -> tuple[Path, Path]:
    """Save benchmark results and report.

    Returns:
        Tuple of (results_path, report_path).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert results to serializable format
    results_data = []
    for r in results:
        d = asdict(r) if hasattr(r, "__dataclass_fields__") else r
        if "tool_calls" in d:
            d["tool_calls"] = [
                asdict(tc) if hasattr(tc, "__dataclass_fields__") else tc
                for tc in d["tool_calls"]
            ]
        results_data.append(d)

    # Build full data structure
    data = {
        "metadata": {
            **metadata,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results_data,
        "analysis": analyze_results(results),
    }

    # Save JSON
    results_path = save_results(benchmark_name, data, prefix="benchmark", timestamp=timestamp)

    # Generate and save report
    report = generate_report(results, metadata)
    report_path = save_report(benchmark_name, report, prefix="report", timestamp=timestamp)

    return results_path, report_path

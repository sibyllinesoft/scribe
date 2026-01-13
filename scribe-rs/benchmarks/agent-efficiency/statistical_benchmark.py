#!/usr/bin/env python3
"""
Statistical Agent Token Efficiency Benchmark

Runs multiple iterations of each target to generate distributions
and compute statistically rigorous comparisons.

Usage:
    ./statistical_benchmark.py                    # Default: 5 iterations
    ./statistical_benchmark.py --iterations 10   # Custom iterations
    ./statistical_benchmark.py --quick           # 3 iterations, subset of targets
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RunResult:
    """Single run result."""
    target_id: str
    iteration: int
    scribe_tokens: int
    scribe_files: int
    scribe_time_ms: float
    naive_tokens: int
    naive_files: int
    naive_tool_calls: int
    naive_time_ms: float
    success: bool = True
    error: str = ""


@dataclass
class TargetStats:
    """Statistical summary for a target."""
    target_id: str
    target_name: str
    category: str
    n_runs: int

    # Scribe stats
    scribe_tokens_mean: float
    scribe_tokens_std: float
    scribe_tokens_min: int
    scribe_tokens_max: int
    scribe_files_mean: float
    scribe_time_mean_ms: float

    # Naive stats
    naive_tokens_mean: float
    naive_tokens_std: float
    naive_tokens_min: int
    naive_tokens_max: int
    naive_files_mean: float
    naive_tool_calls_mean: float
    naive_time_mean_ms: float

    # Derived stats
    token_ratio_mean: float
    token_ratio_std: float
    token_savings_pct_mean: float
    token_savings_pct_std: float
    tool_call_ratio_mean: float

    # Confidence intervals (95%)
    token_savings_ci_low: float
    token_savings_ci_high: float

    # Raw data for distributions
    scribe_tokens_all: list = field(default_factory=list)
    naive_tokens_all: list = field(default_factory=list)
    token_ratios_all: list = field(default_factory=list)


def mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: list) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def confidence_interval_95(values: list) -> tuple[float, float]:
    """Calculate 95% confidence interval using t-distribution approximation."""
    if len(values) < 2:
        m = mean(values)
        return (m, m)

    n = len(values)
    m = mean(values)
    s = std_dev(values)

    # t-value for 95% CI (approximation for common sample sizes)
    t_values = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45,
                8: 2.36, 9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 30: 2.04}
    t = t_values.get(n, 1.96)  # Fall back to z-score for large n

    margin = t * s / math.sqrt(n)
    return (m - margin, m + margin)


def estimate_tokens(content: str) -> int:
    """Rough token estimation (chars / 4)."""
    return len(content) // 4


def count_files_in_output(content: str) -> int:
    """Count files in scribe XML output."""
    return len(re.findall(r"<file>[\s\S]*?</file>", content))


def run_scribe(root: Path, query: str) -> tuple[str, bool, float]:
    """Run scribe and return (output, success, time_ms)."""
    local_bin = root / "target" / "release" / "scribe"
    scribe_cmd = [str(local_bin)] if local_bin.exists() else ["scribe"]

    start = time.perf_counter()
    try:
        result = subprocess.run(
            scribe_cmd + ["--covering-set", query, "--stdout"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if result.returncode == 0:
            return result.stdout, True, elapsed_ms
        return result.stderr, False, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return str(e), False, elapsed_ms


class NaiveDiscovery:
    """Simulates agent discovering dependencies without scribe."""

    def __init__(self, root: Path):
        self.root = root
        self.files_read: set = set()
        self.content_accumulated: str = ""
        self.tool_calls: int = 0

    def grep(self, pattern: str, path: str = ".") -> list[str]:
        """Simulate a grep tool call."""
        self.tool_calls += 1
        try:
            result = subprocess.run(
                ["grep", "-r", "-l", "-E", "--include=*.rs", pattern, path],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
            files = [f for f in files if not f.startswith("target/") and "/tests/" not in f]
            return files
        except Exception:
            return []

    def read_file(self, filepath: str) -> str:
        """Simulate a file read tool call."""
        self.tool_calls += 1
        full_path = self.root / filepath
        if full_path.exists() and filepath not in self.files_read:
            self.files_read.add(filepath)
            try:
                content = full_path.read_text()
                self.content_accumulated += f"\n// === {filepath} ===\n{content}"
                return content
            except Exception:
                return ""
        return ""

    def discover_dependencies(self, target_file: str, target_entity: str, max_depth: int = 3):
        """Simulate agent iteratively discovering dependencies."""
        content = self.read_file(target_file)
        if not content:
            return

        dependencies_to_find = set()

        for match in re.finditer(r"use\s+([\w:]+)", content):
            dep = match.group(1)
            if "scribe" in dep or "crate::" in dep:
                parts = dep.split("::")
                if len(parts) > 1:
                    dependencies_to_find.add(parts[-1])

        for match in re.finditer(r":\s*(&?\s*)(\w+)(?:<|,|\)|\s)", content):
            type_name = match.group(2)
            if type_name[0].isupper() and type_name not in [
                "Result", "Option", "Vec", "HashMap", "HashSet",
                "String", "Box", "Arc", "Mutex", "Self", "Path", "PathBuf"
            ]:
                dependencies_to_find.add(type_name)

        discovered = set()
        to_search = list(dependencies_to_find)[:10]

        for depth in range(max_depth):
            if not to_search:
                break

            next_search = []
            for dep in to_search:
                if dep in discovered:
                    continue
                discovered.add(dep)

                pattern = f"(struct|enum|fn|trait|type|impl)\\s+{dep}"
                files = self.grep(pattern)

                for f in files[:3]:
                    if f.endswith(".rs") and f not in self.files_read:
                        file_content = self.read_file(f)
                        for match in re.finditer(r"use\s+([\w:]+)", file_content):
                            new_dep = match.group(1).split("::")[-1]
                            if new_dep not in discovered:
                                next_search.append(new_dep)

            to_search = next_search[:5]


def run_single_benchmark(root: Path, target: dict) -> RunResult:
    """Run a single benchmark iteration."""
    target_id = target["id"]
    query = target["scribe_query"]
    target_file = query.split(":")[0]
    target_entity = query.split(":")[-1] if ":" in query else ""

    # Scribe approach
    scribe_output, scribe_success, scribe_time = run_scribe(root, query)
    if scribe_success:
        scribe_tokens = estimate_tokens(scribe_output)
        scribe_files = count_files_in_output(scribe_output)
    else:
        scribe_tokens = 0
        scribe_files = 0

    # Naive approach
    start = time.perf_counter()
    discovery = NaiveDiscovery(root=root)
    discovery.discover_dependencies(target_file, target_entity)
    naive_time = (time.perf_counter() - start) * 1000

    naive_tokens = estimate_tokens(discovery.content_accumulated)
    naive_files = len(discovery.files_read)
    naive_tool_calls = discovery.tool_calls

    return RunResult(
        target_id=target_id,
        iteration=0,  # Set by caller
        scribe_tokens=scribe_tokens,
        scribe_files=scribe_files,
        scribe_time_ms=scribe_time,
        naive_tokens=naive_tokens,
        naive_files=naive_files,
        naive_tool_calls=naive_tool_calls,
        naive_time_ms=naive_time,
        success=scribe_success,
        error="" if scribe_success else scribe_output[:100],
    )


def compute_stats(target: dict, results: list[RunResult]) -> TargetStats:
    """Compute statistical summary for a target."""
    successful = [r for r in results if r.success]
    if not successful:
        return None

    scribe_tokens = [r.scribe_tokens for r in successful]
    naive_tokens = [r.naive_tokens for r in successful]
    token_ratios = [n / s if s > 0 else 0 for s, n in zip(scribe_tokens, naive_tokens)]
    savings = [(1 - s / n) * 100 if n > 0 else 0 for s, n in zip(scribe_tokens, naive_tokens)]

    savings_ci = confidence_interval_95(savings)

    return TargetStats(
        target_id=target["id"],
        target_name=target["name"],
        category=target.get("category", "unknown"),
        n_runs=len(successful),
        scribe_tokens_mean=mean(scribe_tokens),
        scribe_tokens_std=std_dev(scribe_tokens),
        scribe_tokens_min=min(scribe_tokens),
        scribe_tokens_max=max(scribe_tokens),
        scribe_files_mean=mean([r.scribe_files for r in successful]),
        scribe_time_mean_ms=mean([r.scribe_time_ms for r in successful]),
        naive_tokens_mean=mean(naive_tokens),
        naive_tokens_std=std_dev(naive_tokens),
        naive_tokens_min=min(naive_tokens),
        naive_tokens_max=max(naive_tokens),
        naive_files_mean=mean([r.naive_files for r in successful]),
        naive_tool_calls_mean=mean([r.naive_tool_calls for r in successful]),
        naive_time_mean_ms=mean([r.naive_time_ms for r in successful]),
        token_ratio_mean=mean(token_ratios),
        token_ratio_std=std_dev(token_ratios),
        token_savings_pct_mean=mean(savings),
        token_savings_pct_std=std_dev(savings),
        tool_call_ratio_mean=mean([r.naive_tool_calls for r in successful]),
        token_savings_ci_low=savings_ci[0],
        token_savings_ci_high=savings_ci[1],
        scribe_tokens_all=scribe_tokens,
        naive_tokens_all=naive_tokens,
        token_ratios_all=token_ratios,
    )


def generate_charts(all_stats: list[TargetStats], output_dir: Path):
    """Generate visualization charts."""
    if not HAS_MATPLOTLIB:
        print("  [skip] matplotlib not installed, skipping charts")
        return

    # Chart 1: Token savings by target
    fig, ax = plt.subplots(figsize=(12, 6))
    targets = [s.target_id for s in all_stats]
    savings = [s.token_savings_pct_mean for s in all_stats]
    errors = [s.token_savings_pct_std for s in all_stats]

    colors = ['#2ecc71' if s > 50 else '#f39c12' if s > 25 else '#e74c3c' for s in savings]
    bars = ax.bar(targets, savings, yerr=errors, capsize=3, color=colors, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Token Savings (%)')
    ax.set_xlabel('Benchmark Target')
    ax.set_title('Token Efficiency: Scribe vs Naive Discovery')
    ax.set_xticklabels(targets, rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars, savings):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'token_savings.png', dpi=150)
    plt.close()

    # Chart 2: Token comparison box plots
    fig, ax = plt.subplots(figsize=(14, 6))

    positions = []
    scribe_data = []
    naive_data = []

    for i, s in enumerate(all_stats):
        positions.append(i)
        scribe_data.append(s.scribe_tokens_all)
        naive_data.append(s.naive_tokens_all)

    bp1 = ax.boxplot(scribe_data, positions=[p - 0.2 for p in positions],
                     widths=0.35, patch_artist=True)
    bp2 = ax.boxplot(naive_data, positions=[p + 0.2 for p in positions],
                     widths=0.35, patch_artist=True)

    for patch in bp1['boxes']:
        patch.set_facecolor('#3498db')
    for patch in bp2['boxes']:
        patch.set_facecolor('#e74c3c')

    ax.set_xticks(positions)
    ax.set_xticklabels([s.target_id for s in all_stats], rotation=45, ha='right')
    ax.set_ylabel('Tokens')
    ax.set_title('Token Distribution: Scribe (blue) vs Naive (red)')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Scribe', 'Naive'], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'token_distributions.png', dpi=150)
    plt.close()

    # Chart 3: Tool calls comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(all_stats))
    width = 0.35

    scribe_calls = [1 for _ in all_stats]  # Always 1
    naive_calls = [s.tool_call_ratio_mean for s in all_stats]

    ax.bar([i - width/2 for i in x], scribe_calls, width, label='Scribe', color='#3498db')
    ax.bar([i + width/2 for i in x], naive_calls, width, label='Naive', color='#e74c3c')

    ax.set_ylabel('Tool Calls')
    ax.set_xlabel('Benchmark Target')
    ax.set_title('Tool Calls: Scribe vs Naive Discovery')
    ax.set_xticks(x)
    ax.set_xticklabels([s.target_id for s in all_stats], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'tool_calls.png', dpi=150)
    plt.close()

    print(f"  Charts saved to {output_dir}")


def generate_report(all_stats: list[TargetStats], all_results: list[RunResult],
                    iterations: int, output_dir: Path) -> str:
    """Generate comprehensive markdown report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Aggregate stats
    total_scribe = sum(s.scribe_tokens_mean for s in all_stats)
    total_naive = sum(s.naive_tokens_mean for s in all_stats)
    overall_ratio = total_naive / total_scribe if total_scribe > 0 else 0
    overall_savings = (1 - total_scribe / total_naive) * 100 if total_naive > 0 else 0

    all_savings = [s.token_savings_pct_mean for s in all_stats]
    savings_std = std_dev(all_savings)

    report = f"""# Agent Token Efficiency Benchmark - Statistical Report

**Generated:** {datetime.now().isoformat()}
**Iterations per target:** {iterations}
**Targets tested:** {len(all_stats)}
**Total runs:** {len(all_results)}

## Executive Summary

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Mean Token Savings** | {mean(all_savings):.1f}% | [{min(s.token_savings_ci_low for s in all_stats):.1f}%, {max(s.token_savings_ci_high for s in all_stats):.1f}%] |
| **Overall Token Ratio** | {overall_ratio:.2f}x | - |
| **Mean Tool Call Reduction** | {mean([s.tool_call_ratio_mean for s in all_stats]):.1f}x | - |

### Key Finding

> **Scribe reduces token usage by {mean(all_savings):.0f}% on average** (std: {savings_std:.1f}%)
> while requiring **{mean([s.tool_call_ratio_mean for s in all_stats]):.0f}x fewer tool calls**.

## Per-Target Results

| Target | Category | Savings | 95% CI | Scribe Tokens | Naive Tokens | Tool Calls |
|--------|----------|---------|--------|---------------|--------------|------------|
"""

    for s in sorted(all_stats, key=lambda x: -x.token_savings_pct_mean):
        report += f"| {s.target_id} | {s.category} | {s.token_savings_pct_mean:.1f}% | [{s.token_savings_ci_low:.0f}%, {s.token_savings_ci_high:.0f}%] | {s.scribe_tokens_mean:,.0f} | {s.naive_tokens_mean:,.0f} | 1/{s.tool_call_ratio_mean:.0f} |\n"

    # Category breakdown
    categories = {}
    for s in all_stats:
        cat = s.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(s.token_savings_pct_mean)

    report += "\n## Results by Category\n\n"
    report += "| Category | Mean Savings | Std Dev | N |\n"
    report += "|----------|--------------|---------|---|\n"
    for cat, savings in sorted(categories.items(), key=lambda x: -mean(x[1])):
        report += f"| {cat} | {mean(savings):.1f}% | {std_dev(savings):.1f}% | {len(savings)} |\n"

    # Depth analysis
    depths = {"shallow": [], "medium": [], "deep": []}
    for s in all_stats:
        # Infer depth from data
        if s.naive_tool_calls_mean < 25:
            depths["shallow"].append(s.token_savings_pct_mean)
        elif s.naive_tool_calls_mean < 35:
            depths["medium"].append(s.token_savings_pct_mean)
        else:
            depths["deep"].append(s.token_savings_pct_mean)

    report += "\n## Results by Dependency Depth\n\n"
    report += "| Depth | Mean Savings | N |\n"
    report += "|-------|--------------|---|\n"
    for depth, savings in depths.items():
        if savings:
            report += f"| {depth} | {mean(savings):.1f}% | {len(savings)} |\n"

    # Statistical methodology
    report += f"""
## Methodology

### Measurement Approach
- Each target was run **{iterations} times** to establish statistical significance
- Token counts estimated as `characters / 4` (standard approximation for code)
- 95% confidence intervals computed using t-distribution

### Scribe Approach
- Single `scribe --covering-set "<file>:<entity>"` call
- Returns target entity with transitive dependencies
- **Always 1 tool call** regardless of complexity

### Naive Approach (Simulated Agent)
1. Read target file (1 tool call)
2. Extract `use` statements and type references
3. For each dependency:
   - Grep for definition (1 tool call)
   - Read matching files (1 tool call each)
4. Repeat for transitive dependencies (max depth: 3)

### Conservative Assumptions
The naive simulation is **optimistic** because real agents:
- Often require multiple grep attempts to find correct files
- May read irrelevant files before finding dependencies
- Have additional overhead from tool call formatting/parsing
- May explore dead ends in large codebases

## Raw Data

<details>
<summary>Click to expand full results JSON</summary>

```json
"""

    # Build JSON data separately to avoid f-string escaping issues
    json_data = [{
        "target_id": s.target_id,
        "n_runs": s.n_runs,
        "scribe": {
            "tokens_mean": round(s.scribe_tokens_mean),
            "tokens_std": round(s.scribe_tokens_std, 1),
            "files_mean": round(s.scribe_files_mean, 1),
        },
        "naive": {
            "tokens_mean": round(s.naive_tokens_mean),
            "tokens_std": round(s.naive_tokens_std, 1),
            "files_mean": round(s.naive_files_mean, 1),
            "tool_calls_mean": round(s.naive_tool_calls_mean, 1),
        },
        "savings": {
            "mean_pct": round(s.token_savings_pct_mean, 1),
            "std_pct": round(s.token_savings_pct_std, 1),
            "ci_95": [round(s.token_savings_ci_low, 1), round(s.token_savings_ci_high, 1)],
        }
    } for s in all_stats]

    report += json.dumps(json_data, indent=2)
    report += """
```

</details>
"""

    # Save report
    report_path = output_dir / f"statistical_report_{timestamp}.md"
    report_path.write_text(report)

    # Also save raw JSON data
    raw_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "iterations": iterations,
            "n_targets": len(all_stats),
        },
        "summary": {
            "mean_savings_pct": mean(all_savings),
            "std_savings_pct": savings_std,
            "overall_token_ratio": overall_ratio,
        },
        "targets": [{
            "id": s.target_id,
            "name": s.target_name,
            "category": s.category,
            "n_runs": s.n_runs,
            "scribe_tokens": s.scribe_tokens_all,
            "naive_tokens": s.naive_tokens_all,
            "token_ratios": s.token_ratios_all,
            "savings_mean": s.token_savings_pct_mean,
            "savings_ci_95": [s.token_savings_ci_low, s.token_savings_ci_high],
        } for s in all_stats]
    }
    json_path = output_dir / f"raw_data_{timestamp}.json"
    json_path.write_text(json.dumps(raw_data, indent=2))

    return report


def main():
    parser = argparse.ArgumentParser(description="Statistical Agent Efficiency Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=5,
                        help="Number of iterations per target (default: 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 iterations, subset of targets")
    parser.add_argument("--targets", "-t", type=str, nargs="*",
                        help="Specific target IDs to run (default: all)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    scribe_root = script_dir.parent.parent
    # Use unified results directory
    results_dir = script_dir.parent / "results" / "agent-efficiency"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load targets
    with open(script_dir / "targets.json") as f:
        data = json.load(f)
        targets = data["targets"]

    # Filter targets if specified
    if args.targets:
        targets = [t for t in targets if t["id"] in args.targets]

    # Quick mode
    iterations = args.iterations
    if args.quick:
        iterations = 3
        # Take a representative subset
        quick_ids = ["token_counter_count", "centrality_calculate", "covering_set_compute",
                     "ast_parse_chunks", "pattern_matcher"]
        targets = [t for t in targets if t["id"] in quick_ids]

    print("=" * 70)
    print("Statistical Agent Token Efficiency Benchmark")
    print("=" * 70)
    print(f"Scribe root: {scribe_root}")
    print(f"Targets: {len(targets)}")
    print(f"Iterations: {iterations}")
    print(f"Total runs: {len(targets) * iterations}")
    print()

    # Run benchmarks
    all_results: list[RunResult] = []
    target_results: dict[str, list[RunResult]] = {}

    for target in targets:
        target_id = target["id"]
        target_results[target_id] = []

        print(f"\n[{target_id}] {target['name']}")
        print(f"  Query: {target['scribe_query']}")

        for i in range(iterations):
            print(f"  Run {i+1}/{iterations}...", end=" ", flush=True)
            result = run_single_benchmark(scribe_root, target)
            result.iteration = i + 1

            if result.success:
                ratio = result.naive_tokens / result.scribe_tokens if result.scribe_tokens > 0 else 0
                print(f"OK (scribe: {result.scribe_tokens:,}, naive: {result.naive_tokens:,}, ratio: {ratio:.2f}x)")
            else:
                print(f"FAILED: {result.error}")

            all_results.append(result)
            target_results[target_id].append(result)

    # Compute statistics
    print("\n" + "=" * 70)
    print("Computing statistics...")
    print("=" * 70)

    all_stats = []
    for target in targets:
        stats = compute_stats(target, target_results[target["id"]])
        if stats:
            all_stats.append(stats)
            print(f"  {stats.target_id}: {stats.token_savings_pct_mean:.1f}% savings "
                  f"(95% CI: [{stats.token_savings_ci_low:.0f}%, {stats.token_savings_ci_high:.0f}%])")

    # Generate charts
    if not args.no_charts:
        print("\nGenerating charts...")
        generate_charts(all_stats, results_dir)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(all_stats, all_results, iterations, results_dir)

    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()

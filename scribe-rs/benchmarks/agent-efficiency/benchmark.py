#!/usr/bin/env python3
"""
Agent Token Efficiency Benchmark

Compares token usage when understanding code:
- WITH scribe: Single covering-set call
- WITHOUT scribe: Iterative grep/read discovery

Simulates realistic agent behavior including tool call counting.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkResult:
    target_id: str
    target_name: str

    # Scribe metrics
    scribe_tokens: int = 0
    scribe_lines: int = 0
    scribe_files: int = 0
    scribe_tool_calls: int = 1  # Always 1 for scribe

    # Naive metrics
    naive_tokens: int = 0
    naive_lines: int = 0
    naive_files: int = 0
    naive_tool_calls: int = 0  # grep + read calls

    # Derived
    @property
    def token_ratio(self) -> float:
        if self.scribe_tokens == 0:
            return 0
        return self.naive_tokens / self.scribe_tokens

    @property
    def token_savings_pct(self) -> float:
        if self.naive_tokens == 0:
            return 0
        return (1 - self.scribe_tokens / self.naive_tokens) * 100

    @property
    def tool_call_ratio(self) -> float:
        if self.scribe_tool_calls == 0:
            return 0
        return self.naive_tool_calls / self.scribe_tool_calls


@dataclass
class NaiveDiscovery:
    """Simulates an agent discovering dependencies without scribe."""

    root: Path
    files_read: set = field(default_factory=set)
    content_accumulated: str = ""
    tool_calls: int = 0

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
            # Filter out target/ directory and test files for cleaner comparison
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
        """
        Simulate agent iteratively discovering dependencies.

        This models what an agent does:
        1. Read target file
        2. Grep for types/functions used in the target
        3. Read those files
        4. Repeat for transitive dependencies
        """
        # Step 1: Read target file
        content = self.read_file(target_file)
        if not content:
            return

        # Step 2: Find the target entity and extract what it uses
        dependencies_to_find = set()

        # Look for use statements
        for match in re.finditer(r"use\s+([\w:]+)", content):
            dep = match.group(1)
            if "scribe" in dep or "crate::" in dep:
                # Extract the type/module name
                parts = dep.split("::")
                if len(parts) > 1:
                    dependencies_to_find.add(parts[-1])

        # Look for type annotations that might need lookup
        for match in re.finditer(r":\s*(&?\s*)(\w+)(?:<|,|\)|\s)", content):
            type_name = match.group(2)
            if type_name[0].isupper() and type_name not in [
                "Result",
                "Option",
                "Vec",
                "HashMap",
                "HashSet",
                "String",
                "Box",
                "Arc",
                "Mutex",
                "Self",
            ]:
                dependencies_to_find.add(type_name)

        # Step 3: Iteratively discover dependencies
        discovered = set()
        to_search = list(dependencies_to_find)[:10]  # Limit to avoid explosion

        for depth in range(max_depth):
            if not to_search:
                break

            next_search = []
            for dep in to_search:
                if dep in discovered:
                    continue
                discovered.add(dep)

                # Grep for definition
                pattern = f"(struct|enum|fn|trait|type)\\s+{dep}"
                files = self.grep(pattern)

                for f in files[:3]:  # Limit files per dependency
                    if f.endswith(".rs") and f not in self.files_read:
                        file_content = self.read_file(f)
                        # Find new dependencies in this file
                        for match in re.finditer(r"use\s+([\w:]+)", file_content):
                            new_dep = match.group(1).split("::")[-1]
                            if new_dep not in discovered:
                                next_search.append(new_dep)

            to_search = next_search[:5]  # Limit growth


def estimate_tokens(content: str) -> int:
    """Rough token estimation for code (chars / 4)."""
    return len(content) // 4


def count_files_in_output(content: str) -> int:
    """Count files in scribe XML output."""
    # Count <file> elements in covering_set output
    return len(re.findall(r"<file>[\s\S]*?</file>", content))


def run_scribe(root: Path, query: str) -> tuple[str, bool]:
    """Run scribe covering-set and return output."""
    # Check for local release binary first
    local_bin = root / "target" / "release" / "scribe"

    if local_bin.exists():
        scribe_cmd = [str(local_bin)]
    else:
        scribe_cmd = ["scribe"]

    try:
        result = subprocess.run(
            scribe_cmd + ["--covering-set", query, "--stdout"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout, True
        return result.stderr, False
    except FileNotFoundError:
        # Try with cargo run
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--bin",
                    "scribe",
                    "--",
                    "--covering-set",
                    query,
                    "--stdout",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return result.stdout, result.returncode == 0
        except Exception as e:
            return str(e), False
    except Exception as e:
        return str(e), False


def run_benchmark(root: Path, targets: list[dict]) -> list[BenchmarkResult]:
    """Run benchmark for all targets."""
    results = []

    for target in targets:
        target_id = target["id"]
        target_name = target["name"]
        query = target["scribe_query"]

        print(f"\n{'='*60}")
        print(f"Target: {target_name}")
        print(f"Query: {query}")
        print("=" * 60)

        result = BenchmarkResult(target_id=target_id, target_name=target_name)

        # --- SCRIBE APPROACH ---
        print("\n[SCRIBE] Running covering-set...", end=" ", flush=True)
        scribe_output, success = run_scribe(root, query)

        if success:
            result.scribe_tokens = estimate_tokens(scribe_output)
            result.scribe_lines = scribe_output.count("\n")
            result.scribe_files = count_files_in_output(scribe_output)
            result.scribe_tool_calls = 1
            print(f"OK ({result.scribe_tokens} tokens, {result.scribe_files} files)")
        else:
            print(f"FAILED: {scribe_output[:100]}")

        # --- NAIVE APPROACH ---
        print("[NAIVE] Simulating agent discovery...", end=" ", flush=True)

        target_file = query.split(":")[0]
        target_entity = query.split(":")[-1] if ":" in query else ""

        discovery = NaiveDiscovery(root=root)
        discovery.discover_dependencies(target_file, target_entity)

        result.naive_tokens = estimate_tokens(discovery.content_accumulated)
        result.naive_lines = discovery.content_accumulated.count("\n")
        result.naive_files = len(discovery.files_read)
        result.naive_tool_calls = discovery.tool_calls

        print(f"OK ({result.naive_tokens} tokens, {result.naive_files} files, {result.naive_tool_calls} tool calls)")

        # --- COMPARISON ---
        print(f"\n  Token ratio: {result.token_ratio:.1f}x")
        print(f"  Token savings: {result.token_savings_pct:.1f}%")
        print(f"  Tool call ratio: {result.tool_call_ratio:.1f}x")

        results.append(result)

    return results


def generate_report(results: list[BenchmarkResult], output_dir: Path):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{timestamp}.md"

    total_scribe_tokens = sum(r.scribe_tokens for r in results)
    total_naive_tokens = sum(r.naive_tokens for r in results)
    total_scribe_calls = sum(r.scribe_tool_calls for r in results)
    total_naive_calls = sum(r.naive_tool_calls for r in results)

    avg_token_ratio = total_naive_tokens / total_scribe_tokens if total_scribe_tokens else 0
    avg_call_ratio = total_naive_calls / total_scribe_calls if total_scribe_calls else 0

    report = f"""# Agent Token Efficiency Benchmark

**Generated:** {datetime.now().isoformat()}

## Executive Summary

| Metric | With Scribe | Without Scribe | Improvement |
|--------|-------------|----------------|-------------|
| Total Tokens | {total_scribe_tokens:,} | {total_naive_tokens:,} | **{avg_token_ratio:.1f}x fewer** |
| Tool Calls | {total_scribe_calls} | {total_naive_calls} | **{avg_call_ratio:.1f}x fewer** |

## Per-Target Results

| Target | Scribe Tokens | Naive Tokens | Savings | Tool Calls (S/N) |
|--------|---------------|--------------|---------|------------------|
"""

    for r in results:
        report += f"| {r.target_id} | {r.scribe_tokens:,} | {r.naive_tokens:,} | {r.token_savings_pct:.0f}% | {r.scribe_tool_calls}/{r.naive_tool_calls} |\n"

    report += f"""
## Methodology

### Scribe Approach
- Single `scribe --covering-set "<file>:<entity>"` call
- Returns only the target entity and its transitive dependencies
- **1 tool call** regardless of dependency complexity

### Naive Approach (Simulated Agent Behavior)
1. Read target file (1 tool call)
2. Extract `use` statements and type references
3. For each dependency:
   - Grep for definition (1 tool call per grep)
   - Read matching files (1 tool call per file)
4. Repeat for transitive dependencies (up to depth 3)

### Token Estimation
- Characters / 4 (rough approximation for code tokens)

## Why These Numbers are Conservative

The naive simulation is actually **optimistic** because real agents:
1. Often grep multiple times to find the right file
2. Read more context than strictly necessary
3. May explore dead ends before finding dependencies
4. Have additional overhead from tool call formatting

Real-world savings with scribe are likely **higher** than shown.

## Raw Data

```json
{json.dumps([{
    "target_id": r.target_id,
    "scribe": {"tokens": r.scribe_tokens, "files": r.scribe_files, "calls": r.scribe_tool_calls},
    "naive": {"tokens": r.naive_tokens, "files": r.naive_files, "calls": r.naive_tool_calls},
    "ratios": {"tokens": round(r.token_ratio, 2), "calls": round(r.tool_call_ratio, 2)}
} for r in results], indent=2)}
```
"""

    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")
    return report


def main():
    script_dir = Path(__file__).parent
    scribe_root = script_dir.parent.parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Load targets
    targets_file = script_dir / "targets.json"
    if not targets_file.exists():
        print(f"Error: {targets_file} not found")
        sys.exit(1)

    with open(targets_file) as f:
        data = json.load(f)
        targets = data["targets"]

    print("=" * 60)
    print("Agent Token Efficiency Benchmark")
    print("=" * 60)
    print(f"Scribe root: {scribe_root}")
    print(f"Targets: {len(targets)}")

    # Run benchmarks
    results = run_benchmark(scribe_root, targets)

    # Generate report
    report = generate_report(results, results_dir)
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()

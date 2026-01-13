#!/usr/bin/env python
"""
Real Agent Benchmark for Code Understanding

Runs actual Claude API calls with tools to measure real token usage
and tool call patterns when understanding code dependencies.

Compares:
- Claude with grep/read tools (naive discovery)
- Scribe covering-set (single call)

Stores all data in JSON for statistical analysis.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    ./real_agent_benchmark.py                    # Default: 3 iterations
    ./real_agent_benchmark.py --iterations 5    # More iterations
    ./real_agent_benchmark.py --model claude-3-5-haiku-20241022  # Use Haiku
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    print("Error: anthropic package required. Install with: pip install anthropic")
    sys.exit(1)


# Tool definitions for Claude
TOOLS = [
    {
        "name": "grep_codebase",
        "description": "Search for a pattern in the codebase. Returns file paths that match. Use this to find where types, functions, or imports are defined.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (e.g., 'struct Config', 'fn process', 'use scribe_core')"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.rs', 'src/**/*.py')",
                    "default": "*.rs"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use this to examine code after finding files with grep.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to repo root)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: start reading from this line (1-indexed)",
                    "default": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: stop reading at this line",
                    "default": -1
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "report_dependencies",
        "description": "Report the dependencies you've found for the target function/type. Call this when you've identified all the key dependencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The function or type you analyzed"
                },
                "dependencies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "file": {"type": "string"},
                            "type": {"type": "string", "enum": ["function", "struct", "enum", "trait", "type", "const", "module"]}
                        },
                        "required": ["name", "file", "type"]
                    },
                    "description": "List of dependencies found"
                },
                "files_examined": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All files you read during analysis"
                }
            },
            "required": ["target", "dependencies", "files_examined"]
        }
    }
]


@dataclass
class ToolCall:
    """Record of a single tool call."""
    name: str
    input: dict
    output: str
    tokens_in_output: int


@dataclass
class AgentRun:
    """Complete record of an agent run."""
    target_id: str
    target_query: str
    model: str
    timestamp: str

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Tool usage
    tool_calls: list = field(default_factory=list)
    num_tool_calls: int = 0

    # Results
    dependencies_found: list = field(default_factory=list)
    files_examined: list = field(default_factory=list)

    # Timing
    duration_ms: float = 0

    # Status
    success: bool = True
    error: str = ""

    # Raw messages for debugging
    messages: list = field(default_factory=list)


@dataclass
class ScribeRun:
    """Record of a scribe covering-set run."""
    target_id: str
    target_query: str
    timestamp: str

    # Output
    output: str = ""
    output_tokens: int = 0
    files_returned: int = 0

    # Timing
    duration_ms: float = 0

    # Status
    success: bool = True
    error: str = ""


class RealAgentBenchmark:
    def __init__(self, repo_root: Path, model: str = "claude-sonnet-4-20250514"):
        self.repo_root = repo_root
        self.model = model
        self.client = anthropic.Anthropic()

    def execute_grep(self, pattern: str, file_pattern: str = "*.rs") -> str:
        """Execute grep and return results."""
        try:
            result = subprocess.run(
                ["grep", "-r", "-l", "-E", f"--include={file_pattern}", pattern, "."],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
            # Filter out target directory and test files
            files = [f for f in files if not f.startswith("./target/") and "/tests/" not in f]
            if not files:
                return "No matches found."
            return "\n".join(files[:20])  # Limit to 20 files
        except subprocess.TimeoutExpired:
            return "Error: grep timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_read_file(self, path: str, start_line: int = 1, end_line: int = -1) -> str:
        """Read file contents."""
        try:
            # Normalize path
            if path.startswith("./"):
                path = path[2:]
            full_path = self.repo_root / path

            if not full_path.exists():
                return f"Error: File not found: {path}"

            content = full_path.read_text()
            lines = content.split("\n")

            if end_line == -1:
                end_line = len(lines)

            # Clamp to valid range
            start_line = max(1, start_line)
            end_line = min(len(lines), end_line)

            selected_lines = lines[start_line-1:end_line]

            # Add line numbers
            numbered = [f"{i+start_line:4d} | {line}" for i, line in enumerate(selected_lines)]

            # Truncate if too long
            result = "\n".join(numbered)
            if len(result) > 15000:
                result = result[:15000] + "\n... (truncated)"

            return result
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result."""
        if tool_name == "grep_codebase":
            return self.execute_grep(
                tool_input["pattern"],
                tool_input.get("file_pattern", "*.rs")
            )
        elif tool_name == "read_file":
            return self.execute_read_file(
                tool_input["path"],
                tool_input.get("start_line", 1),
                tool_input.get("end_line", -1)
            )
        elif tool_name == "report_dependencies":
            # This is the final report - just acknowledge it
            return "Dependencies recorded. Analysis complete."
        else:
            return f"Unknown tool: {tool_name}"

    def run_agent(self, target_id: str, target_query: str) -> AgentRun:
        """Run Claude with tools to discover dependencies for a target."""
        run = AgentRun(
            target_id=target_id,
            target_query=target_query,
            model=self.model,
            timestamp=datetime.now().isoformat()
        )

        # Extract file and entity from query
        parts = target_query.split(":")
        target_file = parts[0]
        target_entity = parts[1] if len(parts) > 1 else ""

        system_prompt = """You are a code analysis assistant. Your task is to find all dependencies
for a given function or type in a Rust codebase.

Use the grep_codebase tool to search for type and function definitions.
Use the read_file tool to examine the code.

Be efficient - don't read files you don't need. Focus on finding:
1. The target function/type definition
2. Types used in the function signature and body
3. Functions called by the target
4. Traits implemented or required

When you've identified the key dependencies, call report_dependencies with your findings."""

        user_message = f"""Find all dependencies for the function/type `{target_entity}` in the file `{target_file}`.

Start by reading the target file to understand what the function does and what it depends on.
Then trace the dependencies to find their definitions.

Be thorough but efficient - focus on direct dependencies that would be needed to understand
how this code works."""

        messages = [{"role": "user", "content": user_message}]

        start_time = time.perf_counter()

        try:
            # Agentic loop
            max_iterations = 20
            for iteration in range(max_iterations):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages
                )

                # Accumulate token usage
                run.input_tokens += response.usage.input_tokens
                run.output_tokens += response.usage.output_tokens

                # Check stop reason
                if response.stop_reason == "end_turn":
                    # Agent finished without calling report_dependencies
                    break

                if response.stop_reason == "tool_use":
                    # Process tool calls
                    tool_results = []

                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input
                            tool_id = content_block.id

                            # Execute the tool
                            result = self.handle_tool_call(tool_name, tool_input)

                            # Record the tool call
                            run.tool_calls.append(ToolCall(
                                name=tool_name,
                                input=tool_input,
                                output=result[:500],  # Truncate for storage
                                tokens_in_output=len(result) // 4
                            ))
                            run.num_tool_calls += 1

                            # Check if this is the final report
                            if tool_name == "report_dependencies":
                                run.dependencies_found = tool_input.get("dependencies", [])
                                run.files_examined = tool_input.get("files_examined", [])

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result
                            })

                    # Add assistant message and tool results to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                    # Check if we got the final report
                    if any(tc.name == "report_dependencies" for tc in run.tool_calls[-len(tool_results):]):
                        break
                else:
                    # Unknown stop reason
                    break

            run.total_tokens = run.input_tokens + run.output_tokens
            run.duration_ms = (time.perf_counter() - start_time) * 1000
            run.success = True

        except Exception as e:
            run.success = False
            run.error = str(e)
            run.duration_ms = (time.perf_counter() - start_time) * 1000

        return run

    def run_scribe(self, target_id: str, target_query: str) -> ScribeRun:
        """Run scribe covering-set for comparison."""
        run = ScribeRun(
            target_id=target_id,
            target_query=target_query,
            timestamp=datetime.now().isoformat()
        )

        scribe_bin = self.repo_root / "target" / "release" / "scribe"
        if not scribe_bin.exists():
            scribe_bin = "scribe"

        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                [str(scribe_bin), "--covering-set", target_query, "--stdout"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            run.duration_ms = (time.perf_counter() - start_time) * 1000

            if result.returncode == 0:
                run.output = result.stdout
                run.output_tokens = len(result.stdout) // 4
                # Count files in XML output
                run.files_returned = len(re.findall(r"<file>[\s\S]*?</file>", result.stdout))
                run.success = True
            else:
                run.success = False
                run.error = result.stderr[:500]

        except Exception as e:
            run.success = False
            run.error = str(e)
            run.duration_ms = (time.perf_counter() - start_time) * 1000

        return run


def run_benchmark(repo_root: Path, targets: list, iterations: int, model: str, output_dir: Path):
    """Run the full benchmark."""
    benchmark = RealAgentBenchmark(repo_root, model)

    all_agent_runs = []
    all_scribe_runs = []

    print(f"\n{'='*70}")
    print(f"Real Agent Benchmark")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Targets: {len(targets)}")
    print(f"Iterations: {iterations}")
    print(f"Total runs: {len(targets) * iterations * 2}")  # agent + scribe
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

            # Run agent
            print(f"    [AGENT] Running Claude with tools...", end=" ", flush=True)
            agent_run = benchmark.run_agent(target_id, query)

            if agent_run.success:
                print(f"OK ({agent_run.total_tokens:,} tokens, {agent_run.num_tool_calls} calls, {agent_run.duration_ms:.0f}ms)")
            else:
                print(f"FAILED: {agent_run.error[:50]}")

            all_agent_runs.append(agent_run)

            # Run scribe
            print(f"    [SCRIBE] Running covering-set...", end=" ", flush=True)
            scribe_run = benchmark.run_scribe(target_id, query)

            if scribe_run.success:
                print(f"OK ({scribe_run.output_tokens:,} tokens, {scribe_run.files_returned} files, {scribe_run.duration_ms:.0f}ms)")
            else:
                print(f"FAILED: {scribe_run.error[:50]}")

            all_scribe_runs.append(scribe_run)

            # Brief comparison
            if agent_run.success and scribe_run.success:
                ratio = agent_run.total_tokens / scribe_run.output_tokens if scribe_run.output_tokens > 0 else 0
                print(f"    [RATIO] Agent used {ratio:.1f}x more tokens than scribe")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to serializable format
    agent_data = []
    for run in all_agent_runs:
        d = asdict(run)
        d["tool_calls"] = [asdict(tc) for tc in run.tool_calls]
        agent_data.append(d)

    scribe_data = [asdict(run) for run in all_scribe_runs]

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "iterations": iterations,
            "n_targets": len(targets)
        },
        "agent_runs": agent_data,
        "scribe_runs": scribe_data
    }

    results_file = output_dir / f"real_benchmark_{timestamp}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\n\nResults saved to: {results_file}")

    # Generate summary
    generate_summary(all_agent_runs, all_scribe_runs, output_dir, timestamp)


def generate_summary(agent_runs: list, scribe_runs: list, output_dir: Path, timestamp: str):
    """Generate summary statistics."""
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")

    # Group by target
    targets = {}
    for agent_run, scribe_run in zip(agent_runs, scribe_runs):
        tid = agent_run.target_id
        if tid not in targets:
            targets[tid] = {"agent": [], "scribe": []}
        targets[tid]["agent"].append(agent_run)
        targets[tid]["scribe"].append(scribe_run)

    summary_data = []

    print(f"\n{'Target':<30} {'Agent Tokens':>15} {'Scribe Tokens':>15} {'Ratio':>10} {'Tool Calls':>12}")
    print("-" * 85)

    total_agent_tokens = 0
    total_scribe_tokens = 0
    total_tool_calls = 0

    for tid, runs in targets.items():
        agent_tokens = [r.total_tokens for r in runs["agent"] if r.success]
        scribe_tokens = [r.output_tokens for r in runs["scribe"] if r.success]
        tool_calls = [r.num_tool_calls for r in runs["agent"] if r.success]

        if agent_tokens and scribe_tokens:
            avg_agent = sum(agent_tokens) / len(agent_tokens)
            avg_scribe = sum(scribe_tokens) / len(scribe_tokens)
            avg_calls = sum(tool_calls) / len(tool_calls)
            ratio = avg_agent / avg_scribe if avg_scribe > 0 else 0

            total_agent_tokens += avg_agent
            total_scribe_tokens += avg_scribe
            total_tool_calls += avg_calls

            print(f"{tid:<30} {avg_agent:>15,.0f} {avg_scribe:>15,.0f} {ratio:>10.1f}x {avg_calls:>12.1f}")

            summary_data.append({
                "target_id": tid,
                "agent_tokens_mean": avg_agent,
                "scribe_tokens_mean": avg_scribe,
                "token_ratio": ratio,
                "tool_calls_mean": avg_calls,
                "n_runs": len(agent_tokens)
            })

    print("-" * 85)

    if total_scribe_tokens > 0:
        overall_ratio = total_agent_tokens / total_scribe_tokens
        avg_tool_calls = total_tool_calls / len(targets)

        print(f"{'TOTAL':<30} {total_agent_tokens:>15,.0f} {total_scribe_tokens:>15,.0f} {overall_ratio:>10.1f}x {avg_tool_calls:>12.1f}")

        print(f"\n\nKey Findings:")
        print(f"  - Agent uses {overall_ratio:.1f}x more tokens than scribe on average")
        print(f"  - Agent makes {avg_tool_calls:.1f} tool calls on average")
        print(f"  - Scribe requires 1 call regardless of complexity")

        savings_pct = (1 - 1/overall_ratio) * 100
        print(f"  - Token savings with scribe: {savings_pct:.0f}%")

    # Save summary
    summary_file = output_dir / f"real_summary_{timestamp}.json"
    summary_file.write_text(json.dumps({
        "overall": {
            "agent_tokens_total": total_agent_tokens,
            "scribe_tokens_total": total_scribe_tokens,
            "token_ratio": overall_ratio if total_scribe_tokens > 0 else 0,
            "avg_tool_calls": total_tool_calls / len(targets) if targets else 0
        },
        "per_target": summary_data
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Real Agent Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=3,
                        help="Number of iterations per target (default: 3)")
    parser.add_argument("--model", "-m", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use")
    parser.add_argument("--targets", "-t", type=str, nargs="*",
                        help="Specific target IDs to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 iteration, 3 targets")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

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

    run_benchmark(repo_root, targets, iterations, args.model, results_dir)


if __name__ == "__main__":
    main()

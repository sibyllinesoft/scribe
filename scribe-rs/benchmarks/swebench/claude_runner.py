#!/usr/bin/env python3
"""
Run SWE-bench tasks using Claude Code CLI with JSON output.
"""

import json
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from common.results import save_results


def load_task(task_id: str, dataset: str = "princeton-nlp/SWE-bench_Lite") -> dict:
    """Load a specific task from SWE-bench."""
    if not HAS_DATASETS:
        raise ImportError("datasets package required")

    ds = load_dataset(dataset, split="test")
    for item in ds:
        if item.get("instance_id") == task_id:
            return {
                "instance_id": item.get("instance_id", ""),
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement", ""),
                "patch": item.get("patch", ""),
                "version": item.get("version", ""),
            }
    raise ValueError(f"Task {task_id} not found")


def get_docker_image(task_id: str) -> str:
    """Get Docker image name for a task."""
    return f"sweb.eval.x86_64.{task_id}:latest"


def setup_container(task_id: str) -> tuple[str, Path]:
    """Start Docker container and copy repo."""
    image = get_docker_image(task_id)

    # Start container
    result = subprocess.run(
        ["docker", "run", "-d", "--rm", image, "sleep", "infinity"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")
    container_id = result.stdout.strip()[:12]

    # Create temp dir and copy repo
    tmp_dir = Path(tempfile.mkdtemp(prefix="swebench_"))
    subprocess.run(
        ["docker", "cp", f"{container_id}:/testbed/.", str(tmp_dir)],
        check=True, capture_output=True
    )

    return container_id, tmp_dir


def cleanup_container(container_id: str, tmp_dir: Path):
    """Stop container and clean up."""
    subprocess.run(["docker", "stop", container_id], capture_output=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def setup_scribe_hooks(work_dir: Path):
    """Set up scribe reminder hooks in the work directory."""
    hooks_dir = work_dir / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Write the scribe reminder hook
    hook_script = hooks_dir / "scribe_reminder.py"
    hook_script.write_text('''#!/usr/bin/env python3
"""Hook to remind agents about scribe when exploring."""
import json
import sys

EXPLORATION_THRESHOLD = 3

def get_counts(sid):
    try:
        with open(f"/tmp/claude_hook_{sid}.json") as f:
            return json.load(f)
    except:
        return {"count": 0, "reminded": False}

def save_counts(sid, c):
    with open(f"/tmp/claude_hook_{sid}.json", "w") as f:
        json.dump(c, f)

def main():
    try:
        data = json.load(sys.stdin)
    except:
        sys.exit(0)

    sid = data.get("session_id", "x")
    tool = data.get("tool_name", "")
    event = data.get("hook_event_name", "")

    if event != "PreToolUse" or tool not in ("Read", "Grep"):
        sys.exit(0)

    counts = get_counts(sid)
    counts["count"] = counts.get("count", 0) + 1

    if counts["count"] >= EXPLORATION_THRESHOLD and not counts.get("reminded"):
        counts["reminded"] = True
        save_counts(sid, counts)

        reminder = f"""EFFICIENCY TIP: You have made {counts["count"]} Read/Grep calls.

Use scribe to get complete context in ONE call instead:

  # Get a function and ALL its dependencies:
  scribe --covering-set "path/to/file.py:function_name" --stdout

  # Get prioritized context for a directory:
  scribe --token-target 8000 path/to/dir --stdout

Scribe returns the target plus all types, functions, and constants it uses."""

        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": reminder
            }
        }
        print(json.dumps(output))
        sys.exit(0)

    save_counts(sid, counts)
    sys.exit(0)

if __name__ == "__main__":
    main()
''')
    hook_script.chmod(0o755)

    # Write settings.json
    settings = work_dir / ".claude" / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Read|Grep",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {hook_script}"
                        }
                    ]
                }
            ]
        }
    }, indent=2))


def run_with_claude(
    task: dict,
    work_dir: Path,
    mode: str = "standard",
    timeout_s: int = 600,
    model: str = "sonnet",
) -> dict:
    """Run Claude Code on a task and capture JSON output."""

    issue = task.get("problem_statement", "")

    # Build prompt based on mode
    if mode == "scribe-context":
        # Pre-fetch scribe context
        from runner import fetch_scribe_context
        context = fetch_scribe_context(work_dir, issue)
        if context:
            prompt = f"""Fix the following issue in this repository.

Here is the COMPLETE relevant code context you need:

{context}

ISSUE:
{issue}

IMPORTANT: The context above contains all the relevant code. DO NOT re-explore the codebase. Go directly to implementing the fix. After fixing, run tests to verify."""
        else:
            prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works."""

    elif mode == "scribe-tool":
        from runner import infer_target_directories
        target_dirs = infer_target_directories(work_dir, issue)
        if target_dirs:
            dir_suggestions = "\n".join(f"  - {d.relative_to(work_dir)}" for d in target_dirs[:3])
            dir_example = str(target_dirs[0].relative_to(work_dir))
        else:
            dir_suggestions = "  - Use grep to find the relevant directory first"
            dir_example = "src/"

        prompt = f"""Fix the following issue in this repository.

STEP 1: Run this scribe command to get all relevant code:
```
scribe --output-format text -o /dev/stdout --token-target 8000 {dir_example}
```

STEP 2: After scribe completes, implement the fix using ONLY Edit/Write tools.

CRITICAL: After running scribe, you MUST NOT use these tools:
- NO Read tool (scribe already showed you the code)
- NO Grep tool (scribe already found the relevant files)
- NO find/grep/cat bash commands (scribe already explored)

The scribe output contains the complete relevant codebase. Trust it and go directly to editing.

STEP 3: Run tests to verify your fix.

ISSUE:
{issue}"""

    elif mode == "scribe-hooks":
        # Standard prompt but with hooks that remind about scribe
        setup_scribe_hooks(work_dir)
        prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works.

NOTE: The 'scribe' tool is available for efficient codebase exploration. Example:
  scribe --covering-set "path/to/file.py:function_name" --stdout"""

    else:  # standard
        prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works."""

    # Run Claude Code with JSON output
    # Write prompt to temp file to avoid shell escaping issues
    prompt_file = work_dir / ".claude_prompt.txt"
    prompt_file.write_text(prompt)

    cmd = [
        "claude",
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--model", model,
        "--dangerously-skip-permissions",
    ]

    print(f"    Running Claude Code ({model})...")
    print(f"    Timeout: {timeout_s}s")
    print(f"    Work dir: {work_dir}")

    start_time = datetime.now()
    try:
        # Pipe prompt via stdin
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=work_dir,
        )
        duration = (datetime.now() - start_time).total_seconds()

        # Parse JSON output
        output = {
            "success": result.returncode == 0,
            "duration_s": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

        # Parse stream-json output (newline-delimited JSON)
        events = []
        total_input_tokens = 0
        total_output_tokens = 0
        tool_calls = []

        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append(event)

                # Extract token usage from message events
                if event.get("type") == "result":
                    usage = event.get("usage", {})
                    total_input_tokens = usage.get("input_tokens", 0)
                    total_output_tokens = usage.get("output_tokens", 0)

                # Track tool calls
                if event.get("type") == "tool_use":
                    tool_calls.append({
                        "name": event.get("tool", ""),
                        "input": event.get("input", {}),
                    })
                elif event.get("type") == "tool_result":
                    if tool_calls:
                        tool_calls[-1]["result_preview"] = str(event.get("output", ""))[:200]

            except json.JSONDecodeError:
                continue

        output["events"] = events
        output["total_tokens"] = total_input_tokens + total_output_tokens
        output["input_tokens"] = total_input_tokens
        output["output_tokens"] = total_output_tokens
        output["tool_calls"] = tool_calls
        output["num_tool_calls"] = len(tool_calls)

        return output

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "duration_s": duration,
            "error": f"Timeout after {timeout_s}s",
            "timeout": True,
        }
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "duration_s": duration,
            "error": str(e),
        }


def run_task(
    task_id: str,
    mode: str = "standard",
    timeout_s: int = 600,
    model: str = "sonnet",
) -> dict:
    """Run a single SWE-bench task with Claude Code."""

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Mode: {mode}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    # Load task
    print("Loading task...")
    task = load_task(task_id)

    # Setup container
    print("Setting up container...")
    container_id, work_dir = setup_container(task_id)
    print(f"  Container: {container_id}")
    print(f"  Work dir: {work_dir}")

    try:
        # Run with Claude
        result = run_with_claude(task, work_dir, mode, timeout_s, model)
        result["task_id"] = task_id
        result["mode"] = mode
        result["model"] = model
        result["timestamp"] = datetime.now().isoformat()

        # Print summary
        if result.get("success"):
            tokens = result.get("total_tokens", "?")
            print(f"    Completed in {result['duration_s']:.1f}s")
            print(f"    Tokens: {tokens}")
        else:
            error = result.get("error", "Unknown error")
            print(f"    FAILED: {error}")

        return result

    finally:
        # Cleanup
        print("Cleaning up...")
        cleanup_container(container_id, work_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SWE-bench tasks with Claude Code")
    parser.add_argument("--task-ids", nargs="+", required=True, help="Task IDs to run")
    parser.add_argument("--mode", choices=["standard", "scribe-context", "scribe-tool", "scribe-hooks", "all"],
                       default="standard", help="Mode to run")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task")
    parser.add_argument("--model", default="sonnet", help="Claude model to use")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    results = []
    modes = ["standard", "scribe-context", "scribe-tool", "scribe-hooks"] if args.mode == "all" else [args.mode]

    for task_id in args.task_ids:
        for mode in modes:
            result = run_task(task_id, mode, args.timeout, args.model)
            results.append(result)

    # Save results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task_ids": args.task_ids,
            "modes": modes,
            "model": args.model,
            "timeout": args.timeout,
        },
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"claude_benchmark_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for r in results:
        status = "OK" if r.get("success") else "FAILED"
        tokens = r.get("total_tokens", 0)
        print(f"{r['task_id']} | {r['mode']:15} | {r['duration_s']:6.1f}s | {tokens:>10} tokens | {status}")


if __name__ == "__main__":
    main()

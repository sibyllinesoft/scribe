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
    """Load a specific task from SWE-bench.

    Supported datasets:
    - princeton-nlp/SWE-bench_Lite (Python only)
    - swe-bench/SWE-bench_Multilingual (TypeScript, Rust, Go, etc.)
    """
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

    # Write the scribe enforcement hook
    hook_script = hooks_dir / "scribe_enforcer.py"
    hook_script.write_text('''#!/usr/bin/env python3
"""Hook to enforce scribe usage: blocks Read on code files, blocks piped scribe."""
import json, sys, os, re

CODE_EXTS = {".py",".pyi",".js",".jsx",".ts",".tsx",".mjs",".cjs",".go",".rs",
    ".java",".kt",".c",".h",".cpp",".hpp",".cc",".rb",".php",".swift",".scala",
    ".cs",".lua",".ex",".exs",".hs",".ml",".clj",".vue",".svelte",".sol"}
ALLOWED_EXTS = {".md",".txt",".json",".yaml",".yml",".toml",".ini",".cfg",
    ".xml",".html",".css",".env",".lock",".sum",".sh",".bash"}

def get_state(sid):
    try:
        with open(f"/tmp/claude_hook_{sid}.json") as f: return json.load(f)
    except: return {"grep_count": 0, "reminded": False}

def save_state(sid, s):
    with open(f"/tmp/claude_hook_{sid}.json", "w") as f: json.dump(s, f)

def main():
    try: data = json.load(sys.stdin)
    except: sys.exit(0)

    sid = data.get("session_id", "x")
    tool = data.get("tool_name", "")
    event = data.get("hook_event_name", "")
    inp = data.get("tool_input", {})

    if event != "PreToolUse": sys.exit(0)

    state = get_state(sid)
    msg, allow = "", True

    # BLOCK: Read on code files
    if tool == "Read":
        fp = inp.get("file_path", "")
        ext = os.path.splitext(fp)[1].lower()
        if ext in CODE_EXTS:
            msg = f"""BLOCKED: Use scribe instead of Read for code files.

You tried to read: {fp}

USE SCRIBE to get the code WITH its dependencies:

  scribe --covering-set "{fp}:FUNCTION_NAME" --stdout

Replace FUNCTION_NAME with the entity you need. This returns the function
AND all types/functions it depends on. One call, complete context.

Read is only allowed for non-code files (config, docs, etc.)."""
            allow = False
        elif ext not in ALLOWED_EXTS and ext:
            pass  # Unknown ext, allow

    # BLOCK: Bash piping scribe output
    elif tool == "Bash":
        cmd = inp.get("command", "")
        pipes = [r"scribe.*\\|\\s*head", r"scribe.*\\|\\s*tail", r"scribe.*\\|\\s*grep",
                 r"scribe.*\\|\\s*awk", r"scribe.*\\|\\s*sed", r"scribe.*>\\s*/dev/null"]
        for p in pipes:
            if re.search(p, cmd, re.I):
                msg = f"""BLOCKED: Do not pipe scribe output.

You ran: {cmd[:80]}...

This defeats scribe's purpose. Scribe returns exactly what you need.
Truncating loses context and wastes tokens.

CORRECT: scribe --covering-set "file.py:func" --stdout
Use --token-target to limit size if needed."""
                allow = False
                break

    # WARN: Too many Greps
    elif tool == "Grep":
        state["grep_count"] = state.get("grep_count", 0) + 1
        if state["grep_count"] >= 3 and not state.get("reminded"):
            state["reminded"] = True
            msg = f"""STOP: {state["grep_count"]} Grep calls. Use scribe instead.

  scribe --covering-set "file.py:function" --stdout

One call = function + all dependencies. Stop grepping."""

    save_state(sid, state)
    if msg:
        out = {"hookSpecificOutput": {"hookEventName": "PreToolUse",
               "permissionDecision": "allow" if allow else "deny",
               "permissionDecisionReason": msg}}
        print(json.dumps(out))
    sys.exit(0)

if __name__ == "__main__": main()
''')
    hook_script.chmod(0o755)

    # Write settings.json - hook applies to Read, Grep, AND Bash
    settings = work_dir / ".claude" / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Read|Grep|Bash",
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
            # No context available - return failure without crashing benchmark
            return {
                "success": False,
                "error": f"scribe-context failed: no context returned for {work_dir}",
                "tokens": 0,
                "duration_s": 0,
                "patch": "",
            }

    elif mode == "scribe-tool":
        from runner import detect_repo_language, extract_code_references

        language = detect_repo_language(work_dir)

        # Set up hooks to remind about scribe when exploring
        setup_scribe_hooks(work_dir)

        # Extract key terms from issue for query-hint
        refs = extract_code_references(issue, language)
        import re
        camel_matches = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b', issue)
        refs.extend(camel_matches[:5])
        keywords = list(set(refs))[:8]
        query_hint = " ".join(keywords) if keywords else "main"

        prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

=== SCRIBE: YOUR PRIMARY EXPLORATION TOOL ===

Scribe returns a function/class AND ALL ITS DEPENDENCIES in a single call.
This replaces grep + read exploration. Use scribe INSTEAD of manual file reading.

**How to use (ALWAYS use --covering-set for specific entities):**
```
scribe --covering-set "path/to/file.py:function_name" --stdout
scribe --covering-set "path/to/file.ts:ClassName" --stdout
scribe --covering-set "path/to/file.go:MethodName" --stdout
```

This returns the target PLUS every type, function, and constant it depends on.
Complete dependency graph. One call. No manual import tracing needed.

=== WORKFLOW ===

1. Grep ONCE to find the file/function mentioned in the issue
2. scribe --covering-set on that target to get ALL context
3. Fix the code using what scribe returned
4. Run tests

=== MANDATORY RULES ===

After scribe returns context:
- DO NOT call Read on files scribe already returned. You have everything.
- DO NOT grep for more context. The dependency graph is complete.
- DO NOT pipe scribe through head/tail/grep. Let it complete.

The purpose of scribe is to ELIMINATE iterative exploration.
One scribe call replaces 10+ grep/read calls. USE IT.

After fixing, run tests to verify."""

    elif mode == "scribe-hooks":
        # Standard prompt but with hooks that remind about scribe
        setup_scribe_hooks(work_dir)
        prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

HINT: Use `scribe --covering-set "file:function" --stdout` to get a function with all its dependencies in one call. This is faster than reading files individually.

After fixing, run the relevant tests to verify your fix works."""

    else:  # standard
        prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works."""

    # Run Claude Code with JSON output
    # Write prompt to temp file to avoid shell escaping issues
    prompt_file = work_dir / ".claude_prompt.txt"
    prompt_file.write_text(prompt)

    # Output file for streaming results (preserves output even if interrupted)
    output_file = work_dir / ".claude_output.jsonl"

    cmd = [
        "claude",
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--model", model,
        "--dangerously-skip-permissions",
    ]

    print(f"    Running Claude Code ({model})...")
    print(f"    Timeout: {timeout_s}s (soft - will wait for completion)")
    print(f"    Work dir: {work_dir}")

    start_time = datetime.now()
    try:
        # Use Popen to stream output to file, preserving it even on interrupt
        with open(output_file, 'w') as outf:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=outf,
                stderr=subprocess.PIPE,
                text=True,
                cwd=work_dir,
            )
            # Send prompt and close stdin
            process.stdin.write(prompt)
            process.stdin.close()

            # Wait for completion (no forced timeout - let Claude finish)
            # Check periodically and warn if exceeding soft timeout
            warned = False
            while process.poll() is None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_s and not warned:
                    print(f"    Warning: Exceeded {timeout_s}s, still running...")
                    warned = True
                import time
                time.sleep(1)

            returncode = process.returncode
            stderr = process.stderr.read()

        duration = (datetime.now() - start_time).total_seconds()

        # Read output from file
        stdout = output_file.read_text() if output_file.exists() else ""

        # Parse JSON output
        output = {
            "success": returncode == 0,
            "duration_s": duration,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
        }

        # Parse stream-json output (newline-delimited JSON)
        events = []
        total_input_tokens = 0
        total_output_tokens = 0
        tool_calls = []
        task_succeeded = False  # Track if task completed successfully

        for line in stdout.strip().split('\n'):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append(event)

                # Extract token usage and success from result events
                if event.get("type") == "result":
                    # Check if this is a successful completion
                    if event.get("subtype") == "success":
                        task_succeeded = True
                        # Use token usage from successful result (include cache tokens)
                        usage = event.get("usage", {})
                        total_input_tokens = (
                            usage.get("input_tokens", 0) +
                            usage.get("cache_creation_input_tokens", 0) +
                            usage.get("cache_read_input_tokens", 0)
                        )
                        total_output_tokens = usage.get("output_tokens", 0)
                    elif not task_succeeded:
                        # Only use tokens from non-success if we haven't seen success
                        usage = event.get("usage", {})
                        if usage.get("input_tokens", 0) > 0 or usage.get("cache_read_input_tokens", 0) > 0:
                            total_input_tokens = (
                                usage.get("input_tokens", 0) +
                                usage.get("cache_creation_input_tokens", 0) +
                                usage.get("cache_read_input_tokens", 0)
                            )
                            total_output_tokens = usage.get("output_tokens", 0)

                # Track tool calls - they're nested in assistant message content
                if event.get("type") == "assistant":
                    content = event.get("message", {}).get("content", [])
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            tool_calls.append({
                                "name": item.get("name", ""),
                                "input": item.get("input", {}),
                            })
                elif event.get("type") == "user":
                    # Tool results are in user messages
                    content = event.get("message", {}).get("content", [])
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            if tool_calls:
                                tool_calls[-1]["result_preview"] = str(item.get("content", ""))[:200]

            except json.JSONDecodeError:
                continue

        output["events"] = events
        output["total_tokens"] = total_input_tokens + total_output_tokens
        output["input_tokens"] = total_input_tokens
        output["output_tokens"] = total_output_tokens
        output["tool_calls"] = tool_calls
        output["num_tool_calls"] = len(tool_calls)

        # Override success based on task completion, not just return code
        # This handles cases where infrastructure errors occur after task completion
        if task_succeeded:
            output["success"] = True

        return output

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        # Try to read any partial output
        stdout = output_file.read_text() if output_file.exists() else ""
        return {
            "success": False,
            "duration_s": duration,
            "error": str(e),
            "stdout": stdout,
        }


def run_task(
    task_id: str,
    mode: str = "standard",
    timeout_s: int = 600,
    model: str = "sonnet",
    dataset: str = "princeton-nlp/SWE-bench_Lite",
) -> dict:
    """Run a single SWE-bench task with Claude Code."""

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Mode: {mode}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    # Load task
    print("Loading task...")
    task = load_task(task_id, dataset)

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
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                       help="Dataset to use (e.g., swe-bench/SWE-bench_Multilingual)")

    args = parser.parse_args()

    results = []
    modes = ["standard", "scribe-context", "scribe-tool"] if args.mode == "all" else [args.mode]

    for task_id in args.task_ids:
        for mode in modes:
            result = run_task(task_id, mode, args.timeout, args.model, args.dataset)
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

"""SWE-bench task runner using OpenCode.

Executes SWE-bench tasks with OpenCode agent and captures results.
Uses OpenCode's `run` command for realistic agent behavior.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def get_scribe_path() -> Optional[str]:
    """Get path to scribe binary."""
    path = shutil.which("scribe")
    if path:
        return path

    # Check common install locations
    home = Path.home()
    candidates = [
        home / ".cargo" / "bin" / "scribe",
        home / ".local" / "bin" / "scribe",
        Path("/usr/local/bin/scribe"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def extract_code_references(issue_text: str) -> list[str]:
    """Extract file paths and code references from issue text.

    Looks for patterns like:
    - module.path.ClassName
    - path/to/file.py
    - function_name()
    """
    refs = []

    # Python dotted paths (e.g., django.core.files.uploadhandler)
    dotted_pattern = r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){2,})\b'
    for match in re.finditer(dotted_pattern, issue_text, re.IGNORECASE):
        refs.append(match.group(1))

    # File paths (e.g., path/to/file.py)
    path_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_/]*\.py)\b'
    for match in re.finditer(path_pattern, issue_text):
        refs.append(match.group(1))

    # Class names in CamelCase (e.g., FileSystemStorage, TemporaryUploadedFile)
    class_pattern = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'
    for match in re.finditer(class_pattern, issue_text):
        refs.append(match.group(1))

    return list(set(refs))[:10]  # Limit to 10 most relevant


def is_test_path(path: Path) -> bool:
    """Check if a path is likely a test directory."""
    path_str = str(path).lower()
    return any(test_indicator in path_str for test_indicator in [
        '/test/', '/tests/', 'test_', '_test', '/testing/', '/fixtures/'
    ])


def infer_target_directories(repo_path: Path, issue_text: str) -> list[Path]:
    """Infer which directories in the repo are relevant to the issue.

    Uses multiple strategies:
    1. Look for Python module paths like 'django.core.files'
    2. Search for files containing class names (excluding tests)
    3. Search for method names mentioned in the issue
    """
    refs = extract_code_references(issue_text)
    dirs = []

    # Strategy 1: Convert dotted paths to directories
    for ref in refs:
        if '.' in ref and not ref.endswith('.py'):
            dir_path = repo_path / ref.replace('.', '/')
            if dir_path.is_dir():
                dirs.append(dir_path)
            else:
                parent = dir_path.parent
                if parent.is_dir() and parent != repo_path:
                    dirs.append(parent)

    # Strategy 2: Search for files containing class names (excluding tests)
    if not dirs:
        for ref in refs[:5]:
            if ref[0].isupper():  # Likely a class name
                try:
                    # Use grep with --exclude-dir to skip test directories
                    result = subprocess.run(
                        ["grep", "-rl", f"class {ref}", str(repo_path),
                         "--include=*.py",
                         "--exclude-dir=test", "--exclude-dir=tests",
                         "--exclude-dir=testing", "--exclude-dir=fixtures",
                         "-m", "5"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        for file_path in result.stdout.strip().split('\n'):
                            if file_path and not is_test_path(Path(file_path)):
                                parent = Path(file_path).parent
                                if parent != repo_path and parent not in dirs:
                                    dirs.append(parent)
                                    if len(dirs) >= 3:
                                        break
                except (subprocess.TimeoutExpired, Exception):
                    continue
            if len(dirs) >= 3:
                break

    # Strategy 3: Search for method names like "get_order_by" or "ordering_parts"
    if not dirs:
        # Look for method-like patterns: word followed by ( or word after def/self.
        method_pattern = r'\b(get_\w+|set_\w+|\w+_\w+)\s*\('
        methods = re.findall(method_pattern, issue_text)
        for method in methods[:3]:
            try:
                result = subprocess.run(
                    ["grep", "-rl", f"def {method}", str(repo_path),
                     "--include=*.py",
                     "--exclude-dir=test", "--exclude-dir=tests",
                     "-m", "3"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    for file_path in result.stdout.strip().split('\n'):
                        if file_path and not is_test_path(Path(file_path)):
                            parent = Path(file_path).parent
                            if parent != repo_path and parent not in dirs:
                                dirs.append(parent)
                                if len(dirs) >= 3:
                                    break
            except (subprocess.TimeoutExpired, Exception):
                continue
            if len(dirs) >= 3:
                break

    # Deduplicate while preserving order, prioritizing non-test paths
    seen = set()
    unique_dirs = []
    # First pass: non-test directories
    for d in dirs:
        if d not in seen and not is_test_path(d):
            seen.add(d)
            unique_dirs.append(d)
    # Second pass: test directories (if we have room)
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)

    return unique_dirs[:3]


def fetch_scribe_context(repo_path: Path, issue_text: str, timeout_s: int = 60) -> str:
    """Run scribe to get relevant code context for an issue.

    First tries to identify specific directories to analyze (fast).
    Falls back to query-hint on full repo with short timeout.

    Args:
        repo_path: Path to the repository.
        issue_text: The issue/problem statement.
        timeout_s: Timeout for scribe commands.

    Returns:
        String with scribe output, or empty string if scribe fails.
    """
    scribe_bin = get_scribe_path()
    if not scribe_bin:
        return ""

    output_file = Path(tempfile.mktemp(suffix='.txt', prefix='scribe_'))
    context_parts = []

    try:
        # Strategy 1: Target specific directories inferred from issue
        target_dirs = infer_target_directories(repo_path, issue_text)

        for target_dir in target_dirs:
            try:
                result = subprocess.run(
                    [
                        scribe_bin,
                        "--output-format", "text",
                        "-o", str(output_file),
                        "--token-target", "4000",  # Smaller target per directory
                        str(target_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,  # Short timeout per directory
                )

                if result.returncode == 0 and output_file.exists():
                    content = output_file.read_text()
                    if len(content) > 100:
                        context_parts.append(f"# Files from {target_dir.relative_to(repo_path)}:\n{content}")
                        output_file.unlink()

            except (subprocess.TimeoutExpired, Exception):
                continue

        # Strategy 2: If no targeted results, try query-hint with short timeout
        if not context_parts:
            refs = extract_code_references(issue_text)
            query_hint = " ".join(refs[:5]) if refs else issue_text[:100].replace('\n', ' ')

            try:
                result = subprocess.run(
                    [
                        scribe_bin,
                        "--query-hint", query_hint,
                        "--output-format", "text",
                        "-o", str(output_file),
                        "--token-target", "6000",
                        "--exclude-tests",
                        str(repo_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,  # Short timeout - if too slow, skip
                )

                if result.returncode == 0 and output_file.exists():
                    content = output_file.read_text()
                    if len(content) > 100:
                        context_parts.append(content)

            except (subprocess.TimeoutExpired, Exception):
                pass  # Too slow, proceed without scribe context

        if not context_parts:
            return ""

        # Combine all context, removing redundant headers
        combined = "\n\n".join(context_parts)

        # Strip the header/summary and just get the file contents
        if "---" in combined:
            parts = combined.split("---", 1)
            if len(parts) > 1:
                combined = "---" + parts[1]

        return combined[:16000]  # Limit total context size

    finally:
        # Clean up temp file
        if output_file.exists():
            output_file.unlink()


@dataclass
class ToolCall:
    """Record of a single tool call."""
    name: str
    input: dict
    output: str
    tokens_in_output: int


@dataclass
class TaskResult:
    """Result of running a single SWE-bench task."""
    task_id: str
    mode: str  # "scribe" or "standard"
    model: str
    timestamp: str

    # Outcome
    resolved: bool = False
    patch: str = ""
    explanation: str = ""

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Tool usage
    tool_calls: list = field(default_factory=list)
    num_tool_calls: int = 0
    scribe_calls: int = 0

    # Timing
    duration_s: float = 0

    # Status
    success: bool = True
    error: str = ""

    # Raw output for debugging
    raw_output: str = ""


def get_opencode_path() -> Optional[str]:
    """Get path to OpenCode binary."""
    # Check PATH first
    path = shutil.which("opencode")
    if path:
        return path

    # Check common install locations
    home = Path.home()
    candidates = [
        home / ".opencode" / "bin" / "opencode",
        home / ".local" / "bin" / "opencode",
        home / "go" / "bin" / "opencode",
        Path("/usr/local/bin/opencode"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def check_opencode_installed() -> bool:
    """Check if OpenCode is installed."""
    return get_opencode_path() is not None


def get_docker_image_name(task: dict) -> str:
    """Get the Docker image name for a SWE-bench task.

    SWE-bench images are named: sweb.eval.x86_64.{instance_id}:latest
    where instance_id is like 'astropy__astropy-12907'
    """
    instance_id = task.get("instance_id", "")
    return f"sweb.eval.x86_64.{instance_id}:latest"


def check_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0


def check_images_available(tasks: list[dict]) -> dict[str, bool]:
    """Check which Docker images are available locally.

    SWE-bench images must be built locally using build_images.py,
    they cannot be pulled from Docker Hub.

    Args:
        tasks: List of SWE-bench task dicts.

    Returns:
        Dict mapping image_name -> available bool.
    """
    results = {}
    missing = []

    for task in tasks:
        image_name = get_docker_image_name(task)
        if image_name not in results:
            available = check_image_exists(image_name)
            results[image_name] = available
            if not available:
                missing.append(task.get("instance_id", "unknown"))

    available_count = sum(1 for v in results.values() if v)
    print(f"Docker images: {available_count}/{len(results)} available")

    if missing:
        print(f"Missing images for: {', '.join(missing[:5])}" +
              (f" (and {len(missing)-5} more)" if len(missing) > 5 else ""))
        print("Build missing images with: python swebench/build_images.py --max-tasks N")

    return results


# Backwards compatibility alias
def prepull_images(tasks: list[dict], max_workers: int = 4) -> dict[str, bool]:
    """Check which images are available (images are built locally, not pulled)."""
    return check_images_available(tasks)


def parse_opencode_output(output: str) -> dict:
    """Parse OpenCode JSON output to extract metrics."""
    metrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": [],
        "scribe_calls": 0,
    }

    # OpenCode with --format json outputs newline-delimited JSON events
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            event_type = event.get("type", "")

            # Track token usage from step_finish events
            # Format: {"type":"step_finish","part":{"tokens":{"input":N,"output":N,...}}}
            if event_type == "step_finish":
                part = event.get("part", {})
                tokens = part.get("tokens", {})
                if tokens:
                    metrics["input_tokens"] += tokens.get("input", 0)
                    metrics["output_tokens"] += tokens.get("output", 0)

            # Track tool calls from tool_use events
            # Format: {"type":"tool_use","part":{"tool":"bash","state":{"input":{...}}}}
            if event_type == "tool_use":
                part = event.get("part", {})
                tool_name = part.get("tool", "")
                tool_input = part.get("state", {}).get("input", {})
                metrics["tool_calls"].append({
                    "name": tool_name,
                    "input": tool_input,
                })
                if tool_name == "scribe" or "scribe" in tool_name.lower():
                    metrics["scribe_calls"] += 1

        except json.JSONDecodeError:
            continue

    return metrics


class TaskRunner:
    """Runs SWE-bench tasks using OpenCode."""

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        scribe_binary: Optional[str] = None,
        use_docker: bool = True,
        task_timeout_s: int = 300,  # Timeout for OpenCode to solve the task
        setup_timeout_s: int = 120,  # Timeout for container/repo setup
    ):
        if not check_opencode_installed():
            raise RuntimeError(
                "OpenCode not installed. Install with: curl -fsSL https://opencode.ai/install | bash"
            )

        self.model = model
        self.scribe_binary = scribe_binary or shutil.which("scribe") or "scribe"
        self.use_docker = use_docker
        self.task_timeout_s = task_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.verbose = True  # Enable verbose output for debugging

        # Current task context
        self.repo_path: Optional[Path] = None
        self.task_id: Optional[str] = None
        self.container_id: Optional[str] = None

    def setup_task(self, task: dict) -> bool:
        """Set up the environment for a task."""
        self.task_id = task.get("instance_id", "unknown")

        if self.use_docker:
            return self._setup_docker(task)
        else:
            return self._setup_local(task)

    def _setup_docker(self, task: dict) -> bool:
        """Set up task in Docker container.

        Assumes image is already built (use build_images.py before batch runs).
        """
        image_name = get_docker_image_name(task)

        try:
            # Verify image exists (should already be built)
            if not check_image_exists(image_name):
                print(f"    Image not found: {image_name}")
                print(f"    Build with: python swebench/build_images.py")
                return False

            # Start container
            if self.verbose:
                print(f"    Starting container from {image_name}...")
            result = subprocess.run(
                ["docker", "run", "-d", "--rm", image_name, "sleep", "3600"],
                capture_output=True,
                text=True,
                timeout=self.setup_timeout_s,
            )

            if result.returncode != 0:
                if self.verbose:
                    print(f"    Container start failed: {result.stderr[:200]}")
                return False

            self.container_id = result.stdout.strip()
            if self.verbose:
                print(f"    Container started: {self.container_id[:12]}")

            # Copy repo out to temp directory for OpenCode to access
            # SWE-bench images have the repo in /testbed/
            self.repo_path = Path(tempfile.mkdtemp(prefix="swebench_"))
            if self.verbose:
                print(f"    Copying repo to {self.repo_path}...")
            cp_result = subprocess.run(
                ["docker", "cp", f"{self.container_id}:/testbed/.", str(self.repo_path)],
                capture_output=True,
                text=True,
                timeout=self.setup_timeout_s,
            )

            if cp_result.returncode != 0:
                if self.verbose:
                    print(f"    Copy failed: {cp_result.stderr[:200]}")
                return False

            if self.verbose:
                # List files to verify
                files = list(self.repo_path.iterdir())[:5]
                print(f"    Repo ready: {len(list(self.repo_path.iterdir()))} items ({', '.join(f.name for f in files)}...)")

            return True

        except Exception as e:
            print(f"    Docker setup failed: {e}")
            return False

    def _setup_local(self, task: dict) -> bool:
        """Set up task locally by cloning repo."""
        self.repo_path = Path(tempfile.mkdtemp(prefix="swebench_"))

        repo_url = f"https://github.com/{task.get('repo', '')}.git"
        base_commit = task.get("base_commit", "")

        try:
            subprocess.run(
                ["git", "clone", "--depth", "100", repo_url, str(self.repo_path)],
                capture_output=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=self.repo_path,
                capture_output=True,
                timeout=30,
            )
            return True
        except Exception as e:
            print(f"    Local setup failed: {e}")
            return False

    def cleanup_task(self) -> None:
        """Clean up task environment."""
        if self.container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass
            self.container_id = None

        if self.repo_path and self.repo_path.exists():
            shutil.rmtree(self.repo_path, ignore_errors=True)
            self.repo_path = None

    def run_task(self, task: dict, mode: str = "scribe") -> TaskResult:
        """Run a single SWE-bench task using OpenCode.

        Args:
            task: SWE-bench task dict.
            mode: "standard", "scribe-context", or "scribe-tool".
                  Legacy "scribe" maps to "scribe-context".

        Returns:
            TaskResult with all metrics.
        """
        # Handle legacy mode name
        if mode == "scribe":
            mode = "scribe-context"

        result = TaskResult(
            task_id=task.get("instance_id", "unknown"),
            mode=mode,
            model=self.model,
            timestamp=datetime.now().isoformat(),
        )

        issue = task.get("problem_statement", "")

        start_time = time.time()

        try:
            # Set up environment
            if not self.setup_task(task):
                result.success = False
                result.error = "Failed to set up task environment"
                return result

            # Build prompt based on mode
            if mode == "scribe-context":
                # Pre-fetch scribe context for relevant code
                if self.verbose:
                    print("    Fetching scribe context...")
                scribe_context = fetch_scribe_context(self.repo_path, issue)

                if scribe_context:
                    if self.verbose:
                        print(f"    Got {len(scribe_context)} chars of context")
                    prompt = f"""Fix the following issue in this repository.

Here is the COMPLETE relevant code context you need:

{scribe_context}

ISSUE:
{issue}

IMPORTANT: The context above contains all the relevant code. DO NOT re-explore the codebase with grep/glob/read to find additional files. Go directly to implementing the fix using only the code shown above. After fixing, run tests to verify."""
                else:
                    if self.verbose:
                        print("    No scribe context available, using standard prompt")
                    prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works."""

            elif mode == "scribe-tool":
                # Pre-identify directories to give agent concrete targets
                target_dirs = infer_target_directories(self.repo_path, issue)
                if target_dirs:
                    dir_suggestions = "\n".join(f"  - {d.relative_to(self.repo_path)}" for d in target_dirs[:3])
                    dir_example = str(target_dirs[0].relative_to(self.repo_path))
                else:
                    # Fallback: suggest looking for main source directory
                    dir_suggestions = "  - Use grep to find the relevant directory first"
                    dir_example = "src/"

                prompt = f"""Fix the following issue in this repository.

You have `scribe` for getting code context. Here's how to use it:

SUGGESTED DIRECTORIES (based on issue analysis):
{dir_suggestions}

SCRIBE COMMAND (copy and modify):
```
scribe --output-format text -o /dev/stdout --token-target 8000 {dir_example}
```

RULES:
- ONLY run scribe on a specific subdirectory (like above)
- NEVER run scribe on "." - it will timeout on large repos
- Use --token-target 8000 to limit output size

ISSUE:
{issue}

Run scribe on one of the suggested directories, then fix the issue and run tests."""

            else:  # standard mode
                prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works."""

            # Run OpenCode
            opencode_bin = get_opencode_path()
            cmd = [
                opencode_bin, "run",
                "--model", self.model,
                "--format", "json",
                prompt,
            ]

            if self.verbose:
                print(f"    Running OpenCode ({self.model})...")
                print(f"    Timeout: {self.task_timeout_s}s")
                print(f"    Command: {' '.join(cmd[:4])}...")

            proc_result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.task_timeout_s,
            )

            result.raw_output = proc_result.stdout[:10000]
            result.duration_s = time.time() - start_time

            if self.verbose:
                print(f"    Completed in {result.duration_s:.1f}s")
                if proc_result.stderr:
                    print(f"    stderr: {proc_result.stderr[:200]}")

            # Parse output for metrics
            if proc_result.returncode == 0:
                metrics = parse_opencode_output(proc_result.stdout)
                result.input_tokens = metrics["input_tokens"]
                result.output_tokens = metrics["output_tokens"]
                result.total_tokens = result.input_tokens + result.output_tokens
                result.num_tool_calls = len(metrics["tool_calls"])
                result.scribe_calls = metrics["scribe_calls"]
                result.tool_calls = [
                    ToolCall(
                        name=tc["name"],
                        input=tc["input"],
                        output="",
                        tokens_in_output=0,
                    )
                    for tc in metrics["tool_calls"]
                ]
                result.success = True
            else:
                result.success = False
                result.error = proc_result.stderr[:500]

            # Get patch (git diff)
            diff_result = subprocess.run(
                ["git", "diff"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result.patch = diff_result.stdout[:5000]
            result.resolved = len(result.patch.strip()) > 0

            # If using Docker, copy changes back and run tests
            if self.use_docker and self.container_id and result.patch:
                # Copy modified files back to container
                subprocess.run(
                    ["docker", "cp", f"{self.repo_path}/.", f"{self.container_id}:/testbed/"],
                    capture_output=True,
                    timeout=60,
                )

                # TODO: Run SWE-bench test harness to verify fix
                # For now, just check if there's a patch

        except subprocess.TimeoutExpired as e:
            result.success = False
            result.error = f"Timeout after {self.task_timeout_s}s"
            result.duration_s = time.time() - start_time
            if self.verbose:
                print(f"    Timeout expired after {result.duration_s:.1f}s")
                if hasattr(e, 'stdout') and e.stdout:
                    print(f"    partial stdout: {e.stdout[:300] if isinstance(e.stdout, str) else e.stdout.decode()[:300]}")

        except Exception as e:
            result.success = False
            result.error = str(e)[:500]
            result.duration_s = time.time() - start_time

        finally:
            self.cleanup_task()

        return result


def run_task_batch(
    tasks: list[dict],
    mode: str = "both",
    model: str = "anthropic/claude-sonnet-4-20250514",
    max_tasks: Optional[int] = None,
    use_docker: bool = True,
    prepull_workers: int = 4,
    task_timeout_s: int = 600,
) -> list[TaskResult]:
    """Run a batch of SWE-bench tasks.

    Args:
        tasks: List of SWE-bench task dicts.
        mode: "standard", "scribe-context", "scribe-tool", "both", or "all".
              Legacy "scribe" = "scribe-context".
              "both" = standard + scribe-context.
              "all" = standard + scribe-context + scribe-tool.
        model: Model to use (format: provider/model for OpenCode).
        max_tasks: Maximum number of tasks to run.
        use_docker: Whether to use Docker for isolation.
        prepull_workers: Number of parallel workers for image pre-pulling.
        task_timeout_s: Timeout per task in seconds.

    Returns:
        List of TaskResult objects.
    """
    # Handle legacy mode name
    if mode == "scribe":
        mode = "scribe-context"

    if max_tasks:
        tasks = tasks[:max_tasks]

    # Pre-pull Docker images before running any tasks
    if use_docker:
        image_status = prepull_images(tasks, max_workers=prepull_workers)
        # Filter out tasks whose images failed to pull
        failed_images = [img for img, ok in image_status.items() if not ok]
        if failed_images:
            print(f"Warning: {len(failed_images)} images failed to pull")
            # Skip tasks with failed images
            tasks = [
                t for t in tasks
                if image_status.get(get_docker_image_name(t), False)
            ]
            if not tasks:
                print("Error: No tasks with available images")
                return []

    runner = TaskRunner(model=model, use_docker=use_docker, task_timeout_s=task_timeout_s)
    results = []

    # Determine which modes to run
    run_standard = mode in ("both", "all", "standard")
    run_scribe_context = mode in ("both", "all", "scribe-context")
    run_scribe_tool = mode in ("all", "scribe-tool")

    for i, task in enumerate(tasks):
        task_id = task.get("instance_id", f"task_{i}")
        print(f"\n[{i+1}/{len(tasks)}] {task_id}")

        if run_standard:
            print("  Running standard mode...")
            result = runner.run_task(task, mode="standard")
            status = "OK" if result.success else f"FAILED: {result.error[:30]}"
            print(f"    {status} - Tokens: {result.total_tokens:,}, Calls: {result.num_tool_calls}")
            results.append(result)

        if run_scribe_context:
            print("  Running scribe-context mode...")
            result = runner.run_task(task, mode="scribe-context")
            status = "OK" if result.success else f"FAILED: {result.error[:30]}"
            print(f"    {status} - Tokens: {result.total_tokens:,}, Calls: {result.num_tool_calls}")
            results.append(result)

        if run_scribe_tool:
            print("  Running scribe-tool mode...")
            result = runner.run_task(task, mode="scribe-tool")
            status = "OK" if result.success else f"FAILED: {result.error[:30]}"
            print(f"    {status} - Tokens: {result.total_tokens:,}, Calls: {result.num_tool_calls}")
            results.append(result)

    return results

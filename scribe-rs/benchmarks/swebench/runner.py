"""SWE-bench task runner using Claude Code.

Executes SWE-bench tasks with Claude Code agent and captures results.
Uses Claude Code's `-p` (print) mode for non-interactive agent behavior.
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


# Load NLTK stopwords + GitHub/programming-specific additions
def _load_stopwords() -> set:
    """Load NLTK English stopwords plus programming-specific additions."""
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        words = set(stopwords.words('english'))
    except Exception:
        # Fallback to minimal set if NLTK unavailable
        words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'is', 'are', 'was', 'were',
                 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
                 'this', 'that', 'these', 'those', 'it', 'its', 'of', 'in', 'to', 'for',
                 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further',
                 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'now'}

    # Add GitHub/programming-specific stopwords
    programming_stopwords = {
        # Issue boilerplate
        'issue', 'bug', 'fix', 'error', 'problem', 'expected', 'actual', 'behavior',
        'behaviour', 'result', 'output', 'example', 'code', 'file', 'line', 'version',
        # Common verbs in issues
        'using', 'used', 'use', 'work', 'works', 'working', 'worked',
        'trying', 'try', 'tried', 'want', 'wanted', 'getting', 'get', 'got',
        'make', 'made', 'makes', 'seem', 'seems', 'seemed', 'look', 'looks', 'looking',
        'think', 'thought', 'know', 'known', 'find', 'found', 'show', 'shown', 'shows',
        'create', 'created', 'run', 'running', 'ran', 'call', 'called',
        # GitHub phrases
        'steps', 'reproduce', 'reproduction', 'minimal', 'repo', 'repository',
        'please', 'thanks', 'thank', 'hi', 'hello', 'following', 'see', 'also',
        # Numbers
        'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next', 'new', 'old',
    }
    words.update(programming_stopwords)
    return words

STOPWORDS = _load_stopwords()


def extract_query_keywords(text: str, max_keywords: int = 10) -> str:
    """Extract meaningful keywords from text by filtering stopwords.

    Args:
        text: Input text (issue description, etc.)
        max_keywords: Maximum number of keywords to return

    Returns:
        Space-separated keywords suitable for scribe --query-hint
    """
    # Tokenize: extract words, keeping underscores for identifiers
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())

    # Filter stopwords and very short words
    keywords = []
    seen = set()
    for word in words:
        if word not in STOPWORDS and word not in seen and len(word) > 2:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= max_keywords:
                break

    return " ".join(keywords) if keywords else text[:100].replace('\n', ' ')


def cleanup_stale_workdirs(max_age_hours: int = 24) -> None:
    """Clean up stale swebench work directories and Go build cache.

    Called at the start of a benchmark run to free disk space.
    """
    tmp_dir = Path(tempfile.gettempdir())
    now = time.time()
    max_age_s = max_age_hours * 3600
    cleaned = 0

    # Clean swebench_* directories
    for d in tmp_dir.glob("swebench_*"):
        try:
            if d.is_dir():
                mtime = d.stat().st_mtime
                if now - mtime > max_age_s:
                    shutil.rmtree(d, ignore_errors=True)
                    cleaned += 1
        except Exception:
            pass

    # Clean go-build* directories (no age check, always clean)
    for d in tmp_dir.glob("go-build*"):
        try:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                cleaned += 1
        except Exception:
            pass

    if cleaned > 0:
        print(f"Cleaned up {cleaned} stale work directories")


def setup_scribe_hooks(work_dir: Path) -> None:
    """Set up Claude Code hooks to enforce scribe usage.

    Creates hooks that:
    1. BLOCK Read/Grep on code files - force scribe usage
    2. ALLOW all other tools (so we don't need --allowedTools which bypasses hooks)
    3. Block bad scribe usage patterns

    IMPORTANT: --allowedTools bypasses hooks entirely, so we must handle all
    permission decisions in this hook instead.
    """
    hooks_dir = work_dir / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Write the scribe enforcement hook
    hook_script = hooks_dir / "scribe_enforce.py"
    hook_script.write_text('''#!/usr/bin/env python3
"""Hook to enforce scribe usage and handle permissions.

IMPORTANT: This hook returns "allow" for all tools EXCEPT:
- Read on code files (blocked)
- Grep (blocked)
- Bad scribe usage patterns (blocked with retry)

This allows us to avoid --allowedTools which bypasses hooks entirely.
"""
import json, sys, os, re, hashlib

CODE_EXTS = {".py",".pyi",".js",".jsx",".ts",".tsx",".mjs",".cjs",".go",".rs",
    ".java",".kt",".c",".h",".cpp",".hpp",".cc",".rb",".php",".swift",".scala",
    ".cs",".lua",".ex",".exs",".hs",".ml",".clj",".vue",".svelte",".sol"}

def get_state(sid):
    try:
        with open(f"/tmp/claude_hook_{sid}.json") as f: return json.load(f)
    except: return {"seen_commands": []}

def save_state(sid, s):
    with open(f"/tmp/claude_hook_{sid}.json", "w") as f: json.dump(s, f)

def hash_command(tool, inp):
    return hashlib.md5(json.dumps({"tool": tool, "input": inp}, sort_keys=True).encode()).hexdigest()

def deny(msg):
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse",
           "permissionDecision": "deny", "permissionDecisionReason": msg}}
    print(json.dumps(out))
    sys.exit(0)

def allow():
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse",
           "permissionDecision": "allow"}}
    print(json.dumps(out))
    sys.exit(0)

def main():
    try: data = json.load(sys.stdin)
    except: sys.exit(0)

    sid = data.get("session_id", "x")
    tool = data.get("tool_name", "")
    event = data.get("hook_event_name", "")
    inp = data.get("tool_input", {})

    if event != "PreToolUse": sys.exit(0)

    state = get_state(sid)

    # BLOCK: Read on code files
    if tool == "Read":
        fp = inp.get("file_path", "")
        ext = os.path.splitext(fp)[1].lower()
        if ext in CODE_EXTS:
            deny(f"""BLOCKED: Cannot Read code files directly. Use scribe.

You tried to read: {fp}

USE SCRIBE to get the code WITH its dependencies:

  scribe --covering-set "{fp}:FUNCTION_NAME" --stdout

Replace FUNCTION_NAME with the function/class you need.
This returns the entity AND all types/functions it depends on.

For non-code files (configs, docs), Read is allowed.""")
        # Non-code files: allow
        allow()

    # Grep: allow for file discovery, block for content reading on code files
    if tool == "Grep":
        output_mode = inp.get("output_mode", "files_with_matches")
        grep_path = inp.get("path", "")

        # Allow file discovery (finding which files match)
        if output_mode == "files_with_matches":
            allow()

        # For content mode, check if targeting code files
        if output_mode == "content":
            ext = os.path.splitext(grep_path)[1].lower() if grep_path else ""
            if ext in CODE_EXTS or not grep_path:
                pattern = inp.get("pattern", "")
                deny(f"""BLOCKED: Use scribe instead of Grep content on code files.

You searched for: {pattern}

USE SCRIBE to get complete context:

  scribe --covering-set "path/to/file:entity_name" --stdout

This returns the entity AND all its dependencies in one call.

Tip: Grep with output_mode="files_with_matches" is allowed for discovery.""")

        # Allow grep on non-code paths
        allow()

    # Check Bash for bad scribe usage patterns
    if tool == "Bash":
        cmd = inp.get("command", "")
        if "scribe" in cmd.lower():
            cmd_hash = hash_command(tool, inp)
            # Allow retry for scribe commands
            if cmd_hash in state.get("seen_commands", []):
                allow()

            # Block: scribe . (root scan without options)
            if re.search(r"scribe\\s+\\.", cmd) and not re.search(r"--(covering-set|token-target|output)", cmd):
                state.setdefault("seen_commands", []).append(cmd_hash)
                save_state(sid, state)
                deny(f"""BLOCKED: Don't run scribe on root without constraints.

You ran: {cmd}

USE TARGETED SCRIBE:

  scribe --covering-set "path/to/file:function" --stdout
  scribe --token-target 8000 src/specific/dir --stdout

Resubmit if you really need this.""")

            # Block: piping scribe output
            if re.search(r"scribe.*\\|\\s*(head|tail|grep|awk|sed|cut)", cmd, re.I):
                state.setdefault("seen_commands", []).append(cmd_hash)
                save_state(sid, state)
                deny(f"""BLOCKED: Don't pipe scribe output.

You ran: {cmd}

Scribe returns exactly what you need. Use --token-target to limit size.

Resubmit if you really need this.""")

            # Block: redirecting scribe to /dev/null
            if re.search(r"scribe.*>\\s*/dev/null", cmd, re.I):
                state.setdefault("seen_commands", []).append(cmd_hash)
                save_state(sid, state)
                deny("""BLOCKED: Don't discard scribe output.

Resubmit if you really need this.""")

    # ALLOW all other tools
    allow()

if __name__ == "__main__": main()
''')
    hook_script.chmod(0o755)

    # Write settings.json - hook applies to ALL tools (.*) since we handle permissions
    # IMPORTANT: We use ".*" matcher because --allowedTools bypasses hooks, so we must
    # handle all permission decisions in the hook instead.
    # Must use absolute path for hook script
    settings = work_dir / ".claude" / "settings.json"
    abs_hook_path = hook_script.resolve()
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": ".*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {abs_hook_path}"
                        }
                    ]
                }
            ]
        }
    }, indent=2))


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


def get_scribe_git_info() -> dict:
    """Get git information about the scribe installation.

    Returns dict with:
        - commit: short git commit hash (e.g., "abc1234")
        - commit_full: full git commit hash
        - branch: current branch name
        - dirty: True if there are uncommitted changes
        - scribe_dir: path to scribe source directory (if found)
    """
    info = {
        "commit": None,
        "commit_full": None,
        "branch": None,
        "dirty": None,
        "scribe_dir": None,
    }

    # Try to find scribe source directory
    # Common locations: relative to this benchmark dir, or in common dev locations
    candidates = [
        Path(__file__).parent.parent.parent,  # scribe-rs directory (benchmarks/swebench -> scribe-rs)
        Path.home() / "Projects" / "scribe" / "scribe-rs",
        Path.home() / "scribe" / "scribe-rs",
        Path.home() / "src" / "scribe-rs",
    ]

    scribe_dir = None
    for candidate in candidates:
        if (candidate / "Cargo.toml").exists():
            scribe_dir = candidate
            break

    if not scribe_dir:
        return info

    info["scribe_dir"] = str(scribe_dir)

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=scribe_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit_full"] = result.stdout.strip()
            info["commit"] = info["commit_full"][:7]

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=scribe_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=scribe_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, Exception):
        pass

    return info


def detect_repo_language(repo_path: Path) -> str:
    """Detect the primary language of a repository.

    Args:
        repo_path: Path to the repository root.

    Returns:
        Language name: "python", "typescript", "javascript", "rust", "go", "java", etc.
    """
    repo_path = Path(repo_path)

    # Count files by extension
    ext_counts = {}
    try:
        for ext in [".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".java", ".vue"]:
            count = len(list(repo_path.rglob(f"*{ext}")))
            if count > 0:
                ext_counts[ext] = count
    except Exception:
        pass

    if not ext_counts:
        return "unknown"

    # Find dominant extension
    dominant_ext = max(ext_counts, key=ext_counts.get)

    ext_to_lang = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".vue": "typescript",  # Vue uses TypeScript/JS
    }

    return ext_to_lang.get(dominant_ext, "unknown")


def extract_code_references(issue_text: str, language: str = "python") -> list[str]:
    """Extract file paths and code references from issue text.

    Looks for patterns like:
    - module.path.ClassName
    - path/to/file.py
    - function_name()

    Args:
        issue_text: The issue/problem statement text.
        language: Primary language of the repo (affects file extension patterns).
    """
    refs = []

    # Language-specific file extension
    ext_map = {
        "python": "py",
        "typescript": "ts",
        "javascript": "js",
        "rust": "rs",
        "go": "go",
        "java": "java",
    }
    ext = ext_map.get(language, "py")

    # Python dotted paths (e.g., django.core.files.uploadhandler)
    dotted_pattern = r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){2,})\b'
    for match in re.finditer(dotted_pattern, issue_text, re.IGNORECASE):
        refs.append(match.group(1))

    # File paths with language-specific extension
    path_pattern = rf'\b([a-zA-Z_][a-zA-Z0-9_/]*\.{ext})\b'
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


def detect_repo_language(repo_path: Path) -> str:
    """Detect primary language of a repository."""
    # Count files by extension
    counts = {}
    for ext in ['.py', '.go', '.rs', '.js', '.ts', '.java', '.rb', '.php']:
        try:
            result = subprocess.run(
                ["find", str(repo_path), "-name", f"*{ext}", "-type", "f"],
                capture_output=True, text=True, timeout=5
            )
            counts[ext] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except:
            counts[ext] = 0

    if not counts:
        return 'unknown'

    max_ext = max(counts, key=counts.get)
    return {
        '.py': 'python', '.go': 'go', '.rs': 'rust', '.js': 'javascript',
        '.ts': 'typescript', '.java': 'java', '.rb': 'ruby', '.php': 'php'
    }.get(max_ext, 'unknown')


def infer_target_directories(repo_path: Path, issue_text: str) -> list[Path]:
    """Infer which directories in the repo are relevant to the issue.

    Supports multiple languages with appropriate search patterns.
    """
    refs = extract_code_references(issue_text)
    dirs = []
    lang = detect_repo_language(repo_path)

    # Language-specific settings
    lang_config = {
        'python': {'ext': '*.py', 'class': 'class {ref}', 'func': 'def {ref}'},
        'go': {'ext': '*.go', 'class': 'type {ref} struct', 'func': 'func {ref}|func \\([^)]+\\) {ref}'},
        'rust': {'ext': '*.rs', 'class': 'struct {ref}|enum {ref}', 'func': 'fn {ref}'},
        'javascript': {'ext': '*.js', 'class': 'class {ref}', 'func': 'function {ref}|const {ref}'},
        'typescript': {'ext': '*.ts', 'class': 'class {ref}|interface {ref}', 'func': 'function {ref}|const {ref}'},
        'java': {'ext': '*.java', 'class': 'class {ref}', 'func': 'void {ref}|public {ref}'},
        'ruby': {'ext': '*.rb', 'class': 'class {ref}', 'func': 'def {ref}'},
    }
    config = lang_config.get(lang, lang_config['python'])

    # Strategy 1: Convert dotted/slashed paths to directories
    for ref in refs:
        if '.' in ref or '/' in ref:
            dir_path = repo_path / ref.replace('.', '/').replace('::', '/')
            if dir_path.is_dir():
                dirs.append(dir_path)
            else:
                parent = dir_path.parent
                if parent.is_dir() and parent != repo_path:
                    dirs.append(parent)

    # Strategy 2: Search for type/class definitions
    if not dirs:
        for ref in refs[:5]:
            if ref[0].isupper():  # Likely a type name
                pattern = config['class'].format(ref=ref)
                try:
                    result = subprocess.run(
                        ["grep", "-rlE", pattern, str(repo_path),
                         "--include", config['ext'],
                         "--exclude-dir=test", "--exclude-dir=tests",
                         "--exclude-dir=testing", "--exclude-dir=vendor",
                         "--exclude-dir=node_modules", "-m", "5"],
                        capture_output=True, text=True, timeout=10,
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

    # Strategy 3: Search for function definitions
    if not dirs:
        method_pattern = r'\b(get_?\w+|set_?\w+|\w+_\w+|[A-Z][a-z]+[A-Z]\w+)\s*\('
        methods = re.findall(method_pattern, issue_text)
        for method in methods[:3]:
            pattern = config['func'].format(ref=method)
            try:
                result = subprocess.run(
                    ["grep", "-rlE", pattern, str(repo_path),
                     "--include", config['ext'],
                     "--exclude-dir=test", "--exclude-dir=tests",
                     "--exclude-dir=vendor", "--exclude-dir=node_modules",
                     "-m", "3"],
                    capture_output=True, text=True, timeout=10,
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

    # Strategy 4: For Go, also search in common directories
    if not dirs and lang == 'go':
        for common_dir in ['internal', 'pkg', 'cmd', 'lib', 'src']:
            d = repo_path / common_dir
            if d.is_dir():
                dirs.append(d)
                if len(dirs) >= 2:
                    break

    # Deduplicate while preserving order, prioritizing non-test paths
    seen = set()
    unique_dirs = []
    for d in dirs:
        if d not in seen and not is_test_path(d):
            seen.add(d)
            unique_dirs.append(d)
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)

    return unique_dirs[:3]


def find_covering_set_targets(repo_path: Path, issue_text: str) -> list[str]:
    """Find file:entity pairs from issue text for --covering-set.

    Returns list of "path/to/file.ext:EntityName" strings.
    """
    targets = []
    lang = detect_repo_language(repo_path)

    # Language-specific settings
    lang_config = {
        'python': {'ext': '*.py', 'patterns': ['class {ref}', 'def {ref}']},
        'go': {'ext': '*.go', 'patterns': ['type {ref} struct', 'func {ref}', r'func \([^)]+\) {ref}']},
        'rust': {'ext': '*.rs', 'patterns': ['struct {ref}', 'enum {ref}', 'fn {ref}']},
        'javascript': {'ext': '*.js', 'patterns': ['class {ref}', 'function {ref}', r'const {ref}\s*=']},
        'typescript': {'ext': '*.ts', 'patterns': ['class {ref}', 'interface {ref}', 'function {ref}', r'const {ref}\s*=']},
        'java': {'ext': '*.java', 'patterns': ['class {ref}', 'interface {ref}']},
    }
    config = lang_config.get(lang, lang_config['python'])

    # Extract potential entity names from issue
    refs = extract_code_references(issue_text, lang)

    # Also look for function/method names in parentheses
    method_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    for match in re.finditer(method_pattern, issue_text):
        name = match.group(1)
        if len(name) > 2 and name not in ['if', 'for', 'while', 'with', 'return', 'print']:
            refs.append(name)

    refs = list(set(refs))[:10]

    # Search for each reference in the codebase
    for ref in refs:
        if len(ref) < 3:
            continue

        for pattern_template in config['patterns']:
            pattern = pattern_template.format(ref=ref)
            try:
                result = subprocess.run(
                    ["grep", "-rlE", pattern, str(repo_path),
                     "--include", config['ext'],
                     "--exclude-dir=test", "--exclude-dir=tests",
                     "--exclude-dir=testing", "--exclude-dir=vendor",
                     "--exclude-dir=node_modules", "--exclude-dir=__pycache__",
                     "-m", "3"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for file_path in result.stdout.strip().split('\n'):
                        if file_path and not is_test_path(Path(file_path)):
                            # Convert absolute path to relative
                            try:
                                rel_path = Path(file_path).relative_to(repo_path)
                                target = f"{rel_path}:{ref}"
                                if target not in targets:
                                    targets.append(target)
                                    if len(targets) >= 5:
                                        return targets
                            except ValueError:
                                pass
            except (subprocess.TimeoutExpired, Exception):
                continue

    return targets


def fetch_scribe_context(repo_path: Path, issue_text: str, timeout_s: int = 60, context_tokens: int = 4000) -> str:
    """Run scribe to get relevant code context for an issue.

    Strategies (in order):
    1. Use --covering-set if specific file:entity pairs can be identified (best for dependencies)
    2. Target specific directories inferred from issue
    3. Fall back to query-hint on full repo

    Args:
        repo_path: Path to the repository.
        issue_text: The issue/problem statement.
        timeout_s: Timeout for scribe commands.
        context_tokens: Target token budget for context (default: 4000).

    Returns:
        String with scribe output, or empty string if scribe fails.
    """
    scribe_bin = get_scribe_path()
    if not scribe_bin:
        return ""

    context_parts = []

    try:
        # Strategy 1: Try covering-set if we can identify specific entities
        # This uses the dependency graph and benefits from TypeScript resolution
        covering_targets = find_covering_set_targets(repo_path, issue_text)

        for target in covering_targets[:3]:  # Limit to 3 covering-set calls
            try:
                result = subprocess.run(
                    [
                        scribe_bin,
                        "--covering-set", target,
                        "--token-target", str(context_tokens // 3),
                        "--stdout",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=45,  # covering-set can take longer
                    cwd=repo_path,
                )

                if result.returncode == 0 and result.stdout:
                    content = result.stdout.strip()
                    if len(content) > 100:
                        context_parts.append(f"# Covering set for {target}:\n{content}")

            except (subprocess.TimeoutExpired, Exception):
                continue

        # Strategy 2: Use query-hint for semantic search (primary fallback)
        # This is more effective than directory scanning as it uses FTS to find relevant files
        if not context_parts:
            refs = extract_code_references(issue_text)
            if refs:
                query_hint = " ".join(refs[:5])
            else:
                # Extract keywords by aggressive stopword filtering
                query_hint = extract_query_keywords(issue_text)
            output_file = Path(tempfile.mktemp(suffix='.txt', prefix='scribe_'))

            try:
                # Determine target path - prefer src/ or lib/ if they exist
                # This avoids noise from demo/docs/examples directories
                target_path = repo_path
                for src_dir in ['src', 'lib', 'pkg', 'packages']:
                    candidate = repo_path / src_dir
                    if candidate.is_dir():
                        target_path = candidate
                        break

                cmd = [
                    scribe_bin,
                    "--query-hint", query_hint,
                    "--output-format", "text",
                    "-o", str(output_file),
                    "--token-target", str(context_tokens),  # Use full budget for query-hint
                    "--exclude-tests",
                    str(target_path),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0 and output_file.exists():
                    content = output_file.read_text()
                    if len(content) > 100:
                        context_parts.append(content)

            except (subprocess.TimeoutExpired, Exception):
                pass
            finally:
                if output_file.exists():
                    output_file.unlink()

        # Strategy 3: Fall back to directory scanning if query-hint didn't work
        if not context_parts:
            target_dirs = infer_target_directories(repo_path, issue_text)
            output_file = Path(tempfile.mktemp(suffix='.txt', prefix='scribe_'))

            try:
                for target_dir in target_dirs:
                    try:
                        result = subprocess.run(
                            [
                                scribe_bin,
                                "--output-format", "text",
                                "-o", str(output_file),
                                "--token-target", str(context_tokens // 4),
                                str(target_dir),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )

                        if result.returncode == 0 and output_file.exists():
                            content = output_file.read_text()
                            if len(content) > 100:
                                context_parts.append(f"# Files from {target_dir.relative_to(repo_path)}:\n{content}")
                                output_file.unlink()

                    except (subprocess.TimeoutExpired, Exception):
                        continue
            finally:
                if output_file.exists():
                    output_file.unlink()

        if not context_parts:
            return ""

        # Combine all context
        combined = "\n\n".join(context_parts)

        # Strip the header/summary and just get the file contents
        if "---" in combined:
            parts = combined.split("---", 1)
            if len(parts) > 1:
                combined = "---" + parts[1]

        # Limit total context size (~4 chars per token)
        max_chars = context_tokens * 4
        return combined[:max_chars]

    except Exception:
        return ""


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
    model: str  # Model argument passed (e.g., "sonnet", "opus")
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

    # Run tracking
    run_number: int = 1

    # Status
    success: bool = True
    error: str = ""

    # Metrics reliability - False if Claude errored during execution
    metrics_reliable: bool = True

    # Raw output for debugging
    raw_output: str = ""

    # Model tracking - actual model ID returned by Claude (e.g., "claude-sonnet-4-5-20250929")
    model_resolved: str = ""


def get_claude_path() -> Optional[str]:
    """Get path to Claude Code binary."""
    # Check PATH first
    path = shutil.which("claude")
    if path:
        return path

    # Check common install locations
    home = Path.home()
    candidates = [
        home / ".claude" / "bin" / "claude",
        home / ".local" / "bin" / "claude",
        Path("/usr/local/bin/claude"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def check_claude_installed() -> bool:
    """Check if Claude Code is installed."""
    return get_claude_path() is not None


# Backwards compatibility aliases
def get_opencode_path() -> Optional[str]:
    """Legacy alias for get_claude_path."""
    return get_claude_path()


def check_opencode_installed() -> bool:
    """Legacy alias for check_claude_installed."""
    return check_claude_installed()


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


def parse_claude_output(output: str) -> dict:
    """Parse Claude Code JSON output to extract metrics.

    Claude Code with --output-format json outputs a single JSON result object.
    """
    metrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": [],
        "scribe_calls": 0,
        "total_cost_usd": 0,
        "num_turns": 0,
        "had_error": False,  # True if Claude reported error_during_execution
        "error_types": [],   # List of error strings from Claude
        "model_resolved": "",  # Actual model ID used (e.g., "claude-sonnet-4-5-20250929")
    }

    try:
        # Parse the single JSON result
        result = json.loads(output.strip())

        # Check for error_during_execution - tokens will be 0 but patch may exist
        subtype = result.get("subtype", "")
        if subtype == "error_during_execution":
            metrics["had_error"] = True
            metrics["error_types"] = result.get("errors", [])

        # Extract usage metrics from main usage field
        usage = result.get("usage", {})
        # Sum all input tokens (regular + cache read + cache creation)
        metrics["input_tokens"] = (
            usage.get("input_tokens", 0) +
            usage.get("cache_read_input_tokens", 0) +
            usage.get("cache_creation_input_tokens", 0)
        )
        metrics["output_tokens"] = usage.get("output_tokens", 0)

        # If main usage is empty, try to get from modelUsage
        if metrics["input_tokens"] == 0 and "modelUsage" in result:
            for model_name, model_usage in result.get("modelUsage", {}).items():
                # Capture the first (usually only) model name as the resolved model
                if not metrics["model_resolved"] and model_name:
                    metrics["model_resolved"] = model_name
                metrics["input_tokens"] += (
                    model_usage.get("inputTokens", 0) +
                    model_usage.get("cacheReadInputTokens", 0) +
                    model_usage.get("cacheCreationInputTokens", 0)
                )
                metrics["output_tokens"] += model_usage.get("outputTokens", 0)
                metrics["total_cost_usd"] += model_usage.get("costUSD", 0)

        # Also check for model field directly in result (some Claude versions include this)
        if not metrics["model_resolved"]:
            metrics["model_resolved"] = result.get("model", "")

        metrics["total_cost_usd"] = result.get("total_cost_usd", metrics["total_cost_usd"])
        metrics["duration_ms"] = result.get("duration_ms", 0)
        metrics["num_turns"] = result.get("num_turns", 0)

        # Count scribe calls by scanning the raw result for bash commands containing "scribe"
        # The result text may contain tool call logs
        result_text = result.get("result", "")
        if isinstance(result_text, str):
            # Count occurrences of scribe commands in the output
            scribe_patterns = [
                r'scribe\s+--covering-set',
                r'scribe\s+--token-target',
                r'scribe\s+--query-hint',
                r'scribe\s+\.',  # scribe .
                r'scribe\s+src/',  # scribe src/...
            ]
            for pattern in scribe_patterns:
                metrics["scribe_calls"] += len(re.findall(pattern, result_text, re.IGNORECASE))

    except json.JSONDecodeError:
        pass

    return metrics


# Backwards compatibility alias
def parse_opencode_output(output: str) -> dict:
    """Legacy alias for parse_claude_output."""
    return parse_claude_output(output)


class TaskRunner:
    """Runs SWE-bench tasks using Claude Code."""

    def __init__(
        self,
        model: str = "sonnet",
        scribe_binary: Optional[str] = None,
        use_docker: bool = True,
        task_timeout_s: int = 300,  # Timeout for Claude Code to solve the task
        setup_timeout_s: int = 120,  # Timeout for container/repo setup
        context_tokens: int = 4000,  # Token budget for scribe-context mode
    ):
        if not check_claude_installed():
            raise RuntimeError(
                "Claude Code not installed. Install with: npm install -g @anthropic-ai/claude-code"
            )

        self.model = model
        self.scribe_binary = scribe_binary or shutil.which("scribe") or "scribe"
        self.use_docker = use_docker
        self.task_timeout_s = task_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.context_tokens = context_tokens
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

        # Clean up Go build cache that may have been created
        go_cache = Path(tempfile.gettempdir())
        for d in go_cache.glob("go-build*"):
            try:
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

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
                scribe_context = fetch_scribe_context(self.repo_path, issue, context_tokens=self.context_tokens)

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
                # Set up hooks to enforce scribe usage
                if self.repo_path:
                    setup_scribe_hooks(self.repo_path)
                    if self.verbose:
                        print("    Set up scribe enforcement hooks")

                # Pre-identify directories to give agent concrete targets
                target_dirs = infer_target_directories(self.repo_path, issue)
                if target_dirs:
                    dir_example = str(target_dirs[0].relative_to(self.repo_path))
                else:
                    dir_example = "src/"

                prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

=== HOW TO USE SCRIBE (CRITICAL - READ THIS) ===

You have `scribe` which returns a function/class AND ALL ITS DEPENDENCIES in one call.
This is the ONLY tool you need for understanding code. Do NOT use grep/read for exploration.

**PRIMARY USAGE - Covering Set (use this!):**
When you identify a relevant function or class, get it WITH all dependencies:
```
scribe --covering-set "path/to/file.py:function_name" --stdout
scribe --covering-set "path/to/file.ts:ClassName" --stdout
scribe --covering-set "path/to/file.go:MethodName" --stdout
```

This returns the target entity PLUS every type, function, and constant it uses.
You get the complete dependency graph in ONE call. No need to trace imports manually.

**SECONDARY USAGE - Directory Overview (only if you don't know the target):**
```
scribe --token-target 8000 {dir_example} --stdout
```

=== WORKFLOW ===

1. Use grep ONCE to find the file/function mentioned in the issue
2. Run scribe --covering-set on that function to get complete context
3. Implement your fix using the context scribe provided
4. Run tests

=== CRITICAL RULES ===

- After scribe returns, DO NOT read the same files again. Scribe already gave you everything.
- After scribe returns, DO NOT grep for more context. You have the dependency graph.
- DO NOT pipe scribe through head/tail/grep. Let it complete fully.
- NEVER run scribe on "." - always target a specific directory or use --covering-set

The whole point of scribe is to REPLACE iterative exploration. Use it once, then fix the code."""

            else:  # standard mode
                prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works."""

            # Run Claude Code
            claude_bin = get_claude_path()

            # Build command based on mode:
            # - standard/scribe-context: Use --allowedTools to grant permissions (no hooks needed)
            # - scribe-tool: Use hooks for permissions (--allowedTools bypasses hooks)
            cmd = [
                claude_bin,
                "-p",  # Print mode (non-interactive)
                "--model", self.model,
                "--output-format", "json",  # Single JSON result at end
            ]

            if mode in ("standard", "scribe-context"):
                # Use --dangerously-skip-permissions instead of --allowedTools
                # --allowedTools was causing "only prompt commands are supported in streaming mode" errors
                cmd.append("--dangerously-skip-permissions")
            # scribe-tool mode: hooks handle permissions, no permission bypass (hooks need to work)

            cmd.append(prompt)

            if self.verbose:
                print(f"    Running Claude Code ({self.model})...")
                print(f"    Timeout: {self.task_timeout_s}s")
                print(f"    Command: claude -p --model {self.model}...")

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

            # Parse output for metrics (even if returncode != 0, Claude may have worked)
            metrics = parse_claude_output(proc_result.stdout)
            result.input_tokens = metrics["input_tokens"]
            result.output_tokens = metrics["output_tokens"]
            result.total_tokens = result.input_tokens + result.output_tokens
            result.num_tool_calls = metrics.get("num_turns", 0)  # Use turns as proxy for tool calls
            result.scribe_calls = metrics["scribe_calls"]
            result.model_resolved = metrics.get("model_resolved", "")
            result.metrics_reliable = not metrics.get("had_error", False)
            if not result.metrics_reliable and self.verbose:
                errors = metrics.get("error_types", [])
                error_summary = errors[0][:50] if errors else "unknown"
                print(f"    Warning: Claude errored during execution ({error_summary}...)")
            result.tool_calls = [
                ToolCall(
                    name=tc["name"],
                    input=tc["input"],
                    output="",
                    tokens_in_output=0,
                )
                for tc in metrics["tool_calls"]
            ]

            # Get patch (git diff) - this is the real measure of success
            diff_result = subprocess.run(
                ["git", "diff"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result.patch = diff_result.stdout[:5000]
            result.resolved = len(result.patch.strip()) > 0

            # Consider successful if we got a patch, regardless of Claude's internal errors
            result.success = result.resolved or proc_result.returncode == 0
            if proc_result.returncode != 0 and not result.resolved:
                result.error = proc_result.stderr[:500] if proc_result.stderr else "Non-zero exit with no patch"

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
            result.duration_s = time.time() - start_time
            if self.verbose:
                print(f"    Timeout expired after {result.duration_s:.1f}s")
                if hasattr(e, 'stdout') and e.stdout:
                    print(f"    partial stdout: {e.stdout[:300] if isinstance(e.stdout, str) else e.stdout.decode()[:300]}")

            # Still try to get the patch - Claude may have made edits before timeout
            try:
                if self.repo_path and self.repo_path.exists():
                    diff_result = subprocess.run(
                        ["git", "diff"],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    result.patch = diff_result.stdout[:5000]
                    result.resolved = len(result.patch.strip()) > 0
                    if result.resolved:
                        result.success = True
                        result.error = f"Timeout after {self.task_timeout_s}s (but patch was generated)"
                        if self.verbose:
                            print(f"    Found patch despite timeout ({len(result.patch)} chars)")
                    else:
                        result.error = f"Timeout after {self.task_timeout_s}s"
                        result.success = False
            except Exception:
                result.error = f"Timeout after {self.task_timeout_s}s"
                result.success = False

        except Exception as e:
            result.success = False
            result.error = str(e)[:500]
            result.duration_s = time.time() - start_time

        finally:
            self.cleanup_task()

        return result


def _run_single_task(args: tuple) -> TaskResult:
    """Worker function for parallel task execution."""
    task, run_mode, model, use_docker, task_timeout_s, task_index, total_tasks, context_tokens = args
    task_id = task.get("instance_id", f"task_{task_index}")

    # Create a new runner for each task (thread-safe)
    runner = TaskRunner(model=model, use_docker=use_docker, task_timeout_s=task_timeout_s, context_tokens=context_tokens)
    runner.verbose = False  # Less verbose in parallel mode

    print(f"  [{task_index+1}/{total_tasks}] {task_id} ({run_mode})...")
    result = runner.run_task(task, mode=run_mode)
    status = "OK" if result.success else f"FAILED"
    print(f"    [{task_index+1}/{total_tasks}] {task_id} ({run_mode}): {status} - {result.total_tokens:,} tokens")
    return result


def run_task_batch(
    tasks: list[dict],
    mode: str = "both",
    model: str = "sonnet",
    max_tasks: Optional[int] = None,
    use_docker: bool = True,
    prepull_workers: int = 4,
    task_timeout_s: int = 600,
    parallel_workers: int = 1,
    context_tokens: int = 4000,
) -> list[TaskResult]:
    """Run a batch of SWE-bench tasks.

    Args:
        tasks: List of SWE-bench task dicts.
        mode: "standard", "scribe-context", "scribe-tool", "both", or "all".
              Legacy "scribe" = "scribe-context".
              "both" = standard + scribe-context.
              "all" = standard + scribe-context + scribe-tool.
        model: Model to use (e.g., "sonnet", "opus", "claude-sonnet-4-5-20250929").
        max_tasks: Maximum number of tasks to run.
        use_docker: Whether to use Docker for isolation.
        prepull_workers: Number of parallel workers for image pre-pulling.
        task_timeout_s: Timeout per task in seconds.
        parallel_workers: Number of tasks to run in parallel (default: 1 = sequential).
        context_tokens: Token budget for scribe-context mode (default: 4000).

    Returns:
        List of TaskResult objects.
    """
    # Handle legacy mode name
    if mode == "scribe":
        mode = "scribe-context"

    # Clean up stale work directories to free disk space
    cleanup_stale_workdirs(max_age_hours=1)

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

    # Determine which modes to run
    run_standard = mode in ("both", "all", "standard")
    run_scribe_context = mode in ("both", "all", "scribe-context")
    run_scribe_tool = mode in ("all", "scribe-tool")

    # Build list of (task, mode) pairs to run
    work_items = []
    for i, task in enumerate(tasks):
        if run_standard:
            work_items.append((task, "standard", model, use_docker, task_timeout_s, i, len(tasks), context_tokens))
        if run_scribe_context:
            work_items.append((task, "scribe-context", model, use_docker, task_timeout_s, i, len(tasks), context_tokens))
        if run_scribe_tool:
            work_items.append((task, "scribe-tool", model, use_docker, task_timeout_s, i, len(tasks), context_tokens))

    results = []

    if parallel_workers > 1:
        print(f"Running {len(work_items)} task/mode combinations with {parallel_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_item = {executor.submit(_run_single_task, item): item for item in work_items}
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"  Task {item[0].get('instance_id')} ({item[1]}) raised exception: {e}")
    else:
        # Sequential execution (original behavior)
        runner = TaskRunner(model=model, use_docker=use_docker, task_timeout_s=task_timeout_s, context_tokens=context_tokens)

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

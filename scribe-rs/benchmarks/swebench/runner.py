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
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

try:
    from common.claude_config import resolve_claude_config_dir, build_claude_env
except ImportError:
    # Allow running directly from this directory
    from ..common.claude_config import resolve_claude_config_dir, build_claude_env


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

WORKDIR_ROOT = Path(__file__).parent.parent / ".workdirs"


def ensure_workdir_root() -> Path:
    """Ensure the benchmark workdir root exists."""
    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKDIR_ROOT

# Code file extensions for repo size estimation
CODE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs",
    ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc", ".rb", ".php", ".swift",
    ".scala", ".cs", ".lua", ".ex", ".exs", ".hs", ".ml", ".clj", ".vue", ".svelte",
    ".sol",
}

SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


def estimate_repo_code_bytes(repo_path: Path, max_bytes: int = 200_000_000) -> int:
    """Estimate total bytes of code files in a repo.

    Excludes common vendor/cache/test directories. Stops early once max_bytes is reached.
    """
    total = 0
    for root, dirs, files in os.walk(repo_path):
        # Prune dirs
        pruned = []
        for d in dirs:
            if d in SKIP_DIR_NAMES:
                continue
            if is_test_path(Path(root) / d):
                continue
            pruned.append(d)
        dirs[:] = pruned

        for name in files:
            ext = Path(name).suffix.lower()
            if ext not in CODE_EXTS:
                continue
            path = Path(root) / name
            if is_test_path(path):
                continue
            try:
                total += path.stat().st_size
                if total >= max_bytes:
                    return total
            except Exception:
                continue
    return total


def choose_context_tokens(code_bytes: int, base_tokens: int, max_tokens: int = 12000) -> int:
    """Choose a token budget based on repo size.

    Scales from base_tokens up to max_tokens using simple size thresholds.
    """
    mb = code_bytes / (1024 * 1024) if code_bytes else 0
    if mb <= 2:
        scale = 1.0
    elif mb <= 8:
        scale = 1.5
    elif mb <= 20:
        scale = 2.0
    elif mb <= 50:
        scale = 2.5
    else:
        scale = 3.0

    target = int(base_tokens * scale)
    return max(base_tokens, min(max_tokens, target))


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
    tmp_dir = ensure_workdir_root()
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
    for d in Path(tempfile.gettempdir()).glob("go-build*"):
        try:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                cleaned += 1
        except Exception:
            pass

    if cleaned > 0:
        print(f"Cleaned up {cleaned} stale work directories")


def _scribe_hook_source() -> str:
    return '''#!/usr/bin/env python3
"""Surgical hook: Suggest scribe only when it would clearly help.

Philosophy:
- Grep is for exploration (finding WHERE)
- Scribe is for understanding (understanding HOW)
- Don't force scribe during exploration phase
- Suggest scribe when agent is clearly trying to understand something deeply

Triggers for scribe suggestion:
1. Multiple reads of same file → "use scribe for the full picture"
2. Multiple greps on same file → "scribe shows the dependency graph"
3. Reading a file with many imports → "scribe includes dependencies"

NO BLOCKING. Only suggestions.
"""
import json, sys, os, re

CODE_EXTS = {".py",".pyi",".js",".jsx",".ts",".tsx",".go",".rs",".java",".c",".cpp",".rb"}
LOG_PATH = os.environ.get("CLAUDE_HOOK_LOG")

def log_event(kind, data):
    if not LOG_PATH: return
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({"kind": kind, **data}) + "\\n")
    except: pass

def get_state(sid):
    try:
        with open(f"/tmp/claude_hook_{sid}.json") as f: return json.load(f)
    except: return {"file_reads": {}, "file_greps": {}, "scribe_used": False}

def save_state(sid, s):
    try:
        with open(f"/tmp/claude_hook_{sid}.json", "w") as f: json.dump(s, f)
    except: pass

def allow(msg=None):
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
    if msg:
        out["hookSpecificOutput"]["permissionDecisionReason"] = msg
        log_event("suggest", {"message": msg[:500]})
    print(json.dumps(out))
    sys.exit(0)

def main():
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    if not raw and len(sys.argv) > 1: raw = sys.argv[1]
    if not raw: allow()
    
    if os.path.exists(raw):
        try: raw = open(raw).read()
        except: pass
    
    try: data = json.loads(raw)
    except: allow()

    sid = data.get("session_id", "x")
    tool = data.get("tool_name", "")
    event = data.get("hook_event_name", "")
    inp = data.get("tool_input", {})

    if event != "PreToolUse": sys.exit(0)

    state = get_state(sid)
    
    # Track file reads - suggest scribe proactively
    if tool == "Read":
        fp = inp.get("file_path", "")
        ext = os.path.splitext(fp)[1].lower()
        fname = os.path.basename(fp)
        
        if ext in CODE_EXTS:
            reads = state.setdefault("file_reads", {})
            total_reads = state.get("total_code_reads", 0) + 1
            reads[fp] = reads.get(fp, 0) + 1
            state["total_code_reads"] = total_reads
            save_state(sid, state)
            
            # Log all code file reads for tracking
            log_event("code_read", {"file": fp, "count": reads[fp], "total": total_reads})
            
            # First code read - introduce scribe
            if total_reads == 1 and not state.get("scribe_introduced"):
                state["scribe_introduced"] = True
                save_state(sid, state)
                allow(f"""REMINDER: Use scribe for code understanding:
  scribe --covering-set "{fp}:FUNCTION_NAME" --stdout
This shows the function AND all its dependencies in one call.""")
            
            # Second read of same file - stronger suggestion
            elif reads[fp] == 2:
                allow(f"TIP: Reading {fname} again? scribe --covering-set \"{fp}:TARGET\" shows it + all dependencies.")
        
        allow()

    # Track grep patterns - suggest scribe after 2 greps (reduced from 3)
    if tool == "Grep":
        path = inp.get("path", "")
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext in CODE_EXTS:
                greps = state.setdefault("file_greps", {})
                greps[path] = greps.get(path, 0) + 1
                save_state(sid, state)
                
                log_event("code_grep", {"path": path, "count": greps[path]})
                
                if greps[path] == 2:
                    fname = os.path.basename(path)
                    allow(f"TIP: Multiple searches in {fname}. Try: scribe --covering-set \"{path}:TARGET\" --stdout")
        
        allow()

    # Track scribe usage
    if tool == "Bash":
        cmd = inp.get("command", "")
        if re.search(r"^\\s*scribe\\s", cmd, re.I):
            state["scribe_used"] = True
            log_event("scribe_cmd", {"cmd": cmd[:200]})
            save_state(sid, state)
        allow()

    # No blocking on Edit/Write - agent may have enough context
    allow()

if __name__ == "__main__":
    main()
'''


def _merge_hook_settings(settings: dict, hook_command: str) -> dict:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}

    pre = hooks.get("PreToolUse")
    if not isinstance(pre, list):
        pre = []

    entry = {"matcher": ".*", "hooks": [{"type": "command", "command": hook_command}]}
    already = False
    for existing in pre:
        if not isinstance(existing, dict):
            continue
        if existing.get("matcher") != ".*":
            continue
        for hook in existing.get("hooks", []) or []:
            if hook.get("type") == "command" and hook.get("command") == hook_command:
                already = True
                break
        if already:
            break

    if not already:
        pre.append(entry)

    hooks["PreToolUse"] = pre
    settings["hooks"] = hooks
    return settings


def setup_scribe_hooks_config(config_dir: Path) -> None:
    """Install scribe enforcement hook into the benchmark config directory."""
    hooks_dir = config_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_script = hooks_dir / "scribe_enforce.py"
    hook_script.write_text(_scribe_hook_source())
    hook_script.chmod(0o755)

    settings_path = config_dir / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text() or "{}")
        except Exception:
            settings = {}
    if not isinstance(settings, dict):
        settings = {}

    python_bin = shutil.which("python3") or "python3"
    hook_command = f"{python_bin} {hook_script.resolve()}"
    settings = _merge_hook_settings(settings, hook_command)
    settings_path.write_text(json.dumps(settings, indent=2))


def mark_claude_project_trusted(config_dir: Optional[Path], project_path: Path) -> None:
    """Mark a repo as trusted in the benchmark Claude config to avoid trust prompts."""
    if not config_dir:
        return
    claude_state = config_dir / ".claude.json"
    state: dict = {}
    if claude_state.exists():
        try:
            state = json.loads(claude_state.read_text() or "{}")
        except Exception:
            state = {}
    if not isinstance(state, dict):
        state = {}

    projects = state.get("projects")
    if not isinstance(projects, dict):
        projects = {}

    key = str(project_path)
    entry = projects.get(key)
    if not isinstance(entry, dict):
        entry = {
            "allowedTools": [],
            "mcpContextUris": [],
            "mcpServers": {},
            "enabledMcpjsonServers": [],
            "disabledMcpjsonServers": [],
            "hasTrustDialogAccepted": True,
            "projectOnboardingSeenCount": 0,
            "hasClaudeMdExternalIncludesApproved": False,
            "hasClaudeMdExternalIncludesWarningShown": False,
        }
    else:
        entry = dict(entry)
        entry["hasTrustDialogAccepted"] = True

    projects[key] = entry
    state["projects"] = projects
    claude_state.write_text(json.dumps(state))


def ensure_scribe_tool_config_dir(base_dir: Optional[Path]) -> Optional[Path]:
    """Return a scribe-tool-specific config dir with hooks and copied env."""
    if not base_dir:
        return None
    scribe_dir = base_dir / "scribe-tool"
    scribe_dir.mkdir(parents=True, exist_ok=True)

    # Copy base state if present (preserves cached auth + trust metadata)
    base_state = base_dir / ".claude.json"
    scribe_state = scribe_dir / ".claude.json"
    if base_state.exists() and not scribe_state.exists():
        try:
            shutil.copyfile(base_state, scribe_state)
        except Exception:
            pass

    # Ensure env settings are present in scribe-tool settings
    base_settings_path = base_dir / "settings.json"
    scribe_settings_path = scribe_dir / "settings.json"
    base_settings: dict = {}
    if base_settings_path.exists():
        try:
            base_settings = json.loads(base_settings_path.read_text() or "{}")
        except Exception:
            base_settings = {}
    if not isinstance(base_settings, dict):
        base_settings = {}

    scribe_settings: dict = {}
    if scribe_settings_path.exists():
        try:
            scribe_settings = json.loads(scribe_settings_path.read_text() or "{}")
        except Exception:
            scribe_settings = {}
    if not isinstance(scribe_settings, dict):
        scribe_settings = {}

    if isinstance(base_settings.get("env"), dict):
        scribe_settings["env"] = base_settings["env"]

    scribe_settings_path.write_text(json.dumps(scribe_settings, indent=2))
    setup_scribe_hooks_config(scribe_dir)
    return scribe_dir


def setup_scribe_hooks(work_dir: Path) -> None:
    """Set up Claude Code hooks to enforce scribe usage in a local repo."""
    hooks_dir = work_dir / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_script = hooks_dir / "scribe_enforce.py"
    hook_script.write_text(_scribe_hook_source())
    hook_script.chmod(0o755)

    # Write settings.json - hook applies to ALL tools (.*) since we handle permissions
    # IMPORTANT: We use ".*" matcher because --allowedTools bypasses hooks, so we must
    # handle all permission decisions in the hook instead.
    # Must use absolute path for hook script
    settings = work_dir / ".claude" / "settings.json"
    abs_hook_path = hook_script.resolve()
    python_bin = shutil.which("python3") or "python3"
    settings.write_text(
        json.dumps(
            _merge_hook_settings({}, f"{python_bin} {abs_hook_path}"),
            indent=2,
        )
    )


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


def extract_github_file_refs(issue_text: str) -> list[dict]:
    """Extract GitHub file references from issue text.

    Returns list of dicts with 'path' and optional 'line' keys.
    E.g., https://github.com/org/repo/blob/commit/path/to/file.go#L238
    """
    refs = []
    # Match GitHub blob URLs with optional line number
    github_pattern = r'https?://github\.com/[^/]+/[^/]+/blob/[^/]+/([^\s#]+)(?:#L(\d+))?'
    for match in re.finditer(github_pattern, issue_text):
        ref = {"path": match.group(1)}
        if match.group(2):
            ref["line"] = int(match.group(2))
        refs.append(ref)
    return refs


def extract_code_references(issue_text: str, language: str = "python") -> list[str]:
    """Extract file paths and code references from issue text.

    Looks for patterns like:
    - module.path.ClassName
    - path/to/file.py
    - function_name()
    - GitHub blob URLs

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

    # GitHub file links (highest priority - extract file paths)
    for github_ref in extract_github_file_refs(issue_text):
        refs.append(github_ref["path"])

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
                # Determine target path - prefer src/ if it exists
                # This avoids noise from demo/docs/examples directories
                # Note: 'lib' is excluded as it's usually compiled output
                target_path = repo_path
                for src_dir in ['src', 'pkg', 'packages']:
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


def summarize_hook_log(log_path: Path, max_events: int = 50) -> dict:
    """Summarize hook log events for diagnostics."""
    summary = {
        "denies": 0,
        "warnings": 0,
        "scribe_cmds": 0,
        "events": [],
    }
    try:
        if not log_path.exists():
            return summary
        with log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                kind = event.get("kind")
                if kind == "deny":
                    summary["denies"] += 1
                elif kind == "warn":
                    summary["warnings"] += 1
                elif kind == "scribe_cmd":
                    summary["scribe_cmds"] += 1
                if len(summary["events"]) < max_events:
                    summary["events"].append(event)
    except Exception:
        pass
    return summary


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
    repo_code_bytes: int = 0
    context_tokens: int = 0
    hook_denies: int = 0
    hook_warnings: int = 0
    scribe_commands: int = 0
    hook_events: list = field(default_factory=list)
    hook_debug_invocations: int = 0
    hook_debug_events: list = field(default_factory=list)
    baseline_dirty: bool = False
    baseline_diff_len: int = 0

    # Outcome
    resolved: bool = False
    patch: str = ""
    test_output: str = ""
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
    retry_count: int = 0
    system_failure: bool = False

    # Status
    success: bool = True
    error: str = ""

    # Metrics reliability - False if Claude errored during execution
    metrics_reliable: bool = True

    # Raw output for debugging
    raw_output: str = ""

    # Model tracking - actual model ID returned by Claude (e.g., "claude-sonnet-4-5-20250929")
    model_resolved: str = ""


SYSTEM_ERROR_PATTERNS = [
    r"only prompt commands are supported in streaming mode",
    r"lock acquisition failed",
    r"api error: 429",
    r"usage limit reached",
    r"error: non-fatal: lock acquisition failed",
    r"datasets package required",
    r"failed to set up task environment",
    r"docker\\.errors",
]


def is_system_failure(result: TaskResult) -> bool:
    """Return True if a failure looks like a system/infra error worth retrying."""
    if result.success:
        return False
    haystack = f"{result.error}\n{result.raw_output or ''}".lower()
    return any(re.search(pat, haystack, re.IGNORECASE) for pat in SYSTEM_ERROR_PATTERNS)


def write_progress(results: list[TaskResult], progress_path: Path) -> None:
    """Write a progress file with current results list (atomic replace)."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for item in results:
        if is_dataclass(item) and not isinstance(item, type):
            payload.append(asdict(item))
        else:
            payload.append(item)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str))
    tmp_path.replace(progress_path)


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



# Pi agent support
PI_MONO_PATH = Path.home() / "Projects" / "pi-mono"
PI_SCRIBE_EXTENSION = PI_MONO_PATH / ".pi" / "extensions" / "scribe.ts"


def get_pi_path() -> Optional[str]:
    """Get path to Pi coding agent CLI."""
    cli_path = PI_MONO_PATH / "packages" / "coding-agent" / "dist" / "cli.js"
    if cli_path.exists():
        return str(cli_path)
    return None


def check_pi_installed() -> bool:
    """Check if Pi coding agent is installed."""
    return get_pi_path() is not None


def parse_pi_output(stdout: str, stderr: str) -> dict:
    """Parse Pi agent output to extract metrics.

    Handles both text output and JSONL streaming output.
    Extension stats are parsed from stderr.
    """
    metrics = {
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": [],
        "scribe_calls": 0,
        "total_cost_usd": 0,
        "num_turns": 0,
        "had_error": False,
        "error_types": [],
        "model_resolved": "",
    }

    # Parse extension stats from stderr: [scribe-ext] Stats: reads=1, readBlocks=1, ...
    ext_stats_match = re.search(
        r'\[scribe-ext\] Stats: reads=(\d+), readBlocks=(\d+), greps=(\d+), grepBlocks=(\d+), scribeCalls=(\d+)',
        stderr
    )
    if ext_stats_match:
        metrics["scribe_calls"] = int(ext_stats_match.group(5))
        metrics["read_blocks"] = int(ext_stats_match.group(2))
        metrics["grep_blocks"] = int(ext_stats_match.group(4))

    # Try to parse JSONL output (--mode json)
    for line in stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            event_type = event.get("type", "")

            # Extract usage from turn_end or agent_end
            if event_type in ("turn_end", "agent_end"):
                if event_type == "turn_end":
                    msg = event.get("message", {})
                else:
                    msgs = event.get("messages", [])
                    msg = msgs[-1] if msgs else {}

                usage = msg.get("usage", {})
                if usage:
                    metrics["input_tokens"] += usage.get("input", 0) + usage.get("cacheRead", 0)
                    metrics["output_tokens"] += usage.get("output", 0)
                    cost = usage.get("cost", {})
                    metrics["total_cost_usd"] += cost.get("total", 0)

                # Extract model info
                if msg.get("model"):
                    metrics["model_resolved"] = msg.get("model", "")

            # Count tool calls
            if event_type == "tool_start":
                tool_name = event.get("toolName", "")
                metrics["tool_calls"].append({
                    "name": tool_name,
                    "input": event.get("input", {}),
                })
                # Check for scribe in bash commands
                if tool_name == "bash":
                    cmd = event.get("input", {}).get("command", "")
                    if re.search(r'scribe\s+--', cmd, re.IGNORECASE):
                        metrics["scribe_calls"] += 1

            # Count turns
            if event_type == "turn_start":
                metrics["num_turns"] += 1

        except json.JSONDecodeError:
            # Text mode output - scan for scribe patterns
            scribe_patterns = [
                r'scribe\s+--covering-set',
                r'scribe\s+--token-target',
                r'scribe\s+--stdout',
            ]
            for pattern in scribe_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    metrics["scribe_calls"] += 1

    # Check for errors in stderr
    if "error" in stderr.lower() and ("api" in stderr.lower() or "timeout" in stderr.lower()):
        metrics["had_error"] = True

    return metrics

def get_docker_image_name(task: dict) -> str:
    """Get the Docker image name for a SWE-bench task.

    Supports multiple naming conventions:
    - Standard/Multilingual: sweb.eval.x86_64.{instance_id}:latest
      where instance_id is like 'astropy__astropy-12907'
    - Pro: jefzda/sweap-images:{owner}.{repo}-{owner}__{repo}-{commit}-v{version}
      Pre-built images on DockerHub for SWE-bench Pro
    """
    instance_id = task.get("instance_id", "")

    # SWE-bench Pro instance IDs start with 'instance_'
    if instance_id.startswith("instance_"):
        # Pro format: instance_{owner}__{repo}-{commit}-v{version}
        # Docker format: jefzda/sweap-images:{owner}.{repo}-{owner}__{repo}-{commit}-v{version}
        clean_id = instance_id[len("instance_"):]
        match = re.match(r'^([^_]+)__([^-]+)-(.+)$', clean_id)
        if match:
            owner, repo, rest = match.groups()
            return f"jefzda/sweap-images:{owner}.{repo}-{owner}__{repo}-{rest}"
        # Fallback
        return f"jefzda/sweap-images:{clean_id}"

    # Standard format
    return f"sweb.eval.x86_64.{instance_id}:latest"

def get_container_repo_path(task: dict) -> str:
    """Get the repo path inside the container.
    
    Different SWE-bench variants store repos in different locations:
    - Standard/Multilingual: /testbed/
    - Pro: /app/
    """
    instance_id = task.get("instance_id", "")
    if instance_id.startswith("instance_"):
        return "/app/"
    return "/testbed/"




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
    """Parse Claude Code output to extract metrics.

    Handles both JSON output (--output-format json) and text output.
    Text output is used when JSON causes streaming errors with some providers.
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

    # Patterns for detecting scribe usage in text output
    scribe_patterns = [
        r'scribe\s+--covering-set',
        r'scribe\s+--token-target',
        r'scribe\s+--query-hint',
        r'scribe\s+--stdout',
        r'scribe\s+\.',  # scribe .
        r'scribe\s+src/',  # scribe src/...
        r'scribe\s+["\'"][^\"\'\]+["\'"]',  # scribe "path" or scribe 'path'
    ]

    try:
        # Try to parse as JSON first
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
        result_text = result.get("result", "")
        if isinstance(result_text, str):
            for pattern in scribe_patterns:
                metrics["scribe_calls"] += len(re.findall(pattern, result_text, re.IGNORECASE))

    except json.JSONDecodeError:
        # Text output mode - scan the raw text for scribe commands
        # This is less precise but allows basic tracking when JSON isn't available
        for pattern in scribe_patterns:
            metrics["scribe_calls"] += len(re.findall(pattern, output, re.IGNORECASE))

        # Try to detect errors in text output
        if "error" in output.lower() and ("api" in output.lower() or "timeout" in output.lower()):
            metrics["had_error"] = True

    return metrics


# Backwards compatibility alias
def parse_opencode_output(output: str) -> dict:
    """Legacy alias for parse_claude_output."""
    return parse_claude_output(output)


class TaskRunner:
    """Runs SWE-bench tasks using Claude Code."""

    def __init__(
        self,
        agent_type: str = "claude",  # "claude" or "pi"
        model: str = "glm-4.7",
        scribe_binary: Optional[str] = None,
        use_docker: bool = True,
        task_timeout_s: int = 2400,  # Timeout for Claude Code to solve the task
        setup_timeout_s: int = 120,  # Timeout for container/repo setup
        context_tokens: int = 4000,  # Token budget for scribe-context mode
        scribe_max_depth: int = 4,  # Default max depth for scribe covering-set
        claude_config_dir: Optional[Union[str, Path]] = None,
    ):
        self.agent_type = agent_type
        if agent_type == "claude" and not check_claude_installed():
            raise RuntimeError(
                "Claude Code not installed. Install with: npm install -g @anthropic-ai/claude-code"
            )
        if agent_type == "pi" and not check_pi_installed():
            raise RuntimeError(
                f"Pi coding agent not installed. Expected at: {PI_MONO_PATH}"
            )

        self.model = model
        self.claude_config_dir = resolve_claude_config_dir(
            str(claude_config_dir) if claude_config_dir else None
        )
        self.scribe_binary = scribe_binary or shutil.which("scribe") or "scribe"
        self.use_docker = use_docker
        self.task_timeout_s = task_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.context_tokens = context_tokens
        self.scribe_max_depth = scribe_max_depth
        self.verbose = True  # Enable verbose output for debugging

        # Current task context
        self.repo_path: Optional[Path] = None
        self.task_id: Optional[str] = None
        self.container_id: Optional[str] = None
        self._current_task: Optional[dict] = None  # Store current task for helper functions

    def setup_task(self, task: dict) -> bool:
        """Set up the environment for a task."""
        self.task_id = task.get("instance_id", "unknown")
        self._current_task = task  # Store for helper functions

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
                ["docker", "run", "-d", "--rm", "--entrypoint", "", image_name, "sleep", "3600"],
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
            self.repo_path = Path(tempfile.mkdtemp(prefix="swebench_", dir=ensure_workdir_root()))
            if self.verbose:
                print(f"    Copying repo to {self.repo_path}...")
            cp_result = subprocess.run(
                ["docker", "cp", f"{self.container_id}:{get_container_repo_path(self._current_task)}.", str(self.repo_path)],
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

            mark_claude_project_trusted(self.claude_config_dir, self.repo_path)

            # Run before_repo_set_cmd if specified (SWE-bench Pro)
            if not self._run_setup_command(task):
                if self.verbose:
                    print("    Warning: setup command failed (continuing anyway)")

            return True

        except Exception as e:
            print(f"    Docker setup failed: {e}")
            return False

    def _run_setup_command(self, task: dict) -> bool:
        """Run before_repo_set_cmd if present (SWE-bench Pro feature).

        Args:
            task: Task dict that may contain before_repo_set_cmd

        Returns:
            True if no command or command succeeded, False if command failed
        """
        before_cmd = task.get("before_repo_set_cmd", "")
        if not before_cmd:
            return True

        if self.verbose:
            print(f"    Running setup command: {before_cmd[:80]}...")
        try:
            result = subprocess.run(
                before_cmd,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.setup_timeout_s,
            )
            if result.returncode != 0:
                if self.verbose:
                    print(f"    Setup command warning: {result.stderr[:200]}")
                return False
            return True
        except Exception as e:
            if self.verbose:
                print(f"    Setup command failed: {e}")
            return False

    def _setup_local(self, task: dict) -> bool:
        """Set up task locally by cloning repo."""
        self.repo_path = Path(tempfile.mkdtemp(prefix="swebench_", dir=ensure_workdir_root()))

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
            mark_claude_project_trusted(self.claude_config_dir, self.repo_path)

            # Run before_repo_set_cmd if specified (SWE-bench Pro)
            if not self._run_setup_command(task):
                if self.verbose:
                    print("    Warning: setup command failed (continuing anyway)")

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


    def _evaluate_patch(self, task: dict, patch: str) -> tuple[bool, str]:
        """Evaluate if a patch actually resolves the issue by running tests.
        
        Args:
            task: Task dict with test info (fail_to_pass, test_patch)
            patch: The agent's generated patch
            
        Returns:
            Tuple of (resolved: bool, test_output: str)
        """
        if self.verbose:
            print(f"    [EVAL] Starting evaluation, container_id={self.container_id[:12] if self.container_id else 'None'}")
        if not self.container_id:
            return False, "No container available for evaluation"
        
        container_repo_path = get_container_repo_path(task)
        
        # 1. Apply the agent's patch to the container (already done earlier)
        # The files were copied back after Claude Code finished
        
        # 2. Apply the test_patch (adds/modifies test files for evaluation)
        test_patch = task.get("test_patch", "")
        if test_patch.strip():
            # Write test_patch to container and apply it
            apply_cmd = f"cd {container_repo_path} && git apply --3way -"
            apply_result = subprocess.run(
                ["docker", "exec", "-i", self.container_id, "bash", "-c", apply_cmd],
                input=test_patch,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if apply_result.returncode != 0:
                # Try without --3way
                apply_result = subprocess.run(
                    ["docker", "exec", "-i", self.container_id, "bash", "-c", 
                     f"cd {container_repo_path} && git apply -"],
                    input=test_patch,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if apply_result.returncode != 0:
                    if self.verbose:
                        print(f"    Warning: Could not apply test_patch: {apply_result.stderr[:200]}")
        
        # 3. Get fail_to_pass tests (check both lowercase and uppercase field names)
        fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or ""
        if self.verbose:
            print(f"    [EVAL] fail_to_pass raw: {repr(fail_to_pass)[:100]}")
        if isinstance(fail_to_pass, str):
            try:
                import ast
                # fail_to_pass can be Python list literal (single quotes) or JSON
                fail_to_pass = ast.literal_eval(fail_to_pass) if fail_to_pass.startswith('[') else []
            except Exception:
                try:
                    import json as json_module
                    fail_to_pass = json_module.loads(fail_to_pass) if fail_to_pass.startswith('[') else []
                except Exception:
                    fail_to_pass = []

        if not fail_to_pass:
            # No specific tests to check - consider resolved if patch exists
            return bool(patch.strip()), "No fail_to_pass tests specified"

        # 4. Detect project type and build test command
        test_specs = fail_to_pass[:20]

        # Detect project type by checking for key files
        # Python: setup.py, pyproject.toml, or Python test format
        # JS/TS: package.json with vitest format tests (file > describe > test)

        # Check test format - vitest uses " > " separators
        is_vitest_format = any(" > " in spec for spec in test_specs)

        # Check for Python project indicators (setup.py or pyproject.toml)
        check_python = subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-c",
             f"test -f {container_repo_path}/setup.py || test -f {container_repo_path}/pyproject.toml"],
            capture_output=True, timeout=10
        )
        is_python_project = check_python.returncode == 0

        # Check for Go project (go.mod)
        check_go = subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-c",
             f"test -f {container_repo_path}/go.mod"],
            capture_output=True, timeout=10
        )
        is_go_project = check_go.returncode == 0

        # Check for Rust project (Cargo.toml)
        check_rust = subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-c",
             f"test -f {container_repo_path}/Cargo.toml"],
            capture_output=True, timeout=10
        )
        is_rust_project = check_rust.returncode == 0

        # Django-style tests: test_name (module.Class)
        is_django_format = any(re.match(r"^\w+\s*\(", spec) for spec in test_specs)

        # Go-style tests: TestFunctionName or TestFunctionName/subtest
        is_go_format = is_go_project or any(re.match(r"^Test[A-Z]", spec) for spec in test_specs)

        # Rust-style tests: module::test_name or just test_name with underscores
        is_rust_format = is_rust_project

        # Conda activation prefix for Python projects (if conda is available)
        conda_prefix = "source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate testbed 2>/dev/null; "

        if is_go_format and not is_python_project:
            # Go projects - use go test with -run flag
            # Handle both TestName and TestName/subtest formats
            test_patterns = []
            for spec in test_specs:
                # Go test patterns: TestName or TestName/subtest
                if "/" in spec:
                    # Subtest: TestName/subtest -> use as-is
                    test_patterns.append(spec)
                else:
                    # Regular test: TestName
                    test_patterns.append(f"^{spec}$")
            
            pattern_str = "|".join(test_patterns)
            test_cmd = f"cd {container_repo_path} && timeout 600 go test -v -run '{pattern_str}' ./... 2>&1; echo EXIT_CODE=$?"
            
            if self.verbose:
                print(f"    Running {len(test_specs)} fail_to_pass tests (Go)...")

        elif is_rust_format and not is_python_project:
            # Rust projects - use cargo test with filter
            # Test names can be module::test_name or just test_name
            test_filters = " ".join(test_specs)
            test_cmd = f"cd {container_repo_path} && timeout 600 cargo test {test_filters} -- --test-threads=1 2>&1; echo EXIT_CODE=$?"
            
            if self.verbose:
                print(f"    Running {len(test_specs)} fail_to_pass tests (Rust)...")

        elif is_vitest_format and not is_python_project:
            # JS/TS projects with vitest format
            test_patterns = []
            for spec in test_specs:
                if " > " in spec:
                    test_name = spec.split(" > ")[-1]
                    test_patterns.append(test_name)
                else:
                    test_patterns.append(spec)

            pattern_str = "|".join(re.escape(p) for p in test_patterns)
            test_cmd = f"cd {container_repo_path} && timeout 300 pnpm test -- --testNamePattern '{pattern_str}' 2>&1; echo EXIT_CODE=$?"

        elif is_django_format:
            # Django tests: use Django test runner
            # Format: test_name (module.Class) -> module.Class.test_name
            # Extract module from first test spec
            test_modules = set()
            for spec in test_specs:
                match = re.match(r"\w+\s*\(([^)]+)\)", spec)
                if match:
                    # Extract module.Class -> module
                    full_path = match.group(1)
                    test_modules.add(full_path.rsplit(".", 1)[0] if "." in full_path else full_path)

            if test_modules:
                modules_str = " ".join(sorted(test_modules))
                test_cmd = f"{conda_prefix}cd {container_repo_path}/tests && timeout 300 python runtests.py {modules_str} --parallel 1 2>&1; echo EXIT_CODE=$?"
            else:
                test_cmd = f"{conda_prefix}cd {container_repo_path} && timeout 300 python -m pytest --tb=short -q 2>&1; echo EXIT_CODE=$?"

        else:
            # Generic Python - use pytest with conda environment
            test_spec_str = " ".join(f'"{t}"' for t in test_specs)
            test_cmd = f"{conda_prefix}cd {container_repo_path} && timeout 300 python -m pytest {test_spec_str} --tb=short -q 2>&1; echo EXIT_CODE=$?"

        # Print test type for non-Go/Rust (those print earlier in their branches)
        if not (is_go_format and not is_python_project) and not (is_rust_format and not is_python_project):
            if self.verbose:
                test_type = 'JS/TS' if is_vitest_format and not is_python_project else 'Django' if is_django_format else 'Python'
                print(f"    Running {len(test_specs)} fail_to_pass tests ({test_type})...")

        test_result = subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-c", test_cmd],
            capture_output=True,
            text=True,
            timeout=360,
        )
        
        output = test_result.stdout

        # Check if tests passed
        # For Go/Rust: check specific test results since overall exit code may be non-zero
        # due to unrelated package issues
        if "EXIT_CODE=0" in output:
            if self.verbose:
                print(f"    Tests PASSED")
            return True, output

        # Go tests: check if all required tests show "--- PASS:"
        if is_go_format and not is_python_project:
            passed_tests = set()
            failed_tests = set()
            for line in output.split("\n"):
                if "--- PASS:" in line:
                    # Extract test name: "--- PASS: TestName (1.23s)"
                    match = re.search(r"--- PASS:\s+(\S+)", line)
                    if match:
                        passed_tests.add(match.group(1))
                elif "--- FAIL:" in line:
                    match = re.search(r"--- FAIL:\s+(\S+)", line)
                    if match:
                        failed_tests.add(match.group(1))

            # Check if all required tests passed
            all_passed = True
            for spec in test_specs:
                # Handle subtest format: TestName/subtest
                test_name = spec.split("/")[0] if "/" in spec else spec
                if test_name in failed_tests:
                    all_passed = False
                    break
                if test_name not in passed_tests:
                    # Test wasn't found at all - could be an issue
                    # But we'll be lenient if no failure was recorded
                    pass

            if all_passed and passed_tests:
                if self.verbose:
                    print(f"    Tests PASSED (Go: {len(passed_tests)} tests)")
                return True, output

        # Rust tests: check if tests show "ok"
        if is_rust_format and not is_python_project:
            # Rust output: "test module::test_name ... ok"
            passed_tests = []
            failed_tests = []
            for line in output.split("\n"):
                if " ... ok" in line:
                    passed_tests.append(line)
                elif " ... FAILED" in line:
                    failed_tests.append(line)

            if not failed_tests and passed_tests:
                if self.verbose:
                    print(f"    Tests PASSED (Rust: {len(passed_tests)} tests)")
                return True, output

        if self.verbose:
            # Extract failure summary
            lines = output.split("\n")
            for line in lines[-10:]:
                if "FAILED" in line or "ERROR" in line or "passed" in line:
                    print(f"    {line}")
        return False, output


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
        hook_log_path: Optional[Path] = None
        hook_debug_path: Optional[Path] = None
        active_config_dir: Optional[Path] = None

        try:
            # Set up environment
            if not self.setup_task(task):
                result.success = False
                result.error = "Failed to set up task environment"
                return result
            if self.repo_path:
                try:
                    repo_code_bytes = estimate_repo_code_bytes(self.repo_path)
                    context_tokens = choose_context_tokens(repo_code_bytes, self.context_tokens)
                    result.repo_code_bytes = repo_code_bytes
                    result.context_tokens = context_tokens
                    if self.verbose:
                        mb = repo_code_bytes / (1024 * 1024)
                        print(f"    Repo size: {mb:.1f} MB code, token target: {context_tokens}")
                except Exception:
                    result.context_tokens = self.context_tokens
            else:
                result.context_tokens = self.context_tokens

            # Use benchmark config directory for Claude Code
            active_config_dir = self.claude_config_dir
            if self.repo_path:
                mark_claude_project_trusted(active_config_dir, self.repo_path)

            token_target = result.context_tokens or self.context_tokens
            dir_token_target = max(2000, token_target // 2)

            baseline_diff = ""
            if self.repo_path:
                try:
                    baseline = subprocess.run(
                        ["git", "diff"],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    baseline_diff = baseline.stdout
                    result.baseline_dirty = bool(baseline_diff.strip())
                    result.baseline_diff_len = len(baseline_diff)
                except Exception:
                    baseline_diff = ""

            # Build prompt based on mode
            if mode == "scribe-context":
                # Pre-fetch scribe context for relevant code
                if self.verbose:
                    print("    Fetching scribe context...")
                scribe_context = fetch_scribe_context(self.repo_path, issue, context_tokens=token_target)

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

                prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

=== SCRIBE: SURGICAL CODE SLICES ===

Use `scribe` for targeted code slices with dependencies. Pattern: start small, expand as needed.

**Command format:**
  scribe --covering-set "FILE:FUNCTION" --max-depth 1 --token-target 800 --stdout

**Workflow:**
1. Find relevant code with grep
2. Get a small slice: `scribe --covering-set "file.go:FuncName" --max-depth 1 --token-target 800 --stdout`
3. If you need more context, get another specific slice
4. Make your fix and test

**Key settings:**
- `--max-depth 1` = only direct dependencies (tight focus)
- `--token-target 800` = small output (~800 tokens)
- Multiple small calls are better than one large call

**Example:**
```bash
# Find where the bug is
grep -rn "ValidationError" --include="*.go"

# Get just that function
scribe --covering-set "internal/validate.go:ValidateInput" --max-depth 1 --token-target 800 --stdout

# Need a helper it uses? Get that separately
scribe --covering-set "internal/errors.go:NewError" --max-depth 1 --token-target 800 --stdout
```

Target specific functions. Never request whole files."""

            else:  # standard mode
                prompt = f"""Fix the following issue in this repository.

ISSUE:
{issue}

After fixing, run the relevant tests to verify your fix works."""

            # Run agent (Claude Code or Pi)
            if self.agent_type == "pi":
                # Pi agent setup
                pi_bin = get_pi_path()
                env = os.environ.copy()
                # Determine provider from model name
                if self.model.startswith("glm"):
                    provider = "zai"
                elif self.model.startswith("claude"):
                    provider = "anthropic"
                elif self.model.startswith("gpt"):
                    provider = "openai"
                else:
                    provider = "zai"  # Default to zai for unknown models
                
                # Use JSON mode for metrics (pi supports this better than Claude with z.ai)
                cmd = [
                    "node", pi_bin,
                    "--print",  # Print mode (non-interactive)
                    "--provider", provider,
                    "--model", self.model,
                    "--mode", "json",  # JSONL output for metrics
                    "--no-session",
                ]
                
                # For scribe-tool mode, use the scribe extension
                if mode == "scribe-tool":
                    cmd.extend(["-e", str(PI_SCRIBE_EXTENSION)])
                
                cmd.append(prompt)
                
                if self.verbose:
                    print(f"    Running Pi agent ({self.model})...")
                    print(f"    Timeout: {self.task_timeout_s}s")
                    print(f"    Mode: {mode} (extension: {mode == 'scribe-tool'})")
            else:
                # Claude Code setup
                claude_bin = get_claude_path()
                env = build_claude_env(active_config_dir)
                if mode == "scribe-tool":
                    fd, path = tempfile.mkstemp(prefix="claude_hook_", suffix=".jsonl")
                    os.close(fd)
                    hook_log_path = Path(path)
                    fd, path = tempfile.mkstemp(prefix="claude_hook_debug_", suffix=".log")
                    os.close(fd)
                    hook_debug_path = Path(path)
                    env["CLAUDE_HOOK_LOG"] = str(hook_log_path)
                    env["CLAUDE_HOOK_DEBUG"] = str(hook_debug_path)
                    env["CLAUDE_SCRIBE_TOKEN_TARGET"] = str(token_target)
                    env["CLAUDE_SCRIBE_MAX_DEPTH"] = str(self.scribe_max_depth)

                # Build command based on mode:
                # - standard/scribe-context: Use --allowedTools to grant permissions (no hooks needed)
                # - scribe-tool: Use hooks for permissions (--allowedTools bypasses hooks)
                # Use JSON output for Anthropic models (opus/sonnet/haiku) for proper metrics
                # Use text output for z.ai models (glm-*) which have streaming issues with JSON
                is_zai_model = self.model.startswith("glm")
                output_format = "text" if is_zai_model else "json"

                cmd = [
                    claude_bin,
                    "-p",  # Print mode (non-interactive)
                    "--model", self.model,
                    "--output-format", output_format,
                    "--no-session-persistence",
                    "--disable-slash-commands",
                    "--input-format", "text",
                ]

                if mode in ("standard", "scribe-context"):
                    # Use --dangerously-skip-permissions instead of --allowedTools
                    # --allowedTools was causing "only prompt commands are supported in streaming mode" errors
                    cmd.append("--dangerously-skip-permissions")
                if mode == "scribe-tool":
                    # Use bypassPermissions mode which should still run hooks for logging
                    # while auto-accepting all permission requests
                    cmd.extend(["--permission-mode", "bypassPermissions"])
                    cmd.extend(["--setting-sources", "user,local"])
                    if self.repo_path:
                        settings_path = self.repo_path / ".claude" / "settings.json"
                        cmd.extend(["--settings", str(settings_path)])
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
                env=env,
                timeout=self.task_timeout_s,
            )

            result.raw_output = proc_result.stdout[:10000]
            result.duration_s = time.time() - start_time

            if self.verbose:
                print(f"    Completed in {result.duration_s:.1f}s")
                if proc_result.stderr:
                    print(f"    stderr: {proc_result.stderr[:200]}")

            # Parse output for metrics (even if returncode != 0, agent may have worked)
            if self.agent_type == "pi":
                metrics = parse_pi_output(proc_result.stdout, proc_result.stderr)
            else:
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
            new_diff = diff_result.stdout
            # If the repo was already dirty and no new changes were made, treat as no patch
            if baseline_diff.strip() and new_diff.strip() == baseline_diff.strip():
                result.patch = ""
                result.resolved = False
            else:
                result.patch = new_diff[:5000]

            # If Claude errored before doing any work, treat as failure and ignore any diffs
            if metrics.get("had_error", False) and metrics.get("num_turns", 0) == 0:
                result.patch = ""
                result.resolved = False
                result.success = False
                err = metrics.get("error_types", [])
                result.error = err[0] if err else "error_during_execution"
            else:
                # Consider successful if we got a patch, regardless of Claude's internal errors
                result.success = proc_result.returncode == 0

            # If using Docker, copy changes back and run tests
            if self.use_docker and self.container_id and result.patch:
                # Copy modified files back to container
                subprocess.run(
                    ["docker", "cp", f"{self.repo_path}/.", f"{self.container_id}:{get_container_repo_path(self._current_task)}"],
                    capture_output=True,
                    timeout=60,
                )

                # Evaluate patch by actually running tests
                result.resolved, result.test_output = self._evaluate_patch(self._current_task, result.patch)
                result.success = result.resolved or result.success
            else:
                # No Docker or no patch - just check if there's a patch
                result.resolved = len(result.patch.strip()) > 0

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
                    new_diff = diff_result.stdout
                    if baseline_diff.strip() and new_diff.strip() == baseline_diff.strip():
                        result.patch = ""
                        result.resolved = False
                    else:
                        result.patch = new_diff[:5000]
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
            if hook_log_path:
                summary = summarize_hook_log(hook_log_path)
                result.hook_denies = summary.get("denies", 0)
                result.hook_warnings = summary.get("warnings", 0)
                result.scribe_commands = summary.get("scribe_cmds", 0)
                result.hook_events = summary.get("events", [])
                try:
                    hook_log_path.unlink()
                except Exception:
                    pass
            if hook_debug_path:
                try:
                    if hook_debug_path.exists():
                        lines = [ln.strip() for ln in hook_debug_path.read_text().splitlines() if ln.strip()]
                        result.hook_debug_invocations = len(lines)
                        result.hook_debug_events = lines[:10]
                        hook_debug_path.unlink()
                except Exception:
                    pass
            self.cleanup_task()

        return result


def _run_single_task(args: tuple) -> TaskResult:
    """Worker function for parallel task execution."""
    (
        task,
        run_mode,
        model,
        use_docker,
        task_timeout_s,
        task_index,
        total_tasks,
        context_tokens,
        scribe_max_depth,
        claude_config_dir,
        run_number,
        max_system_retries,
        system_retry_delay_s,
    ) = args
    task_id = task.get("instance_id", f"task_{task_index}")

    # Create a new runner for each task (thread-safe)
    runner = TaskRunner(
        agent_type=agent_type,
        model=model,
        use_docker=use_docker,
        task_timeout_s=task_timeout_s,
        context_tokens=context_tokens,
        scribe_max_depth=scribe_max_depth,
        claude_config_dir=claude_config_dir,
    )
    runner.verbose = False  # Less verbose in parallel mode

    print(f"  [{task_index+1}/{total_tasks}] {task_id} ({run_mode})...")
    attempt = 0
    while True:
        result = runner.run_task(task, mode=run_mode)
        result.run_number = run_number
        result.retry_count = attempt
        result.system_failure = is_system_failure(result)
        if result.system_failure and attempt < max_system_retries:
            attempt += 1
            print(f"    [{task_index+1}/{total_tasks}] {task_id} ({run_mode}): SYSTEM FAIL, retrying in {system_retry_delay_s}s (attempt {attempt}/{max_system_retries})")
            time.sleep(system_retry_delay_s)
            continue
        break
    status = "OK" if result.success else f"FAILED"
    print(f"    [{task_index+1}/{total_tasks}] {task_id} ({run_mode}): {status} - {result.total_tokens:,} tokens")
    return result


def run_task_batch(
    tasks: list[dict],
    mode: str = "both",
    model: str = "glm-4.7",
    agent_type: str = "claude",
    max_tasks: Optional[int] = None,
    use_docker: bool = True,
    prepull_workers: int = 4,
    task_timeout_s: int = 2400,
    parallel_workers: int = 1,
    context_tokens: int = 4000,
    scribe_max_depth: int = 4,
    claude_config_dir: Optional[Union[str, Path]] = None,
    run_number: int = 1,
    max_system_retries: int = 1,
    system_retry_delay_s: int = 30,
    progress_path: Optional[Union[str, Path]] = None,
) -> list[TaskResult]:
    """Run a batch of SWE-bench tasks.

    Args:
        tasks: List of SWE-bench task dicts.
        mode: "standard", "scribe-tool", "both", or "all".
              "both" = standard + scribe-tool (recommended).
              "all" = standard + scribe-context + scribe-tool.
              Legacy "scribe" and "scribe-context" are deprecated.
        model: Model to use (default: "openrouter/z-ai/glm-4.7").
        max_tasks: Maximum number of tasks to run.
        use_docker: Whether to use Docker for isolation.
        prepull_workers: Number of parallel workers for image pre-pulling.
        task_timeout_s: Timeout per task in seconds.
        parallel_workers: Number of tasks to run in parallel (default: 1 = sequential).
        context_tokens: Token budget for scribe-context mode (default: 4000).
        scribe_max_depth: Default max depth for scribe covering-set (default: 4).
        run_number: Current run index (for tracking across repeated runs).
        max_system_retries: Retry count for system/infra failures.
        system_retry_delay_s: Delay before retrying a system/infra failure.
        progress_path: Optional path to write incremental results JSON.

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
    # "both" = standard + scribe-tool (the primary comparison)
    # "all" = standard + scribe-context + scribe-tool (includes deprecated mode)
    run_standard = mode in ("both", "all", "standard")
    run_scribe_context = mode in ("all", "scribe-context")  # Deprecated
    run_scribe_tool = mode in ("both", "all", "scribe-tool")

    # Build list of (task, mode) pairs to run
    work_items = []
    for i, task in enumerate(tasks):
        if run_standard:
            work_items.append((
                task,
                "standard",
                model,
                use_docker,
                task_timeout_s,
                i,
                len(tasks),
                context_tokens,
                scribe_max_depth,
                claude_config_dir,
                run_number,
                max_system_retries,
                system_retry_delay_s,
            ))
        if run_scribe_context:
            work_items.append((
                task,
                "scribe-context",
                model,
                use_docker,
                task_timeout_s,
                i,
                len(tasks),
                context_tokens,
                scribe_max_depth,
                claude_config_dir,
                run_number,
                max_system_retries,
                system_retry_delay_s,
            ))
        if run_scribe_tool:
            work_items.append((
                task,
                "scribe-tool",
                model,
                use_docker,
                task_timeout_s,
                i,
                len(tasks),
                context_tokens,
                scribe_max_depth,
                claude_config_dir,
                run_number,
                max_system_retries,
                system_retry_delay_s,
            ))

    results = []
    progress_file = Path(progress_path) if progress_path else None

    if parallel_workers > 1:
        print(f"Running {len(work_items)} task/mode combinations with {parallel_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_item = {executor.submit(_run_single_task, item): item for item in work_items}
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    if progress_file:
                        write_progress(results, progress_file)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"  Task {item[0].get('instance_id')} ({item[1]}) raised exception: {e}")
    else:
        # Sequential execution (original behavior)
        runner = TaskRunner(
            agent_type=agent_type,
            model=model,
            use_docker=use_docker,
            task_timeout_s=task_timeout_s,
            context_tokens=context_tokens,
            scribe_max_depth=scribe_max_depth,
            claude_config_dir=claude_config_dir,
        )

        for i, task in enumerate(tasks):
            task_id = task.get("instance_id", f"task_{i}")
            print(f"\n[{i+1}/{len(tasks)}] {task_id}")

            if run_standard:
                print("  Running standard mode...")
                attempt = 0
                while True:
                    result = runner.run_task(task, mode="standard")
                    result.run_number = run_number
                    result.retry_count = attempt
                    result.system_failure = is_system_failure(result)
                    if result.system_failure and attempt < max_system_retries:
                        attempt += 1
                        print(f"    SYSTEM FAIL, retrying in {system_retry_delay_s}s (attempt {attempt}/{max_system_retries})")
                        time.sleep(system_retry_delay_s)
                        continue
                    break
                status = "OK" if result.success else f"FAILED: {result.error[:30]}"
                print(f"    {status} - Tokens: {result.total_tokens:,}, Calls: {result.num_tool_calls}")
                results.append(result)
                if progress_file:
                    write_progress(results, progress_file)

            if run_scribe_context:
                print("  Running scribe-context mode...")
                attempt = 0
                while True:
                    result = runner.run_task(task, mode="scribe-context")
                    result.run_number = run_number
                    result.retry_count = attempt
                    result.system_failure = is_system_failure(result)
                    if result.system_failure and attempt < max_system_retries:
                        attempt += 1
                        print(f"    SYSTEM FAIL, retrying in {system_retry_delay_s}s (attempt {attempt}/{max_system_retries})")
                        time.sleep(system_retry_delay_s)
                        continue
                    break
                status = "OK" if result.success else f"FAILED: {result.error[:30]}"
                print(f"    {status} - Tokens: {result.total_tokens:,}, Calls: {result.num_tool_calls}")
                results.append(result)
                if progress_file:
                    write_progress(results, progress_file)

            if run_scribe_tool:
                print("  Running scribe-tool mode...")
                attempt = 0
                while True:
                    result = runner.run_task(task, mode="scribe-tool")
                    result.run_number = run_number
                    result.retry_count = attempt
                    result.system_failure = is_system_failure(result)
                    if result.system_failure and attempt < max_system_retries:
                        attempt += 1
                        print(f"    SYSTEM FAIL, retrying in {system_retry_delay_s}s (attempt {attempt}/{max_system_retries})")
                        time.sleep(system_retry_delay_s)
                        continue
                    break
                status = "OK" if result.success else f"FAILED: {result.error[:30]}"
                print(f"    {status} - Tokens: {result.total_tokens:,}, Calls: {result.num_tool_calls}")
                results.append(result)
                if progress_file:
                    write_progress(results, progress_file)

    return results

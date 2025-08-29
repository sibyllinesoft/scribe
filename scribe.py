#!/usr/bin/env python3
"""
Scribe: Advanced Repository Intelligence for LLM Code Analysis

Intelligently render repositories for LLM analysis with automatic file selection,
optimal token usage, and multiple output formats. Scribe automatically chooses
between intelligent selection and traditional filtering based on repository complexity.
"""

from __future__ import annotations
import argparse
import fnmatch
import html
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from dataclasses import dataclass
from typing import List, Optional, Set

# External deps
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename, TextLexer
import markdown
from tqdm import tqdm

# PackRepo integration
try:
    from packrepo.library import RepositoryPacker, PackRepoError
    from packrepo.packer.tokenizer import TokenizerType
    PACKREPO_AVAILABLE = True
except ImportError:
    PACKREPO_AVAILABLE = False
    RepositoryPacker = None
    PackRepoError = None

# Scribe FastPath integration
try:
    from packrepo.fastpath.integrated_v5 import FastPathEngine, create_fastpath_engine, get_variant_flag_configuration
    from packrepo.fastpath.fast_scan import FastScanner
    from packrepo.fastpath.types import FastPathVariant, ScribeConfig
    from packrepo.packer.tokenizer import estimate_tokens_scan_result
    FASTPATH_AVAILABLE = True
except ImportError as e:
    FASTPATH_AVAILABLE = False
    FastScanner = None
    create_fastpath_engine = None
    get_variant_flag_configuration = None
    FastPathVariant = None
    ScribeConfig = None
    FastPathEngine = None
    estimate_tokens_scan_result = None

MAX_DEFAULT_BYTES = 200 * 1024  # Increased from 50KB to 200KB for modern source files
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".ogg", ".flac",
    ".ttf", ".otf", ".eot", ".woff", ".woff2",
    ".so", ".dll", ".dylib", ".class", ".jar", ".exe", ".bin",
}
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd", ".mkdn"}

@dataclass
class RenderDecision:
    include: bool
    reason: str  # "ok" | "binary" | "too_large" | "ignored"

@dataclass
class FileInfo:
    path: pathlib.Path  # absolute path on disk
    rel: str            # path relative to repo root (slash-separated)
    size: int
    decision: RenderDecision
    content: Optional[str] = None  # File content when loaded
    token_estimate: Optional[int] = None  # Token count estimate


def run(cmd: List[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def parse_gitignore_patterns(repo_root: pathlib.Path) -> Set[str]:
    """Parse .gitignore files and return a set of normalized patterns."""
    patterns = set()
    
    # Add some essential patterns that should always be ignored even without .gitignore
    essential_patterns = {
        ".git/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "Thumbs.db",
    }
    patterns.update(essential_patterns)
    
    # Parse the main .gitignore file in the repo root
    main_gitignore = repo_root / ".gitignore"
    if main_gitignore.exists():
        try:
            with main_gitignore.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    # Remove negation patterns for simplicity (!)
                    if line.startswith("!"):
                        continue
                    patterns.add(line)
        except Exception:
            # If we can't read the .gitignore file, just skip it
            pass
    
    return patterns


def should_ignore_path(rel_path: str, patterns: Set[str]) -> bool:
    """Check if a relative path should be ignored based on gitignore patterns."""
    # Check each pattern
    for pattern in patterns:
        if match_gitignore_pattern(rel_path, pattern):
            return True
    return False


def match_gitignore_pattern(rel_path: str, pattern: str) -> bool:
    """Match a single gitignore pattern against a relative path."""
    # Handle directory patterns (ending with /)
    if pattern.endswith("/"):
        pattern_dir = pattern.rstrip("/")
        # For directory patterns, check if any directory in the path matches
        path_parts = rel_path.split("/")
        for i, part in enumerate(path_parts[:-1]):  # Exclude the filename from directory matching
            if fnmatch.fnmatch(part, pattern_dir):
                return True
        # Also check if it's a directory itself
        if fnmatch.fnmatch(rel_path, pattern_dir) and rel_path.endswith("/"):
            return True
        # Check if the path starts with this directory
        if rel_path.startswith(pattern_dir + "/"):
            return True
    else:
        # For file patterns, match against the full path and filename
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # Also check just the filename
        filename = rel_path.split("/")[-1]
        if fnmatch.fnmatch(filename, pattern):
            return True
        # Handle patterns that might be meant to match anywhere in the path
        if "/" not in pattern:
            path_parts = rel_path.split("/")
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
    
    return False


def git_clone(url: str, dst: str) -> None:
    run(["git", "clone", "--depth", "1", url, dst])


def git_head_commit(repo_dir: str) -> str:
    try:
        cp = run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
        return cp.stdout.strip()
    except Exception:
        return "(unknown)"


def bytes_human(n: int) -> str:
    """Human-readable bytes: 1 decimal for KiB and above, integer for B."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    i = 0
    while f >= 1024.0 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    if i == 0:
        return f"{int(f)} {units[i]}"
    else:
        return f"{f:.1f} {units[i]}"


def looks_binary(path: pathlib.Path) -> bool:
    ext = path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return True
        # Heuristic: try UTF-8 decode; if it hard-fails, likely binary
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False
    except Exception:
        # If unreadable, treat as binary to be safe
        return True


def decide_file(path: pathlib.Path, repo_root: pathlib.Path, max_bytes: int, ignore_patterns: Set[str]) -> FileInfo:
    rel = str(path.relative_to(repo_root)).replace(os.sep, "/")
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    
    # Check if the file should be ignored based on gitignore patterns
    if should_ignore_path(rel, ignore_patterns):
        return FileInfo(path, rel, size, RenderDecision(False, "ignored"))
    
    if size > max_bytes:
        return FileInfo(path, rel, size, RenderDecision(False, "too_large"))
    if looks_binary(path):
        return FileInfo(path, rel, size, RenderDecision(False, "binary"))
    return FileInfo(path, rel, size, RenderDecision(True, "ok"))


def should_use_intelligent_mode(repo_root: pathlib.Path) -> bool:
    """Determine if repository should use intelligent file selection."""
    if not FASTPATH_AVAILABLE:
        return False
    
    # Count files to estimate complexity
    try:
        result = run(["git", "ls-files"], cwd=str(repo_root), check=True)
        git_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        file_count = len([f for f in git_files if f.strip()])
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to filesystem count
        file_count = sum(1 for f in repo_root.rglob("*") if f.is_file() and not f.is_symlink())
    
    # Use intelligent mode for repos with more than 50 files
    return file_count > 50


def collect_files(repo_root: pathlib.Path, max_bytes: int) -> List[FileInfo]:
    """Collect files from the repository, preferring git ls-files if available."""
    infos: List[FileInfo] = []
    
    # Try to use git ls-files first (respects .gitignore automatically)
    try:
        result = run(["git", "ls-files"], cwd=str(repo_root), check=True)
        git_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        for rel_path in git_files:
            if not rel_path:  # Skip empty lines
                continue
            abs_path = repo_root / rel_path
            if abs_path.exists() and abs_path.is_file() and not abs_path.is_symlink():
                infos.append(decide_file_simple(abs_path, repo_root, max_bytes))
        
        print(f"✓ Using git ls-files: found {len(git_files)} tracked files", file=sys.stderr)
        return infos
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to filesystem walk if git is not available or not a git repo
        print("⚠️  Git not available, falling back to filesystem walk", file=sys.stderr)
        
        # Parse gitignore patterns for manual filtering
        ignore_patterns = parse_gitignore_patterns(repo_root)
        
        for p in sorted(repo_root.rglob("*")):
            if p.is_symlink():
                continue
            if p.is_file():
                infos.append(decide_file(p, repo_root, max_bytes, ignore_patterns))
        return infos


def decide_file_simple(path: pathlib.Path, repo_root: pathlib.Path, max_bytes: int) -> FileInfo:
    """Simplified file decision for git-tracked files (no ignore checking needed)."""
    rel = str(path.relative_to(repo_root)).replace(os.sep, "/")
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    
    if size > max_bytes:
        return FileInfo(path, rel, size, RenderDecision(False, "too_large"))
    if looks_binary(path):
        return FileInfo(path, rel, size, RenderDecision(False, "binary"))
    return FileInfo(path, rel, size, RenderDecision(True, "ok"))


def generate_tree_fallback(root: pathlib.Path) -> str:
    """Minimal tree-like output if `tree` command is missing."""
    lines: List[str] = []
    prefix_stack: List[str] = []

    def walk(dir_path: pathlib.Path, prefix: str = ""):
        entries = [e for e in dir_path.iterdir() if e.name != ".git"]
        entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
        for i, e in enumerate(entries):
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            lines.append(prefix + branch + e.name)
            if e.is_dir():
                extension = "    " if last else "│   "
                walk(e, prefix + extension)

    lines.append(root.name)
    walk(root)
    return "\n".join(lines)


def try_tree_command(root: pathlib.Path) -> str:
    try:
        cp = run(["tree", "-a", "."], cwd=str(root))
        return cp.stdout
    except Exception:
        return generate_tree_fallback(root)


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def render_markdown_text(md_text: str) -> str:
    return markdown.markdown(md_text, extensions=["fenced_code", "tables", "toc"])  # type: ignore


def highlight_code(text: str, filename: str, formatter: HtmlFormatter) -> str:
    try:
        lexer = get_lexer_for_filename(filename, stripall=False)
    except Exception:
        lexer = TextLexer(stripall=False)
    return highlight(text, lexer, formatter)


def slugify(path_str: str) -> str:
    # Simple slug: keep alnum, dash, underscore; replace others with '-'
    out = []
    for ch in path_str:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def generate_cxml_text(infos: List[FileInfo], repo_url: str, head: str, diff_content: Optional[str] = None) -> str:
    """Generate CXML format text for LLM consumption."""
    lines = ["<documents>"]
    lines.append(f"<!-- Repository: {repo_url} (HEAD: {head}) -->")
    
    rendered = [i for i in infos if i.decision.include]
    total_tokens = sum(i.token_estimate or 0 for i in rendered)
    lines.append(f"<!-- Files: {len(rendered)}, Estimated tokens: {total_tokens:,} -->")

    for index, i in enumerate(rendered, 1):
        lines.append(f'<document index="{index}">')
        lines.append(f"<source>{i.rel}</source>")
        if i.size or i.token_estimate:
            metadata_parts = []
            if i.size:
                metadata_parts.append(f"Size: {bytes_human(i.size)}")
            if i.token_estimate:
                metadata_parts.append(f"Tokens: ~{i.token_estimate}")
            lines.append(f"<!-- {', '.join(metadata_parts)} -->")
        lines.append("<document_content>")

        if i.content is not None:
            lines.append(html.escape(i.content))
        else:
            try:
                text = read_text(i.path)
                lines.append(html.escape(text))
            except Exception as e:
                lines.append(f"<!-- Failed to read: {str(e)} -->")

        lines.append("</document_content>")
        lines.append("</document>")

    # Add diff content if available
    if diff_content:
        lines.append("")
        lines.append('<document index="diffs">')
        lines.append("<source>Repository Diffs (Relevance Filtered)</source>")
        lines.append("<!-- Recent changes relevant to the specified context -->")
        lines.append("<document_content>")
        lines.append(html.escape(diff_content))
        lines.append("</document_content>")
        lines.append("</document>")

    lines.append("</documents>")
    return "\n".join(lines)


def generate_repomix_text(infos: List[FileInfo], repo_url: str, head: str, diff_content: Optional[str] = None) -> str:
    """Generate repomix format text for LLM consumption."""
    lines = []
    
    # Repomix header with metadata
    lines.append(f"Repository: {repo_url}")
    lines.append(f"Commit: {head}")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append("")
    
    rendered = [i for i in infos if i.decision.include]
    total_tokens = sum(i.token_estimate or 0 for i in rendered)
    lines.append(f"Files: {len(rendered)}")
    lines.append(f"Estimated tokens: {total_tokens:,}")
    lines.append("")
    
    for i in rendered:
        lines.append("---")
        lines.append(f"File: {i.rel}")
        if i.size:
            lines.append(f"Size: {bytes_human(i.size)}")
        if i.token_estimate:
            lines.append(f"Tokens: {i.token_estimate}")
        lines.append("")
        
        if i.content is not None:
            lines.append(i.content)
        else:
            try:
                content = read_text(i.path)
                lines.append(content)
            except Exception as e:
                lines.append(f"Failed to read: {str(e)}")
        lines.append("")
    
    # Add diff content if available
    if diff_content:
        lines.append("---")
        lines.append("File: Repository Diffs (Relevance Filtered)")
        lines.append("Type: Git diffs")
        lines.append("")
        lines.append(diff_content)
        lines.append("")
    
    return "\n".join(lines)


def build_html(repo_url: str, repo_dir: pathlib.Path, head_commit: str, infos: List[FileInfo], diff_content: Optional[str] = None) -> str:
    return "<html><body><h1>Simple HTML Placeholder</h1></body></html>"


def estimate_tokens_simple(text: str) -> int:
    """Simple token estimation (roughly 4 chars per token for English)."""
    return max(1, len(text) // 4)


def load_file_content(file_info: FileInfo) -> FileInfo:
    """Load content for a file and estimate tokens."""
    try:
        content = read_text(file_info.path)
        token_estimate = estimate_tokens_simple(content)
        return FileInfo(
            path=file_info.path,
            rel=file_info.rel,
            size=file_info.size,
            decision=file_info.decision,
            content=content,
            token_estimate=token_estimate
        )
    except Exception:
        return FileInfo(
            path=file_info.path,
            rel=file_info.rel,
            size=file_info.size,
            decision=RenderDecision(False, "read_error"),
            content=None,
            token_estimate=None
        )


def select_files_fastpath(
    repo_dir: pathlib.Path, 
    token_budget: int, 
    variant_str: str = "v5_integrated", 
    query_hint: str = "",
    entry_points: List[str] = None,
    entry_functions: List[str] = None,
    personalization_alpha: float = 0.15,
    include_diffs: bool = False,
    diff_commits: int = 10,
    diff_branch: str = "",
    diff_relevance_threshold: float = 0.1
) -> Tuple[List[FileInfo], Optional[str]]:
    """Use Scribe intelligent algorithms to select files within token budget with optional entry points and diffs."""
    if not FASTPATH_AVAILABLE:
        raise RuntimeError("Scribe intelligent selection not available")
    
    # Map variant string to enum
    variant_mapping = {
        'v1_baseline': FastPathVariant.V1_BASELINE,
        'v2_quotas': FastPathVariant.V2_QUOTAS,
        'v3_centrality': FastPathVariant.V3_CENTRALITY,
        'v4_demotion': FastPathVariant.V4_DEMOTION,
        'v5_integrated': FastPathVariant.V5_INTEGRATED,
    }
    variant = variant_mapping[variant_str]
    
    # Scan repository files
    scanner = FastScanner(repo_dir)
    scan_results = scanner.scan_repository()
    
    # Build entry points configuration
    processed_entry_points = []
    if entry_points:
        for ep in entry_points:
            from packrepo.fastpath.types import EntryPointSpec
            processed_entry_points.append(EntryPointSpec(file_path=ep))
    
    # Process entry functions (file.py:function_name format)
    if entry_functions:
        for ef in entry_functions:
            if ':' in ef:
                file_path, func_name = ef.split(':', 1)
                from packrepo.fastpath.types import EntryPointSpec
                processed_entry_points.append(EntryPointSpec(
                    file_path=file_path, 
                    function_name=func_name
                ))
    
    # Set up diff packing if requested
    diff_options = None
    if include_diffs:
        from packrepo.fastpath.types import DiffPackingOptions
        
        # Determine commit range or branch comparison
        commit_range = None
        branch_comparison = None
        
        if diff_branch:
            branch_comparison = diff_branch
        else:
            commit_range = f"HEAD~{diff_commits}..HEAD"
        
        diff_options = DiffPackingOptions(
            enabled=True,
            commit_range=commit_range,
            branch_comparison=branch_comparison,
            max_commits=diff_commits,
            relevance_threshold=diff_relevance_threshold
        )
    
    # Create Scribe configuration
    config = ScribeConfig(
        variant=variant,
        total_budget=token_budget,
        entry_points=processed_entry_points,
        personalization_alpha=personalization_alpha,
        diff_options=diff_options
    )
    
    # Execute enhanced Scribe selection
    result = execute_enhanced_fastpath(repo_dir, scan_results, config, query_hint)
    
    # Convert Scribe results back to FileInfo objects
    selected_infos = []
    for scan_result in result.selected_files:
        file_path = repo_dir / scan_result.stats.path
        file_info = FileInfo(
            path=file_path,
            rel=scan_result.stats.path,
            size=scan_result.stats.size_bytes,
            decision=RenderDecision(True, "scribe_selected"),
            content=None,  # Will be loaded later
            token_estimate=estimate_tokens_scan_result(scan_result) if estimate_tokens_scan_result else None
        )
        selected_infos.append(file_info)
    
    # Return files and optional diff content
    return selected_infos, result.diff_content


def execute_enhanced_fastpath(repo_dir, scan_results, config, query_hint=""):
    """Execute Scribe with enhanced features (entry points and diffs)."""
    # Create base Scribe engine
    engine = FastPathEngine()
    
    # If no entry points or diffs, use standard execution
    if not config.entry_points and not config.diff_options:
        return engine.execute_variant(scan_results, config, query_hint)
    
    # Enhanced execution with personalized centrality and diff packing
    from packrepo.fastpath.result_builder import create_result_builder
    result_builder = create_result_builder(config.variant)
    
    # Phase 1: Apply personalized centrality if entry points specified
    if config.entry_points:
        from packrepo.fastpath.personalized_centrality import create_personalized_calculator
        from packrepo.fastpath.personalized_centrality import EntryPoint
        
        # Convert EntryPointSpec to EntryPoint
        entry_points = []
        for ep_spec in config.entry_points:
            entry_points.append(EntryPoint(
                file_path=ep_spec.file_path,
                function_name=ep_spec.function_name,
                class_name=ep_spec.class_name,
                weight=ep_spec.weight,
                description=ep_spec.description
            ))
        
        # Calculate personalized centrality
        centrality_calc = create_personalized_calculator(
            entry_points=entry_points,
            personalization_alpha=config.personalization_alpha
        )
        centrality_scores = centrality_calc.calculate_personalized_centrality(scan_results)
        
        # Use centrality scores to influence selection
        for scan_result in scan_results:
            file_path = scan_result.stats.path
            centrality_score = centrality_scores.pagerank_scores.get(file_path, 0.0)
            
            # Boost heuristic scores based on centrality
            if hasattr(scan_result, 'heuristic_score'):
                scan_result.heuristic_score = (
                    scan_result.heuristic_score * (1 - config.centrality_weight) +
                    centrality_score * config.centrality_weight
                )
        
        result_builder.with_entry_point_stats({
            'num_entry_points': len(config.entry_points),
            'personalization_alpha': config.personalization_alpha,
            'avg_centrality_score': sum(centrality_scores.pagerank_scores.values()) / len(centrality_scores.pagerank_scores) if centrality_scores.pagerank_scores else 0
        })
    
    # Phase 2: Execute standard Scribe selection
    base_result = engine.execute_variant(scan_results, config, query_hint)
    
    # Phase 3: Add diff content if requested
    diff_content = None
    included_diffs = []
    
    if config.diff_options and config.diff_options.enabled:
        from packrepo.fastpath.diff_packer import create_diff_packer
        
        # Set up diff packer with relevance gating if entry points specified
        entry_points_for_diff = None
        if config.entry_points:
            entry_points_for_diff = [ep.file_path for ep in config.entry_points]
        
        diff_packer = create_diff_packer(
            repo_path=str(repo_dir),
            entry_points=entry_points_for_diff,
            commit_range=config.diff_options.commit_range,
            branch_comparison=config.diff_options.branch_comparison,
            max_commits=config.diff_options.max_commits,
            relevance_threshold=config.diff_options.relevance_threshold
        )
        
        # Extract and pack relevant diffs
        included_diffs, diff_content = diff_packer.pack_diffs(scan_results)
    
    # Build enhanced result
    enhanced_result = result_builder.with_selection(
        base_result.selected_files,
        base_result.total_files_considered
    ).with_budget(
        base_result.budget_allocated,
        base_result.budget_used
    ).with_performance(
        base_result.selection_time_ms,
        base_result.memory_usage_mb
    ).with_scores(
        base_result.heuristic_scores,
        base_result.final_scores
    ).with_diffs(
        included_diffs,
        diff_content
    ).build()
    
    return enhanced_result


def derive_temp_output_path(repo_url: str) -> pathlib.Path:
    """Derive a temporary output path from the repo URL."""
    # Extract repo name from URL like https://github.com/owner/repo or https://github.com/owner/repo.git
    parts = repo_url.rstrip('/').split('/')
    if len(parts) >= 2:
        repo_name = parts[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        filename = f"{repo_name}.html"
    else:
        filename = "repo.html"

    return pathlib.Path(tempfile.gettempdir()) / filename


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scribe: Intelligent repository analysis for LLM code consumption",
        epilog="""
Examples:
  %(prog)s                                                         # Intelligently process current directory
  %(prog)s /path/to/local/repo                                     # Process local directory
  %(prog)s https://github.com/user/repo                           # Process GitHub repository
  %(prog)s --output-format cxml                                    # CXML format for current directory
  %(prog)s --output-format repomix --token-target 30000           # Repomix format with 30K token limit
  %(prog)s --query-hint "authentication" --token-target 50000     # Focus on authentication-related code
  
  # Advanced options:
  %(prog)s --force-traditional --max-bytes 100000                 # Force traditional filtering
  %(prog)s --entry-points src/main.py api/routes.py               # Focus on specific entry points
  %(prog)s --include-diffs --diff-commits 5                       # Include recent git changes
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("repo_url", nargs="?", help="GitHub repo URL (https://github.com/owner/repo[.git]) or local directory path. If not provided, uses current directory.")
    ap.add_argument("-o", "--out", help="Output file path (default: uses config file setting or saves to current directory with auto-generated name)")
    ap.add_argument("--no-open", action="store_true", help="Don't open the HTML file in browser after generation (HTML mode only)")
    
    # Output format selection
    ap.add_argument("--output-format", choices=["html", "cxml", "repomix"], default="html",
                   help="Output format: 'html' for web page, 'cxml' for LLM consumption, 'repomix' for repomix format")
    
    # Token budget (replaces token-target)
    ap.add_argument("--token-target", "--token-budget", type=int, default=50000, dest="token_target",
                   help="Target token count for intelligent selection (default: 50000)")
    
    # Mode selection
    ap.add_argument("--force-traditional", action="store_true",
                   help="Force traditional file filtering instead of intelligent selection")
    ap.add_argument("--max-bytes", type=int, default=MAX_DEFAULT_BYTES, 
                   help="Max file size to include (bytes); larger files are skipped (default: 200KB)")
    
    # Intelligent selection options (organized into Advanced group)
    advanced_group = ap.add_argument_group('Advanced Options', 
                                         'Fine-tune intelligent selection behavior')
    
    if FASTPATH_AVAILABLE:
        advanced_group.add_argument("--algorithm", "--variant", default="v5_integrated", dest="algorithm",
                       choices=["v1_baseline", "v2_quotas", "v3_centrality", "v4_demotion", "v5_integrated"],
                       help="Selection algorithm (default: v5_integrated)")
        advanced_group.add_argument("--query-hint", default="",
                       help="Query hint to guide file selection (e.g., 'authentication', 'database')")
        advanced_group.add_argument("--show-metrics", action="store_true",
                       help="Show detailed performance and quality metrics")
        
        # Entry point relevance
        advanced_group.add_argument("--entry-points", nargs="*", default=[],
                       help="Focus on specific entry point files (e.g., 'src/main.py' 'api/routes.js')")
        advanced_group.add_argument("--entry-functions", nargs="*", default=[],
                       help="Focus on specific functions (format: 'file.py:function_name')")
        advanced_group.add_argument("--personalization-alpha", type=float, default=0.15,
                       help="Entry point focus strength (0.0-1.0, default: 0.15)")
        
        # Git integration
        advanced_group.add_argument("--include-diffs", action="store_true",
                       help="Include relevant Git diffs")
        advanced_group.add_argument("--diff-commits", type=int, default=10,
                       help="Number of recent commits to analyze (default: 10)")
        advanced_group.add_argument("--diff-branch", default="",
                       help="Compare with specific branch")
        advanced_group.add_argument("--diff-relevance-threshold", type=float, default=0.1,
                       help="Minimum relevance score for including diffs (default: 0.1)")
    
    args = ap.parse_args()

    # No validation needed - we'll automatically choose the best mode

    # Determine if we're working with a URL or local directory
    if args.repo_url is None:
        # Use current directory
        repo_url_for_display = f"file://{os.getcwd()}"
        repo_dir = pathlib.Path.cwd()
        is_local = True
        tmpdir = None
    elif args.repo_url.startswith(('http://', 'https://')):
        # It's a URL
        repo_url_for_display = args.repo_url
        tmpdir = tempfile.mkdtemp(prefix="rendergit_")
        repo_dir = pathlib.Path(tmpdir, "repo")
        is_local = False
    else:
        # It's a local path
        repo_path = pathlib.Path(args.repo_url)
        if not repo_path.exists():
            print(f"❌ Directory does not exist: {args.repo_url}", file=sys.stderr)
            return 1
        if not repo_path.is_dir():
            print(f"❌ Path is not a directory: {args.repo_url}", file=sys.stderr)
            return 1
        repo_url_for_display = f"file://{repo_path.resolve()}"
        repo_dir = repo_path
        is_local = True
        tmpdir = None

    # Load configuration from scribe.config.json if available
    config = None
    try:
        if PACKREPO_AVAILABLE:
            from packrepo.fastpath.config_manager import load_config
            config = load_config(repo_dir)
    except Exception:
        # If config loading fails, continue without config
        config = None

    # Set default output path if not provided
    if args.out is None:
        # Priority order: 1. CLI args, 2. Config file, 3. Current directory with auto-generated name
        if config and config.output_file_path:
            # Use the path from configuration
            args.out = str(pathlib.Path(config.output_file_path).expanduser().resolve())
        else:
            # Generate default filename in current directory
            if is_local:
                base_name = repo_dir.name
            else:
                base_name = derive_temp_output_path(args.repo_url).stem
            ext_map = {'html': '.html', 'cxml': '.xml', 'repomix': '.txt'}
            ext = ext_map.get(args.output_format, '.html')
            args.out = str(pathlib.Path.cwd() / f"{base_name}{ext}")

    try:
        # Phase 1: Repository preparation
        if is_local:
            head = git_head_commit(str(repo_dir))
            print(f"✅ Repository ready (HEAD: {head[:8] if head != '(unknown)' else 'no git'})", file=sys.stderr)
        else:
            print(f"📥 Cloning repository...", file=sys.stderr)
            git_clone(args.repo_url, str(repo_dir))
            head = git_head_commit(str(repo_dir))
            print(f"✅ Clone complete (HEAD: {head[:8]})", file=sys.stderr)

        # Phase 2: File Selection with automatic mode detection
        print(f"\n🎯 Phase 1: File Selection", file=sys.stderr)
        diff_content = None
        
        # Automatically choose between intelligent and traditional modes
        use_intelligent = not args.force_traditional and should_use_intelligent_mode(repo_dir)
        
        if use_intelligent:
            print(f"🧠 Using intelligent selection (algorithm: {getattr(args, 'algorithm', 'v5_integrated')})", file=sys.stderr)
            # Use Scribe intelligent selection with enhanced features
            try:
                selected_infos, diff_content = select_files_fastpath(
                    repo_dir, 
                    args.token_target, 
                    getattr(args, 'algorithm', 'v5_integrated'),
                    getattr(args, 'query_hint', ''),
                    entry_points=getattr(args, 'entry_points', []),
                    entry_functions=getattr(args, 'entry_functions', []),
                    personalization_alpha=getattr(args, 'personalization_alpha', 0.15),
                    include_diffs=getattr(args, 'include_diffs', False),
                    diff_commits=getattr(args, 'diff_commits', 10),
                    diff_branch=getattr(args, 'diff_branch', ''),
                    diff_relevance_threshold=getattr(args, 'diff_relevance_threshold', 0.1)
                )
                
                # Enhanced status message
                status_parts = [f"Selected {len(selected_infos)} files"]
                if getattr(args, 'entry_points', []) or getattr(args, 'entry_functions', []):
                    entry_count = len(getattr(args, 'entry_points', [])) + len(getattr(args, 'entry_functions', []))
                    status_parts.append(f"with {entry_count} entry points")
                if diff_content:
                    status_parts.append("including relevant diffs")
                status_parts.append(f"(target: {args.token_target:,} tokens)")
                
                print(f"✅ {' '.join(status_parts)}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  Intelligent selection failed: {e}", file=sys.stderr)
                print(f"🔄 Falling back to traditional filtering", file=sys.stderr)
                use_intelligent = False
        
        if not use_intelligent:
            print(f"🗂️  Using traditional file filtering (max size: {bytes_human(args.max_bytes)})", file=sys.stderr)
            all_infos = collect_files(repo_dir, args.max_bytes)
            selected_infos = [i for i in all_infos if i.decision.include]
            print(f"✅ Selected {len(selected_infos)} files after filtering", file=sys.stderr)

        if not selected_infos:
            print("❌ No files to process", file=sys.stderr)
            return 1

        # Phase 3: Content Loading with better progress
        print(f"\n📚 Phase 2: Content Loading", file=sys.stderr)
        loaded_infos = []
        total_tokens = 0
        
        with tqdm(selected_infos, desc="📄 Loading files", unit="file", file=sys.stderr) as pbar:
            for file_info in pbar:
                # Update progress bar description with current file
                filename = file_info.rel[-40:] if len(file_info.rel) > 40 else file_info.rel
                pbar.set_postfix_str(filename)
                
                loaded_info = load_file_content(file_info)
                if loaded_info.decision.include and loaded_info.content is not None:
                    loaded_infos.append(loaded_info)
                    total_tokens += loaded_info.token_estimate or 0
                elif not loaded_info.decision.include:
                    pbar.write(f"⚠️  Skipping {file_info.rel}: {loaded_info.decision.reason}")
        
        print(f"✅ Loaded {len(loaded_infos)} files (~{total_tokens:,} tokens)", file=sys.stderr)

        # Phase 4: Output Generation
        print(f"\n🔨 Phase 3: Output Generation", file=sys.stderr)
        print(f"🎨 Generating {args.output_format} format...", file=sys.stderr)
        
        if args.output_format == 'html':
            content = build_html(repo_url_for_display, repo_dir, head, loaded_infos, diff_content)
        elif args.output_format == 'cxml':
            content = generate_cxml_text(loaded_infos, repo_url_for_display, head, diff_content)
        elif args.output_format == 'repomix':
            content = generate_repomix_text(loaded_infos, repo_url_for_display, head, diff_content)
        else:
            print(f"❌ Unknown output format: {args.output_format}", file=sys.stderr)
            return 1

        # Write output
        out_path = pathlib.Path(args.out)
        print(f"💾 Writing to: {out_path.resolve()}", file=sys.stderr)
        
        # Ensure the parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        out_path.write_text(content, encoding="utf-8")
        file_size = out_path.stat().st_size
        print(f"✅ Complete! Wrote {bytes_human(file_size)} to {out_path.name}", file=sys.stderr)
        
        # Show configuration source info
        if config and config.output_file_path and args.out == str(pathlib.Path(config.output_file_path).expanduser().resolve()):
            print(f"📋 Output path from scribe.config.json", file=sys.stderr)

        # Show metrics if requested
        if use_intelligent and getattr(args, 'show_metrics', False):
            print(f"\n📊 Selection Metrics:", file=sys.stderr)
            print(f"  Selection method: {getattr(args, 'algorithm', 'v5_integrated')}", file=sys.stderr)
            print(f"  Token target: {args.token_target:,}", file=sys.stderr)
            print(f"  Actual tokens: ~{total_tokens:,}", file=sys.stderr)
            print(f"  Files selected: {len(loaded_infos)}", file=sys.stderr)
            if getattr(args, 'query_hint', ''):
                print(f"  Query hint: '{args.query_hint}'", file=sys.stderr)
        elif not use_intelligent and getattr(args, 'show_metrics', False):
            print(f"\n📊 Filtering Stats:", file=sys.stderr)
            print(f"  Max file size: {bytes_human(args.max_bytes)}", file=sys.stderr)
            print(f"  Files processed: {len(loaded_infos)}", file=sys.stderr)
            print(f"  Total tokens: ~{total_tokens:,}", file=sys.stderr)

        # Open HTML in browser if requested
        if args.output_format == 'html' and not args.no_open:
            print(f"🌐 Opening {out_path} in browser...", file=sys.stderr)
            webbrowser.open(f"file://{out_path.resolve()}")

        return 0

    finally:
        if tmpdir:
            print(f"🧹 Cleaning up temporary files...", file=sys.stderr)
            shutil.rmtree(tmpdir, ignore_errors=True)




if __name__ == "__main__":
    main()

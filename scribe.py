#!/usr/bin/env python3
"""
Scribe: Advanced Repository Intelligence for LLM Code Analysis

Render GitHub repositories into multiple formats with intelligent file selection.
Supports traditional file filtering, Scribe's intelligent selection algorithms, and multiple output formats
optimized for LLM consumption.
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
    formatter = HtmlFormatter(nowrap=False)
    pygments_css = formatter.get_style_defs('.highlight')

    # Stats
    rendered = [i for i in infos if i.decision.include]
    skipped_binary = [i for i in infos if i.decision.reason == "binary"]
    skipped_large = [i for i in infos if i.decision.reason == "too_large"]
    skipped_ignored = [i for i in infos if i.decision.reason == "ignored"]
    total_files = len(rendered) + len(skipped_binary) + len(skipped_large) + len(skipped_ignored)

    # Directory tree
    tree_text = try_tree_command(repo_dir)

    # Generate CXML text for LLM view
    cxml_text = generate_cxml_text(infos, repo_url, head_commit)

    # Table of contents
    toc_items: List[str] = []
    for i in rendered:
        anchor = slugify(i.rel)
        toc_items.append(
            f'<li><a href="#file-{anchor}">{html.escape(i.rel)}</a> '
            f'<span class="muted">({bytes_human(i.size)})</span></li>'
        )
    toc_html = "".join(toc_items)

    # Render file sections
    sections: List[str] = []
    for i in rendered:
        anchor = slugify(i.rel)
        p = i.path
        ext = p.suffix.lower()
        try:
            text = read_text(p)
            if ext in MARKDOWN_EXTENSIONS:
                body_html = render_markdown_text(text)
            else:
                code_html = highlight_code(text, i.rel, formatter)
                body_html = f'<div class="highlight">{code_html}</div>'
        except Exception as e:
            body_html = f'<pre class="error">Failed to render: {html.escape(str(e))}</pre>'
        sections.append(f"""
<section class="file-section" id="file-{anchor}">
  <h2>{html.escape(i.rel)} <span class="muted">({bytes_human(i.size)})</span></h2>
  <div class="file-body">{body_html}</div>
  <div class="back-top"><a href="#top">↑ Back to top</a></div>
</section>
""")

    # Skips lists
    def render_skip_list(title: str, items: List[FileInfo]) -> str:
        if not items:
            return ""
        lis = [
            f"<li><code>{html.escape(i.rel)}</code> "
            f"<span class='muted'>({bytes_human(i.size)})</span></li>"
            for i in items
        ]
        return (
            f"<details open><summary>{html.escape(title)} ({len(items)})</summary>"
            f"<ul class='skip-list'>\n" + "\n".join(lis) + "\n</ul></details>"
        )

    skipped_html = (
        render_skip_list("Skipped binaries", skipped_binary) +
        render_skip_list("Skipped large files", skipped_large)
    )

    # HTML with left sidebar TOC
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Flattened repo – {html.escape(repo_url)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, 'Apple Color Emoji','Segoe UI Emoji';
    margin: 0; padding: 0; line-height: 1.45;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 0 1rem; }}
  .meta small {{ color: #666; }}
  .counts {{ margin-top: 0.25rem; color: #333; }}
  .muted {{ color: #777; font-weight: normal; font-size: 0.9em; }}

  /* Layout with sidebar */
  .page {{ display: grid; grid-template-columns: 320px minmax(0,1fr); gap: 0; }}
  #sidebar {{
    position: sticky; top: 0; align-self: start;
    height: 100vh; overflow: auto;
    border-right: 1px solid #eee; background: #fafbfc;
  }}
  #sidebar .sidebar-inner {{ padding: 0.75rem; }}
  #sidebar h2 {{ margin: 0 0 0.5rem 0; font-size: 1rem; }}

  .toc {{ list-style: none; padding-left: 0; margin: 0; overflow-x: auto; }}
  .toc li {{ padding: 0.15rem 0; white-space: nowrap; }}
  .toc a {{ text-decoration: none; color: #0366d6; display: inline-block; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}

  main.container {{ padding-top: 1rem; }}

  pre {{ background: #f6f8fa; padding: 0.75rem; overflow: auto; border-radius: 6px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono','Courier New', monospace; }}
  .highlight {{ overflow-x: auto; }}
  .file-section {{ padding: 1rem; border-top: 1px solid #eee; }}
  .file-section h2 {{ margin: 0 0 0.5rem 0; font-size: 1.1rem; }}
  .file-body {{ margin-bottom: 0.5rem; }}
  .back-top {{ font-size: 0.9rem; }}
  .skip-list code {{ background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  .error {{ color: #b00020; background: #fff3f3; }}

  /* Hide duplicate top TOC on wide screens */
  .toc-top {{ display: block; }}
  @media (min-width: 1000px) {{ .toc-top {{ display: none; }} }}

  :target {{ scroll-margin-top: 8px; }}

  /* View toggle */
  .view-toggle {{
    margin: 1rem 0;
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }}
  .toggle-btn {{
    padding: 0.5rem 1rem;
    border: 1px solid #d1d9e0;
    background: white;
    cursor: pointer;
    border-radius: 6px;
    font-size: 0.9rem;
  }}
  .toggle-btn.active {{
    background: #0366d6;
    color: white;
    border-color: #0366d6;
  }}
  .toggle-btn:hover:not(.active) {{
    background: #f6f8fa;
  }}

  /* LLM view */
  #llm-view {{ display: none; }}
  #llm-text {{
    width: 100%;
    height: 70vh;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85em;
    border: 1px solid #d1d9e0;
    border-radius: 6px;
    padding: 1rem;
    resize: vertical;
  }}
  .copy-hint {{
    margin-top: 0.5rem;
    color: #666;
    font-size: 0.9em;
  }}

  /* Pygments */
  {pygments_css}
</style>
</head>
<body>
<a id="top"></a>

<div class="page">
  <nav id="sidebar"><div class="sidebar-inner">
      <h2>Contents ({len(rendered)})</h2>
      <ul class="toc toc-sidebar">
        <li><a href="#top">↑ Back to top</a></li>
        {toc_html}
      </ul>
  </div></nav>

  <main class="container">

    <section>
        <div class="meta">
        <div><strong>Repository:</strong> <a href="{html.escape(repo_url)}">{html.escape(repo_url)}</a></div>
        <small><strong>HEAD commit:</strong> {html.escape(head_commit)}</small>
        <div class="counts">
            <strong>Total files:</strong> {total_files} · <strong>Rendered:</strong> {len(rendered)} · <strong>Skipped:</strong> {len(skipped_binary) + len(skipped_large) + len(skipped_ignored)}
        </div>
        </div>
    </section>

    <div class="view-toggle">
      <strong>View:</strong>
      <button class="toggle-btn active" onclick="showHumanView()">👤 Human</button>
      <button class="toggle-btn" onclick="showLLMView()">🤖 LLM</button>
    </div>

    <div id="human-view">
      <section>
        <h2>Directory tree</h2>
        <pre>{html.escape(tree_text)}</pre>
      </section>

      <section class="toc-top">
        <h2>Table of contents ({len(rendered)})</h2>
        <ul class="toc">{toc_html}</ul>
      </section>

      <section>
        <h2>Skipped items</h2>
        {skipped_html}
      </section>

      {''.join(sections)}
      
      {f'''
      <section>
        <h2>Git Diffs</h2>
        <div class="file-body">
          <pre class="highlight">{html.escape(diff_content)}</pre>
        </div>
      </section>
      ''' if diff_content else ''}
    </div>

    <div id="llm-view">
      <section>
        <h2>🤖 LLM View - CXML Format</h2>
        <p>Copy the text below and paste it to an LLM for analysis:</p>
        <textarea id="llm-text" readonly>{html.escape(cxml_text)}</textarea>
        <div class="copy-hint">
          💡 <strong>Tip:</strong> Click in the text area and press Ctrl+A (Cmd+A on Mac) to select all, then Ctrl+C (Cmd+C) to copy.
        </div>
      </section>
    </div>
  </main>
</div>

<script>
function showHumanView() {{
  document.getElementById('human-view').style.display = 'block';
  document.getElementById('llm-view').style.display = 'none';
  document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
}}

function showLLMView() {{
  document.getElementById('human-view').style.display = 'none';
  document.getElementById('llm-view').style.display = 'block';
  document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');

  // Auto-select all text when switching to LLM view for easy copying
  setTimeout(() => {{
    const textArea = document.getElementById('llm-text');
    textArea.focus();
    textArea.select();
  }}, 100);
}}
</script>
</body>
</html>
"""


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
    """Use Scribe's intelligent algorithms to select files within token budget with optional entry points and diffs."""
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


def show_progress_bar(current: int, total: int, prefix: str = "Progress", suffix: str = "", length: int = 40) -> None:
    """Show a simple progress bar in the terminal."""
    if total == 0:
        return
    
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%) {suffix}', end='', file=sys.stderr)
    if current == total:
        print(file=sys.stderr)  # Newline when complete


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scribe: Render GitHub repositories with advanced intelligence for LLM analysis",
        epilog="""
Examples:
  %(prog)s                                                         # Process current directory with HTML output
  %(prog)s /path/to/local/repo                                     # Process local directory
  %(prog)s https://github.com/user/repo                           # Traditional HTML output from GitHub
  %(prog)s --output-format html https://github.com/user/repo      # Same as above
  %(prog)s --output-format cxml                                    # CXML format for current directory
  %(prog)s --output-format repomix --token-target 50000           # Repomix format with token limit for current directory
  
  # Scribe intelligent file selection:
  %(prog)s --use-fastpath                                          # HTML with Scribe intelligence for current directory
  %(prog)s --use-fastpath https://github.com/user/repo            # HTML with Scribe intelligence from GitHub
  %(prog)s --use-fastpath --fastpath-variant v5_integrated --output-format cxml /path/to/repo
  %(prog)s --use-fastpath --token-target 30000 --query-hint "authentication" --output-format repomix
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("repo_url", nargs="?", help="GitHub repo URL (https://github.com/owner/repo[.git]) or local directory path. If not provided, uses current directory.")
    ap.add_argument("-o", "--out", help="Output file path (default: uses config file setting or saves to current directory with auto-generated name)")
    ap.add_argument("--max-bytes", type=int, default=MAX_DEFAULT_BYTES, help="Max file size to include (bytes); larger files are skipped (traditional mode only)")
    ap.add_argument("--no-open", action="store_true", help="Don't open the HTML file in browser after generation (HTML mode only)")
    
    # Output format selection
    ap.add_argument("--output-format", choices=["html", "cxml", "repomix"], default="html",
                   help="Output format: 'html' for web page, 'cxml' for LLM consumption, 'repomix' for repomix format")
    
    # File selection method
    ap.add_argument("--use-fastpath", action="store_true", 
                   help="Use Scribe intelligent file selection instead of traditional filtering")
    ap.add_argument("--token-target", type=int, default=50000,
                   help="Target token count for intelligent selection modes (default: 50000)")
    
    # Scribe-specific options (only available if Scribe is installed)
    if FASTPATH_AVAILABLE:
        ap.add_argument("--fastpath-variant", default="v5_integrated",
                       choices=["v1_baseline", "v2_quotas", "v3_centrality", "v4_demotion", "v5_integrated"],
                       help="Scribe algorithm variant (default: v5_integrated)")
        ap.add_argument("--query-hint", default="",
                       help="Query hint for Scribe optimization (helps guide selection)")
        ap.add_argument("--show-fastpath-metrics", action="store_true",
                       help="Show detailed Scribe performance and quality metrics")
        
        # Entry point relevance (NEW)
        ap.add_argument("--entry-points", nargs="*", default=[],
                       help="Entry point files for personalized relevance (e.g., 'src/main.py' 'api/handler.js')")
        ap.add_argument("--entry-functions", nargs="*", default=[],
                       help="Specific functions to focus on (format: 'file.py:function_name')")
        ap.add_argument("--personalization-alpha", type=float, default=0.15,
                       help="Strength of entry point bias (0.0-1.0, default: 0.15)")
        
        # Diff packing (NEW)
        ap.add_argument("--include-diffs", action="store_true",
                       help="Include Git diffs in the repository pack")
        ap.add_argument("--diff-commits", type=int, default=10,
                       help="Number of recent commits to include (default: 10)")
        ap.add_argument("--diff-branch", default="",
                       help="Compare with specific branch (e.g., 'main..feature-branch')")
        ap.add_argument("--diff-relevance-threshold", type=float, default=0.1,
                       help="Minimum relevance score for including diffs (default: 0.1)")
    
    args = ap.parse_args()

    # Validate Scribe availability if requested
    if args.use_fastpath and not FASTPATH_AVAILABLE:
        print("❌ Scribe intelligent selection requested but not available. Install Scribe components or remove --use-fastpath", file=sys.stderr)
        return 1

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
        if is_local:
            print(f"📁 Processing local directory: {repo_dir}", file=sys.stderr)
            head = git_head_commit(str(repo_dir))
            print(f"✓ Local directory ready (HEAD: {head[:8] if head != '(unknown)' else 'no git'})", file=sys.stderr)
        else:
            print(f"📁 Cloning {args.repo_url} to temporary directory: {repo_dir}", file=sys.stderr)
            git_clone(args.repo_url, str(repo_dir))
            head = git_head_commit(str(repo_dir))
            print(f"✓ Clone complete (HEAD: {head[:8]})", file=sys.stderr)

        # Phase 1: File Selection  
        print(f"📊 Selecting files...", file=sys.stderr)
        diff_content = None
        if args.use_fastpath:
            # Use Scribe intelligent selection with enhanced features
            selected_infos, diff_content = select_files_fastpath(
                repo_dir, 
                args.token_target, 
                getattr(args, 'fastpath_variant', 'v5_integrated'),
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
            status_parts = [f"Scribe selected {len(selected_infos)} files"]
            if getattr(args, 'entry_points', []) or getattr(args, 'entry_functions', []):
                entry_count = len(getattr(args, 'entry_points', [])) + len(getattr(args, 'entry_functions', []))
                status_parts.append(f"with {entry_count} entry points")
            if diff_content:
                status_parts.append("including relevant diffs")
            status_parts.append(f"(target: {args.token_target} tokens)")
            
            print(f"✓ {' '.join(status_parts)}", file=sys.stderr)
        else:
            # Use traditional file filtering
            all_infos = collect_files(repo_dir, args.max_bytes)
            selected_infos = [i for i in all_infos if i.decision.include]
            print(f"✓ Traditional filtering selected {len(selected_infos)} files (max size: {bytes_human(args.max_bytes)})", file=sys.stderr)

        # Phase 2: Content Loading
        print(f"📄 Loading file contents...", file=sys.stderr)
        loaded_infos = []
        total_tokens = 0
        
        for i, file_info in enumerate(selected_infos):
            # Show progress bar
            show_progress_bar(i, len(selected_infos), "Loading", file_info.rel[-30:] if len(file_info.rel) > 30 else file_info.rel)
            
            loaded_info = load_file_content(file_info)
            if loaded_info.decision.include and loaded_info.content is not None:
                loaded_infos.append(loaded_info)
                total_tokens += loaded_info.token_estimate or 0
            elif not loaded_info.decision.include:
                print(f"\n⚠️  Skipping {file_info.rel}: {loaded_info.decision.reason}", file=sys.stderr)
        
        # Complete the progress bar
        show_progress_bar(len(selected_infos), len(selected_infos), "Loading", "Complete")
        
        print(f"✓ Loaded {len(loaded_infos)} files (~{total_tokens:,} tokens total)", file=sys.stderr)

        if not loaded_infos:
            print("❌ No files to process", file=sys.stderr)
            return 1

        # Phase 3: Output Generation
        print(f"🔨 Generating {args.output_format} output...", file=sys.stderr)
        
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
        print(f"💾 Writing output: {out_path.resolve()}", file=sys.stderr)
        
        # Ensure the parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        out_path.write_text(content, encoding="utf-8")
        file_size = out_path.stat().st_size
        print(f"✓ Wrote {bytes_human(file_size)} to {out_path}", file=sys.stderr)
        
        # Show configuration source info
        if config and config.output_file_path and args.out == str(pathlib.Path(config.output_file_path).expanduser().resolve()):
            print(f"📋 Output path from scribe.config.json", file=sys.stderr)

        # Show Scribe metrics if requested
        if args.use_fastpath and getattr(args, 'show_fastpath_metrics', False):
            print(f"\n📊 Scribe Metrics:", file=sys.stderr)
            print(f"  Selection method: {getattr(args, 'fastpath_variant', 'v5_integrated')}", file=sys.stderr)
            print(f"  Token target: {args.token_target:,}", file=sys.stderr)
            print(f"  Actual tokens: ~{total_tokens:,}", file=sys.stderr)
            print(f"  Files selected: {len(loaded_infos)}", file=sys.stderr)
            if getattr(args, 'query_hint', ''):
                print(f"  Query hint: '{args.query_hint}'", file=sys.stderr)

        # Open HTML in browser if requested
        if args.output_format == 'html' and not args.no_open:
            print(f"🌐 Opening {out_path} in browser...", file=sys.stderr)
            webbrowser.open(f"file://{out_path.resolve()}")

        return 0

    finally:
        if tmpdir:
            print(f"🗑️  Cleaning up temporary directory: {tmpdir}", file=sys.stderr)
            shutil.rmtree(tmpdir, ignore_errors=True)




if __name__ == "__main__":
    main()

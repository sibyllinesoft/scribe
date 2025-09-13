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
import re
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

# Import from our modules
from scribe.glob_patterns import (
    parse_comma_separated_globs, 
    should_include_path, 
    match_glob_pattern
)
from scribe.git_utils import (
    run, 
    parse_gitignore_patterns, 
    should_ignore_path, 
    match_gitignore_pattern,
    git_clone, 
    git_head_commit
)
from scribe.file_analysis import (
    RenderDecision, 
    FileInfo, 
    bytes_human, 
    looks_binary, 
    decide_file, 
    decide_file_simple,
    estimate_tokens_simple,
    load_file_content,
    collect_files
)
from scribe.tree_utils import (
    generate_tree_fallback, 
    try_tree_command
)
from scribe.output_formats import (
    slugify,
    get_file_icon,
    generate_cxml_text,
    generate_repomix_text,
    build_html,
    read_text,
    derive_temp_output_path
)
from scribe.fastpath import (
    should_use_intelligent_mode,
    select_files_fastpath,
    FASTPATH_AVAILABLE
)

# PackRepo integration
try:
    from packrepo.library import RepositoryPacker, PackRepoError
    from packrepo.packer.tokenizer import TokenizerType
    PACKREPO_AVAILABLE = True
except ImportError:
    PACKREPO_AVAILABLE = False
    RepositoryPacker = None
    PackRepoError = None

# FastPath configuration
try:
    from packrepo.fastpath.integrated_v5 import get_variant_flag_configuration
    FASTPATH_CONFIG_AVAILABLE = True
except ImportError:
    FASTPATH_CONFIG_AVAILABLE = False
    get_variant_flag_configuration = None

MAX_DEFAULT_BYTES = 200 * 1024  # Increased from 50KB to 200KB for modern source files











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
  %(prog)s --include "*.py,*.js" --exclude "*.test.*,node_modules/**"  # Filter by file patterns
  
  # Advanced options:
  %(prog)s --force-traditional --max-bytes 100000                 # Force traditional filtering
  %(prog)s --entry-points src/main.py api/routes.py               # Focus on specific entry points
  %(prog)s --include-diffs --diff-commits 5                       # Include recent git changes
  %(prog)s --editor --token-target 20000 --open                   # Interactive bundle editor
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("repo_url", nargs="?", help="GitHub repo URL (https://github.com/owner/repo[.git]) or local directory path. If not provided, uses current directory.")
    ap.add_argument("-o", "--out", help="Output file path (default: uses config file setting or saves to current directory with auto-generated name)")
    ap.add_argument("--open", action="store_true", help="Open the HTML file in browser after generation (HTML mode only)")
    ap.add_argument("--editor", action="store_true", help="Launch interactive bundle editor instead of generating static output")
    
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
    
    # File filtering options
    ap.add_argument("--include", type=str, default="",
                   help="Comma-separated glob patterns for files to include (e.g., '*.py,*.js,src/**')")
    ap.add_argument("--exclude", type=str, default="",
                   help="Comma-separated glob patterns for files to exclude (e.g., '*.test.js,build/**')")
    
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

    # Check if editor mode is requested
    if args.editor:
        # Import and launch bundle editor
        try:
            from scribe_editor import create_bundle_editor
            
            # Only support local directories for editor mode
            if args.repo_url and args.repo_url.startswith(('http://', 'https://')):
                print("❌ Editor mode only supports local repositories", file=sys.stderr)
                return 1
            
            repo_dir = pathlib.Path(args.repo_url or ".").resolve()
            if not repo_dir.exists() or not repo_dir.is_dir():
                print(f"❌ Directory does not exist or is not a directory: {repo_dir}", file=sys.stderr)
                return 1
            
            # Set default output for editor
            if args.out is None:
                args.out = f"{repo_dir.name}-bundle-editor.html"
            
            # Create bundle editor
            create_bundle_editor(
                repo_dir=repo_dir,
                output_path=pathlib.Path(args.out),
                max_bytes=args.max_bytes,
                use_intelligent=not args.force_traditional,
                token_target=args.token_target
            )
            
            # Open in browser if requested
            if args.open:
                print(f"🌐 Opening bundle editor in browser...", file=sys.stderr)
                webbrowser.open(f"file://{pathlib.Path(args.out).resolve()}")
            
            return 0
            
        except ImportError:
            print("❌ Bundle editor not available - scribe_editor.py not found", file=sys.stderr)
            return 1

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
        repo_dir = repo_path.resolve()
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

        # Parse include/exclude patterns
        include_patterns = parse_comma_separated_globs(args.include)
        exclude_patterns = parse_comma_separated_globs(args.exclude)
        
        # Show pattern information if patterns are provided
        if include_patterns or exclude_patterns:
            pattern_info = []
            if include_patterns:
                pattern_info.append(f"include: {', '.join(include_patterns)}")
            if exclude_patterns:
                pattern_info.append(f"exclude: {', '.join(exclude_patterns)}")
            print(f"📋 Using file patterns: {'; '.join(pattern_info)}", file=sys.stderr)

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
            all_infos = collect_files(repo_dir, args.max_bytes, include_patterns, exclude_patterns)
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
        if args.output_format == 'html' and args.open:
            print(f"🌐 Opening {out_path} in browser...", file=sys.stderr)
            webbrowser.open(f"file://{out_path.resolve()}")

        return 0

    finally:
        if tmpdir:
            print(f"🧹 Cleaning up temporary files...", file=sys.stderr)
            shutil.rmtree(tmpdir, ignore_errors=True)




if __name__ == "__main__":
    main()

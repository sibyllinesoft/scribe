#!/usr/bin/env python3
"""
Enhanced Scribe - Original scribe.py with intelligent defaults integration

This module enhances the original scribe with automatic intelligent defaults
while maintaining full backward compatibility.
"""

import argparse
import pathlib
import sys
import tempfile
import subprocess
import time
from typing import Optional, List

# Import original scribe functions (we'll modify them)
from scribe import (
    git_clone, git_head_commit, collect_files, build_html, 
    generate_cxml_text, generate_repomix_text, derive_temp_output_path,
    FileInfo, RenderDecision, bytes_human, MAX_DEFAULT_BYTES
)

# Import intelligent components
from repository_analyzer import RepositoryAnalyzer
from config_generator import ConfigGenerator
from smart_filter import SmartFilter
from intelligent_scribe import IntelligentScribe


def enhanced_collect_files(repo_root: pathlib.Path, smart_filter: SmartFilter) -> List[FileInfo]:
    """Enhanced file collection using smart filtering."""
    all_files = []
    
    # Collect all files
    for root, dirs, filenames in repo_root.walk():
        for filename in filenames:
            file_path = root / filename
            all_files.append(file_path)
    
    # Apply smart filtering
    included_files, filter_stats = smart_filter.filter_files(all_files, repo_root)
    
    # Convert to FileInfo objects
    file_infos = []
    for file_path in included_files:
        try:
            rel_path = file_path.relative_to(repo_root)
            size = file_path.stat().st_size
            
            file_info = FileInfo(
                path=file_path,
                rel=str(rel_path).replace('\\', '/'),  # Normalize path separators
                size=size,
                decision=RenderDecision(True, "smart_filter_included"),
                content=None,
                token_estimate=None
            )
            file_infos.append(file_info)
            
        except (OSError, PermissionError):
            continue
    
    return file_infos


def enhanced_main(args_override=None) -> int:
    """Enhanced main function with intelligent defaults."""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Scribe: Repository Intelligence with Automatic Optimization",
        epilog="""
Examples:
  %(prog)s https://github.com/user/repo                           # Auto-optimized HTML output  
  %(prog)s --intelligent https://github.com/user/repo             # Use intelligent defaults
  %(prog)s --intelligent --analyze-only /path/to/repo             # Analyze and show recommendations
  %(prog)s --output-format cxml --intelligent https://github.com/user/repo  # CXML with intelligent selection
  
Traditional mode (original scribe behavior):
  %(prog)s --traditional --max-bytes 1000000 https://github.com/user/repo
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Original arguments
    parser.add_argument("repo_url", help="GitHub repo URL or local path")
    parser.add_argument("-o", "--out", help="Output file path")
    parser.add_argument("--no-open", action="store_true", help="Don't open HTML in browser")
    parser.add_argument("--output-format", choices=["html", "cxml", "repomix"], default="html",
                       help="Output format")
    
    # Enhanced arguments
    parser.add_argument("--intelligent", action="store_true", default=True,
                       help="Use intelligent defaults (enabled by default)")
    parser.add_argument("--traditional", action="store_true", 
                       help="Use traditional scribe behavior (disables intelligent defaults)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze repository and show recommendations")
    parser.add_argument("--force-analysis", action="store_true",
                       help="Force re-analysis of repository")
    
    # Traditional mode arguments (for backward compatibility)
    parser.add_argument("--max-bytes", type=int, default=MAX_DEFAULT_BYTES,
                       help="Max file size in traditional mode")
    
    # Parse arguments
    if args_override:
        args = parser.parse_args(args_override)
    else:
        args = parser.parse_args()
    
    # Disable intelligent defaults if traditional mode is requested
    if args.traditional:
        args.intelligent = False
    
    # Determine if we're working with a local path or URL
    repo_path = pathlib.Path(args.repo_url)
    is_local = repo_path.exists()
    
    if args.intelligent:
        return _run_intelligent_mode(args, is_local)
    else:
        return _run_traditional_mode(args, is_local)


def _run_intelligent_mode(args, is_local: bool) -> int:
    """Run in intelligent mode with automatic optimization."""
    
    print("🧠 Running Enhanced Scribe with Intelligent Defaults")
    
    try:
        # Handle local vs remote repositories
        if is_local:
            repo_dir = pathlib.Path(args.repo_url).resolve()
            head_commit = git_head_commit(str(repo_dir)) if (repo_dir / ".git").exists() else "local"
            cleanup_temp = False
        else:
            # Clone repository to temporary directory
            tmpdir = tempfile.mkdtemp(prefix="enhanced_scribe_")
            repo_dir = pathlib.Path(tmpdir) / "repo"
            print(f"📁 Cloning {args.repo_url}...")
            git_clone(args.repo_url, str(repo_dir))
            head_commit = git_head_commit(str(repo_dir))
            cleanup_temp = True
        
        # Create intelligent scribe instance
        intelligent_scribe = IntelligentScribe()
        
        # Analyze repository and generate configuration
        print(f"🔍 Analyzing repository structure...")
        analysis, config = intelligent_scribe.analyze_and_configure(
            repo_dir, force_analysis=args.force_analysis
        )
        
        # Display analysis results
        intelligent_scribe._display_recommendations(analysis, config)
        
        # If analyze-only mode, stop here
        if args.analyze_only:
            print(f"📊 Analysis complete. Configuration saved to: {repo_dir / 'scribe.config.json'}")
            return 0
        
        # Collect files using intelligent filtering
        print(f"📄 Applying intelligent file selection...")
        smart_filter = SmartFilter(config.max_file_size)
        file_infos = enhanced_collect_files(repo_dir, smart_filter)
        
        # Load file contents and estimate tokens
        print(f"💾 Loading file contents...")
        loaded_infos = []
        total_tokens = 0
        
        for file_info in file_infos:
            try:
                with open(file_info.path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Estimate tokens
                token_estimate = max(1, len(content) // 4)
                
                # Check if adding this file would exceed budget
                if total_tokens + token_estimate > config.token_budget:
                    print(f"⚠️  Reached token budget limit, stopping file inclusion")
                    break
                
                file_info.content = content
                file_info.token_estimate = token_estimate
                loaded_infos.append(file_info)
                total_tokens += token_estimate
                
            except (OSError, UnicodeDecodeError, PermissionError):
                continue
        
        print(f"✅ Loaded {len(loaded_infos)} files (~{total_tokens:,} tokens)")
        
        # Generate output
        output_path = _determine_output_path(args, repo_dir, args.output_format)
        print(f"🔨 Generating {args.output_format} output...")
        
        if args.output_format == "html":
            content = build_html(args.repo_url, repo_dir, head_commit, loaded_infos)
        elif args.output_format == "cxml":
            content = generate_cxml_text(loaded_infos, args.repo_url, head_commit)
        elif args.output_format == "repomix":
            content = generate_repomix_text(loaded_infos, args.repo_url, head_commit)
        else:
            raise ValueError(f"Unknown output format: {args.output_format}")
        
        # Write output
        output_path.write_text(content, encoding='utf-8')
        file_size = output_path.stat().st_size
        
        print(f"✅ Success! Output written to: {output_path}")
        print(f"📊 File size: {bytes_human(file_size)}")
        print(f"🎯 Token utilization: {total_tokens / config.token_budget:.1%}")
        
        # Open in browser if HTML and not disabled
        if args.output_format == "html" and not args.no_open:
            import webbrowser
            webbrowser.open(f"file://{output_path.resolve()}")
        
        # Cleanup
        if cleanup_temp:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in intelligent mode: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _run_traditional_mode(args, is_local: bool) -> int:
    """Run in traditional mode (original scribe behavior)."""
    
    print("🔧 Running in Traditional Scribe Mode")
    
    # For traditional mode, we'll use the original scribe logic
    # This maintains full backward compatibility
    
    try:
        if is_local:
            repo_dir = pathlib.Path(args.repo_url).resolve()
            head_commit = git_head_commit(str(repo_dir)) if (repo_dir / ".git").exists() else "local"
            cleanup_temp = False
        else:
            tmpdir = tempfile.mkdtemp(prefix="scribe_traditional_")
            repo_dir = pathlib.Path(tmpdir) / "repo"
            print(f"📁 Cloning {args.repo_url}...")
            git_clone(args.repo_url, str(repo_dir))
            head_commit = git_head_commit(str(repo_dir))
            cleanup_temp = True
        
        # Use original file collection
        print(f"📄 Collecting files (traditional mode)...")
        all_infos = collect_files(repo_dir, args.max_bytes)
        rendered_infos = [info for info in all_infos if info.decision.include]
        
        print(f"📊 Found {len(all_infos)} files, including {len(rendered_infos)}")
        
        # Load content
        for info in rendered_infos:
            try:
                with open(info.path, 'r', encoding='utf-8', errors='replace') as f:
                    info.content = f.read()
                info.token_estimate = max(1, len(info.content) // 4)
            except Exception:
                info.decision = RenderDecision(False, "read_error")
        
        # Filter out files that failed to load
        final_infos = [info for info in rendered_infos if info.decision.include]
        
        # Generate output
        output_path = _determine_output_path(args, repo_dir, args.output_format)
        
        if args.output_format == "html":
            content = build_html(args.repo_url, repo_dir, head_commit, final_infos)
        elif args.output_format == "cxml":
            content = generate_cxml_text(final_infos, args.repo_url, head_commit)
        elif args.output_format == "repomix":
            content = generate_repomix_text(final_infos, args.repo_url, head_commit)
        
        output_path.write_text(content, encoding='utf-8')
        
        print(f"✅ Traditional mode output written to: {output_path}")
        
        # Cleanup
        if cleanup_temp:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in traditional mode: {e}")
        return 1


def _determine_output_path(args, repo_dir: pathlib.Path, output_format: str) -> pathlib.Path:
    """Determine the output file path."""
    
    if args.out:
        return pathlib.Path(args.out)
    
    # Auto-generate output path
    repo_name = repo_dir.name
    extensions = {"html": ".html", "cxml": ".xml", "repomix": ".txt"}
    extension = extensions.get(output_format, ".txt")
    
    if args.repo_url.startswith(("http://", "https://")):
        # For URLs, use temp directory
        temp_dir = pathlib.Path(tempfile.gettempdir())
        return temp_dir / f"{repo_name}{extension}"
    else:
        # For local paths, save alongside repository
        return repo_dir / f"{repo_name}_scribe{extension}"


if __name__ == "__main__":
    sys.exit(enhanced_main())
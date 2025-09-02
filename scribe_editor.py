#!/usr/bin/env python3
"""
Scribe Interactive Bundle Editor

Creates an interactive web interface for editing Scribe bundles.
Shows all files in the repository categorized by inclusion status,
allows adding/removing files from the bundle, and saves the modified bundle.
"""

import argparse
import html
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import webbrowser
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Any
import os

# Import from main scribe module
from scribe import (
    collect_files, FileInfo, RenderDecision, bytes_human, get_file_icon,
    git_head_commit, parse_gitignore_patterns, should_ignore_path,
    looks_binary, decide_file_simple, should_use_intelligent_mode,
    select_files_fastpath, MAX_DEFAULT_BYTES, read_text, estimate_tokens_simple
)

try:
    from intelligent_scribe import IntelligentScribe
    INTELLIGENT_AVAILABLE = True
except ImportError:
    INTELLIGENT_AVAILABLE = False

@dataclass
class BundleState:
    """Tracks the current state of the bundle being edited."""
    repo_dir: pathlib.Path
    all_files: List[FileInfo]  # All discovered files
    included_files: Set[str]   # Currently included relative paths
    excluded_categories: Dict[str, List[FileInfo]]  # Category -> files
    token_estimate: int = 0
    total_size: int = 0
    
    def get_stats(self):
        """Get current bundle statistics."""
        included_count = len(self.included_files)
        excluded_count = len(self.all_files) - included_count
        return {
            'included_count': included_count,
            'excluded_count': excluded_count,
            'total_count': len(self.all_files),
            'token_estimate': self.token_estimate,
            'total_size': self.total_size
        }

def categorize_files(all_files: List[FileInfo], selected_files: Set[str]) -> Dict[str, List[FileInfo]]:
    """
    Categorize files by their exclusion reason or inclusion status.
    
    Args:
        all_files: All files discovered in the repository
        selected_files: Set of relative paths that are currently selected
    
    Returns:
        Dictionary mapping categories to lists of files
    """
    categories = {
        'included': [],
        'didn_t_fit': [],  # Files excluded due to token/size constraints
        'binary': [],
        'too_large': [],
        'ignored': [],
        'other': []
    }
    
    for file_info in all_files:
        if file_info.rel in selected_files:
            categories['included'].append(file_info)
        else:
            # File is not selected - categorize by reason
            if file_info.decision.reason == 'binary':
                categories['binary'].append(file_info)
            elif file_info.decision.reason == 'too_large':
                categories['too_large'].append(file_info)
            elif file_info.decision.reason == 'ignored':
                categories['ignored'].append(file_info)
            elif file_info.decision.reason == 'ok' or not file_info.decision.include:
                # Files that would normally be included but aren't selected
                # These are likely files that "didn't fit" in the token budget
                categories['didn_t_fit'].append(file_info)
            else:
                categories['other'].append(file_info)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def build_interactive_html(
    repo_url: str, 
    repo_dir: pathlib.Path, 
    head_commit: str, 
    bundle_state: BundleState,
    diff_content: Optional[str] = None
) -> str:
    """Generate interactive HTML interface for bundle editing."""
    
    stats = bundle_state.get_stats()
    categories = categorize_files(bundle_state.all_files, bundle_state.included_files)
    
    # Start building HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scribe Bundle Editor: {repo_url}</title>
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
    <style>
        {get_editor_css()}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <a href="https://sibylline.dev" style="display: flex; align-items: center; gap: 12px; color: inherit; text-decoration: none;" target="_blank">
                    <img src="https://sibylline.dev/img/logo.svg" alt="Sibylline" style="width: 32px; height: 32px;">
                    Scribe Bundle Editor
                </a>
            </h1>
            <div class="meta">
                <div class="meta-item">
                    <i data-lucide="git-branch" class="icon"></i>
                    <span><strong>Repository:</strong> {html.escape(repo_url)}</span>
                </div>
                <div class="meta-item">
                    <i data-lucide="git-commit" class="icon"></i>
                    <span><strong>Commit:</strong> {html.escape(head_commit)}</span>
                </div>
                <div class="meta-item">
                    <i data-lucide="clock" class="icon"></i>
                    <span><strong>Last Updated:</strong> <span id="last-updated">{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</span></span>
                </div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">
                    <i data-lucide="check-circle" class="icon-lg"></i>
                    <span id="included-count">{stats['included_count']}</span>
                </div>
                <div class="stat-label">Included Files</div>
            </div>
            <div class="stat">
                <div class="stat-value">
                    <i data-lucide="x-circle" class="icon-lg"></i>
                    <span id="excluded-count">{stats['excluded_count']}</span>
                </div>
                <div class="stat-label">Excluded Files</div>
            </div>
            <div class="stat">
                <div class="stat-value">
                    <i data-lucide="hash" class="icon-lg"></i>
                    <span id="token-estimate">~{stats['token_estimate']:,}</span>
                </div>
                <div class="stat-label">Estimated Tokens</div>
            </div>
            <div class="stat">
                <div class="stat-value">
                    <i data-lucide="hard-drive" class="icon-lg"></i>
                    <span id="total-size">{bytes_human(stats['total_size'])}</span>
                </div>
                <div class="stat-label">Total Size</div>
            </div>
        </div>
        
        <div class="controls">
            <div class="export-section">
                <select id="export-format" class="format-select">
                    <option value="html">HTML Bundle</option>
                    <option value="cxml">CXML Format</option>
                    <option value="repomix">Repomix Format</option>
                </select>
                <button id="export-bundle" class="btn-primary">
                    <i data-lucide="download" class="icon"></i>
                    Export Bundle
                </button>
            </div>
            <div class="config-section">
                <button id="export-config" class="btn-secondary">
                    <i data-lucide="settings" class="icon"></i>
                    Export Config
                </button>
                <button id="save-json" class="btn-secondary">
                    <i data-lucide="save" class="icon"></i>
                    Save JSON
                </button>
            </div>
            <div class="selection-section">
                <button id="select-all-visible" class="btn-secondary">
                    <i data-lucide="check-square" class="icon"></i>
                    Select All Visible
                </button>
                <button id="deselect-all-visible" class="btn-secondary">
                    <i data-lucide="square" class="icon"></i>
                    Deselect All Visible
                </button>
            </div>
        </div>
        
        <div class="file-browser">
"""

    # Generate file categories
    for category_key, files in categories.items():
        if not files:
            continue
            
        category_title = category_key.replace('_', ' ').title()
        category_icon = get_category_icon(category_key)
        is_included = category_key == 'included'
        is_open = category_key in ['included', 'didn_t_fit']  # Open included and "didn't fit" by default
        
        html_content += f"""
            <div class="category" data-category="{category_key}">
                <div class="category-header {'open' if is_open else ''}" onclick="toggleCategory('{category_key}')">
                    <i data-lucide="{category_icon}" class="icon"></i>
                    <span class="category-title">{category_title}</span>
                    <span class="file-count">({len(files)} files)</span>
                    <i data-lucide="chevron-down" class="chevron icon"></i>
                </div>
                <div class="category-content {'open' if is_open else ''}">
        """
        
        # Build and render file tree for this category
        file_tree = build_file_tree(files)
        tree_html = render_file_tree_html(file_tree, bundle_state.included_files)
        html_content += tree_html
        
        html_content += """
                </div>
            </div>
        """
    
    # Add the JavaScript and closing HTML
    # Store bundle data for JavaScript access
    bundle_data = {
        'repo_url': repo_url,
        'repo_dir': str(bundle_state.repo_dir),
        'head_commit': head_commit,
        'diff_content': diff_content
    }
    
    html_content += f"""
        </div>
        
        <div class="status-bar">
            <span id="status-message">Ready</span>
        </div>
    </div>
    
    <script>
        // Bundle metadata for export functionality
        window.bundleData = {json.dumps(bundle_data)};
        {get_editor_javascript()}
    </script>
</body>
</html>"""
    
    return html_content

def get_category_icon(category: str) -> str:
    """Get appropriate icon for file category."""
    icons = {
        'included': 'check-circle',
        'didn_t_fit': 'clock',
        'binary': 'file-image',
        'too_large': 'file-x',
        'ignored': 'eye-off',
        'other': 'file'
    }
    return icons.get(category, 'file')

def build_file_tree(files: List[FileInfo]) -> Dict[str, Any]:
    """Build a hierarchical tree structure from a flat list of files."""
    tree = {'_directories': {}, '_files': []}
    
    for file_info in files:
        path_parts = file_info.rel.split('/')
        
        if len(path_parts) == 1:
            # File in root directory
            tree['_files'].append(file_info)
        else:
            # File in subdirectory - build the directory structure
            current_level = tree
            
            # Navigate/create the directory structure
            for part in path_parts[:-1]:  # All parts except the filename
                if part not in current_level['_directories']:
                    current_level['_directories'][part] = {'_directories': {}, '_files': []}
                current_level = current_level['_directories'][part]
            
            # Add the file to the final directory
            current_level['_files'].append(file_info)
    
    return tree

def render_file_tree_html(tree: Dict[str, Any], included_files: Set[str], prefix: str = "", level: int = 0) -> str:
    """Render a file tree as HTML with collapsible directories."""
    html_content = ""
    
    # Count files in a tree recursively
    def count_files_in_tree(t):
        count = len(t.get('_files', []))
        for subdir in t.get('_directories', {}).values():
            count += count_files_in_tree(subdir)
        return count
    
    # Render directories first
    for dir_name in sorted(tree.get('_directories', {}).keys()):
        dir_tree = tree['_directories'][dir_name]
        file_count = count_files_in_tree(dir_tree)
        
        if file_count > 0:  # Only show directories that contain files
            # Calculate how many files in this directory are selected
            def count_selected_in_tree(t, prefix_path):
                selected_count = 0
                for file_info in t.get('_files', []):
                    if file_info.rel in included_files:
                        selected_count += 1
                for subdir_name, subdir_tree in t.get('_directories', {}).items():
                    selected_count += count_selected_in_tree(subdir_tree, f"{prefix_path}{subdir_name}/")
                return selected_count
            
            selected_count = count_selected_in_tree(dir_tree, f"{prefix}{dir_name}/")
            dir_path = f"{prefix}{dir_name}/"
            
            # Determine checkbox state
            if selected_count == 0:
                checkbox_class = "square"
                dir_selected_class = ""
            elif selected_count == file_count:
                checkbox_class = "check-square" 
                dir_selected_class = "dir-selected"
            else:
                checkbox_class = "minus-square"  # Partially selected
                dir_selected_class = "dir-partial"
            
            html_content += f"""
            <div class="tree-directory" style="margin-left: {level * 20}px;">
                <div class="tree-directory-header {dir_selected_class}" data-dir-path="{html.escape(dir_path)}">
                    <div class="directory-checkbox" onclick="toggleDirectory(this.parentElement, event)">
                        <i data-lucide="{checkbox_class}" class="icon checkbox-icon"></i>
                    </div>
                    <div class="directory-expander" onclick="toggleDirectoryExpansion(this.parentElement)">
                        <i data-lucide="chevron-right" class="icon chevron-icon"></i>
                    </div>
                    <i data-lucide="folder" class="icon folder-icon"></i>
                    <span class="directory-name">{html.escape(dir_name)}</span>
                    <span class="file-count">({selected_count}/{file_count})</span>
                </div>
                <div class="tree-directory-content" style="display: none;">
                    {render_file_tree_html(dir_tree, included_files, f"{prefix}{dir_name}/", level + 1)}
                </div>
            </div>
            """
    
    # Render files
    for file_info in sorted(tree.get('_files', []), key=lambda f: f.rel):
        is_selected = file_info.rel in included_files
        file_icon = get_file_icon(file_info.rel)
        
        # Format file metadata using the same helper as the original
        file_meta = []
        if file_info.size:
            file_meta.append(bytes_human(file_info.size))
        if file_info.token_estimate:
            file_meta.append(f"~{file_info.token_estimate} tokens")
        
        file_meta_str = " • ".join(file_meta) if file_meta else ""
        reason_text = ""
        
        if not is_selected and file_info.decision.reason != "ok":
            reason_text = f" ({file_info.decision.reason})"
        
        indent = level * 20
        html_content += f"""
        <div class="file-item {'selected' if is_selected else ''}" 
             style="margin-left: {indent + 20}px;"
             data-path="{html.escape(file_info.rel)}"
             data-size="{file_info.size or 0}"
             data-tokens="{file_info.token_estimate or 0}">
            <div class="file-checkbox" onclick="toggleFile('{html.escape(file_info.rel, quote=True)}')">
                <i data-lucide="{'check-square' if is_selected else 'square'}" class="icon checkbox-icon"></i>
            </div>
            <div class="file-info">
                <div class="file-name">
                    <i data-lucide="{file_icon}" class="icon"></i>
                    <span class="path">{html.escape(pathlib.Path(file_info.rel).name)}</span>
                    <span class="reason">{reason_text}</span>
                </div>
                <div class="file-meta">{file_meta_str}</div>
            </div>
        </div>
        """
    
    return html_content

def get_editor_css() -> str:
    """Get CSS styles for the interactive editor."""
    return """
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2a2a2a;
            --bg-tertiary: #3a3a3a;
            --text-primary: #e5e5e5;
            --text-secondary: #b5b5b5;
            --text-muted: #888;
            --accent-primary: #4f9cf9;
            --accent-secondary: #6b7280;
            --border-color: #404040;
            --hover-color: #333333;
            --success-color: #22c55e;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-size: 14px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .header {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
            color: white;
            padding: 32px;
            position: relative;
            overflow: hidden;
        }
        
        .header h1 {
            margin: 0;
            font-size: 32px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            position: relative;
            z-index: 1;
        }
        
        .header .meta {
            margin-top: 20px;
            opacity: 0.9;
            font-size: 13px;
            position: relative;
            z-index: 1;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.08);
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .stats {
            background: var(--bg-tertiary);
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 24px;
        }
        
        .stat {
            text-align: center;
            padding: 20px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .stat-label {
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        
        .controls {
            background: var(--bg-tertiary);
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .export-section {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        
        .config-section {
            display: flex;
            gap: 8px;
        }
        
        .selection-section {
            display: flex;
            gap: 8px;
        }
        
        .format-select {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 14px;
            cursor: pointer;
            outline: none;
            transition: all 0.2s ease;
        }
        
        .format-select:hover {
            background: var(--hover-color);
        }
        
        .format-select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(79, 156, 249, 0.2);
        }
        
        .btn-primary, .btn-secondary {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
            white-space: nowrap;
        }
        
        .btn-primary {
            background: var(--accent-primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: #3d8bfd;
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .btn-secondary:hover {
            background: var(--hover-color);
            transform: translateY(-1px);
        }
        
        .file-browser {
            max-height: 70vh;
            overflow-y: auto;
            background: var(--bg-secondary);
        }
        
        .category {
            border-bottom: 1px solid var(--border-color);
        }
        
        .category:last-child {
            border-bottom: none;
        }
        
        .category-header {
            padding: 16px 24px;
            background: var(--bg-tertiary);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
            user-select: none;
        }
        
        .category-header:hover {
            background: var(--hover-color);
        }
        
        .category-title {
            flex: 1;
        }
        
        .file-count {
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 400;
        }
        
        .chevron {
            transition: transform 0.2s ease;
        }
        
        .category-header.open .chevron {
            transform: rotate(180deg);
        }
        
        .category-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        
        .category-content.open {
            max-height: 2000px;
        }
        
        .file-item {
            padding: 12px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .file-item:hover {
            background: var(--hover-color);
        }
        
        .file-item.selected {
            background: rgba(79, 156, 249, 0.1);
            border-left: 3px solid var(--accent-primary);
        }
        
        /* Tree structure styles */
        .tree-directory {
            margin-bottom: 4px;
        }
        
        .tree-directory-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 6px;
            margin-bottom: 2px;
            transition: all 0.2s ease;
        }
        
        .tree-directory-header:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .tree-directory-header .chevron-icon {
            transition: transform 0.2s ease;
            width: 16px;
            height: 16px;
            color: var(--text-muted);
        }
        
        .tree-directory-header.expanded .chevron-icon {
            transform: rotate(90deg);
        }
        
        .tree-directory-header .folder-icon {
            width: 16px;
            height: 16px;
            color: var(--accent-secondary);
        }
        
        .directory-name {
            font-weight: 500;
            color: var(--text-primary);
        }
        
        .file-count {
            color: var(--text-muted);
            font-size: 12px;
            margin-left: auto;
        }
        
        .tree-directory-content {
            margin-top: 4px;
            border-left: 1px solid rgba(255, 255, 255, 0.08);
            margin-left: 12px;
            padding-left: 8px;
        }
        
        .directory-checkbox {
            display: flex;
            align-items: center;
            padding: 2px;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.2s ease;
        }
        
        .directory-checkbox:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .directory-checkbox .checkbox-icon {
            width: 16px;
            height: 16px;
            color: var(--text-muted);
            transition: color 0.2s ease;
        }
        
        .directory-expander {
            display: flex;
            align-items: center;
            padding: 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        
        .directory-expander:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .tree-directory-header.dir-selected .directory-checkbox .checkbox-icon {
            color: var(--accent-primary);
        }
        
        .tree-directory-header.dir-partial .directory-checkbox .checkbox-icon {
            color: var(--warning-color);
        }
        
        .file-checkbox {
            color: var(--text-muted);
            transition: color 0.2s ease;
        }
        
        .file-item.selected .file-checkbox {
            color: var(--accent-primary);
        }
        
        .file-info {
            flex: 1;
            min-width: 0;
        }
        
        .file-name {
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            margin-bottom: 2px;
        }
        
        .file-name .path {
            color: var(--text-primary);
        }
        
        .file-name .reason {
            color: var(--text-muted);
            font-size: 11px;
        }
        
        .file-meta {
            font-size: 11px;
            color: var(--text-muted);
        }
        
        .status-bar {
            background: var(--bg-tertiary);
            padding: 8px 24px;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
        }
        
        .icon {
            width: 16px;
            height: 16px;
        }
        
        .icon-lg {
            width: 20px;
            height: 20px;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 12px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 24px;
            }
            
            .stats {
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                padding: 16px;
            }
            
            .controls {
                padding: 12px 16px;
                flex-direction: column;
                gap: 12px;
                align-items: stretch;
            }
            
            .export-section,
            .config-section,
            .selection-section {
                justify-content: center;
            }
            
            .file-item {
                padding: 10px 16px;
            }
        }
    """

def get_editor_javascript() -> str:
    """Get JavaScript for interactive functionality."""
    return """
        // Initialize Lucide icons
        lucide.createIcons();
        
        // Global state
        let bundleState = {
            includedFiles: new Set(),
            tokenEstimate: 0,
            totalSize: 0,
            allFileData: new Map() // Store file metadata
        };
        
        // Initialize bundle state from existing data
        function initializeBundleState() {
            const fileItems = document.querySelectorAll('.file-item');
            fileItems.forEach(item => {
                const path = item.dataset.path;
                const tokens = parseInt(item.dataset.tokens) || 0;
                const size = parseInt(item.dataset.size) || 0;
                const isSelected = item.classList.contains('selected');
                
                // Store file metadata
                bundleState.allFileData.set(path, {
                    path: path,
                    tokens: tokens,
                    size: size,
                    element: item.cloneNode(true), // Store original element
                    category: item.closest('.category').dataset.category,
                    originalCategory: item.closest('.category').dataset.category
                });
                
                if (isSelected) {
                    bundleState.includedFiles.add(path);
                    bundleState.tokenEstimate += tokens;
                    bundleState.totalSize += size;
                }
            });
            updateStats();
        }
        
        // Toggle category open/closed
        function toggleCategory(categoryKey) {
            const category = document.querySelector(`[data-category="${categoryKey}"]`);
            if (!category) return;
            
            const header = category.querySelector('.category-header');
            const content = category.querySelector('.category-content');
            
            header.classList.toggle('open');
            content.classList.toggle('open');
        }
        
        // Toggle directory selection (files within directory)
        function toggleDirectory(headerElement, event) {
            event.stopPropagation();
            
            const dirPath = headerElement.dataset.dirPath;
            const checkbox = headerElement.querySelector('.directory-checkbox .checkbox-icon');
            const isCurrentlySelected = checkbox.getAttribute('data-lucide') === 'check-square';
            
            // Get all files in this directory
            const filesInDirectory = [];
            for (const [filePath, fileData] of bundleState.allFileData) {
                if (filePath.startsWith(dirPath)) {
                    filesInDirectory.push(filePath);
                }
            }
            
            // Toggle all files in directory
            const targetState = !isCurrentlySelected;
            let statusMessage = '';
            
            for (const filePath of filesInDirectory) {
                const currentlyIncluded = bundleState.includedFiles.has(filePath);
                
                if (targetState && !currentlyIncluded) {
                    // Add to bundle
                    const fileData = bundleState.allFileData.get(filePath);
                    bundleState.includedFiles.add(filePath);
                    bundleState.tokenEstimate += fileData.tokens;
                    bundleState.totalSize += fileData.size;
                    moveFileToCategory(filePath, 'included');
                } else if (!targetState && currentlyIncluded) {
                    // Remove from bundle
                    const fileData = bundleState.allFileData.get(filePath);
                    bundleState.includedFiles.delete(filePath);
                    bundleState.tokenEstimate -= fileData.tokens;
                    bundleState.totalSize -= fileData.size;
                    moveFileToCategory(filePath, fileData.originalCategory);
                }
            }
            
            // Update directory checkbox state
            updateDirectoryCheckboxes();
            
            // Update stats and UI
            updateStats();
            updateLastModified();
            updateCategoryCounts();
            
            statusMessage = targetState ? 
                `Added ${filesInDirectory.length} files from ${dirPath}` :
                `Removed ${filesInDirectory.length} files from ${dirPath}`;
            setStatusMessage(statusMessage);
        }
        
        // Toggle directory expansion in file trees
        function toggleDirectoryExpansion(headerElement) {
            const content = headerElement.nextElementSibling;
            const chevron = headerElement.querySelector('.chevron-icon');
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                headerElement.classList.add('expanded');
            } else {
                content.style.display = 'none';
                headerElement.classList.remove('expanded');
            }
        }
        
        // Update directory checkbox states based on file selection
        function updateDirectoryCheckboxes() {
            const directories = document.querySelectorAll('.tree-directory-header[data-dir-path]');
            
            for (const dirHeader of directories) {
                const dirPath = dirHeader.dataset.dirPath;
                const checkbox = dirHeader.querySelector('.directory-checkbox .checkbox-icon');
                const fileCountSpan = dirHeader.querySelector('.file-count');
                
                // Count files in this directory
                let totalFiles = 0;
                let selectedFiles = 0;
                
                for (const [filePath, fileData] of bundleState.allFileData) {
                    if (filePath.startsWith(dirPath)) {
                        totalFiles++;
                        if (bundleState.includedFiles.has(filePath)) {
                            selectedFiles++;
                        }
                    }
                }
                
                // Update checkbox icon and class
                dirHeader.classList.remove('dir-selected', 'dir-partial');
                
                if (selectedFiles === 0) {
                    checkbox.setAttribute('data-lucide', 'square');
                } else if (selectedFiles === totalFiles) {
                    checkbox.setAttribute('data-lucide', 'check-square');
                    dirHeader.classList.add('dir-selected');
                } else {
                    checkbox.setAttribute('data-lucide', 'minus-square');
                    dirHeader.classList.add('dir-partial');
                }
                
                // Update file count display
                fileCountSpan.textContent = `(${selectedFiles}/${totalFiles})`;
                
                // Refresh lucide icons
                lucide.createIcons();
            }
        }
        
        // Toggle file selection and move between categories
        function toggleFile(filePath) {
            const fileData = bundleState.allFileData.get(filePath);
            if (!fileData) return;
            
            const isCurrentlySelected = bundleState.includedFiles.has(filePath);
            
            if (isCurrentlySelected) {
                // Remove from bundle - move to original category
                bundleState.includedFiles.delete(filePath);
                bundleState.tokenEstimate -= fileData.tokens;
                bundleState.totalSize -= fileData.size;
                
                moveFileToCategory(filePath, fileData.originalCategory);
                setStatusMessage(`Removed ${filePath} from bundle`);
            } else {
                // Add to bundle - move to included category
                bundleState.includedFiles.add(filePath);
                bundleState.tokenEstimate += fileData.tokens;
                bundleState.totalSize += fileData.size;
                
                moveFileToCategory(filePath, 'included');
                setStatusMessage(`Added ${filePath} to bundle`);
            }
            
            updateStats();
            updateLastModified();
            updateCategoryCounts();
            updateDirectoryCheckboxes();
        }
        
        // Move file to specified category
        function moveFileToCategory(filePath, targetCategory) {
            const fileData = bundleState.allFileData.get(filePath);
            if (!fileData) return;
            
            // Remove from current location
            const currentElement = document.querySelector(`[data-path="${filePath}"]`);
            if (currentElement) {
                currentElement.remove();
            }
            
            // Find or create target category
            let targetCategoryElement = document.querySelector(`[data-category="${targetCategory}"]`);
            if (!targetCategoryElement) {
                targetCategoryElement = createCategoryElement(targetCategory);
            }
            
            // Create new file element
            const fileElement = createFileElement(fileData, targetCategory === 'included');
            
            // Add to target category
            const categoryContent = targetCategoryElement.querySelector('.category-content');
            
            // Insert in alphabetical order
            const existingFiles = Array.from(categoryContent.querySelectorAll('.file-item'));
            let inserted = false;
            
            for (const existingFile of existingFiles) {
                const existingPath = existingFile.dataset.path;
                if (filePath.toLowerCase() < existingPath.toLowerCase()) {
                    categoryContent.insertBefore(fileElement, existingFile);
                    inserted = true;
                    break;
                }
            }
            
            if (!inserted) {
                categoryContent.appendChild(fileElement);
            }
            
            // Update file data category
            fileData.category = targetCategory;
            
            // Re-initialize icons
            lucide.createIcons();
        }
        
        // Create file element
        function createFileElement(fileData, isSelected) {
            const fileElement = fileData.element.cloneNode(true);
            
            // Update selection state
            if (isSelected) {
                fileElement.classList.add('selected');
                fileElement.querySelector('.checkbox-icon').setAttribute('data-lucide', 'check-square');
            } else {
                fileElement.classList.remove('selected');
                fileElement.querySelector('.checkbox-icon').setAttribute('data-lucide', 'square');
            }
            
            // Update onclick handler
            const checkbox = fileElement.querySelector('.file-checkbox');
            checkbox.setAttribute('onclick', `toggleFile('${fileData.path.replace(/'/g, "\\\\'")}'))`);
            
            return fileElement;
        }
        
        // Create category element if it doesn't exist
        function createCategoryElement(categoryKey) {
            const categoryTitles = {
                'included': 'Included',
                'didn_t_fit': "Didn't Fit",
                'binary': 'Binary',
                'too_large': 'Too Large', 
                'ignored': 'Ignored',
                'other': 'Other'
            };
            
            const categoryIcons = {
                'included': 'check-circle',
                'didn_t_fit': 'clock',
                'binary': 'file-image',
                'too_large': 'file-x',
                'ignored': 'eye-off',
                'other': 'file'
            };
            
            const categoryTitle = categoryTitles[categoryKey] || categoryKey.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase());
            const categoryIcon = categoryIcons[categoryKey] || 'file';
            const isOpen = ['included', 'didn_t_fit'].includes(categoryKey);
            
            const categoryHTML = `
                <div class="category" data-category="${categoryKey}">
                    <div class="category-header ${isOpen ? 'open' : ''}" onclick="toggleCategory('${categoryKey}')">
                        <i data-lucide="${categoryIcon}" class="icon"></i>
                        <span class="category-title">${categoryTitle}</span>
                        <span class="file-count">(0 files)</span>
                        <i data-lucide="chevron-down" class="chevron icon"></i>
                    </div>
                    <div class="category-content ${isOpen ? 'open' : ''}">
                    </div>
                </div>
            `;
            
            const parser = new DOMParser();
            const doc = parser.parseFromString(categoryHTML, 'text/html');
            const categoryElement = doc.querySelector('.category');
            
            // Insert in proper order (included first, then others)
            const fileBrowser = document.querySelector('.file-browser');
            const categoryOrder = ['included', 'didn_t_fit', 'binary', 'too_large', 'ignored', 'other'];
            const categoryIndex = categoryOrder.indexOf(categoryKey);
            
            let inserted = false;
            for (const existingCategory of fileBrowser.querySelectorAll('.category')) {
                const existingKey = existingCategory.dataset.category;
                const existingIndex = categoryOrder.indexOf(existingKey);
                
                if (categoryIndex < existingIndex) {
                    fileBrowser.insertBefore(categoryElement, existingCategory);
                    inserted = true;
                    break;
                }
            }
            
            if (!inserted) {
                fileBrowser.appendChild(categoryElement);
            }
            
            lucide.createIcons();
            return categoryElement;
        }
        
        // Update category file counts
        function updateCategoryCounts() {
            document.querySelectorAll('.category').forEach(category => {
                const fileCount = category.querySelectorAll('.file-item').length;
                const countElement = category.querySelector('.file-count');
                if (countElement) {
                    countElement.textContent = `(${fileCount} files)`;
                }
                
                // Hide empty categories except included
                const categoryKey = category.dataset.category;
                if (fileCount === 0 && categoryKey !== 'included') {
                    category.style.display = 'none';
                } else {
                    category.style.display = 'block';
                }
            });
        }
        
        // Update statistics display
        function updateStats() {
            const includedCount = bundleState.includedFiles.size;
            const totalFiles = bundleState.allFileData.size;
            const excludedCount = totalFiles - includedCount;
            
            document.getElementById('included-count').textContent = includedCount;
            document.getElementById('excluded-count').textContent = excludedCount;
            document.getElementById('token-estimate').textContent = `~${bundleState.tokenEstimate.toLocaleString()}`;
            document.getElementById('total-size').textContent = formatBytes(bundleState.totalSize);
        }
        
        // Format bytes for display
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }
        
        // Update last modified timestamp
        function updateLastModified() {
            const now = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
            document.getElementById('last-updated').textContent = now;
        }
        
        // Set status message
        function setStatusMessage(message) {
            document.getElementById('status-message').textContent = message;
            setTimeout(() => {
                document.getElementById('status-message').textContent = 'Ready';
            }, 3000);
        }
        
        // Select all visible files
        function selectAllVisible() {
            const visibleFiles = document.querySelectorAll('.category-content.open .file-item:not(.selected)');
            visibleFiles.forEach(item => {
                const path = item.dataset.path;
                toggleFile(path);
            });
            setStatusMessage(`Selected ${visibleFiles.length} files`);
        }
        
        // Deselect all visible files
        function deselectAllVisible() {
            const visibleFiles = document.querySelectorAll('.category-content.open .file-item.selected');
            visibleFiles.forEach(item => {
                const path = item.dataset.path;
                toggleFile(path);
            });
            setStatusMessage(`Deselected ${visibleFiles.length} files`);
        }
        
        // Export bundle in selected format
        async function exportBundle() {
            const format = document.getElementById('export-format').value;
            const includedFiles = Array.from(bundleState.includedFiles);
            
            if (includedFiles.length === 0) {
                setStatusMessage('No files selected for export');
                return;
            }
            
            setStatusMessage(`Preparing ${format.toUpperCase()} export...`);
            
            try {
                // Prepare the export data
                const exportData = {
                    format: format,
                    included_files: includedFiles,
                    bundle_metadata: window.bundleData,
                    stats: {
                        file_count: includedFiles.length,
                        token_estimate: bundleState.tokenEstimate,
                        total_size: bundleState.totalSize
                    },
                    timestamp: new Date().toISOString()
                };
                
                // Since we can't call the Python backend from pure JS, we'll export the file list
                // and instructions for the user to run the export command
                const instructions = generateExportInstructions(format, includedFiles);
                
                // Create download with instructions and file list
                const content = JSON.stringify({
                    instructions: instructions,
                    file_list: includedFiles,
                    export_data: exportData
                }, null, 2);
                
                const blob = new Blob([content], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `scribe-export-${format}.json`;
                a.click();
                URL.revokeObjectURL(url);
                
                setStatusMessage(`Export configuration saved! See instructions in downloaded file.`);
                
            } catch (error) {
                console.error('Export failed:', error);
                setStatusMessage('Export failed - check console for details');
            }
        }
        
        // Generate export instructions for the user
        function generateExportInstructions(format, includedFiles) {
            const repoDir = window.bundleData.repo_dir || '.';
            const fileListArg = includedFiles.map(f => `"${f}"`).join(' ');
            
            return {
                format: format,
                description: `Instructions to export your customized bundle in ${format.toUpperCase()} format`,
                steps: [
                    "1. Save this file list to use with scribe",
                    "2. Run one of the following commands:",
                    "",
                    "Option A - Use the file list directly:",
                    `python scribe.py "${repoDir}" --output-format ${format} --explicit-includes ${fileListArg}`,
                    "",
                    "Option B - Create a config file first:",
                    "1. Use 'Export Config' button to save scribe.config.json",
                    "2. Place scribe.config.json in your repository root",
                    `3. Run: python scribe.py "${repoDir}" --output-format ${format}`,
                    "",
                    "Option C - Generate bundle with current token estimate:",
                    `python scribe.py "${repoDir}" --output-format ${format} --token-target ${bundleState.tokenEstimate}`
                ],
                file_count: includedFiles.length,
                token_estimate: bundleState.tokenEstimate
            };
        }
        
        // Save bundle (export file list as JSON)
        function saveJSON() {
            const includedFiles = Array.from(bundleState.includedFiles);
            const bundleData = {
                timestamp: new Date().toISOString(),
                included_files: includedFiles,
                stats: {
                    file_count: includedFiles.length,
                    token_estimate: bundleState.tokenEstimate,
                    total_size: bundleState.totalSize
                },
                bundle_metadata: window.bundleData
            };
            
            // Create and download JSON file
            const dataStr = JSON.stringify(bundleData, null, 2);
            const blob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'scribe-bundle-selection.json';
            a.click();
            URL.revokeObjectURL(url);
            
            setStatusMessage(`Bundle selection saved with ${includedFiles.length} files`);
        }
        
        // Export configuration
        function exportConfig() {
            const includedFiles = Array.from(bundleState.includedFiles);
            const config = {
                version: "1.0",
                include_patterns: [],
                exclude_patterns: [],
                explicit_includes: includedFiles,
                explicit_excludes: [],
                max_tokens: bundleState.tokenEstimate,
                created: new Date().toISOString()
            };
            
            const dataStr = JSON.stringify(config, null, 2);
            const blob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'scribe.config.json';
            a.click();
            URL.revokeObjectURL(url);
            
            setStatusMessage('Configuration exported - place in repo root for automatic use');
        }
        
        // Event listeners
        document.getElementById('export-bundle').addEventListener('click', exportBundle);
        document.getElementById('save-json').addEventListener('click', saveJSON);
        document.getElementById('export-config').addEventListener('click', exportConfig);
        document.getElementById('select-all-visible').addEventListener('click', selectAllVisible);
        document.getElementById('deselect-all-visible').addEventListener('click', deselectAllVisible);
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', () => {
            initializeBundleState();
            updateCategoryCounts();
        });
        
        // Global functions for HTML onclick handlers
        window.toggleCategory = toggleCategory;
        window.toggleFile = toggleFile;
    """

def create_bundle_editor(repo_dir: pathlib.Path, output_path: pathlib.Path, 
                        max_bytes: int = MAX_DEFAULT_BYTES, 
                        use_intelligent: bool = True,
                        token_target: int = 50000) -> BundleState:
    """
    Create a bundle editor for the given repository.
    
    Args:
        repo_dir: Path to the repository
        output_path: Where to save the HTML file
        max_bytes: Maximum file size to consider
        use_intelligent: Whether to use intelligent selection initially
        token_target: Token budget for intelligent selection
    
    Returns:
        BundleState object with current bundle configuration
    """
    print(f"📁 Analyzing repository: {repo_dir}")
    
    # Collect all files in the repository
    all_files = collect_files(repo_dir, max_bytes)
    print(f"🔍 Found {len(all_files)} files")
    
    # Determine initial selection
    selected_files = set()
    diff_content = None
    
    if use_intelligent and should_use_intelligent_mode(repo_dir):
        print(f"🧠 Using intelligent selection (target: {token_target:,} tokens)")
        try:
            selected_infos, diff_content = select_files_fastpath(repo_dir, token_target)
            selected_files = {info.rel for info in selected_infos}
            print(f"✅ Intelligent selection complete: {len(selected_files)} files selected")
        except Exception as e:
            print(f"⚠️ Intelligent selection failed: {e}")
            print("📋 Falling back to traditional filtering")
            use_intelligent = False
    
    if not use_intelligent:
        # Use traditional filtering
        selected_files = {info.rel for info in all_files if info.decision.include}
        print(f"📋 Traditional filtering: {len(selected_files)} files selected")
    
    # Calculate token estimates for all files
    for file_info in all_files:
        if file_info.token_estimate is None:
            # Calculate token estimate if not already available
            if file_info.content:
                file_info.token_estimate = estimate_tokens_simple(file_info.content)
            elif file_info.size and not looks_binary(file_info.path):
                # Estimate tokens for text files based on size
                file_info.token_estimate = file_info.size // 4  # Rough estimate: 4 chars per token
            else:
                file_info.token_estimate = 0
    
    # Calculate initial statistics for selected files
    token_estimate = 0
    total_size = 0
    
    for file_info in all_files:
        if file_info.rel in selected_files:
            token_estimate += file_info.token_estimate or 0
            total_size += file_info.size or 0
    
    # Create bundle state
    bundle_state = BundleState(
        repo_dir=repo_dir,
        all_files=all_files,
        included_files=selected_files,
        excluded_categories={},
        token_estimate=token_estimate,
        total_size=total_size
    )
    
    # Generate HTML
    repo_url = f"file://{repo_dir.resolve()}"
    head_commit = git_head_commit(str(repo_dir))
    
    html_content = build_interactive_html(
        repo_url, repo_dir, head_commit, bundle_state, diff_content
    )
    
    # Write HTML file
    print(f"💾 Writing interactive bundle editor to: {output_path}")
    output_path.write_text(html_content, encoding='utf-8')
    
    return bundle_state

def main():
    """Main entry point for the bundle editor."""
    parser = argparse.ArgumentParser(
        description="Scribe Interactive Bundle Editor - Edit repository bundles with a web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("repo_path", nargs="?", default=".",
                       help="Path to repository (default: current directory)")
    parser.add_argument("-o", "--output", default="scribe-bundle-editor.html",
                       help="Output HTML file path (default: scribe-bundle-editor.html)")
    parser.add_argument("--max-bytes", type=int, default=MAX_DEFAULT_BYTES,
                       help=f"Maximum file size to consider (default: {MAX_DEFAULT_BYTES})")
    parser.add_argument("--token-target", type=int, default=50000,
                       help="Initial token budget for intelligent selection (default: 50000)")
    parser.add_argument("--force-traditional", action="store_true",
                       help="Skip intelligent selection and use traditional filtering")
    parser.add_argument("--open", action="store_true",
                       help="Open the HTML file in browser after generation")
    
    args = parser.parse_args()
    
    # Validate repository path
    repo_dir = pathlib.Path(args.repo_path).resolve()
    if not repo_dir.exists():
        print(f"❌ Directory does not exist: {repo_dir}")
        return 1
    
    if not repo_dir.is_dir():
        print(f"❌ Path is not a directory: {repo_dir}")
        return 1
    
    # Set output path
    output_path = pathlib.Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create bundle editor
        bundle_state = create_bundle_editor(
            repo_dir=repo_dir,
            output_path=output_path,
            max_bytes=args.max_bytes,
            use_intelligent=not args.force_traditional,
            token_target=args.token_target
        )
        
        stats = bundle_state.get_stats()
        print(f"✅ Bundle editor created successfully!")
        print(f"   📊 {stats['included_count']} files included, {stats['excluded_count']} excluded")
        print(f"   🔢 ~{stats['token_estimate']:,} tokens, {bytes_human(stats['total_size'])}")
        print(f"   📄 Saved to: {output_path}")
        
        # Open in browser if requested
        if args.open:
            print(f"🌐 Opening in browser...")
            webbrowser.open(f"file://{output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating bundle editor: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
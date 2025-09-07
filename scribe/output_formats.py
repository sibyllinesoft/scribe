#!/usr/bin/env python3
"""Output format generation utilities."""

import html
import pathlib
import tempfile
from typing import List, Optional

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename, TextLexer

from .file_analysis import bytes_human

from .file_analysis import FileInfo


def slugify(path_str: str) -> str:
    """Simple slug: keep ASCII alnum, dash, underscore; replace others with '-'."""
    out = []
    for ch in path_str:
        if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def get_file_icon(file_path: str) -> str:
    """Return appropriate Lucide icon for file type."""
    ext = pathlib.Path(file_path).suffix.lower()
    name = pathlib.Path(file_path).name.lower()
    
    # Special files
    if name in ['readme.md', 'readme.txt', 'readme']:
        return 'book-open'
    elif name in ['license', 'licence']:
        return 'scale'
    elif name in ['dockerfile', 'docker-compose.yml', 'docker-compose.yaml']:
        return 'box'
    elif name in ['makefile']:
        return 'settings'
    elif name.startswith('.git'):
        return 'git-branch'
    elif name in ['package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml']:
        return 'package'
    elif name in ['tsconfig.json', 'jsconfig.json']:
        return 'settings'
    elif name in ['requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg']:
        return 'package'
    elif name in ['cargo.toml', 'cargo.lock']:
        return 'package'
    elif name in ['go.mod', 'go.sum']:
        return 'package'
    elif name in ['.env', '.envrc']:
        return 'key'
    elif name.endswith('.config.js') or name.endswith('.config.ts') or name.endswith('.config.json'):
        return 'settings'
    
    # Extensions
    if ext in ['.py', '.pyw']:
        return 'file-code'
    elif ext in ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']:
        return 'file-code'
    elif ext in ['.html', '.htm', '.xml', '.xhtml']:
        return 'globe'
    elif ext in ['.css', '.scss', '.sass', '.less', '.styl']:
        return 'palette'
    elif ext in ['.json', '.jsonc', '.json5']:
        return 'braces'
    elif ext in ['.yml', '.yaml']:
        return 'list'
    elif ext in ['.md', '.markdown', '.mdx']:
        return 'file-text'
    elif ext in ['.txt', '.text']:
        return 'file-text'
    elif ext in ['.rs']:
        return 'file-code'
    elif ext in ['.go']:
        return 'file-code'
    elif ext in ['.java', '.kt', '.scala', '.groovy']:
        return 'file-code'
    elif ext in ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx']:
        return 'file-code'
    elif ext in ['.cs', '.fs', '.vb']:
        return 'file-code'
    elif ext in ['.php', '.rb', '.pl', '.pm', '.r', '.swift', '.dart']:
        return 'file-code'
    elif ext in ['.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd']:
        return 'terminal'
    elif ext in ['.sql', '.sqlite', '.db']:
        return 'database'
    elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico']:
        return 'image'
    elif ext in ['.pdf']:
        return 'file-text'
    elif ext in ['.zip', '.tar', '.gz', '.bz2', '.7z', '.rar']:
        return 'archive'
    elif ext in ['.env', '.envrc']:
        return 'key'
    else:
        return 'file'


def _detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext = pathlib.Path(file_path).suffix.lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.html': 'html', '.htm': 'html', '.xhtml': 'html',
        '.css': 'css', '.scss': 'scss', '.sass': 'sass', '.less': 'less',
        '.json': 'json', '.jsonc': 'json', '.json5': 'json',
        '.xml': 'xml', '.xsl': 'xml', '.xsd': 'xml',
        '.yml': 'yaml', '.yaml': 'yaml',
        '.md': 'markdown', '.markdown': 'markdown', '.mdown': 'markdown',
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh', '.fish': 'fish',
        '.rs': 'rust',
        '.go': 'go',
        '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
        '.c': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
        '.h': 'c', '.hpp': 'cpp', '.hxx': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.sql': 'sql',
        '.r': 'r',
        '.swift': 'swift',
        '.dart': 'dart',
    }
    
    return language_map.get(ext, 'text')


def _highlight_code(text: str, filename: str) -> str:
    """Apply syntax highlighting to code."""
    try:
        lexer = get_lexer_for_filename(filename, stripall=False)
    except Exception:
        lexer = TextLexer(stripall=False)
        
    formatter = HtmlFormatter(
        style='github-dark',
        noclasses=True,
        wrapcode=True,
        lineanchors='line',
        linenos='table'
    )
    return highlight(text, lexer, formatter)


def read_text(path: pathlib.Path) -> str:
    """Read text content from file with UTF-8 encoding."""
    return path.read_text(encoding="utf-8", errors="replace")


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
    import time
    
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
    """Generate HTML format for repository content."""
    import time
    
    # Filter to only included files and calculate stats
    rendered = [i for i in infos if i.decision.include]
    total_tokens = sum(i.token_estimate or 0 for i in rendered)
    
    # HTML template with embedded CSS
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Repository Analysis: {repo_url}</title>
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a1a;
            --bg-secondary: #2a2a2a;
            --bg-tertiary: #3a3a3a;
            --text-primary: #e5e5e5;
            --text-secondary: #b5b5b5;
            --text-muted: #888;
            --accent-primary: #4f9cf9;
            --accent-secondary: #7c3aed;
            --border-color: #404040;
            --hover-color: #333333;
            --code-bg: #252525;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-size: 14px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
            color: white;
            padding: 32px;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.02) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.01) 0%, transparent 50%);
            pointer-events: none;
        }}
        
        .header::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/svg+xml,%3csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3e%3cg fill='none' fill-rule='evenodd'%3e%3cg fill='%23ffffff' fill-opacity='0.02'%3e%3ccircle cx='20' cy='20' r='1'/%3e%3c/g%3e%3c/g%3e%3c/svg%3e");
            pointer-events: none;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 32px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            position: relative;
            z-index: 1;
        }}
        
        .header .meta {{
            margin-top: 20px;
            opacity: 0.9;
            font-size: 13px;
            position: relative;
            z-index: 1;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }}
        
        .meta-item {{
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
        }}
        
        .meta-item:hover {{
            background: rgba(255, 255, 255, 0.12);
            transform: translateY(-1px);
        }}
        
        .stats {{
            background: var(--bg-tertiary);
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 24px;
        }}
        
        .stat {{
            text-align: center;
            padding: 20px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }}
        
        .stat:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            font-weight: 500;
        }}
        
        .toc {{
            background: var(--bg-tertiary);
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .toc h3 {{
            margin: 0 0 20px 0;
            font-size: 18px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
        }}
        
        .toc ul {{
            margin: 0;
            padding: 0;
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 8px;
        }}
        
        .toc li {{
            margin: 0;
        }}
        
        .toc a {{
            color: var(--text-secondary);
            text-decoration: none;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            transition: all 0.2s ease;
        }}
        
        .toc a:hover {{
            color: var(--accent-primary);
            background: var(--hover-color);
        }}
        
        .file-list {{
            max-height: 400px;
            overflow-y: auto;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }}
        
        .file-item {{
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s ease;
        }}
        
        .file-item:hover {{
            background-color: var(--hover-color);
        }}
        
        .file-item:last-child {{
            border-bottom: none;
        }}
        
        .file-name {{
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .file-meta {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        
        .content {{
            padding: 24px;
            background: var(--bg-secondary);
        }}
        
        .file-section {{
            margin-bottom: 32px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background: var(--bg-primary);
        }}
        
        .file-header {{
            background: var(--bg-tertiary);
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-weight: 600;
            font-size: 14px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .file-content {{
            max-height: 600px;
            overflow-y: auto;
            position: relative;
        }}
        
        .file-content::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .file-content::-webkit-scrollbar-track {{
            background: var(--bg-secondary);
        }}
        
        .file-content::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}
        
        .file-content::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}
        
        pre {{
            margin: 0;
            padding: 24px;
            background: var(--code-bg);
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: var(--text-primary);
        }}
        
        .diff-section {{
            margin-top: 32px;
            border: 1px solid var(--accent-primary);
            border-radius: 8px;
            overflow: hidden;
            background: var(--bg-primary);
        }}
        
        .diff-header {{
            background: var(--bg-tertiary);
            padding: 16px 20px;
            border-bottom: 1px solid var(--accent-primary);
            font-weight: 600;
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .icon {{
            width: 16px;
            height: 16px;
        }}
        
        .icon-lg {{
            width: 20px;
            height: 20px;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 12px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 24px;
            }}
            
            .header .meta {{
                flex-direction: column;
                align-items: stretch;
                gap: 8px;
            }}
            
            .meta-item {{
                justify-content: center;
            }}
            
            .stats {{
                grid-template-columns: 1fr;
                gap: 16px;
                padding: 16px;
            }}
            
            .toc ul {{
                grid-template-columns: 1fr;
            }}
            
            .content {{
                padding: 16px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <a href="https://sibylline.dev" style="display: flex; align-items: center; gap: 12px; color: inherit; text-decoration: none;" target="_blank">
                    <img src="https://sibylline.dev/img/logo.svg" alt="Sibylline" style="width: 32px; height: 32px;">
                    Repository Analysis
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
                    <span><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</span>
                </div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">
                    <i data-lucide="files" class="icon-lg"></i>
                    {len(rendered)}
                </div>
                <div class="stat-label">Files</div>
            </div>
            <div class="stat">
                <div class="stat-value">
                    <i data-lucide="hash" class="icon-lg"></i>
                    {total_tokens:,}
                </div>
                <div class="stat-label">Estimated Tokens</div>
            </div>
            <div class="stat">
                <div class="stat-value">
                    <i data-lucide="hard-drive" class="icon-lg"></i>
                    {bytes_human(sum(i.size or 0 for i in rendered))}
                </div>
                <div class="stat-label">Total Size</div>
            </div>
        </div>
        
        <div class="toc">
            <h3>
                <i data-lucide="list" class="icon"></i>
                Table of Contents
            </h3>
            <ul>"""
    
    # Add file links to TOC
    for i, file_info in enumerate(rendered, 1):
        file_icon = get_file_icon(file_info.rel)
        html_content += f"""
                <li><a href="#file-{i}"><i data-lucide="{file_icon}" class="icon"></i>{html.escape(file_info.rel)}</a></li>"""
    
    if diff_content:
        html_content += """
                <li><a href="#diffs"><i data-lucide="git-pull-request" class="icon"></i>Repository Diffs</a></li>"""
    
    html_content += """
            </ul>
        </div>
        
        <div class="file-list">"""
    
    # Add file list with metadata
    for i, file_info in enumerate(rendered, 1):
        file_meta_parts = []
        if file_info.size:
            file_meta_parts.append(f"{bytes_human(file_info.size)}")
        if file_info.token_estimate:
            file_meta_parts.append(f"~{file_info.token_estimate} tokens")
        file_meta = " • ".join(file_meta_parts)
        file_icon = get_file_icon(file_info.rel)
        
        html_content += f"""
            <div class="file-item">
                <span class="file-name"><i data-lucide="{file_icon}" class="icon"></i>{html.escape(file_info.rel)}</span>
                <span class="file-meta">{file_meta}</span>
            </div>"""
    
    html_content += """
        </div>
        
        <div class="content">"""
    
    # Add file contents
    for i, file_info in enumerate(rendered, 1):
        file_icon = get_file_icon(file_info.rel)
        html_content += f"""
            <div class="file-section" id="file-{i}">
                <div class="file-header"><i data-lucide="{file_icon}" class="icon"></i>{html.escape(file_info.rel)}</div>
                <div class="file-content">
                    <pre>"""
        
        if file_info.content is not None:
            html_content += html.escape(file_info.content)
        else:
            try:
                content = read_text(file_info.path)
                html_content += html.escape(content)
            except Exception as e:
                html_content += f"Failed to read file: {html.escape(str(e))}"
        
        html_content += """</pre>
                </div>
            </div>"""
    
    # Add diff content if available
    if diff_content:
        html_content += f"""
            <div class="diff-section" id="diffs">
                <div class="diff-header"><i data-lucide="git-pull-request" class="icon"></i>Repository Diffs (Relevance Filtered)</div>
                <div class="file-content">
                    <pre>{html.escape(diff_content)}</pre>
                </div>
            </div>"""
    
    html_content += """
        </div>
    </div>
    <script>
        // Initialize Lucide icons
        lucide.createIcons();
    </script>
</body>
</html>"""
    
    return html_content


def derive_temp_output_path(repo_url: str) -> pathlib.Path:
    """Derive a temporary output path from the repo URL."""
    # Extract repo name from URL like https://github.com/owner/repo or https://github.com/owner/repo.git
    parts = repo_url.rstrip('/').split('/')
    
    # Handle malformed URLs and edge cases
    if not repo_url or not repo_url.startswith(('http://', 'https://')):
        filename = "repo.html"
    elif len(parts) >= 5 and parts[-1]:  # Has at least protocol/empty/domain/owner/repo
        repo_name = parts[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        # Ensure we have a valid repo name
        if repo_name and repo_name not in ['', '/']:
            filename = f"{repo_name}.html"
        else:
            filename = "repo.html"
    else:
        # No valid owner/repo structure (e.g., https://github.com or https://github.com/user)
        filename = "repo.html"

    return pathlib.Path(tempfile.gettempdir()) / filename
"""
Output formatting for different file formats.

Supports multiple output formats including JSON, XML, Markdown, and Plain text
compatible with repomix-style output options.
"""

import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from ..fastpath.fast_scan import ScanResult
from ..fastpath.git_integration import GitIntegration, GitCommit, GitDiff


@dataclass
class OutputConfig:
    """Configuration for output formatting."""
    style: str = "plain"  # "json", "xml", "markdown", "plain"
    include_file_summary: bool = True
    include_directory_structure: bool = True
    include_files: bool = True
    show_line_numbers: bool = False
    include_git_info: bool = False
    include_diffs: bool = False
    include_commit_history: bool = False
    max_commits: int = 50
    custom_header: Optional[str] = None
    copy_to_clipboard: bool = False


@dataclass
class DirectoryNode:
    """Represents a directory structure node."""
    name: str
    path: str
    is_file: bool
    size_bytes: Optional[int] = None
    children: Optional[List['DirectoryNode']] = None


@dataclass
class PackSummary:
    """Summary information about the packed repository."""
    total_files: int
    total_size_bytes: int
    total_lines: int
    total_tokens: int
    languages: Dict[str, int]
    file_types: Dict[str, int]
    generation_time: float
    git_info: Optional[Dict[str, Any]] = None


class OutputFormatter:
    """Base class for output formatters."""
    
    def __init__(self, config: OutputConfig):
        self.config = config
        
    def format_output(
        self,
        selected_files: List[ScanResult],
        summary: PackSummary,
        directory_structure: Optional[DirectoryNode] = None,
        git_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format the complete output."""
        raise NotImplementedError


class JSONFormatter(OutputFormatter):
    """JSON output formatter."""
    
    def format_output(
        self,
        selected_files: List[ScanResult],
        summary: PackSummary,
        directory_structure: Optional[DirectoryNode] = None,
        git_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format output as JSON."""
        output_data = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "generator": "Scribe FastPath",
                "version": "1.0",
                "config": asdict(self.config)
            }
        }
        
        if self.config.include_file_summary:
            output_data["summary"] = asdict(summary)
            
        if self.config.include_directory_structure and directory_structure:
            output_data["directory_structure"] = self._serialize_directory_tree(directory_structure)
            
        if self.config.include_git_info and git_info:
            output_data["git_info"] = git_info
            
        if self.config.include_files:
            output_data["files"] = []
            for scan_result in selected_files:
                file_data = {
                    "path": scan_result.stats.path,
                    "language": scan_result.stats.language,
                    "size_bytes": scan_result.stats.size_bytes,
                    "lines": scan_result.stats.lines,
                    "is_readme": scan_result.stats.is_readme,
                    "is_test": scan_result.stats.is_test,
                    "is_config": scan_result.stats.is_config,
                    "is_docs": scan_result.stats.is_docs,
                    "priority_boost": scan_result.priority_boost,
                    "churn_score": scan_result.churn_score,
                    "content": self._get_file_content(scan_result)
                }
                
                if scan_result.imports:
                    file_data["imports"] = {
                        "count": scan_result.imports.import_count,
                        "external": scan_result.imports.external_imports,
                        "relative": scan_result.imports.relative_imports,
                        "modules": list(scan_result.imports.imports)
                    }
                    
                if scan_result.doc_analysis:
                    file_data["document_analysis"] = asdict(scan_result.doc_analysis)
                    
                output_data["files"].append(file_data)
                
        return json.dumps(output_data, indent=2, ensure_ascii=False)
        
    def _serialize_directory_tree(self, node: DirectoryNode) -> Dict[str, Any]:
        """Serialize directory tree to JSON-compatible format."""
        result = {
            "name": node.name,
            "path": node.path,
            "is_file": node.is_file,
        }
        
        if node.size_bytes is not None:
            result["size_bytes"] = node.size_bytes
            
        if node.children:
            result["children"] = [
                self._serialize_directory_tree(child) for child in node.children
            ]
            
        return result
        
    def _get_file_content(self, scan_result: ScanResult) -> str:
        """Get file content with optional line numbers."""
        try:
            # In a real implementation, this would read the file content
            # For now, return a placeholder since we don't store content in ScanResult
            return f"# Content of {scan_result.stats.path}\\n# (Content would be read from file)"
        except Exception:
            return f"# Unable to read content of {scan_result.stats.path}"


class MarkdownFormatter(OutputFormatter):
    """Markdown output formatter."""
    
    def format_output(
        self,
        selected_files: List[ScanResult],
        summary: PackSummary,
        directory_structure: Optional[DirectoryNode] = None,
        git_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format output as Markdown."""
        lines = []
        
        # Header
        if self.config.custom_header:
            lines.append(self.config.custom_header)
            lines.append("")
            
        lines.append("# Repository Pack")
        lines.append("")
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        lines.append(f"Generator: Scribe FastPath")
        lines.append("")
        
        # Summary
        if self.config.include_file_summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(f"- **Total Files**: {summary.total_files}")
            lines.append(f"- **Total Size**: {self._format_bytes(summary.total_size_bytes)}")
            lines.append(f"- **Total Lines**: {summary.total_lines:,}")
            lines.append(f"- **Estimated Tokens**: {summary.total_tokens:,}")
            lines.append("")
            
            if summary.languages:
                lines.append("### Languages")
                lines.append("")
                for lang, count in sorted(summary.languages.items()):
                    lines.append(f"- {lang}: {count} files")
                lines.append("")
                
        # Directory Structure
        if self.config.include_directory_structure and directory_structure:
            lines.append("## Directory Structure")
            lines.append("")
            lines.append("```")
            self._render_directory_tree(directory_structure, lines, "")
            lines.append("```")
            lines.append("")
            
        # Git Information
        if self.config.include_git_info and git_info:
            lines.append("## Git Information")
            lines.append("")
            
            if "commits" in git_info:
                lines.append("### Recent Commits")
                lines.append("")
                for commit in git_info["commits"][:10]:
                    lines.append(f"- `{commit.hash[:8]}` - {commit.message} ({commit.author})")
                lines.append("")
                
            if "diffs" in git_info and git_info["diffs"]:
                lines.append("### Working Tree Changes")
                lines.append("")
                for diff in git_info["diffs"]:
                    lines.append(f"- `{diff.status}` {diff.file_path} (+{diff.additions}/-{diff.deletions})")
                lines.append("")
                
        # Files
        if self.config.include_files:
            lines.append("## Files")
            lines.append("")
            
            for scan_result in selected_files:
                lang = scan_result.stats.language or "text"
                lines.append(f"### {scan_result.stats.path}")
                lines.append("")
                lines.append(f"**Language**: {lang} | **Size**: {self._format_bytes(scan_result.stats.size_bytes)} | **Lines**: {scan_result.stats.lines}")
                lines.append("")
                lines.append(f"```{lang}")
                lines.append(self._get_file_content(scan_result))
                lines.append("```")
                lines.append("")
                
        return "\\n".join(lines)
        
    def _render_directory_tree(self, node: DirectoryNode, lines: List[str], prefix: str):
        """Render directory tree as text."""
        if node.name:  # Skip root
            icon = "📄" if node.is_file else "📁"
            size_info = f" ({self._format_bytes(node.size_bytes)})" if node.size_bytes else ""
            lines.append(f"{prefix}{icon} {node.name}{size_info}")
            
        if node.children:
            for i, child in enumerate(node.children):
                is_last = i == len(node.children) - 1
                child_prefix = prefix + ("└── " if is_last else "├── ")
                next_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{child_prefix}{child.name}")
                if child.children:
                    self._render_directory_tree(child, lines, next_prefix)
                    
    def _format_bytes(self, bytes_count: int) -> str:
        """Format byte count as human readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"
        
    def _get_file_content(self, scan_result: ScanResult) -> str:
        """Get file content with optional line numbers."""
        try:
            # Placeholder - would read actual file content
            content = f"// Content of {scan_result.stats.path}\\n// (Content would be read from file)"
            
            if self.config.show_line_numbers:
                lines = content.split("\\n")
                numbered_lines = [f"{i+1:3d}: {line}" for i, line in enumerate(lines)]
                return "\\n".join(numbered_lines)
            else:
                return content
        except Exception:
            return f"// Unable to read content of {scan_result.stats.path}"


class PlainFormatter(OutputFormatter):
    """Plain text output formatter."""
    
    def format_output(
        self,
        selected_files: List[ScanResult],
        summary: PackSummary,
        directory_structure: Optional[DirectoryNode] = None,
        git_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format output as plain text."""
        lines = []
        
        # Header
        if self.config.custom_header:
            lines.append(self.config.custom_header)
            lines.append("=" * 80)
            lines.append("")
            
        lines.append("REPOSITORY PACK")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        lines.append(f"Generator: Scribe FastPath")
        lines.append("")
        
        # Summary
        if self.config.include_file_summary:
            lines.append("SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Total Files: {summary.total_files}")
            lines.append(f"Total Size: {self._format_bytes(summary.total_size_bytes)}")
            lines.append(f"Total Lines: {summary.total_lines:,}")
            lines.append(f"Estimated Tokens: {summary.total_tokens:,}")
            lines.append("")
            
        # Files
        if self.config.include_files:
            lines.append("FILES")
            lines.append("-" * 40)
            lines.append("")
            
            for scan_result in selected_files:
                lines.append(f"FILE: {scan_result.stats.path}")
                lines.append(f"Language: {scan_result.stats.language or 'unknown'}")
                lines.append(f"Size: {self._format_bytes(scan_result.stats.size_bytes)}")
                lines.append(f"Lines: {scan_result.stats.lines}")
                lines.append("")
                lines.append(self._get_file_content(scan_result))
                lines.append("")
                lines.append("=" * 80)
                lines.append("")
                
        return "\\n".join(lines)
        
    def _format_bytes(self, bytes_count: int) -> str:
        """Format byte count as human readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"
        
    def _get_file_content(self, scan_result: ScanResult) -> str:
        """Get file content with optional line numbers."""
        # Placeholder - would read actual file content
        return f"Content of {scan_result.stats.path}\\n(Content would be read from file)"


def create_directory_structure(repo_path: Path, selected_files: List[ScanResult]) -> DirectoryNode:
    """Create directory structure from selected files."""
    root = DirectoryNode(
        name="",
        path=str(repo_path),
        is_file=False,
        children=[]
    )
    
    # Build tree from selected files
    for scan_result in selected_files:
        parts = Path(scan_result.stats.path).parts
        current_node = root
        
        for i, part in enumerate(parts):
            is_file = i == len(parts) - 1
            
            # Find existing child or create new one
            child_node = None
            if current_node.children:
                child_node = next(
                    (child for child in current_node.children if child.name == part),
                    None
                )
                
            if not child_node:
                child_node = DirectoryNode(
                    name=part,
                    path="/".join(parts[:i+1]),
                    is_file=is_file,
                    size_bytes=scan_result.stats.size_bytes if is_file else None,
                    children=[] if not is_file else None
                )
                if current_node.children is None:
                    current_node.children = []
                current_node.children.append(child_node)
                
            current_node = child_node
            
    return root


def create_pack_summary(
    selected_files: List[ScanResult],
    total_tokens: int,
    git_info: Optional[Dict[str, Any]] = None
) -> PackSummary:
    """Create pack summary from selected files."""
    total_size = sum(f.stats.size_bytes for f in selected_files)
    total_lines = sum(f.stats.lines for f in selected_files)
    
    # Language distribution
    languages = {}
    file_types = {}
    
    for scan_result in selected_files:
        lang = scan_result.stats.language or "unknown"
        languages[lang] = languages.get(lang, 0) + 1
        
        ext = Path(scan_result.stats.path).suffix.lower() or "no_ext"
        file_types[ext] = file_types.get(ext, 0) + 1
        
    return PackSummary(
        total_files=len(selected_files),
        total_size_bytes=total_size,
        total_lines=total_lines,
        total_tokens=total_tokens,
        languages=languages,
        file_types=file_types,
        generation_time=time.time(),
        git_info=git_info
    )


def create_formatter(config: OutputConfig) -> OutputFormatter:
    """Create output formatter based on configuration."""
    formatters = {
        "json": JSONFormatter,
        "markdown": MarkdownFormatter,
        "plain": PlainFormatter,
        "xml": PlainFormatter,  # TODO: Implement XML formatter
    }
    
    formatter_class = formatters.get(config.style, PlainFormatter)
    return formatter_class(config)


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Test output formatting
    config = OutputConfig(
        style="json",
        include_file_summary=True,
        include_directory_structure=True,
        include_files=True,
        show_line_numbers=False
    )
    
    formatter = create_formatter(config)
    
    # Create sample data (would come from actual scanning)
    sample_files = []  # Would be populated with ScanResult objects
    summary = PackSummary(
        total_files=10,
        total_size_bytes=50000,
        total_lines=2000,
        total_tokens=8000,
        languages={"python": 5, "markdown": 2, "yaml": 3},
        file_types={".py": 5, ".md": 2, ".yaml": 3},
        generation_time=time.time()
    )
    
    # output = formatter.format_output(sample_files, summary)
    # print(output)
"""
Diff Packing with Relevance Gating

Extends FastPath to include Git diffs in the repository packing process.
Diffs are subjected to the same relevance gating as regular code files,
allowing users to pack changes that are relevant to their problem context.

This enables analysis of:
- Recent changes relevant to specific features
- Bug fix contexts with related code
- Development history for targeted areas
- Change impact analysis
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .fast_scan import ScanResult
from .personalized_centrality import PersonalizedCentralityCalculator, EntryPoint
from .utils import FilePatternMatcher, ErrorHandler


@dataclass
class DiffEntry:
    """Represents a single diff/change in the repository."""
    file_path: str
    change_type: str  # 'added', 'modified', 'deleted', 'renamed'
    diff_content: str
    line_additions: int
    line_deletions: int
    commit_hash: Optional[str] = None
    commit_message: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[str] = None
    old_file_path: Optional[str] = None  # For renames


@dataclass
class DiffPackingConfig:
    """Configuration for diff packing functionality."""
    # Git options
    include_staged: bool = True
    include_unstaged: bool = True
    include_commits: Optional[List[str]] = None  # Specific commit hashes
    commit_range: Optional[str] = None  # e.g., "HEAD~5..HEAD"
    branch_comparison: Optional[str] = None  # e.g., "main..feature-branch"
    
    # Filtering options
    max_commits: int = 50
    max_diff_size_kb: int = 100  # Skip very large diffs
    ignore_patterns: List[str] = None  # Patterns to ignore (e.g., "*.lock", "node_modules/*")
    
    # Relevance gating
    enable_relevance_gating: bool = True
    relevance_threshold: float = 0.1  # Minimum centrality score to include diff
    relevance_calculator: Optional[PersonalizedCentralityCalculator] = None
    
    # Content filtering
    include_binary_diffs: bool = False
    include_generated_files: bool = False
    max_lines_per_diff: int = 1000
    
    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = [
                "*.lock", "*.log", "*.tmp", "*.cache",
                "node_modules/*", ".git/*", "__pycache__/*",
                "*.min.js", "*.min.css", "build/*", "dist/*"
            ]


class GitDiffExtractor:
    """Extracts diff information from Git repositories."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self._validate_git_repo()
    
    def _validate_git_repo(self):
        """Ensure the path is a valid Git repository."""
        if not (self.repo_path / '.git').exists():
            raise ValueError(f"Not a Git repository: {self.repo_path}")
    
    def extract_diffs(self, config: DiffPackingConfig) -> List[DiffEntry]:
        """Extract diffs according to the configuration."""
        all_diffs = []
        
        if config.include_staged:
            all_diffs.extend(self._get_staged_diffs())
        
        if config.include_unstaged:
            all_diffs.extend(self._get_unstaged_diffs())
        
        if config.include_commits:
            for commit_hash in config.include_commits:
                all_diffs.extend(self._get_commit_diffs(commit_hash))
        
        if config.commit_range:
            all_diffs.extend(self._get_range_diffs(config.commit_range))
        
        if config.branch_comparison:
            all_diffs.extend(self._get_branch_comparison_diffs(config.branch_comparison))
        
        # Apply filtering
        filtered_diffs = self._apply_filters(all_diffs, config)
        
        return filtered_diffs[:config.max_commits] if config.max_commits > 0 else filtered_diffs
    
    def _get_staged_diffs(self) -> List[DiffEntry]:
        """Get staged changes."""
        return self._parse_diff_output(
            self._run_git_command(['diff', '--cached', '--name-status']),
            self._run_git_command(['diff', '--cached'])
        )
    
    def _get_unstaged_diffs(self) -> List[DiffEntry]:
        """Get unstaged changes."""
        return self._parse_diff_output(
            self._run_git_command(['diff', '--name-status']),
            self._run_git_command(['diff'])
        )
    
    def _get_commit_diffs(self, commit_hash: str) -> List[DiffEntry]:
        """Get diffs for a specific commit."""
        commit_info = self._get_commit_info(commit_hash)
        
        name_status = self._run_git_command(['diff', '--name-status', f'{commit_hash}~1', commit_hash])
        diff_content = self._run_git_command(['show', '--no-merges', commit_hash])
        
        diffs = self._parse_diff_output(name_status, diff_content)
        
        # Add commit metadata
        for diff in diffs:
            diff.commit_hash = commit_hash
            diff.commit_message = commit_info.get('message')
            diff.author = commit_info.get('author')
            diff.timestamp = commit_info.get('timestamp')
        
        return diffs
    
    def _get_range_diffs(self, commit_range: str) -> List[DiffEntry]:
        """Get diffs for a range of commits."""
        try:
            commits = self._run_git_command(['rev-list', '--reverse', commit_range]).strip().split('\n')
        except RuntimeError as e:
            # If the range is invalid, try to get recent commits instead
            try:
                available_commits = self._run_git_command(['rev-list', '--max-count=10', 'HEAD']).strip().split('\n')
                if len(available_commits) <= 1:
                    return []  # Not enough history
                # Use last few commits
                commits = available_commits[:min(5, len(available_commits)-1)]
            except RuntimeError:
                return []  # No git history available
        
        all_diffs = []
        
        for commit_hash in commits:
            if commit_hash.strip():
                all_diffs.extend(self._get_commit_diffs(commit_hash.strip()))
        
        return all_diffs
    
    def _get_branch_comparison_diffs(self, branch_spec: str) -> List[DiffEntry]:
        """Get diffs between branches."""
        name_status = self._run_git_command(['diff', '--name-status', branch_spec])
        diff_content = self._run_git_command(['diff', branch_spec])
        
        return self._parse_diff_output(name_status, diff_content)
    
    def _parse_diff_output(self, name_status_output: str, diff_content: str) -> List[DiffEntry]:
        """Parse Git diff output into DiffEntry objects."""
        entries = []
        
        # Parse name-status to get file changes
        for line in name_status_output.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            status = parts[0]
            file_path = parts[1]
            old_file_path = parts[2] if len(parts) > 2 else None
            
            # Map Git status codes to change types
            change_type_map = {
                'A': 'added',
                'M': 'modified', 
                'D': 'deleted',
                'R': 'renamed',
                'C': 'copied'
            }
            
            change_type = change_type_map.get(status[0], 'modified')
            
            # Extract diff for this specific file
            file_diff = self._extract_file_diff(diff_content, file_path, old_file_path)
            line_stats = self._count_diff_lines(file_diff)
            
            entry = DiffEntry(
                file_path=file_path,
                change_type=change_type,
                diff_content=file_diff,
                line_additions=line_stats['additions'],
                line_deletions=line_stats['deletions'],
                old_file_path=old_file_path
            )
            
            entries.append(entry)
        
        return entries
    
    def _extract_file_diff(self, full_diff: str, file_path: str, old_file_path: str = None) -> str:
        """Extract diff content for a specific file."""
        # Look for the file's diff section
        patterns = [
            rf"diff --git a/{re.escape(file_path)} b/{re.escape(file_path)}.*?(?=diff --git|\Z)",
        ]
        
        if old_file_path:
            patterns.append(
                rf"diff --git a/{re.escape(old_file_path)} b/{re.escape(file_path)}.*?(?=diff --git|\Z)"
            )
        
        for pattern in patterns:
            match = re.search(pattern, full_diff, re.DOTALL)
            if match:
                return match.group(0)
        
        return ""
    
    def _count_diff_lines(self, diff_content: str) -> Dict[str, int]:
        """Count addition and deletion lines in diff content."""
        additions = len(re.findall(r'^\\+(?!\\+\\+)', diff_content, re.MULTILINE))
        deletions = len(re.findall(r'^-(?!---)', diff_content, re.MULTILINE))
        
        return {'additions': additions, 'deletions': deletions}
    
    def _get_commit_info(self, commit_hash: str) -> Dict[str, str]:
        """Get metadata for a commit."""
        try:
            output = self._run_git_command([
                'show', '--no-patch', '--format=%H%n%an%n%ae%n%at%n%s%n%b',
                commit_hash
            ])
            
            lines = output.strip().split('\n', 5)
            return {
                'hash': lines[0] if len(lines) > 0 else commit_hash,
                'author': lines[1] if len(lines) > 1 else '',
                'email': lines[2] if len(lines) > 2 else '',
                'timestamp': lines[3] if len(lines) > 3 else '',
                'message': lines[4] if len(lines) > 4 else '',
                'body': lines[5] if len(lines) > 5 else ''
            }
        except subprocess.CalledProcessError:
            return {'hash': commit_hash}
    
    def _apply_filters(self, diffs: List[DiffEntry], config: DiffPackingConfig) -> List[DiffEntry]:
        """Apply filtering rules to the diff list."""
        filtered = []
        
        for diff in diffs:
            # Skip if matches ignore patterns
            if self._should_ignore_file(diff.file_path, config.ignore_patterns):
                continue
            
            # Skip very large diffs
            if len(diff.diff_content) > config.max_diff_size_kb * 1024:
                continue
            
            # Skip binary diffs if not enabled
            if not config.include_binary_diffs and self._is_binary_diff(diff.diff_content):
                continue
            
            # Skip generated files if not enabled
            if not config.include_generated_files and self._is_generated_file(diff.file_path):
                continue
            
            # Truncate very long diffs
            if config.max_lines_per_diff > 0:
                diff.diff_content = self._truncate_diff(diff.diff_content, config.max_lines_per_diff)
            
            filtered.append(diff)
        
        return filtered
    
    def _should_ignore_file(self, file_path: str, ignore_patterns: List[str]) -> bool:
        """Check if file should be ignored based on patterns."""
        from .utils.file_patterns import FilePatternMatcher
        matcher = FilePatternMatcher()
        return matcher.should_ignore(file_path, ignore_patterns)
    
    def _is_binary_diff(self, diff_content: str) -> bool:
        """Check if diff contains binary content."""
        return 'Binary files' in diff_content or b'\\x00' in diff_content.encode('utf-8', errors='ignore')
    
    def _is_generated_file(self, file_path: str) -> bool:
        """Check if file appears to be generated."""
        generated_indicators = [
            '.generated.', '_pb2.py', '.proto.py',
            'package-lock.json', 'yarn.lock', 'poetry.lock',
            '.min.js', '.min.css', 'bundle.js'
        ]
        
        return any(indicator in file_path for indicator in generated_indicators)
    
    def _truncate_diff(self, diff_content: str, max_lines: int) -> str:
        """Truncate diff content to maximum number of lines."""
        lines = diff_content.split('\n')
        if len(lines) <= max_lines:
            return diff_content
        
        truncated_lines = lines[:max_lines]
        truncated_lines.append(f"... [Diff truncated - {len(lines) - max_lines} more lines]")
        return '\n'.join(truncated_lines)
    
    def _run_git_command(self, args: List[str]) -> str:
        """Run a git command and return its output."""
        from .utils.error_handling import ErrorHandler, ErrorContext
        
        handler = ErrorHandler("GitDiffExtractor")
        context = ErrorContext(
            operation="git_command",
            component="GitDiffExtractor", 
            additional_info={"command": args, "cwd": self.repo_path}
        )
        
        # Use subprocess directly as ErrorHandler.safe_subprocess_call needs proper setup
        import subprocess
        
        def run_command():
            return subprocess.run(
                ['git'] + args,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_path,
                timeout=30
            ).stdout
        
        result = handler.safe_execute(run_command, context)
        if result.is_failure:
            raise result.error
        
        return result.value


class DiffRelevanceGate:
    """Applies relevance filtering to diffs based on centrality scores."""
    
    def __init__(self, centrality_calculator: Optional[PersonalizedCentralityCalculator] = None):
        self.centrality_calculator = centrality_calculator
    
    def filter_relevant_diffs(
        self,
        diffs: List[DiffEntry],
        scan_results: List[ScanResult],
        threshold: float = 0.1
    ) -> List[DiffEntry]:
        """Filter diffs to include only those that are relevant."""
        if not self.centrality_calculator:
            # No filtering - return all diffs
            return diffs
        
        # Calculate centrality scores
        centrality_scores = self.centrality_calculator.calculate_personalized_centrality(scan_results)
        
        relevant_diffs = []
        for diff in diffs:
            # Get centrality score for the changed file
            file_score = centrality_scores.pagerank_scores.get(diff.file_path, 0.0)
            
            # Include diff if it meets the relevance threshold
            if file_score >= threshold:
                relevant_diffs.append(diff)
        
        return relevant_diffs


class DiffPacker:
    """
    Main interface for packing diffs with relevance gating.
    
    Integrates diff extraction, relevance filtering, and formatting
    into the FastPath repository packing workflow.
    """
    
    def __init__(self, repo_path: str, config: DiffPackingConfig):
        self.repo_path = repo_path
        self.config = config
        self.diff_extractor = GitDiffExtractor(repo_path)
        
        # Set up relevance gating if enabled
        self.relevance_gate = None
        if config.enable_relevance_gating and config.relevance_calculator:
            self.relevance_gate = DiffRelevanceGate(config.relevance_calculator)
    
    def pack_diffs(self, scan_results: List[ScanResult]) -> Tuple[List[DiffEntry], str]:
        """
        Extract and pack relevant diffs.
        
        Args:
            scan_results: Scanned repository files for relevance context
            
        Returns:
            Tuple of (relevant_diffs, formatted_diff_content)
        """
        # Extract diffs according to configuration
        all_diffs = self.diff_extractor.extract_diffs(self.config)
        
        # Apply relevance filtering if enabled
        relevant_diffs = all_diffs
        if self.relevance_gate:
            relevant_diffs = self.relevance_gate.filter_relevant_diffs(
                all_diffs, scan_results, self.config.relevance_threshold
            )
        
        # Format diffs for inclusion in pack
        formatted_content = self._format_diffs_for_pack(relevant_diffs)
        
        return relevant_diffs, formatted_content
    
    def _format_diffs_for_pack(self, diffs: List[DiffEntry]) -> str:
        """Format diffs for inclusion in the repository pack."""
        if not diffs:
            return ""
        
        sections = []
        sections.append("# Repository Diffs (Relevance Filtered)\n")
        sections.append(f"Found {len(diffs)} relevant changes:\n")
        
        for i, diff in enumerate(diffs, 1):
            sections.append(f"## Diff {i}: {diff.file_path}")
            
            # Add metadata
            metadata = [
                f"**Change Type:** {diff.change_type}",
                f"**Lines:** +{diff.line_additions}/-{diff.line_deletions}",
            ]
            
            if diff.commit_hash:
                metadata.append(f"**Commit:** {diff.commit_hash[:8]}")
            
            if diff.commit_message:
                metadata.append(f"**Message:** {diff.commit_message}")
            
            if diff.old_file_path:
                metadata.append(f"**Old Path:** {diff.old_file_path}")
            
            sections.append(" | ".join(metadata))
            sections.append("")
            
            # Add diff content
            sections.append("```diff")
            sections.append(diff.diff_content)
            sections.append("```")
            sections.append("")
        
        return "\n".join(sections)


# Convenience functions

def create_diff_packer(
    repo_path: str,
    entry_points: Optional[List[Union[str, EntryPoint]]] = None,
    **config_kwargs
) -> DiffPacker:
    """Create a DiffPacker with optional entry point relevance."""
    config = DiffPackingConfig(**config_kwargs)
    
    # Set up personalized centrality if entry points provided
    if entry_points:
        from .personalized_centrality import create_personalized_calculator
        config.relevance_calculator = create_personalized_calculator(entry_points)
        config.enable_relevance_gating = True
    
    return DiffPacker(repo_path, config)


def pack_recent_diffs(
    repo_path: str,
    num_commits: int = 10,
    entry_points: Optional[List[Union[str, EntryPoint]]] = None,
    relevance_threshold: float = 0.1
) -> Tuple[List[DiffEntry], str]:
    """Quick function to pack recent diffs with optional entry point filtering."""
    config = DiffPackingConfig(
        commit_range=f"HEAD~{num_commits}..HEAD",
        relevance_threshold=relevance_threshold
    )
    
    diff_packer = create_diff_packer(repo_path, entry_points, **config.__dict__)
    
    # Need scan results for relevance calculation
    from .fast_scan import FastScanner
    scanner = FastScanner(repo_path)
    scan_results = scanner.scan_repository()
    
    return diff_packer.pack_diffs(scan_results)
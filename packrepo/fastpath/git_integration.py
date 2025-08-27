"""
Git integration for repository analysis and remote support.

Provides git-aware file selection, change frequency analysis, and remote repository support.
"""

import os
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from urllib.parse import urlparse


class GitFileInfo(NamedTuple):
    """Information about a file from Git."""
    path: str
    change_count: int
    last_modified: float
    author_count: int
    lines_added: int
    lines_deleted: int


@dataclass
class GitCommit:
    """Git commit information."""
    hash: str
    author: str
    date: str
    message: str
    files_changed: List[str]


@dataclass
class GitDiff:
    """Git diff information."""
    file_path: str
    status: str  # A, M, D, R, C (added, modified, deleted, renamed, copied)
    additions: int
    deletions: int
    diff_text: str


class GitIntegration:
    """
    Git integration for repository analysis.
    
    Provides file change frequency analysis, commit history, and diff support.
    Compatible with repomix-style git features.
    """
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._git_available = self._check_git_availability()
        
    def _check_git_availability(self) -> bool:
        """Check if git is available and repo is a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def is_git_repo(self) -> bool:
        """Check if the repository is under git version control."""
        return self._git_available
        
    def get_file_change_frequencies(
        self,
        max_commits: int = 100,
        file_paths: Optional[List[str]] = None
    ) -> Dict[str, GitFileInfo]:
        """
        Get change frequencies for files based on git log.
        
        Args:
            max_commits: Maximum number of commits to analyze
            file_paths: Specific files to analyze (None = all files)
            
        Returns:
            Dictionary mapping file paths to GitFileInfo
        """
        if not self._git_available:
            return {}
            
        file_stats = {}
        
        try:
            # Get git log with file statistics
            cmd = [
                'git', 'log', '--numstat', '--pretty=format:%H|%an|%at|%s',
                f'-{max_commits}'
            ]
            
            if file_paths:
                cmd.extend(['--'] + file_paths)
                
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {}
                
            lines = result.stdout.strip().split('\\n')
            current_commit = None
            
            for line in lines:
                if not line.strip():
                    continue
                    
                if '|' in line and not '\\t' in line:
                    # Commit line: hash|author|timestamp|message
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        current_commit = {
                            'hash': parts[0],
                            'author': parts[1],
                            'timestamp': float(parts[2]),
                            'message': parts[3]
                        }
                elif '\\t' in line:
                    # File stat line: additions\\tdeletions\\tfilename
                    parts = line.split('\\t')
                    if len(parts) >= 3 and current_commit:
                        additions = int(parts[0]) if parts[0].isdigit() else 0
                        deletions = int(parts[1]) if parts[1].isdigit() else 0
                        filename = parts[2]
                        
                        if filename not in file_stats:
                            file_stats[filename] = {
                                'change_count': 0,
                                'last_modified': 0,
                                'authors': set(),
                                'lines_added': 0,
                                'lines_deleted': 0
                            }
                            
                        stats = file_stats[filename]
                        stats['change_count'] += 1
                        stats['last_modified'] = max(
                            stats['last_modified'], 
                            current_commit['timestamp']
                        )
                        stats['authors'].add(current_commit['author'])
                        stats['lines_added'] += additions
                        stats['lines_deleted'] += deletions
                        
            # Convert to GitFileInfo objects
            result_stats = {}
            for filename, stats in file_stats.items():
                result_stats[filename] = GitFileInfo(
                    path=filename,
                    change_count=stats['change_count'],
                    last_modified=stats['last_modified'],
                    author_count=len(stats['authors']),
                    lines_added=stats['lines_added'],
                    lines_deleted=stats['lines_deleted']
                )
                
            return result_stats
            
        except (subprocess.SubprocessError, ValueError, KeyError):
            return {}
            
    def get_recent_commits(self, count: int = 50) -> List[GitCommit]:
        """Get recent commit history."""
        if not self._git_available:
            return []
            
        try:
            result = subprocess.run([
                'git', 'log', '--pretty=format:%H|%an|%ad|%s', 
                '--date=iso', f'-{count}'
            ], 
            cwd=self.repo_path,
            capture_output=True, 
            text=True,
            timeout=10
            )
            
            if result.returncode != 0:
                return []
                
            commits = []
            for line in result.stdout.strip().split('\\n'):
                if line.strip():
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        commits.append(GitCommit(
                            hash=parts[0],
                            author=parts[1],
                            date=parts[2],
                            message=parts[3],
                            files_changed=[]  # Could be populated with additional git show
                        ))
                        
            return commits
            
        except subprocess.SubprocessError:
            return []
            
    def get_working_tree_diffs(self) -> List[GitDiff]:
        """Get current working tree changes (staged + unstaged)."""
        if not self._git_available:
            return []
            
        diffs = []
        
        try:
            # Get staged changes
            staged_result = subprocess.run([
                'git', 'diff', '--cached', '--numstat'
            ], 
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=10
            )
            
            if staged_result.returncode == 0:
                for line in staged_result.stdout.strip().split('\\n'):
                    if line.strip():
                        parts = line.split('\\t')
                        if len(parts) >= 3:
                            diffs.append(GitDiff(
                                file_path=parts[2],
                                status='M',  # Modified (staged)
                                additions=int(parts[0]) if parts[0].isdigit() else 0,
                                deletions=int(parts[1]) if parts[1].isdigit() else 0,
                                diff_text=""  # Could be populated with git diff content
                            ))
                            
            # Get unstaged changes
            unstaged_result = subprocess.run([
                'git', 'diff', '--numstat'
            ], 
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=10
            )
            
            if unstaged_result.returncode == 0:
                for line in unstaged_result.stdout.strip().split('\\n'):
                    if line.strip():
                        parts = line.split('\\t')
                        if len(parts) >= 3:
                            diffs.append(GitDiff(
                                file_path=parts[2],
                                status='M',  # Modified (unstaged)
                                additions=int(parts[0]) if parts[0].isdigit() else 0,
                                deletions=int(parts[1]) if parts[1].isdigit() else 0,
                                diff_text=""
                            ))
                            
            return diffs
            
        except subprocess.SubprocessError:
            return []
            
    def sort_files_by_changes(
        self,
        files: List[str],
        max_commits: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Sort files by change frequency (repomix-compatible).
        
        Returns list of (filename, change_score) tuples sorted by score (desc).
        """
        change_frequencies = self.get_file_change_frequencies(max_commits, files)
        
        if not change_frequencies:
            # Fallback to file modification time if git not available
            scored_files = []
            for file_path in files:
                try:
                    full_path = self.repo_path / file_path
                    if full_path.exists():
                        mtime = full_path.stat().st_mtime
                        # Convert to recency score (newer = higher)
                        days_old = (time.time() - mtime) / (24 * 3600)
                        score = max(0, 1.0 - (days_old / 365))  # Score decays over 1 year
                        scored_files.append((file_path, score))
                except (OSError, AttributeError):
                    scored_files.append((file_path, 0.0))
                    
            return sorted(scored_files, key=lambda x: x[1], reverse=True)
        
        # Use git change frequency data
        scored_files = []
        for file_path in files:
            if file_path in change_frequencies:
                info = change_frequencies[file_path]
                # Combine change count with recency
                days_since_modified = (time.time() - info.last_modified) / (24 * 3600)
                recency_bonus = max(0, 1.0 - (days_since_modified / 180))  # 6 month decay
                score = info.change_count * (1.0 + recency_bonus)
                scored_files.append((file_path, score))
            else:
                scored_files.append((file_path, 0.0))
                
        return sorted(scored_files, key=lambda x: x[1], reverse=True)


class RemoteRepoHandler:
    """Handle remote repository cloning and processing."""
    
    @staticmethod
    def is_remote_url(url: str) -> bool:
        """Check if URL is a remote git repository."""
        parsed = urlparse(url)
        return (parsed.scheme in ('http', 'https', 'git', 'ssh') or
                url.startswith('git@') or
                'github.com' in url or
                'gitlab.com' in url or
                'bitbucket.org' in url)
                
    @staticmethod
    def clone_repository(
        url: str, 
        branch: Optional[str] = None,
        depth: Optional[int] = 1
    ) -> Optional[Path]:
        """
        Clone remote repository to temporary directory.
        
        Args:
            url: Git repository URL
            branch: Specific branch/tag/commit to checkout
            depth: Clone depth (1 for shallow clone)
            
        Returns:
            Path to cloned repository or None if failed
        """
        try:
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp(prefix='scribe_remote_'))
            
            # Build git clone command
            cmd = ['git', 'clone']
            
            if depth:
                cmd.extend(['--depth', str(depth)])
                
            if branch:
                cmd.extend(['--branch', branch])
                
            cmd.extend([url, str(temp_dir / 'repo')])
            
            # Clone repository
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None
                
            repo_path = temp_dir / 'repo'
            return repo_path if repo_path.exists() else None
            
        except (subprocess.SubprocessError, OSError):
            return None
            
    @staticmethod
    def cleanup_repository(repo_path: Path):
        """Clean up cloned repository."""
        if repo_path.exists():
            # Remove parent temp directory
            temp_dir = repo_path.parent
            shutil.rmtree(temp_dir, ignore_errors=True)


def create_git_integration(repo_path: Path) -> GitIntegration:
    """Create git integration instance."""
    return GitIntegration(repo_path)


# Example usage
if __name__ == "__main__":
    # Test git integration
    repo_path = Path.cwd()
    git = GitIntegration(repo_path)
    
    if git.is_git_repo():
        print("Git repository detected")
        
        # Test change frequencies
        files = ['README.md', 'main.py']
        frequencies = git.get_file_change_frequencies(max_commits=10, file_paths=files)
        for filename, info in frequencies.items():
            print(f"{filename}: {info.change_count} changes, {info.author_count} authors")
            
        # Test file sorting
        sorted_files = git.sort_files_by_changes(files)
        print("Files by change frequency:")
        for filename, score in sorted_files:
            print(f"  {filename}: {score:.2f}")
    else:
        print("Not a git repository")
        
    # Test remote URL detection
    remote_urls = [
        'https://github.com/user/repo.git',
        'git@github.com:user/repo.git',
        '/local/path/repo'
    ]
    
    for url in remote_urls:
        is_remote = RemoteRepoHandler.is_remote_url(url)
        print(f"{url}: {'REMOTE' if is_remote else 'LOCAL'}")
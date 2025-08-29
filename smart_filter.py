#!/usr/bin/env python3
"""
Smart Filtering System for Scribe

This module provides intelligent file filtering that automatically excludes noise
while preserving all important source code files. It uses heuristics and patterns
to identify generated files, build artifacts, and other non-essential content.
"""

import pathlib
import re
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False


class FilterReason(Enum):
    """Reasons why a file might be filtered out."""
    INCLUDED = "included"
    BINARY_EXTENSION = "binary_extension"  
    BINARY_CONTENT = "binary_content"
    TOO_LARGE = "too_large"
    BUILD_ARTIFACT = "build_artifact"
    GENERATED_FILE = "generated_file"
    TEMPORARY_FILE = "temporary_file"
    LOG_FILE = "log_file"
    CONFIG_CACHE = "config_cache"
    DOCUMENTATION_ASSET = "documentation_asset"
    VCS_FILE = "vcs_file"
    IDE_FILE = "ide_file"
    PACKAGE_MANAGER = "package_manager"
    VIRTUAL_ENV = "virtual_env"


@dataclass
class FilterDecision:
    """Decision about whether to include a file."""
    include: bool
    reason: FilterReason
    confidence: float  # 0-1, how confident we are in this decision
    details: Optional[str] = None


class SmartFilter:
    """Intelligent file filtering system for scribe."""
    
    def __init__(self, max_file_size: int = 1024 * 1024):  # 1MB default
        self.max_file_size = max_file_size
        
        # Initialize python-magic if available
        self._init_magic()
        
        # Core patterns for different categories
        self._init_filter_patterns()

    
    def _init_magic(self):
        """Initialize python-magic objects if available."""
        if MAGIC_AVAILABLE:
            try:
                self.magic_mime = magic.Magic(mime=True)
                self.magic_description = magic.Magic()
                self.magic_available = True
            except Exception:
                # Fallback if magic fails to initialize
                self.magic_available = False
        else:
            self.magic_available = False
    
    def _init_filter_patterns(self):
        """Initialize all filtering patterns."""
        
        # Binary file extensions (definitive exclusions)
        self.binary_extensions = {
            # Images
            '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico', '.svg',
            '.tiff', '.tif', '.raw', '.psd', '.ai', '.eps',
            
            # Audio/Video
            '.mp3', '.mp4', '.avi', '.mov', '.mkv', '.wav', '.ogg', '.flac',
            '.m4a', '.aac', '.wma', '.wmv', '.flv', '.webm',
            
            # Archives
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar', '.dmg', '.iso',
            
            # Fonts
            '.ttf', '.otf', '.eot', '.woff', '.woff2',
            
            # Executables and libraries
            '.exe', '.dll', '.so', '.dylib', '.bin', '.app',
            '.class', '.jar', '.war', '.ear',
            
            # Compiled Python
            '.pyc', '.pyo', '.pyd',
            
            # Documents (usually not source code)
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        }
        
        # Directory patterns to exclude entirely
        self.excluded_directories = {
            # Version control
            '.git', '.svn', '.hg', '.bzr',
            
            # Build outputs
            'build', 'dist', 'out', 'target', 'bin', 'obj',
            
            # Dependencies
            'node_modules', 'vendor', 'Pods',
            
            # Python environments and caches
            '__pycache__', '.pytest_cache', '.mypy_cache', '.tox', '.nox',
            'venv', '.venv', 'env', '.env', 'site-packages',
            
            # IDE and editor files
            '.idea', '.vscode', '.vs', '.settings',
            
            # Coverage and testing artifacts
            'coverage', '.coverage', 'htmlcov', '.nyc_output',
            
            # Package manager caches
            '.npm', '.yarn', '.pnpm-store', '.gradle', '.m2',
            
            # System directories
            '.DS_Store', '.Trash', 'Thumbs.db',
        }
        
        # File patterns that indicate generated content
        self.generated_patterns = [
            # Auto-generated markers
            re.compile(r'.*\.generated\.\w+$'),
            re.compile(r'.*\.gen\.\w+$'),
            re.compile(r'.*_pb2\.py$'),  # Protocol buffers
            re.compile(r'.*_pb2_grpc\.py$'),
            re.compile(r'.*\.pb\.go$'),
            
            # Build outputs
            re.compile(r'.*\.min\.\w+$'),
            re.compile(r'.*\.bundle\.\w+$'),
            re.compile(r'.*\.chunk\.\w+$'),
            
            # Lockfiles (keep package.json but exclude lock files)
            re.compile(r'.*lock\.json$'),
            re.compile(r'.*yarn\.lock$'),
            re.compile(r'.*pnpm-lock\.yaml$'),
            re.compile(r'.*Gemfile\.lock$'),
            re.compile(r'.*Pipfile\.lock$'),
            re.compile(r'.*poetry\.lock$'),
            re.compile(r'.*Cargo\.lock$'),
        ]
        
        # Temporary and log file patterns
        self.temporary_patterns = [
            re.compile(r'.*\.tmp$'),
            re.compile(r'.*\.temp$'),
            re.compile(r'.*\.log$'),
            re.compile(r'.*\.bak$'),
            re.compile(r'.*\.backup$'),
            re.compile(r'.*~$'),
            re.compile(r'.*\.swp$'),
            re.compile(r'.*\.swo$'),
        ]
        
        # Source code extensions (high priority for inclusion)
        self.source_extensions = {
            # Programming languages
            '.py', '.js', '.ts', '.jsx', '.tsx',
            '.go', '.rs', '.java', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
            '.cs', '.rb', '.php', '.kt', '.swift', '.scala', '.r',
            '.m', '.mm', '.sh', '.bash', '.zsh', '.fish', '.ps1',
            
            # Web technologies
            '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
            
            # Template engines (commonly used in web development)
            '.njk', '.nunjucks',  # Nunjucks templates
            '.hbs', '.handlebars',  # Handlebars templates  
            '.mustache',  # Mustache templates
            '.ejs',  # Embedded JavaScript templates
            '.pug', '.jade',  # Pug templates (formerly Jade)
            '.liquid',  # Liquid templates (Shopify, Jekyll)
            '.erb',  # ERB templates (Ruby)
            '.twig',  # Twig templates (PHP)
            '.j2', '.jinja', '.jinja2',  # Jinja2 templates (Python)
            '.dust',  # Dust templates
            '.eta',  # Eta templates
            
            # Data and config (often important for understanding)
            '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.env',
            '.sql', '.graphql', '.proto',
            
            # Documentation (selective inclusion)
            '.md', '.rst', '.txt', '.adoc',
            
            # Build and project files
            'Dockerfile', 'Makefile', '.gitignore', '.dockerignore',
        }
        
        # Files that are always important (override size limits)
        self.critical_files = {
            'README.md', 'README.rst', 'README.txt', 'README',
            'CHANGELOG.md', 'CHANGELOG.rst', 'CHANGELOG.txt',
            'LICENSE', 'LICENSE.txt', 'LICENSE.md',
            'package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod',
            'requirements.txt', 'Pipfile', 'Gemfile',
            'tsconfig.json', 'babel.config.js', 'webpack.config.js',
            'Dockerfile', 'docker-compose.yml', 'Makefile',
            '.gitignore', '.gitattributes',
        }
    
    def should_include_file(self, file_path: pathlib.Path, repo_root: pathlib.Path) -> FilterDecision:
        """Determine if a file should be included in scribe output."""
        try:
            # Get relative path for pattern matching
            rel_path = file_path.relative_to(repo_root)
            filename = file_path.name.lower()
            extension = file_path.suffix.lower()
            
            # Check if file is in excluded directory
            for part in rel_path.parts:
                if part in self.excluded_directories or part.startswith('.'):
                    # Exception for some dotfiles that are important
                    if filename not in {'.gitignore', '.gitattributes', '.env.example'}:
                        return FilterDecision(
                            include=False,
                            reason=FilterReason.VCS_FILE if part.startswith('.git') else FilterReason.BUILD_ARTIFACT,
                            confidence=0.95,
                            details=f"In excluded directory: {part}"
                        )
            
            # Get file size
            try:
                file_size = file_path.stat().st_size
            except (OSError, PermissionError):
                return FilterDecision(
                    include=False,
                    reason=FilterReason.VCS_FILE,
                    confidence=0.9,
                    details="Cannot access file"
                )
            
            # Skip empty files
            if file_size == 0:
                return FilterDecision(
                    include=False,
                    reason=FilterReason.TEMPORARY_FILE,
                    confidence=0.8,
                    details="Empty file"
                )
            
            # Critical files are always included (with size limit exception)
            if filename in self.critical_files:
                if file_size <= self.max_file_size * 5:  # Allow 5x size for critical files
                    return FilterDecision(
                        include=True,
                        reason=FilterReason.INCLUDED,
                        confidence=1.0,
                        details="Critical project file"
                    )
            
            # Check for binary extensions
            if extension in self.binary_extensions:
                return FilterDecision(
                    include=False,
                    reason=FilterReason.BINARY_EXTENSION,
                    confidence=0.95,
                    details=f"Binary extension: {extension}"
                )
            
            # Check file size before reading content
            if file_size > self.max_file_size:
                # Allow larger source files but warn
                if extension in self.source_extensions:
                    return FilterDecision(
                        include=False,
                        reason=FilterReason.TOO_LARGE,
                        confidence=0.7,
                        details=f"Large source file: {file_size} bytes"
                    )
                else:
                    return FilterDecision(
                        include=False,
                        reason=FilterReason.TOO_LARGE,
                        confidence=0.9,
                        details=f"File too large: {file_size} bytes"
                    )
            
            # Check for generated file patterns
            rel_path_str = str(rel_path)
            for pattern in self.generated_patterns:
                if pattern.match(rel_path_str):
                    return FilterDecision(
                        include=False,
                        reason=FilterReason.GENERATED_FILE,
                        confidence=0.8,
                        details=f"Matches generated pattern"
                    )
            
            # Check for temporary file patterns
            for pattern in self.temporary_patterns:
                if pattern.match(rel_path_str):
                    return FilterDecision(
                        include=False,
                        reason=FilterReason.TEMPORARY_FILE,
                        confidence=0.9,
                        details="Temporary file pattern"
                    )
            
            # Check if file contains binary content
            if self._contains_binary_content(file_path):
                return FilterDecision(
                    include=False,
                    reason=FilterReason.BINARY_CONTENT,
                    confidence=0.85,
                    details="Contains binary content"
                )
            
            # Check if file looks generated by inspecting content
            if extension in self.source_extensions or not extension:
                generated_confidence = self._analyze_generated_content(file_path)
                if generated_confidence > 0.7:
                    return FilterDecision(
                        include=False,
                        reason=FilterReason.GENERATED_FILE,
                        confidence=generated_confidence,
                        details="Content appears generated"
                    )
            
            # If it's a known source extension, include it
            if extension in self.source_extensions:
                return FilterDecision(
                    include=True,
                    reason=FilterReason.INCLUDED,
                    confidence=0.9,
                    details=f"Source code file: {extension}"
                )
            
            # For unknown extensions, be more conservative but still include text files
            if self._appears_to_be_text(file_path):
                return FilterDecision(
                    include=True,
                    reason=FilterReason.INCLUDED,
                    confidence=0.6,
                    details="Appears to be text file"
                )
            
            # Default to exclude unknown files
            return FilterDecision(
                include=False,
                reason=FilterReason.BINARY_CONTENT,
                confidence=0.5,
                details="Unknown file type"
            )
            
        except Exception as e:
            return FilterDecision(
                include=False,
                reason=FilterReason.VCS_FILE,
                confidence=0.9,
                details=f"Error analyzing file: {e}"
            )
    
    def _contains_binary_content(self, file_path: pathlib.Path) -> bool:
        """Check if file contains binary content using magic if available."""
        try:
            # Use python-magic if available for accurate detection
            if self.magic_available:
                try:
                    mime_type = self.magic_mime.from_file(str(file_path))
                    # Text files should have text/ mime types
                    if mime_type.startswith('text/'):
                        return False
                    # Some common text types that don't start with text/
                    text_mimes = {
                        'application/json',
                        'application/xml',
                        'application/javascript',
                        'application/x-sh',
                        'application/x-shellscript',
                        'inode/x-empty',  # Empty files
                    }
                    if mime_type in text_mimes:
                        return False
                    # If mime type suggests binary, it's binary
                    binary_prefixes = ('image/', 'video/', 'audio/', 'application/octet-stream', 'application/pdf')
                    if any(mime_type.startswith(prefix) for prefix in binary_prefixes):
                        return True
                except Exception:
                    # Fallback to heuristic detection if magic fails
                    pass
            
            # Fallback to heuristic detection
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)  # Read first 8KB
            
            # If contains null bytes, it's binary
            if b'\x00' in chunk:
                return True
            
            # Check for high ratio of non-printable characters
            if len(chunk) > 0:
                printable_chars = sum(1 for b in chunk if 32 <= b <= 126 or b in (9, 10, 13))
                ratio = printable_chars / len(chunk)
                return ratio < 0.75  # If less than 75% printable, consider binary
            
            return False
            
        except (OSError, PermissionError):
            return True  # If we can't read it, assume binary  # If we can't read it, assume binary
    
    def _analyze_generated_content(self, file_path: pathlib.Path) -> float:
        """Analyze file content to determine if it's generated. Returns confidence 0-1."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Read first few lines and last few lines
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i > 50:  # Don't read entire large files
                        break
            
            if not lines:
                return 0.0
            
            confidence = 0.0
            
            # Check for generated file markers
            header_text = '\n'.join(lines[:10]).lower()
            generated_markers = [
                'auto-generated', 'autogenerated', 'automatically generated',
                'do not edit', 'do not modify', 'generated by',
                'this file is generated', 'code generated by',
                'machine generated', 'computer generated'
            ]
            
            for marker in generated_markers:
                if marker in header_text:
                    confidence += 0.4
            
            # Check for repetitive patterns (common in generated code)
            if len(set(lines)) < len(lines) * 0.3:  # High repetition
                confidence += 0.2
            
            # Check for very long lines (common in minified/generated files)
            long_lines = sum(1 for line in lines if len(line) > 200)
            if long_lines > len(lines) * 0.1:  # > 10% are very long lines
                confidence += 0.3
            
            # Check for lack of comments (generated code often has no comments)
            comment_patterns = ['#', '//', '/*', '*/', '<!--', '-->', "'''", '"""']
            has_comments = any(any(pattern in line for pattern in comment_patterns) for line in lines)
            if not has_comments and len(lines) > 10:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except (OSError, UnicodeDecodeError, PermissionError):
            return 0.0
    
    def _appears_to_be_text(self, file_path: pathlib.Path) -> bool:
        """Check if file appears to be a text file using magic-enhanced detection."""
        try:
            # Use python-magic if available for accurate detection
            if self.magic_available:
                try:
                    mime_type = self.magic_mime.from_file(str(file_path))
                    # Text files should have text/ mime types
                    if mime_type.startswith('text/'):
                        return True
                    # Some common text types that don't start with text/
                    text_mimes = {
                        'application/json',
                        'application/xml',
                        'application/javascript',
                        'application/x-sh',
                        'application/x-shellscript',
                        'application/x-python',
                        'application/x-perl',
                        'application/x-ruby',
                        'inode/x-empty',  # Empty files
                        # Template files might be detected as text/plain or application/octet-stream
                        # so we'll also check the fallback heuristics for unknown types
                    }
                    if mime_type in text_mimes:
                        return True
                    # For application/octet-stream or other ambiguous types,
                    # fall through to heuristic detection
                    if mime_type in ('application/octet-stream', 'text/plain'):
                        # Continue to heuristic check below
                        pass
                    else:
                        # For clearly binary types, return False
                        binary_prefixes = ('image/', 'video/', 'audio/', 'application/pdf', 'application/zip')
                        if any(mime_type.startswith(prefix) for prefix in binary_prefixes):
                            return False
                except Exception:
                    # Fallback to heuristic detection if magic fails
                    pass
            
            # Fallback heuristic detection (always run for ambiguous mime types)
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)  # Read first 1KB
            
            if not chunk:
                return True  # Empty files are text
            
            # Try to decode as UTF-8
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                pass
            
            # Check if it's mostly printable ASCII
            printable = sum(1 for b in chunk if 32 <= b <= 126 or b in (9, 10, 13))
            return printable / len(chunk) > 0.8
            
        except (OSError, PermissionError):
            return False

    
    def get_file_info(self, file_path: pathlib.Path) -> Dict[str, str]:
        """Get detailed file type information using magic if available."""
        info = {
            'extension': file_path.suffix.lower(),
            'mime_type': 'unknown',
            'description': 'unknown',
            'detection_method': 'extension_only'
        }
        
        try:
            if self.magic_available:
                try:
                    info['mime_type'] = self.magic_mime.from_file(str(file_path))
                    info['description'] = self.magic_description.from_file(str(file_path))
                    info['detection_method'] = 'magic'
                except Exception as e:
                    info['detection_method'] = f'magic_failed: {e}'
            
            # Add our own classification
            if info['extension'] in self.source_extensions:
                info['scribe_classification'] = 'source_code'
            elif info['extension'] in self.binary_extensions:
                info['scribe_classification'] = 'binary'
            else:
                info['scribe_classification'] = 'unknown'
                
        except Exception as e:
            info['error'] = str(e)
            
        return info
    
    def filter_files(self, files: List[pathlib.Path], repo_root: pathlib.Path) -> Tuple[List[pathlib.Path], Dict[FilterReason, int]]:
        """Filter a list of files, returning included files and statistics."""
        included_files = []
        stats = {reason: 0 for reason in FilterReason}
        
        for file_path in files:
            decision = self.should_include_file(file_path, repo_root)
            stats[decision.reason] += 1
            
            if decision.include:
                included_files.append(file_path)
        
        return included_files, stats
    
    def get_exclusion_patterns_for_scribe(self) -> List[str]:
        """Get exclusion patterns in format suitable for scribe configuration."""
        patterns = []
        
        # Add directory exclusions
        for dirname in self.excluded_directories:
            patterns.extend([
                f"**/{dirname}/**",
                f"{dirname}/**",
                f"**/{dirname}",
            ])
        
        # Add file extension exclusions
        for ext in self.binary_extensions:
            patterns.append(f"**/*{ext}")
        
        # Add generated file patterns
        patterns.extend([
            "**/*.generated.*",
            "**/*.gen.*",
            "**/*_pb2.py",
            "**/*_pb2_grpc.py",
            "**/*.pb.go",
            "**/*.min.*",
            "**/*.bundle.*",
            "**/yarn.lock",
            "**/package-lock.json",
            "**/pnpm-lock.yaml",
            "**/Pipfile.lock",
            "**/poetry.lock",
            "**/Cargo.lock",
            "**/*.tmp",
            "**/*.log",
            "**/*.bak",
            "**/*~",
            "**/*.swp",
        ])
        
        return patterns


def create_smart_filter(max_file_size: int = 1024 * 1024) -> SmartFilter:
    """Factory function to create a SmartFilter instance."""
    return SmartFilter(max_file_size)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python smart_filter.py <repository_path>")
        sys.exit(1)
    
    repo_path = pathlib.Path(sys.argv[1])
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    # Test the smart filter
    smart_filter = SmartFilter()
    
    # Collect all files
    all_files = []
    for root, dirs, filenames in repo_path.walk():
        for filename in filenames:
            all_files.append(root / filename)
    
    print(f"📁 Found {len(all_files)} total files")
    
    # Filter files
    included_files, stats = smart_filter.filter_files(all_files, repo_path)
    
    print(f"✅ Included: {len(included_files)} files")
    print(f"❌ Excluded: {len(all_files) - len(included_files)} files")
    print(f"\nExclusion breakdown:")
    for reason, count in stats.items():
        if count > 0 and reason != FilterReason.INCLUDED:
            print(f"  {reason.value}: {count}")
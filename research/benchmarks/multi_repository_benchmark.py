#!/usr/bin/env python3
"""
Multi-Repository Benchmark Framework
====================================

Comprehensive framework for evaluating retrieval systems across diverse repository types.
Supports automatic repository discovery, classification, and evaluation dataset generation.

Repository Types Supported:
- Web Applications (React, Django, Rails)
- CLI Tools (Command-line utilities)
- Libraries (Open source packages)
- Data Science (Jupyter notebooks, ML projects)
- Documentation-heavy (Docs, guides, wikis)
"""

import os
import json
import shutil
import random
import hashlib
import tempfile
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
import numpy as np
from github import Github
from git import Repo as GitRepo


class RepositoryType(Enum):
    """Types of repositories for evaluation."""
    WEB_APPLICATION = "web_application"
    CLI_TOOL = "cli_tool"
    LIBRARY = "library"
    DATA_SCIENCE = "data_science"
    DOCUMENTATION_HEAVY = "documentation_heavy"


@dataclass
class RepositoryMetadata:
    """Metadata about a repository."""
    name: str
    owner: str
    url: str
    description: str
    language: str
    stars: int
    forks: int
    size_kb: int
    last_updated: str
    topics: List[str]
    
    # Analysis metadata
    file_count: int = 0
    code_file_count: int = 0
    doc_file_count: int = 0
    test_file_count: int = 0
    config_file_count: int = 0
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    
    # Complexity metrics
    cyclomatic_complexity: float = 0.0
    average_file_size: float = 0.0
    dependency_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class Repository:
    """Represents a repository for evaluation."""
    
    def __init__(
        self, 
        metadata: RepositoryMetadata, 
        repo_type: RepositoryType,
        local_path: Optional[str] = None
    ):
        self.metadata = metadata
        self.type = repo_type
        self.local_path = local_path
        self.id = f"{metadata.owner}_{metadata.name}"
        self.prepared = False
        
        # Cached data
        self._text_files_cache = None
        self._file_contents_cache = {}
        self._analysis_cache = {}
    
    @property
    def name(self) -> str:
        """Repository name."""
        return f"{self.metadata.owner}/{self.metadata.name}"
    
    def prepare_for_evaluation(self, base_dir: str = "./temp_repos") -> None:
        """Prepare repository for evaluation (clone, analyze, cache)."""
        if self.prepared:
            return
        
        try:
            # Create base directory
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            
            # Clone repository if not already local
            if not self.local_path:
                self.local_path = self._clone_repository(base_dir)
            
            # Analyze repository structure
            self._analyze_repository_structure()
            
            # Generate file cache
            self._cache_text_files()
            
            self.prepared = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to prepare repository {self.name}: {str(e)}")
    
    def _clone_repository(self, base_dir: str) -> str:
        """Clone repository to local directory."""
        repo_dir = Path(base_dir) / f"{self.metadata.owner}_{self.metadata.name}"
        
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        
        # Clone with shallow history for speed
        GitRepo.clone_from(
            self.metadata.url,
            str(repo_dir),
            depth=1
        )
        
        return str(repo_dir)
    
    def _analyze_repository_structure(self) -> None:
        """Analyze repository structure and update metadata."""
        if not self.local_path or not Path(self.local_path).exists():
            return
        
        repo_path = Path(self.local_path)
        
        # Count files by type
        file_counts = {
            'total': 0,
            'code': 0,
            'doc': 0,
            'test': 0,
            'config': 0
        }
        
        line_counts = {
            'total': 0,
            'code': 0,
            'comment': 0,
            'blank': 0
        }
        
        total_file_size = 0
        
        # Define file type patterns
        code_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.clj',
            '.html', '.css', '.scss', '.less', '.vue', '.jsx', '.tsx'
        }
        
        doc_extensions = {
            '.md', '.rst', '.txt', '.adoc', '.org'
        }
        
        config_extensions = {
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.xml'
        }
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    total_file_size += file_size
                    file_counts['total'] += 1
                    
                    # Classify file type
                    extension = file_path.suffix.lower()
                    file_name = file_path.name.lower()
                    
                    if 'test' in file_name or 'spec' in file_name:
                        file_counts['test'] += 1
                    elif extension in code_extensions:
                        file_counts['code'] += 1
                    elif extension in doc_extensions:
                        file_counts['doc'] += 1
                    elif extension in config_extensions:
                        file_counts['config'] += 1
                    
                    # Analyze lines if text file
                    if file_size < 1024 * 1024:  # Skip large files
                        try:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            lines = content.split('\n')
                            
                            line_counts['total'] += len(lines)
                            
                            for line in lines:
                                stripped = line.strip()
                                if not stripped:
                                    line_counts['blank'] += 1
                                elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                                    line_counts['comment'] += 1
                                else:
                                    line_counts['code'] += 1
                                    
                        except Exception:
                            pass
                            
                except Exception:
                    continue
        
        # Update metadata
        self.metadata.file_count = file_counts['total']
        self.metadata.code_file_count = file_counts['code']
        self.metadata.doc_file_count = file_counts['doc']
        self.metadata.test_file_count = file_counts['test']
        self.metadata.config_file_count = file_counts['config']
        self.metadata.total_lines = line_counts['total']
        self.metadata.code_lines = line_counts['code']
        self.metadata.comment_lines = line_counts['comment']
        self.metadata.blank_lines = line_counts['blank']
        self.metadata.average_file_size = total_file_size / max(file_counts['total'], 1)
    
    def _cache_text_files(self) -> None:
        """Cache list of text files."""
        if not self.local_path:
            return
        
        repo_path = Path(self.local_path)
        text_files = []
        
        # Text file patterns
        text_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.clj',
            '.html', '.css', '.scss', '.less', '.vue', '.jsx', '.tsx',
            '.md', '.rst', '.txt', '.json', '.yaml', '.yml', '.toml',
            '.ini', '.cfg', '.conf', '.xml', '.sql', '.sh', '.bat'
        }
        
        # Skip directories
        skip_dirs = {
            '.git', '.svn', '.hg', '__pycache__', '.pytest_cache',
            'node_modules', 'venv', 'env', '.venv', '.env',
            'build', 'dist', 'target', 'bin', 'obj',
            '.idea', '.vscode', '.vs'
        }
        
        for file_path in repo_path.rglob('*'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                # Skip very large files
                if file_path.stat().st_size > 1024 * 1024:  # 1MB
                    continue
                    
                relative_path = str(file_path.relative_to(repo_path))
                text_files.append(relative_path)
        
        self._text_files_cache = sorted(text_files)
    
    def get_text_files(self) -> List[str]:
        """Get list of text files in repository."""
        if self._text_files_cache is None:
            self._cache_text_files()
        return self._text_files_cache or []
    
    def read_file(self, file_path: str) -> str:
        """Read file content."""
        if file_path in self._file_contents_cache:
            return self._file_contents_cache[file_path]
        
        if not self.local_path:
            return ""
        
        full_path = Path(self.local_path) / file_path
        
        try:
            if full_path.exists() and full_path.is_file():
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                self._file_contents_cache[file_path] = content
                return content
        except Exception:
            pass
        
        return ""
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get repository metadata."""
        return self.metadata.to_dict()
    
    def generate_questions(
        self, 
        question_type: str, 
        count: int = 10, 
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Generate evaluation questions for this repository."""
        random.seed(random_seed)
        
        generators = {
            'architectural': self._generate_architectural_questions,
            'implementation_detail': self._generate_implementation_questions,
            'bug_analysis': self._generate_bug_analysis_questions,
            'feature_request': self._generate_feature_questions,
            'documentation_query': self._generate_documentation_questions
        }
        
        generator = generators.get(question_type, self._generate_generic_questions)
        return generator(count)
    
    def _generate_architectural_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate architectural questions."""
        questions = []
        
        templates = [
            "How is the project structured and what are the main components?",
            "What architectural patterns are used in this codebase?",
            "How do the different modules interact with each other?",
            "What is the data flow through the application?",
            "How is dependency injection handled?",
            "What is the overall system architecture?",
            "How are external services integrated?",
            "What design patterns are implemented?"
        ]
        
        for i in range(count):
            template = random.choice(templates)
            questions.append({
                'question': template,
                'type': 'architectural',
                'answer': f"Architectural answer for {self.name} regarding: {template}",
                'difficulty': random.choice(['easy', 'medium', 'hard']),
                'expected_files': self._get_relevant_files_for_architecture()
            })
        
        return questions
    
    def _generate_implementation_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate implementation detail questions."""
        questions = []
        
        # Get some actual files to base questions on
        files = self.get_text_files()
        code_files = [f for f in files if any(ext in f for ext in ['.py', '.js', '.java', '.cpp'])]
        
        templates = [
            "How does the {function_name} function work?",
            "What is the purpose of the {file_name} file?",
            "How is {feature_name} implemented?",
            "What algorithms are used in {component_name}?",
            "How does error handling work in {module_name}?",
            "What data structures are used for {purpose}?",
            "How is {functionality} optimized for performance?",
            "What is the implementation strategy for {feature}?"
        ]
        
        for i in range(count):
            template = random.choice(templates)
            file_name = random.choice(code_files) if code_files else "main"
            
            question = template.format(
                function_name=self._extract_sample_function_name(file_name),
                file_name=Path(file_name).stem,
                feature_name=self._generate_feature_name(),
                component_name=self._generate_component_name(),
                module_name=Path(file_name).stem,
                purpose=random.choice(['storage', 'processing', 'caching', 'validation']),
                functionality=random.choice(['authentication', 'data processing', 'API calls']),
                feature=self._generate_feature_name()
            )
            
            questions.append({
                'question': question,
                'type': 'implementation_detail',
                'answer': f"Implementation details for {question}",
                'difficulty': random.choice(['easy', 'medium', 'hard']),
                'expected_files': [file_name] if code_files else []
            })
        
        return questions
    
    def _generate_bug_analysis_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate bug analysis questions."""
        questions = []
        
        bug_templates = [
            "The application crashes when {scenario}. What could be causing this?",
            "Why is {feature} not working correctly?",
            "Performance is slow when {operation}. How can this be optimized?",
            "Memory usage increases over time. Where might the memory leak be?",
            "The {component} fails under high load. What are potential issues?",
            "Error {error_code} occurs during {operation}. What's the root cause?",
            "The {functionality} produces incorrect results. What might be wrong?",
            "Intermittent failures happen in {scenario}. How to debug this?"
        ]
        
        for i in range(count):
            template = random.choice(bug_templates)
            question = template.format(
                scenario=random.choice(['user login', 'data import', 'file upload', 'API request']),
                feature=self._generate_feature_name(),
                operation=random.choice(['data processing', 'file operations', 'network requests']),
                component=self._generate_component_name(),
                error_code=random.choice(['404', '500', 'TimeoutError', 'ValidationError']),
                functionality=random.choice(['authentication', 'data validation', 'file processing'])
            )
            
            questions.append({
                'question': question,
                'type': 'bug_analysis',
                'answer': f"Bug analysis for: {question}",
                'difficulty': random.choice(['medium', 'hard']),
                'expected_files': self._get_relevant_files_for_debugging()
            })
        
        return questions
    
    def _generate_feature_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate feature request questions."""
        questions = []
        
        feature_templates = [
            "How would you implement {feature} in this codebase?",
            "What changes are needed to add {functionality}?",
            "How can {existing_feature} be extended to support {new_capability}?",
            "What architecture changes are needed for {feature}?",
            "How would you integrate {external_service} into this system?",
            "What is the best approach to implement {user_story}?",
            "How can performance be improved for {operation}?",
            "What security measures are needed for {feature}?"
        ]
        
        for i in range(count):
            template = random.choice(feature_templates)
            question = template.format(
                feature=random.choice(['user notifications', 'data export', 'real-time updates', 'multi-tenancy']),
                functionality=random.choice(['caching', 'logging', 'monitoring', 'backup']),
                existing_feature=self._generate_feature_name(),
                new_capability=random.choice(['mobile support', 'offline mode', 'bulk operations']),
                external_service=random.choice(['payment gateway', 'email service', 'analytics']),
                user_story=random.choice(['user dashboard', 'admin panel', 'reporting system']),
                operation=random.choice(['data queries', 'file uploads', 'batch processing'])
            )
            
            questions.append({
                'question': question,
                'type': 'feature_request',
                'answer': f"Feature implementation plan for: {question}",
                'difficulty': random.choice(['medium', 'hard']),
                'expected_files': self._get_relevant_files_for_features()
            })
        
        return questions
    
    def _generate_documentation_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate documentation questions."""
        questions = []
        
        doc_templates = [
            "How do you install and set up this project?",
            "What are the main configuration options?",
            "How do you run the tests?",
            "What are the API endpoints and how do you use them?",
            "How do you contribute to this project?",
            "What are the system requirements?",
            "How do you deploy this application?",
            "What troubleshooting steps are available?"
        ]
        
        for i in range(count):
            question = random.choice(doc_templates)
            
            questions.append({
                'question': question,
                'type': 'documentation_query',
                'answer': f"Documentation answer for: {question}",
                'difficulty': 'easy',
                'expected_files': self._get_relevant_documentation_files()
            })
        
        return questions
    
    def _generate_generic_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate generic questions."""
        return self._generate_implementation_questions(count)
    
    def _extract_sample_function_name(self, file_path: str) -> str:
        """Extract a sample function name from file."""
        try:
            content = self.read_file(file_path)
            # Simple regex to find function names
            import re
            functions = re.findall(r'def\s+(\w+)|function\s+(\w+)', content)
            if functions:
                return next(name for group in functions for name in group if name)
        except Exception:
            pass
        return "process_data"
    
    def _generate_feature_name(self) -> str:
        """Generate a realistic feature name."""
        features = [
            'user authentication', 'data validation', 'file processing',
            'API integration', 'caching system', 'logging framework',
            'error handling', 'configuration management', 'testing utilities'
        ]
        return random.choice(features)
    
    def _generate_component_name(self) -> str:
        """Generate a realistic component name."""
        components = [
            'authentication module', 'database layer', 'API handler',
            'file processor', 'cache manager', 'configuration loader',
            'error handler', 'logging system', 'validation engine'
        ]
        return random.choice(components)
    
    def _get_relevant_files_for_architecture(self) -> List[str]:
        """Get files relevant for architectural questions."""
        files = self.get_text_files()
        return [f for f in files if any(pattern in f.lower() for pattern in [
            'main', 'app', 'server', 'client', 'config', 'setup', 'init'
        ])][:5]
    
    def _get_relevant_files_for_debugging(self) -> List[str]:
        """Get files relevant for debugging questions."""
        files = self.get_text_files()
        return [f for f in files if any(pattern in f.lower() for pattern in [
            'error', 'exception', 'log', 'debug', 'test', 'util'
        ])][:3]
    
    def _get_relevant_files_for_features(self) -> List[str]:
        """Get files relevant for feature questions."""
        files = self.get_text_files()
        code_files = [f for f in files if any(ext in f for ext in ['.py', '.js', '.java', '.cpp'])]
        return code_files[:5]
    
    def _get_relevant_documentation_files(self) -> List[str]:
        """Get documentation files."""
        files = self.get_text_files()
        return [f for f in files if any(ext in f for ext in ['.md', '.rst', '.txt'])][:3]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'metadata': self.metadata.to_dict(),
            'local_path': self.local_path,
            'prepared': self.prepared,
            'text_file_count': len(self.get_text_files())
        }


class RepositoryBenchmark:
    """
    Main benchmark framework for multi-repository evaluation.
    
    Handles repository discovery, classification, and evaluation coordination.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token for higher API limits."""
        self.github_token = github_token
        self.github_client = Github(github_token) if github_token else Github()
        
        # Repository classification patterns
        self.classification_patterns = {
            RepositoryType.WEB_APPLICATION: {
                'topics': ['web', 'webapp', 'react', 'angular', 'vue', 'django', 'rails', 'express'],
                'keywords': ['react', 'angular', 'vue', 'django', 'rails', 'express', 'webapp', 'frontend', 'backend'],
                'files': ['package.json', 'requirements.txt', 'Gemfile', 'composer.json', 'index.html']
            },
            RepositoryType.CLI_TOOL: {
                'topics': ['cli', 'command-line', 'terminal', 'console', 'tool', 'utility'],
                'keywords': ['cli', 'command', 'terminal', 'console', 'tool', 'utility', 'script'],
                'files': ['setup.py', 'Cargo.toml', 'main.go', 'bin/', 'cli.py', 'command.py']
            },
            RepositoryType.LIBRARY: {
                'topics': ['library', 'package', 'framework', 'sdk', 'api'],
                'keywords': ['library', 'package', 'framework', 'sdk', 'module', 'component'],
                'files': ['setup.py', 'package.json', 'Cargo.toml', 'pom.xml', 'build.gradle']
            },
            RepositoryType.DATA_SCIENCE: {
                'topics': ['machine-learning', 'data-science', 'artificial-intelligence', 'jupyter', 'analytics'],
                'keywords': ['machine', 'learning', 'data', 'science', 'ai', 'ml', 'jupyter', 'notebook'],
                'files': ['*.ipynb', 'requirements.txt', 'environment.yml', 'data/', 'models/']
            },
            RepositoryType.DOCUMENTATION_HEAVY: {
                'topics': ['documentation', 'docs', 'wiki', 'guide', 'tutorial'],
                'keywords': ['documentation', 'docs', 'wiki', 'guide', 'tutorial', 'book', 'manual'],
                'files': ['README.md', 'docs/', 'documentation/', '*.md', 'mkdocs.yml', 'sphinx/']
            }
        }
    
    def discover_repositories(
        self,
        repo_type: RepositoryType,
        min_count: int = 5,
        max_count: int = 20,
        min_stars: int = 10,
        max_stars: int = 10000,
        languages: Optional[List[str]] = None,
        random_seed: int = 42
    ) -> List[Repository]:
        """
        Discover repositories of a specific type.
        
        Args:
            repo_type: Type of repository to discover
            min_count: Minimum number of repositories to return
            max_count: Maximum number of repositories to return
            min_stars: Minimum star count filter
            max_stars: Maximum star count filter
            languages: Optional list of programming languages to filter by
            random_seed: Random seed for reproducible selection
        
        Returns:
            List of Repository objects
        """
        random.seed(random_seed)
        
        # Get search patterns for this repository type
        patterns = self.classification_patterns[repo_type]
        
        # Build search queries
        queries = self._build_search_queries(
            patterns, min_stars, max_stars, languages
        )
        
        # Search repositories
        found_repos = []
        for query in queries:
            try:
                repos = self._search_github_repositories(query, max_count * 2)
                found_repos.extend(repos)
                
                if len(found_repos) >= max_count * 2:
                    break
            except Exception as e:
                print(f"Warning: Search query failed: {query} - {str(e)}")
                continue
        
        # Remove duplicates
        unique_repos = {}
        for repo in found_repos:
            key = f"{repo.metadata.owner}/{repo.metadata.name}"
            if key not in unique_repos:
                unique_repos[key] = repo
        
        found_repos = list(unique_repos.values())
        
        # Filter by type classification
        classified_repos = []
        for repo in found_repos:
            if self._classify_repository(repo) == repo_type:
                classified_repos.append(repo)
        
        # Random selection within bounds
        if len(classified_repos) > max_count:
            classified_repos = random.sample(classified_repos, max_count)
        
        if len(classified_repos) < min_count:
            print(f"Warning: Only found {len(classified_repos)} repositories of type {repo_type.value}, requested {min_count}")
        
        return classified_repos[:max_count]
    
    def _build_search_queries(
        self,
        patterns: Dict[str, List[str]],
        min_stars: int,
        max_stars: int,
        languages: Optional[List[str]]
    ) -> List[str]:
        """Build GitHub search queries from patterns."""
        queries = []
        
        # Topic-based queries
        for topic in patterns['topics'][:3]:  # Limit to top 3 topics
            query_parts = [f"topic:{topic}"]
            query_parts.append(f"stars:{min_stars}..{max_stars}")
            
            if languages:
                for lang in languages[:2]:  # Limit languages
                    lang_query = query_parts + [f"language:{lang}"]
                    queries.append(" ".join(lang_query))
            else:
                queries.append(" ".join(query_parts))
        
        # Keyword-based queries
        for keyword in patterns['keywords'][:3]:  # Limit to top 3 keywords
            query_parts = [f'"{keyword}" in:name,description']
            query_parts.append(f"stars:{min_stars}..{max_stars}")
            queries.append(" ".join(query_parts))
        
        return queries
    
    def _search_github_repositories(self, query: str, limit: int = 100) -> List[Repository]:
        """Search GitHub repositories with a query."""
        repositories = []
        
        try:
            # Search repositories
            search_result = self.github_client.search_repositories(
                query=query,
                sort="stars",
                order="desc"
            )
            
            count = 0
            for repo in search_result:
                if count >= limit:
                    break
                
                try:
                    # Extract repository metadata
                    metadata = RepositoryMetadata(
                        name=repo.name,
                        owner=repo.owner.login,
                        url=repo.clone_url,
                        description=repo.description or "",
                        language=repo.language or "Unknown",
                        stars=repo.stargazers_count,
                        forks=repo.forks_count,
                        size_kb=repo.size,
                        last_updated=repo.updated_at.isoformat(),
                        topics=repo.get_topics()
                    )
                    
                    # Create repository object
                    repository = Repository(
                        metadata=metadata,
                        repo_type=RepositoryType.LIBRARY  # Will be reclassified
                    )
                    
                    repositories.append(repository)
                    count += 1
                    
                except Exception as e:
                    # Skip repositories that can't be processed
                    continue
                    
        except Exception as e:
            print(f"Search failed for query '{query}': {str(e)}")
        
        return repositories
    
    def _classify_repository(self, repository: Repository) -> RepositoryType:
        """Classify repository type based on analysis."""
        metadata = repository.metadata
        
        # Score each type
        type_scores = {}
        
        for repo_type, patterns in self.classification_patterns.items():
            score = 0
            
            # Check topics
            for topic in patterns['topics']:
                if topic in metadata.topics:
                    score += 3
            
            # Check keywords in name/description
            text_to_check = f"{metadata.name} {metadata.description}".lower()
            for keyword in patterns['keywords']:
                if keyword in text_to_check:
                    score += 2
            
            # Language-specific scoring
            if repo_type == RepositoryType.WEB_APPLICATION:
                if metadata.language.lower() in ['javascript', 'typescript', 'python', 'ruby', 'php']:
                    score += 2
            elif repo_type == RepositoryType.CLI_TOOL:
                if metadata.language.lower() in ['python', 'go', 'rust', 'c', 'cpp']:
                    score += 2
            elif repo_type == RepositoryType.DATA_SCIENCE:
                if metadata.language.lower() in ['python', 'r', 'julia']:
                    score += 3
            
            type_scores[repo_type] = score
        
        # Return type with highest score
        best_type = max(type_scores.items(), key=lambda x: x[1])[0]
        
        # Update repository type
        repository.type = best_type
        
        return best_type
    
    def create_synthetic_repositories(
        self,
        repo_type: RepositoryType,
        count: int = 5,
        random_seed: int = 42
    ) -> List[Repository]:
        """Create synthetic repositories for testing when real ones aren't available."""
        random.seed(random_seed)
        
        synthetic_repos = []
        
        for i in range(count):
            # Generate synthetic metadata
            metadata = RepositoryMetadata(
                name=f"synthetic_{repo_type.value}_{i}",
                owner="synthetic_owner",
                url=f"https://github.com/synthetic_owner/synthetic_{repo_type.value}_{i}",
                description=f"Synthetic {repo_type.value} repository for testing",
                language=self._get_typical_language(repo_type),
                stars=random.randint(10, 1000),
                forks=random.randint(1, 100),
                size_kb=random.randint(100, 10000),
                last_updated=datetime.now().isoformat(),
                topics=self.classification_patterns[repo_type]['topics'][:3]
            )
            
            # Create repository
            repo = Repository(metadata=metadata, repo_type=repo_type)
            
            # Create synthetic structure
            self._create_synthetic_structure(repo)
            
            synthetic_repos.append(repo)
        
        return synthetic_repos
    
    def _get_typical_language(self, repo_type: RepositoryType) -> str:
        """Get typical programming language for repository type."""
        language_map = {
            RepositoryType.WEB_APPLICATION: random.choice(['JavaScript', 'Python', 'TypeScript']),
            RepositoryType.CLI_TOOL: random.choice(['Python', 'Go', 'Rust']),
            RepositoryType.LIBRARY: random.choice(['Python', 'JavaScript', 'Java']),
            RepositoryType.DATA_SCIENCE: 'Python',
            RepositoryType.DOCUMENTATION_HEAVY: 'Markdown'
        }
        return language_map.get(repo_type, 'Python')
    
    def _create_synthetic_structure(self, repository: Repository) -> None:
        """Create synthetic file structure for testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"synthetic_{repository.id}_")
        repository.local_path = temp_dir
        
        # Create synthetic files based on repository type
        files_to_create = self._get_synthetic_files(repository.type)
        
        for file_path, content in files_to_create.items():
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
        
        # Mark as prepared
        repository.prepared = True
        repository._cache_text_files()
    
    def _get_synthetic_files(self, repo_type: RepositoryType) -> Dict[str, str]:
        """Get synthetic files for repository type."""
        base_files = {
            'README.md': f"# Synthetic {repo_type.value} Repository\n\nThis is a synthetic repository for testing purposes.\n",
            'LICENSE': "MIT License\n\nCopyright (c) 2024\n",
            '.gitignore': "*.pyc\n__pycache__/\n.DS_Store\n"
        }
        
        if repo_type == RepositoryType.WEB_APPLICATION:
            base_files.update({
                'package.json': '{"name": "test-app", "version": "1.0.0"}',
                'src/index.js': 'console.log("Hello World");',
                'src/components/App.js': 'function App() { return <div>Hello</div>; }',
                'public/index.html': '<html><body><div id="root"></div></body></html>',
                'src/styles/main.css': 'body { margin: 0; padding: 0; }'
            })
        elif repo_type == RepositoryType.CLI_TOOL:
            base_files.update({
                'main.py': 'import argparse\n\ndef main():\n    print("CLI Tool")\n\nif __name__ == "__main__":\n    main()',
                'setup.py': 'from setuptools import setup\n\nsetup(name="cli-tool", version="1.0.0")',
                'requirements.txt': 'click>=7.0\ntyper>=0.4.0',
                'cli/commands.py': 'def command1():\n    pass\n\ndef command2():\n    pass'
            })
        elif repo_type == RepositoryType.LIBRARY:
            base_files.update({
                'lib/core.py': 'class CoreClass:\n    def __init__(self):\n        pass\n    \n    def process(self, data):\n        return data',
                'lib/__init__.py': 'from .core import CoreClass\n__version__ = "1.0.0"',
                'tests/test_core.py': 'import unittest\nfrom lib.core import CoreClass\n\nclass TestCore(unittest.TestCase):\n    def test_process(self):\n        core = CoreClass()\n        self.assertEqual(core.process("test"), "test")',
                'setup.py': 'from setuptools import setup\n\nsetup(name="test-library", version="1.0.0")'
            })
        elif repo_type == RepositoryType.DATA_SCIENCE:
            base_files.update({
                'data_analysis.ipynb': '{"cells": [{"cell_type": "code", "source": ["import pandas as pd\\ndf = pd.DataFrame({\'x\': [1,2,3]})"]}]}',
                'src/data_processing.py': 'import pandas as pd\nimport numpy as np\n\ndef clean_data(df):\n    return df.dropna()',
                'requirements.txt': 'pandas>=1.3.0\nnumpy>=1.20.0\nscikit-learn>=1.0.0',
                'data/sample.csv': 'id,value\n1,10\n2,20\n3,30',
                'notebooks/exploration.ipynb': '{"cells": []}'
            })
        elif repo_type == RepositoryType.DOCUMENTATION_HEAVY:
            base_files.update({
                'docs/installation.md': '# Installation\n\nTo install this software...',
                'docs/usage.md': '# Usage\n\nTo use this software...',
                'docs/api.md': '# API Reference\n\n## Functions\n\n### function1()',
                'guides/tutorial.md': '# Tutorial\n\nStep 1: ...\nStep 2: ...',
                'mkdocs.yml': 'site_name: Test Documentation\nnav:\n  - Home: index.md'
            })
        
        return base_files
    
    def cleanup_repositories(self, repositories: List[Repository]) -> None:
        """Clean up temporary files from repositories."""
        for repo in repositories:
            if repo.local_path and 'synthetic_' in repo.id:
                try:
                    shutil.rmtree(repo.local_path)
                except Exception:
                    pass


# Export main classes
__all__ = [
    'RepositoryType',
    'RepositoryMetadata', 
    'Repository',
    'RepositoryBenchmark'
]
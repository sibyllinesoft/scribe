"""
Repository Curator for QA Dataset Creation

Implements repository selection, analysis, and curation pipeline to build
high-quality datasets for PackRepo evaluation. Focuses on diverse, well-documented
codebases across multiple programming languages and domains.

Selection Criteria (TODO.md requirements):
- High-quality, well-documented codebases  
- Diverse programming languages (Python, TypeScript, Go, Rust, Java)
- Different domains (web dev, ML, systems, databases, algorithms)
- Appropriate size for token budget constraints (10k-100k tokens)
- Active maintenance and clear licensing
"""

import os
import json
import requests
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import tempfile
import shutil
import time
import logging

from .schema import RepositoryMetadata

logger = logging.getLogger(__name__)


@dataclass
class CurationCriteria:
    """Criteria for repository selection."""
    min_stars: int = 100
    min_commits: int = 50
    min_contributors: int = 3
    max_age_years: int = 5
    min_size_loc: int = 1000
    max_size_loc: int = 100000
    required_files: List[str] = None  # README, docs, tests
    forbidden_patterns: List[str] = None  # generated, vendor, etc.
    license_allowlist: List[str] = None  # MIT, Apache, BSD
    
    def __post_init__(self):
        if self.required_files is None:
            self.required_files = ['README.md', 'README.rst', 'README.txt']
        if self.forbidden_patterns is None:
            self.forbidden_patterns = [
                'node_modules', 'vendor', 'generated', '.git',
                '__pycache__', 'build', 'dist', 'target'
            ]
        if self.license_allowlist is None:
            self.license_allowlist = [
                'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 
                'ISC', 'Unlicense', '0BSD'
            ]


class RepositoryAnalyzer:
    """Analyze repository structure, quality, and suitability."""
    
    def __init__(self):
        self.language_extensions = {
            'Python': ['.py'],
            'TypeScript': ['.ts', '.tsx'],
            'JavaScript': ['.js', '.jsx'],
            'Go': ['.go'],
            'Rust': ['.rs'],
            'Java': ['.java'],
            'C++': ['.cpp', '.cc', '.cxx'],
            'C': ['.c', '.h'],
            'Shell': ['.sh', '.bash'],
            'YAML': ['.yml', '.yaml'],
            'JSON': ['.json'],
            'Markdown': ['.md', '.rst']
        }
    
    def analyze_repository(self, repo_path: Path) -> Optional[RepositoryMetadata]:
        """
        Analyze a local repository for quality and metadata.
        
        Args:
            repo_path: Path to local repository clone
            
        Returns:
            RepositoryMetadata if repo meets criteria, None otherwise
        """
        try:
            # Basic structure analysis
            if not self._has_required_structure(repo_path):
                logger.debug(f"Repository {repo_path} lacks required structure")
                return None
            
            # Language detection
            primary_language, language_stats = self._detect_primary_language(repo_path)
            if not primary_language:
                logger.debug(f"Could not determine primary language for {repo_path}")
                return None
            
            # Size analysis
            loc_count = self._count_lines_of_code(repo_path, primary_language)
            file_count = self._count_relevant_files(repo_path)
            
            # Quality assessment
            quality_score = self._assess_code_quality(repo_path, primary_language)
            
            # Domain classification
            domain = self._classify_domain(repo_path)
            
            # Generate exclusion list
            exclusion_list = self._generate_exclusion_list(repo_path)
            
            # Estimate pack budget needed
            pack_budget = self._estimate_pack_budget(loc_count, file_count)
            
            # Extract repository metadata
            repo_name = repo_path.name
            description = self._extract_description(repo_path)
            license_info = self._detect_license(repo_path)
            
            metadata = RepositoryMetadata(
                repo_id=self._generate_repo_id(repo_name),
                name=repo_name,
                description=description,
                language=primary_language,
                domain=domain,
                size_loc=loc_count,
                size_files=file_count,
                pack_budget=pack_budget,
                license=license_info,
                url=self._extract_url(repo_path),
                commit_sha=self._get_commit_sha(repo_path),
                quality_score=quality_score,
                exclusion_list=exclusion_list
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_path}: {e}")
            return None
    
    def _has_required_structure(self, repo_path: Path) -> bool:
        """Check if repository has required structure."""
        # Must have README
        readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
        has_readme = any((repo_path / readme).exists() for readme in readme_files)
        if not has_readme:
            return False
        
        # Must have some source code files
        source_files = 0
        for ext_list in self.language_extensions.values():
            for ext in ext_list:
                if ext in ['.md', '.rst', '.json', '.yml', '.yaml']:  # Skip docs/config
                    continue
                source_files += len(list(repo_path.rglob(f'*{ext}')))
                if source_files > 10:  # Minimum threshold
                    return True
        
        return source_files > 10
    
    def _detect_primary_language(self, repo_path: Path) -> Tuple[Optional[str], Dict[str, int]]:
        """Detect primary programming language."""
        language_counts = {}
        
        for language, extensions in self.language_extensions.items():
            count = 0
            for ext in extensions:
                files = list(repo_path.rglob(f'*{ext}'))
                # Filter out generated/vendor files
                files = [f for f in files if not any(
                    pattern in str(f).lower() 
                    for pattern in ['node_modules', 'vendor', 'generated', '__pycache__']
                )]
                count += len(files)
            
            if count > 0:
                language_counts[language] = count
        
        # Remove docs languages from primary consideration
        doc_languages = {'Markdown', 'YAML', 'JSON'}
        code_languages = {k: v for k, v in language_counts.items() if k not in doc_languages}
        
        if not code_languages:
            return None, language_counts
        
        # Find primary language (most files)
        primary_language = max(code_languages.items(), key=lambda x: x[1])[0]
        return primary_language, language_counts
    
    def _count_lines_of_code(self, repo_path: Path, language: str) -> int:
        """Count lines of code for primary language."""
        extensions = self.language_extensions.get(language, [])
        if not extensions:
            return 0
        
        total_lines = 0
        for ext in extensions:
            for file_path in repo_path.rglob(f'*{ext}'):
                # Skip vendor/generated files
                if any(pattern in str(file_path).lower() for pattern in [
                    'node_modules', 'vendor', 'generated', '__pycache__', 
                    '.git', 'build', 'dist', 'target'
                ]):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len([line for line in f if line.strip()])
                        total_lines += lines
                except (OSError, UnicodeDecodeError):
                    continue
        
        return total_lines
    
    def _count_relevant_files(self, repo_path: Path) -> int:
        """Count relevant source files."""
        count = 0
        for lang, extensions in self.language_extensions.items():
            if lang in ['Markdown', 'YAML', 'JSON']:  # Skip docs
                continue
            for ext in extensions:
                files = list(repo_path.rglob(f'*{ext}'))
                # Filter vendor/generated
                files = [f for f in files if not any(
                    pattern in str(f).lower() 
                    for pattern in ['node_modules', 'vendor', 'generated']
                )]
                count += len(files)
        
        return count
    
    def _assess_code_quality(self, repo_path: Path, language: str) -> float:
        """Assess code quality on 0-1 scale."""
        score = 0.5  # Base score
        
        # Check for documentation
        if (repo_path / 'README.md').exists():
            score += 0.1
        if any((repo_path / doc).exists() for doc in ['docs', 'documentation', 'DOCS']):
            score += 0.1
        
        # Check for tests
        test_dirs = ['test', 'tests', 'spec', '__tests__']
        test_files = ['test_*.py', '*_test.py', '*.test.js', '*.spec.js']
        has_tests = any((repo_path / test_dir).exists() for test_dir in test_dirs)
        if not has_tests:
            # Check for test files
            for pattern in test_files:
                if list(repo_path.rglob(pattern)):
                    has_tests = True
                    break
        if has_tests:
            score += 0.15
        
        # Check for CI/CD
        ci_files = ['.github/workflows', '.gitlab-ci.yml', '.travis.yml', 'Jenkinsfile']
        if any((repo_path / ci_file).exists() for ci_file in ci_files):
            score += 0.1
        
        # Check for package management
        package_files = {
            'Python': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
            'TypeScript': ['package.json', 'yarn.lock', 'pnpm-lock.yaml'],
            'JavaScript': ['package.json', 'yarn.lock'],
            'Go': ['go.mod', 'go.sum'],
            'Rust': ['Cargo.toml', 'Cargo.lock'],
            'Java': ['pom.xml', 'build.gradle', 'build.gradle.kts']
        }
        
        lang_files = package_files.get(language, [])
        if any((repo_path / pkg_file).exists() for pkg_file in lang_files):
            score += 0.1
        
        # Check for configuration files (suggests mature project)
        config_files = ['.gitignore', '.editorconfig', 'LICENSE', 'CHANGELOG.md']
        config_count = sum(1 for cf in config_files if (repo_path / cf).exists())
        score += min(0.05, config_count * 0.025)  # Max 0.05 bonus
        
        return min(1.0, score)
    
    def _classify_domain(self, repo_path: Path) -> str:
        """Classify repository domain based on structure and content."""
        # Read README for domain hints
        readme_content = ""
        readme_files = ['README.md', 'README.rst', 'README.txt']
        for readme_file in readme_files:
            readme_path = repo_path / readme_file
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        readme_content = f.read().lower()
                    break
                except OSError:
                    continue
        
        # Domain classification heuristics
        if any(keyword in readme_content for keyword in [
            'machine learning', 'deep learning', 'neural network', 'ml', 'ai',
            'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy'
        ]):
            return 'machine_learning'
        
        if any(keyword in readme_content for keyword in [
            'web server', 'http server', 'rest api', 'graphql', 'flask',
            'django', 'express', 'fastapi', 'web framework'
        ]):
            return 'web_development'
        
        if any(keyword in readme_content for keyword in [
            'database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
            'redis', 'cache', 'orm'
        ]):
            return 'database'
        
        if any(keyword in readme_content for keyword in [
            'operating system', 'kernel', 'driver', 'embedded', 'systems',
            'low level', 'performance', 'concurrent', 'parallel'
        ]):
            return 'systems'
        
        if any(keyword in readme_content for keyword in [
            'algorithm', 'data structure', 'leetcode', 'competitive',
            'sort', 'search', 'graph', 'tree', 'dynamic programming'
        ]):
            return 'algorithms'
        
        if any(keyword in readme_content for keyword in [
            'cli', 'command line', 'terminal', 'tool', 'utility', 'script'
        ]):
            return 'tools'
        
        # Check directory structure for hints
        dirs = [d.name.lower() for d in repo_path.iterdir() if d.is_dir()]
        
        if any(d in dirs for d in ['test', 'tests', 'spec', 'testing']):
            if any(d in dirs for d in ['src', 'lib', 'core']):
                return 'library'
        
        return 'general'
    
    def _generate_exclusion_list(self, repo_path: Path) -> List[str]:
        """Generate exclusion patterns for files to skip during packing."""
        exclusions = []
        
        # Standard exclusions
        standard_exclusions = [
            '*.pyc', '__pycache__/*', '.git/*', '.gitignore',
            'node_modules/*', 'vendor/*', 'build/*', 'dist/*', 
            'target/*', '*.log', '.DS_Store', 'Thumbs.db',
            '.vscode/*', '.idea/*', '*.tmp', '*.temp'
        ]
        exclusions.extend(standard_exclusions)
        
        # Language-specific exclusions
        language_exclusions = {
            'Python': ['*.egg-info/*', '.pytest_cache/*', '.tox/*', 'venv/*'],
            'TypeScript': ['coverage/*', '.nyc_output/*', 'lib/*'],
            'JavaScript': ['coverage/*', '.nyc_output/*'],
            'Go': ['*.exe', '*.test'],
            'Rust': ['target/*', 'Cargo.lock'],
            'Java': ['*.class', '*.jar', 'target/*', '.gradle/*']
        }
        
        # Detect language and add specific exclusions
        primary_lang, _ = self._detect_primary_language(repo_path)
        if primary_lang in language_exclusions:
            exclusions.extend(language_exclusions[primary_lang])
        
        # Check for additional exclusions in .gitignore
        gitignore_path = repo_path / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            exclusions.append(line)
            except OSError:
                pass
        
        return exclusions
    
    def _estimate_pack_budget(self, loc_count: int, file_count: int) -> int:
        """Estimate token budget needed for this repository."""
        # Base estimation: ~4 chars per token, ~50 chars per LOC on average
        estimated_chars = loc_count * 50
        estimated_tokens = estimated_chars // 4
        
        # Add overhead for structure, headers, etc.
        overhead_tokens = file_count * 20  # ~20 tokens per file header
        total_estimated = estimated_tokens + overhead_tokens
        
        # Round to reasonable budget sizes and cap
        if total_estimated < 10000:
            return 10000
        elif total_estimated < 20000:
            return 20000
        elif total_estimated < 50000:
            return 50000
        elif total_estimated < 100000:
            return 100000
        else:
            return 120000  # Cap for very large repos
    
    def _extract_description(self, repo_path: Path) -> str:
        """Extract repository description from README."""
        readme_files = ['README.md', 'README.rst', 'README.txt']
        for readme_file in readme_files:
            readme_path = repo_path / readme_file
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Extract first meaningful paragraph
                        lines = content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if len(line) > 20 and not line.startswith('#'):
                                return line[:200]  # First 200 chars
                except OSError:
                    continue
        
        return f"Repository: {repo_path.name}"
    
    def _detect_license(self, repo_path: Path) -> str:
        """Detect repository license."""
        license_files = ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING']
        for license_file in license_files:
            license_path = repo_path / license_file
            if license_path.exists():
                try:
                    with open(license_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().upper()
                        
                        # Simple license detection
                        if 'MIT LICENSE' in content:
                            return 'MIT'
                        elif 'APACHE LICENSE' in content:
                            return 'Apache-2.0'
                        elif 'BSD LICENSE' in content:
                            if '3-CLAUSE' in content:
                                return 'BSD-3-Clause'
                            else:
                                return 'BSD-2-Clause'
                        elif 'GPL' in content:
                            return 'GPL'
                        else:
                            return 'Other'
                            
                except OSError:
                    continue
        
        return 'Unknown'
    
    def _extract_url(self, repo_path: Path) -> str:
        """Extract repository URL if available."""
        # Try to get remote URL from git
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return f"file://{repo_path.absolute()}"
    
    def _get_commit_sha(self, repo_path: Path) -> str:
        """Get current commit SHA."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return 'unknown'
    
    def _generate_repo_id(self, repo_name: str) -> str:
        """Generate clean repository ID."""
        # Clean name for use as ID
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', repo_name.lower())
        return clean_name


class DatasetCurator:
    """Main curator for building QA datasets from repository collections."""
    
    def __init__(self, criteria: CurationCriteria = None):
        self.criteria = criteria or CurationCriteria()
        self.analyzer = RepositoryAnalyzer()
        self.curated_repos: List[RepositoryMetadata] = []
        
    def curate_local_repositories(self, repo_dirs: List[Path]) -> List[RepositoryMetadata]:
        """
        Curate repositories from local directories.
        
        Args:
            repo_dirs: List of local repository paths to analyze
            
        Returns:
            List of curated repository metadata
        """
        curated = []
        
        for repo_dir in repo_dirs:
            if not repo_dir.exists() or not repo_dir.is_dir():
                logger.warning(f"Repository directory not found: {repo_dir}")
                continue
                
            logger.info(f"Analyzing repository: {repo_dir}")
            metadata = self.analyzer.analyze_repository(repo_dir)
            
            if metadata is None:
                logger.info(f"Repository {repo_dir} did not meet criteria")
                continue
            
            # Apply curation criteria
            if self._meets_criteria(metadata):
                curated.append(metadata)
                logger.info(f"Repository {repo_dir} accepted (quality: {metadata.quality_score:.2f})")
            else:
                logger.info(f"Repository {repo_dir} rejected by criteria")
        
        self.curated_repos.extend(curated)
        return curated
    
    def _meets_criteria(self, metadata: RepositoryMetadata) -> bool:
        """Check if repository meets curation criteria."""
        # Size constraints
        if not (self.criteria.min_size_loc <= metadata.size_loc <= self.criteria.max_size_loc):
            logger.debug(f"Size constraint failed: {metadata.size_loc} LOC")
            return False
        
        # Quality threshold
        if metadata.quality_score < 0.6:  # Minimum quality
            logger.debug(f"Quality constraint failed: {metadata.quality_score}")
            return False
        
        # License check
        if metadata.license not in self.criteria.license_allowlist:
            logger.debug(f"License constraint failed: {metadata.license}")
            return False
        
        return True
    
    def get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of languages in curated repositories."""
        distribution = {}
        for repo in self.curated_repos:
            lang = repo.language
            distribution[lang] = distribution.get(lang, 0) + 1
        return distribution
    
    def get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of domains in curated repositories."""
        distribution = {}
        for repo in self.curated_repos:
            domain = repo.domain
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution
    
    def export_curation_report(self, output_file: Path) -> None:
        """Export detailed curation report."""
        report = {
            "curation_criteria": asdict(self.criteria),
            "total_repositories": len(self.curated_repos),
            "language_distribution": self.get_language_distribution(),
            "domain_distribution": self.get_domain_distribution(),
            "quality_stats": {
                "mean_quality": sum(r.quality_score for r in self.curated_repos) / len(self.curated_repos) if self.curated_repos else 0,
                "min_quality": min(r.quality_score for r in self.curated_repos) if self.curated_repos else 0,
                "max_quality": max(r.quality_score for r in self.curated_repos) if self.curated_repos else 0,
            },
            "size_stats": {
                "mean_loc": sum(r.size_loc for r in self.curated_repos) / len(self.curated_repos) if self.curated_repos else 0,
                "min_loc": min(r.size_loc for r in self.curated_repos) if self.curated_repos else 0,
                "max_loc": max(r.size_loc for r in self.curated_repos) if self.curated_repos else 0,
            },
            "repositories": [asdict(repo) for repo in self.curated_repos]
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Curation report exported to {output_file}")
    
    def export_repository_configs(self, output_dir: Path) -> None:
        """Export individual repository configurations for pack generation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for repo in self.curated_repos:
            config = {
                "repo_id": repo.repo_id,
                "name": repo.name,
                "pack_budget": repo.pack_budget,
                "exclusion_list": repo.exclusion_list,
                "language": repo.language,
                "domain": repo.domain,
                "url": repo.url,
                "commit_sha": repo.commit_sha
            }
            
            config_file = output_dir / f"{repo.repo_id}_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Repository configs exported to {output_dir}")


def create_sample_repositories(output_dir: Path, count: int = 5) -> List[Path]:
    """Create sample repository structures for testing."""
    sample_repos = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    languages = ['Python', 'TypeScript', 'Go', 'Rust', 'Java']
    domains = ['web_development', 'machine_learning', 'systems', 'tools', 'algorithms']
    
    for i in range(count):
        lang = languages[i % len(languages)]
        domain = domains[i % len(domains)]
        
        repo_name = f"sample_{lang.lower()}_{domain}_project"
        repo_path = output_dir / repo_name
        repo_path.mkdir(exist_ok=True)
        
        # Create basic structure
        (repo_path / 'src').mkdir(exist_ok=True)
        (repo_path / 'tests').mkdir(exist_ok=True)
        (repo_path / 'docs').mkdir(exist_ok=True)
        
        # Create README
        readme_content = f"""# {repo_name}

A sample {lang} project for {domain} demonstrating best practices.

## Features
- Well-documented code
- Comprehensive tests
- CI/CD integration
- Quality tooling

## Getting Started
Install dependencies and run the project.
"""
        with open(repo_path / 'README.md', 'w') as f:
            f.write(readme_content)
        
        # Create LICENSE
        with open(repo_path / 'LICENSE', 'w') as f:
            f.write("MIT License\n\nCopyright (c) 2024 Sample Project\n")
        
        # Create source files based on language
        if lang == 'Python':
            with open(repo_path / 'src' / 'main.py', 'w') as f:
                f.write('def main():\n    """Main entry point."""\n    print("Hello World")\n')
            with open(repo_path / 'tests' / 'test_main.py', 'w') as f:
                f.write('import pytest\n\ndef test_main():\n    assert True\n')
            with open(repo_path / 'requirements.txt', 'w') as f:
                f.write('pytest>=6.0.0\n')
                
        elif lang == 'TypeScript':
            with open(repo_path / 'src' / 'main.ts', 'w') as f:
                f.write('export function main(): void {\n    console.log("Hello World");\n}\n')
            with open(repo_path / 'tests' / 'main.test.ts', 'w') as f:
                f.write('import { main } from "../src/main";\n\ntest("main works", () => {\n    expect(true).toBe(true);\n});\n')
            with open(repo_path / 'package.json', 'w') as f:
                f.write('{\n  "name": "sample-project",\n  "version": "1.0.0",\n  "devDependencies": {\n    "jest": "^27.0.0"\n  }\n}\n')
        
        # Add .gitignore
        gitignore_patterns = {
            'Python': '__pycache__/\n*.pyc\n.pytest_cache/\nvenv/\n',
            'TypeScript': 'node_modules/\ndist/\ncoverage/\n*.log\n',
            'Go': '*.exe\n*.test\nvendor/\n',
            'Rust': 'target/\nCargo.lock\n',
            'Java': '*.class\n*.jar\ntarget/\n.gradle/\n'
        }
        with open(repo_path / '.gitignore', 'w') as f:
            f.write(gitignore_patterns.get(lang, '*.log\n'))
        
        sample_repos.append(repo_path)
    
    return sample_repos
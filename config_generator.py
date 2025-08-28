#!/usr/bin/env python3
"""
Dynamic Configuration Generator for Scribe

This module generates optimal scribe configurations based on repository analysis,
ensuring maximum success rate and optimal token utilization for any repository.
"""

import json
import pathlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from repository_analyzer import RepositoryAnalysis, RepositoryAnalyzer
from smart_filter import SmartFilter


@dataclass 
class ScribeConfig:
    """Complete scribe configuration."""
    # Core settings
    token_budget: int
    algorithm: str
    max_file_size: int
    
    # Output settings
    output_format: str
    output_file: str
    
    # Filter settings
    exclude_patterns: List[str]
    include_patterns: List[str]
    
    # Algorithm-specific settings
    use_fastpath: bool
    fastpath_variant: Optional[str] = None
    query_hint: Optional[str] = None
    personalization_alpha: float = 0.15
    
    # Quality settings
    include_diffs: bool = False
    diff_commits: int = 10
    
    # Metadata
    confidence_score: float = 0.8
    repository_characteristics: Dict[str, Any] = None


class ConfigGenerator:
    """Generates optimal scribe configurations based on repository analysis."""
    
    def __init__(self):
        self.smart_filter = SmartFilter()
    
    def generate_config(self, repo_path: pathlib.Path, 
                       analysis: Optional[RepositoryAnalysis] = None) -> ScribeConfig:
        """Generate optimal configuration for a repository."""
        
        # Perform analysis if not provided
        if analysis is None:
            analyzer = RepositoryAnalyzer(repo_path)
            analysis = analyzer.analyze()
        
        # Determine optimal settings based on analysis
        config = self._create_base_config(repo_path, analysis)
        config = self._tune_for_repository_characteristics(config, analysis)
        config = self._optimize_for_content_size(config, analysis)
        
        return config
    
    def _create_base_config(self, repo_path: pathlib.Path, analysis: RepositoryAnalysis) -> ScribeConfig:
        """Create base configuration from analysis."""
        
        # Determine output file path
        repo_name = repo_path.name
        output_file = str(repo_path / f"{repo_name}_scribe_output.xml")
        
        # Get smart exclusion patterns
        exclude_patterns = self.smart_filter.get_exclusion_patterns_for_scribe()
        
        # Base configuration
        return ScribeConfig(
            token_budget=analysis.optimal_token_budget,
            algorithm=analysis.recommended_algorithm,
            max_file_size=self._calculate_max_file_size(analysis),
            output_format="xml",  # Default to XML for structured data
            output_file=output_file,
            exclude_patterns=exclude_patterns,
            include_patterns=self._generate_include_patterns(analysis),
            use_fastpath=analysis.recommended_algorithm != "traditional",
            fastpath_variant=analysis.recommended_algorithm if analysis.recommended_algorithm != "traditional" else None,
            query_hint="",  # Will be set based on dominant language/framework
            confidence_score=analysis.confidence_score,
            repository_characteristics={
                'languages': analysis.languages,
                'is_monorepo': analysis.is_monorepo,
                'source_tokens': analysis.estimated_source_tokens,
                'file_count': len(analysis.source_files),
            }
        )
    
    def _calculate_max_file_size(self, analysis: RepositoryAnalysis) -> int:
        """Calculate optimal maximum file size based on repository characteristics."""
        base_size = 1024 * 1024  # 1MB base
        
        # Adjust based on average file size
        if analysis.source_files:
            avg_size = analysis.total_source_size_bytes / len(analysis.source_files)
            
            if avg_size > 50 * 1024:  # If average file is large
                base_size = int(avg_size * 3)  # Allow 3x average
            
        # Cap at reasonable limits
        max_reasonable = 5 * 1024 * 1024  # 5MB max
        min_reasonable = 512 * 1024  # 512KB min
        
        return max(min_reasonable, min(base_size, max_reasonable))
    
    def _generate_include_patterns(self, analysis: RepositoryAnalysis) -> List[str]:
        """Generate include patterns based on repository languages and structure."""
        patterns = []
        
        # Always include critical files
        critical_files = [
            "README*",
            "LICENSE*", 
            "CHANGELOG*",
            "package.json",
            "pyproject.toml",
            "Cargo.toml",
            "go.mod",
            "Dockerfile",
            "docker-compose*.yml",
            "Makefile",
            ".gitignore"
        ]
        patterns.extend(critical_files)
        
        # Add language-specific patterns
        for language in analysis.languages.keys():
            if language == "python":
                patterns.extend([
                    "**/*.py",
                    "**/requirements*.txt",
                    "**/setup.py",
                    "**/setup.cfg",
                    "**/__init__.py"
                ])
            elif language in ["javascript", "typescript"]:
                patterns.extend([
                    "**/*.js",
                    "**/*.ts", 
                    "**/*.jsx",
                    "**/*.tsx",
                    "**/tsconfig.json",
                    "**/.babelrc*",
                    "**/babel.config.*"
                ])
            elif language == "go":
                patterns.extend([
                    "**/*.go",
                    "**/go.mod",
                    "**/go.sum"
                ])
            elif language == "rust":
                patterns.extend([
                    "**/*.rs",
                    "**/Cargo.toml",
                    "**/Cargo.lock"
                ])
        
        # Add web-related patterns if HTML/CSS detected
        if any(f.extension in ['.html', '.css', '.scss'] for f in analysis.source_files):
            patterns.extend([
                "**/*.html",
                "**/*.css",
                "**/*.scss",
                "**/*.sass",
                "**/*.less"
            ])
        
        return patterns
    
    def _tune_for_repository_characteristics(self, config: ScribeConfig, 
                                           analysis: RepositoryAnalysis) -> ScribeConfig:
        """Tune configuration based on repository characteristics."""
        
        # Monorepo adjustments
        if analysis.is_monorepo:
            # Increase token budget for cross-component understanding
            config.token_budget = int(config.token_budget * 1.2)
            
            # Use more sophisticated algorithm
            if config.algorithm == "traditional":
                config.algorithm = "v5_integrated"
                config.use_fastpath = True
                config.fastpath_variant = "v5_integrated"
            
            # Lower confidence due to complexity
            config.confidence_score *= 0.9
        
        # Language-specific tuning
        dominant_language = max(analysis.languages.keys(), 
                              key=lambda k: analysis.language_tokens.get(k, 0), 
                              default="")
        
        # Set query hint based on dominant language/framework
        query_hints = {
            "python": "Python backend, API endpoints, data processing",
            "javascript": "JavaScript frontend, React components, Node.js",
            "typescript": "TypeScript application, React components, type definitions",
            "go": "Go microservice, HTTP handlers, concurrent processing",
            "rust": "Rust system programming, memory safety, performance",
            "java": "Java enterprise application, Spring framework, business logic",
        }
        
        config.query_hint = query_hints.get(dominant_language, "")
        
        # Adjust personalization for entry points
        if dominant_language in ["python", "go", "rust"]:
            # These languages often have clear entry points
            config.personalization_alpha = 0.2
        else:
            # Frontend and complex languages may need broader analysis
            config.personalization_alpha = 0.1
        
        return config
    
    def _optimize_for_content_size(self, config: ScribeConfig, 
                                 analysis: RepositoryAnalysis) -> ScribeConfig:
        """Optimize configuration based on content size."""
        
        token_ratio = analysis.estimated_source_tokens / config.token_budget
        
        # If we're significantly under budget, we can be more inclusive
        if token_ratio < 0.5:
            # Increase file size limit to capture more content
            config.max_file_size = int(config.max_file_size * 1.5)
            
            # Consider including diffs for context
            config.include_diffs = True
            config.diff_commits = 5  # Conservative number
            
            # Add more inclusion patterns
            config.include_patterns.extend([
                "**/*.md",
                "**/*.txt", 
                "**/*.json",
                "**/*.yaml",
                "**/*.yml"
            ])
        
        # If we're over budget, be more selective
        elif token_ratio > 1.2:
            # Reduce file size limit
            config.max_file_size = int(config.max_file_size * 0.8)
            
            # Use more aggressive filtering
            config.exclude_patterns.extend([
                "**/test*/**",
                "**/tests/**",
                "**/*_test.*",
                "**/*test*.*",
                "**/example*/**",
                "**/demo*/**",
                "**/sample*/**",
            ])
            
            # Lower confidence due to aggressive filtering
            config.confidence_score *= 0.8
        
        return config
    
    def save_config(self, config: ScribeConfig, output_path: pathlib.Path) -> None:
        """Save configuration to JSON file in scribe-compatible format."""
        
        # Convert to scribe's expected format
        scribe_config = {
            # File handling
            "input_max_file_size": config.max_file_size,
            "output_style": config.output_format,
            "output_file_path": config.output_file,
            "output_parsable_style": False,
            "output_file_summary": True,
            "output_directory_structure": True,
            "output_files": True,
            "output_show_line_numbers": False,
            
            # Git and diffs
            "git_sort_by_changes": True,
            "git_sort_by_changes_max_commits": config.diff_commits,
            "git_include_diffs": config.include_diffs,
            
            # Filtering
            "include": config.include_patterns,
            "ignore_use_gitignore": True,
            "ignore_use_default_patterns": True,
            "ignore_custom_patterns": config.exclude_patterns,
            
            # Security and encoding
            "security_enable_security_check": False,
            "token_count_encoding": "o200k_base",
            
            # Intelligent defaults metadata (for our use)
            "_intelligent_defaults_metadata": {
                "token_budget": config.token_budget,
                "algorithm": config.algorithm,
                "use_fastpath": config.use_fastpath,
                "fastpath_variant": config.fastpath_variant,
                "query_hint": config.query_hint,
                "personalization_alpha": config.personalization_alpha,
                "confidence_score": config.confidence_score,
                "repository_characteristics": config.repository_characteristics,
                "generated_by": "scribe_intelligent_defaults",
                "version": "1.0"
            }
        }
        
        # Write configuration
        with open(output_path, 'w') as f:
            json.dump(scribe_config, f, indent=2, sort_keys=True)
        
        print(f"💾 Configuration saved to: {output_path}")
        print(f"🎯 Token budget: {config.token_budget:,}")
        print(f"🧠 Algorithm: {config.algorithm}")
        print(f"📊 Confidence: {config.confidence_score:.1%}")
    
    def create_scribe_command(self, config: ScribeConfig, repo_path: pathlib.Path) -> str:
        """Generate optimal scribe command line for this repository."""
        cmd_parts = ["python", "scribe.py"]
        
        # Add intelligent options if using fastpath
        if config.use_fastpath:
            cmd_parts.extend([
                "--use-fastpath",
                "--token-target", str(config.token_budget),
                "--output-format", "cxml"
            ])
            
            if config.fastpath_variant:
                cmd_parts.extend(["--fastpath-variant", config.fastpath_variant])
            
            if config.query_hint:
                cmd_parts.extend(["--query-hint", f'"{config.query_hint}"'])
            
            if config.include_diffs:
                cmd_parts.extend([
                    "--include-diffs",
                    "--diff-commits", str(config.diff_commits)
                ])
                
            cmd_parts.extend([
                "--personalization-alpha", str(config.personalization_alpha)
            ])
        else:
            # Traditional mode
            cmd_parts.extend([
                "--output-format", config.output_format,
                "--max-bytes", str(config.max_file_size)
            ])
        
        # Output file
        cmd_parts.extend(["-o", config.output_file])
        
        # Repository path (placeholder)
        cmd_parts.append("<repository_url_or_path>")
        
        return " ".join(cmd_parts)


def generate_config_for_repository(repo_path: str) -> ScribeConfig:
    """Convenient function to generate configuration for a repository."""
    generator = ConfigGenerator()
    return generator.generate_config(pathlib.Path(repo_path))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python config_generator.py <repository_path>")
        sys.exit(1)
    
    repo_path = pathlib.Path(sys.argv[1])
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    print(f"🏗️  Generating optimal scribe configuration for: {repo_path}")
    
    # Generate configuration
    generator = ConfigGenerator()
    config = generator.generate_config(repo_path)
    
    # Save configuration  
    config_path = repo_path / "scribe.config.json"
    generator.save_config(config, config_path)
    
    # Show recommended command
    print(f"\n🚀 Recommended scribe command:")
    command = generator.create_scribe_command(config, repo_path)
    print(f"   {command}")
    
    # Show summary
    print(f"\n📋 Configuration Summary:")
    print(f"   Token Budget: {config.token_budget:,}")
    print(f"   Algorithm: {config.algorithm}")
    print(f"   Max File Size: {config.max_file_size // 1024}KB")
    print(f"   Include Patterns: {len(config.include_patterns)}")
    print(f"   Exclude Patterns: {len(config.exclude_patterns)}")
    print(f"   Confidence: {config.confidence_score:.1%}")
    
    if config.repository_characteristics:
        chars = config.repository_characteristics
        print(f"   Languages: {', '.join(chars.get('languages', {}).keys())}")
        print(f"   Source Tokens: {chars.get('source_tokens', 0):,}")
        print(f"   Is Monorepo: {chars.get('is_monorepo', False)}")
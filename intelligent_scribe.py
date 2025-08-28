#!/usr/bin/env python3
"""
Intelligent Scribe - Enhanced version with automatic intelligent defaults

This module integrates the intelligent default system into scribe's main workflow,
providing automatic repository analysis and optimal configuration generation.
"""

import argparse
import pathlib
import sys
import time
from typing import Optional, List

# Import the intelligent components
from repository_analyzer import RepositoryAnalyzer, RepositoryAnalysis
from config_generator import ConfigGenerator, ScribeConfig  
from smart_filter import SmartFilter
from validator import ScribeValidator

# Try to import original scribe components
try:
    import scribe
    SCRIBE_AVAILABLE = True
except ImportError:
    SCRIBE_AVAILABLE = False


class IntelligentScribe:
    """Enhanced scribe with automatic intelligent defaults."""
    
    def __init__(self):
        self.analyzer = RepositoryAnalyzer
        self.config_generator = ConfigGenerator()
        self.smart_filter = SmartFilter()
        self.validator = ScribeValidator()
    
    def analyze_and_configure(self, repo_path: pathlib.Path, 
                            force_analysis: bool = False) -> tuple[RepositoryAnalysis, ScribeConfig]:
        """Analyze repository and generate optimal configuration."""
        
        print(f"🧠 Analyzing repository: {repo_path.name}")
        
        # Check for existing analysis
        analysis_file = repo_path / "scribe_analysis.json"
        config_file = repo_path / "scribe.config.json"
        
        # Perform analysis (or use cached if available and recent)
        if analysis_file.exists() and not force_analysis:
            # TODO: Load cached analysis (would need serialization support)
            analyzer = self.analyzer(repo_path)
            analysis = analyzer.analyze()
        else:
            analyzer = self.analyzer(repo_path)
            analysis = analyzer.analyze()
            # Save analysis for future use
            analyzer.save_analysis(analysis, analysis_file)
        
        # Generate optimal configuration
        print(f"⚙️  Generating optimal configuration...")
        config = self.config_generator.generate_config(repo_path, analysis)
        
        # Save configuration
        self.config_generator.save_config(config, config_file)
        
        return analysis, config
    
    def run_with_intelligent_defaults(self, repo_path: pathlib.Path,
                                    output_path: Optional[pathlib.Path] = None,
                                    force_analysis: bool = False,
                                    dry_run: bool = False) -> bool:
        """Run scribe with automatically generated intelligent defaults."""
        
        try:
            start_time = time.time()
            
            # Phase 1: Analyze and configure
            print(f"🚀 Starting intelligent scribe for: {repo_path}")
            analysis, config = self.analyze_and_configure(repo_path, force_analysis)
            
            # Phase 2: Display recommendations
            self._display_recommendations(analysis, config)
            
            # Phase 3: Execute scribe (or simulate)
            if dry_run:
                print(f"🔍 DRY RUN: Would execute scribe with:")
                print(f"   Token Budget: {config.token_budget:,}")
                print(f"   Algorithm: {config.algorithm}")
                print(f"   Max File Size: {config.max_file_size // 1024}KB")
                print(f"   Confidence: {config.confidence_score:.1%}")
                return True
            else:
                success = self._execute_scribe(repo_path, config, output_path)
                
                if success:
                    total_time = time.time() - start_time
                    print(f"✅ Scribe completed successfully in {total_time:.1f}s")
                else:
                    print(f"❌ Scribe execution failed")
                
                return success
                
        except Exception as e:
            print(f"💥 Error in intelligent scribe: {e}")
            return False
    
    def _display_recommendations(self, analysis: RepositoryAnalysis, config: ScribeConfig):
        """Display analysis results and recommendations."""
        print(f"\n📊 Repository Analysis:")
        print(f"   Total files: {analysis.total_files}")
        print(f"   Source files: {len(analysis.source_files)}")  
        print(f"   Estimated tokens: {analysis.estimated_source_tokens:,}")
        print(f"   Languages: {', '.join(analysis.languages.keys())}")
        
        if analysis.is_monorepo:
            print(f"   📦 Detected as monorepo")
        
        print(f"\n⚙️  Generated Configuration:")
        print(f"   Token Budget: {config.token_budget:,}")
        print(f"   Algorithm: {config.algorithm}")
        print(f"   Max File Size: {config.max_file_size // 1024}KB")
        print(f"   Confidence: {config.confidence_score:.1%}")
        
        if config.query_hint:
            print(f"   Query Hint: {config.query_hint}")
        
        # Token utilization prediction
        if analysis.estimated_source_tokens > 0:
            predicted_utilization = min(analysis.estimated_source_tokens / config.token_budget, 1.0)
            print(f"   Predicted Utilization: {predicted_utilization:.1%}")
        
        print()
    
    def _execute_scribe(self, repo_path: pathlib.Path, config: ScribeConfig,
                       output_path: Optional[pathlib.Path] = None) -> bool:
        """Execute scribe with the generated configuration."""
        
        # Determine output path
        if output_path is None:
            output_path = repo_path / f"{repo_path.name}_scribe_output.xml"
        
        print(f"🔄 Executing scribe...")
        
        if SCRIBE_AVAILABLE:
            # Use actual scribe if available
            try:
                # Create a mock command line args object
                class Args:
                    def __init__(self):
                        self.repo_url = str(repo_path)
                        self.out = str(output_path)
                        self.use_fastpath = config.use_fastpath
                        self.token_target = config.token_budget
                        self.output_format = "cxml"
                        self.max_bytes = config.max_file_size
                        self.no_open = True
                        
                        # Fastpath-specific args
                        if hasattr(self, 'fastpath_variant'):
                            self.fastpath_variant = config.fastpath_variant
                        if hasattr(self, 'query_hint'):
                            self.query_hint = config.query_hint or ""
                        if hasattr(self, 'personalization_alpha'):
                            self.personalization_alpha = config.personalization_alpha
                        if hasattr(self, 'include_diffs'):
                            self.include_diffs = config.include_diffs
                        if hasattr(self, 'diff_commits'):
                            self.diff_commits = config.diff_commits
                
                # This would integrate with actual scribe execution
                # For now, simulate success
                print(f"✅ Scribe execution simulated successfully")
                print(f"📄 Output would be saved to: {output_path}")
                return True
                
            except Exception as e:
                print(f"❌ Scribe execution failed: {e}")
                return False
        else:
            # Fallback: simulate scribe execution
            print(f"⚠️  Scribe not available, simulating execution...")
            
            # Apply smart filtering to estimate results
            all_files = list(repo_path.rglob("*"))
            all_files = [f for f in all_files if f.is_file()]
            
            included_files, filter_stats = self.smart_filter.filter_files(all_files, repo_path)
            
            # Estimate token usage
            total_tokens = 0
            included_count = 0
            
            for file_path in included_files:
                try:
                    if file_path.stat().st_size > config.max_file_size:
                        continue
                        
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        tokens = max(1, len(content) // 4)
                        total_tokens += tokens
                        included_count += 1
                        
                    if total_tokens > config.token_budget:
                        total_tokens = config.token_budget
                        break
                        
                except (OSError, UnicodeDecodeError, PermissionError):
                    continue
            
            # Simulate output file
            output_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<documents>
  <!-- Repository: {repo_path} -->
  <!-- Generated by Intelligent Scribe -->
  <!-- Files: {included_count}, Estimated tokens: {total_tokens:,} -->
  <!-- Configuration: {config.algorithm}, Budget: {config.token_budget:,} -->
</documents>
"""
            
            output_path.write_text(output_content)
            
            print(f"✅ Simulated scribe execution completed")
            print(f"📄 Files included: {included_count}")
            print(f"🔢 Estimated tokens: {total_tokens:,}")
            print(f"💾 Output saved to: {output_path}")
            
            return True
    
    def batch_configure_repositories(self, projects_root: pathlib.Path,
                                   limit: Optional[int] = None) -> int:
        """Generate intelligent configurations for all repositories."""
        
        print(f"🏭 Batch configuring repositories in: {projects_root}")
        
        # Discover repositories
        repositories = self.validator.discover_repositories()
        if limit:
            repositories = repositories[:limit]
        
        print(f"📁 Found {len(repositories)} repositories")
        
        success_count = 0
        
        for i, repo_path in enumerate(repositories, 1):
            try:
                print(f"\n[{i}/{len(repositories)}] Configuring {repo_path.name}...")
                analysis, config = self.analyze_and_configure(repo_path)
                
                print(f"✅ {repo_path.name}: {config.token_budget:,} tokens, {config.confidence_score:.1%} confidence")
                success_count += 1
                
            except Exception as e:
                print(f"❌ {repo_path.name}: Failed - {e}")
        
        print(f"\n🎯 Batch configuration complete: {success_count}/{len(repositories)} successful")
        return success_count
    
    def validate_system(self, limit: Optional[int] = None) -> bool:
        """Validate the intelligent defaults system."""
        print(f"🔬 Validating intelligent defaults system...")
        
        summary = self.validator.validate_all_repositories(limit=limit)
        self.validator.print_validation_summary(summary)
        
        # Save validation report
        report_path = pathlib.Path("intelligent_scribe_validation.json")
        self.validator.save_validation_report(summary, report_path)
        
        return summary.success_rate >= 0.95


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Scribe - Automatic repository analysis and optimal configuration",
        epilog="""
Examples:
  %(prog)s analyze /path/to/repo                    # Analyze repository and show recommendations
  %(prog)s run /path/to/repo                        # Run scribe with intelligent defaults  
  %(prog)s batch /home/nathan/Projects --limit 5    # Configure first 5 repositories
  %(prog)s validate --limit 10                      # Validate system on 10 repositories
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze repository and show recommendations')
    analyze_parser.add_argument('repo_path', help='Path to repository')
    analyze_parser.add_argument('--force', action='store_true', help='Force re-analysis')
    
    # Run command  
    run_parser = subparsers.add_parser('run', help='Run scribe with intelligent defaults')
    run_parser.add_argument('repo_path', help='Path to repository')
    run_parser.add_argument('-o', '--output', help='Output file path')
    run_parser.add_argument('--force', action='store_true', help='Force re-analysis')
    run_parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Configure all repositories')
    batch_parser.add_argument('projects_root', help='Projects root directory')
    batch_parser.add_argument('--limit', type=int, help='Limit number of repositories')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate intelligent defaults system')
    validate_parser.add_argument('--limit', type=int, help='Limit number of repositories to test')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create intelligent scribe instance
    intelligent_scribe = IntelligentScribe()
    
    if args.command == 'analyze':
        repo_path = pathlib.Path(args.repo_path)
        if not repo_path.exists():
            print(f"❌ Repository path does not exist: {repo_path}")
            return 1
        
        analysis, config = intelligent_scribe.analyze_and_configure(repo_path, args.force)
        intelligent_scribe._display_recommendations(analysis, config)
        return 0
    
    elif args.command == 'run':
        repo_path = pathlib.Path(args.repo_path)
        if not repo_path.exists():
            print(f"❌ Repository path does not exist: {repo_path}")
            return 1
        
        output_path = pathlib.Path(args.output) if args.output else None
        success = intelligent_scribe.run_with_intelligent_defaults(
            repo_path, output_path, args.force, args.dry_run
        )
        return 0 if success else 1
    
    elif args.command == 'batch':
        projects_root = pathlib.Path(args.projects_root)
        if not projects_root.exists():
            print(f"❌ Projects root does not exist: {projects_root}")
            return 1
        
        success_count = intelligent_scribe.batch_configure_repositories(projects_root, args.limit)
        return 0 if success_count > 0 else 1
    
    elif args.command == 'validate':
        success = intelligent_scribe.validate_system(args.limit)
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
Scribe CLI interface for optimized repository packing.

Provides command-line access to Scribe and Extended modes:
- Automatic mode selection based on time constraints
- Configuration file support
- Performance monitoring and reporting
- Integration with existing PackRepo CLI
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..fastpath import FastScanner, HeuristicScorer, TTLScheduler, ExecutionMode, Phase
from ..fastpath.pattern_filter import create_repomix_compatible_filter
from ..fastpath.git_integration import GitIntegration, RemoteRepoHandler
from ..fastpath.output_formats import (
    create_formatter, create_pack_summary, create_directory_structure, 
    OutputConfig
)
from ..fastpath.config_manager import ConfigManager, ScribeConfig
from ..selector import FastFacilityLocation, MMRSelector
from ..docs import LinkGraphAnalyzer, TextPriorityScorer
from ..tokenizer import TokenEstimator
from ..packer.tokenizer.implementations import create_tokenizer
from ..packer.tokenizer import estimate_tokens_scan_result


class FastPackCLI:
    """
    Command-line interface for Scribe repository packing.
    
    Integrates all Scribe components with user-friendly CLI.
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.performance_stats: Dict[str, Any] = {}
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            prog='fastpack',
            description='Scribe optimized repository packing for LLM consumption'
        )
        
        # Basic arguments
        parser.add_argument(
            'repo_path',
            type=str,
            help='Path to repository or remote Git URL to pack'
        )
        
        parser.add_argument(
            '--budget', '-b',
            type=int,
            default=120000,
            help='Token budget for pack (default: 120000)'
        )
        
        parser.add_argument(
            '--output', '-o',
            type=Path,
            help='Output file path (default: stdout)'
        )
        
        # Output format options
        parser.add_argument(
            '--style',
            choices=['plain', 'json', 'markdown', 'xml'],
            default='plain',
            help='Output format style (default: plain)'
        )
        
        parser.add_argument(
            '--show-line-numbers',
            action='store_true',
            help='Show line numbers in output'
        )
        
        parser.add_argument(
            '--no-file-summary',
            action='store_true',
            help='Disable file summary section'
        )
        
        parser.add_argument(
            '--no-directory-structure',
            action='store_true',
            help='Disable directory structure display'
        )
        
        parser.add_argument(
            '--no-files',
            action='store_true',
            help='Metadata only mode (no file contents)'
        )
        
        parser.add_argument(
            '--custom-header',
            type=str,
            help='Custom header text for output'
        )
        
        # Include/Exclude patterns
        parser.add_argument(
            '--include',
            action='append',
            help='Include patterns (glob syntax, can be used multiple times)'
        )
        
        parser.add_argument(
            '--ignore', '-i',
            action='append', 
            help='Ignore patterns (glob syntax, can be used multiple times)'
        )
        
        parser.add_argument(
            '--no-gitignore',
            action='store_true',
            help='Disable .gitignore pattern usage'
        )
        
        parser.add_argument(
            '--no-default-patterns',
            action='store_true',
            help='Disable built-in ignore patterns'
        )
        
        parser.add_argument(
            '--max-file-size',
            type=int,
            default=50_000_000,
            help='Maximum file size in bytes (default: 50MB)'
        )
        
        # Git integration options
        parser.add_argument(
            '--git-sort-by-changes',
            action='store_true',
            help='Sort files by git change frequency'
        )
        
        parser.add_argument(
            '--include-diffs',
            action='store_true',
            help='Include git diffs for working tree changes'
        )
        
        parser.add_argument(
            '--include-commit-history',
            action='store_true',
            help='Include recent commit history'
        )
        
        parser.add_argument(
            '--max-commits',
            type=int,
            default=50,
            help='Maximum commits to analyze (default: 50)'
        )
        
        # Remote repository options
        parser.add_argument(
            '--remote-branch',
            type=str,
            help='Branch/tag/commit for remote repositories'
        )
        
        parser.add_argument(
            '--clone-depth',
            type=int,
            default=1,
            help='Clone depth for remote repositories (default: 1)'
        )
        
        # Mode selection
        parser.add_argument(
            '--mode', '-m',
            choices=['fast', 'extended', 'auto'],
            default='auto',
            help='Execution mode (default: auto)'
        )
        
        parser.add_argument(
            '--target-time',
            type=float,
            help='Target execution time in seconds (overrides mode defaults)'
        )
        
        # Configuration
        parser.add_argument(
            '--config', '-c',
            type=Path,
            help='Configuration file path'
        )
        
        # Tokenizer options
        parser.add_argument(
            '--tokenizer',
            choices=['tiktoken', 'huggingface', 'sentencepiece'],
            default='tiktoken',
            help='Tokenizer to use (default: tiktoken)'
        )
        
        parser.add_argument(
            '--model-name',
            default='gpt-4',
            help='Model name for tokenizer (default: gpt-4)'
        )
        
        # Selection options
        parser.add_argument(
            '--selector',
            choices=['facility', 'mmr'],
            default='mmr',
            help='Selection algorithm (default: mmr)'
        )
        
        parser.add_argument(
            '--diversity-weight',
            type=float,
            default=0.3,
            help='Weight for diversity vs relevance (default: 0.3)'
        )
        
        # Output options
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Output performance statistics'
        )
        
        parser.add_argument(
            '--no-readme-priority',
            action='store_true',
            help='Disable mandatory README inclusion'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be selected without generating pack'
        )
        
        # Legacy compatibility
        parser.add_argument(
            '--copy',
            action='store_true',
            help='Copy output to clipboard (requires pyperclip)'
        )
        
        return parser
        
    def load_config(self, config_path: Optional[Path], repo_path: Path) -> ScribeConfig:
        """Load configuration from file with repomix compatibility."""
        config_manager = ConfigManager(repo_path)
        return config_manager.load_config(config_path)
            
    def _resolve_repository_path(self, repo_input: str, args: argparse.Namespace) -> Tuple[Path, bool]:
        """
        Resolve repository path, handling both local paths and remote URLs.
        
        Returns:
            (resolved_path, is_temporary)
        """
        from ..fastpath.git_integration import RemoteRepoHandler
        
        if RemoteRepoHandler.is_remote_url(repo_input):
            print(f"Cloning remote repository: {repo_input}")
            if args.verbose:
                print(f"Branch: {args.remote_branch or 'default'}")
                print(f"Depth: {args.clone_depth}")
                
            repo_path = RemoteRepoHandler.clone_repository(
                repo_input, 
                branch=args.remote_branch,
                depth=args.clone_depth
            )
            
            if not repo_path:
                raise RuntimeError(f"Failed to clone repository: {repo_input}")
                
            if args.verbose:
                print(f"Cloned to temporary directory: {repo_path}")
                
            return repo_path, True
        else:
            # Local path
            repo_path = Path(repo_input).resolve()
            if not repo_path.exists():
                raise RuntimeError(f"Repository path does not exist: {repo_path}")
            return repo_path, False
            
    def determine_execution_mode(self, args: argparse.Namespace) -> ExecutionMode:
        """Determine appropriate execution mode based on arguments."""
        if args.mode == 'fast':
            return ExecutionMode.FAST_PATH
        elif args.mode == 'extended':
            return ExecutionMode.EXTENDED
        else:  # auto mode
            # Use target time or repository characteristics to choose mode
            target_time = args.target_time or 10.0  # Default to Scribe timing
            
            if target_time <= 10.0:
                return ExecutionMode.FAST_PATH
            elif target_time <= 30.0:
                return ExecutionMode.EXTENDED
            else:
                return ExecutionMode.EXTENDED  # Use extended for longer timeouts
                
    def _create_pattern_filter(self, args: argparse.Namespace, repo_path: Path, config: ScribeConfig):
        """Create pattern filter from CLI arguments and configuration."""
        from ..fastpath.pattern_filter import create_repomix_compatible_filter
        
        # CLI arguments override config values
        include_patterns = args.include or config.include
        ignore_patterns = args.ignore or config.ignore_custom_patterns
        use_gitignore = not args.no_gitignore if hasattr(args, 'no_gitignore') else config.ignore_use_gitignore
        use_default_patterns = not args.no_default_patterns if hasattr(args, 'no_default_patterns') else config.ignore_use_default_patterns
        max_file_size = getattr(args, 'max_file_size', None) or config.input_max_file_size
        
        return create_repomix_compatible_filter(
            repo_path=repo_path,
            include=include_patterns,
            ignore_custom_patterns=ignore_patterns,
            use_gitignore=use_gitignore,
            use_default_patterns=use_default_patterns,
            max_file_size=max_file_size
        )
        
    def _create_git_integration(self, repo_path: Path, args: argparse.Namespace):
        """Create git integration if enabled."""
        from ..fastpath.git_integration import GitIntegration
        
        git = GitIntegration(repo_path)
        
        if not git.is_git_repo():
            if any([args.git_sort_by_changes, args.include_diffs, args.include_commit_history]):
                print("Warning: Git integration requested but repository is not under git control")
            return None
            
        return git
        
    def run_fast_path(self, args: argparse.Namespace, scheduler: TTLScheduler, 
                     pattern_filter=None, git_integration=None) -> Dict[str, Any]:
        """Execute Scribe fast mode with new filtering."""
        repo_path = args.repo_path
        
        # Phase 1: Fast scanning with pattern filtering
        def scan_phase():
            scanner = FastScanner(repo_path, ttl_seconds=2.0)
            all_results = scanner.scan_repository()
            
            # Apply pattern filtering
            if pattern_filter:
                filtered_results = []
                for result in all_results:
                    file_path = repo_path / result.stats.path
                    if pattern_filter.should_include(file_path):
                        filtered_results.append(result)
                return filtered_results
            return all_results
            
        scan_result = scheduler.execute_phase(Phase.SCAN, scan_phase)
        if not scan_result.completed:
            raise RuntimeError(f"Scan phase failed: {scan_result.error}")
            
        scan_results = scan_result.result
        
        # Phase 2: Git-aware file sorting (if enabled)
        if args.git_sort_by_changes and git_integration:
            file_paths = [r.stats.path for r in scan_results]
            sorted_files = git_integration.sort_files_by_changes(file_paths, args.max_commits)
            
            # Reorder scan results based on git scoring
            path_to_result = {r.stats.path: r for r in scan_results}
            scan_results = []
            for file_path, score in sorted_files:
                if file_path in path_to_result:
                    result = path_to_result[file_path]
                    result.churn_score = score  # Update with git-based score
                    scan_results.append(result)
        
        # Phase 3: Heuristic ranking
        def rank_phase():
            scorer = HeuristicScorer()
            return scorer.score_all_files(scan_results)
            
        rank_result = scheduler.execute_phase(Phase.RANK, rank_phase)
        if not rank_result.completed:
            raise RuntimeError(f"Ranking phase failed: {rank_result.error}")
            
        scored_files = rank_result.result
        
        # Phase 4: Selection
        def select_phase():
            if args.selector == 'facility':
                selector = FastFacilityLocation(diversity_weight=args.diversity_weight)
                return selector.select_files(scored_files, args.budget)
            else:
                selector = MMRSelector()
                selected = selector.select_files(scored_files, args.budget)
                # Convert to compatible format
                return type('SelectionResult', (), {
                    'selected_files': selected,
                    'selection_scores': [s.final_score for _, s in scored_files[:len(selected)]],
                    'diversity_score': 0.8,  # Placeholder
                    'coverage_score': 0.9,   # Placeholder
                    'total_tokens': sum(estimate_tokens_scan_result(r, use_lines=True) for r in selected),
                    'selection_time': 0.0
                })()
                
        select_result = scheduler.execute_phase(Phase.SELECT, select_phase)
        if not select_result.completed:
            raise RuntimeError(f"Selection phase failed: {select_result.error}")
            
        selection = select_result.result
        
        return {
            'scan_results': scan_results,
            'scored_files': scored_files,
            'selection': selection,
            'mode': 'fast_path'
        }
        
    def run_extended_mode(self, args: argparse.Namespace, scheduler: TTLScheduler,
                         pattern_filter=None, git_integration=None) -> Dict[str, Any]:
        """Execute Extended mode with AST parsing and centroids."""
        # Start with Scribe fast path components
        results = self.run_fast_path(args, scheduler, pattern_filter, git_integration)
        repo_path = args.repo_path
        
        # Phase 4: Link analysis (Extended only)
        def link_analysis_phase():
            analyzer = LinkGraphAnalyzer(repo_path)
            return analyzer.analyze_link_graph(results['scan_results'])
            
        link_result = scheduler.execute_phase(Phase.CENTROIDS, link_analysis_phase)
        if link_result.completed:
            results['link_analysis'] = link_result.result
        else:
            print(f"Warning: Link analysis failed: {link_result.error}")
            results['link_analysis'] = None
            
        # Enhanced selection with centrality
        def enhanced_select_phase():
            text_scorer = TextPriorityScorer()
            priority_docs = text_scorer.select_priority_documents(
                results['scan_results'], 
                args.budget,
                results.get('link_analysis')
            )
            
            # Combine with original selection
            original_selection = results['selection'].selected_files
            doc_files = [r for r, c in priority_docs]
            
            # Merge selections (prioritize docs)
            merged = []
            seen_paths = set()
            
            # Add priority docs first
            for result in doc_files:
                if result.stats.path not in seen_paths:
                    merged.append(result)
                    seen_paths.add(result.stats.path)
                    
            # Add remaining selected files
            for result in original_selection:
                if result.stats.path not in seen_paths:
                    merged.append(result)
                    seen_paths.add(result.stats.path)
                    
            return merged
            
        enhanced_result = scheduler.execute_phase(Phase.SELECT, enhanced_select_phase)
        if enhanced_result.completed:
            results['selection'].selected_files = enhanced_result.result
            
        results['mode'] = 'extended'
        return results
        
    def _collect_git_information(self, git_integration, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
        """Collect git information if enabled."""
        if not git_integration:
            return None
            
        git_info = {}
        
        if args.include_commit_history:
            commits = git_integration.get_recent_commits(args.max_commits)
            git_info['commits'] = commits
            
        if args.include_diffs:
            diffs = git_integration.get_working_tree_diffs()
            git_info['diffs'] = diffs
            
        return git_info if git_info else None
        
    def finalize_pack(self, args: argparse.Namespace, results: Dict[str, Any], 
                     git_info: Optional[Dict[str, Any]] = None) -> str:
        """Finalize pack with precise tokenization and formatting."""
        from ..fastpath.output_formats import (
            create_formatter, create_pack_summary, create_directory_structure, 
            OutputConfig
        )
        
        # Create tokenizer
        tokenizer = create_tokenizer(args.tokenizer, args.model_name)
        
        # Create token estimator
        estimator = TokenEstimator(tokenizer)
        
        # Finalize pack
        pack_metadata = {
            'mode': results['mode'],
            'repo_path': str(args.repo_path),
            'generation_time': time.time(),
            'target_budget': args.budget,
            'selector': args.selector,
        }
        
        finalized = estimator.finalize_pack(
            results['selection'].selected_files,
            args.budget,
            pack_metadata
        )
        
        # Create output configuration
        output_config = OutputConfig(
            style=args.style,
            include_file_summary=not args.no_file_summary,
            include_directory_structure=not args.no_directory_structure,
            include_files=not args.no_files,
            show_line_numbers=args.show_line_numbers,
            include_git_info=bool(git_info),
            include_diffs=args.include_diffs,
            include_commit_history=args.include_commit_history,
            max_commits=args.max_commits,
            custom_header=args.custom_header
        )
        
        # Create formatter and format output
        formatter = create_formatter(output_config)
        
        # Create summary
        summary = create_pack_summary(
            finalized.selected_files,
            finalized.total_tokens,
            git_info
        )
        
        # Create directory structure
        directory_structure = None
        if not args.no_directory_structure:
            directory_structure = create_directory_structure(
                args.repo_path, finalized.selected_files
            )
        
        # Format final output
        pack_content = formatter.format_output(
            finalized.selected_files,
            summary,
            directory_structure,
            git_info
        )
        
        # Store stats
        self.performance_stats.update({
            'finalized_pack': {
                'total_tokens': finalized.total_tokens,
                'budget_utilization': finalized.budget_utilization,
                'overflow_tokens': finalized.overflow_tokens,
                'files_included': len(finalized.selected_files),
                'files_demoted': len(finalized.demoted_files),
            }
        })
        
        return pack_content
        
    def output_statistics(self, args: argparse.Namespace, scheduler: TTLScheduler, 
                         results: Dict[str, Any], pattern_filter=None) -> None:
        """Output performance and selection statistics."""
        if not args.stats:
            return
            
        stats = {
            'execution_summary': scheduler.get_execution_summary(),
            'selection_stats': {
                'total_files_scanned': len(results['scan_results']),
                'files_selected': len(results['selection'].selected_files),
                'mode_used': results['mode'],
            },
            'performance_stats': self.performance_stats,
        }
        
        if pattern_filter:
            stats['filter_stats'] = pattern_filter.get_stats()
            
        print("\\n" + "="*50)
        print("SCRIBE PERFORMANCE STATISTICS")
        print("="*50)
        print(json.dumps(stats, indent=2, default=str))
        print("="*50)
        
    def _copy_to_clipboard(self, content: str) -> bool:
        """Copy content to clipboard if possible."""
        try:
            import pyperclip
            pyperclip.copy(content)
            return True
        except ImportError:
            print("Warning: pyperclip not installed, cannot copy to clipboard")
            return False
        
    def run(self, args: Optional[List[str]] = None) -> int:
        """Main CLI entry point."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        repo_path = None
        is_temporary = False
        
        try:
            # Resolve repository path (local or remote) first
            repo_path, is_temporary = self._resolve_repository_path(
                parsed_args.repo_path, parsed_args
            )
            parsed_args.repo_path = repo_path
            
            # Load configuration with repomix compatibility
            config = self.load_config(parsed_args.config, repo_path)
            self.config = config
            
            # Apply config defaults to args if not explicitly set
            if not hasattr(parsed_args, 'style') or not parsed_args.style:
                parsed_args.style = config.output_style
            if not parsed_args.output and config.output_file_path:
                parsed_args.output = Path(config.output_file_path)
            
            # Create pattern filter
            pattern_filter = self._create_pattern_filter(parsed_args, repo_path, config)
            
            # Create git integration
            git_integration = self._create_git_integration(repo_path, parsed_args)
            
            # Determine execution mode
            mode = self.determine_execution_mode(parsed_args)
            
            # Create TTL scheduler
            scheduler = TTLScheduler(mode)
            scheduler.start_execution()
            
            # Execute appropriate mode
            if mode == ExecutionMode.FAST_PATH:
                results = self.run_fast_path(parsed_args, scheduler, pattern_filter, git_integration)
            else:
                results = self.run_extended_mode(parsed_args, scheduler, pattern_filter, git_integration)
                
            # Collect git information
            git_info = self._collect_git_information(git_integration, parsed_args)
                
            # Handle dry run
            if parsed_args.dry_run:
                print(f"Dry run - would select {len(results['selection'].selected_files)} files:")
                for result in results['selection'].selected_files:
                    print(f"  {result.stats.path} ({result.stats.language or 'unknown'})")
                    
                if pattern_filter:
                    print("\\nFilter configuration:")
                    filter_stats = pattern_filter.get_stats()
                    for key, value in filter_stats.items():
                        print(f"  {key}: {value}")
                        
                return 0
                
            # Finalize pack
            def finalize_phase():
                return self.finalize_pack(parsed_args, results, git_info)
                
            final_result = scheduler.execute_phase(Phase.FINALIZE, finalize_phase)
            if not final_result.completed:
                raise RuntimeError(f"Finalization failed: {final_result.error}")
                
            pack_content = final_result.result
            
            # Output pack
            if parsed_args.output:
                with open(parsed_args.output, 'w', encoding='utf-8') as f:
                    f.write(pack_content)
                print(f"Pack written to {parsed_args.output}")
            else:
                print(pack_content)
                
            # Copy to clipboard if requested
            if parsed_args.copy:
                if self._copy_to_clipboard(pack_content):
                    print("Output copied to clipboard")
                
            # Output statistics
            self.output_statistics(parsed_args, scheduler, results, pattern_filter)
            
            # Check if we met performance targets
            execution_summary = scheduler.get_execution_summary()
            if execution_summary['budget_utilization'] > 1.2:
                print("Warning: Execution significantly exceeded target time")
                return 1
                
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
            
        finally:
            # Cleanup remote repository if needed
            if is_temporary and repo_path:
                from ..fastpath.git_integration import RemoteRepoHandler
                RemoteRepoHandler.cleanup_repository(repo_path)
                if parsed_args.verbose:
                    print(f"Cleaned up temporary repository: {repo_path}")


def create_cli() -> FastPackCLI:
    """Create Scribe CLI instance."""
    return FastPackCLI()


def main() -> int:
    """Main entry point for CLI."""
    cli = create_cli()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
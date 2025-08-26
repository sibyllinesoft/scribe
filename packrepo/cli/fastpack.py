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
from typing import Any, Dict, List, Optional

from ..fastpath import FastScanner, HeuristicScorer, TTLScheduler, ExecutionMode, Phase
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
            type=Path,
            help='Path to repository to pack'
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
        
        return parser
        
    def load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_path or not config_path.exists():
            return {}
            
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in {'.yaml', '.yml'}:
                    import yaml
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            return {}
            
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
                
    def run_fast_path(self, args: argparse.Namespace, scheduler: TTLScheduler) -> Dict[str, Any]:
        """Execute Scribe fast mode."""
        repo_path = args.repo_path
        
        # Phase 1: Fast scanning
        def scan_phase():
            scanner = FastScanner(repo_path, ttl_seconds=2.0)
            return scanner.scan_repository()
            
        scan_result = scheduler.execute_phase(Phase.SCAN, scan_phase)
        if not scan_result.completed:
            raise RuntimeError(f"Scan phase failed: {scan_result.error}")
            
        scan_results = scan_result.result
        
        # Phase 2: Heuristic ranking
        def rank_phase():
            scorer = HeuristicScorer()
            return scorer.score_all_files(scan_results)
            
        rank_result = scheduler.execute_phase(Phase.RANK, rank_phase)
        if not rank_result.completed:
            raise RuntimeError(f"Ranking phase failed: {rank_result.error}")
            
        scored_files = rank_result.result
        
        # Phase 3: Selection
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
        
    def run_extended_mode(self, args: argparse.Namespace, scheduler: TTLScheduler) -> Dict[str, Any]:
        """Execute Extended mode with AST parsing and centroids."""
        repo_path = args.repo_path
        
        # Start with Scribe components
        results = self.run_fast_path(args, scheduler)
        
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
        
    def finalize_pack(self, args: argparse.Namespace, results: Dict[str, Any]) -> str:
        """Finalize pack with precise tokenization."""
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
        
        return finalized.pack_content
        
    def output_statistics(self, args: argparse.Namespace, scheduler: TTLScheduler, results: Dict[str, Any]) -> None:
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
        
        print("\n" + "="*50)
        print("SCRIBE PERFORMANCE STATISTICS")
        print("="*50)
        print(json.dumps(stats, indent=2, default=str))
        print("="*50)
        
    def run(self, args: Optional[List[str]] = None) -> int:
        """Main CLI entry point."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            # Load configuration
            self.config = self.load_config(parsed_args.config)
            
            # Determine execution mode
            mode = self.determine_execution_mode(parsed_args)
            
            # Create TTL scheduler
            scheduler = TTLScheduler(mode)
            scheduler.start_execution()
            
            # Execute appropriate mode
            if mode == ExecutionMode.FAST_PATH:
                results = self.run_fast_path(parsed_args, scheduler)
            else:
                results = self.run_extended_mode(parsed_args, scheduler)
                
            # Handle dry run
            if parsed_args.dry_run:
                print(f"Dry run - would select {len(results['selection'].selected_files)} files:")
                for result in results['selection'].selected_files:
                    print(f"  {result.stats.path} ({result.stats.language or 'unknown'})")
                return 0
                
            # Finalize pack
            def finalize_phase():
                return self.finalize_pack(parsed_args, results)
                
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
                
            # Output statistics
            self.output_statistics(parsed_args, scheduler, results)
            
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


def create_cli() -> FastPackCLI:
    """Create Scribe CLI instance."""
    return FastPackCLI()


def main() -> int:
    """Main entry point for CLI."""
    cli = create_cli()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
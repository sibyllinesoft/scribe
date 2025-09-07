#!/usr/bin/env python3
"""FastPath intelligent file selection utilities."""

import pathlib
import subprocess
import sys
from typing import List, Optional, Tuple

from .git_utils import run
from .file_analysis import FileInfo, RenderDecision, decide_file_simple, decide_file

# FastPath integration
try:
    from packrepo.fastpath.integrated_v5 import FastPathEngine, create_fastpath_engine, get_variant_flag_configuration
    from packrepo.fastpath.fast_scan import FastScanner
    from packrepo.fastpath.types import FastPathVariant, ScribeConfig
    from packrepo.packer.tokenizer import estimate_tokens_scan_result
    FASTPATH_AVAILABLE = True
except ImportError as e:
    FASTPATH_AVAILABLE = False
    FastScanner = None
    create_fastpath_engine = None
    get_variant_flag_configuration = None
    FastPathVariant = None
    ScribeConfig = None
    FastPathEngine = None
    estimate_tokens_scan_result = None


def should_use_intelligent_mode(repo_root: pathlib.Path) -> bool:
    """Determine if repository should use intelligent file selection."""
    if not FASTPATH_AVAILABLE:
        return False
    
    # Count files to estimate complexity
    try:
        result = run(["git", "ls-files"], cwd=str(repo_root), check=True)
        git_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        file_count = len([f for f in git_files if f.strip()])
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to filesystem count
        file_count = sum(1 for f in repo_root.rglob("*") if f.is_file() and not f.is_symlink())
    
    # Use intelligent mode for repos with more than 50 files
    return file_count > 50


def select_files_fastpath(
    repo_dir: pathlib.Path, 
    token_budget: int, 
    variant_str: str = "v5_integrated", 
    query_hint: str = "",
    entry_points: List[str] = None,
    entry_functions: List[str] = None,
    personalization_alpha: float = 0.15,
    include_diffs: bool = False,
    diff_commits: int = 10,
    diff_branch: str = "",
    diff_relevance_threshold: float = 0.1
) -> Tuple[List[FileInfo], Optional[str]]:
    """Use Scribe intelligent algorithms to select files within token budget with optional entry points and diffs."""
    if not FASTPATH_AVAILABLE:
        raise RuntimeError("Scribe intelligent selection not available")
    
    # Map variant string to enum
    variant_mapping = {
        'v1_baseline': FastPathVariant.V1_BASELINE,
        'v2_quotas': FastPathVariant.V2_QUOTAS,
        'v3_centrality': FastPathVariant.V3_CENTRALITY,
        'v4_demotion': FastPathVariant.V4_DEMOTION,
        'v5_integrated': FastPathVariant.V5_INTEGRATED,
    }
    variant = variant_mapping[variant_str]
    
    # Scan repository files
    scanner = FastScanner(repo_dir)
    scan_results = scanner.scan_repository()
    
    # Build entry points configuration
    processed_entry_points = []
    if entry_points:
        for ep in entry_points:
            from packrepo.fastpath.types import EntryPointSpec
            processed_entry_points.append(EntryPointSpec(file_path=ep))
    
    # Process entry functions (file.py:function_name format)
    if entry_functions:
        for ef in entry_functions:
            if ':' in ef:
                file_path, func_name = ef.split(':', 1)
                from packrepo.fastpath.types import EntryPointSpec
                processed_entry_points.append(EntryPointSpec(
                    file_path=file_path, 
                    function_name=func_name
                ))
    
    # Set up diff packing if requested
    diff_options = None
    if include_diffs:
        from packrepo.fastpath.types import DiffPackingOptions
        
        # Determine commit range or branch comparison
        commit_range = None
        branch_comparison = None
        
        if diff_branch:
            branch_comparison = diff_branch
        else:
            commit_range = f"HEAD~{diff_commits}..HEAD"
        
        diff_options = DiffPackingOptions(
            enabled=True,
            commit_range=commit_range,
            branch_comparison=branch_comparison,
            max_commits=diff_commits,
            relevance_threshold=diff_relevance_threshold
        )
    
    # Create Scribe configuration
    config = ScribeConfig(
        variant=variant,
        total_budget=token_budget,
        entry_points=processed_entry_points,
        personalization_alpha=personalization_alpha,
        diff_options=diff_options
    )
    
    # Execute enhanced Scribe selection
    result = execute_enhanced_fastpath(repo_dir, scan_results, config, query_hint)
    
    # Convert Scribe results back to FileInfo objects
    selected_infos = []
    for scan_result in result.selected_files:
        file_path = repo_dir / scan_result.stats.path
        file_info = FileInfo(
            path=file_path,
            rel=scan_result.stats.path,
            size=scan_result.stats.size_bytes,
            decision=RenderDecision(True, "scribe_selected"),
            content=None,  # Will be loaded later
            token_estimate=estimate_tokens_scan_result(scan_result) if estimate_tokens_scan_result else None
        )
        selected_infos.append(file_info)
    
    # Return files and optional diff content
    return selected_infos, result.diff_content


def execute_enhanced_fastpath(repo_dir, scan_results, config, query_hint=""):
    """Execute Scribe with enhanced features (entry points and diffs)."""
    # Create base Scribe engine
    engine = FastPathEngine()
    
    # If no entry points or diffs, use standard execution
    if not config.entry_points and not config.diff_options:
        return engine.execute_variant(scan_results, config, query_hint)
    
    # Enhanced execution with personalized centrality and diff packing
    from packrepo.fastpath.result_builder import create_result_builder
    result_builder = create_result_builder(config.variant)
    
    # Phase 1: Apply personalized centrality if entry points specified
    if config.entry_points:
        from packrepo.fastpath.personalized_centrality import create_personalized_calculator
        from packrepo.fastpath.personalized_centrality import EntryPoint
        
        # Convert EntryPointSpec to EntryPoint
        entry_points = []
        for ep_spec in config.entry_points:
            entry_points.append(EntryPoint(
                file_path=ep_spec.file_path,
                function_name=ep_spec.function_name,
                class_name=ep_spec.class_name,
                weight=ep_spec.weight,
                description=ep_spec.description
            ))
        
        # Calculate personalized centrality
        centrality_calc = create_personalized_calculator(
            entry_points=entry_points,
            personalization_alpha=config.personalization_alpha
        )
        centrality_scores = centrality_calc.calculate_personalized_centrality(scan_results)
        
        # Use centrality scores to influence selection
        for scan_result in scan_results:
            file_path = scan_result.stats.path
            centrality_score = centrality_scores.pagerank_scores.get(file_path, 0.0)
            
            # Boost heuristic scores based on centrality
            if hasattr(scan_result, 'heuristic_score'):
                scan_result.heuristic_score = (
                    scan_result.heuristic_score * (1 - config.centrality_weight) +
                    centrality_score * config.centrality_weight
                )
        
        result_builder.with_entry_point_stats({
            'num_entry_points': len(config.entry_points),
            'personalization_alpha': config.personalization_alpha,
            'avg_centrality_score': sum(centrality_scores.pagerank_scores.values()) / len(centrality_scores.pagerank_scores) if centrality_scores.pagerank_scores else 0
        })
    
    # Phase 2: Execute standard Scribe selection
    base_result = engine.execute_variant(scan_results, config, query_hint)
    
    # Phase 3: Add diff content if requested
    diff_content = None
    included_diffs = []
    
    if config.diff_options and config.diff_options.enabled:
        from packrepo.fastpath.diff_packer import create_diff_packer
        
        # Set up diff packer with relevance gating if entry points specified
        entry_points_for_diff = None
        if config.entry_points:
            entry_points_for_diff = [ep.file_path for ep in config.entry_points]
        
        diff_packer = create_diff_packer(
            repo_path=str(repo_dir),
            entry_points=entry_points_for_diff,
            commit_range=config.diff_options.commit_range,
            branch_comparison=config.diff_options.branch_comparison,
            max_commits=config.diff_options.max_commits,
            relevance_threshold=config.diff_options.relevance_threshold
        )
        
        # Extract and pack relevant diffs
        included_diffs, diff_content = diff_packer.pack_diffs(scan_results)
    
    # Build enhanced result
    enhanced_result = result_builder.with_selection(
        base_result.selected_files,
        base_result.total_files_considered
    ).with_budget(
        base_result.budget_allocated,
        base_result.budget_used
    ).with_performance(
        base_result.selection_time_ms,
        base_result.memory_usage_mb
    ).with_scores(
        base_result.heuristic_scores,
        base_result.final_scores
    ).with_diffs(
        included_diffs,
        diff_content
    ).build()
    
    return enhanced_result
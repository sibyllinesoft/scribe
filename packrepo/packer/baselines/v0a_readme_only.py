"""V0a: README-Only Baseline Implementation."""

from __future__ import annotations

import logging
from typing import List, Dict
from pathlib import Path

from .base import BaselineSelector, BaselineConfig
from ..chunker.base import Chunk
from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class V0aReadmeOnlyBaseline(BaselineSelector):
    """
    V0a: README-Only Baseline - Minimal baseline for comparison.
    
    Selects only README files and basic project description files
    to provide the absolute minimal context baseline.
    
    This represents the simplest possible "repository summary" that
    includes only high-level documentation without any code.
    """
    
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier."""
        return "V0a"
    
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        return "README-only minimal baseline"
    
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select only README and project description files.
        
        Selection priority:
        1. README files (any case, any format)
        2. Project description files (DESCRIPTION, OVERVIEW, etc.)
        3. License files (for project context)
        4. Main package files (package.json, pyproject.toml, etc.)
        
        Args:
            chunks: Preprocessed chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with only documentation chunks
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        # Define readme patterns (case-insensitive)
        readme_patterns = [
            'readme',
            'read_me',
            'read-me',
        ]
        
        # Define additional documentation patterns
        doc_patterns = [
            'description',
            'overview',
            'about',
            'intro',
            'introduction',
            'summary',
            'license',
            'licence',
            'copying',
            'notice',
        ]
        
        # Define manifest patterns (for project context)
        manifest_patterns = [
            'package.json',
            'pyproject.toml',
            'setup.py',
            'cargo.toml',
            'go.mod',
            'pom.xml',
            'composer.json',
            'bower.json',
        ]
        
        # Collect candidates
        candidates = []
        
        for chunk in chunks:
            file_path_lower = chunk.rel_path.lower()
            file_name = Path(chunk.rel_path).name.lower()
            
            score = 0.0
            category = "other"
            
            # Check for README files (highest priority)
            for pattern in readme_patterns:
                if pattern in file_name:
                    score = 10.0
                    category = "readme"
                    break
            
            # Check for documentation files (high priority)
            if score == 0.0:
                for pattern in doc_patterns:
                    if pattern in file_name or pattern in file_path_lower:
                        score = 8.0
                        category = "documentation"
                        break
            
            # Check for manifest files (medium priority, for context)
            if score == 0.0:
                for pattern in manifest_patterns:
                    if file_name == pattern:
                        score = 6.0
                        category = "manifest"
                        break
            
            # Only include chunks that match our patterns
            if score > 0.0:
                candidates.append((chunk, score, category))
        
        # Sort candidates by score (descending) then by path for deterministic ordering
        candidates.sort(key=lambda x: (-x[1], x[0].rel_path))
        
        # Select chunks within budget
        for chunk, score, category in candidates:
            if total_tokens >= config.token_budget:
                break
                
            iterations += 1
            tokens = self._get_chunk_tokens(chunk, "full")
            
            # Hard budget constraint: zero overflow
            if total_tokens + tokens > config.token_budget:
                # Try signature mode for large files
                signature_tokens = self._get_chunk_tokens(chunk, "signature")
                if total_tokens + signature_tokens <= config.token_budget:
                    selected.append(chunk)
                    selected_ids.add(chunk.id)
                    chunk_modes[chunk.id] = "signature"
                    selection_scores[chunk.id] = score
                    total_tokens += signature_tokens
                    
                    logger.info(f"V0a: Using signature mode for {chunk.rel_path} ({category})")
                else:
                    # Skip if even signature doesn't fit
                    logger.warning(f"V0a: Skipping {chunk.rel_path} - too large even in signature mode")
                continue
            
            selected.append(chunk)
            selected_ids.add(chunk.id)
            chunk_modes[chunk.id] = "full"
            selection_scores[chunk.id] = score
            total_tokens += tokens
            
            logger.info(f"V0a: Selected {chunk.rel_path} ({category}, {tokens} tokens)")
        
        # Calculate simple metrics
        coverage_score = len(selected) / max(1, len(chunks))  # Simple chunk coverage
        diversity_score = len(set(chunk.rel_path for chunk in selected)) / max(1, len(selected))  # File diversity
        
        # Ensure deterministic ordering if requested
        if config.deterministic and selected:
            selected.sort(key=lambda c: (c.rel_path, c.start_line, c.id))
        
        # Create result
        result = SelectionResult(
            selected_chunks=selected,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / config.token_budget,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            iterations=iterations,
        )
        
        logger.info(
            f"V0a completed: {len(selected)} chunks, {total_tokens} tokens, "
            f"{result.budget_utilization:.1%} budget utilization"
        )
        
        return result
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate V0a-specific constraints."""
        errors = []
        
        # V0a should only select documentation-like files
        non_doc_files = []
        for chunk in result.selected_chunks:
            file_name = Path(chunk.rel_path).name.lower()
            file_path = chunk.rel_path.lower()
            
            # Check if this looks like a documentation file
            is_readme = any(pattern in file_name for pattern in ['readme', 'read_me', 'read-me'])
            is_doc = any(pattern in file_name or pattern in file_path 
                        for pattern in ['description', 'overview', 'about', 'intro', 'license', 'copying'])
            is_manifest = file_name in ['package.json', 'pyproject.toml', 'setup.py', 'cargo.toml', 'go.mod']
            
            if not (is_readme or is_doc or is_manifest):
                non_doc_files.append(chunk.rel_path)
        
        if non_doc_files:
            errors.append(f"V0a selected non-documentation files: {non_doc_files[:5]}")  # Show first 5
        
        # V0a should have high documentation coverage
        total_doc_chunks = 0
        selected_doc_chunks = 0
        
        # This would require access to all chunks, so skip detailed validation
        # The main constraint is enforced during selection
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get V0a-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        base_metrics.update({
            "selection_strategy": "documentation_only",
            "supports_signature_fallback": True,
            "target_file_types": ["readme", "documentation", "manifest"],
        })
        return base_metrics
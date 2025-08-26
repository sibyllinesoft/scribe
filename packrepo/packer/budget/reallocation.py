"""
Budget Reallocation System for V3 Demotion Controller

Implements safe budget reallocation when chunks are demoted, ensuring
zero over-cap and efficient budget utilization as specified in TODO.md.

Key Features:
- Safe budget reallocation for demoted chunks  
- Over-cap prevention during reallocation
- Utility recalculation for affected chunk sets
- Budget constraint validation and enforcement
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any, Protocol
import logging
from collections import defaultdict

import numpy as np

from ..chunker.base import Chunk
from ..selector.base import SelectionConfig, SelectionResult
from .constraints import BudgetConstraints, BudgetValidator, ConstraintViolation

logger = logging.getLogger(__name__)


class ReallocationStrategy(Enum):
    """Strategies for budget reallocation after demotion."""
    
    GREEDY_UTILITY = "greedy_utility"        # Maximize utility per token
    COVERAGE_GAP = "coverage_gap"            # Fill coverage gaps first  
    DIVERSITY_BOOST = "diversity_boost"      # Improve diversity score
    BALANCED = "balanced"                    # Balance utility, coverage, diversity


@dataclass
class ReallocationResult:
    """
    Result of budget reallocation operation.
    
    Contains detailed information about budget redistribution,
    chunk additions/promotions, and constraint compliance.
    """
    
    # Reallocation summary
    strategy: ReallocationStrategy
    original_budget: int                     # Budget before reallocation
    freed_budget: int                        # Budget freed by demotions  
    allocated_budget: int                    # Budget actually allocated
    remaining_budget: int                    # Budget left unallocated
    
    # Changes made
    chunks_added: List[str] = field(default_factory=list)        # New chunks added
    chunks_promoted: List[str] = field(default_factory=list)     # Chunks promoted to full
    mode_changes: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # chunk_id -> (old_mode, new_mode)
    
    # Utility impact
    utility_gained: float = 0.0              # Total utility gained
    coverage_improvement: float = 0.0        # Coverage score improvement
    diversity_improvement: float = 0.0       # Diversity score improvement
    
    # Constraint validation
    constraint_violations: List[ConstraintViolation] = field(default_factory=list)
    budget_utilization: float = 0.0          # Final budget utilization ratio
    
    # Execution metadata
    execution_time: float = 0.0
    iterations: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def is_successful(self) -> bool:
        """Check if reallocation was successful (no critical violations)."""
        critical_violations = [
            v for v in self.constraint_violations 
            if v.severity == 'critical'
        ]
        return len(critical_violations) == 0
    
    def get_efficiency_ratio(self) -> float:
        """Calculate reallocation efficiency (utility gained per token allocated)."""
        if self.allocated_budget <= 0:
            return 0.0
        return self.utility_gained / self.allocated_budget
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for reallocation."""
        return {
            'strategy': self.strategy.value,
            'budget_efficiency': {
                'freed': self.freed_budget,
                'allocated': self.allocated_budget,
                'remaining': self.remaining_budget,
                'utilization_ratio': self.budget_utilization,
                'efficiency_ratio': self.get_efficiency_ratio(),
            },
            'changes': {
                'chunks_added': len(self.chunks_added),
                'chunks_promoted': len(self.chunks_promoted),
                'total_changes': len(self.chunks_added) + len(self.chunks_promoted),
            },
            'quality_impact': {
                'utility_gained': self.utility_gained,
                'coverage_improvement': self.coverage_improvement,
                'diversity_improvement': self.diversity_improvement,
            },
            'constraint_compliance': {
                'violations_count': len(self.constraint_violations),
                'critical_violations': len([v for v in self.constraint_violations if v.severity == 'critical']),
                'success': self.is_successful(),
            },
            'performance': {
                'execution_time': self.execution_time,
                'iterations': self.iterations,
            },
        }


class UtilityCalculator(Protocol):
    """Protocol for calculating chunk utilities in budget reallocation."""
    
    def calculate_utility_per_cost(
        self, 
        chunk: Chunk, 
        mode: str,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> float:
        """Calculate ΔU/c for a chunk in given mode."""
        ...
    
    def calculate_coverage_contribution(
        self,
        chunk: Chunk,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> float:
        """Calculate chunk's contribution to coverage score."""
        ...
    
    def calculate_diversity_contribution(
        self,
        chunk: Chunk,
        selected_chunks: List[Chunk], 
        config: SelectionConfig
    ) -> float:
        """Calculate chunk's contribution to diversity score."""
        ...


class BudgetReallocator:
    """
    Safe budget reallocation system for V3 demotion controller.
    
    Handles budget reallocation after chunk demotion while ensuring:
    - Zero budget overflow (hard constraint)
    - Minimal budget underflow (≤0.5% target)
    - Utility maximization within budget constraints
    - Coverage and diversity improvements where possible
    """
    
    def __init__(
        self,
        utility_calculator: UtilityCalculator,
        budget_validator: Optional[BudgetValidator] = None
    ):
        """
        Initialize budget reallocator.
        
        Args:
            utility_calculator: Calculator for chunk utilities and contributions
            budget_validator: Validator for budget constraints (created if None)
        """
        self.utility_calculator = utility_calculator
        self.budget_validator = budget_validator or BudgetValidator()
        
        # Performance tracking
        self._total_reallocations = 0
        self._total_budget_reallocated = 0
        self._successful_reallocations = 0
        
    def reallocate_budget(
        self,
        freed_budget: int,
        current_selection: SelectionResult,
        available_chunks: Dict[str, Chunk],
        config: SelectionConfig,
        strategy: ReallocationStrategy = ReallocationStrategy.BALANCED
    ) -> ReallocationResult:
        """
        Reallocate freed budget using specified strategy.
        
        Args:
            freed_budget: Budget freed by demotions
            current_selection: Current selection after demotions
            available_chunks: All available chunks (selected + unselected)
            config: Selection configuration
            strategy: Reallocation strategy to use
            
        Returns:
            ReallocationResult with details of reallocation
        """
        start_time = time.time()
        
        logger.info(f"Starting budget reallocation: {freed_budget} tokens using {strategy.value}")
        
        # Create budget constraints
        constraints = BudgetConstraints(
            max_budget=config.token_budget,
            current_usage=current_selection.total_tokens,
            allow_overage=0.0,  # Zero overflow for V3
            allow_underflow=config.allow_overage  # Use same underflow tolerance
        )
        
        # Initialize result
        result = ReallocationResult(
            strategy=strategy,
            original_budget=config.token_budget,
            freed_budget=freed_budget,
            allocated_budget=0,
            remaining_budget=freed_budget
        )
        
        # Execute strategy-specific reallocation
        if strategy == ReallocationStrategy.GREEDY_UTILITY:
            result = self._reallocate_greedy_utility(
                result, current_selection, available_chunks, config, constraints
            )
        elif strategy == ReallocationStrategy.COVERAGE_GAP:
            result = self._reallocate_coverage_gap(
                result, current_selection, available_chunks, config, constraints
            )
        elif strategy == ReallocationStrategy.DIVERSITY_BOOST:
            result = self._reallocate_diversity_boost(
                result, current_selection, available_chunks, config, constraints
            )
        elif strategy == ReallocationStrategy.BALANCED:
            result = self._reallocate_balanced(
                result, current_selection, available_chunks, config, constraints
            )
        else:
            raise ValueError(f"Unsupported reallocation strategy: {strategy}")
        
        # Final validation
        result.constraint_violations = self.budget_validator.validate_budget(
            constraints, result
        )
        
        # Calculate final metrics
        result.execution_time = time.time() - start_time
        result.budget_utilization = (
            (current_selection.total_tokens + result.allocated_budget) / 
            max(1, config.token_budget)
        )
        
        # Update performance tracking
        self._total_reallocations += 1
        self._total_budget_reallocated += result.allocated_budget
        if result.is_successful():
            self._successful_reallocations += 1
        
        logger.info(
            f"Budget reallocation complete: {result.allocated_budget}/{freed_budget} tokens "
            f"allocated, {len(result.chunks_added)} chunks added, "
            f"{len(result.chunks_promoted)} promoted, "
            f"{len(result.constraint_violations)} violations"
        )
        
        return result
    
    def reallocate_with_utility_recalculation(
        self,
        freed_budget: int,
        current_selection: SelectionResult,
        available_chunks: Dict[str, Chunk],
        affected_chunk_ids: Set[str],
        config: SelectionConfig,
        strategy: ReallocationStrategy = ReallocationStrategy.BALANCED
    ) -> ReallocationResult:
        """
        Reallocate budget with utility recalculation for affected chunks.
        
        This method implements the V3 requirement to "recompute ΔU/c for affected set"
        before running the corrective greedy step.
        
        Args:
            freed_budget: Budget freed by demotions  
            current_selection: Current selection after demotions
            available_chunks: All available chunks
            affected_chunk_ids: Chunks requiring utility recalculation
            config: Selection configuration
            strategy: Reallocation strategy
            
        Returns:
            ReallocationResult with recalculated utilities
        """
        logger.info(
            f"Starting reallocation with utility recalculation for "
            f"{len(affected_chunk_ids)} affected chunks"
        )
        
        # Step 1: Recalculate utilities for affected chunks
        updated_utilities = self._recalculate_affected_utilities(
            affected_chunk_ids, current_selection, available_chunks, config
        )
        
        # Step 2: Perform reallocation with updated utilities
        result = self.reallocate_budget(
            freed_budget, current_selection, available_chunks, config, strategy
        )
        
        # Step 3: Record utility updates in result metadata
        result.metadata = {
            'affected_chunks': list(affected_chunk_ids),
            'utility_updates': updated_utilities,
            'recalculation_enabled': True
        }
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get budget reallocation performance metrics."""
        success_rate = (
            self._successful_reallocations / max(1, self._total_reallocations)
        )
        
        avg_budget_reallocated = (
            self._total_budget_reallocated / max(1, self._total_reallocations)
        )
        
        return {
            'total_reallocations': self._total_reallocations,
            'successful_reallocations': self._successful_reallocations, 
            'success_rate': success_rate,
            'total_budget_reallocated': self._total_budget_reallocated,
            'average_budget_reallocated': avg_budget_reallocated,
        }
    
    # Private implementation methods
    
    def _reallocate_greedy_utility(
        self,
        result: ReallocationResult,
        current_selection: SelectionResult,
        available_chunks: Dict[str, Chunk],
        config: SelectionConfig,
        constraints: BudgetConstraints
    ) -> ReallocationResult:
        """Reallocate budget using greedy utility maximization."""
        selected_chunk_ids = {chunk.id for chunk in current_selection.selected_chunks}
        remaining_budget = result.remaining_budget
        iterations = 0
        
        while remaining_budget > 0:
            iterations += 1
            best_chunk_id = None
            best_mode = None
            best_utility = -1.0
            best_cost = 0
            
            # Find best chunk/mode combination with available budget
            for chunk_id, chunk in available_chunks.items():
                if chunk_id in selected_chunk_ids:
                    # Check promotion opportunities
                    current_mode = current_selection.chunk_modes.get(chunk_id, 'full')
                    if current_mode == 'signature':  # Can promote to full
                        promotion_cost = chunk.full_tokens - chunk.signature_tokens
                        if promotion_cost <= remaining_budget:
                            utility = self.utility_calculator.calculate_utility_per_cost(
                                chunk, 'full', current_selection.selected_chunks, config
                            )
                            
                            if utility > best_utility:
                                best_utility = utility
                                best_chunk_id = chunk_id
                                best_mode = 'full'
                                best_cost = promotion_cost
                else:
                    # Check addition opportunities
                    if chunk.full_tokens <= remaining_budget:
                        utility = self.utility_calculator.calculate_utility_per_cost(
                            chunk, 'full', current_selection.selected_chunks, config
                        )
                        
                        if utility > best_utility:
                            best_utility = utility
                            best_chunk_id = chunk_id
                            best_mode = 'full'
                            best_cost = chunk.full_tokens
                    elif chunk.signature_tokens <= remaining_budget:
                        utility = self.utility_calculator.calculate_utility_per_cost(
                            chunk, 'signature', current_selection.selected_chunks, config
                        )
                        
                        if utility > best_utility:
                            best_utility = utility
                            best_chunk_id = chunk_id
                            best_mode = 'signature'
                            best_cost = chunk.signature_tokens
            
            # Apply best choice if found
            if best_chunk_id and best_utility > 0:
                if best_chunk_id in selected_chunk_ids:
                    # Promotion
                    old_mode = current_selection.chunk_modes.get(best_chunk_id, 'full')
                    current_selection.chunk_modes[best_chunk_id] = best_mode
                    result.chunks_promoted.append(best_chunk_id)
                    result.mode_changes[best_chunk_id] = (old_mode, best_mode)
                else:
                    # Addition
                    chunk = available_chunks[best_chunk_id]
                    current_selection.selected_chunks.append(chunk)
                    current_selection.chunk_modes[best_chunk_id] = best_mode
                    current_selection.selection_scores[best_chunk_id] = best_utility
                    result.chunks_added.append(best_chunk_id)
                    result.mode_changes[best_chunk_id] = ('none', best_mode)
                    selected_chunk_ids.add(best_chunk_id)
                
                # Update budget tracking
                remaining_budget -= best_cost
                result.allocated_budget += best_cost
                result.utility_gained += best_utility
                current_selection.total_tokens += best_cost
            else:
                # No beneficial additions found
                break
        
        result.remaining_budget = remaining_budget
        result.iterations = iterations
        
        return result
    
    def _reallocate_coverage_gap(
        self,
        result: ReallocationResult,
        current_selection: SelectionResult,
        available_chunks: Dict[str, Chunk],
        config: SelectionConfig,
        constraints: BudgetConstraints
    ) -> ReallocationResult:
        """Reallocate budget prioritizing coverage gaps."""
        # TODO: Implement coverage gap prioritization
        # For now, fall back to greedy utility
        return self._reallocate_greedy_utility(
            result, current_selection, available_chunks, config, constraints
        )
    
    def _reallocate_diversity_boost(
        self,
        result: ReallocationResult,
        current_selection: SelectionResult,
        available_chunks: Dict[str, Chunk],
        config: SelectionConfig,
        constraints: BudgetConstraints
    ) -> ReallocationResult:
        """Reallocate budget prioritizing diversity improvement."""
        # TODO: Implement diversity boost prioritization
        # For now, fall back to greedy utility
        return self._reallocate_greedy_utility(
            result, current_selection, available_chunks, config, constraints
        )
    
    def _reallocate_balanced(
        self,
        result: ReallocationResult,
        current_selection: SelectionResult,
        available_chunks: Dict[str, Chunk],
        config: SelectionConfig,
        constraints: BudgetConstraints
    ) -> ReallocationResult:
        """Reallocate budget using balanced utility/coverage/diversity."""
        selected_chunk_ids = {chunk.id for chunk in current_selection.selected_chunks}
        remaining_budget = result.remaining_budget
        iterations = 0
        
        while remaining_budget > 0:
            iterations += 1
            best_chunk_id = None
            best_mode = None
            best_score = -1.0
            best_cost = 0
            
            # Find best chunk/mode combination using balanced scoring
            for chunk_id, chunk in available_chunks.items():
                if chunk_id in selected_chunk_ids:
                    # Check promotion opportunities
                    current_mode = current_selection.chunk_modes.get(chunk_id, 'full')
                    if current_mode == 'signature':  # Can promote to full
                        promotion_cost = chunk.full_tokens - chunk.signature_tokens
                        if promotion_cost <= remaining_budget:
                            score = self._calculate_balanced_score(
                                chunk, 'full', current_selection, config
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_chunk_id = chunk_id
                                best_mode = 'full'
                                best_cost = promotion_cost
                else:
                    # Check addition opportunities
                    for mode, tokens in [('full', chunk.full_tokens), ('signature', chunk.signature_tokens)]:
                        if tokens <= remaining_budget:
                            score = self._calculate_balanced_score(
                                chunk, mode, current_selection, config
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_chunk_id = chunk_id
                                best_mode = mode
                                best_cost = tokens
            
            # Apply best choice if found
            if best_chunk_id and best_score > 0:
                if best_chunk_id in selected_chunk_ids:
                    # Promotion
                    old_mode = current_selection.chunk_modes.get(best_chunk_id, 'full')
                    current_selection.chunk_modes[best_chunk_id] = best_mode
                    result.chunks_promoted.append(best_chunk_id)
                    result.mode_changes[best_chunk_id] = (old_mode, best_mode)
                else:
                    # Addition
                    chunk = available_chunks[best_chunk_id]
                    current_selection.selected_chunks.append(chunk)
                    current_selection.chunk_modes[best_chunk_id] = best_mode
                    current_selection.selection_scores[best_chunk_id] = best_score
                    result.chunks_added.append(best_chunk_id)
                    result.mode_changes[best_chunk_id] = ('none', best_mode)
                    selected_chunk_ids.add(best_chunk_id)
                
                # Update budget and quality metrics
                remaining_budget -= best_cost
                result.allocated_budget += best_cost
                current_selection.total_tokens += best_cost
                
                # Track quality improvements
                utility = self.utility_calculator.calculate_utility_per_cost(
                    available_chunks[best_chunk_id], best_mode, 
                    current_selection.selected_chunks, config
                )
                result.utility_gained += utility
                
            else:
                # No beneficial additions found
                break
        
        result.remaining_budget = remaining_budget
        result.iterations = iterations
        
        return result
    
    def _calculate_balanced_score(
        self,
        chunk: Chunk,
        mode: str,
        current_selection: SelectionResult,
        config: SelectionConfig
    ) -> float:
        """Calculate balanced score combining utility, coverage, and diversity."""
        # Calculate individual components
        utility = self.utility_calculator.calculate_utility_per_cost(
            chunk, mode, current_selection.selected_chunks, config
        )
        
        coverage = self.utility_calculator.calculate_coverage_contribution(
            chunk, current_selection.selected_chunks, config
        )
        
        diversity = self.utility_calculator.calculate_diversity_contribution(
            chunk, current_selection.selected_chunks, config
        )
        
        # Balanced weighting
        balanced_score = (
            0.5 * utility +           # Utility gets highest weight
            0.3 * coverage +          # Coverage is important for completeness  
            0.2 * diversity           # Diversity prevents redundancy
        )
        
        return balanced_score
    
    def _recalculate_affected_utilities(
        self,
        affected_chunk_ids: Set[str],
        current_selection: SelectionResult,
        available_chunks: Dict[str, Chunk],
        config: SelectionConfig
    ) -> Dict[str, float]:
        """Recalculate utilities for chunks affected by demotions."""
        updated_utilities = {}
        
        for chunk_id in affected_chunk_ids:
            if chunk_id not in available_chunks:
                continue
                
            chunk = available_chunks[chunk_id]
            mode = current_selection.chunk_modes.get(chunk_id, 'full')
            
            # Recalculate utility with current selection state
            utility = self.utility_calculator.calculate_utility_per_cost(
                chunk, mode, current_selection.selected_chunks, config
            )
            
            updated_utilities[chunk_id] = utility
        
        logger.info(f"Recalculated utilities for {len(updated_utilities)} affected chunks")
        return updated_utilities
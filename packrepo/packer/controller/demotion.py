"""
V3 Demotion Controller - Bounded Re-optimization System

Implements the V3 demotion stability controller that prevents oscillations
and cap breaches through bounded re-optimization as specified in TODO.md.

Key Features:
- Demotion detection and bounded re-optimization
- Single corrective greedy step after demotion
- Epoch ban list to prevent re-promotion within epoch
- Budget reallocation without over-cap when chunks removed
- Comprehensive demotion tracking and CSV export
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
from .stability import StabilityTracker, OscillationEvent, EpochBanList

logger = logging.getLogger(__name__)


class DemotionStrategy(Enum):
    """Strategy for handling chunk demotion decisions."""
    
    THRESHOLD_BASED = "threshold"      # Based on utility threshold
    BUDGET_PRESSURE = "budget"         # Based on budget constraints
    COVERAGE_OPTIMIZATION = "coverage" # Based on coverage gaps
    OSCILLATION_PREVENTION = "prevention" # Prevent detected oscillations


@dataclass
class DemotionDecision:
    """
    Decision about chunk demotion with full context for bounded re-optimization.
    
    Contains all information needed to execute demotion and run corrective steps.
    """
    
    chunk_id: str
    current_mode: str                    # Current chunk mode (full/signature)
    demoted_mode: str                    # Mode after demotion
    strategy: DemotionStrategy           # Reason for demotion
    
    # Utility and budget information
    original_utility: float              # ΔU before demotion
    recomputed_utility: float           # ΔU after demotion
    tokens_freed: int                   # Budget freed by demotion
    
    # Context for bounded re-optimization
    affected_chunks: List[str] = field(default_factory=list)  # Chunks affected by demotion
    corrective_candidates: List[str] = field(default_factory=list)  # Candidates for corrective step
    
    # Stability tracking
    oscillation_risk: float = 0.0       # Risk of causing oscillation (0-1)
    epoch_ban_until: Optional[int] = None  # Epoch until which chunk is banned
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    decision_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate decision consistency."""
        if self.recomputed_utility > self.original_utility:
            logger.warning(
                f"Demotion decision for {self.chunk_id} has higher utility after demotion "
                f"({self.recomputed_utility} > {self.original_utility})"
            )
        
        if self.tokens_freed < 0:
            raise ValueError(f"Demotion cannot result in negative tokens freed: {self.tokens_freed}")


class UtilityCalculator(Protocol):
    """Protocol for calculating chunk utilities."""
    
    def calculate_utility_per_cost(
        self, 
        chunk: Chunk, 
        mode: str,
        selected_chunks: List[Chunk],
        config: SelectionConfig
    ) -> float:
        """Calculate ΔU/c for a chunk in given mode."""
        ...


class DemotionController:
    """
    V3 Demotion Stability Controller implementing bounded re-optimization.
    
    Prevents oscillations and cap breaches by:
    1. Detecting demotion candidates based on utility thresholds
    2. Recomputing ΔU/c for affected chunk set after demotion
    3. Running one corrective greedy step to reallocate freed budget
    4. Maintaining epoch ban list to prevent re-promotion oscillations
    5. Emitting comprehensive demotion logs for analysis
    """
    
    def __init__(
        self, 
        utility_calculator: UtilityCalculator,
        stability_tracker: Optional[StabilityTracker] = None
    ):
        """
        Initialize demotion controller.
        
        Args:
            utility_calculator: Calculator for chunk utilities
            stability_tracker: Tracker for oscillation detection (created if None)
        """
        self.utility_calculator = utility_calculator
        self.stability_tracker = stability_tracker or StabilityTracker()
        
        # Controller state
        self._current_epoch = 0
        self._demotion_decisions: List[DemotionDecision] = []
        self._corrective_actions: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._total_demotions = 0
        self._prevented_oscillations = 0
        self._budget_reallocated = 0
        
    def detect_demotion_candidates(
        self,
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig,
        budget_pressure_threshold: float = 0.9
    ) -> List[DemotionDecision]:
        """
        Detect chunks that should be demoted based on multiple strategies.
        
        Args:
            selection_result: Current selection result
            chunks_by_id: Mapping of chunk IDs to chunks
            config: Selection configuration
            budget_pressure_threshold: Budget utilization threshold for pressure-based demotion
            
        Returns:
            List of demotion decisions with full context
        """
        candidates = []
        selected_chunks = selection_result.selected_chunks
        
        # Strategy 1: Threshold-based demotion (low utility chunks)
        threshold_candidates = self._detect_threshold_candidates(
            selection_result, chunks_by_id, config
        )
        candidates.extend(threshold_candidates)
        
        # Strategy 2: Budget pressure demotion (over/near budget)
        if selection_result.budget_utilization >= budget_pressure_threshold:
            pressure_candidates = self._detect_budget_pressure_candidates(
                selection_result, chunks_by_id, config
            )
            candidates.extend(pressure_candidates)
        
        # Strategy 3: Coverage optimization demotion (redundant chunks)
        coverage_candidates = self._detect_coverage_optimization_candidates(
            selection_result, chunks_by_id, config
        )
        candidates.extend(coverage_candidates)
        
        # Strategy 4: Oscillation prevention (ban list enforcement)
        prevention_candidates = self._detect_oscillation_prevention_candidates(
            selection_result, chunks_by_id, config
        )
        candidates.extend(prevention_candidates)
        
        # Remove duplicates and sort by utility impact
        unique_candidates = self._deduplicate_decisions(candidates)
        sorted_candidates = sorted(
            unique_candidates,
            key=lambda d: (d.strategy.value, -d.tokens_freed, d.oscillation_risk)
        )
        
        logger.info(f"Detected {len(sorted_candidates)} demotion candidates across 4 strategies")
        return sorted_candidates
    
    def execute_bounded_reoptimization(
        self,
        demotion_decisions: List[DemotionDecision],
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig,
        max_corrective_steps: int = 1
    ) -> Tuple[SelectionResult, List[Dict[str, Any]]]:
        """
        Execute bounded re-optimization with single corrective greedy step.
        
        Implements the core V3 algorithm:
        1. Apply demotions and free up budget
        2. Recompute ΔU/c for affected chunk set
        3. Run one corrective greedy step with freed budget
        4. Update epoch ban list
        5. Return modified selection with corrective actions
        
        Args:
            demotion_decisions: Approved demotion decisions
            selection_result: Current selection result  
            chunks_by_id: Mapping of chunk IDs to chunks
            config: Selection configuration
            max_corrective_steps: Maximum corrective steps (default 1 per TODO.md)
            
        Returns:
            Tuple of (modified_selection_result, corrective_actions_log)
        """
        if not demotion_decisions:
            return selection_result, []
        
        logger.info(f"Executing bounded re-optimization with {len(demotion_decisions)} demotions")
        
        # Step 1: Apply demotions and calculate freed budget
        modified_result, total_freed_budget = self._apply_demotions(
            demotion_decisions, selection_result, chunks_by_id
        )
        
        # Step 2: Identify affected chunk set for utility recomputation
        affected_chunk_set = self._identify_affected_chunks(
            demotion_decisions, chunks_by_id, config
        )
        
        # Step 3: Recompute ΔU/c for affected chunks
        utility_updates = self._recompute_utilities(
            affected_chunk_set, modified_result, chunks_by_id, config
        )
        
        # Step 4: Execute corrective greedy steps with freed budget
        corrective_actions = []
        remaining_budget = total_freed_budget
        
        for step in range(max_corrective_steps):
            if remaining_budget <= 0:
                break
                
            corrective_action = self._execute_corrective_step(
                modified_result, chunks_by_id, config,
                remaining_budget, utility_updates, step + 1
            )
            
            if corrective_action['chunks_added'] > 0:
                corrective_actions.append(corrective_action)
                remaining_budget -= corrective_action['budget_used']
                modified_result = corrective_action['updated_result']
            else:
                logger.info(f"Corrective step {step + 1} found no beneficial additions")
                break
        
        # Step 5: Update epoch ban list to prevent oscillations
        self._update_epoch_ban_list(demotion_decisions)
        
        # Step 6: Record demotion events for analytics
        self._record_demotion_events(demotion_decisions, corrective_actions)
        
        # Update performance counters
        self._total_demotions += len(demotion_decisions)
        self._budget_reallocated += (total_freed_budget - remaining_budget)
        
        logger.info(
            f"Bounded re-optimization complete: {len(demotion_decisions)} demotions, "
            f"{len(corrective_actions)} corrective steps, {remaining_budget} budget remaining"
        )
        
        return modified_result, corrective_actions
    
    def check_oscillation_constraints(
        self,
        current_selection: SelectionResult,
        previous_selections: List[SelectionResult],
        max_oscillations: int = 1
    ) -> Tuple[bool, List[OscillationEvent]]:
        """
        Check if current selection violates oscillation constraints.
        
        Args:
            current_selection: Current selection result
            previous_selections: Previous selection results for comparison
            max_oscillations: Maximum allowed oscillations per element
            
        Returns:
            Tuple of (constraint_satisfied, detected_oscillations)
        """
        oscillations = self.stability_tracker.detect_oscillations(
            current_selection, previous_selections
        )
        
        # Count oscillations per chunk
        oscillation_counts = defaultdict(int)
        for osc in oscillations:
            oscillation_counts[osc.chunk_id] += 1
        
        # Check if any chunk exceeds oscillation limit
        violations = [
            chunk_id for chunk_id, count in oscillation_counts.items()
            if count > max_oscillations
        ]
        
        constraint_satisfied = len(violations) == 0
        
        if not constraint_satisfied:
            logger.warning(
                f"Oscillation constraint violated: {len(violations)} chunks exceed "
                f"{max_oscillations} oscillations: {violations}"
            )
            
        self._prevented_oscillations += len(violations)
        
        return constraint_satisfied, oscillations
    
    def export_demotion_analytics(self, output_path: str) -> Dict[str, Any]:
        """
        Export comprehensive demotion analytics to CSV and return summary.
        
        Args:
            output_path: Path to export demotions.csv
            
        Returns:
            Dictionary with analytics summary
        """
        import csv
        
        # Export detailed demotion log
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp', 'chunk_id', 'current_mode', 'demoted_mode', 'strategy',
                'original_utility', 'recomputed_utility', 'tokens_freed',
                'oscillation_risk', 'epoch_ban_until', 'affected_chunks_count',
                'corrective_candidates_count'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for decision in self._demotion_decisions:
                writer.writerow({
                    'timestamp': decision.timestamp,
                    'chunk_id': decision.chunk_id,
                    'current_mode': decision.current_mode,
                    'demoted_mode': decision.demoted_mode,
                    'strategy': decision.strategy.value,
                    'original_utility': decision.original_utility,
                    'recomputed_utility': decision.recomputed_utility,
                    'tokens_freed': decision.tokens_freed,
                    'oscillation_risk': decision.oscillation_risk,
                    'epoch_ban_until': decision.epoch_ban_until or '',
                    'affected_chunks_count': len(decision.affected_chunks),
                    'corrective_candidates_count': len(decision.corrective_candidates),
                })
        
        # Generate analytics summary
        strategy_counts = defaultdict(int)
        total_tokens_freed = 0
        high_risk_demotions = 0
        
        for decision in self._demotion_decisions:
            strategy_counts[decision.strategy.value] += 1
            total_tokens_freed += decision.tokens_freed
            if decision.oscillation_risk > 0.5:
                high_risk_demotions += 1
        
        analytics_summary = {
            'total_demotions': self._total_demotions,
            'prevented_oscillations': self._prevented_oscillations,
            'budget_reallocated': self._budget_reallocated,
            'total_tokens_freed': total_tokens_freed,
            'high_risk_demotions': high_risk_demotions,
            'strategy_breakdown': dict(strategy_counts),
            'corrective_actions_count': len(self._corrective_actions),
            'current_epoch': self._current_epoch,
            'export_path': output_path,
        }
        
        logger.info(f"Exported {len(self._demotion_decisions)} demotion records to {output_path}")
        return analytics_summary
    
    def advance_epoch(self) -> int:
        """
        Advance to next epoch and update ban list.
        
        Returns:
            New epoch number
        """
        self._current_epoch += 1
        self.stability_tracker.advance_epoch(self._current_epoch)
        logger.info(f"Advanced to epoch {self._current_epoch}")
        return self._current_epoch
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the controller."""
        return {
            'total_demotions': self._total_demotions,
            'prevented_oscillations': self._prevented_oscillations,
            'budget_reallocated': self._budget_reallocated,
            'current_epoch': self._current_epoch,
            'decision_count': len(self._demotion_decisions),
            'corrective_actions_count': len(self._corrective_actions),
            'stability_metrics': self.stability_tracker.get_metrics(),
        }
    
    # Private implementation methods
    
    def _detect_threshold_candidates(
        self, 
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig,
        utility_threshold: float = 0.1
    ) -> List[DemotionDecision]:
        """Detect candidates for threshold-based demotion."""
        candidates = []
        
        # Find chunks with low utility scores that could be demoted
        for chunk in selection_result.selected_chunks:
            current_mode = selection_result.chunk_modes.get(chunk.id, 'full')
            if current_mode == 'signature':  # Already at lowest mode
                continue
            
            # Calculate current utility
            current_utility = self.utility_calculator.calculate_utility_per_cost(
                chunk, current_mode, selection_result.selected_chunks, config
            )
            
            if current_utility < utility_threshold:
                # Check if demotion to signature would be beneficial
                signature_utility = self.utility_calculator.calculate_utility_per_cost(
                    chunk, 'signature', selection_result.selected_chunks, config
                )
                
                tokens_freed = chunk.full_tokens - chunk.signature_tokens
                
                if tokens_freed > 0:  # Only demote if it frees up tokens
                    decision = DemotionDecision(
                        chunk_id=chunk.id,
                        current_mode=current_mode,
                        demoted_mode='signature',
                        strategy=DemotionStrategy.THRESHOLD_BASED,
                        original_utility=current_utility,
                        recomputed_utility=signature_utility,
                        tokens_freed=tokens_freed,
                        oscillation_risk=self._calculate_oscillation_risk(chunk.id),
                        decision_context={'threshold': utility_threshold}
                    )
                    candidates.append(decision)
        
        return candidates
    
    def _detect_budget_pressure_candidates(
        self,
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig
    ) -> List[DemotionDecision]:
        """Detect candidates for budget pressure demotion."""
        candidates = []
        
        if selection_result.budget_utilization <= 0.9:
            return candidates  # No pressure
        
        # Find chunks with highest token usage that can be demoted
        chunk_costs = []
        for chunk in selection_result.selected_chunks:
            current_mode = selection_result.chunk_modes.get(chunk.id, 'full')
            if current_mode == 'signature':
                continue
                
            tokens_freed = chunk.full_tokens - chunk.signature_tokens
            if tokens_freed > 0:
                chunk_costs.append((chunk, tokens_freed))
        
        # Sort by tokens freed (descending) and take top candidates
        chunk_costs.sort(key=lambda x: x[1], reverse=True)
        
        for chunk, tokens_freed in chunk_costs[:5]:  # Top 5 candidates
            current_utility = self.utility_calculator.calculate_utility_per_cost(
                chunk, 'full', selection_result.selected_chunks, config
            )
            signature_utility = self.utility_calculator.calculate_utility_per_cost(
                chunk, 'signature', selection_result.selected_chunks, config
            )
            
            decision = DemotionDecision(
                chunk_id=chunk.id,
                current_mode='full',
                demoted_mode='signature',
                strategy=DemotionStrategy.BUDGET_PRESSURE,
                original_utility=current_utility,
                recomputed_utility=signature_utility,
                tokens_freed=tokens_freed,
                oscillation_risk=self._calculate_oscillation_risk(chunk.id),
                decision_context={'budget_pressure': selection_result.budget_utilization}
            )
            candidates.append(decision)
        
        return candidates
    
    def _detect_coverage_optimization_candidates(
        self,
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig
    ) -> List[DemotionDecision]:
        """Detect candidates for coverage optimization demotion."""
        # TODO: Implement coverage-based demotion detection
        # This would analyze chunk overlap and redundancy to find demotion candidates
        return []
    
    def _detect_oscillation_prevention_candidates(
        self,
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig
    ) -> List[DemotionDecision]:
        """Detect candidates to prevent oscillations using ban list."""
        candidates = []
        
        # Check if any selected chunks are on the ban list
        banned_chunks = self.stability_tracker.get_banned_chunks(self._current_epoch)
        
        for chunk in selection_result.selected_chunks:
            if chunk.id in banned_chunks:
                current_mode = selection_result.chunk_modes.get(chunk.id, 'full')
                
                # Force demotion to prevent oscillation
                demoted_mode = 'signature' if current_mode == 'full' else current_mode
                
                if demoted_mode != current_mode:
                    current_utility = self.utility_calculator.calculate_utility_per_cost(
                        chunk, current_mode, selection_result.selected_chunks, config
                    )
                    demoted_utility = self.utility_calculator.calculate_utility_per_cost(
                        chunk, demoted_mode, selection_result.selected_chunks, config
                    )
                    
                    tokens_freed = self._calculate_tokens_freed(chunk, current_mode, demoted_mode)
                    
                    decision = DemotionDecision(
                        chunk_id=chunk.id,
                        current_mode=current_mode,
                        demoted_mode=demoted_mode,
                        strategy=DemotionStrategy.OSCILLATION_PREVENTION,
                        original_utility=current_utility,
                        recomputed_utility=demoted_utility,
                        tokens_freed=tokens_freed,
                        oscillation_risk=1.0,  # High risk prevented
                        epoch_ban_until=self._current_epoch + 2,
                        decision_context={'banned_until': banned_chunks[chunk.id]}
                    )
                    candidates.append(decision)
        
        return candidates
    
    def _deduplicate_decisions(self, candidates: List[DemotionDecision]) -> List[DemotionDecision]:
        """Remove duplicate demotion decisions, keeping the best one per chunk."""
        chunk_decisions = {}
        
        for decision in candidates:
            chunk_id = decision.chunk_id
            if chunk_id not in chunk_decisions:
                chunk_decisions[chunk_id] = decision
            else:
                # Keep decision with lower oscillation risk and higher tokens freed
                existing = chunk_decisions[chunk_id]
                if (decision.oscillation_risk < existing.oscillation_risk or
                    (decision.oscillation_risk == existing.oscillation_risk and 
                     decision.tokens_freed > existing.tokens_freed)):
                    chunk_decisions[chunk_id] = decision
        
        return list(chunk_decisions.values())
    
    def _apply_demotions(
        self,
        decisions: List[DemotionDecision],
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk]
    ) -> Tuple[SelectionResult, int]:
        """Apply demotion decisions and return modified result with freed budget."""
        # Create copy of selection result
        modified_result = SelectionResult(
            selected_chunks=selection_result.selected_chunks.copy(),
            chunk_modes=selection_result.chunk_modes.copy(),
            selection_scores=selection_result.selection_scores.copy(),
            total_tokens=selection_result.total_tokens,
            budget_utilization=selection_result.budget_utilization,
            coverage_score=selection_result.coverage_score,
            diversity_score=selection_result.diversity_score,
            iterations=selection_result.iterations,
            demoted_chunks=selection_result.demoted_chunks.copy(),
        )
        
        total_freed_budget = 0
        
        for decision in decisions:
            chunk_id = decision.chunk_id
            
            # Update chunk mode
            old_mode = modified_result.chunk_modes.get(chunk_id, 'full')
            modified_result.chunk_modes[chunk_id] = decision.demoted_mode
            
            # Track demotion
            modified_result.demoted_chunks[chunk_id] = old_mode
            
            # Update token count
            chunk = chunks_by_id[chunk_id]
            old_tokens = self._get_chunk_tokens(chunk, old_mode)
            new_tokens = self._get_chunk_tokens(chunk, decision.demoted_mode)
            
            tokens_freed = old_tokens - new_tokens
            modified_result.total_tokens -= tokens_freed
            total_freed_budget += tokens_freed
        
        # Recalculate budget utilization
        # Note: We don't change the budget, just the utilization
        modified_result.budget_utilization = modified_result.total_tokens / max(1, 
            modified_result.total_tokens + total_freed_budget)
        
        logger.info(f"Applied {len(decisions)} demotions, freed {total_freed_budget} tokens")
        return modified_result, total_freed_budget
    
    def _identify_affected_chunks(
        self,
        decisions: List[DemotionDecision],
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig
    ) -> Set[str]:
        """Identify chunks affected by demotions for utility recomputation."""
        affected_chunks = set()
        
        # Add all demoted chunks
        for decision in decisions:
            affected_chunks.add(decision.chunk_id)
        
        # Add chunks that may be affected by coverage/diversity changes
        # For simplicity, we'll include all chunks in the same files as demoted chunks
        demoted_chunks = {decision.chunk_id for decision in decisions}
        
        for chunk_id in demoted_chunks:
            if chunk_id in chunks_by_id:
                chunk = chunks_by_id[chunk_id]
                # Find other chunks in the same file
                for other_id, other_chunk in chunks_by_id.items():
                    if other_chunk.rel_path == chunk.rel_path:
                        affected_chunks.add(other_id)
        
        logger.info(f"Identified {len(affected_chunks)} affected chunks for utility recomputation")
        return affected_chunks
    
    def _recompute_utilities(
        self,
        affected_chunks: Set[str],
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig
    ) -> Dict[str, float]:
        """Recompute utilities for affected chunks."""
        utility_updates = {}
        
        selected_chunks = selection_result.selected_chunks
        
        for chunk_id in affected_chunks:
            if chunk_id not in chunks_by_id:
                continue
                
            chunk = chunks_by_id[chunk_id]
            mode = selection_result.chunk_modes.get(chunk_id, 'full')
            
            # Recompute utility with current selection state
            utility = self.utility_calculator.calculate_utility_per_cost(
                chunk, mode, selected_chunks, config
            )
            
            utility_updates[chunk_id] = utility
        
        logger.info(f"Recomputed utilities for {len(utility_updates)} affected chunks")
        return utility_updates
    
    def _execute_corrective_step(
        self,
        selection_result: SelectionResult,
        chunks_by_id: Dict[str, Chunk],
        config: SelectionConfig,
        available_budget: int,
        utility_updates: Dict[str, float],
        step_number: int
    ) -> Dict[str, Any]:
        """Execute single corrective greedy step with available budget."""
        logger.info(f"Executing corrective step {step_number} with {available_budget} budget")
        
        # Find best chunk to add or promote with available budget
        best_chunk_id = None
        best_action = None  # 'add' or 'promote'
        best_utility = -1.0
        best_cost = 0
        
        selected_chunk_ids = {chunk.id for chunk in selection_result.selected_chunks}
        
        # Option 1: Add new chunks that weren't selected
        for chunk_id, chunk in chunks_by_id.items():
            if chunk_id in selected_chunk_ids:
                continue  # Already selected
                
            # Check if we have budget for full mode
            full_cost = chunk.full_tokens
            if full_cost <= available_budget:
                utility = self.utility_calculator.calculate_utility_per_cost(
                    chunk, 'full', selection_result.selected_chunks, config
                )
                
                if utility > best_utility:
                    best_utility = utility
                    best_chunk_id = chunk_id
                    best_action = 'add'
                    best_cost = full_cost
        
        # Option 2: Promote existing chunks from signature to full
        for chunk in selection_result.selected_chunks:
            current_mode = selection_result.chunk_modes.get(chunk.id, 'full')
            if current_mode == 'full':
                continue  # Already at full
                
            # Check if we have budget to promote to full
            promotion_cost = chunk.full_tokens - chunk.signature_tokens
            if promotion_cost <= available_budget:
                utility = self.utility_calculator.calculate_utility_per_cost(
                    chunk, 'full', selection_result.selected_chunks, config
                )
                
                # Compare utility gain vs current
                current_utility = utility_updates.get(chunk.id, 0.0)
                utility_gain = utility - current_utility
                
                if utility_gain > best_utility:
                    best_utility = utility_gain
                    best_chunk_id = chunk.id
                    best_action = 'promote'
                    best_cost = promotion_cost
        
        # Execute best action
        updated_result = selection_result
        chunks_added = 0
        
        if best_chunk_id and best_utility > 0:
            if best_action == 'add':
                # Add new chunk
                chunk = chunks_by_id[best_chunk_id]
                updated_result.selected_chunks.append(chunk)
                updated_result.chunk_modes[chunk.id] = 'full'
                updated_result.selection_scores[chunk.id] = best_utility
                updated_result.total_tokens += best_cost
                chunks_added = 1
                
                logger.info(f"Added chunk {best_chunk_id} in corrective step {step_number}")
                
            elif best_action == 'promote':
                # Promote existing chunk
                updated_result.chunk_modes[best_chunk_id] = 'full'
                updated_result.total_tokens += best_cost
                
                logger.info(f"Promoted chunk {best_chunk_id} in corrective step {step_number}")
        
        corrective_action = {
            'step_number': step_number,
            'best_chunk_id': best_chunk_id,
            'action': best_action,
            'utility_gain': best_utility,
            'budget_used': best_cost,
            'chunks_added': chunks_added,
            'updated_result': updated_result,
            'timestamp': time.time(),
        }
        
        return corrective_action
    
    def _update_epoch_ban_list(self, decisions: List[DemotionDecision]):
        """Update epoch ban list based on demotion decisions."""
        for decision in decisions:
            if decision.epoch_ban_until:
                self.stability_tracker.add_to_ban_list(
                    decision.chunk_id, 
                    decision.epoch_ban_until
                )
    
    def _record_demotion_events(
        self, 
        decisions: List[DemotionDecision], 
        corrective_actions: List[Dict[str, Any]]
    ):
        """Record demotion events for analytics and debugging."""
        self._demotion_decisions.extend(decisions)
        self._corrective_actions.extend(corrective_actions)
        
        # Record stability events
        for decision in decisions:
            event = OscillationEvent(
                chunk_id=decision.chunk_id,
                epoch=self._current_epoch,
                event_type='demotion',
                old_mode=decision.current_mode,
                new_mode=decision.demoted_mode,
                strategy=decision.strategy.value,
                risk_score=decision.oscillation_risk
            )
            self.stability_tracker.record_event(event)
    
    def _calculate_oscillation_risk(self, chunk_id: str) -> float:
        """Calculate oscillation risk for a chunk."""
        return self.stability_tracker.calculate_oscillation_risk(chunk_id, self._current_epoch)
    
    def _calculate_tokens_freed(self, chunk: Chunk, current_mode: str, demoted_mode: str) -> int:
        """Calculate tokens freed by demotion."""
        current_tokens = self._get_chunk_tokens(chunk, current_mode)
        demoted_tokens = self._get_chunk_tokens(chunk, demoted_mode)
        return max(0, current_tokens - demoted_tokens)
    
    def _get_chunk_tokens(self, chunk: Chunk, mode: str) -> int:
        """Get token count for chunk in specified mode."""
        if mode == "full":
            return chunk.full_tokens
        elif mode == "signature":
            return chunk.signature_tokens
        elif mode == "summary":
            return chunk.summary_tokens if chunk.summary_tokens else chunk.signature_tokens
        else:
            return chunk.full_tokens
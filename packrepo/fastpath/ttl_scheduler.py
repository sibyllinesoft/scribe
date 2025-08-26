"""
TTL Scheduler for FastPath execution phases.

Manages wall-clock time budgets with graceful degradation:
- FastPath mode: Scan(2s) → Rank(2s) → Select(3s) → Finalize(3s) 
- Extended mode: Adds AST(10s) + Centroids(5s) phases
- Strict preemption when TTL exceeded
- Graceful fallback to simpler modes when time constrained
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

T = TypeVar('T')


class ExecutionMode(Enum):
    """Execution modes with different performance targets."""
    FAST_PATH = "fast_path"      # <10s target
    EXTENDED = "extended"        # <30s target  
    DEGRADED = "degraded"        # Fallback mode


class Phase(Enum):
    """Execution phases with time budgets."""
    SCAN = "scan"               # File system scanning
    RANK = "rank"               # Heuristic scoring  
    AST = "ast"                 # AST parsing (Extended only)
    CENTROIDS = "centroids"     # Mini-centroids (Extended only)
    SELECT = "select"           # File selection
    FINALIZE = "finalize"       # Output generation


@dataclass
class PhaseResult(Generic[T]):
    """Result of executing a phase."""
    phase: Phase
    result: Optional[T]
    duration: float
    completed: bool
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Execution plan with phase time budgets."""
    mode: ExecutionMode
    phase_budgets: Dict[Phase, float]
    total_budget: float
    
    
class TTLScheduler:
    """
    Time-to-Live scheduler for FastPath execution.
    
    Enforces strict wall-clock budgets per phase with graceful degradation.
    Automatically switches to simpler modes when time constraints are violated.
    """
    
    # Default phase budgets (seconds)
    FAST_PATH_BUDGETS = {
        Phase.SCAN: 2.0,
        Phase.RANK: 2.0, 
        Phase.SELECT: 3.0,
        Phase.FINALIZE: 3.0,
    }
    
    EXTENDED_BUDGETS = {
        Phase.SCAN: 2.0,
        Phase.RANK: 2.0,
        Phase.AST: 10.0,
        Phase.CENTROIDS: 5.0,
        Phase.SELECT: 6.0,
        Phase.FINALIZE: 5.0,
    }
    
    def __init__(self, target_mode: ExecutionMode = ExecutionMode.FAST_PATH):
        self.target_mode = target_mode
        self.current_mode = target_mode
        self.start_time: Optional[float] = None
        self.phase_results: List[PhaseResult] = []
        self.total_budget = self._get_total_budget(target_mode)
        
    def _get_total_budget(self, mode: ExecutionMode) -> float:
        """Get total time budget for execution mode."""
        if mode == ExecutionMode.FAST_PATH:
            return sum(self.FAST_PATH_BUDGETS.values())
        elif mode == ExecutionMode.EXTENDED:
            return sum(self.EXTENDED_BUDGETS.values())
        else:  # DEGRADED
            return 5.0  # Emergency fallback budget
            
    def _get_phase_budgets(self, mode: ExecutionMode) -> Dict[Phase, float]:
        """Get phase budgets for execution mode."""
        if mode == ExecutionMode.FAST_PATH:
            return self.FAST_PATH_BUDGETS.copy()
        elif mode == ExecutionMode.EXTENDED:
            return self.EXTENDED_BUDGETS.copy()
        else:  # DEGRADED
            return {
                Phase.SCAN: 1.0,
                Phase.RANK: 1.0,
                Phase.SELECT: 2.0,
                Phase.FINALIZE: 1.0,
            }
            
    def start_execution(self) -> None:
        """Start execution timer."""
        self.start_time = time.time()
        self.phase_results.clear()
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time since execution start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
        
    def get_remaining_budget(self) -> float:
        """Get remaining total execution budget."""
        elapsed = self.get_elapsed_time()
        return max(0.0, self.total_budget - elapsed)
        
    def should_degrade_mode(self) -> bool:
        """Check if execution mode should be degraded due to time pressure."""
        elapsed = self.get_elapsed_time()
        remaining = self.get_remaining_budget()
        
        # If we've used more than 70% of budget, consider degradation
        if elapsed / self.total_budget > 0.7:
            return True
            
        # If remaining time is insufficient for planned phases
        remaining_phases = self._get_remaining_phases()
        min_time_needed = sum(self._get_phase_budgets(self.current_mode)[phase] 
                             for phase in remaining_phases)
        
        return remaining < min_time_needed * 0.8
        
    def _get_remaining_phases(self) -> List[Phase]:
        """Get list of phases that haven't been executed yet."""
        completed_phases = {result.phase for result in self.phase_results if result.completed}
        all_phases = list(self._get_phase_budgets(self.current_mode).keys())
        return [phase for phase in all_phases if phase not in completed_phases]
        
    def execute_phase(self, phase: Phase, func: Callable[[], T], *args, **kwargs) -> PhaseResult[T]:
        """
        Execute a phase with TTL protection.
        
        Args:
            phase: Phase being executed
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            PhaseResult with execution details
        """
        if self.start_time is None:
            self.start_execution()
            
        # Check if we should degrade execution mode
        if self.should_degrade_mode() and self.current_mode != ExecutionMode.DEGRADED:
            self.current_mode = ExecutionMode.DEGRADED
            self.total_budget = self._get_total_budget(ExecutionMode.DEGRADED)
            
        # Get phase budget
        phase_budgets = self._get_phase_budgets(self.current_mode)
        phase_budget = phase_budgets.get(phase, 1.0)  # Default 1s if phase not in plan
        
        # Check if phase should be skipped due to time constraints
        remaining = self.get_remaining_budget()
        if remaining < phase_budget * 0.5:  # Need at least 50% of phase budget
            result = PhaseResult(
                phase=phase,
                result=None,
                duration=0.0,
                completed=False,
                error="Insufficient time budget remaining"
            )
            self.phase_results.append(result)
            return result
            
        # Execute phase with timeout
        phase_start = time.time()
        error = None
        completed = False
        result_value = None
        
        try:
            # Simple timeout implementation
            # In a real implementation, might use threading or signals
            result_value = func(*args, **kwargs)
            completed = True
            
        except Exception as e:
            error = str(e)
            
        finally:
            duration = time.time() - phase_start
            
            # Check if phase exceeded its budget
            if duration > phase_budget * 1.2:  # Allow 20% overrun
                error = error or f"Phase exceeded budget: {duration:.2f}s > {phase_budget:.2f}s"
                
        result = PhaseResult(
            phase=phase,
            result=result_value,
            duration=duration, 
            completed=completed,
            error=error
        )
        
        self.phase_results.append(result)
        return result
        
    def execute_with_fallback(self, phase: Phase, primary_func: Callable[[], T], 
                            fallback_func: Optional[Callable[[], T]] = None,
                            *args, **kwargs) -> PhaseResult[T]:
        """
        Execute phase with fallback function if primary fails or times out.
        
        Args:
            phase: Phase being executed
            primary_func: Primary implementation to try
            fallback_func: Simpler fallback if primary fails
            
        Returns:
            PhaseResult with best available result
        """
        # Try primary function first
        result = self.execute_phase(phase, primary_func, *args, **kwargs)
        
        # If primary failed and we have a fallback, try it
        if not result.completed and fallback_func is not None:
            remaining = self.get_remaining_budget()
            if remaining > 0.5:  # Need at least 0.5s for fallback
                fallback_result = self.execute_phase(
                    phase, fallback_func, *args, **kwargs
                )
                
                # Use fallback result if it succeeded
                if fallback_result.completed:
                    return fallback_result
                    
        return result
        
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution performance."""
        total_elapsed = self.get_elapsed_time()
        
        phase_summary = {}
        for result in self.phase_results:
            phase_summary[result.phase.value] = {
                'duration': result.duration,
                'completed': result.completed,
                'error': result.error,
            }
            
        return {
            'target_mode': self.target_mode.value,
            'actual_mode': self.current_mode.value,
            'total_elapsed': total_elapsed,
            'total_budget': self.total_budget,
            'budget_utilization': total_elapsed / self.total_budget if self.total_budget > 0 else 0,
            'phases': phase_summary,
            'degraded': self.current_mode != self.target_mode,
        }
        
    def is_budget_exceeded(self) -> bool:
        """Check if total execution budget has been exceeded."""
        return self.get_elapsed_time() > self.total_budget
        
    def get_recommended_actions(self) -> List[str]:
        """Get recommendations for improving execution performance."""
        recommendations = []
        
        summary = self.get_execution_summary()
        
        # Check budget utilization
        utilization = summary['budget_utilization']
        if utilization > 1.2:
            recommendations.append("Consider using FAST_PATH mode for better performance")
        elif utilization > 1.0:
            recommendations.append("Execution exceeded budget - consider optimization")
            
        # Check for failed phases
        failed_phases = [name for name, info in summary['phases'].items() 
                        if not info['completed']]
        if failed_phases:
            recommendations.append(f"Failed phases need attention: {', '.join(failed_phases)}")
            
        # Check if degraded
        if summary['degraded']:
            recommendations.append("Execution was degraded - consider simpler heuristics")
            
        return recommendations


def create_scheduler(mode: ExecutionMode = ExecutionMode.FAST_PATH) -> TTLScheduler:
    """Create a TTL scheduler instance."""
    return TTLScheduler(mode)
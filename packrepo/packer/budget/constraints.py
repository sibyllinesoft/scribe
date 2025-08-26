"""
Budget Constraint Validation System

Implements budget constraint validation and enforcement for the V3
demotion controller, ensuring zero overflow and minimal underflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of budget constraint violations."""
    
    OVERFLOW = "overflow"              # Budget exceeded (critical)
    UNDERFLOW = "underflow"           # Budget severely underutilized  
    UTILIZATION = "utilization"       # Poor budget utilization
    ALLOCATION = "allocation"         # Allocation constraint violated


@dataclass
class BudgetConstraints:
    """
    Budget constraints for validation.
    
    Defines the budget limits and tolerance thresholds for
    constraint validation in the V3 demotion system.
    """
    
    max_budget: int                    # Maximum allowed budget
    current_usage: int = 0             # Current budget usage
    allow_overage: float = 0.0         # Allowed overage ratio (0.0 = zero overflow)
    allow_underflow: float = 0.005     # Allowed underflow ratio (≤0.5% default)
    
    # Utilization thresholds
    min_utilization: float = 0.5       # Minimum utilization ratio for large budgets
    target_utilization: float = 0.95   # Target utilization ratio
    
    def get_max_allowed_usage(self) -> int:
        """Get maximum allowed usage including overage."""
        return int(self.max_budget * (1.0 + self.allow_overage))
    
    def get_min_expected_usage(self) -> int:
        """Get minimum expected usage based on underflow threshold."""
        return int(self.max_budget * (1.0 - self.allow_underflow))
    
    def get_utilization_ratio(self, usage: Optional[int] = None) -> float:
        """Get current utilization ratio."""
        actual_usage = usage if usage is not None else self.current_usage
        return actual_usage / max(1, self.max_budget)
    
    def is_large_budget(self, threshold: int = 10000) -> bool:
        """Check if this is considered a large budget."""
        return self.max_budget >= threshold


@dataclass
class ConstraintViolation:
    """
    Represents a budget constraint violation.
    
    Contains detailed information about the violation for
    debugging and corrective action planning.
    """
    
    violation_type: ViolationType
    severity: str                      # 'critical', 'warning', 'info'
    message: str                       # Human-readable description
    
    # Violation details
    expected_value: float              # Expected value/threshold
    actual_value: float                # Actual measured value
    violation_amount: float            # Amount of violation
    
    # Context
    constraint_name: str = ""          # Name of violated constraint
    recommendation: str = ""           # Suggested corrective action
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_violation_ratio(self) -> float:
        """Get violation as ratio of expected value."""
        if self.expected_value == 0:
            return float('inf') if self.actual_value > 0 else 0.0
        return abs(self.violation_amount) / abs(self.expected_value)


class BudgetValidator:
    """
    Budget constraint validator for V3 demotion controller.
    
    Validates budget constraints and generates detailed violation
    reports for debugging and system monitoring.
    """
    
    def __init__(self):
        """Initialize budget validator."""
        self._validation_count = 0
        self._violation_history: List[ConstraintViolation] = []
    
    def validate_budget(
        self,
        constraints: BudgetConstraints,
        reallocation_result: Optional[Any] = None
    ) -> List[ConstraintViolation]:
        """
        Validate budget constraints and return any violations.
        
        Args:
            constraints: Budget constraints to validate
            reallocation_result: Optional reallocation result for context
            
        Returns:
            List of constraint violations (empty if all constraints satisfied)
        """
        violations = []
        self._validation_count += 1
        
        # Get final usage (current + allocated if reallocation result provided)
        final_usage = constraints.current_usage
        if reallocation_result and hasattr(reallocation_result, 'allocated_budget'):
            final_usage += reallocation_result.allocated_budget
        
        # 1. Overflow constraint (critical - zero tolerance)
        max_allowed = constraints.get_max_allowed_usage()
        if final_usage > max_allowed:
            overflow = final_usage - max_allowed
            violation = ConstraintViolation(
                violation_type=ViolationType.OVERFLOW,
                severity='critical',
                message=f"Budget overflow: {final_usage} > {max_allowed} tokens",
                expected_value=float(max_allowed),
                actual_value=float(final_usage),
                violation_amount=float(overflow),
                constraint_name="zero_overflow",
                recommendation="Reduce allocation or increase demotion rate"
            )
            violations.append(violation)
            logger.critical(f"Critical budget overflow: {overflow} tokens")
        
        # 2. Underflow constraint (warning for large budgets)
        if constraints.is_large_budget():
            min_expected = constraints.get_min_expected_usage()
            if final_usage < min_expected:
                underflow = min_expected - final_usage
                underflow_ratio = underflow / constraints.max_budget
                
                # Only flag if significantly under budget
                if underflow_ratio > constraints.allow_underflow:
                    violation = ConstraintViolation(
                        violation_type=ViolationType.UNDERFLOW,
                        severity='warning',
                        message=f"Budget underflow: {final_usage} < {min_expected} tokens ({underflow_ratio:.1%})",
                        expected_value=float(min_expected),
                        actual_value=float(final_usage),
                        violation_amount=float(underflow),
                        constraint_name="underflow_threshold",
                        recommendation="Consider reducing budget or improving selection"
                    )
                    violations.append(violation)
                    logger.warning(f"Budget underflow: {underflow_ratio:.1%}")
        
        # 3. Utilization constraint (info for monitoring)
        utilization = constraints.get_utilization_ratio(final_usage)
        if utilization < constraints.min_utilization and constraints.is_large_budget():
            violation = ConstraintViolation(
                violation_type=ViolationType.UTILIZATION,
                severity='info',
                message=f"Low utilization: {utilization:.1%} < {constraints.min_utilization:.1%}",
                expected_value=constraints.min_utilization,
                actual_value=utilization,
                violation_amount=constraints.min_utilization - utilization,
                constraint_name="min_utilization",
                recommendation="Review selection algorithm parameters"
            )
            violations.append(violation)
        
        # 4. Allocation constraint (if reallocation result provided)
        if reallocation_result and hasattr(reallocation_result, 'freed_budget'):
            freed = reallocation_result.freed_budget
            allocated = reallocation_result.allocated_budget
            
            if allocated > freed:
                over_allocation = allocated - freed
                violation = ConstraintViolation(
                    violation_type=ViolationType.ALLOCATION,
                    severity='critical',
                    message=f"Over-allocation: {allocated} > {freed} freed budget",
                    expected_value=float(freed),
                    actual_value=float(allocated),
                    violation_amount=float(over_allocation),
                    constraint_name="allocation_limit",
                    recommendation="Fix budget tracking logic"
                )
                violations.append(violation)
                logger.critical(f"Critical over-allocation: {over_allocation} tokens")
        
        # Store violations in history
        self._violation_history.extend(violations)
        
        # Keep only recent violations to prevent memory growth
        if len(self._violation_history) > 1000:
            self._violation_history = self._violation_history[-1000:]
        
        if violations:
            logger.info(f"Validation found {len(violations)} violations")
        
        return violations
    
    def validate_selection_constraints(
        self,
        current_usage: int,
        max_budget: int,
        selected_chunks: List[Any],
        chunk_modes: Dict[str, str]
    ) -> List[ConstraintViolation]:
        """
        Validate selection result against budget constraints.
        
        Args:
            current_usage: Current token usage
            max_budget: Maximum allowed budget
            selected_chunks: Selected chunks
            chunk_modes: Chunk mode mappings
            
        Returns:
            List of constraint violations
        """
        constraints = BudgetConstraints(
            max_budget=max_budget,
            current_usage=current_usage
        )
        
        violations = self.validate_budget(constraints)
        
        # Additional selection-specific validations
        if selected_chunks:
            # Check for duplicate chunks
            chunk_ids = [chunk.id for chunk in selected_chunks if hasattr(chunk, 'id')]
            if len(chunk_ids) != len(set(chunk_ids)):
                violation = ConstraintViolation(
                    violation_type=ViolationType.ALLOCATION,
                    severity='critical', 
                    message="Duplicate chunks in selection",
                    expected_value=float(len(chunk_ids)),
                    actual_value=float(len(set(chunk_ids))),
                    violation_amount=float(len(chunk_ids) - len(set(chunk_ids))),
                    constraint_name="unique_chunks",
                    recommendation="Fix selection algorithm to prevent duplicates"
                )
                violations.append(violation)
            
            # Check for mode consistency
            for chunk in selected_chunks:
                if hasattr(chunk, 'id') and chunk.id not in chunk_modes:
                    violation = ConstraintViolation(
                        violation_type=ViolationType.ALLOCATION,
                        severity='warning',
                        message=f"Chunk {chunk.id} has no mode mapping",
                        expected_value=1.0,
                        actual_value=0.0,
                        violation_amount=1.0,
                        constraint_name="mode_consistency",
                        recommendation="Ensure all selected chunks have mode mappings"
                    )
                    violations.append(violation)
        
        return violations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics and history summary."""
        # Count violations by type and severity
        violation_counts = {
            'by_type': {},
            'by_severity': {}
        }
        
        for violation in self._violation_history:
            # Count by type
            type_key = violation.violation_type.value
            violation_counts['by_type'][type_key] = \
                violation_counts['by_type'].get(type_key, 0) + 1
            
            # Count by severity
            severity_key = violation.severity
            violation_counts['by_severity'][severity_key] = \
                violation_counts['by_severity'].get(severity_key, 0) + 1
        
        # Recent violations (last 10)
        recent_violations = [
            {
                'type': v.violation_type.value,
                'severity': v.severity,
                'message': v.message,
                'violation_ratio': v.get_violation_ratio(),
                'timestamp': v.timestamp,
            }
            for v in self._violation_history[-10:]
        ]
        
        return {
            'total_validations': self._validation_count,
            'total_violations': len(self._violation_history),
            'violation_rate': len(self._violation_history) / max(1, self._validation_count),
            'violation_counts': violation_counts,
            'recent_violations': recent_violations,
        }
    
    def clear_history(self):
        """Clear violation history (useful for testing)."""
        self._violation_history.clear()
        self._validation_count = 0
        logger.info("Cleared budget validator history")
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on violation history."""
        recommendations = set()
        
        # Analyze recent violations for patterns
        recent_violations = self._violation_history[-50:]  # Last 50 violations
        
        overflow_count = sum(1 for v in recent_violations if v.violation_type == ViolationType.OVERFLOW)
        underflow_count = sum(1 for v in recent_violations if v.violation_type == ViolationType.UNDERFLOW)
        
        if overflow_count > 0:
            recommendations.add(
                "Frequent budget overflow detected. Consider increasing demotion "
                "aggressiveness or implementing stricter budget tracking."
            )
        
        if underflow_count > 5:
            recommendations.add(
                "Frequent budget underflow detected. Consider reducing budget "
                "targets or improving selection algorithm efficiency."
            )
        
        utilization_violations = [
            v for v in recent_violations 
            if v.violation_type == ViolationType.UTILIZATION
        ]
        if len(utilization_violations) > 10:
            recommendations.add(
                "Poor budget utilization detected consistently. Review selection "
                "algorithm parameters or reduce budget allocation."
            )
        
        if not recommendations:
            recommendations.add("Budget constraints are well maintained. Continue current approach.")
        
        return list(recommendations)
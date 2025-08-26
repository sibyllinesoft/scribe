"""Budget validation oracle for PackRepo."""

from __future__ import annotations

import time
from typing import Dict, Any, Optional

from . import Oracle, OracleReport, OracleResult
from ..packfmt.base import PackFormat


class BudgetOracle(Oracle):
    """Oracle for validating budget constraints and token accounting."""
    
    category = "budget"
    
    def name(self) -> str:
        return "budget_validation"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate budget constraints."""
        start_time = time.time()
        
        try:
            errors = []
            details = {}
            
            # Get budget information
            index = pack.index
            target_budget = index.target_budget
            actual_tokens = index.actual_tokens
            details["target_budget"] = target_budget
            details["actual_tokens"] = actual_tokens
            
            if target_budget <= 0:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="No target budget set, skipping budget validation",
                    details=details,
                    execution_time=time.time() - start_time
                )
            
            # Budget constraint: 0 overflow (hard constraint)
            if actual_tokens > target_budget:
                overflow = actual_tokens - target_budget
                overflow_pct = (overflow / target_budget) * 100
                errors.append(f"Budget overflow: {actual_tokens} > {target_budget} tokens (+{overflow} tokens, +{overflow_pct:.1f}%)")
                details["budget_overflow"] = overflow
                details["overflow_percentage"] = overflow_pct
            
            # Underflow constraint: ≤0.5% allowed
            utilization = actual_tokens / target_budget
            details["utilization"] = utilization
            
            if utilization < 0.995:  # Less than 99.5%
                underflow = (target_budget - actual_tokens) / target_budget
                details["underflow_percentage"] = underflow * 100
                
                if underflow > 0.005:  # More than 0.5%
                    errors.append(f"Excessive underflow: {underflow:.3f} ({underflow*100:.1f}%) > 0.5%")
                    details["underflow_violation"] = True
            
            # Validate token sum consistency
            if index.chunks:
                chunk_token_sum = sum(chunk.get('selected_tokens', 0) for chunk in index.chunks)
                details["chunk_token_sum"] = chunk_token_sum
                
                if abs(chunk_token_sum - actual_tokens) > 1:  # Allow 1 token rounding
                    errors.append(f"Token sum mismatch: chunks={chunk_token_sum}, actual={actual_tokens}")
                    details["token_sum_mismatch"] = abs(chunk_token_sum - actual_tokens)
            
            # Validate budget utilization calculation
            calculated_utilization = actual_tokens / target_budget if target_budget > 0 else 0
            reported_utilization = index.budget_utilization
            
            if abs(calculated_utilization - reported_utilization) > 0.001:  # Allow small float differences
                errors.append(f"Budget utilization mismatch: calculated={calculated_utilization:.3f}, reported={reported_utilization:.3f}")
                details["utilization_mismatch"] = abs(calculated_utilization - reported_utilization)
            
            # Check for negative tokens
            if actual_tokens < 0:
                errors.append(f"Negative token count: {actual_tokens}")
                details["negative_tokens"] = True
            
            # Validate per-chunk token counts
            negative_chunk_tokens = []
            for i, chunk in enumerate(index.chunks or []):
                chunk_tokens = chunk.get('selected_tokens', 0)
                if chunk_tokens < 0:
                    negative_chunk_tokens.append((i, chunk_tokens, chunk.get('id', 'unknown')))
            
            if negative_chunk_tokens:
                errors.append(f"Negative chunk token counts: {negative_chunk_tokens}")
                details["negative_chunk_tokens"] = negative_chunk_tokens
            
            # Overall result
            if errors:
                result = OracleResult.FAIL
                message = f"Budget validation failed: {'; '.join(errors)}"
            else:
                result = OracleResult.PASS
                message = f"Budget validation passed: {actual_tokens}/{target_budget} tokens ({utilization:.1%} utilization)"
            
            return OracleReport(
                oracle_name=self.name(),
                result=result,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.ERROR,
                message=f"Budget oracle execution failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )


class BudgetEnforcementOracle(Oracle):
    """Runtime budget enforcement oracle."""
    
    category = "budget"
    
    def name(self) -> str:
        return "budget_enforcement"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Enforce budget during selection process."""
        start_time = time.time()
        
        try:
            # This oracle checks that budget enforcement was properly applied during selection
            index = pack.index
            target_budget = index.target_budget
            actual_tokens = index.actual_tokens
            
            details = {
                "target_budget": target_budget,
                "actual_tokens": actual_tokens,
                "enforcement_enabled": context.get("budget_enforcement", True) if context else True
            }
            
            if target_budget <= 0:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="No budget enforcement needed (no target budget)",
                    details=details,
                    execution_time=time.time() - start_time
                )
            
            # Check that selection process respected budget (zero overflow tolerance)
            if actual_tokens > target_budget:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.FAIL,
                    message=f"Budget enforcement failed: selection exceeded budget by {actual_tokens - target_budget} tokens",
                    details={**details, "overflow": actual_tokens - target_budget},
                    execution_time=time.time() - start_time
                )
            
            # Check that selection made good use of budget (not excessive underutilization)
            utilization = actual_tokens / target_budget
            if utilization < 0.5:  # Less than 50% utilization might indicate poor selection
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.FAIL,
                    message=f"Very low budget utilization: {utilization:.1%} suggests selection algorithm issue",
                    details={**details, "utilization": utilization, "low_utilization": True},
                    execution_time=time.time() - start_time
                )
            
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.PASS,
                message=f"Budget enforcement successful: {actual_tokens}/{target_budget} tokens ({utilization:.1%})",
                details={**details, "utilization": utilization},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.ERROR,
                message=f"Budget enforcement oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )
"""Budget enforcement system for parity-controlled comparisons."""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..tokenizer.base import Tokenizer
from ..selector.base import SelectionResult, PackResult

logger = logging.getLogger(__name__)


@dataclass
class BudgetConstraints:
    """Budget constraints for parity enforcement."""
    
    # Target budget
    target_budget: int
    
    # Tolerance (default ±5% as per TODO.md)
    tolerance_percent: float = 5.0
    
    # Separate selection and decode budgets
    separate_decode_budget: bool = True
    decode_budget_multiplier: float = 2.0  # Decode budget = selection * multiplier
    
    @property
    def min_budget(self) -> int:
        """Minimum acceptable budget."""
        return int(self.target_budget * (1 - self.tolerance_percent / 100))
    
    @property
    def max_budget(self) -> int:
        """Maximum acceptable budget."""
        return int(self.target_budget * (1 + self.tolerance_percent / 100))
    
    @property
    def target_decode_budget(self) -> int:
        """Target decode budget if separate."""
        if self.separate_decode_budget:
            return int(self.target_budget * self.decode_budget_multiplier)
        return self.target_budget
    
    def is_within_tolerance(self, actual_budget: int) -> bool:
        """Check if actual budget is within tolerance."""
        return self.min_budget <= actual_budget <= self.max_budget
    
    def get_budget_deviation_percent(self, actual_budget: int) -> float:
        """Get budget deviation as percentage."""
        return ((actual_budget - self.target_budget) / self.target_budget) * 100


@dataclass
class BudgetReport:
    """Report on budget enforcement results."""
    
    variant_id: str
    target_budget: int
    actual_budget: int
    constraints: BudgetConstraints
    
    # Compliance
    within_tolerance: bool
    deviation_percent: float
    
    # Token breakdown
    selection_tokens: int
    decode_tokens: Optional[int] = None
    
    # Metadata
    tokenizer_name: str = "unknown"
    tokenizer_version: str = "unknown"
    
    @property
    def compliance_status(self) -> str:
        """Get compliance status string."""
        if self.within_tolerance:
            return "COMPLIANT"
        elif self.deviation_percent > 0:
            return "OVER_BUDGET"
        else:
            return "UNDER_BUDGET"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "variant_id": self.variant_id,
            "target_budget": self.target_budget,
            "actual_budget": self.actual_budget,
            "deviation_percent": round(self.deviation_percent, 2),
            "compliance_status": self.compliance_status,
            "within_tolerance": self.within_tolerance,
            "tokenizer": f"{self.tokenizer_name}:{self.tokenizer_version}",
            "selection_tokens": self.selection_tokens,
            "decode_tokens": self.decode_tokens,
        }


class BudgetEnforcer:
    """
    Budget enforcement system for parity-controlled comparisons.
    
    Ensures all variants use the same downstream token budget (±5% tolerance)
    as required by TODO.md for fair comparison.
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """Initialize budget enforcer."""
        self.tokenizer = tokenizer
        self._reports: Dict[str, BudgetReport] = {}
    
    def create_constraints(
        self, 
        target_budget: int, 
        tolerance_percent: float = 5.0,
        separate_decode_budget: bool = True
    ) -> BudgetConstraints:
        """Create budget constraints for parity enforcement."""
        return BudgetConstraints(
            target_budget=target_budget,
            tolerance_percent=tolerance_percent,
            separate_decode_budget=separate_decode_budget,
        )
    
    def validate_budget_parity(
        self,
        results: Dict[str, PackResult],
        constraints: BudgetConstraints
    ) -> Dict[str, BudgetReport]:
        """
        Validate budget parity across multiple variant results.
        
        Args:
            results: Dictionary of variant_id -> PackResult
            constraints: Budget constraints to enforce
            
        Returns:
            Dictionary of variant_id -> BudgetReport
        """
        reports = {}
        
        for variant_id, pack_result in results.items():
            report = self.validate_single_budget(
                variant_id, pack_result, constraints
            )
            reports[variant_id] = report
            self._reports[variant_id] = report
        
        # Log parity summary
        self._log_parity_summary(reports, constraints)
        
        return reports
    
    def validate_single_budget(
        self,
        variant_id: str,
        pack_result: PackResult,
        constraints: BudgetConstraints
    ) -> BudgetReport:
        """
        Validate budget for a single variant result.
        
        Args:
            variant_id: Variant identifier
            pack_result: Pack result to validate
            constraints: Budget constraints
            
        Returns:
            Budget report for the variant
        """
        actual_budget = pack_result.selection.total_tokens
        deviation_percent = constraints.get_budget_deviation_percent(actual_budget)
        within_tolerance = constraints.is_within_tolerance(actual_budget)
        
        # Get tokenizer information
        tokenizer_name = getattr(self.tokenizer, 'name', 'unknown')
        tokenizer_version = getattr(self.tokenizer, 'version', 'unknown')
        
        # Create report
        report = BudgetReport(
            variant_id=variant_id,
            target_budget=constraints.target_budget,
            actual_budget=actual_budget,
            constraints=constraints,
            within_tolerance=within_tolerance,
            deviation_percent=deviation_percent,
            selection_tokens=actual_budget,
            decode_tokens=None,  # Would be filled by decode phase
            tokenizer_name=tokenizer_name,
            tokenizer_version=tokenizer_version,
        )
        
        # Log result
        status_emoji = "✅" if within_tolerance else "❌"
        logger.info(
            f"{status_emoji} Budget validation {variant_id}: "
            f"{actual_budget}/{constraints.target_budget} tokens "
            f"({deviation_percent:+.1f}%) - {report.compliance_status}"
        )
        
        return report
    
    def enforce_budget_during_selection(
        self,
        current_tokens: int,
        additional_tokens: int,
        constraints: BudgetConstraints
    ) -> Tuple[bool, str]:
        """
        Enforce budget constraints during selection process.
        
        Args:
            current_tokens: Current token count
            additional_tokens: Tokens to be added
            constraints: Budget constraints
            
        Returns:
            Tuple of (can_add, reason)
        """
        projected_total = current_tokens + additional_tokens
        
        # Hard constraint: never exceed max budget
        if projected_total > constraints.max_budget:
            return False, f"Would exceed max budget ({projected_total} > {constraints.max_budget})"
        
        # Warning: approaching target budget
        if projected_total > constraints.target_budget:
            return True, f"Approaching target budget ({projected_total}/{constraints.target_budget})"
        
        return True, "Within budget"
    
    def get_budget_status(self, variant_id: str) -> Optional[BudgetReport]:
        """Get budget status for a variant."""
        return self._reports.get(variant_id)
    
    def get_all_reports(self) -> Dict[str, BudgetReport]:
        """Get all budget reports."""
        return self._reports.copy()
    
    def export_budget_report(self, output_path: Path) -> Dict[str, Any]:
        """
        Export comprehensive budget report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report summary dictionary
        """
        import json
        from datetime import datetime
        
        # Collect all report data
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "tokenizer": {
                "name": getattr(self.tokenizer, 'name', 'unknown'),
                "version": getattr(self.tokenizer, 'version', 'unknown'),
            },
            "variants": {
                variant_id: report.get_summary() 
                for variant_id, report in self._reports.items()
            },
        }
        
        # Calculate summary statistics
        if self._reports:
            deviations = [report.deviation_percent for report in self._reports.values()]
            compliant_count = sum(1 for report in self._reports.values() if report.within_tolerance)
            
            report_data["summary"] = {
                "total_variants": len(self._reports),
                "compliant_variants": compliant_count,
                "compliance_rate": compliant_count / len(self._reports),
                "avg_deviation_percent": sum(deviations) / len(deviations),
                "max_deviation_percent": max(deviations),
                "min_deviation_percent": min(deviations),
            }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Budget report exported to {output_path}")
        
        return report_data["summary"] if "summary" in report_data else {}
    
    def _log_parity_summary(self, reports: Dict[str, BudgetReport], constraints: BudgetConstraints):
        """Log parity validation summary."""
        compliant = [r for r in reports.values() if r.within_tolerance]
        non_compliant = [r for r in reports.values() if not r.within_tolerance]
        
        logger.info(
            f"Budget parity validation: {len(compliant)}/{len(reports)} variants compliant "
            f"(target={constraints.target_budget}, tolerance=±{constraints.tolerance_percent}%)"
        )
        
        if non_compliant:
            logger.warning("Non-compliant variants:")
            for report in non_compliant:
                logger.warning(
                    f"  {report.variant_id}: {report.actual_budget} tokens "
                    f"({report.deviation_percent:+.1f}%) - {report.compliance_status}"
                )
    
    def clear_reports(self):
        """Clear all stored reports."""
        self._reports.clear()
        logger.info("Cleared all budget reports")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get budget enforcer performance metrics."""
        return {
            "total_validations": len(self._reports),
            "compliant_validations": sum(1 for r in self._reports.values() if r.within_tolerance),
            "tokenizer_name": getattr(self.tokenizer, 'name', 'unknown'),
            "tokenizer_version": getattr(self.tokenizer, 'version', 'unknown'),
        }
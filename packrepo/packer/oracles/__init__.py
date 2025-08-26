"""Oracle system for PackRepo runtime validation and contracts."""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ..packfmt.base import PackFormat, PackIndex


class OracleResult(Enum):
    """Oracle validation result."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class OracleReport:
    """Report from oracle validation."""
    oracle_name: str
    result: OracleResult
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class Oracle(ABC):
    """Base oracle class for pack validation."""
    
    @abstractmethod
    def name(self) -> str:
        """Return oracle name."""
        pass
    
    @abstractmethod
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate pack and return report."""
        pass
    
    def should_run(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if oracle should run in current context."""
        return True


class OracleRegistry:
    """Registry for managing oracles."""
    
    def __init__(self):
        self._oracles: List[Oracle] = []
    
    def register(self, oracle: Oracle):
        """Register an oracle."""
        self._oracles.append(oracle)
    
    def get_oracles(self, category: Optional[str] = None) -> List[Oracle]:
        """Get oracles, optionally filtered by category."""
        if category is None:
            return self._oracles.copy()
        
        return [o for o in self._oracles if hasattr(o, 'category') and o.category == category]
    
    def validate_pack(
        self, 
        pack: PackFormat, 
        context: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None
    ) -> List[OracleReport]:
        """Run all applicable oracles on a pack."""
        reports = []
        context = context or {}
        
        for oracle in self._oracles:
            # Skip if category filtering is enabled and oracle doesn't match
            if categories and hasattr(oracle, 'category') and oracle.category not in categories:
                continue
                
            if oracle.should_run(context):
                try:
                    report = oracle.validate(pack, context)
                    reports.append(report)
                except Exception as e:
                    reports.append(OracleReport(
                        oracle_name=oracle.name(),
                        result=OracleResult.ERROR,
                        message=f"Oracle execution failed: {str(e)}",
                        details={"exception": type(e).__name__}
                    ))
        
        return reports
    
    def check_all_passed(self, reports: List[OracleReport]) -> bool:
        """Check if all oracle reports passed."""
        return all(r.result == OracleResult.PASS for r in reports if r.result != OracleResult.SKIP)


# Global oracle registry
_global_registry = OracleRegistry()


def get_global_registry() -> OracleRegistry:
    """Get the global oracle registry."""
    return _global_registry


def register_oracle(oracle: Oracle):
    """Register oracle in global registry."""
    _global_registry.register(oracle)


def validate_pack_with_oracles(
    pack: PackFormat, 
    context: Optional[Dict[str, Any]] = None,
    categories: Optional[List[str]] = None
) -> Tuple[bool, List[OracleReport]]:
    """Validate pack with all registered oracles."""
    reports = _global_registry.validate_pack(pack, context, categories)
    success = _global_registry.check_all_passed(reports)
    return success, reports
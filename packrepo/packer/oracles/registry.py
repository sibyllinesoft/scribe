"""Oracle registry and initialization for PackRepo."""

from __future__ import annotations

from . import get_global_registry, register_oracle
from .budget import BudgetOracle, BudgetEnforcementOracle
from .determinism import DeterminismOracle, HashConsistencyOracle
from .anchors import AnchorResolutionOracle, SectionIntegrityOracle
from .selection import SelectionPropertiesOracle, BudgetEfficiencyOracle
from .metamorphic import MetamorphicPropertiesOracle, PropertyCoverageOracle


def register_all_oracles():
    """Register all built-in oracles with the global registry."""
    registry = get_global_registry()
    
    # Budget oracles
    register_oracle(BudgetOracle())
    register_oracle(BudgetEnforcementOracle())
    
    # Determinism oracles
    register_oracle(DeterminismOracle())
    register_oracle(HashConsistencyOracle())
    
    # Anchor oracles
    register_oracle(AnchorResolutionOracle())
    register_oracle(SectionIntegrityOracle())
    
    # Selection oracles
    register_oracle(SelectionPropertiesOracle())
    register_oracle(BudgetEfficiencyOracle())
    
    # Metamorphic oracles
    register_oracle(MetamorphicPropertiesOracle())
    register_oracle(PropertyCoverageOracle())


def get_oracle_categories():
    """Get list of available oracle categories."""
    return [
        "budget",
        "determinism", 
        "anchors",
        "selection",
        "metamorphic"
    ]


def validate_pack_with_category_oracles(pack, context=None, categories=None):
    """Convenience function to validate pack with specific oracle categories."""
    from . import validate_pack_with_oracles
    
    # Ensure all oracles are registered
    register_all_oracles()
    
    # Run validation
    return validate_pack_with_oracles(pack, context, categories)


# Auto-register on import
register_all_oracles()
"""
FastPath Feature Flag System

Centralized feature flag management for FastPath enhancements.
All flags default to False/0 to maintain backward compatibility.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class FastPathFeatureFlags:
    """Feature flags for FastPath system enhancements."""
    
    # V2 Policy Framework
    policy_v2: bool = False
    
    # Demotion System
    demote_enabled: bool = False
    
    # Patch System
    patch_enabled: bool = False
    
    # Router Enhancement
    router_enabled: bool = False
    
    # PageRank Centrality
    centrality_enabled: bool = False
    
    # Quotas + Density-Greedy (Workstream A)
    quotas_enabled: bool = False
    
    # Thompson Sampling Bandit (Workstream E)
    bandit_enabled: bool = False
    
    # Negative Controls for validation
    negative_control: Optional[str] = None  # scramble, flip, random_quota


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Get boolean environment variable with safe default."""
    value = os.environ.get(name, '').lower()
    if value in ('1', 'true', 'yes', 'on', 'enable', 'enabled'):
        return True
    elif value in ('0', 'false', 'no', 'off', 'disable', 'disabled'):
        return False
    else:
        return default


def _get_int_env(name: str, default: int = 0) -> int:
    """Get integer environment variable with safe default."""
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def load_feature_flags() -> FastPathFeatureFlags:
    """
    Load feature flags from environment variables.
    
    Environment Variables:
        FASTPATH_POLICY_V2: Enable V2 policy framework (default: False)
        FASTPATH_DEMOTE: Enable demotion system (default: False)  
        FASTPATH_PATCH: Enable patch system (default: False)
        FASTPATH_ROUTER: Enable router enhancement (default: False)
        FASTPATH_CENTRALITY: Enable PageRank centrality (default: False)
        FASTPATH_QUOTAS: Enable quotas + density-greedy (default: False)
        FASTPATH_BANDIT: Enable Thompson sampling bandit (default: False)
    
    Returns:
        FastPathFeatureFlags instance with current flag values
    """
    return FastPathFeatureFlags(
        policy_v2=_get_bool_env('FASTPATH_POLICY_V2'),
        demote_enabled=_get_bool_env('FASTPATH_DEMOTE'),
        patch_enabled=_get_bool_env('FASTPATH_PATCH'),
        router_enabled=_get_bool_env('FASTPATH_ROUTER'),
        centrality_enabled=_get_bool_env('FASTPATH_CENTRALITY'),
        quotas_enabled=_get_bool_env('FASTPATH_QUOTAS'),
        bandit_enabled=_get_bool_env('FASTPATH_BANDIT'),
        negative_control=os.environ.get('FASTPATH_NEGCTRL')
    )


# Global feature flags instance (lazy-loaded)
_feature_flags: Optional[FastPathFeatureFlags] = None


def get_feature_flags() -> FastPathFeatureFlags:
    """Get the global feature flags instance (singleton pattern)."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = load_feature_flags()
    return _feature_flags


def reload_feature_flags() -> FastPathFeatureFlags:
    """Force reload of feature flags from environment variables."""
    global _feature_flags
    _feature_flags = load_feature_flags()
    return _feature_flags
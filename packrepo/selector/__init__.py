"""
FastPath file selection algorithms.

Provides efficient selection methods for FastPath optimization:
- fast_facloc.py: Submodular facility location approximation
- mmr_sparse.py: Maximal Marginal Relevance with sparse features
"""

from __future__ import annotations

from .fast_facloc import FastFacilityLocation, SelectionResult
from .mmr_sparse import MMRSelector, MMRConfig

__all__ = [
    "FastFacilityLocation",
    "SelectionResult", 
    "MMRSelector",
    "MMRConfig",
]
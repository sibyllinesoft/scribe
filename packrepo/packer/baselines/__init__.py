"""Baseline implementations for PackRepo evaluation."""

from .base import BaselineSelector, BaselineConfig
from .v0a_readme_only import V0aReadmeOnlyBaseline
from .v0b_naive_concat import V0bNaiveConcatBaseline  
from .v0c_bm25_baseline import V0cBM25Baseline
from .v1_random import V1RandomBaseline
from .v2_recency import V2RecencyBaseline
from .v3_tfidf_pure import V3TfIdfPureBaseline
from .v4_semantic import V4SemanticBaseline

__all__ = [
    'BaselineSelector',
    'BaselineConfig',
    'V0aReadmeOnlyBaseline',
    'V0bNaiveConcatBaseline',
    'V0cBM25Baseline',
    'V1RandomBaseline',
    'V2RecencyBaseline', 
    'V3TfIdfPureBaseline',
    'V4SemanticBaseline',
]

def create_baseline_selector(variant: str) -> BaselineSelector:
    """
    Factory function to create baseline selectors.
    
    Args:
        variant: Baseline variant ('V0a', 'V0b', 'V0c', 'V1', 'V2', 'V3', 'V4')
        
    Returns:
        Appropriate baseline selector instance
        
    Raises:
        ValueError: For unknown variant
    """
    variant = variant.upper()
    
    if variant == 'V0A':
        return V0aReadmeOnlyBaseline()
    elif variant == 'V0B':
        return V0bNaiveConcatBaseline()
    elif variant == 'V0C':
        return V0cBM25Baseline()
    elif variant == 'V1':
        return V1RandomBaseline()
    elif variant == 'V2':
        return V2RecencyBaseline()
    elif variant == 'V3':
        return V3TfIdfPureBaseline()
    elif variant == 'V4':
        return V4SemanticBaseline()
    else:
        raise ValueError(
            f"Unknown baseline variant: {variant}. "
            f"Must be one of: V0a, V0b, V0c, V1, V2, V3, V4"
        )
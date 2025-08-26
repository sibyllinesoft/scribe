"""
Scribe - Advanced Repository Intelligence for LLM Code Analysis.

A submodular, budget-aware repository packer with multi-fidelity code chunks,
supporting deterministic and objective-driven modes with local LLM integration.
"""

from __future__ import annotations

__version__ = "1.0.0"

# Core library exports
from .packer.chunker import Chunk, ChunkKind, CodeChunker
from .packer.tokenizer import Tokenizer, TokenizerType
from .packer.selector import PackRequest, PackResult, RepositorySelector, SelectionConfig, SelectionMode, SelectionVariant
from .packer.packfmt import PackFormat, PackIndex, PackBody

# Scribe optimization system exports
from .fastpath import FastScanner, HeuristicScorer, TTLScheduler, ExecutionMode, Phase
from .selector import FastFacilityLocation, MMRSelector, SelectionResult, MMRConfig
from .docs import LinkGraphAnalyzer, TextPriorityScorer, LinkAnalysisResult, CentralityResult
from .tokenizer import TokenEstimator, EstimationResult, FinalizedPack
from .cli import FastPackCLI

# High-level API for easier usage
from .library import RepositoryPacker, ScribeConfig

__all__ = [
    # High-level API (recommended for most users)
    "RepositoryPacker",
    "ScribeConfig",
    
    # Original PackRepo exports
    "Chunk",
    "ChunkKind", 
    "CodeChunker",
    "Tokenizer",
    "TokenizerType",
    "PackRequest",
    "PackResult",
    "RepositorySelector",
    "SelectionConfig",
    "SelectionMode",
    "SelectionVariant",
    "PackFormat",
    "PackIndex",
    "PackBody",
    
    # Scribe system exports
    "FastScanner",
    "HeuristicScorer", 
    "TTLScheduler",
    "ExecutionMode",
    "Phase",
    "FastFacilityLocation",
    "MMRSelector",
    "SelectionResult",
    "MMRConfig",
    "LinkGraphAnalyzer",
    "TextPriorityScorer",
    "LinkAnalysisResult", 
    "CentralityResult",
    "TokenEstimator",
    "EstimationResult",
    "FinalizedPack",
    "FastPackCLI",
]
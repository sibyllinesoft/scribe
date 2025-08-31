"""
FastPath optimization system for PackRepo.

Provides zero-training, latency-bounded optimization with:
- FastPath mode: <10s performance using heuristics only
- Extended mode: <30s performance with AST parsing and centroids
- TTL-driven graceful degradation
- Rule-based text centrality (README++)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .fast_scan import FastScanner, ScanResult
from .heuristics import HeuristicScorer, ScoreComponents
from .ttl_scheduler import TTLScheduler, Phase, ExecutionMode
from .centrality import CentralityCalculator, PageRankComputer, DependencyGraph
from .incremental_pagerank import (
    IncrementalPageRankEngine,
    GraphDelta,
    PersonalizedPageRankQuery,
    IncrementalUpdateResult,
    create_graph_delta,
    create_incremental_pagerank_engine
)
from .integrated_v5 import FastPathEngine

__all__ = [
    "FastScanner",
    "ScanResult", 
    "HeuristicScorer",
    "ScoreComponents",
    "TTLScheduler",
    "Phase",
    "ExecutionMode",
    "CentralityCalculator",
    "PageRankComputer", 
    "DependencyGraph",
    "IncrementalPageRankEngine",
    "GraphDelta",
    "PersonalizedPageRankQuery",
    "IncrementalUpdateResult",
    "create_graph_delta",
    "create_incremental_pagerank_engine",
    "FastPathEngine",
]
"""
Performance gate tests for FastPath system.

Validates that FastPath meets strict performance requirements:
- FastPath p95 < 10s on representative repositories
- Extended p95 < 30s with full feature set
- Graceful degradation under time pressure
- Deterministic execution with --no-llm flag
"""

import pytest
import tempfile
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any

from ...fastpath import FastScanner, HeuristicScorer, TTLScheduler, ExecutionMode
from ...selector import MMRSelector
from ...docs import LinkGraphAnalyzer, TextPriorityScorer
from ...cli.fastpack import FastPackCLI


class TestPerformanceGates:
    """Test suite for FastPath performance requirements."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create test repository structure
            self._create_test_repo(repo_path)
            yield repo_path
            
    def _create_test_repo(self, repo_path: Path):
        """Create a representative test repository."""
        # Create README files
        (repo_path / "README.md").write_text("""
# Test Repository

This is a test repository for FastPath performance validation.

## Features
- FastPath scanning
- Heuristic scoring
- Document centrality

## Getting Started
See the documentation for more information.
        """)
        
        # Create source files
        src_dir = repo_path / "src"
        src_dir.mkdir()
        
        # Python files
        (src_dir / "main.py").write_text("""
import os
import sys
from typing import List, Dict, Optional

def main():
    \"\"\"Main entry point.\"\"\"
    print("Hello, world!")

class DataProcessor:
    \"\"\"Process data efficiently.\"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def process(self, data: List[Any]) -> List[Any]:
        \"\"\"Process input data.\"\"\"
        return [item for item in data if item is not None]

if __name__ == "__main__":
    main()
        """)
        
        # JavaScript files
        (src_dir / "app.js").write_text("""
const express = require('express');
const path = require('path');

class WebServer {
    constructor(port) {
        this.port = port;
        this.app = express();
        this.setupRoutes();
    }
    
    setupRoutes() {
        this.app.get('/', (req, res) => {
            res.json({ message: 'Hello FastPath!' });
        });
        
        this.app.get('/api/data', (req, res) => {
            res.json({ data: [1, 2, 3, 4, 5] });
        });
    }
    
    start() {
        this.app.listen(this.port, () => {
            console.log(`Server running on port ${this.port}`);
        });
    }
}

module.exports = WebServer;
        """)
        
        # Test files
        tests_dir = repo_path / "tests"
        tests_dir.mkdir()
        
        (tests_dir / "test_main.py").write_text("""
import unittest
from src.main import DataProcessor

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor({})
        
    def test_process_filters_none(self):
        data = [1, None, 2, None, 3]
        result = self.processor.process(data)
        self.assertEqual(result, [1, 2, 3])
        
    def test_process_empty_list(self):
        result = self.processor.process([])
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
        """)
        
        # Documentation
        docs_dir = repo_path / "docs"
        docs_dir.mkdir()
        
        (docs_dir / "architecture.md").write_text("""
# Architecture Overview

This document describes the system architecture.

## Components

### FastPath Scanner
Rapid file analysis using heuristics.

### Selector
File selection with diversity optimization.

### TTL Scheduler
Time-bounded execution management.
        """)
        
        (docs_dir / "api.md").write_text("""
# API Reference

## Classes

### FastScanner
Scans repository files quickly.

### HeuristicScorer  
Scores files using multiple heuristics.
        """)
        
        # Configuration files
        (repo_path / "package.json").write_text("""
{
    "name": "test-repo",
    "version": "1.0.0",
    "description": "Test repository for FastPath",
    "main": "src/app.js",
    "dependencies": {
        "express": "^4.18.0"
    }
}
        """)
        
        (repo_path / "requirements.txt").write_text("""
pytest>=7.0.0
flask>=2.0.0
requests>=2.28.0
        """)
        
    def test_fast_path_latency_gate(self, temp_repo):
        """Test FastPath meets <10s latency requirement."""
        execution_times = []
        
        # Run multiple iterations for statistical validation
        for i in range(5):  # Reduced for CI performance
            start_time = time.time()
            
            # Execute FastPath pipeline
            scheduler = TTLScheduler(ExecutionMode.FAST_PATH)
            scheduler.start_execution()
            
            # Scanning phase
            scanner = FastScanner(temp_repo, ttl_seconds=2.0)
            scan_result = scheduler.execute_phase(
                scheduler.Phase.SCAN,
                lambda: scanner.scan_repository()
            )
            assert scan_result.completed
            
            # Ranking phase
            scorer = HeuristicScorer()
            rank_result = scheduler.execute_phase(
                scheduler.Phase.RANK,
                lambda: scorer.score_all_files(scan_result.result)
            )
            assert rank_result.completed
            
            # Selection phase
            selector = MMRSelector()
            select_result = scheduler.execute_phase(
                scheduler.Phase.SELECT,
                lambda: selector.select_files(rank_result.result, 120000)
            )
            assert select_result.completed
            
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
        # Statistical validation
        p95_time = statistics.quantiles(execution_times, n=20)[18]  # 95th percentile
        avg_time = statistics.mean(execution_times)
        
        print(f"FastPath performance: avg={avg_time:.2f}s, p95={p95_time:.2f}s")
        
        # Assert performance gate
        assert p95_time < 10.0, f"FastPath p95 latency {p95_time:.2f}s exceeds 10s gate"
        assert avg_time < 8.0, f"FastPath average latency {avg_time:.2f}s exceeds expected 8s"
        
    def test_extended_mode_latency_gate(self, temp_repo):
        """Test Extended mode meets <30s latency requirement."""
        execution_times = []
        
        # Run multiple iterations
        for i in range(3):  # Fewer iterations for extended mode
            start_time = time.time()
            
            # Execute Extended pipeline
            scheduler = TTLScheduler(ExecutionMode.EXTENDED)
            scheduler.start_execution()
            
            # Fast phases (reuse FastPath implementation)
            scanner = FastScanner(temp_repo, ttl_seconds=2.0)
            scan_result = scheduler.execute_phase(
                scheduler.Phase.SCAN,
                lambda: scanner.scan_repository()
            )
            assert scan_result.completed
            
            scorer = HeuristicScorer()
            rank_result = scheduler.execute_phase(
                scheduler.Phase.RANK,
                lambda: scorer.score_all_files(scan_result.result)
            )
            assert rank_result.completed
            
            # Extended phases
            link_analyzer = LinkGraphAnalyzer(temp_repo)
            link_result = scheduler.execute_phase(
                scheduler.Phase.CENTROIDS,
                lambda: link_analyzer.analyze_link_graph(scan_result.result)
            )
            # Link analysis may fail gracefully
            
            text_scorer = TextPriorityScorer()
            enhanced_result = scheduler.execute_phase(
                scheduler.Phase.SELECT,
                lambda: text_scorer.select_priority_documents(
                    scan_result.result, 120000, 
                    link_result.result if link_result.completed else None
                )
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
        # Statistical validation
        max_time = max(execution_times)
        avg_time = statistics.mean(execution_times)
        
        print(f"Extended mode performance: avg={avg_time:.2f}s, max={max_time:.2f}s")
        
        # Assert performance gate
        assert max_time < 30.0, f"Extended mode max latency {max_time:.2f}s exceeds 30s gate"
        assert avg_time < 25.0, f"Extended mode average latency {avg_time:.2f}s exceeds expected 25s"
        
    def test_graceful_degradation(self, temp_repo):
        """Test graceful degradation under time pressure."""
        # Create scheduler with very tight time budget
        scheduler = TTLScheduler(ExecutionMode.EXTENDED)
        scheduler.total_budget = 5.0  # Force degradation
        scheduler.start_execution()
        
        scanner = FastScanner(temp_repo, ttl_seconds=1.0)  # Tight scanning budget
        scan_result = scheduler.execute_phase(
            scheduler.Phase.SCAN,
            lambda: scanner.scan_repository()
        )
        
        # Should complete even with tight budget
        assert scan_result.completed or scan_result.duration < 1.5
        
        # Check if mode was degraded
        execution_summary = scheduler.get_execution_summary()
        if execution_summary['degraded']:
            assert execution_summary['actual_mode'] == ExecutionMode.DEGRADED.value
            
    def test_deterministic_execution(self, temp_repo):
        """Test deterministic execution for reproducibility."""
        # Run same configuration multiple times
        results = []
        
        for i in range(3):
            scanner = FastScanner(temp_repo, ttl_seconds=2.0)
            scan_results = scanner.scan_repository()
            
            scorer = HeuristicScorer()
            scored_files = scorer.score_all_files(scan_results)
            
            # Extract deterministic data
            file_paths = [result.stats.path for result, score in scored_files]
            scores = [score.final_score for result, score in scored_files]
            
            results.append({
                'paths': file_paths,
                'scores': scores
            })
            
        # Verify deterministic results
        for i in range(1, len(results)):
            assert results[i]['paths'] == results[0]['paths'], "File ordering must be deterministic"
            
            # Scores should be identical (within floating point precision)
            for j, score in enumerate(results[i]['scores']):
                assert abs(score - results[0]['scores'][j]) < 1e-10, \
                    f"Scores must be deterministic: {score} vs {results[0]['scores'][j]}"
                    
    def test_token_budget_enforcement(self, temp_repo):
        """Test zero overflow budget enforcement."""
        from ...tokenizer import TokenEstimator
        from ...packer.tokenizer.implementations import create_tokenizer
        
        tokenizer = create_tokenizer('tiktoken', 'gpt-4')
        estimator = TokenEstimator(tokenizer)
        
        # Create selection
        scanner = FastScanner(temp_repo)
        scan_results = scanner.scan_repository()
        
        # Test with tight budget
        tight_budget = 1000  # Very small budget
        
        finalized = estimator.finalize_pack(scan_results[:5], tight_budget)
        
        # Assert zero overflow
        assert finalized.overflow_tokens == 0, \
            f"Budget overflow detected: {finalized.overflow_tokens} tokens"
            
        # Assert reasonable underflow
        underflow_rate = 1 - finalized.budget_utilization
        assert underflow_rate <= 0.005, \
            f"Underflow rate {underflow_rate:.3f} exceeds 0.5% limit"
            
    def test_readme_retention_gate(self, temp_repo):
        """Test 100% README retention requirement."""
        scanner = FastScanner(temp_repo)
        scan_results = scanner.scan_repository()
        
        # Find README files
        readme_files = [r for r in scan_results if r.stats.is_readme]
        assert len(readme_files) > 0, "Test repo must have README files"
        
        # Run selection
        scorer = HeuristicScorer()
        scored_files = scorer.score_all_files(scan_results)
        
        selector = MMRSelector()
        selected_files = selector.select_files(scored_files, 120000)
        
        # Check README retention
        selected_paths = {f.stats.path for f in selected_files}
        readme_paths = {f.stats.path for f in readme_files}
        
        retained_readmes = readme_paths & selected_paths
        retention_rate = len(retained_readmes) / len(readme_paths)
        
        assert retention_rate == 1.0, \
            f"README retention rate {retention_rate:.2f} below 100% requirement"
            
    @pytest.mark.slow
    def test_cli_performance_integration(self, temp_repo):
        """Integration test for CLI performance."""
        cli = FastPackCLI()
        
        # Test FastPath mode via CLI
        start_time = time.time()
        result = cli.run([
            str(temp_repo),
            '--mode', 'fast',
            '--budget', '50000',
            '--dry-run'
        ])
        execution_time = time.time() - start_time
        
        assert result == 0, "CLI should succeed"
        assert execution_time < 10.0, f"CLI FastPath took {execution_time:.2f}s > 10s limit"
        
    def test_memory_usage_bounds(self, temp_repo):
        """Test memory usage stays within reasonable bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss
        
        # Execute memory-intensive operations
        scanner = FastScanner(temp_repo)
        scan_results = scanner.scan_repository()
        
        scorer = HeuristicScorer()
        scored_files = scorer.score_all_files(scan_results)
        
        link_analyzer = LinkGraphAnalyzer(temp_repo)
        link_analysis = link_analyzer.analyze_link_graph(scan_results)
        
        end_memory = process.memory_info().rss
        memory_delta = end_memory - start_memory
        memory_mb = memory_delta / (1024 * 1024)
        
        print(f"Memory usage: {memory_mb:.2f} MB")
        
        # Assert reasonable memory usage (adjust based on test environment)
        assert memory_mb < 100, f"Memory usage {memory_mb:.2f}MB exceeds 100MB limit"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
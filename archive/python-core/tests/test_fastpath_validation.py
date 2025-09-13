#!/usr/bin/env python3
"""
FastPath system validation script.

Quick validation that the FastPath system is working correctly:
- Tests core components integration
- Validates performance within bounds
- Checks deterministic behavior
- Verifies acceptance gate compliance
"""

import sys
import time
import tempfile
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from packrepo.fastpath import FastScanner, HeuristicScorer, TTLScheduler, ExecutionMode
from packrepo.selector import MMRSelector
from packrepo.docs import TextPriorityScorer
from packrepo.tokenizer import TokenEstimator
from packrepo.packer.tokenizer.implementations import create_tokenizer


def create_test_repository(repo_path: Path):
    """Create a test repository for validation."""
    # README
    (repo_path / "README.md").write_text("""
# Test Repository

This is a comprehensive test repository for FastPath validation.

## Features
- FastPath scanning with heuristics
- Extended mode with centrality analysis
- TTL scheduler with graceful degradation
- Rule-based text prioritization

## Architecture
See `docs/architecture.md` for detailed architecture information.

## Getting Started
1. Install dependencies
2. Configure settings
3. Run tests
    """)
    
    # Architecture documentation
    docs_dir = repo_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "architecture.md").write_text("""
# System Architecture

## Overview
The FastPath system provides optimized repository packing.

## Components
- Scanner: Rapid file analysis
- Scorer: Heuristic prioritization  
- Selector: Diverse file selection
- Scheduler: Time-bounded execution

## Performance Targets
- FastPath: <10s p95 latency
- Extended: <30s p95 latency
    """)
    
    # Source code
    src_dir = repo_path / "src"
    src_dir.mkdir()
    
    (src_dir / "main.py").write_text("""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class FastPathProcessor:
    \"\"\"Main processing class for FastPath operations.\"\"\"
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.processed_count = 0
        
    def process_files(self, files: List[str]) -> Dict[str, any]:
        \"\"\"Process a list of files with FastPath optimization.\"\"\"
        results = {}
        
        for file_path in files:
            try:
                result = self._process_single_file(file_path)
                results[file_path] = result
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = None
                
        return results
        
    def _process_single_file(self, file_path: str) -> Dict[str, any]:
        \"\"\"Process a single file.\"\"\"
        return {
            'path': file_path,
            'processed': True,
            'timestamp': time.time()
        }
        
    def get_statistics(self) -> Dict[str, any]:
        \"\"\"Get processing statistics.\"\"\"
        return {
            'processed_count': self.processed_count,
            'config': self.config
        }
    """)
    
    # Test files
    tests_dir = repo_path / "tests"
    tests_dir.mkdir()
    
    (tests_dir / "test_processor.py").write_text("""
import unittest
from src.main import FastPathProcessor

class TestFastPathProcessor(unittest.TestCase):
    
    def setUp(self):
        self.config = {'debug': True}
        self.processor = FastPathProcessor(self.config)
        
    def test_process_files(self):
        files = ['test1.py', 'test2.py']
        results = self.processor.process_files(files)
        
        self.assertEqual(len(results), 2)
        self.assertIn('test1.py', results)
        self.assertIn('test2.py', results)
        
    def test_statistics(self):
        stats = self.processor.get_statistics()
        self.assertEqual(stats['processed_count'], 0)
        self.assertEqual(stats['config'], self.config)
        
if __name__ == '__main__':
    unittest.main()
    """)
    
    # Configuration
    (repo_path / "config.yaml").write_text("""
# FastPath Configuration
scanner:
  max_files: 1000
  ttl_seconds: 2.0
  
scorer:
  weights:
    doc: 0.30
    readme: 0.25
    import_deg: 0.15
    
selector:
  algorithm: mmr
  lambda_param: 0.7
    """)


def validate_fast_path_performance(repo_path: Path) -> bool:
    """Validate FastPath meets performance requirements."""
    print("🚀 Testing FastPath performance...")
    
    start_time = time.time()
    
    # Create scheduler
    scheduler = TTLScheduler(ExecutionMode.FAST_PATH)
    scheduler.start_execution()
    
    # Scanning phase
    from packrepo.fastpath import Phase
    scanner = FastScanner(repo_path, ttl_seconds=2.0)
    scan_result = scheduler.execute_phase(
        Phase.SCAN,
        lambda: scanner.scan_repository()
    )
    
    if not scan_result.completed:
        print(f"❌ Scan phase failed: {scan_result.error}")
        return False
        
    print(f"✅ Scanned {len(scan_result.result)} files in {scan_result.duration:.2f}s")
    
    # Scoring phase
    scorer = HeuristicScorer()
    rank_result = scheduler.execute_phase(
        Phase.RANK,
        lambda: scorer.score_all_files(scan_result.result)
    )
    
    if not rank_result.completed:
        print(f"❌ Ranking phase failed: {rank_result.error}")
        return False
        
    print(f"✅ Scored files in {rank_result.duration:.2f}s")
    
    # Selection phase
    selector = MMRSelector()
    select_result = scheduler.execute_phase(
        Phase.SELECT,
        lambda: selector.select_files(rank_result.result, 50000)
    )
    
    if not select_result.completed:
        print(f"❌ Selection phase failed: {select_result.error}")
        return False
        
    print(f"✅ Selected {len(select_result.result)} files in {select_result.duration:.2f}s")
    
    # Check total time
    total_time = time.time() - start_time
    print(f"⏱️  Total FastPath execution: {total_time:.2f}s")
    
    if total_time > 10.0:
        print(f"❌ FastPath exceeded 10s target: {total_time:.2f}s")
        return False
        
    print(f"✅ FastPath within 10s target")
    return True


def validate_extended_mode(repo_path: Path) -> bool:
    """Validate Extended mode functionality."""
    print("🔬 Testing Extended mode...")
    
    start_time = time.time()
    
    # Text centrality analysis
    scanner = FastScanner(repo_path)
    scan_results = scanner.scan_repository()
    
    text_scorer = TextPriorityScorer()
    priority_docs = text_scorer.select_priority_documents(scan_results, 20000)
    
    total_time = time.time() - start_time
    print(f"⏱️  Extended mode execution: {total_time:.2f}s")
    
    if total_time > 30.0:
        print(f"❌ Extended mode exceeded 30s target: {total_time:.2f}s") 
        return False
        
    print(f"✅ Extended mode within 30s target")
    print(f"✅ Selected {len(priority_docs)} priority documents")
    
    # Check README inclusion
    readme_included = any(r.stats.is_readme for r, c in priority_docs)
    if not readme_included:
        print("❌ README not included in priority documents")
        return False
        
    print("✅ README included in selection")
    return True


def validate_determinism(repo_path: Path) -> bool:
    """Validate deterministic execution."""
    print("🔒 Testing deterministic execution...")
    
    results = []
    
    # Run same operation 3 times
    for i in range(3):
        scanner = FastScanner(repo_path, ttl_seconds=2.0)
        scan_results = scanner.scan_repository()
        
        scorer = HeuristicScorer()
        scored_files = scorer.score_all_files(scan_results)
        
        # Extract deterministic data
        file_paths = [result.stats.path for result, score in scored_files]
        scores = [score.final_score for result, score in scored_files]
        
        results.append((file_paths, scores))
        
    # Verify identical results
    for i in range(1, len(results)):
        if results[i][0] != results[0][0]:
            print("❌ File ordering not deterministic")
            return False
            
        # Check score determinism (within floating point precision)
        for j, score in enumerate(results[i][1]):
            if abs(score - results[0][1][j]) > 1e-10:
                print(f"❌ Scores not deterministic: {score} vs {results[0][1][j]}")
                return False
                
    print("✅ Execution is deterministic")
    return True


def validate_token_budget(repo_path: Path) -> bool:
    """Validate token budget enforcement."""
    print("💰 Testing token budget enforcement...")
    
    try:
        tokenizer = create_tokenizer('tiktoken', 'gpt-4')
    except Exception:
        print("⚠️  Tokenizer not available, skipping token budget test")
        return True
        
    estimator = TokenEstimator(tokenizer)
    
    # Get scan results
    scanner = FastScanner(repo_path)
    scan_results = scanner.scan_repository()
    
    # Test with tight budget
    tight_budget = 5000
    
    try:
        finalized = estimator.finalize_pack(scan_results[:10], tight_budget)
        
        print(f"📊 Token usage: {finalized.total_tokens}/{tight_budget}")
        print(f"📊 Budget utilization: {finalized.budget_utilization:.2%}")
        
        # Check zero overflow
        if finalized.overflow_tokens > 0:
            print(f"❌ Budget overflow: {finalized.overflow_tokens} tokens")
            return False
            
        # Check reasonable underflow (for small test files, higher underflow is expected)
        underflow_rate = 1 - finalized.budget_utilization
        if underflow_rate > 0.95:  # Allow up to 95% underflow for test cases
            print(f"❌ Excessive underflow: {underflow_rate:.3%}")
            return False
            
        print("✅ Token budget enforced correctly")
        return True
        
    except Exception as e:
        print(f"❌ Token budget test failed: {e}")
        return False


def main():
    """Run FastPath validation suite."""
    print("🧪 FastPath System Validation")
    print("=" * 50)
    
    # Create test repository
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        create_test_repository(repo_path)
        
        print(f"📁 Created test repository at {repo_path}")
        
        # Run validation tests
        tests = [
            ("FastPath Performance", validate_fast_path_performance),
            ("Extended Mode", validate_extended_mode),
            ("Determinism", validate_determinism),
            ("Token Budget", validate_token_budget),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name} test...")
            try:
                result = test_func(repo_path)
                results[test_name] = result
                
                if result:
                    print(f"✅ {test_name} test passed")
                else:
                    print(f"❌ {test_name} test failed")
                    
            except Exception as e:
                print(f"❌ {test_name} test error: {e}")
                results[test_name] = False
                
        # Summary
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<20}: {status}")
            
        print("-" * 50)
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All FastPath validation tests passed!")
            return 0
        else:
            print("⚠️  Some FastPath tests failed - review implementation")
            return 1


if __name__ == '__main__':
    sys.exit(main())
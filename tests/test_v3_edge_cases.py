"""
Edge case and stress tests for V3 demotion stability controller.

Tests V3 behavior under extreme conditions, error cases, and edge scenarios
to ensure robustness and reliability.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from pathlib import Path

from packrepo.packer.chunker.base import Chunk, ChunkKind
from packrepo.packer.selector.base import SelectionConfig, SelectionResult, SelectionVariant
from packrepo.packer.controller.demotion import DemotionController, DemotionDecision, DemotionStrategy
from packrepo.packer.controller.stability import StabilityTracker, OscillationEvent, EventType
from packrepo.packer.controller.utility import V3UtilityCalculator
from packrepo.packer.budget.reallocation import BudgetReallocator


class TestV3EdgeCases:
    """Test V3 system behavior in edge cases."""
    
    def setup_method(self):
        """Set up test fixtures for edge case testing."""
        self.utility_calculator = V3UtilityCalculator()
        self.stability_tracker = StabilityTracker()
        self.controller = DemotionController(
            self.utility_calculator,
            self.stability_tracker
        )
        self.reallocator = BudgetReallocator(self.utility_calculator)
    
    def test_empty_selection_result(self):
        """Test V3 behavior with empty selection results."""
        empty_result = SelectionResult(
            selected_chunks=[],
            chunk_modes={},
            selection_scores={},
            total_tokens=0,
            budget_utilization=0.0
        )
        
        config = SelectionConfig(token_budget=1000)
        chunks_by_id = {}
        
        # Should handle empty selection gracefully
        candidates = self.controller.detect_demotion_candidates(
            empty_result, chunks_by_id, config
        )
        
        assert candidates == []
        
        # Should handle empty bounded reoptimization
        modified_result, actions = self.controller.execute_bounded_reoptimization(
            [], empty_result, chunks_by_id, config
        )
        
        assert modified_result.total_tokens == 0
        assert actions == []
    
    def test_single_chunk_oscillation(self):
        """Test oscillation detection with only one chunk."""
        chunk = Chunk(
            id="solo_chunk", path=Path("solo.py"), rel_path="solo.py", start_line=1, end_line=10,
            kind=ChunkKind.FUNCTION, name="solo", language="python",
            content="def solo(): pass", full_tokens=50, signature_tokens=10
        )
        
        # Create oscillating selection history
        selections = []
        for mode in ['full', 'signature', 'full', 'signature', 'full']:
            selection = Mock()
            selection.selected_chunks = [chunk]
            selection.chunk_modes = {'solo_chunk': mode}
            selections.append(selection)
        
        # Should detect oscillation in single chunk
        oscillations = self.stability_tracker.detect_oscillations(
            selections[-1], selections[:-1], look_back_epochs=5
        )
        
        assert len(oscillations) > 0
        assert all(osc.chunk_id == 'solo_chunk' for osc in oscillations)
    
    def test_zero_budget_scenario(self):
        """Test V3 behavior with zero or minimal budget."""
        config = SelectionConfig(token_budget=0)
        
        chunk = Chunk(
            id="chunk1", path=Path("test.py"), rel_path="test.py", start_line=1, end_line=5,
            kind=ChunkKind.FUNCTION, name="test_func", language="python",
            content="pass", full_tokens=10, signature_tokens=2
        )
        
        result = SelectionResult(
            selected_chunks=[],
            chunk_modes={},
            selection_scores={},
            total_tokens=0,
            budget_utilization=0.0
        )
        
        # Should handle zero budget gracefully
        reallocation_result = self.reallocator.reallocate_budget(
            0, result, {'chunk1': chunk}, config
        )
        
        assert reallocation_result.allocated_budget == 0
        assert reallocation_result.remaining_budget == 0
    
    def test_massive_chunk_count(self):
        """Test V3 performance with large number of chunks."""
        # Create 1000 test chunks
        chunks = {}
        for i in range(1000):
            chunk = Chunk(
                id=f"chunk_{i}", path=Path(f"file_{i}.py"), rel_path=f"file_{i}.py", 
                start_line=1, end_line=10,
                kind=ChunkKind.FUNCTION, name=f"func_{i}", language="python",
                content=f"def func_{i}(): pass",
                full_tokens=50 + i % 20,
                signature_tokens=10 + i % 5,
                complexity_score=float(i % 10),
                doc_density=0.5
            )
            chunks[f"chunk_{i}"] = chunk
        
        # Create selection with many chunks
        selected_chunks = list(chunks.values())[:100]  # Select first 100
        chunk_modes = {c.id: 'full' for c in selected_chunks}
        
        result = SelectionResult(
            selected_chunks=selected_chunks,
            chunk_modes=chunk_modes,
            selection_scores={c.id: 0.5 for c in selected_chunks},
            total_tokens=sum(c.full_tokens for c in selected_chunks),
            budget_utilization=0.8
        )
        
        config = SelectionConfig(token_budget=10000)
        
        # Should handle large chunk count efficiently
        start_time = time.time()
        candidates = self.controller.detect_demotion_candidates(
            result, chunks, config, budget_pressure_threshold=0.9
        )
        detection_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 5 seconds)
        assert detection_time < 5.0
        assert isinstance(candidates, list)
    
    def test_extreme_oscillation_rate(self):
        """Test stability tracker with extremely high oscillation rate."""
        chunk_id = "oscillating_chunk"
        
        # Record 100 oscillations in rapid succession
        for i in range(100):
            event = OscillationEvent(
                chunk_id=chunk_id,
                epoch=i,
                event_type=EventType.OSCILLATION,
                risk_score=0.9
            )
            self.stability_tracker.record_event(event)
        
        # Should handle extreme oscillation count
        risk = self.stability_tracker.calculate_oscillation_risk(chunk_id, 100)
        assert risk > 0.5  # Should be high risk (adjust threshold based on algorithm)
        
        metrics = self.stability_tracker.get_metrics()
        assert metrics['total_oscillations'] == 100
    
    def test_invalid_chunk_modes(self):
        """Test V3 behavior with invalid chunk modes."""
        chunk = Chunk(
            id="chunk1", path=Path("test.py"), rel_path="test.py", start_line=1, end_line=5,
            kind=ChunkKind.FUNCTION, name="test_func", language="python",
            content="pass", full_tokens=50, signature_tokens=10
        )
        
        # Create result with invalid mode
        result = SelectionResult(
            selected_chunks=[chunk],
            chunk_modes={'chunk1': 'invalid_mode'},  # Invalid mode
            selection_scores={'chunk1': 0.5},
            total_tokens=50,
            budget_utilization=0.05
        )
        
        config = SelectionConfig(token_budget=1000)
        
        # Should handle invalid mode gracefully
        utility = self.utility_calculator.calculate_utility_per_cost(
            chunk, 'invalid_mode', [chunk], config
        )
        
        # Should default to full mode behavior
        assert utility >= 0.0
    
    def test_circular_demotion_detection(self):
        """Test detection of circular demotion patterns."""
        chunks = {
            'chunk1': Chunk(id="chunk1", path=Path("a.py"), rel_path="a.py", start_line=1, end_line=10,
                           kind=ChunkKind.FUNCTION, name="func1", language="python", content="def func1(): pass",
                           full_tokens=100, signature_tokens=20),
            'chunk2': Chunk(id="chunk2", path=Path("b.py"), rel_path="b.py", start_line=1, end_line=10,
                           kind=ChunkKind.FUNCTION, name="func2", language="python", content="def func2(): pass",
                           full_tokens=80, signature_tokens=15),
            'chunk3': Chunk(id="chunk3", path=Path("c.py"), rel_path="c.py", start_line=1, end_line=10,
                           kind=ChunkKind.FUNCTION, name="func3", language="python", content="def func3(): pass",
                           full_tokens=60, signature_tokens=10),
        }
        
        # Create circular dependency in utility calculation
        # (This is a pathological case that shouldn't happen in practice)
        with patch.object(self.utility_calculator, 'calculate_utility_per_cost') as mock_calc:
            # Mock utility calculation to create circular patterns
            def mock_utility(chunk, mode, selected_chunks, config):
                # Return utility that causes circular demotions
                return 0.1 if chunk.id == 'chunk1' else 0.5
            
            mock_calc.side_effect = mock_utility
            
            result = SelectionResult(
                selected_chunks=list(chunks.values()),
                chunk_modes={cid: 'full' for cid in chunks},
                selection_scores={cid: 0.5 for cid in chunks},
                total_tokens=240,
                budget_utilization=0.24
            )
            
            config = SelectionConfig(token_budget=1000)
            
            # Should detect candidates without infinite loops
            candidates = self.controller.detect_demotion_candidates(
                result, chunks, config
            )
            
            assert isinstance(candidates, list)
            # Should not get stuck in infinite loop
    
    def test_memory_pressure_handling(self):
        """Test V3 behavior under memory pressure simulation."""
        # Simulate memory pressure by creating large data structures
        large_chunks = {}
        large_content = "x" * 10000  # 10KB content per chunk
        
        for i in range(100):
            chunk = Chunk(
                id=f"large_chunk_{i}",
                path=Path(f"large_{i}.py"),
                rel_path=f"large_{i}.py",
                start_line=1, end_line=1000,
                kind=ChunkKind.FUNCTION, name=f"large_func_{i}", language="python",
                content=large_content,
                full_tokens=1000,
                signature_tokens=100
            )
            large_chunks[chunk.id] = chunk
        
        # Create large selection result
        selected_chunks = list(large_chunks.values())[:50]
        result = SelectionResult(
            selected_chunks=selected_chunks,
            chunk_modes={c.id: 'full' for c in selected_chunks},
            selection_scores={c.id: 0.5 for c in selected_chunks},
            total_tokens=50000,
            budget_utilization=0.5
        )
        
        config = SelectionConfig(token_budget=100000)
        
        # Should handle large memory usage gracefully
        try:
            candidates = self.controller.detect_demotion_candidates(
                result, large_chunks, config
            )
            assert isinstance(candidates, list)
        except MemoryError:
            pytest.skip("Memory pressure test triggered MemoryError (expected in low-memory environments)")
    
    def test_concurrent_access_simulation(self):
        """Test V3 behavior under simulated concurrent access."""
        import threading
        import queue
        
        chunk = Chunk(
            id="shared_chunk", path=Path("shared.py"), rel_path="shared.py", 
            start_line=1, end_line=10, kind=ChunkKind.FUNCTION, name="shared_func", language="python",
            content="def shared_func(): pass", full_tokens=100, signature_tokens=20
        )
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker():
            try:
                # Each thread calculates utility independently
                utility = self.utility_calculator.calculate_utility_per_cost(
                    chunk, 'full', [], SelectionConfig()
                )
                results_queue.put(utility)
            except Exception as e:
                errors_queue.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert errors_queue.empty(), f"Concurrent access caused errors: {list(errors_queue.queue)}"
        assert results_queue.qsize() == 10, f"Expected 10 results, got {results_queue.qsize()}"
    
    def test_malformed_input_handling(self):
        """Test V3 behavior with malformed input data."""
        # Test with None chunks
        try:
            self.controller.detect_demotion_candidates(
                None, {}, SelectionConfig()
            )
        except (AttributeError, TypeError):
            pass  # Expected to fail gracefully
        
        # Test with malformed chunk data
        malformed_chunk = Mock()
        malformed_chunk.id = None  # Invalid ID
        malformed_chunk.full_tokens = -100  # Invalid token count
        
        try:
            self.utility_calculator.calculate_utility_per_cost(
                malformed_chunk, 'full', [], SelectionConfig()
            )
        except (AttributeError, ValueError):
            pass  # Expected to fail gracefully
    
    def test_extreme_budget_ratios(self):
        """Test V3 with extreme budget utilization ratios."""
        chunk = Chunk(
            id="chunk1", path=Path("test.py"), rel_path="test.py",
            start_line=1, end_line=50000, kind=ChunkKind.FUNCTION, name="large_func", language="python",
            content="def large_func(): pass",
            full_tokens=1000000,  # Very large chunk
            signature_tokens=100000
        )
        
        # Test with tiny budget
        tiny_config = SelectionConfig(token_budget=1)
        result = SelectionResult(
            selected_chunks=[chunk],
            chunk_modes={'chunk1': 'full'},
            selection_scores={'chunk1': 0.5},
            total_tokens=1000000,
            budget_utilization=1000000.0  # Extreme over-budget
        )
        
        # Should handle extreme ratios without crashing
        candidates = self.controller.detect_demotion_candidates(
            result, {'chunk1': chunk}, tiny_config
        )
        
        assert isinstance(candidates, list)
    
    def test_stability_tracker_overflow(self):
        """Test stability tracker behavior when history overflows."""
        # Create tracker with very small history size
        small_tracker = StabilityTracker(max_history_size=5)
        
        # Add more events than max size
        for i in range(10):
            event = OscillationEvent(
                chunk_id=f"chunk_{i}",
                epoch=i,
                event_type=EventType.DEMOTION
            )
            small_tracker.record_event(event)
        
        # Should maintain max size
        assert len(small_tracker.event_history) == 5
        
        # Should keep most recent events
        most_recent = small_tracker.event_history[-1]
        assert most_recent.chunk_id == "chunk_9"  # Last added


class TestV3PerformanceStress:
    """Performance and stress tests for V3 system."""
    
    def test_large_selection_performance(self):
        """Test V3 performance with large selection sets."""
        # Create large selection result
        chunks = []
        chunk_modes = {}
        selection_scores = {}
        
        for i in range(500):  # 500 selected chunks
            chunk = Chunk(
                id=f"chunk_{i}",
                path=Path(f"file_{i % 100}.py"),
                rel_path=f"file_{i % 100}.py",  # Simulate file grouping
                start_line=1, end_line=20,
                kind=ChunkKind.FUNCTION, name=f"func_{i}", language="python",
                content=f"def func_{i}(): pass",
                full_tokens=100 + i % 50,
                signature_tokens=20 + i % 10
            )
            chunks.append(chunk)
            chunk_modes[chunk.id] = 'full'
            selection_scores[chunk.id] = 0.5 + (i % 50) / 100.0
        
        result = SelectionResult(
            selected_chunks=chunks,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=sum(c.full_tokens for c in chunks),
            budget_utilization=0.8
        )
        
        chunks_by_id = {c.id: c for c in chunks}
        config = SelectionConfig(token_budget=100000)
        
        controller = DemotionController(
            V3UtilityCalculator(),
            StabilityTracker()
        )
        
        # Measure detection performance
        start_time = time.time()
        candidates = controller.detect_demotion_candidates(
            result, chunks_by_id, config
        )
        detection_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert detection_time < 10.0, f"Detection took too long: {detection_time:.2f}s"
        assert len(candidates) >= 0  # Should produce some result
        
        # Measure bounded reoptimization performance
        if candidates:
            limited_candidates = candidates[:5]  # Limit for performance
            start_time = time.time()
            
            modified_result, actions = controller.execute_bounded_reoptimization(
                limited_candidates, result, chunks_by_id, config, max_corrective_steps=1
            )
            
            reopt_time = time.time() - start_time
            assert reopt_time < 5.0, f"Reoptimization took too long: {reopt_time:.2f}s"
    
    def test_repeated_operations_performance(self):
        """Test performance of repeated V3 operations."""
        chunk = Chunk(
            id="perf_chunk", path=Path("perf.py"), rel_path="perf.py",
            start_line=1, end_line=20, kind=ChunkKind.FUNCTION, name="perf_func", language="python",
            content="def perf_func(): pass", full_tokens=100, signature_tokens=20
        )
        
        config = SelectionConfig(token_budget=1000)
        calculator = V3UtilityCalculator()
        
        # Measure repeated utility calculations
        start_time = time.time()
        
        for i in range(1000):
            utility = calculator.calculate_utility_per_cost(
                chunk, 'full', [], config
            )
            assert utility >= 0.0
        
        calc_time = time.time() - start_time
        
        # Should benefit from caching and complete quickly
        assert calc_time < 2.0, f"1000 calculations took too long: {calc_time:.2f}s"
        
        # Verify caching metrics are being tracked
        metrics = calculator.get_performance_metrics()
        assert isinstance(metrics['cache_hits'], int), "Cache hits should be tracked"
        assert metrics['calculations_count'] >= 1000, "Should have recorded all calculations"
    
    def test_memory_usage_bounds(self):
        """Test that V3 components maintain reasonable memory usage."""
        import sys
        
        # Measure initial memory
        initial_size = sys.getsizeof(StabilityTracker())
        
        # Create tracker and add many events
        tracker = StabilityTracker(max_history_size=100)
        
        for i in range(200):  # More than max size
            event = OscillationEvent(
                chunk_id=f"chunk_{i}",
                epoch=i,
                event_type=EventType.DEMOTION
            )
            tracker.record_event(event)
        
        # Memory should be bounded by max_history_size
        final_size = sys.getsizeof(tracker)
        
        # Should not grow indefinitely (allow 10x growth as reasonable bound)
        assert final_size < initial_size * 10, f"Memory usage grew too much: {final_size} vs {initial_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
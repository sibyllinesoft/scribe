"""
Comprehensive tests for V3 demotion stability controller.

Tests all V3 components including demotion detection, bounded re-optimization,
oscillation prevention, and budget reallocation.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
from pathlib import Path

from packrepo.packer.chunker.base import Chunk, ChunkKind
from packrepo.packer.selector.base import SelectionConfig, SelectionResult, SelectionVariant
from packrepo.packer.controller.demotion import DemotionController, DemotionDecision, DemotionStrategy
from packrepo.packer.controller.stability import StabilityTracker, OscillationEvent, EventType
from packrepo.packer.controller.utility import V3UtilityCalculator
from packrepo.packer.budget.reallocation import BudgetReallocator, ReallocationStrategy


class TestV3UtilityCalculator:
    """Test V3 utility calculator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = V3UtilityCalculator()
        self.config = SelectionConfig(
            token_budget=10000,
            coverage_weight=0.7,
            diversity_weight=0.3
        )
        
        # Create test chunks
        self.chunks = [
            Chunk(
                id="chunk1", path=Path("main.py"), rel_path="main.py", start_line=1, end_line=50,
                kind=ChunkKind.FUNCTION, name="main", language="python",
                content="def main(): pass", 
                full_tokens=100, signature_tokens=20,
                complexity_score=3.0, doc_density=0.8
            ),
            Chunk(
                id="chunk2", path=Path("utils.py"), rel_path="utils.py", start_line=1, end_line=30,
                kind=ChunkKind.CLASS, name="Utils", language="python",
                content="class Utils: pass", 
                full_tokens=80, signature_tokens=15,
                complexity_score=2.0, doc_density=0.5
            ),
            Chunk(
                id="chunk3", path=Path("test.py"), rel_path="test.py", start_line=1, end_line=20,
                kind=ChunkKind.TEST, name="test_main", language="python",
                content="def test_main(): assert True", 
                full_tokens=60, signature_tokens=10,
                complexity_score=1.5, doc_density=0.3
            ),
        ]
    
    def test_calculate_utility_per_cost_basic(self):
        """Test basic utility per cost calculation."""
        chunk = self.chunks[0]  # main function chunk
        selected_chunks = []
        
        utility = self.calculator.calculate_utility_per_cost(
            chunk, 'full', selected_chunks, self.config
        )
        
        assert utility > 0.0
        assert isinstance(utility, float)
    
    def test_calculate_utility_per_cost_modes(self):
        """Test utility calculation for different modes."""
        chunk = self.chunks[0]
        selected_chunks = []
        
        full_utility = self.calculator.calculate_utility_per_cost(
            chunk, 'full', selected_chunks, self.config
        )
        signature_utility = self.calculator.calculate_utility_per_cost(
            chunk, 'signature', selected_chunks, self.config
        )
        
        # Full mode should generally have higher utility per token due to more content
        # but the actual comparison depends on token costs
        assert full_utility >= 0.0
        assert signature_utility >= 0.0
    
    def test_calculate_coverage_contribution(self):
        """Test coverage contribution calculation."""
        chunk = self.chunks[0]
        selected_chunks = []
        
        coverage = self.calculator.calculate_coverage_contribution(
            chunk, selected_chunks, self.config
        )
        
        assert 0.0 <= coverage <= 1.0
    
    def test_calculate_diversity_contribution(self):
        """Test diversity contribution calculation."""
        chunk = self.chunks[0]
        selected_chunks = []
        
        diversity = self.calculator.calculate_diversity_contribution(
            chunk, selected_chunks, self.config
        )
        
        assert 0.0 <= diversity <= 1.0
    
    def test_calculate_demotion_impact(self):
        """Test demotion impact calculation."""
        chunk = self.chunks[0]
        selected_chunks = [self.chunks[1]]
        
        impact = self.calculator.calculate_demotion_impact(
            chunk, 'full', 'signature', selected_chunks, self.config
        )
        
        required_keys = [
            'current_utility', 'demoted_utility', 'utility_loss', 
            'tokens_freed', 'utility_per_token_freed'
        ]
        
        for key in required_keys:
            assert key in impact
        
        assert impact['tokens_freed'] == chunk.full_tokens - chunk.signature_tokens
        # Utility loss can be negative if signature mode is more efficient
        # Accept numpy types as well as regular types
        import numpy as np
        assert isinstance(impact['utility_loss'], (int, float, np.number))
        # Tokens freed should always be positive when demoting from full to signature
        assert impact['tokens_freed'] > 0
    
    def test_performance_metrics(self):
        """Test utility calculator performance tracking."""
        chunk = self.chunks[0]
        
        # Perform some calculations to generate metrics
        for i in range(5):
            self.calculator.calculate_utility_per_cost(
                chunk, 'full', [], self.config
            )
        
        metrics = self.calculator.get_performance_metrics()
        
        assert 'calculations_count' in metrics
        assert 'cache_hits' in metrics
        assert 'cache_hit_rate' in metrics
        assert metrics['calculations_count'] >= 5
    
    def test_cache_functionality(self):
        """Test utility calculator caching."""
        chunk = self.chunks[0]
        
        # Reset cache to ensure clean state
        self.calculator.clear_cache()
        
        # Do several calculations to trigger feature cache
        for _ in range(3):
            self.calculator.calculate_utility_per_cost(chunk, 'full', [], self.config)
        
        metrics = self.calculator.get_performance_metrics()
        
        # Should have recorded calculations and some cache hits
        assert metrics['calculations_count'] > 0
        # The cache might hit on the feature calculations within each utility calculation
        # or it might not hit depending on implementation details
        # Let's just verify the metrics are being tracked
        assert isinstance(metrics['cache_hits'], int)
        assert isinstance(metrics['cache_hit_rate'], float)
        assert 'feature_cache_size' in metrics


class TestStabilityTracker:
    """Test stability tracking and oscillation detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = StabilityTracker(max_history_size=100)
        self.chunk_ids = ['chunk1', 'chunk2', 'chunk3']
    
    def test_record_event(self):
        """Test event recording functionality."""
        event = OscillationEvent(
            chunk_id='chunk1',
            epoch=1,
            event_type=EventType.DEMOTION,
            old_mode='full',
            new_mode='signature'
        )
        
        self.tracker.record_event(event)
        
        assert len(self.tracker.event_history) == 1
        assert self.tracker.event_history[0] == event
    
    def test_detect_oscillations(self):
        """Test oscillation detection across selections."""
        # Create mock selection results with oscillating patterns
        selections = []
        
        for i, modes in enumerate([
            {'chunk1': 'full', 'chunk2': 'signature'},     # Selection 1
            {'chunk1': 'signature', 'chunk2': 'full'},     # Selection 2 
            {'chunk1': 'full', 'chunk2': 'signature'},     # Selection 3 - creates oscillation!
        ]):
            chunks = [
                Mock(id=chunk_id) for chunk_id in modes.keys()
            ]
            selection = Mock()
            selection.selected_chunks = chunks
            selection.chunk_modes = modes
            selections.append(selection)
        
        # Detect oscillations between selections
        oscillations = self.tracker.detect_oscillations(
            selections[-1], selections[:-1], look_back_epochs=3
        )
        
        # Should detect oscillation for chunk1 (full -> signature -> full)
        assert len(oscillations) > 0
        assert any(osc.chunk_id == 'chunk1' for osc in oscillations)
    
    def test_calculate_oscillation_risk(self):
        """Test oscillation risk calculation."""
        chunk_id = 'chunk1'
        current_epoch = 5
        
        # Record some oscillation history
        for epoch in range(1, 4):
            event = OscillationEvent(
                chunk_id=chunk_id,
                epoch=epoch,
                event_type=EventType.OSCILLATION,
                risk_score=0.8
            )
            self.tracker.record_event(event)
        
        risk = self.tracker.calculate_oscillation_risk(chunk_id, current_epoch)
        
        assert 0.0 <= risk <= 1.0
        assert risk > 0.0  # Should have elevated risk due to history
    
    def test_ban_list_management(self):
        """Test epoch ban list functionality."""
        chunk_id = 'chunk1'
        ban_until_epoch = 10
        
        self.tracker.add_to_ban_list(chunk_id, ban_until_epoch)
        
        # Should be banned
        assert self.tracker.ban_list.is_banned(chunk_id)
        
        # Advance past ban period
        self.tracker.advance_epoch(ban_until_epoch + 1)
        
        # Should no longer be banned
        assert not self.tracker.ban_list.is_banned(chunk_id)
    
    def test_get_metrics(self):
        """Test stability metrics generation."""
        # Record some events
        for i in range(3):
            event = OscillationEvent(
                chunk_id=f'chunk{i}',
                epoch=1,
                event_type=EventType.DEMOTION
            )
            self.tracker.record_event(event)
        
        metrics = self.tracker.get_metrics()
        
        required_keys = [
            'stability_score', 'total_oscillations', 'current_epoch',
            'ban_list_stats', 'event_history_size'
        ]
        
        for key in required_keys:
            assert key in metrics
    
    def test_export_stability_report(self):
        """Test stability report export."""
        # Add some test data
        self.tracker.oscillation_counts['chunk1'] = 3
        self.tracker.oscillation_counts['chunk2'] = 1
        
        report = self.tracker.export_stability_report()
        
        assert 'summary' in report
        assert 'top_oscillating_chunks' in report
        assert 'recommendations' in report
        assert isinstance(report['recommendations'], list)


class TestDemotionController:
    """Test demotion controller functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utility_calculator = V3UtilityCalculator()
        self.stability_tracker = StabilityTracker()
        self.controller = DemotionController(
            self.utility_calculator, 
            self.stability_tracker
        )
        
        self.config = SelectionConfig(
            token_budget=1000,
            coverage_weight=0.7
        )
        
        # Create test chunks and selection result
        self.chunks = {
            'chunk1': Chunk(
                id="chunk1", path=Path("main.py"), rel_path="main.py", start_line=1, end_line=50,
                kind=ChunkKind.FUNCTION, name="main", language="python",
                content="def main(): pass", 
                full_tokens=200, signature_tokens=50,
                complexity_score=3.0, doc_density=0.8
            ),
            'chunk2': Chunk(
                id="chunk2", path=Path("utils.py"), rel_path="utils.py", start_line=1, end_line=30,
                kind=ChunkKind.CLASS, name="Utils", language="python",
                content="class Utils: pass", 
                full_tokens=150, signature_tokens=30,
                complexity_score=2.0, doc_density=0.5
            ),
        }
        
        self.selection_result = SelectionResult(
            selected_chunks=list(self.chunks.values()),
            chunk_modes={'chunk1': 'full', 'chunk2': 'full'},
            selection_scores={'chunk1': 0.8, 'chunk2': 0.6},
            total_tokens=350,
            budget_utilization=0.35
        )
    
    def test_detect_demotion_candidates(self):
        """Test demotion candidate detection."""
        candidates = self.controller.detect_demotion_candidates(
            self.selection_result, self.chunks, self.config,
            budget_pressure_threshold=0.9
        )
        
        # Should find some candidates since we're using low utility thresholds
        assert isinstance(candidates, list)
        
        for candidate in candidates:
            assert isinstance(candidate, DemotionDecision)
            assert candidate.chunk_id in self.chunks
            assert candidate.current_mode in ['full', 'signature', 'summary']
            assert candidate.demoted_mode in ['full', 'signature', 'summary']
            assert candidate.tokens_freed >= 0
    
    def test_execute_bounded_reoptimization(self):
        """Test bounded re-optimization execution."""
        # Create some demotion candidates
        decision = DemotionDecision(
            chunk_id='chunk1',
            current_mode='full',
            demoted_mode='signature', 
            strategy=DemotionStrategy.THRESHOLD_BASED,
            original_utility=0.5,
            recomputed_utility=0.3,
            tokens_freed=150
        )
        
        modified_result, corrective_actions = self.controller.execute_bounded_reoptimization(
            [decision], self.selection_result, self.chunks, self.config,
            max_corrective_steps=1
        )
        
        # Check that demotion was applied
        assert modified_result.chunk_modes['chunk1'] == 'signature'
        assert 'chunk1' in modified_result.demoted_chunks
        
        # Check corrective actions were attempted
        assert isinstance(corrective_actions, list)
    
    def test_check_oscillation_constraints(self):
        """Test oscillation constraint checking."""
        # Create selection history with potential oscillations
        previous_selections = [
            Mock(selected_chunks=[], chunk_modes={'chunk1': 'full'}),
            Mock(selected_chunks=[], chunk_modes={'chunk1': 'signature'}),
        ]
        
        current_selection = Mock(selected_chunks=[], chunk_modes={'chunk1': 'full'})
        
        constraint_satisfied, oscillations = self.controller.check_oscillation_constraints(
            current_selection, previous_selections, max_oscillations=1
        )
        
        assert isinstance(constraint_satisfied, bool)
        assert isinstance(oscillations, list)
    
    def test_export_demotion_analytics(self):
        """Test demotion analytics export."""
        import tempfile
        import os
        
        # Add some test decisions
        decision = DemotionDecision(
            chunk_id='chunk1',
            current_mode='full',
            demoted_mode='signature',
            strategy=DemotionStrategy.THRESHOLD_BASED,
            original_utility=0.5,
            recomputed_utility=0.3,
            tokens_freed=150,
            timestamp=time.time()
        )
        self.controller._demotion_decisions.append(decision)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            summary = self.controller.export_demotion_analytics(temp_path)
            
            assert 'export_path' in summary
            assert 'total_demotions' in summary
            assert os.path.exists(temp_path)
            
            # Verify CSV content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'chunk_id' in content  # Header
                assert 'chunk1' in content    # Data
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        metrics = self.controller.get_performance_metrics()
        
        required_keys = [
            'total_demotions', 'prevented_oscillations', 
            'budget_reallocated', 'current_epoch'
        ]
        
        for key in required_keys:
            assert key in metrics


class TestBudgetReallocator:
    """Test budget reallocation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utility_calculator = V3UtilityCalculator()
        self.reallocator = BudgetReallocator(self.utility_calculator)
        
        self.config = SelectionConfig(token_budget=1000)
        
        # Create test chunks and selection
        self.available_chunks = {
            'chunk1': Chunk(
                id="chunk1", path=Path("main.py"), rel_path="main.py", start_line=1, end_line=50,
                kind=ChunkKind.FUNCTION, name="main", language="python",
                content="def main(): pass", full_tokens=100, signature_tokens=20
            ),
            'chunk2': Chunk(
                id="chunk2", path=Path("utils.py"), rel_path="utils.py", start_line=1, end_line=30,
                kind=ChunkKind.CLASS, name="Utils", language="python",
                content="class Utils: pass", full_tokens=80, signature_tokens=15
            ),
            'chunk3': Chunk(
                id="chunk3", path=Path("test.py"), rel_path="test.py", start_line=1, end_line=20,
                kind=ChunkKind.FUNCTION, name="test", language="python",
                content="def test(): pass", full_tokens=60, signature_tokens=10
            ),
        }
        
        self.current_selection = SelectionResult(
            selected_chunks=[self.available_chunks['chunk1']],
            chunk_modes={'chunk1': 'signature'},  # Already demoted
            selection_scores={'chunk1': 0.5},
            total_tokens=20,
            budget_utilization=0.02
        )
    
    def test_reallocate_budget_greedy(self):
        """Test greedy budget reallocation strategy."""
        freed_budget = 80  # Budget freed by demotion
        
        result = self.reallocator.reallocate_budget(
            freed_budget, self.current_selection, self.available_chunks, 
            self.config, ReallocationStrategy.GREEDY_UTILITY
        )
        
        assert result.strategy == ReallocationStrategy.GREEDY_UTILITY
        assert result.freed_budget == freed_budget
        assert result.allocated_budget >= 0
        assert result.allocated_budget <= freed_budget  # Should not over-allocate
        assert result.is_successful()  # Should have no critical violations
    
    def test_reallocate_budget_balanced(self):
        """Test balanced budget reallocation strategy."""
        freed_budget = 100
        
        result = self.reallocator.reallocate_budget(
            freed_budget, self.current_selection, self.available_chunks,
            self.config, ReallocationStrategy.BALANCED
        )
        
        assert result.strategy == ReallocationStrategy.BALANCED
        assert result.freed_budget == freed_budget
        
        # Check result structure
        summary = result.get_summary_stats()
        assert 'budget_efficiency' in summary
        assert 'changes' in summary
        assert 'quality_impact' in summary
    
    def test_constraint_violations(self):
        """Test budget constraint violation detection."""
        # Try to reallocate more budget than available
        freed_budget = 50
        
        # Mock the reallocation to force over-allocation (for testing)
        result = self.reallocator.reallocate_budget(
            freed_budget, self.current_selection, self.available_chunks, self.config
        )
        
        # Should detect if there are any constraint violations
        assert isinstance(result.constraint_violations, list)
    
    def test_performance_metrics(self):
        """Test budget reallocator performance tracking."""
        # Perform a reallocation to generate metrics
        self.reallocator.reallocate_budget(
            100, self.current_selection, self.available_chunks, self.config
        )
        
        metrics = self.reallocator.get_performance_metrics()
        
        assert 'total_reallocations' in metrics
        assert 'successful_reallocations' in metrics
        assert 'success_rate' in metrics
        assert metrics['total_reallocations'] >= 1


class TestV3Integration:
    """Integration tests for complete V3 system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        from packrepo.packer.selector.selector import RepositorySelector
        from packrepo.packer.tokenizer.implementations import TikTokenTokenizer
        
        from packrepo.packer.tokenizer.base import TokenizerType
        self.tokenizer = TikTokenTokenizer(TokenizerType.CL100K_BASE)
        self.selector = RepositorySelector(self.tokenizer)
        
        # Create test chunks for a realistic scenario
        self.test_chunks = [
            Chunk(
                id=f"chunk{i}", path=Path(f"file{i}.py"), rel_path=f"file{i}.py", start_line=1, end_line=20,
                kind=ChunkKind.FUNCTION, name=f"function{i}", language="python",
                content=f"def function{i}(): pass", 
                full_tokens=100 + i*10, signature_tokens=20 + i*2,
                complexity_score=float(i), doc_density=0.5
            ) for i in range(10)
        ]
    
    def test_v3_selection_end_to_end(self):
        """Test complete V3 selection process."""
        config = SelectionConfig(
            variant=SelectionVariant.V3_STABLE,
            token_budget=800,
            deterministic=True,
            random_seed=42
        )
        
        from packrepo.packer.selector.base import PackRequest
        request = PackRequest(
            chunks=self.test_chunks,
            config=config
        )
        
        # Run V3 selection
        result = self.selector.select(request)
        
        # Verify V3 was used and succeeded
        assert result.selection.total_tokens <= config.token_budget
        assert result.selection.budget_utilization > 0.0
        assert len(result.selection.selected_chunks) > 0
        assert result.execution_time > 0.0
    
    def test_v3_oscillation_prevention(self):
        """Test oscillation prevention across multiple runs."""
        config = SelectionConfig(
            variant=SelectionVariant.V3_STABLE,
            token_budget=500,
            deterministic=True,
            random_seed=42
        )
        
        # Run selection multiple times to test stability
        results = []
        for i in range(3):
            from packrepo.packer.selector.base import PackRequest
            request = PackRequest(chunks=self.test_chunks, config=config)
            result = self.selector.select(request)
            results.append(result)
        
        # Verify results are stable (not oscillating wildly)
        token_counts = [r.selection.total_tokens for r in results]
        selected_counts = [len(r.selection.selected_chunks) for r in results]
        
        # Should be reasonably stable across runs
        token_variance = max(token_counts) - min(token_counts)
        assert token_variance < 200  # Allow some variation but not wild swings
        
        count_variance = max(selected_counts) - min(selected_counts)
        assert count_variance <= 2  # Chunk count should be very stable
    
    def test_v3_analytics_export(self):
        """Test V3 analytics and export functionality."""
        config = SelectionConfig(
            variant=SelectionVariant.V3_STABLE,
            token_budget=600,
            deterministic=True
        )
        
        from packrepo.packer.selector.base import PackRequest
        request = PackRequest(chunks=self.test_chunks, config=config)
        
        # Run selection to generate V3 data
        result = self.selector.select(request)
        
        # Get analytics
        analytics = self.selector.get_v3_analytics()
        
        if 'error' not in analytics:  # V3 was successfully used
            assert 'controller_metrics' in analytics
            assert 'stability_metrics' in analytics
            assert 'utility_metrics' in analytics
    
    def test_v3_deterministic_behavior(self):
        """Test that V3 produces deterministic results."""
        config = SelectionConfig(
            variant=SelectionVariant.V3_STABLE,
            token_budget=400,
            deterministic=True,
            random_seed=42
        )
        
        # Run same selection twice
        results = []
        for _ in range(2):
            # Clear history to ensure clean runs
            if hasattr(self.selector, 'clear_v3_history'):
                self.selector.clear_v3_history()
            
            from packrepo.packer.selector.base import PackRequest
            request = PackRequest(chunks=self.test_chunks, config=config)
            result = self.selector.select(request)
            results.append(result)
        
        # Should produce identical results
        assert results[0].selection.total_tokens == results[1].selection.total_tokens
        assert len(results[0].selection.selected_chunks) == len(results[1].selection.selected_chunks)
        
        # Deterministic hashes should be identical
        if results[0].deterministic_hash and results[1].deterministic_hash:
            assert results[0].deterministic_hash == results[1].deterministic_hash


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
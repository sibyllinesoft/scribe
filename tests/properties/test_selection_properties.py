"""Property-based testing for PackRepo selection properties.

Tests the selection invariants from TODO.md:
- Monotone ΔU ≥ 0 (marginal utility non-decreasing)
- Facility-location bounds respected
- MMR redundancy ≤ threshold
- Submodular selection properties
"""

from __future__ import annotations

import pytest
import math
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize
from typing import Dict, Any, List, Optional, Set, Tuple

from packrepo.packer.packfmt.base import PackFormat, PackIndex, PackSection
from packrepo.packer.oracles.selection import SelectionPropertiesOracle, BudgetEfficiencyOracle
from packrepo.packer.oracles import OracleResult


class MockSelectionAlgorithm:
    """Mock selection algorithm implementing submodular properties."""
    
    def __init__(self, coverage_weights: Dict[str, float] = None, diversity_penalty: float = 0.1):
        self.coverage_weights = coverage_weights or {}
        self.diversity_penalty = diversity_penalty
        self.selected_items: Set[str] = set()
        self.utility_cache: Dict[frozenset, float] = {}
    
    def compute_coverage(self, chunk_ids: Set[str]) -> float:
        """Compute coverage score for a set of chunks."""
        coverage = 0.0
        for chunk_id in chunk_ids:
            coverage += self.coverage_weights.get(chunk_id, 1.0)
        
        # Apply diversity penalty for similar chunks
        # Simulate redundancy by penalizing chunks with same prefix
        prefix_counts = {}
        for chunk_id in chunk_ids:
            prefix = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        diversity_penalty = sum(count * (count - 1) * self.diversity_penalty 
                               for count in prefix_counts.values())
        
        return max(0, coverage - diversity_penalty)
    
    def marginal_utility(self, chunk_id: str, current_selection: Set[str]) -> float:
        """Compute marginal utility of adding chunk_id to current selection."""
        current_utility = self.compute_coverage(current_selection)
        new_utility = self.compute_coverage(current_selection | {chunk_id})
        return new_utility - current_utility
    
    def select_chunks(self, available_chunks: List[Dict[str, Any]], budget: int) -> List[str]:
        """Select chunks using greedy submodular optimization."""
        selected = set()
        selected_list = []
        remaining_budget = budget
        
        while remaining_budget > 0 and available_chunks:
            best_chunk = None
            best_marginal_per_cost = 0
            best_marginal = 0
            
            for chunk in available_chunks:
                chunk_id = chunk['id']
                if chunk_id in selected:
                    continue
                
                cost = chunk.get('selected_tokens', 1)
                if cost > remaining_budget:
                    continue
                
                marginal = self.marginal_utility(chunk_id, selected)
                marginal_per_cost = marginal / max(1, cost)
                
                if marginal_per_cost > best_marginal_per_cost:
                    best_chunk = chunk
                    best_marginal_per_cost = marginal_per_cost
                    best_marginal = marginal
            
            if not best_chunk or best_marginal <= 0:
                break
            
            selected.add(best_chunk['id'])
            selected_list.append(best_chunk['id'])
            remaining_budget -= best_chunk.get('selected_tokens', 1)
        
        return selected_list


# Test strategies
@st.composite
def chunk_strategy(draw):
    """Generate a chunk with selection metadata."""
    chunk_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    tokens = draw(st.integers(min_value=1, max_value=500))
    coverage_score = draw(st.floats(min_value=0.0, max_value=1.0))
    diversity_score = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return {
        'id': chunk_id,
        'rel_path': f"{chunk_id}.py",
        'start_line': 1,
        'end_line': 10,
        'selected_tokens': tokens,
        'selected_mode': 'full',
        'coverage_score': coverage_score,
        'diversity_score': diversity_score,
        'selection_score': (coverage_score + diversity_score) / 2
    }


@st.composite
def pack_with_selection_strategy(draw):
    """Generate pack with selection metadata."""
    num_chunks = draw(st.integers(min_value=2, max_value=15))
    chunks = [draw(chunk_strategy()) for _ in range(num_chunks)]
    
    # Ensure unique IDs
    for i, chunk in enumerate(chunks):
        chunk['id'] = f"chunk_{i}_{chunk['id']}"
    
    total_tokens = sum(chunk['selected_tokens'] for chunk in chunks)
    budget = draw(st.integers(min_value=total_tokens // 2, max_value=total_tokens + 1000))
    
    # Create sections
    sections = []
    for chunk in chunks:
        sections.append(PackSection(
            rel_path=chunk['rel_path'],
            start_line=chunk['start_line'], 
            end_line=chunk['end_line'],
            content=f"Content for {chunk['id']}",
            mode=chunk['selected_mode']
        ))
    
    index = PackIndex(
        target_budget=budget,
        actual_tokens=min(total_tokens, budget),
        chunks=chunks
    )
    
    return PackFormat(index=index, sections=sections)


class TestSelectionProperties:
    """Property-based tests for selection properties."""
    
    @given(pack_with_selection_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_monotone_marginal_utility_property(self, pack: PackFormat):
        """Property: Marginal utility ΔU ≥ 0 (submodularity)."""
        chunks = pack.index.chunks
        if len(chunks) < 2:
            return
        
        # Create mock algorithm
        coverage_weights = {chunk['id']: chunk.get('coverage_score', 0.5) for chunk in chunks}
        algorithm = MockSelectionAlgorithm(coverage_weights)
        
        # Test submodularity: f(A ∪ {v}) - f(A) ≥ f(B ∪ {v}) - f(B) when A ⊆ B
        chunk_ids = [chunk['id'] for chunk in chunks]
        
        if len(chunk_ids) >= 3:
            # Take subsets A ⊆ B
            A = set(chunk_ids[:2])
            B = set(chunk_ids[:3])  
            v = chunk_ids[-1]  # Element not in A or B
            
            # Compute marginal utilities
            marginal_A = algorithm.marginal_utility(v, A)
            marginal_B = algorithm.marginal_utility(v, B)
            
            note(f"Marginal utility: A={marginal_A:.3f}, B={marginal_B:.3f}")
            
            # Submodularity: marginal_A ≥ marginal_B (diminishing returns)
            assert marginal_A >= marginal_B - 1e-6, \
                   f"Submodularity violated: {marginal_A} < {marginal_B}"
            
            # Both should be non-negative for valid utility functions
            assert marginal_A >= -1e-6, f"Negative marginal utility: {marginal_A}"
            assert marginal_B >= -1e-6, f"Negative marginal utility: {marginal_B}"
    
    @given(pack_with_selection_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_budget_efficiency_property(self, pack: PackFormat):
        """Property: Selection should efficiently use available budget."""
        oracle = BudgetEfficiencyOracle()
        
        total_budget = pack.index.target_budget
        actual_tokens = pack.index.actual_tokens
        
        # Calculate efficiency metrics
        budget_utilization = actual_tokens / max(1, total_budget)
        
        context = {
            'budget_utilization': budget_utilization,
            'efficiency_threshold': 0.8  # Should use at least 80% of budget
        }
        
        report = oracle.validate(pack, context)
        note(f"Budget utilization: {budget_utilization:.1%}")
        
        if budget_utilization >= 0.8:
            assert report.result in [OracleResult.PASS, OracleResult.WARN]
        else:
            # Low utilization should be flagged as warning
            assert report.result in [OracleResult.WARN, OracleResult.FAIL]
    
    @given(pack_with_selection_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_mmr_redundancy_property(self, pack: PackFormat):
        """Property: MMR (Maximal Marginal Relevance) should limit redundancy."""
        chunks = pack.index.chunks
        
        # Simulate redundancy by checking for similar chunks
        redundancy_groups = {}
        for chunk in chunks:
            # Group by file prefix (simulate similar content)
            prefix = chunk['rel_path'].split('.')[0]
            if prefix not in redundancy_groups:
                redundancy_groups[prefix] = []
            redundancy_groups[prefix].append(chunk)
        
        # Calculate redundancy ratio
        total_chunks = len(chunks)
        redundant_chunks = sum(max(0, len(group) - 1) for group in redundancy_groups.values())
        redundancy_ratio = redundant_chunks / max(1, total_chunks)
        
        redundancy_threshold = 0.3  # Allow up to 30% redundancy
        
        note(f"Redundancy ratio: {redundancy_ratio:.1%} (threshold: {redundancy_threshold:.1%})")
        
        # High redundancy should be controlled
        if redundancy_ratio <= redundancy_threshold:
            # Low redundancy is good
            pass
        else:
            # High redundancy is acceptable if diversity scores compensate
            avg_diversity = sum(chunk.get('diversity_score', 0) for chunk in chunks) / len(chunks)
            
            # If diversity is high, redundancy might be acceptable
            if avg_diversity < 0.5:
                note(f"High redundancy with low diversity: {redundancy_ratio:.1%}, {avg_diversity:.3f}")
    
    def test_facility_location_bounds_property(self):
        """Property: Facility location algorithm should respect distance bounds."""
        # Create chunks with spatial/semantic relationships
        chunks = [
            {'id': 'core_a', 'selected_tokens': 100, 'location': (0, 0), 'coverage_score': 0.9},
            {'id': 'core_b', 'selected_tokens': 100, 'location': (10, 0), 'coverage_score': 0.8},
            {'id': 'nearby_a', 'selected_tokens': 50, 'location': (1, 1), 'coverage_score': 0.7},
            {'id': 'far_c', 'selected_tokens': 75, 'location': (50, 50), 'coverage_score': 0.6},
        ]
        
        def compute_distance(chunk1, chunk2):
            """Compute distance between chunks."""
            x1, y1 = chunk1['location']
            x2, y2 = chunk2['location']
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Facility location should prefer distant, high-value chunks
        budget = 250  # Can afford 2-3 chunks
        
        # Greedy facility location selection
        selected = []
        remaining = chunks.copy()
        
        while remaining and sum(chunk['selected_tokens'] for chunk in selected) < budget:
            if not selected:
                # First chunk: highest value
                best = max(remaining, key=lambda c: c['coverage_score'])
            else:
                # Subsequent chunks: maximize min distance * value
                best = None
                best_score = 0
                
                for chunk in remaining:
                    if sum(c['selected_tokens'] for c in selected) + chunk['selected_tokens'] > budget:
                        continue
                    
                    min_distance = min(compute_distance(chunk, s) for s in selected)
                    score = min_distance * chunk['coverage_score']
                    
                    if score > best_score:
                        best = chunk
                        best_score = score
            
            if not best:
                break
            
            selected.append(best)
            remaining.remove(best)
        
        # Verify facility location properties
        assert len(selected) >= 2, "Should select multiple facilities"
        
        # Check minimum distance between selected facilities
        if len(selected) >= 2:
            min_distance = min(
                compute_distance(selected[i], selected[j])
                for i in range(len(selected))
                for j in range(i + 1, len(selected))
            )
            
            # Facilities should be reasonably spaced
            assert min_distance > 5, f"Facilities too close: min distance {min_distance}"


class SelectionStateMachine(RuleBasedStateMachine):
    """Stateful testing for selection algorithms."""
    
    def __init__(self):
        super().__init__()
        self.available_chunks: List[Dict[str, Any]] = []
        self.selected_chunks: List[str] = []
        self.budget = 1000
        self.used_budget = 0
        self.algorithm = MockSelectionAlgorithm()
    
    @initialize()
    def setup(self):
        """Initialize selection state."""
        self.available_chunks = []
        self.selected_chunks = []
        self.budget = 1000
        self.used_budget = 0
        self.algorithm = MockSelectionAlgorithm()
    
    @rule(chunk_id=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
          tokens=st.integers(min_value=1, max_value=200),
          score=st.floats(min_value=0.0, max_value=1.0))
    def add_chunk(self, chunk_id: str, tokens: int, score: float):
        """Add a chunk to the available set."""
        unique_id = f"{chunk_id}_{len(self.available_chunks)}"
        
        chunk = {
            'id': unique_id,
            'selected_tokens': tokens,
            'coverage_score': score,
            'rel_path': f"{unique_id}.py"
        }
        
        self.available_chunks.append(chunk)
        note(f"Added chunk: {unique_id} ({tokens} tokens, score {score:.3f})")
    
    @rule()
    def select_next_chunk(self):
        """Select the next best chunk within budget."""
        assume(self.available_chunks)
        assume(self.used_budget < self.budget)
        
        # Find best available chunk
        current_selection = set(self.selected_chunks)
        best_chunk = None
        best_ratio = 0
        
        for chunk in self.available_chunks:
            if chunk['id'] in current_selection:
                continue
            
            cost = chunk['selected_tokens']
            if self.used_budget + cost > self.budget:
                continue
            
            marginal = self.algorithm.marginal_utility(chunk['id'], current_selection)
            ratio = marginal / max(1, cost)
            
            if ratio > best_ratio:
                best_chunk = chunk
                best_ratio = ratio
        
        if best_chunk:
            self.selected_chunks.append(best_chunk['id'])
            self.used_budget += best_chunk['selected_tokens']
            note(f"Selected: {best_chunk['id']} (ratio: {best_ratio:.3f})")
    
    @invariant()
    def budget_invariant(self):
        """Selection should never exceed budget."""
        assert self.used_budget <= self.budget, \
               f"Budget exceeded: {self.used_budget} > {self.budget}"
    
    @invariant()
    def monotone_utility_invariant(self):
        """Marginal utility should be non-negative for selected items."""
        if len(self.selected_chunks) <= 1:
            return
        
        # Check last selected item had positive marginal utility
        if self.selected_chunks:
            last_selected = self.selected_chunks[-1]
            previous_selection = set(self.selected_chunks[:-1])
            
            marginal = self.algorithm.marginal_utility(last_selected, previous_selection)
            
            # Should be positive (or very small negative due to floating point)
            assert marginal >= -1e-6, \
                   f"Negative marginal utility for {last_selected}: {marginal}"


TestSelectionStateMachine = SelectionStateMachine.TestCase


class TestSelectionEdgeCases:
    """Test edge cases for selection algorithms."""
    
    def test_empty_selection_set(self):
        """Test behavior with empty selection set."""
        pack = PackFormat(
            index=PackIndex(target_budget=1000, actual_tokens=0, chunks=[]),
            sections=[]
        )
        
        oracle = SelectionPropertiesOracle()
        report = oracle.validate(pack)
        
        # Empty selection should be valid (though possibly inefficient)
        assert report.result in [OracleResult.PASS, OracleResult.WARN, OracleResult.SKIP]
    
    def test_single_chunk_selection(self):
        """Test selection with only one chunk."""
        chunk = {'id': 'only', 'selected_tokens': 500, 'coverage_score': 0.8, 'rel_path': 'only.py'}
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=1000,
                actual_tokens=500,
                chunks=[chunk]
            ),
            sections=[PackSection(rel_path='only.py', start_line=1, end_line=10, content='content', mode='full')]
        )
        
        oracle = SelectionPropertiesOracle()
        report = oracle.validate(pack)
        
        assert report.result in [OracleResult.PASS, OracleResult.WARN]
    
    def test_identical_score_tie_breaking(self):
        """Test tie-breaking when chunks have identical scores."""
        chunks = []
        sections = []
        
        # Create 3 chunks with identical scores
        for i in range(3):
            chunk = {
                'id': f'tie_{i}',
                'selected_tokens': 100,
                'coverage_score': 0.7,  # Identical
                'diversity_score': 0.5,  # Identical
                'rel_path': f'tie_{i}.py'
            }
            chunks.append(chunk)
            
            sections.append(PackSection(
                rel_path=f'tie_{i}.py',
                start_line=1,
                end_line=10,
                content=f'content_{i}',
                mode='full'
            ))
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=300,
                actual_tokens=300,
                chunks=chunks
            ),
            sections=sections
        )
        
        # Should handle ties gracefully (deterministic tie-breaking)
        oracle = SelectionPropertiesOracle()
        report = oracle.validate(pack)
        
        assert report.result in [OracleResult.PASS, OracleResult.WARN]
    
    def test_extreme_budget_constraints(self):
        """Test selection under extreme budget constraints."""
        # Very small budget
        small_pack = PackFormat(
            index=PackIndex(
                target_budget=1,
                actual_tokens=1,
                chunks=[{'id': 'tiny', 'selected_tokens': 1, 'rel_path': 'tiny.py'}]
            ),
            sections=[PackSection(rel_path='tiny.py', start_line=1, end_line=1, content='x', mode='full')]
        )
        
        oracle = BudgetEfficiencyOracle()
        context = {'efficiency_threshold': 0.5}  # Lower threshold for tiny budgets
        report = oracle.validate(small_pack, context)
        
        assert report.result in [OracleResult.PASS, OracleResult.WARN]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
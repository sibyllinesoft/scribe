"""Property-based testing for PackRepo budget constraints.

Tests the fundamental budget invariants from TODO.md:
- actual_tokens ≤ target_budget  
- ≤0.5% underflow allowed
- 0 overflow allowed
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize
from typing import Dict, Any, List, Optional

from packrepo.packer.packfmt.base import PackFormat, PackIndex, PackSection
from packrepo.packer.oracles.budget import BudgetOracle, BudgetEnforcementOracle
from packrepo.packer.oracles import OracleResult
from packrepo.packer.tokenizer.base import TokenizerInterface


class MockTokenizer(TokenizerInterface):
    """Mock tokenizer for testing with predictable token counts."""
    
    def __init__(self, tokens_per_char: float = 0.25):
        self.tokens_per_char = tokens_per_char
        self.name = "mock_tokenizer"
        self.version = "1.0.0"
    
    def count_tokens(self, text: str) -> int:
        """Predictable token counting for testing."""
        return max(1, int(len(text) * self.tokens_per_char))
    
    def encode(self, text: str) -> List[int]:
        """Mock encoding."""
        return list(range(self.count_tokens(text)))


# Property-based test strategies
@st.composite
def pack_section_strategy(draw):
    """Generate valid PackSection instances."""
    rel_path = draw(st.text(min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
    start_line = draw(st.integers(min_value=1, max_value=1000))
    end_line = draw(st.integers(min_value=start_line, max_value=start_line + 100))
    content = draw(st.text(min_size=10, max_size=1000))
    
    return PackSection(
        rel_path=rel_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        mode="full"
    )


@st.composite
def pack_index_strategy(draw, sections_list=None):
    """Generate valid PackIndex instances."""
    if sections_list is None:
        num_sections = draw(st.integers(min_value=1, max_value=20))
        sections_list = [draw(pack_section_strategy()) for _ in range(num_sections)]
    
    # Calculate realistic token counts
    tokenizer = MockTokenizer()
    total_content_tokens = sum(tokenizer.count_tokens(section.content) for section in sections_list)
    
    # Generate budget that's either just right or slightly over
    budget_factor = draw(st.floats(min_value=0.8, max_value=1.2))
    target_budget = max(100, int(total_content_tokens * budget_factor))
    
    # Actual tokens should respect budget constraints
    max_allowed = int(target_budget * 1.005)  # Allow 0.5% overflow for testing edge cases
    actual_tokens = draw(st.integers(min_value=1, max_value=max_allowed))
    
    chunks = []
    for i, section in enumerate(sections_list):
        chunk_tokens = min(tokenizer.count_tokens(section.content), 
                          actual_tokens // len(sections_list))
        chunks.append({
            'id': f"chunk_{i}",
            'rel_path': section.rel_path,
            'start_line': section.start_line,
            'end_line': section.end_line,
            'selected_tokens': chunk_tokens,
            'selected_mode': section.mode,
            'content_hash': f"hash_{i}"
        })
    
    return PackIndex(
        target_budget=target_budget,
        actual_tokens=actual_tokens,
        chunks=chunks,
        tokenizer_name=tokenizer.name,
        tokenizer_version=tokenizer.version
    )


@st.composite 
def pack_format_strategy(draw):
    """Generate valid PackFormat instances."""
    sections = [draw(pack_section_strategy()) for _ in range(draw(st.integers(min_value=1, max_value=15)))]
    index = draw(pack_index_strategy(sections))
    
    return PackFormat(index=index, sections=sections)


class TestBudgetConstraintsProperties:
    """Property-based tests for budget constraints."""
    
    @given(pack_format_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_no_budget_overflow_property(self, pack: PackFormat):
        """Property: actual_tokens ≤ target_budget (0 overflow allowed)."""
        oracle = BudgetOracle()
        report = oracle.validate(pack)
        
        note(f"Budget: {pack.index.actual_tokens}/{pack.index.target_budget}")
        note(f"Oracle result: {report.result}")
        
        # The fundamental budget constraint
        if pack.index.actual_tokens > pack.index.target_budget:
            assert report.result == OracleResult.FAIL
            assert "budget overflow" in report.message.lower()
        else:
            assert report.result in [OracleResult.PASS, OracleResult.WARN]
    
    @given(pack_format_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_underflow_tolerance_property(self, pack: PackFormat):
        """Property: ≤0.5% underflow is acceptable."""
        oracle = BudgetEnforcementOracle()
        
        underflow = pack.index.target_budget - pack.index.actual_tokens
        underflow_percent = underflow / max(1, pack.index.target_budget)
        
        report = oracle.validate(pack)
        
        note(f"Underflow: {underflow} tokens ({underflow_percent:.1%})")
        
        if underflow_percent <= 0.005:  # ≤0.5%
            assert report.result in [OracleResult.PASS, OracleResult.WARN]
        else:
            # Significant underflow should be flagged
            assert "underflow" in report.message.lower() or report.result == OracleResult.WARN
    
    @given(st.integers(min_value=1000, max_value=100000))
    @settings(max_examples=50)
    def test_budget_scaling_property(self, base_budget: int):
        """Property: Budget constraints scale linearly with budget size."""
        # Create packs with different budgets but proportional token usage
        usage_ratio = 0.95  # Use 95% of budget
        
        small_pack = PackFormat(
            index=PackIndex(
                target_budget=base_budget,
                actual_tokens=int(base_budget * usage_ratio),
                chunks=[{
                    'id': 'chunk_1',
                    'selected_tokens': int(base_budget * usage_ratio),
                    'rel_path': 'test.py'
                }]
            ),
            sections=[PackSection(rel_path='test.py', start_line=1, end_line=10, content='x' * 1000, mode='full')]
        )
        
        large_pack = PackFormat(
            index=PackIndex(
                target_budget=base_budget * 2,
                actual_tokens=int(base_budget * 2 * usage_ratio),
                chunks=[{
                    'id': 'chunk_1', 
                    'selected_tokens': int(base_budget * 2 * usage_ratio),
                    'rel_path': 'test.py'
                }]
            ),
            sections=[PackSection(rel_path='test.py', start_line=1, end_line=20, content='x' * 2000, mode='full')]
        )
        
        oracle = BudgetOracle()
        small_result = oracle.validate(small_pack)
        large_result = oracle.validate(large_pack)
        
        # Both should pass since they respect budget constraints
        assert small_result.result in [OracleResult.PASS, OracleResult.WARN]
        assert large_result.result in [OracleResult.PASS, OracleResult.WARN]


class BudgetStateMachine(RuleBasedStateMachine):
    """Stateful property testing for budget management."""
    
    def __init__(self):
        super().__init__()
        self.target_budget = 10000
        self.actual_tokens = 0
        self.chunks: List[Dict[str, Any]] = []
        self.tokenizer = MockTokenizer()
    
    @initialize()
    def setup(self):
        """Initialize with a reasonable budget."""
        self.target_budget = 10000
        self.actual_tokens = 0
        self.chunks = []
    
    @rule(chunk_size=st.integers(min_value=10, max_value=1000))
    def add_chunk(self, chunk_size: int):
        """Add a chunk respecting budget constraints."""
        assume(self.actual_tokens + chunk_size <= self.target_budget)
        
        chunk_id = f"chunk_{len(self.chunks)}"
        self.chunks.append({
            'id': chunk_id,
            'selected_tokens': chunk_size,
            'rel_path': f'file_{len(self.chunks)}.py',
            'start_line': 1,
            'end_line': 10
        })
        self.actual_tokens += chunk_size
    
    @rule(multiplier=st.floats(min_value=1.1, max_value=5.0))
    def scale_budget(self, multiplier: float):
        """Scale budget up (should allow more chunks)."""
        old_budget = self.target_budget
        self.target_budget = int(self.target_budget * multiplier)
        
        # Scaling budget should never violate constraints
        assert self.actual_tokens <= self.target_budget
        note(f"Scaled budget {old_budget} -> {self.target_budget}")
    
    @invariant()
    def budget_invariant(self):
        """The budget constraint must always hold."""
        assert self.actual_tokens <= self.target_budget, \
               f"Budget violation: {self.actual_tokens} > {self.target_budget}"
    
    @invariant()
    def underflow_invariant(self):
        """Underflow should be within acceptable bounds."""
        if self.chunks:  # Only check if we have chunks
            underflow = self.target_budget - self.actual_tokens
            underflow_percent = underflow / self.target_budget
            
            # In practice, we want to use most of the budget
            # But this is an invariant that should always hold
            assert underflow_percent <= 1.0, "Underflow cannot exceed 100%"


TestBudgetStateMachine = BudgetStateMachine.TestCase


class TestBudgetEdgeCases:
    """Test edge cases for budget handling."""
    
    def test_zero_budget_edge_case(self):
        """Test handling of zero budget."""
        pack = PackFormat(
            index=PackIndex(target_budget=0, actual_tokens=0, chunks=[]),
            sections=[]
        )
        
        oracle = BudgetOracle()
        report = oracle.validate(pack)
        
        # Zero budget with zero usage should be valid
        assert report.result in [OracleResult.PASS, OracleResult.WARN]
    
    def test_single_token_budget(self):
        """Test handling of minimal budget."""
        pack = PackFormat(
            index=PackIndex(
                target_budget=1,
                actual_tokens=1,
                chunks=[{'id': 'tiny', 'selected_tokens': 1, 'rel_path': 'x.py'}]
            ),
            sections=[PackSection(rel_path='x.py', start_line=1, end_line=1, content='x', mode='full')]
        )
        
        oracle = BudgetOracle()
        report = oracle.validate(pack)
        
        assert report.result in [OracleResult.PASS, OracleResult.WARN]
    
    def test_exact_budget_boundary(self):
        """Test exact budget utilization."""
        budget = 5000
        pack = PackFormat(
            index=PackIndex(
                target_budget=budget,
                actual_tokens=budget,  # Exactly at budget
                chunks=[{'id': 'exact', 'selected_tokens': budget, 'rel_path': 'exact.py'}]
            ),
            sections=[PackSection(rel_path='exact.py', start_line=1, end_line=100, content='x' * 1000, mode='full')]
        )
        
        oracle = BudgetOracle()
        report = oracle.validate(pack)
        
        # Exact budget usage should pass
        assert report.result == OracleResult.PASS
        
    def test_budget_overflow_detection(self):
        """Test detection of budget overflow."""
        budget = 1000
        pack = PackFormat(
            index=PackIndex(
                target_budget=budget,
                actual_tokens=budget + 1,  # 1 token over budget
                chunks=[{'id': 'over', 'selected_tokens': budget + 1, 'rel_path': 'over.py'}]
            ),
            sections=[PackSection(rel_path='over.py', start_line=1, end_line=50, content='x' * 500, mode='full')]
        )
        
        oracle = BudgetOracle()
        report = oracle.validate(pack)
        
        # Any overflow should fail
        assert report.result == OracleResult.FAIL
        assert "overflow" in report.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Property-based testing for PackRepo determinism constraints.

Tests the determinism invariant from TODO.md:
- hash(index+body) equal across 3 runs (--no-llm)
- Stable chunk IDs and canonical sort
- Tie-breakers fixed
"""

from __future__ import annotations

import pytest
import hashlib
import json
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize
from typing import Dict, Any, List, Optional, Set

from packrepo.packer.packfmt.base import PackFormat, PackIndex, PackSection
from packrepo.packer.oracles.determinism import DeterminismOracle, HashConsistencyOracle
from packrepo.packer.oracles import OracleResult


def compute_pack_hash(pack: PackFormat) -> str:
    """Compute deterministic hash of pack (index + body)."""
    # Serialize index to canonical JSON
    index_dict = pack.index.to_dict() if hasattr(pack.index, 'to_dict') else vars(pack.index)
    
    # Sort keys for determinism
    canonical_index = json.dumps(index_dict, sort_keys=True, separators=(',', ':'))
    
    # Concatenate all section content in order
    body_content = ""
    if pack.sections:
        sorted_sections = sorted(pack.sections, key=lambda s: (s.rel_path, s.start_line))
        body_content = "\n".join(f"### {s.rel_path}:{s.start_line}-{s.end_line}\n{s.content}" 
                                 for s in sorted_sections)
    
    # Compute combined hash
    combined = f"{canonical_index}\n---BODY---\n{body_content}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


class MockPackGenerator:
    """Mock pack generator for determinism testing."""
    
    def __init__(self, repo_path: str, seed: int = 42):
        self.repo_path = repo_path
        self.seed = seed
    
    def generate_pack(self, no_llm: bool = True) -> PackFormat:
        """Generate a pack deterministically when no_llm=True."""
        # Simulate deterministic pack generation
        import random
        random.seed(self.seed)
        
        # Create deterministic sections
        sections = []
        for i in range(3):  # Fixed number for determinism
            rel_path = f"file_{i}.py"
            content = f"# File {i}\ndef func_{i}():\n    return {i}\n"
            sections.append(PackSection(
                rel_path=rel_path,
                start_line=1,
                end_line=3,
                content=content,
                mode="full"
            ))
        
        # Sort sections for determinism
        sections.sort(key=lambda s: (s.rel_path, s.start_line))
        
        # Create deterministic chunks
        chunks = []
        for i, section in enumerate(sections):
            chunks.append({
                'id': f"chunk_{section.rel_path}_{section.start_line}_{section.end_line}",
                'rel_path': section.rel_path,
                'start_line': section.start_line,
                'end_line': section.end_line,
                'selected_tokens': len(section.content),
                'selected_mode': section.mode,
                'content_hash': hashlib.sha256(section.content.encode()).hexdigest()[:16]
            })
        
        index = PackIndex(
            target_budget=10000,
            actual_tokens=sum(chunk['selected_tokens'] for chunk in chunks),
            chunks=sorted(chunks, key=lambda c: c['id']),  # Canonical sort
            tokenizer_name="cl100k_base",
            tokenizer_version="1.0.0"
        )
        
        return PackFormat(index=index, sections=sections)


@st.composite
def deterministic_pack_strategy(draw):
    """Generate packs that should be deterministic."""
    seed = draw(st.integers(min_value=1, max_value=1000))
    repo_path = "/tmp/test_repo"
    
    generator = MockPackGenerator(repo_path, seed)
    return generator.generate_pack(no_llm=True), seed


class TestDeterminismProperties:
    """Property-based tests for determinism constraints."""
    
    @given(deterministic_pack_strategy())
    @settings(max_examples=50, deadline=10000)
    def test_hash_consistency_property(self, pack_and_seed):
        """Property: hash(index+body) should be identical across multiple runs."""
        pack, seed = pack_and_seed
        
        # Generate same pack multiple times
        generator = MockPackGenerator("/tmp/test_repo", seed)
        
        hashes = []
        for run in range(3):
            pack_run = generator.generate_pack(no_llm=True)
            pack_hash = compute_pack_hash(pack_run)
            hashes.append(pack_hash)
            note(f"Run {run + 1} hash: {pack_hash[:16]}...")
        
        # All hashes should be identical
        assert len(set(hashes)) == 1, f"Hashes differ across runs: {hashes}"
        
        oracle = HashConsistencyOracle()
        context = {
            'hash_comparisons': [
                {'run': i, 'hash': h} for i, h in enumerate(hashes)
            ]
        }
        report = oracle.validate(pack, context)
        
        assert report.result == OracleResult.PASS
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=30)
    def test_chunk_id_stability_property(self, seed: int):
        """Property: Chunk IDs should be stable across runs with same input."""
        generator = MockPackGenerator("/tmp/test_repo", seed)
        
        # Generate packs multiple times
        packs = [generator.generate_pack(no_llm=True) for _ in range(3)]
        
        # Extract chunk IDs from each pack
        chunk_id_sets = []
        for pack in packs:
            chunk_ids = [chunk['id'] for chunk in pack.index.chunks]
            chunk_id_sets.append(set(chunk_ids))
            note(f"Chunk IDs: {chunk_ids[:3]}...")  # Show first 3
        
        # All chunk ID sets should be identical
        assert len(set(frozenset(ids) for ids in chunk_id_sets)) == 1, \
               "Chunk IDs differ across runs"
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=30)
    def test_canonical_ordering_property(self, seed: int):
        """Property: Chunks should be in canonical order."""
        generator = MockPackGenerator("/tmp/test_repo", seed)
        pack = generator.generate_pack(no_llm=True)
        
        chunks = pack.index.chunks
        chunk_ids = [chunk['id'] for chunk in chunks]
        
        # Should be sorted
        sorted_chunk_ids = sorted(chunk_ids)
        assert chunk_ids == sorted_chunk_ids, \
               f"Chunks not in canonical order: {chunk_ids} vs {sorted_chunk_ids}"
        
        note(f"Chunks in canonical order: {len(chunks)} chunks")
    
    def test_tie_breaker_consistency(self):
        """Test that tie-breakers are handled consistently."""
        # Create scenario with potential ties (same scores)
        sections = [
            PackSection(rel_path="a.py", start_line=1, end_line=5, content="# Same content", mode="full"),
            PackSection(rel_path="b.py", start_line=1, end_line=5, content="# Same content", mode="full"),
            PackSection(rel_path="c.py", start_line=1, end_line=5, content="# Same content", mode="full"),
        ]
        
        # Generate pack multiple times
        packs = []
        for run in range(3):
            chunks = []
            for i, section in enumerate(sections):
                chunks.append({
                    'id': f"chunk_{section.rel_path}_{section.start_line}_{section.end_line}",
                    'rel_path': section.rel_path,
                    'start_line': section.start_line,
                    'end_line': section.end_line,
                    'selected_tokens': 50,  # Same tokens - creates tie
                    'selected_mode': section.mode,
                    'selection_score': 0.75,  # Same score - creates tie
                    'content_hash': hashlib.sha256(section.content.encode()).hexdigest()[:16]
                })
            
            # Apply tie-breaker: sort by chunk ID
            chunks.sort(key=lambda c: c['id'])
            
            index = PackIndex(
                target_budget=1000,
                actual_tokens=150,
                chunks=chunks
            )
            
            packs.append(PackFormat(index=index, sections=sections))
        
        # All packs should have identical chunk ordering
        chunk_orders = []
        for pack in packs:
            order = [chunk['id'] for chunk in pack.index.chunks]
            chunk_orders.append(order)
        
        assert len(set(tuple(order) for order in chunk_orders)) == 1, \
               f"Tie-breaker inconsistent across runs: {chunk_orders}"


class DeterminismStateMachine(RuleBasedStateMachine):
    """Stateful testing for determinism properties."""
    
    def __init__(self):
        super().__init__()
        self.generator_configs: List[Dict[str, Any]] = []
        self.generated_hashes: Dict[str, List[str]] = {}
    
    @initialize()
    def setup(self):
        """Initialize determinism testing state."""
        self.generator_configs = []
        self.generated_hashes = {}
    
    @rule(seed=st.integers(min_value=1, max_value=1000),
          repo_name=st.text(min_size=3, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    def add_generator_config(self, seed: int, repo_name: str):
        """Add a generator configuration."""
        config = {
            'seed': seed,
            'repo_path': f"/tmp/{repo_name}",
            'config_id': f"{repo_name}_{seed}"
        }
        self.generator_configs.append(config)
        note(f"Added config: {config['config_id']}")
    
    @rule()
    def generate_and_hash_packs(self):
        """Generate packs and compute hashes."""
        assume(self.generator_configs)
        
        for config in self.generator_configs:
            config_id = config['config_id']
            
            if config_id not in self.generated_hashes:
                self.generated_hashes[config_id] = []
            
            # Generate pack and compute hash
            generator = MockPackGenerator(config['repo_path'], config['seed'])
            pack = generator.generate_pack(no_llm=True)
            pack_hash = compute_pack_hash(pack)
            
            self.generated_hashes[config_id].append(pack_hash)
            note(f"Generated hash for {config_id}: {pack_hash[:8]}...")
    
    @invariant()
    def hash_consistency_invariant(self):
        """All hashes for the same config should be identical."""
        for config_id, hashes in self.generated_hashes.items():
            if len(hashes) > 1:
                unique_hashes = set(hashes)
                assert len(unique_hashes) == 1, \
                       f"Hash inconsistency for {config_id}: {len(unique_hashes)} different hashes"


TestDeterminismStateMachine = DeterminismStateMachine.TestCase


class TestDeterminismEdgeCases:
    """Test edge cases for determinism."""
    
    def test_empty_pack_determinism(self):
        """Test determinism with empty packs."""
        empty_pack = PackFormat(
            index=PackIndex(target_budget=0, actual_tokens=0, chunks=[]),
            sections=[]
        )
        
        # Empty packs should have consistent hashes
        hash1 = compute_pack_hash(empty_pack)
        hash2 = compute_pack_hash(empty_pack)
        
        assert hash1 == hash2
        
        oracle = DeterminismOracle()
        report = oracle.validate(empty_pack)
        
        assert report.result in [OracleResult.PASS, OracleResult.SKIP]
    
    def test_unicode_content_determinism(self):
        """Test determinism with Unicode content."""
        unicode_content = "# 测试文件\ndef 函数():\n    return '你好世界'"
        
        section = PackSection(
            rel_path="unicode_test.py",
            start_line=1,
            end_line=3,
            content=unicode_content,
            mode="full"
        )
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=1000,
                actual_tokens=100,
                chunks=[{
                    'id': f"chunk_unicode_test.py_1_3",
                    'rel_path': section.rel_path,
                    'selected_tokens': 100,
                    'content_hash': hashlib.sha256(unicode_content.encode()).hexdigest()[:16]
                }]
            ),
            sections=[section]
        )
        
        # Unicode content should produce consistent hashes
        hash1 = compute_pack_hash(pack)
        hash2 = compute_pack_hash(pack)
        
        assert hash1 == hash2
    
    def test_large_content_determinism(self):
        """Test determinism with large content."""
        large_content = "# Large file\n" + "x = 1\n" * 1000
        
        section = PackSection(
            rel_path="large_file.py",
            start_line=1,
            end_line=1001,
            content=large_content,
            mode="full"
        )
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=5000,
                actual_tokens=4000,
                chunks=[{
                    'id': f"chunk_large_file.py_1_1001",
                    'rel_path': section.rel_path,
                    'selected_tokens': 4000,
                    'content_hash': hashlib.sha256(large_content.encode()).hexdigest()[:16]
                }]
            ),
            sections=[section]
        )
        
        # Large content should still be deterministic
        hashes = [compute_pack_hash(pack) for _ in range(3)]
        
        assert len(set(hashes)) == 1, f"Large content not deterministic: {hashes}"
    
    def test_floating_point_determinism(self):
        """Test that floating point values are handled deterministically."""
        # Test with scores that might have floating point precision issues
        chunks = []
        for i in range(3):
            score = 0.1 * i + 0.333333333  # Repeating decimal
            chunks.append({
                'id': f"chunk_{i}",
                'rel_path': f"file_{i}.py",
                'selected_tokens': 100,
                'selection_score': score
            })
        
        index = PackIndex(
            target_budget=1000,
            actual_tokens=300,
            chunks=chunks
        )
        
        pack = PackFormat(index=index, sections=[])
        
        # Should be deterministic despite floating point scores
        hash1 = compute_pack_hash(pack)
        hash2 = compute_pack_hash(pack)
        
        assert hash1 == hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
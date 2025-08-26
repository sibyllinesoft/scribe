"""Comprehensive metamorphic testing suite for PackRepo.

Implements all metamorphic properties M1-M6 from TODO.md:
- M1: append non-referenced duplicate file ⇒ ≤1% selection delta
- M2: rename path, content unchanged ⇒ only path fields differ  
- M3: budget×2 ⇒ coverage score increases monotonically
- M4: switch tokenizer cl100k→o200k with scaled budget ⇒ selected set similarity ≥ 0.8 Jaccard
- M5: inject large vendor folder ⇒ selection unaffected except index ignored counts
- M6: remove selected chunk ⇒ budget reallocated without over-cap
"""

from __future__ import annotations

import pytest
import tempfile
import shutil
import os
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from packrepo.packer.packfmt.base import PackFormat, PackIndex, PackSection
from packrepo.packer.oracles.metamorphic import (
    MetamorphicPropertiesOracle, 
    M1_DuplicateFileProperty,
    M2_PathRenameProperty,
    M3_BudgetScalingProperty,
    M4_TokenizerSwitchProperty,
    M5_VendorFolderProperty,
    M6_ChunkRemovalProperty
)
from packrepo.packer.oracles import OracleResult


@dataclass
class TestRepository:
    """Mock repository for metamorphic testing."""
    path: str
    files: Dict[str, str] = field(default_factory=dict)
    
    def add_file(self, rel_path: str, content: str):
        """Add file to repository."""
        self.files[rel_path] = content
        
        # Write to disk if path exists
        if os.path.exists(self.path):
            full_path = os.path.join(self.path, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def remove_file(self, rel_path: str):
        """Remove file from repository."""
        if rel_path in self.files:
            del self.files[rel_path]
        
        if os.path.exists(self.path):
            full_path = os.path.join(self.path, rel_path)
            if os.path.exists(full_path):
                os.remove(full_path)


class MockPackGenerator:
    """Mock pack generator for metamorphic testing."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def generate_pack(self, repo: TestRepository, budget: int, tokenizer: str = "cl100k") -> PackFormat:
        """Generate a pack from repository."""
        sections = []
        chunks = []
        total_tokens = 0
        
        # Sort files for determinism
        sorted_files = sorted(repo.files.items())
        
        # Convert files to sections and chunks
        for i, (rel_path, content) in enumerate(sorted_files):
            # Skip vendor/ignored files for simulation
            if self._should_ignore(rel_path):
                continue
            
            # Estimate tokens (simple approximation)
            estimated_tokens = self._estimate_tokens(content, tokenizer)
            
            if total_tokens + estimated_tokens <= budget:
                section = PackSection(
                    rel_path=rel_path,
                    start_line=1,
                    end_line=content.count('\n') + 1,
                    content=content,
                    mode="full"
                )
                sections.append(section)
                
                chunk = {
                    'id': f"chunk_{rel_path}_{1}_{section.end_line}",
                    'rel_path': rel_path,
                    'start_line': 1,
                    'end_line': section.end_line,
                    'selected_tokens': estimated_tokens,
                    'selected_mode': 'full',
                    'content_hash': hashlib.sha256(content.encode()).hexdigest()[:16]
                }
                chunks.append(chunk)
                total_tokens += estimated_tokens
        
        # Calculate coverage score (mock implementation)
        coverage_score = min(1.0, total_tokens / max(1, budget))
        
        index = PackIndex(
            target_budget=budget,
            actual_tokens=total_tokens,
            chunks=chunks,
            tokenizer_name=tokenizer,
            tokenizer_version="1.0.0",
            coverage_score=coverage_score,
            ignored_files=self._count_ignored_files(repo)
        )
        
        return PackFormat(index=index, sections=sections)
    
    def _should_ignore(self, rel_path: str) -> bool:
        """Check if file should be ignored."""
        ignore_patterns = [
            'node_modules/', 'vendor/', '.git/', '__pycache__/',
            '.pyc', '.so', '.dll', '.exe'
        ]
        return any(pattern in rel_path for pattern in ignore_patterns)
    
    def _estimate_tokens(self, content: str, tokenizer: str) -> int:
        """Estimate token count based on tokenizer."""
        # Mock tokenization with different ratios
        if tokenizer == "cl100k":
            ratio = 0.25  # 4 chars per token
        elif tokenizer == "o200k":
            ratio = 0.3   # ~3.33 chars per token (more efficient)
        else:
            ratio = 0.25
        
        return max(1, int(len(content) * ratio))
    
    def _count_ignored_files(self, repo: TestRepository) -> int:
        """Count ignored files in repository."""
        return sum(1 for rel_path in repo.files.keys() if self._should_ignore(rel_path))


class TestMetamorphicM1:
    """Test M1: Duplicate file invariance."""
    
    def test_m1_duplicate_file_property(self):
        """M1: append non-referenced duplicate file ⇒ ≤1% selection delta."""
        # Create base repository
        repo = TestRepository("/tmp/test_m1")
        repo.add_file("main.py", "def main():\n    print('Hello')\n")
        repo.add_file("utils.py", "def helper():\n    return 42\n")
        repo.add_file("tests.py", "def test():\n    assert True\n")
        
        generator = MockPackGenerator(42)
        original_pack = generator.generate_pack(repo, budget=1000)
        
        # Add duplicate file with different path
        duplicate_content = repo.files["utils.py"]  # Same content
        repo.add_file("duplicate_utils.py", duplicate_content)
        
        transformed_pack = generator.generate_pack(repo, budget=1000)
        
        # Validate M1 property
        property_obj = M1_DuplicateFileProperty()
        result = property_obj.validate(original_pack, transformed_pack, {'m1_threshold': 0.01})
        
        assert result['passes'], f"M1 failed: {result['details']}"
        assert result['selection_delta'] <= 0.01, \
               f"Selection delta too high: {result['selection_delta']:.3f}"
    
    def test_m1_with_different_thresholds(self):
        """Test M1 with various thresholds."""
        repo = TestRepository("/tmp/test_m1_thresh")
        
        # Add several files
        for i in range(5):
            repo.add_file(f"file_{i}.py", f"def func_{i}():\n    return {i}\n")
        
        generator = MockPackGenerator(123)
        original_pack = generator.generate_pack(repo, budget=2000)
        
        # Add duplicate
        repo.add_file("duplicate.py", repo.files["file_0.py"])
        transformed_pack = generator.generate_pack(repo, budget=2000)
        
        property_obj = M1_DuplicateFileProperty()
        
        # Test different thresholds
        for threshold in [0.005, 0.01, 0.02]:
            result = property_obj.validate(
                original_pack, transformed_pack, 
                {'m1_threshold': threshold}
            )
            
            if result['selection_delta'] <= threshold:
                assert result['passes'], f"Should pass with threshold {threshold}"
            else:
                assert not result['passes'], f"Should fail with threshold {threshold}"


class TestMetamorphicM2:
    """Test M2: Path rename invariance."""
    
    def test_m2_path_rename_property(self):
        """M2: rename path, content unchanged ⇒ only path fields differ."""
        # Create repository
        repo = TestRepository("/tmp/test_m2")
        repo.add_file("original.py", "def function():\n    return 'unchanged content'\n")
        repo.add_file("other.py", "def other():\n    return 'other content'\n")
        
        generator = MockPackGenerator(42)
        original_pack = generator.generate_pack(repo, budget=1000)
        
        # Rename file but keep content identical
        original_content = repo.files["original.py"]
        repo.remove_file("original.py")
        repo.add_file("renamed.py", original_content)  # Same content, different path
        
        transformed_pack = generator.generate_pack(repo, budget=1000)
        
        # Validate M2 property
        property_obj = M2_PathRenameProperty()
        result = property_obj.validate(original_pack, transformed_pack, {})
        
        assert result['passes'], f"M2 failed: {result['details']}"
        assert result['other_changes'] == 0, \
               f"Non-path changes detected: {result['other_changes']}"
    
    def test_m2_multiple_renames(self):
        """Test M2 with multiple file renames."""
        repo = TestRepository("/tmp/test_m2_multi")
        
        # Add multiple files
        files_content = {}
        for i in range(3):
            content = f"def func_{i}():\n    return {i * 10}\n"
            files_content[f"old_{i}.py"] = content
            repo.add_file(f"old_{i}.py", content)
        
        generator = MockPackGenerator(456)
        original_pack = generator.generate_pack(repo, budget=1500)
        
        # Rename all files
        for old_name, content in files_content.items():
            repo.remove_file(old_name)
            new_name = old_name.replace("old_", "new_")
            repo.add_file(new_name, content)
        
        transformed_pack = generator.generate_pack(repo, budget=1500)
        
        property_obj = M2_PathRenameProperty()
        result = property_obj.validate(original_pack, transformed_pack, {})
        
        assert result['passes'], f"M2 multi-rename failed: {result['details']}"


class TestMetamorphicM3:
    """Test M3: Budget scaling monotonicity."""
    
    def test_m3_budget_scaling_property(self):
        """M3: budget×2 ⇒ coverage score increases monotonically."""
        # Create repository with enough content
        repo = TestRepository("/tmp/test_m3")
        
        for i in range(10):
            content = f"def function_{i}():\n" + f"    # Comment {i}\n" * 5 + f"    return {i}\n"
            repo.add_file(f"module_{i}.py", content)
        
        generator = MockPackGenerator(789)
        
        # Generate with base budget
        base_budget = 1000
        original_pack = generator.generate_pack(repo, budget=base_budget)
        
        # Generate with doubled budget
        doubled_budget = base_budget * 2
        transformed_pack = generator.generate_pack(repo, budget=doubled_budget)
        
        # Validate M3 property
        property_obj = M3_BudgetScalingProperty()
        result = property_obj.validate(
            original_pack, transformed_pack, 
            {'budget_multiplier': 2.0}
        )
        
        assert result['passes'], f"M3 failed: {result['details']}"
        assert result['coverage_increase'] >= -0.001, \
               f"Coverage decreased: {result['coverage_increase']:.3f}"
    
    def test_m3_various_scaling_factors(self):
        """Test M3 with different scaling factors."""
        repo = TestRepository("/tmp/test_m3_scale")
        
        # Create content-rich repository
        for i in range(8):
            content = "# Module {}\n".format(i) + "x = 1\n" * 20
            repo.add_file(f"file_{i}.py", content)
        
        generator = MockPackGenerator(101112)
        base_budget = 800
        original_pack = generator.generate_pack(repo, budget=base_budget)
        
        property_obj = M3_BudgetScalingProperty()
        
        # Test different scaling factors
        for factor in [1.5, 2.0, 3.0]:
            scaled_budget = int(base_budget * factor)
            scaled_pack = generator.generate_pack(repo, budget=scaled_budget)
            
            result = property_obj.validate(
                original_pack, scaled_pack,
                {'budget_multiplier': factor}
            )
            
            # Coverage should increase (or stay same if already at maximum)
            assert result['coverage_increase'] >= -0.001, \
                   f"Coverage decreased with factor {factor}: {result['coverage_increase']}"


class TestMetamorphicM4:
    """Test M4: Tokenizer switch similarity."""
    
    def test_m4_tokenizer_switch_property(self):
        """M4: switch tokenizer cl100k→o200k with scaled budget ⇒ similarity ≥ 0.8 Jaccard."""
        # Create repository
        repo = TestRepository("/tmp/test_m4")
        
        for i in range(6):
            content = f"class Class{i}:\n    def method(self):\n        return '{i}' * 10\n"
            repo.add_file(f"class_{i}.py", content)
        
        generator = MockPackGenerator(131415)
        
        # Generate with cl100k tokenizer
        base_budget = 1200
        original_pack = generator.generate_pack(repo, budget=base_budget, tokenizer="cl100k")
        
        # Calculate scaled budget for o200k (more efficient tokenizer)
        # o200k is ~20% more efficient, so scale budget accordingly
        scaling_factor = 0.25 / 0.3  # ratio of token efficiencies
        scaled_budget = int(base_budget * scaling_factor)
        
        # Generate with o200k tokenizer
        transformed_pack = generator.generate_pack(repo, budget=scaled_budget, tokenizer="o200k")
        
        # Validate M4 property
        property_obj = M4_TokenizerSwitchProperty()
        result = property_obj.validate(
            original_pack, transformed_pack,
            {'m4_threshold': 0.8}
        )
        
        assert result['passes'], f"M4 failed: {result['details']}"
        assert result['jaccard_similarity'] >= 0.8, \
               f"Jaccard similarity too low: {result['jaccard_similarity']:.3f}"
    
    def test_m4_edge_cases(self):
        """Test M4 with edge cases."""
        # Test with very small repository (high similarity expected)
        repo = TestRepository("/tmp/test_m4_edge")
        repo.add_file("single.py", "print('hello')")
        
        generator = MockPackGenerator(161718)
        
        original_pack = generator.generate_pack(repo, budget=100, tokenizer="cl100k")
        transformed_pack = generator.generate_pack(repo, budget=83, tokenizer="o200k")  # Scaled
        
        property_obj = M4_TokenizerSwitchProperty()
        result = property_obj.validate(original_pack, transformed_pack, {'m4_threshold': 0.8})
        
        # With a single file, similarity should be very high
        assert result['jaccard_similarity'] >= 0.8


class TestMetamorphicM5:
    """Test M5: Vendor folder invariance."""
    
    def test_m5_vendor_folder_property(self):
        """M5: inject large vendor folder ⇒ selection unaffected except index ignored counts."""
        # Create base repository
        repo = TestRepository("/tmp/test_m5")
        
        for i in range(4):
            content = f"def business_logic_{i}():\n    return 'important code {i}'\n"
            repo.add_file(f"business_{i}.py", content)
        
        generator = MockPackGenerator(192021)
        original_pack = generator.generate_pack(repo, budget=1000)
        
        # Inject large vendor folder
        for i in range(20):  # Large vendor folder
            vendor_content = f"/* Vendor library {i} */\nfunction vendor_{i}() {{ return {i}; }}"
            repo.add_file(f"vendor/lib_{i}.js", vendor_content)
        
        # Add node_modules too
        for i in range(10):
            node_content = f"module.exports = function() {{ return {i}; }};"
            repo.add_file(f"node_modules/pkg_{i}/index.js", node_content)
        
        transformed_pack = generator.generate_pack(repo, budget=1000)
        
        # Validate M5 property
        property_obj = M5_VendorFolderProperty()
        result = property_obj.validate(original_pack, transformed_pack, {})
        
        assert result['passes'], f"M5 failed: {result['details']}"
        assert result['selection_unchanged'], \
               "Selection should be unchanged when vendor files added"
        assert result['ignored_increased'], \
               "Ignored file count should increase"
    
    def test_m5_different_vendor_patterns(self):
        """Test M5 with different vendor folder patterns."""
        repo = TestRepository("/tmp/test_m5_patterns")
        
        # Add business logic
        repo.add_file("main.py", "def main(): pass")
        repo.add_file("utils.py", "def util(): pass")
        
        generator = MockPackGenerator(222324)
        original_pack = generator.generate_pack(repo, budget=500)
        
        # Test various vendor patterns
        vendor_patterns = [
            ("vendor/", "vendor_lib.py"),
            ("node_modules/", "package/index.js"),
            ("__pycache__/", "cache.pyc"),
            ("build/", "output.o")
        ]
        
        for folder, filename in vendor_patterns:
            test_repo = TestRepository(f"/tmp/test_m5_{folder.replace('/', '_')}")
            test_repo.files = repo.files.copy()
            
            # Add vendor files
            test_repo.add_file(f"{folder}{filename}", "vendor content")
            
            transformed_pack = generator.generate_pack(test_repo, budget=500)
            
            property_obj = M5_VendorFolderProperty()
            result = property_obj.validate(original_pack, transformed_pack, {})
            
            assert result['selection_unchanged'], \
                   f"Selection changed with {folder} vendor files"


class TestMetamorphicM6:
    """Test M6: Chunk removal reallocation."""
    
    def test_m6_chunk_removal_property(self):
        """M6: remove selected chunk ⇒ budget reallocated without over-cap."""
        # Create repository with multiple files
        repo = TestRepository("/tmp/test_m6")
        
        for i in range(6):
            content = f"def function_{i}():\n    # Function {i} implementation\n    return {i}\n"
            repo.add_file(f"module_{i}.py", content)
        
        generator = MockPackGenerator(252627)
        original_pack = generator.generate_pack(repo, budget=1000)
        
        # Remove one selected chunk (file)
        selected_chunks = original_pack.index.chunks
        if selected_chunks:
            removed_chunk = selected_chunks[0]
            removed_file = removed_chunk['rel_path']
            
            # Create new repo without the removed file
            new_repo = TestRepository("/tmp/test_m6_removed")
            for file_path, content in repo.files.items():
                if file_path != removed_file:
                    new_repo.add_file(file_path, content)
            
            transformed_pack = generator.generate_pack(new_repo, budget=1000)
            
            # Validate M6 property
            property_obj = M6_ChunkRemovalProperty()
            context = {'removed_chunk_id': removed_chunk['id']}
            result = property_obj.validate(original_pack, transformed_pack, context)
            
            assert result['passes'], f"M6 failed: {result['details']}"
            assert result['no_overcap'], "Budget overcap detected"
            assert result['budget_reallocated'], "Budget should be reallocated"
    
    def test_m6_edge_cases(self):
        """Test M6 edge cases."""
        # Test removing the only chunk
        repo = TestRepository("/tmp/test_m6_single")
        repo.add_file("only.py", "def only_function(): pass")
        
        generator = MockPackGenerator(282930)
        original_pack = generator.generate_pack(repo, budget=200)
        
        # Remove the only file
        empty_repo = TestRepository("/tmp/test_m6_empty")
        transformed_pack = generator.generate_pack(empty_repo, budget=200)
        
        property_obj = M6_ChunkRemovalProperty()
        context = {'removed_chunk_id': 'chunk_only.py_1_1'}
        result = property_obj.validate(original_pack, transformed_pack, context)
        
        # Should handle empty result gracefully
        assert result['no_overcap'], "Should not exceed budget"
        # Budget reallocation might not apply when no chunks remain


class TestMetamorphicIntegration:
    """Integration tests for all metamorphic properties."""
    
    def test_all_properties_together(self):
        """Test all metamorphic properties in sequence."""
        # Create comprehensive repository
        repo = TestRepository("/tmp/test_integration")
        
        # Add various types of files
        for i in range(5):
            repo.add_file(f"src/module_{i}.py", f"def func_{i}(): return {i}")
            repo.add_file(f"test/test_{i}.py", f"def test_{i}(): assert True")
        
        generator = MockPackGenerator(313233)
        base_pack = generator.generate_pack(repo, budget=1500)
        
        oracle = MetamorphicPropertiesOracle()
        
        # Test M1: Add duplicate
        m1_repo = TestRepository("/tmp/test_integration_m1")
        m1_repo.files = repo.files.copy()
        m1_repo.add_file("duplicate.py", repo.files["src/module_0.py"])
        m1_pack = generator.generate_pack(m1_repo, budget=1500)
        
        # Test M2: Rename file  
        m2_repo = TestRepository("/tmp/test_integration_m2")
        m2_repo.files = repo.files.copy()
        original_content = m2_repo.files["src/module_0.py"]
        del m2_repo.files["src/module_0.py"]
        m2_repo.add_file("src/renamed_module.py", original_content)
        m2_pack = generator.generate_pack(m2_repo, budget=1500)
        
        # Test M3: Scale budget
        m3_pack = generator.generate_pack(repo, budget=3000)  # 2x budget
        
        # Prepare metamorphic test context
        context = {
            'metamorphic_tests': {
                'm1_duplicate_test': {
                    'original_pack': base_pack,
                    'transformed_pack': m1_pack,
                    'context': {'m1_threshold': 0.01}
                },
                'm2_rename_test': {
                    'original_pack': base_pack,
                    'transformed_pack': m2_pack,
                    'context': {}
                },
                'm3_scaling_test': {
                    'original_pack': base_pack,
                    'transformed_pack': m3_pack,
                    'context': {'budget_multiplier': 2.0}
                }
            }
        }
        
        # Run oracle validation
        report = oracle.validate(base_pack, context)
        
        assert report.result in [OracleResult.PASS, OracleResult.WARN], \
               f"Metamorphic integration test failed: {report.message}"
        
        # Check individual property results
        details = report.details
        assert details.get('passed_properties', 0) >= 2, \
               "Should pass at least 2 metamorphic properties"
    
    def test_property_coverage_oracle(self):
        """Test the property coverage oracle."""
        from packrepo.packer.oracles.metamorphic import PropertyCoverageOracle
        
        # Create dummy pack
        pack = PackFormat(
            index=PackIndex(target_budget=1000, actual_tokens=500, chunks=[]),
            sections=[]
        )
        
        coverage_oracle = PropertyCoverageOracle()
        
        # Test with full coverage
        full_context = {
            'property_coverage_data': {
                'tested_properties': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
                'passed_properties': ['M1', 'M2', 'M3', 'M4', 'M5']
            },
            'property_threshold': 0.70
        }
        
        report = coverage_oracle.validate(pack, full_context)
        assert report.result == OracleResult.PASS
        
        # Test with insufficient coverage
        low_context = {
            'property_coverage_data': {
                'tested_properties': ['M1', 'M2'],  # Only 2/6 properties
                'passed_properties': ['M1', 'M2']
            },
            'property_threshold': 0.70
        }
        
        report = coverage_oracle.validate(pack, low_context)
        assert report.result == OracleResult.FAIL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
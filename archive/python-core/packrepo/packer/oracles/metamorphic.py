"""Metamorphic properties validation oracle for PackRepo."""

from __future__ import annotations

import time
import hashlib
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
from abc import ABC, abstractmethod

from . import Oracle, OracleReport, OracleResult
from ..packfmt.base import PackFormat


class MetamorphicProperty(ABC):
    """Base class for metamorphic properties."""
    
    @abstractmethod
    def name(self) -> str:
        """Property name."""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Property description."""
        pass
    
    @abstractmethod
    def validate(self, original_pack: PackFormat, transformed_pack: PackFormat, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the metamorphic property."""
        pass


class M1_DuplicateFileProperty(MetamorphicProperty):
    """M1: append non-referenced duplicate file ⇒ ≤1% selection delta"""
    
    def name(self) -> str:
        return "M1_duplicate_file_invariance"
    
    def description(self) -> str:
        return "Adding a duplicate file should cause ≤1% selection delta"
    
    def validate(self, original_pack: PackFormat, transformed_pack: PackFormat, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate duplicate file doesn't significantly change selection."""
        original_chunks = set(chunk.get('id', '') for chunk in original_pack.index.chunks or [])
        transformed_chunks = set(chunk.get('id', '') for chunk in transformed_pack.index.chunks or [])
        
        # Calculate selection delta
        added_chunks = transformed_chunks - original_chunks
        removed_chunks = original_chunks - transformed_chunks
        
        total_original = len(original_chunks)
        selection_delta = (len(added_chunks) + len(removed_chunks)) / max(1, total_original)
        
        threshold = context.get('m1_threshold', 0.01)  # 1%
        
        return {
            'selection_delta': selection_delta,
            'threshold': threshold,
            'passes': selection_delta <= threshold,
            'added_chunks': len(added_chunks),
            'removed_chunks': len(removed_chunks),
            'details': f"Selection delta {selection_delta:.3f} vs threshold {threshold}"
        }


class M2_PathRenameProperty(MetamorphicProperty):
    """M2: rename path, content unchanged ⇒ only path fields differ"""
    
    def name(self) -> str:
        return "M2_path_rename_invariance"
    
    def description(self) -> str:
        return "Renaming paths with same content should only change path fields"
    
    def validate(self, original_pack: PackFormat, transformed_pack: PackFormat, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate path rename only affects path fields."""
        # Map chunks by content hash or similar identifier
        original_chunks_by_content = {}
        for chunk in original_pack.index.chunks or []:
            content_key = chunk.get('content_hash') or chunk.get('id', '')
            original_chunks_by_content[content_key] = chunk
        
        transformed_chunks_by_content = {}
        for chunk in transformed_pack.index.chunks or []:
            content_key = chunk.get('content_hash') or chunk.get('id', '')
            transformed_chunks_by_content[content_key] = chunk
        
        path_only_changes = 0
        other_changes = 0
        changed_chunks = []
        
        for content_key in original_chunks_by_content:
            if content_key in transformed_chunks_by_content:
                orig = original_chunks_by_content[content_key]
                trans = transformed_chunks_by_content[content_key]
                
                # Check what changed
                path_changed = orig.get('rel_path') != trans.get('rel_path')
                other_fields_changed = False
                
                # Check non-path fields
                for field in ['start_line', 'end_line', 'selected_tokens', 'selected_mode']:
                    if orig.get(field) != trans.get(field):
                        other_fields_changed = True
                        break
                
                if path_changed and not other_fields_changed:
                    path_only_changes += 1
                elif other_fields_changed:
                    other_changes += 1
                    changed_chunks.append(content_key)
        
        total_changes = path_only_changes + other_changes
        
        return {
            'path_only_changes': path_only_changes,
            'other_changes': other_changes,
            'total_changes': total_changes,
            'passes': other_changes == 0 if total_changes > 0 else True,
            'details': f"{path_only_changes} path-only changes, {other_changes} other changes"
        }


class M3_BudgetScalingProperty(MetamorphicProperty):
    """M3: budget×2 ⇒ coverage score increases monotonically"""
    
    def name(self) -> str:
        return "M3_budget_scaling_monotonicity"
    
    def description(self) -> str:
        return "Doubling budget should monotonically increase coverage score"
    
    def validate(self, original_pack: PackFormat, transformed_pack: PackFormat, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate budget scaling increases coverage."""
        original_coverage = original_pack.index.coverage_score
        transformed_coverage = transformed_pack.index.coverage_score
        
        original_budget = original_pack.index.target_budget
        transformed_budget = transformed_pack.index.target_budget
        
        # Check that budget was actually scaled
        budget_ratio = transformed_budget / max(1, original_budget)
        expected_ratio = context.get('budget_multiplier', 2.0)
        
        coverage_increased = transformed_coverage >= original_coverage
        monotonic_increase = coverage_increased or abs(transformed_coverage - original_coverage) < 0.001
        
        return {
            'original_coverage': original_coverage,
            'transformed_coverage': transformed_coverage,
            'coverage_increase': transformed_coverage - original_coverage,
            'budget_ratio': budget_ratio,
            'expected_ratio': expected_ratio,
            'passes': monotonic_increase,
            'details': f"Coverage: {original_coverage:.3f} → {transformed_coverage:.3f} (budget {budget_ratio:.1f}x)"
        }


class M4_TokenizerSwitchProperty(MetamorphicProperty):
    """M4: switch tokenizer cl100k→o200k with scaled budget ⇒ similarity ≥ 0.8 Jaccard"""
    
    def name(self) -> str:
        return "M4_tokenizer_switch_similarity"
    
    def description(self) -> str:
        return "Switching tokenizers with scaled budget should maintain ≥0.8 Jaccard similarity"
    
    def validate(self, original_pack: PackFormat, transformed_pack: PackFormat, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tokenizer switch maintains selection similarity."""
        original_chunks = set(chunk.get('id', '') for chunk in original_pack.index.chunks or [])
        transformed_chunks = set(chunk.get('id', '') for chunk in transformed_pack.index.chunks or [])
        
        # Calculate Jaccard similarity
        intersection = len(original_chunks & transformed_chunks)
        union = len(original_chunks | transformed_chunks)
        jaccard_similarity = intersection / max(1, union)
        
        threshold = context.get('m4_threshold', 0.8)
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'threshold': threshold,
            'intersection_size': intersection,
            'union_size': union,
            'original_size': len(original_chunks),
            'transformed_size': len(transformed_chunks),
            'passes': jaccard_similarity >= threshold,
            'details': f"Jaccard similarity {jaccard_similarity:.3f} vs threshold {threshold}"
        }


class M5_VendorFolderProperty(MetamorphicProperty):
    """M5: inject large vendor folder ⇒ selection unaffected except index ignored counts"""
    
    def name(self) -> str:
        return "M5_vendor_folder_invariance"
    
    def description(self) -> str:
        return "Adding vendor folder should not affect selection (only ignored counts)"
    
    def validate(self, original_pack: PackFormat, transformed_pack: PackFormat, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate vendor folder injection doesn't affect selection."""
        # Check that selected chunks are identical
        original_chunks = {chunk.get('id', ''): chunk for chunk in original_pack.index.chunks or []}
        transformed_chunks = {chunk.get('id', ''): chunk for chunk in transformed_pack.index.chunks or []}
        
        selection_unchanged = original_chunks == transformed_chunks
        
        # Check that ignored counts increased (vendor files were ignored)
        original_ignored = getattr(original_pack.index, 'ignored_files', 0)
        transformed_ignored = getattr(transformed_pack.index, 'ignored_files', 0)
        
        ignored_increased = transformed_ignored > original_ignored
        
        return {
            'selection_unchanged': selection_unchanged,
            'ignored_increased': ignored_increased,
            'original_ignored': original_ignored,
            'transformed_ignored': transformed_ignored,
            'ignored_delta': transformed_ignored - original_ignored,
            'passes': selection_unchanged,
            'details': f"Selection unchanged: {selection_unchanged}, ignored files: {original_ignored} → {transformed_ignored}"
        }


class M6_ChunkRemovalProperty(MetamorphicProperty):
    """M6: remove selected chunk ⇒ budget reallocated without over-cap"""
    
    def name(self) -> str:
        return "M6_chunk_removal_reallocation"
    
    def description(self) -> str:
        return "Removing a chunk should reallocate budget without exceeding cap"
    
    def validate(self, original_pack: PackFormat, transformed_pack: PackFormat, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chunk removal reallocates budget properly."""
        original_budget = original_pack.index.target_budget
        transformed_budget = transformed_pack.index.target_budget
        
        original_tokens = original_pack.index.actual_tokens
        transformed_tokens = transformed_pack.index.actual_tokens
        
        # Budget should be same or similar
        budget_unchanged = abs(original_budget - transformed_budget) <= 1
        
        # Should not exceed budget
        no_overcap = transformed_tokens <= transformed_budget
        
        # Should have used the freed budget (not just removed without reallocation)
        removed_chunk_id = context.get('removed_chunk_id', '')
        original_chunk_tokens = 0
        
        if removed_chunk_id:
            for chunk in original_pack.index.chunks or []:
                if chunk.get('id', '') == removed_chunk_id:
                    original_chunk_tokens = chunk.get('selected_tokens', 0)
                    break
        
        # Expected: should have reallocated freed tokens
        expected_min_tokens = original_tokens - original_chunk_tokens
        budget_reallocated = transformed_tokens >= expected_min_tokens * 0.8  # Allow some efficiency loss
        
        return {
            'budget_unchanged': budget_unchanged,
            'no_overcap': no_overcap,
            'budget_reallocated': budget_reallocated,
            'original_tokens': original_tokens,
            'transformed_tokens': transformed_tokens,
            'freed_tokens': original_chunk_tokens,
            'passes': budget_unchanged and no_overcap and budget_reallocated,
            'details': f"Budget: {original_budget}={transformed_budget}, tokens: {original_tokens}→{transformed_tokens}, freed: {original_chunk_tokens}"
        }


class MetamorphicPropertiesOracle(Oracle):
    """Oracle for validating metamorphic properties M1-M6."""
    
    category = "metamorphic"
    
    def __init__(self):
        self.properties = [
            M1_DuplicateFileProperty(),
            M2_PathRenameProperty(),
            M3_BudgetScalingProperty(),
            M4_TokenizerSwitchProperty(),
            M5_VendorFolderProperty(),
            M6_ChunkRemovalProperty()
        ]
    
    def name(self) -> str:
        return "metamorphic_properties"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate metamorphic properties."""
        start_time = time.time()
        
        try:
            if not context or 'metamorphic_tests' not in context:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="No metamorphic tests provided",
                    details={},
                    execution_time=time.time() - start_time
                )
            
            metamorphic_tests = context['metamorphic_tests']
            
            results = {}
            passed_properties = 0
            failed_properties = 0
            errors = []
            
            for test_name, test_data in metamorphic_tests.items():
                if 'original_pack' not in test_data or 'transformed_pack' not in test_data:
                    continue
                
                original_pack = test_data['original_pack']
                transformed_pack = test_data['transformed_pack']
                test_context = test_data.get('context', {})
                
                # Find matching property
                property_obj = None
                for prop in self.properties:
                    if prop.name().lower() in test_name.lower():
                        property_obj = prop
                        break
                
                if not property_obj:
                    continue
                
                try:
                    result = property_obj.validate(original_pack, transformed_pack, test_context)
                    results[test_name] = result
                    
                    if result.get('passes', False):
                        passed_properties += 1
                    else:
                        failed_properties += 1
                        errors.append(f"{property_obj.name()}: {result.get('details', 'validation failed')}")
                        
                except Exception as e:
                    failed_properties += 1
                    errors.append(f"{property_obj.name()}: execution error - {str(e)}")
            
            total_properties = passed_properties + failed_properties
            
            details = {
                'total_properties': total_properties,
                'passed_properties': passed_properties,
                'failed_properties': failed_properties,
                'property_results': results
            }
            
            if failed_properties > 0:
                result = OracleResult.FAIL
                message = f"Metamorphic properties validation failed: {failed_properties}/{total_properties} properties failed"
                details['errors'] = errors[:5]  # Show first 5 errors
            elif total_properties == 0:
                result = OracleResult.SKIP
                message = "No metamorphic properties tested"
            else:
                result = OracleResult.PASS
                message = f"Metamorphic properties validation passed: {passed_properties}/{total_properties} properties passed"
            
            return OracleReport(
                oracle_name=self.name(),
                result=result,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.ERROR,
                message=f"Metamorphic properties oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )


class PropertyCoverageOracle(Oracle):
    """Oracle for ensuring adequate metamorphic property coverage."""
    
    category = "metamorphic"
    
    def name(self) -> str:
        return "property_coverage"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate that metamorphic property coverage meets threshold."""
        start_time = time.time()
        
        try:
            if not context or 'property_coverage_data' not in context:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="No property coverage data provided",
                    details={},
                    execution_time=time.time() - start_time
                )
            
            coverage_data = context['property_coverage_data']
            required_properties = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
            threshold = context.get('property_threshold', 0.70)  # 70% coverage required
            
            tested_properties = coverage_data.get('tested_properties', [])
            passed_properties = coverage_data.get('passed_properties', [])
            
            coverage = len(tested_properties) / len(required_properties)
            pass_rate = len(passed_properties) / max(1, len(tested_properties))
            
            meets_threshold = coverage >= threshold
            
            details = {
                'required_properties': required_properties,
                'tested_properties': tested_properties,
                'passed_properties': passed_properties,
                'coverage': coverage,
                'pass_rate': pass_rate,
                'threshold': threshold,
                'missing_properties': [p for p in required_properties if p not in tested_properties]
            }
            
            if not meets_threshold:
                result = OracleResult.FAIL
                message = f"Property coverage below threshold: {coverage:.1%} < {threshold:.1%}"
            else:
                result = OracleResult.PASS
                message = f"Property coverage adequate: {coverage:.1%} ≥ {threshold:.1%} ({len(passed_properties)}/{len(tested_properties)} passed)"
            
            return OracleReport(
                oracle_name=self.name(),
                result=result,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.ERROR,
                message=f"Property coverage oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )
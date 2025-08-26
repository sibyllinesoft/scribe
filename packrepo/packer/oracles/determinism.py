"""Determinism validation oracle for PackRepo."""

from __future__ import annotations

import time
import hashlib
from typing import Dict, Any, Optional, List

from . import Oracle, OracleReport, OracleResult
from ..packfmt.base import PackFormat


class DeterminismOracle(Oracle):
    """Oracle for validating deterministic output properties."""
    
    category = "determinism"
    
    def name(self) -> str:
        return "determinism_validation"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate deterministic properties of the pack."""
        start_time = time.time()
        
        try:
            errors = []
            details = {}
            
            # Check if deterministic mode was requested
            deterministic_mode = context.get("deterministic", False) if context else False
            details["deterministic_mode"] = deterministic_mode
            
            if not deterministic_mode:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="Deterministic mode not enabled, skipping determinism validation",
                    details=details,
                    execution_time=time.time() - start_time
                )
            
            index = pack.index
            
            # Validate chunk ordering is deterministic
            if index.chunks:
                # Check that chunks are sorted deterministically
                prev_file = ""
                prev_line = 0
                prev_id = ""
                
                for i, chunk in enumerate(index.chunks):
                    file_path = chunk.get('rel_path', '')
                    start_line = chunk.get('start_line', 0)
                    chunk_id = chunk.get('id', '')
                    
                    # Primary sort: file path
                    if file_path < prev_file:
                        errors.append(f"Chunks not sorted by file path at index {i}: {file_path} < {prev_file}")
                        break
                    
                    # Secondary sort: within file, by start line
                    elif file_path == prev_file:
                        if start_line < prev_line:
                            errors.append(f"Chunks not sorted by line number in {file_path} at index {i}")
                            break
                        # Tertiary sort: if same file and line, by chunk ID for tie-breaking
                        elif start_line == prev_line and chunk_id < prev_id:
                            errors.append(f"Chunks not sorted by chunk ID for tie-breaking at index {i}")
                            break
                    
                    prev_file = file_path
                    prev_line = start_line if file_path != prev_file else start_line
                    prev_id = chunk_id if file_path == prev_file and start_line == prev_line else chunk_id
                
                details["chunk_count"] = len(index.chunks)
                
                # Check for stable chunk IDs (should be deterministic, not random)
                chunk_ids = [chunk.get('id', '') for chunk in index.chunks]
                
                # Validate chunk ID uniqueness
                if len(chunk_ids) != len(set(chunk_ids)):
                    errors.append("Non-unique chunk IDs detected")
                    details["duplicate_chunk_ids"] = True
                
                # Check that chunk IDs appear to be content-based (not random UUIDs)
                random_looking_ids = []
                for chunk_id in chunk_ids[:5]:  # Check first 5 as sample
                    # Heuristic: random UUIDs vs content hashes
                    if len(chunk_id) == 36 and chunk_id.count('-') == 4:  # UUID format
                        random_looking_ids.append(chunk_id)
                    elif len(chunk_id) < 8:  # Very short IDs might be incremental
                        random_looking_ids.append(chunk_id)
                
                if random_looking_ids:
                    errors.append(f"Potentially non-deterministic chunk IDs detected: {random_looking_ids[:3]}...")
                    details["suspicious_chunk_ids"] = len(random_looking_ids)
            
            # Validate manifest digest exists and is valid
            if index.manifest_digest:
                details["manifest_digest"] = index.manifest_digest
                
                # Verify manifest digest is correct
                body_content = pack.body.format_body()
                expected_digest = index.generate_manifest_digest(body_content)
                
                if index.manifest_digest != expected_digest:
                    errors.append(f"Manifest digest mismatch: stored={index.manifest_digest}, calculated={expected_digest}")
                    details["digest_mismatch"] = True
            else:
                if deterministic_mode:
                    errors.append("Manifest digest missing in deterministic mode")
                    details["missing_manifest_digest"] = True
            
            # Validate canonical JSON structure
            try:
                canonical_json = index.to_json(canonical=True)
                details["canonical_json_length"] = len(canonical_json)
                
                # Check that keys are properly ordered (basic check)
                import json
                canonical_data = json.loads(canonical_json)
                
                # Check top-level key ordering
                keys = list(canonical_data.keys())
                expected_first_keys = ['format_version', 'created_at', 'packrepo_version']
                
                for i, expected_key in enumerate(expected_first_keys):
                    if i < len(keys) and keys[i] != expected_key and expected_key in keys:
                        errors.append(f"Canonical JSON key ordering incorrect: expected {expected_key} at position {i}, got {keys[i]}")
                        break
                        
            except Exception as e:
                errors.append(f"Failed to generate canonical JSON: {str(e)}")
                details["canonical_json_error"] = str(e)
            
            # Check for deterministic timestamps (should be fixed in deterministic mode)
            created_at = index.created_at
            if context and "fixed_timestamp" in context:
                expected_timestamp = context["fixed_timestamp"]
                if created_at != expected_timestamp:
                    errors.append(f"Non-deterministic timestamp: got {created_at}, expected {expected_timestamp}")
                    details["timestamp_mismatch"] = True
            
            # Overall validation result
            if errors:
                result = OracleResult.FAIL
                message = f"Determinism validation failed: {'; '.join(errors[:3])}{'...' if len(errors) > 3 else ''}"
                details["error_count"] = len(errors)
            else:
                result = OracleResult.PASS
                message = f"Determinism validation passed: pack exhibits deterministic properties"
            
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
                message=f"Determinism oracle execution failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )


class HashConsistencyOracle(Oracle):
    """Oracle for validating hash consistency across multiple runs."""
    
    category = "determinism"
    
    def name(self) -> str:
        return "hash_consistency"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate hash consistency for deterministic output."""
        start_time = time.time()
        
        try:
            details = {}
            
            # Generate pack hash
            pack_hash = pack.generate_deterministic_hash()
            details["current_hash"] = pack_hash
            
            # Check against previous hashes if provided
            if context and "previous_hashes" in context:
                previous_hashes = context["previous_hashes"]
                details["previous_hashes"] = previous_hashes
                
                # All hashes should be identical for deterministic output
                all_hashes = previous_hashes + [pack_hash]
                unique_hashes = set(all_hashes)
                
                if len(unique_hashes) > 1:
                    return OracleReport(
                        oracle_name=self.name(),
                        result=OracleResult.FAIL,
                        message=f"Hash inconsistency detected: {len(unique_hashes)} different hashes across runs",
                        details={**details, "unique_hashes": list(unique_hashes), "hash_mismatches": True},
                        execution_time=time.time() - start_time
                    )
                else:
                    return OracleReport(
                        oracle_name=self.name(),
                        result=OracleResult.PASS,
                        message=f"Hash consistency validated across {len(all_hashes)} runs",
                        details={**details, "consistent_runs": len(all_hashes)},
                        execution_time=time.time() - start_time
                    )
            else:
                # First run - just record hash
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.PASS,
                    message=f"Hash recorded for consistency checking: {pack_hash}",
                    details=details,
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.ERROR,
                message=f"Hash consistency oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )
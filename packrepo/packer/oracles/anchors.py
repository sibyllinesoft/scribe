"""Anchor resolution validation oracle for PackRepo."""

from __future__ import annotations

import time
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from . import Oracle, OracleReport, OracleResult
from ..packfmt.base import PackFormat


class AnchorResolutionOracle(Oracle):
    """Oracle for validating that line ranges resolve correctly to actual file content."""
    
    category = "anchors"
    
    def name(self) -> str:
        return "anchor_resolution"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate that all chunk anchors resolve to valid file ranges."""
        start_time = time.time()
        
        try:
            errors = []
            details = {}
            
            index = pack.index
            chunks = index.chunks or []
            
            details["total_chunks"] = len(chunks)
            
            if not chunks:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="No chunks to validate anchors for",
                    details=details,
                    execution_time=time.time() - start_time
                )
            
            # Get repository path from context if available
            repo_path = context.get("repo_path") if context else None
            details["repo_path_available"] = repo_path is not None
            
            invalid_ranges = []
            overlapping_chunks = []
            unresolvable_files = []
            valid_chunks = 0
            
            # Group chunks by file for overlap detection
            file_chunks: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
            
            for i, chunk in enumerate(chunks):
                file_path = chunk.get('rel_path', '')
                start_line = chunk.get('start_line', 0)
                end_line = chunk.get('end_line', 0)
                chunk_id = chunk.get('id', f'chunk_{i}')
                
                # Validate line range is sensible
                if start_line <= 0:
                    invalid_ranges.append(f"Chunk {chunk_id}: invalid start_line {start_line} (must be > 0)")
                    continue
                    
                if end_line < start_line:
                    invalid_ranges.append(f"Chunk {chunk_id}: invalid range {start_line}-{end_line} (end < start)")
                    continue
                
                # Group by file for overlap checking
                if file_path not in file_chunks:
                    file_chunks[file_path] = []
                file_chunks[file_path].append((i, chunk))
                
                # If we have repo path, validate against actual file
                if repo_path:
                    full_path = Path(repo_path) / file_path
                    
                    if not full_path.exists():
                        unresolvable_files.append(f"Chunk {chunk_id}: file {file_path} does not exist")
                        continue
                    
                    try:
                        # Read file and check line range
                        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()
                        
                        total_lines = len(lines)
                        
                        if end_line > total_lines:
                            invalid_ranges.append(f"Chunk {chunk_id}: end_line {end_line} > file length {total_lines}")
                            continue
                        
                        # Extract actual content for validation
                        chunk_lines = lines[start_line-1:end_line]  # Convert to 0-based indexing
                        chunk_content = ''.join(chunk_lines)
                        
                        # Check if content looks reasonable (not all whitespace)
                        if chunk_content.strip() == "":
                            invalid_ranges.append(f"Chunk {chunk_id}: extracted content is empty/whitespace only")
                            continue
                        
                        valid_chunks += 1
                        
                    except Exception as e:
                        unresolvable_files.append(f"Chunk {chunk_id}: failed to read {file_path}: {str(e)}")
                        continue
                else:
                    # Without repo access, just validate the range looks reasonable
                    valid_chunks += 1
            
            # Check for overlapping chunks within files
            for file_path, chunks_in_file in file_chunks.items():
                if len(chunks_in_file) <= 1:
                    continue
                
                # Sort by start line
                sorted_chunks = sorted(chunks_in_file, key=lambda x: x[1].get('start_line', 0))
                
                for j in range(len(sorted_chunks) - 1):
                    current_chunk = sorted_chunks[j][1]
                    next_chunk = sorted_chunks[j+1][1]
                    
                    current_end = current_chunk.get('end_line', 0)
                    next_start = next_chunk.get('start_line', 0)
                    
                    if current_end >= next_start:
                        current_id = current_chunk.get('id', 'unknown')
                        next_id = next_chunk.get('id', 'unknown')
                        overlapping_chunks.append(
                            f"File {file_path}: chunks {current_id} and {next_id} overlap "
                            f"(lines {current_chunk.get('start_line')}-{current_end} vs {next_start}-{next_chunk.get('end_line')})"
                        )
            
            # Validate section headers format
            body_sections = pack.body.sections or []
            header_format_errors = []
            
            for section in body_sections:
                expected_header = section.format_header()
                # Check header format: "### path: {path} lines: {start}-{end} mode: {mode}"
                header_pattern = r'^### path: .+ lines: \d+-\d+ mode: \w+$'
                
                if not re.match(header_pattern, expected_header):
                    header_format_errors.append(f"Invalid header format for chunk {section.chunk_id}: {expected_header}")
            
            # Collect all errors
            all_errors = invalid_ranges + overlapping_chunks + unresolvable_files + header_format_errors
            
            details.update({
                "valid_chunks": valid_chunks,
                "invalid_ranges": len(invalid_ranges),
                "overlapping_chunks": len(overlapping_chunks),
                "unresolvable_files": len(unresolvable_files),
                "header_format_errors": len(header_format_errors),
                "files_checked": len(file_chunks)
            })
            
            if all_errors:
                result = OracleResult.FAIL
                message = f"Anchor resolution failed: {len(all_errors)} issues found"
                details["errors"] = all_errors[:10]  # Include first 10 errors
                if len(all_errors) > 10:
                    details["additional_errors"] = len(all_errors) - 10
            else:
                result = OracleResult.PASS
                message = f"Anchor resolution passed: all {valid_chunks} chunks have valid line ranges"
            
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
                message=f"Anchor resolution oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )


class SectionIntegrityOracle(Oracle):
    """Oracle for validating pack section integrity and consistency."""
    
    category = "anchors"
    
    def name(self) -> str:
        return "section_integrity"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate consistency between index and body sections."""
        start_time = time.time()
        
        try:
            errors = []
            details = {}
            
            index_chunks = {chunk.get('id', ''): chunk for chunk in pack.index.chunks or []}
            body_sections = {section.chunk_id: section for section in pack.body.sections or []}
            
            details["index_chunks"] = len(index_chunks)
            details["body_sections"] = len(body_sections)
            
            # Check that every index chunk has a corresponding body section
            missing_sections = []
            for chunk_id, chunk in index_chunks.items():
                if chunk_id not in body_sections:
                    missing_sections.append(chunk_id)
            
            # Check that every body section has a corresponding index chunk
            orphaned_sections = []
            for chunk_id, section in body_sections.items():
                if chunk_id not in index_chunks:
                    orphaned_sections.append(chunk_id)
            
            # Validate consistency between index and body data
            inconsistent_metadata = []
            for chunk_id in set(index_chunks.keys()) & set(body_sections.keys()):
                chunk = index_chunks[chunk_id]
                section = body_sections[chunk_id]
                
                # Check path consistency
                if chunk.get('rel_path', '') != section.path:
                    inconsistent_metadata.append(f"Chunk {chunk_id}: path mismatch")
                
                # Check line range consistency
                if chunk.get('start_line', 0) != section.start_line:
                    inconsistent_metadata.append(f"Chunk {chunk_id}: start_line mismatch")
                
                if chunk.get('end_line', 0) != section.end_line:
                    inconsistent_metadata.append(f"Chunk {chunk_id}: end_line mismatch")
                
                # Check mode consistency
                if chunk.get('selected_mode', '') != section.mode:
                    inconsistent_metadata.append(f"Chunk {chunk_id}: mode mismatch")
                
                # Check token count consistency
                if chunk.get('selected_tokens', 0) != section.token_count:
                    inconsistent_metadata.append(f"Chunk {chunk_id}: token_count mismatch")
            
            all_errors = missing_sections + orphaned_sections + inconsistent_metadata
            
            details.update({
                "missing_sections": len(missing_sections),
                "orphaned_sections": len(orphaned_sections),
                "inconsistent_metadata": len(inconsistent_metadata)
            })
            
            if all_errors:
                result = OracleResult.FAIL
                message = f"Section integrity failed: {len(all_errors)} issues found"
                details["errors"] = all_errors[:10]
            else:
                result = OracleResult.PASS
                message = f"Section integrity passed: {len(index_chunks)} chunks consistent between index and body"
            
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
                message=f"Section integrity oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )
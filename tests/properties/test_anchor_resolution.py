"""Property-based testing for PackRepo anchor resolution.

Tests the anchor invariants from TODO.md:
- Line ranges non-overlapping
- All headers well-formed  
- Anchors resolve correctly
- 100% sections have valid anchors
"""

from __future__ import annotations

import pytest
import re
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize
from typing import Dict, Any, List, Optional, Tuple, Set

from packrepo.packer.packfmt.base import PackFormat, PackIndex, PackSection
from packrepo.packer.oracles.anchors import AnchorResolutionOracle, SectionIntegrityOracle
from packrepo.packer.oracles import OracleResult


def generate_pack_body(sections: List[PackSection]) -> str:
    """Generate pack body with proper anchor headers."""
    body_parts = []
    
    for section in sections:
        # Generate well-formed header
        header = f"### {section.rel_path}: lines {section.start_line}-{section.end_line} mode: {section.mode}"
        body_parts.append(header)
        body_parts.append(section.content)
        body_parts.append("")  # Empty line between sections
    
    return "\n".join(body_parts)


def parse_anchors_from_body(body: str) -> List[Dict[str, Any]]:
    """Parse anchor headers from pack body."""
    anchors = []
    
    # Pattern for well-formed headers
    header_pattern = r"^### (.+?): lines (\d+)-(\d+) mode: (\w+)$"
    
    lines = body.split('\n')
    current_content = []
    current_anchor = None
    
    for line in lines:
        match = re.match(header_pattern, line)
        if match:
            # Save previous anchor if exists
            if current_anchor:
                current_anchor['content'] = '\n'.join(current_content)
                anchors.append(current_anchor)
            
            # Start new anchor
            rel_path, start_line, end_line, mode = match.groups()
            current_anchor = {
                'rel_path': rel_path,
                'start_line': int(start_line),
                'end_line': int(end_line),
                'mode': mode,
                'header_line': line
            }
            current_content = []
        else:
            if current_anchor:
                current_content.append(line)
    
    # Add final anchor
    if current_anchor:
        current_anchor['content'] = '\n'.join(current_content)
        anchors.append(current_anchor)
    
    return anchors


# Strategies for generating test data
@st.composite
def line_range_strategy(draw):
    """Generate valid line ranges."""
    start = draw(st.integers(min_value=1, max_value=1000))
    length = draw(st.integers(min_value=1, max_value=100))
    return start, start + length - 1


@st.composite  
def non_overlapping_sections_strategy(draw, max_sections=10):
    """Generate non-overlapping sections."""
    num_sections = draw(st.integers(min_value=1, max_value=max_sections))
    
    # Generate file paths
    file_paths = []
    for i in range(num_sections):
        path = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
        file_paths.append(f"{path}_{i}.py")  # Make unique
    
    sections = []
    used_ranges_per_file = {}
    
    for i in range(num_sections):
        file_path = file_paths[i]
        
        if file_path not in used_ranges_per_file:
            used_ranges_per_file[file_path] = []
        
        # Generate non-overlapping range for this file
        attempts = 0
        while attempts < 50:  # Avoid infinite loops
            start, end = draw(line_range_strategy())
            
            # Check for overlap with existing ranges in same file
            overlaps = any(
                not (end < existing_start or start > existing_end)
                for existing_start, existing_end in used_ranges_per_file[file_path]
            )
            
            if not overlaps:
                used_ranges_per_file[file_path].append((start, end))
                
                content = draw(st.text(min_size=10, max_size=200))
                mode = draw(st.sampled_from(['full', 'summary', 'signature']))
                
                sections.append(PackSection(
                    rel_path=file_path,
                    start_line=start,
                    end_line=end,
                    content=content,
                    mode=mode
                ))
                break
            
            attempts += 1
        
        if attempts >= 50:
            # Skip this section if we can't find non-overlapping range
            continue
    
    return sections


@st.composite
def pack_with_anchors_strategy(draw):
    """Generate pack with proper anchor structure."""
    sections = draw(non_overlapping_sections_strategy())
    assume(len(sections) > 0)
    
    # Generate corresponding chunks
    chunks = []
    for i, section in enumerate(sections):
        chunks.append({
            'id': f"chunk_{i}",
            'rel_path': section.rel_path,
            'start_line': section.start_line,
            'end_line': section.end_line,
            'selected_tokens': len(section.content),
            'selected_mode': section.mode
        })
    
    index = PackIndex(
        target_budget=10000,
        actual_tokens=sum(chunk['selected_tokens'] for chunk in chunks),
        chunks=chunks
    )
    
    pack = PackFormat(index=index, sections=sections)
    
    # Generate body with anchors
    body = generate_pack_body(sections)
    pack._body = body  # Store for testing
    
    return pack


class TestAnchorResolutionProperties:
    """Property-based tests for anchor resolution."""
    
    @given(pack_with_anchors_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_no_overlapping_ranges_property(self, pack: PackFormat):
        """Property: Line ranges should not overlap within same file."""
        # Group chunks by file
        chunks_by_file = {}
        for chunk in pack.index.chunks:
            rel_path = chunk['rel_path']
            if rel_path not in chunks_by_file:
                chunks_by_file[rel_path] = []
            chunks_by_file[rel_path].append(chunk)
        
        # Check for overlaps within each file
        for file_path, chunks in chunks_by_file.items():
            ranges = [(chunk['start_line'], chunk['end_line']) for chunk in chunks]
            
            # Check all pairs for overlaps
            for i, (start1, end1) in enumerate(ranges):
                for j, (start2, end2) in enumerate(ranges[i+1:], i+1):
                    # Ranges overlap if not (end1 < start2 or start1 > end2)
                    overlaps = not (end1 < start2 or start1 > end2)
                    
                    assert not overlaps, \
                           f"Overlapping ranges in {file_path}: {start1}-{end1} and {start2}-{end2}"
            
            note(f"File {file_path}: {len(ranges)} non-overlapping ranges")
        
        oracle = SectionIntegrityOracle()
        report = oracle.validate(pack)
        
        assert report.result in [OracleResult.PASS, OracleResult.WARN]
    
    @given(pack_with_anchors_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_well_formed_headers_property(self, pack: PackFormat):
        """Property: All headers should be well-formed."""
        if not hasattr(pack, '_body') or not pack._body:
            # Generate body if not present
            pack._body = generate_pack_body(pack.sections)
        
        anchors = parse_anchors_from_body(pack._body)
        
        # Check that each anchor has well-formed header
        header_pattern = r"^### (.+?): lines (\d+)-(\d+) mode: (\w+)$"
        
        for anchor in anchors:
            header = anchor['header_line']
            match = re.match(header_pattern, header)
            
            assert match, f"Malformed header: {header}"
            
            rel_path, start_line, end_line, mode = match.groups()
            
            # Validate components
            assert rel_path == anchor['rel_path']
            assert int(start_line) == anchor['start_line']
            assert int(end_line) == anchor['end_line']
            assert mode == anchor['mode']
            
            note(f"Well-formed header: {header}")
        
        oracle = AnchorResolutionOracle()
        context = {'anchors': anchors}
        report = oracle.validate(pack, context)
        
        assert report.result in [OracleResult.PASS, OracleResult.WARN]
    
    @given(pack_with_anchors_strategy())
    @settings(max_examples=50, deadline=5000) 
    def test_anchor_chunk_consistency_property(self, pack: PackFormat):
        """Property: Anchors should be consistent with index chunks."""
        if not hasattr(pack, '_body') or not pack._body:
            pack._body = generate_pack_body(pack.sections)
        
        anchors = parse_anchors_from_body(pack._body)
        chunks = pack.index.chunks
        
        # Should have same number of anchors and chunks
        assert len(anchors) == len(chunks), \
               f"Anchor/chunk count mismatch: {len(anchors)} anchors, {len(chunks)} chunks"
        
        # Sort both by consistent order
        anchors_sorted = sorted(anchors, key=lambda a: (a['rel_path'], a['start_line']))
        chunks_sorted = sorted(chunks, key=lambda c: (c['rel_path'], c['start_line']))
        
        # Check consistency
        for anchor, chunk in zip(anchors_sorted, chunks_sorted):
            assert anchor['rel_path'] == chunk['rel_path'], \
                   f"Path mismatch: anchor={anchor['rel_path']}, chunk={chunk['rel_path']}"
            assert anchor['start_line'] == chunk['start_line'], \
                   f"Start line mismatch: anchor={anchor['start_line']}, chunk={chunk['start_line']}"
            assert anchor['end_line'] == chunk['end_line'], \
                   f"End line mismatch: anchor={anchor['end_line']}, chunk={chunk['end_line']}"
            assert anchor['mode'] == chunk['selected_mode'], \
                   f"Mode mismatch: anchor={anchor['mode']}, chunk={chunk['selected_mode']}"
        
        note(f"All {len(anchors)} anchors consistent with chunks")
    
    def test_line_number_validity_property(self):
        """Property: Line numbers should be valid (start ≤ end, both > 0)."""
        sections = [
            PackSection(rel_path="test.py", start_line=1, end_line=5, content="valid", mode="full"),
            PackSection(rel_path="test.py", start_line=10, end_line=15, content="also valid", mode="full"),
        ]
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=1000,
                actual_tokens=100,
                chunks=[
                    {'rel_path': 'test.py', 'start_line': 1, 'end_line': 5, 'selected_mode': 'full'},
                    {'rel_path': 'test.py', 'start_line': 10, 'end_line': 15, 'selected_mode': 'full'},
                ]
            ),
            sections=sections
        )
        
        oracle = SectionIntegrityOracle()
        report = oracle.validate(pack)
        
        assert report.result == OracleResult.PASS


class AnchorStateMachine(RuleBasedStateMachine):
    """Stateful testing for anchor resolution."""
    
    def __init__(self):
        super().__init__()
        self.sections: List[PackSection] = []
        self.file_ranges: Dict[str, List[Tuple[int, int]]] = {}
    
    @initialize()
    def setup(self):
        """Initialize anchor testing state."""
        self.sections = []
        self.file_ranges = {}
    
    @rule(file_path=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
          start_line=st.integers(min_value=1, max_value=100),
          length=st.integers(min_value=1, max_value=50))
    def add_section(self, file_path: str, start_line: int, length: int):
        """Add a section with non-overlapping line range."""
        file_path = f"{file_path}.py"
        end_line = start_line + length - 1
        
        # Check for overlaps with existing ranges in same file
        if file_path in self.file_ranges:
            for existing_start, existing_end in self.file_ranges[file_path]:
                overlaps = not (end_line < existing_start or start_line > existing_end)
                assume(not overlaps)
        
        # Add the range
        if file_path not in self.file_ranges:
            self.file_ranges[file_path] = []
        self.file_ranges[file_path].append((start_line, end_line))
        
        # Create section
        content = f"Content for {file_path}:{start_line}-{end_line}"
        section = PackSection(
            rel_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            mode="full"
        )
        self.sections.append(section)
        
        note(f"Added section: {file_path}:{start_line}-{end_line}")
    
    @invariant()
    def no_overlaps_invariant(self):
        """Line ranges should never overlap within same file."""
        for file_path, ranges in self.file_ranges.items():
            for i, (start1, end1) in enumerate(ranges):
                for j, (start2, end2) in enumerate(ranges[i+1:], i+1):
                    overlaps = not (end1 < start2 or start1 > end2)
                    assert not overlaps, \
                           f"Overlapping ranges in {file_path}: {start1}-{end1} and {start2}-{end2}"
    
    @invariant()
    def valid_line_numbers_invariant(self):
        """All line numbers should be valid."""
        for section in self.sections:
            assert section.start_line > 0, f"Invalid start line: {section.start_line}"
            assert section.end_line > 0, f"Invalid end line: {section.end_line}"
            assert section.start_line <= section.end_line, \
                   f"Start > end: {section.start_line} > {section.end_line}"


TestAnchorStateMachine = AnchorStateMachine.TestCase


class TestAnchorEdgeCases:
    """Test edge cases for anchor resolution."""
    
    def test_single_line_section(self):
        """Test sections with single line (start_line == end_line)."""
        section = PackSection(
            rel_path="single.py",
            start_line=42,
            end_line=42,
            content="x = 1",
            mode="full"
        )
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=100,
                actual_tokens=5,
                chunks=[{
                    'rel_path': 'single.py',
                    'start_line': 42,
                    'end_line': 42,
                    'selected_tokens': 5,
                    'selected_mode': 'full'
                }]
            ),
            sections=[section]
        )
        
        oracle = SectionIntegrityOracle()
        report = oracle.validate(pack)
        
        assert report.result == OracleResult.PASS
    
    def test_large_line_numbers(self):
        """Test handling of large line numbers."""
        section = PackSection(
            rel_path="large.py",
            start_line=999999,
            end_line=1000000,
            content="# Very large file",
            mode="full"
        )
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=100,
                actual_tokens=20,
                chunks=[{
                    'rel_path': 'large.py',
                    'start_line': 999999,
                    'end_line': 1000000,
                    'selected_tokens': 20,
                    'selected_mode': 'full'
                }]
            ),
            sections=[section]
        )
        
        oracle = SectionIntegrityOracle()
        report = oracle.validate(pack)
        
        assert report.result == OracleResult.PASS
    
    def test_adjacent_non_overlapping_ranges(self):
        """Test adjacent ranges that don't overlap."""
        sections = [
            PackSection(rel_path="adjacent.py", start_line=1, end_line=10, content="first", mode="full"),
            PackSection(rel_path="adjacent.py", start_line=11, end_line=20, content="second", mode="full"),
            PackSection(rel_path="adjacent.py", start_line=21, end_line=30, content="third", mode="full"),
        ]
        
        chunks = []
        for i, section in enumerate(sections):
            chunks.append({
                'rel_path': section.rel_path,
                'start_line': section.start_line,
                'end_line': section.end_line,
                'selected_tokens': 10,
                'selected_mode': section.mode
            })
        
        pack = PackFormat(
            index=PackIndex(
                target_budget=100,
                actual_tokens=30,
                chunks=chunks
            ),
            sections=sections
        )
        
        oracle = SectionIntegrityOracle()
        report = oracle.validate(pack)
        
        assert report.result == OracleResult.PASS
    
    def test_malformed_header_detection(self):
        """Test detection of malformed headers."""
        # This would be tested by checking pack body parsing
        # Simulating malformed headers
        malformed_headers = [
            "### missing_colon_lines 1-10 mode: full",  # Missing colon after filename
            "### file.py lines 1-10 mode",  # Missing colon before mode
            "### file.py: lines abc-def mode: full",  # Invalid line numbers
            "### file.py: lines 10-5 mode: full",  # Start > end
        ]
        
        for header in malformed_headers:
            header_pattern = r"^### (.+?): lines (\d+)-(\d+) mode: (\w+)$"
            match = re.match(header_pattern, header)
            
            # Should not match well-formed pattern
            assert not match, f"Malformed header incorrectly validated: {header}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
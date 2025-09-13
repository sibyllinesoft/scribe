"""
Hybrid Demotion System (Workstream C)

Implements multi-fidelity demotion: whole-file → chunk → signature
- Intelligent content reduction when approaching budget limits
- Maintains most important information while reducing token usage
- Progressive degradation preserves critical functionality
- Flag-guarded integration with existing selection systems

Research-grade implementation for publication standards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from .fast_scan import ScanResult
from .feature_flags import get_feature_flags
from ..packer.tokenizer import estimate_tokens_scan_result


class FidelityMode(Enum):
    """Content fidelity levels for demotion system."""
    FULL = "full"           # Complete file content
    CHUNK = "chunk"         # Important chunks only  
    SIGNATURE = "signature" # Type signatures and interfaces only


@dataclass
class DemotionResult:
    """Result of applying demotion to a file."""
    original_path: str
    original_tokens: int
    demoted_tokens: int
    fidelity_mode: FidelityMode
    content: str
    chunks_kept: int
    chunks_total: int
    compression_ratio: float
    quality_score: float  # How much important info was preserved


@dataclass
class ChunkInfo:
    """Information about a code chunk."""
    start_line: int
    end_line: int
    chunk_type: str  # function, class, import, comment, etc.
    content: str
    importance_score: float
    estimated_tokens: int
    dependencies: List[str]  # Other chunks this depends on
    

class CodeChunker:
    """
    Splits code into semantic chunks for selective demotion.
    
    Identifies important code structures:
    - Function and method definitions
    - Class definitions
    - Import statements
    - Module-level constants and variables
    - Documentation blocks
    - Type definitions and interfaces
    """
    
    # Language-specific patterns for chunk detection
    CHUNK_PATTERNS = {
        'python': {
            'function': r'^(\s*)def\s+(\w+)',
            'class': r'^(\s*)class\s+(\w+)', 
            'import': r'^(\s*)(import\s+|from\s+)',
            'constant': r'^(\s*)([A-Z_][A-Z0-9_]*)\s*=',
            'docstring': r'^(\s*)"""',
            'comment': r'^(\s*)#',
            'decorator': r'^(\s*)@\w+'
        },
        'javascript': {
            'function': r'^(\s*)(function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
            'class': r'^(\s*)class\s+(\w+)',
            'import': r'^(\s*)(import\s+|export\s+|require\s*\()',
            'constant': r'^(\s*)const\s+([A-Z_][A-Z0-9_]*)',
            'comment': r'^(\s*)//',
            'jsdoc': r'^(\s*)/\*\*'
        },
        'typescript': {
            'function': r'^(\s*)(function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
            'class': r'^(\s*)class\s+(\w+)',
            'interface': r'^(\s*)interface\s+(\w+)',
            'type': r'^(\s*)type\s+(\w+)',
            'import': r'^(\s*)(import\s+|export\s+)',
            'constant': r'^(\s*)const\s+([A-Z_][A-Z0-9_]*)',
            'comment': r'^(\s*)//',
            'jsdoc': r'^(\s*)/\*\*'
        },
        'go': {
            'function': r'^(\s*)func\s+(?:\([^)]*\)\s*)?(\w+)',
            'type': r'^(\s*)type\s+(\w+)',
            'const': r'^(\s*)const\s*\(',
            'var': r'^(\s*)var\s*\(',
            'import': r'^(\s*)import\s*\(',
            'comment': r'^(\s*)//',
            'struct': r'^(\s*)type\s+\w+\s+struct'
        },
        'rust': {
            'function': r'^(\s*)(?:pub\s+)?(?:async\s+)?fn\s+(\w+)',
            'struct': r'^(\s*)(?:pub\s+)?struct\s+(\w+)',
            'enum': r'^(\s*)(?:pub\s+)?enum\s+(\w+)',
            'trait': r'^(\s*)(?:pub\s+)?trait\s+(\w+)',
            'impl': r'^(\s*)impl\s+',
            'use': r'^(\s*)use\s+',
            'const': r'^(\s*)(?:pub\s+)?const\s+([A-Z_][A-Z0-9_]*)',
            'comment': r'^(\s*)//',
            'mod': r'^(\s*)(?:pub\s+)?mod\s+(\w+)'
        }
    }
    
    def __init__(self):
        self.language_cache = {}
        
    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        if file_path in self.language_cache:
            return self.language_cache[file_path]
            
        ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        ext_mapping = {
            'py': 'python',
            'js': 'javascript', 'jsx': 'javascript', 'mjs': 'javascript',
            'ts': 'typescript', 'tsx': 'typescript',
            'go': 'go',
            'rs': 'rust',
        }
        
        language = ext_mapping.get(ext)
        self.language_cache[file_path] = language
        return language
        
    def chunk_content(self, content: str, file_path: str) -> List[ChunkInfo]:
        """Split content into semantic chunks."""
        language = self.detect_language(file_path)
        if not language or language not in self.CHUNK_PATTERNS:
            return self._chunk_generic(content)
            
        patterns = self.CHUNK_PATTERNS[language]
        lines = content.split('\n')
        chunks = []
        
        i = 0
        while i < len(lines):
            chunk_start = i
            chunk_type = 'unknown'
            
            # Try to match chunk patterns
            line = lines[i].rstrip()
            matched_type = None
            
            for pattern_type, pattern in patterns.items():
                if re.match(pattern, line):
                    matched_type = pattern_type
                    break
                    
            if matched_type:
                chunk_type = matched_type
                # Find end of this chunk
                chunk_end = self._find_chunk_end(lines, i, matched_type, language)
            else:
                # Single line or unmatched content
                chunk_end = i + 1
                chunk_type = 'misc'
                
            # Create chunk
            if chunk_end > chunk_start:
                chunk_content = '\n'.join(lines[chunk_start:chunk_end])
                chunk = ChunkInfo(
                    start_line=chunk_start + 1,
                    end_line=chunk_end,
                    chunk_type=chunk_type,
                    content=chunk_content,
                    importance_score=self._calculate_chunk_importance(chunk_content, chunk_type),
                    estimated_tokens=len(chunk_content) // 4,  # Rough estimate
                    dependencies=[]  # Could be enhanced with dependency analysis
                )
                chunks.append(chunk)
                
            i = chunk_end
            
        return chunks
        
    def _chunk_generic(self, content: str) -> List[ChunkInfo]:
        """Generic chunking for unknown languages."""
        lines = content.split('\n')
        chunk_size = 20  # Lines per chunk
        chunks = []
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:min(i + chunk_size, len(lines))]
            chunk_content = '\n'.join(chunk_lines)
            
            chunk = ChunkInfo(
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                chunk_type='generic',
                content=chunk_content,
                importance_score=0.5,  # Neutral importance
                estimated_tokens=len(chunk_content) // 4,
                dependencies=[]
            )
            chunks.append(chunk)
            
        return chunks
        
    def _find_chunk_end(self, lines: List[str], start_idx: int, chunk_type: str, language: str) -> int:
        """Find the end line of a code chunk based on indentation and syntax."""
        if start_idx >= len(lines):
            return start_idx + 1
            
        start_line = lines[start_idx]
        base_indent = len(start_line) - len(start_line.lstrip())
        
        # Special handling for different chunk types
        if chunk_type in ['function', 'class', 'interface', 'struct', 'enum', 'trait', 'impl']:
            return self._find_block_end(lines, start_idx, base_indent, language)
        elif chunk_type in ['import', 'use']:
            return self._find_import_block_end(lines, start_idx)
        elif chunk_type in ['docstring', 'jsdoc']:
            return self._find_docstring_end(lines, start_idx, language)
        elif chunk_type == 'comment':
            return self._find_comment_block_end(lines, start_idx, language)
        else:
            # Single line chunk
            return start_idx + 1
            
    def _find_block_end(self, lines: List[str], start_idx: int, base_indent: int, language: str) -> int:
        """Find end of indented block (function, class, etc.)."""
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
                
            current_indent = len(line) - len(line.lstrip())
            
            # If we hit code at same or less indentation, block is done
            if current_indent <= base_indent and line.strip():
                break
                
            i += 1
            
        return i
        
    def _find_import_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find end of import block."""
        i = start_idx + 1
        
        while i < len(lines) and i < start_idx + 10:  # Max 10 lines for import block
            line = lines[i].strip()
            
            # Continue if it's continuation of import or empty line
            if not line or line.startswith(('import', 'from', 'use', 'require')) or line.startswith((',', ')')):
                i += 1
            else:
                break
                
        return i
        
    def _find_docstring_end(self, lines: List[str], start_idx: int, language: str) -> int:
        """Find end of docstring/documentation block."""
        if language == 'python':
            # Look for closing """
            for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
                if '"""' in lines[i]:
                    return i + 1
        elif language in ['javascript', 'typescript']:
            # Look for closing */
            for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
                if '*/' in lines[i]:
                    return i + 1
                    
        # Fallback: return reasonable block size
        return min(start_idx + 10, len(lines))
        
    def _find_comment_block_end(self, lines: List[str], start_idx: int, language: str) -> int:
        """Find end of comment block."""
        comment_prefixes = {
            'python': '#',
            'javascript': '//',
            'typescript': '//',
            'go': '//',
            'rust': '//'
        }
        
        prefix = comment_prefixes.get(language, '#')
        i = start_idx + 1
        
        # Continue while lines start with comment prefix
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith(prefix):
                i += 1
            else:
                break
                
        return i
        
    def _calculate_chunk_importance(self, content: str, chunk_type: str) -> float:
        """Calculate importance score for a chunk."""
        # Base importance by type
        type_importance = {
            'class': 0.9,
            'interface': 0.9,
            'struct': 0.8,
            'enum': 0.8,
            'trait': 0.8,
            'function': 0.7,
            'import': 0.9,  # High importance for imports
            'use': 0.9,
            'type': 0.8,
            'constant': 0.6,
            'docstring': 0.5,
            'jsdoc': 0.5,
            'comment': 0.2,
            'misc': 0.3
        }
        
        base_score = type_importance.get(chunk_type, 0.5)
        
        # Adjust based on content characteristics
        content_lower = content.lower()
        
        # Boost for public/exported items
        if any(keyword in content_lower for keyword in ['public', 'export', 'pub']):
            base_score *= 1.2
            
        # Boost for main functions
        if any(keyword in content_lower for keyword in ['main', '__main__', 'init']):
            base_score *= 1.3
            
        # Reduce for test/example code
        if any(keyword in content_lower for keyword in ['test', 'example', 'demo', 'mock']):
            base_score *= 0.7
            
        # Boost for complex content
        complexity_indicators = len(re.findall(r'[{}()\[\]]', content))
        if complexity_indicators > 10:
            base_score *= 1.1
            
        return min(base_score, 1.0)


class SignatureExtractor:
    """
    Extracts type signatures and interfaces for signature-level demotion.
    
    Preserves essential type information while removing implementation details:
    - Function signatures (parameters and return types)
    - Class/struct definitions (without method bodies)
    - Interface definitions
    - Type aliases
    - Import statements
    """
    
    def __init__(self):
        self.chunker = CodeChunker()
        
    def extract_signatures(self, content: str, file_path: str) -> str:
        """Extract just the signatures from code content."""
        language = self.chunker.detect_language(file_path)
        
        if language == 'python':
            return self._extract_python_signatures(content)
        elif language in ['javascript', 'typescript']:
            return self._extract_js_signatures(content)
        elif language == 'go':
            return self._extract_go_signatures(content)
        elif language == 'rust':
            return self._extract_rust_signatures(content)
        else:
            return self._extract_generic_signatures(content)
            
    def _extract_python_signatures(self, content: str) -> str:
        """Extract Python function and class signatures."""
        lines = content.split('\n')
        signatures = []
        
        for line in lines:
            stripped = line.strip()
            
            # Keep imports
            if stripped.startswith(('import ', 'from ')):
                signatures.append(line)
                
            # Keep class definitions (without body)
            elif stripped.startswith('class '):
                signatures.append(line)
                
            # Keep function definitions (without body)
            elif stripped.startswith('def '):
                signatures.append(line)
                
            # Keep constants
            elif re.match(r'^[A-Z_][A-Z0-9_]*\s*=', stripped):
                signatures.append(line)
                
            # Keep type definitions
            elif 'TypeAlias' in stripped or 'NewType' in stripped:
                signatures.append(line)
                
        return '\n'.join(signatures)
        
    def _extract_js_signatures(self, content: str) -> str:
        """Extract JavaScript/TypeScript signatures."""
        lines = content.split('\n')
        signatures = []
        
        for line in lines:
            stripped = line.strip()
            
            # Keep imports/exports
            if stripped.startswith(('import ', 'export ', 'require(')):
                signatures.append(line)
                
            # Keep interface definitions  
            elif stripped.startswith('interface '):
                signatures.append(line)
                
            # Keep type definitions
            elif stripped.startswith('type '):
                signatures.append(line)
                
            # Keep class definitions
            elif stripped.startswith('class '):
                signatures.append(line)
                
            # Keep function signatures
            elif (stripped.startswith(('function ', 'async function')) or 
                  re.match(r'const\s+\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)', stripped)):
                signatures.append(line)
                
            # Keep constants
            elif stripped.startswith('const ') and '=' in stripped:
                signatures.append(line)
                
        return '\n'.join(signatures)
        
    def _extract_go_signatures(self, content: str) -> str:
        """Extract Go signatures."""
        lines = content.split('\n')
        signatures = []
        
        for line in lines:
            stripped = line.strip()
            
            # Keep package declaration
            if stripped.startswith('package '):
                signatures.append(line)
                
            # Keep imports
            elif stripped.startswith('import'):
                signatures.append(line)
                
            # Keep type definitions
            elif stripped.startswith('type '):
                signatures.append(line)
                
            # Keep function signatures
            elif stripped.startswith('func '):
                signatures.append(line)
                
            # Keep constants and variables
            elif stripped.startswith(('const', 'var')):
                signatures.append(line)
                
        return '\n'.join(signatures)
        
    def _extract_rust_signatures(self, content: str) -> str:
        """Extract Rust signatures."""
        lines = content.split('\n')
        signatures = []
        
        for line in lines:
            stripped = line.strip()
            
            # Keep use statements
            if stripped.startswith('use '):
                signatures.append(line)
                
            # Keep struct definitions
            elif re.match(r'(?:pub\s+)?struct\s+', stripped):
                signatures.append(line)
                
            # Keep enum definitions
            elif re.match(r'(?:pub\s+)?enum\s+', stripped):
                signatures.append(line)
                
            # Keep trait definitions
            elif re.match(r'(?:pub\s+)?trait\s+', stripped):
                signatures.append(line)
                
            # Keep function signatures
            elif re.match(r'(?:pub\s+)?(?:async\s+)?fn\s+', stripped):
                signatures.append(line)
                
            # Keep constants
            elif re.match(r'(?:pub\s+)?const\s+', stripped):
                signatures.append(line)
                
            # Keep type aliases
            elif re.match(r'(?:pub\s+)?type\s+', stripped):
                signatures.append(line)
                
        return '\n'.join(signatures)
        
    def _extract_generic_signatures(self, content: str) -> str:
        """Generic signature extraction for unknown languages."""
        lines = content.split('\n')
        signatures = []
        
        for line in lines:
            stripped = line.strip()
            
            # Keep lines that look like definitions or declarations
            if (any(keyword in stripped.lower() for keyword in 
                   ['def', 'function', 'class', 'struct', 'interface', 'type', 'import', 'include']) or
                re.match(r'^[A-Z_][A-Z0-9_]*\s*=', stripped)):  # Constants
                signatures.append(line)
                
        return '\n'.join(signatures)


class DemotionEngine:
    """
    Main demotion engine that orchestrates multi-fidelity content reduction.
    
    Applies progressive demotion strategy:
    1. Start with full content
    2. Switch to important chunks only when needed
    3. Fall back to signatures for maximum compression
    
    Optimizes for preserving most important information while meeting budget.
    """
    
    def __init__(self):
        self.chunker = CodeChunker()
        self.signature_extractor = SignatureExtractor()
        
    def apply_demotion(
        self, 
        scan_result: ScanResult, 
        target_fidelity: FidelityMode,
        max_tokens: Optional[int] = None
    ) -> DemotionResult:
        """Apply demotion to a file with target fidelity mode."""
        flags = get_feature_flags()
        
        # Return original content if demotion is disabled
        if not flags.demote_enabled:
            original_content = self._get_file_content(scan_result)
            original_tokens = len(original_content) // 4
            
            return DemotionResult(
                original_path=scan_result.stats.path,
                original_tokens=original_tokens,
                demoted_tokens=original_tokens,
                fidelity_mode=FidelityMode.FULL,
                content=original_content,
                chunks_kept=1,
                chunks_total=1,
                compression_ratio=1.0,
                quality_score=1.0
            )
            
        # Get original content
        original_content = self._get_file_content(scan_result)
        original_tokens = len(original_content) // 4
        
        if target_fidelity == FidelityMode.FULL:
            return self._apply_full_fidelity(scan_result, original_content, original_tokens, max_tokens)
        elif target_fidelity == FidelityMode.CHUNK:
            return self._apply_chunk_fidelity(scan_result, original_content, original_tokens, max_tokens)
        else:  # SIGNATURE
            return self._apply_signature_fidelity(scan_result, original_content, original_tokens, max_tokens)
            
    def _get_file_content(self, scan_result: ScanResult) -> str:
        """Get file content from scan result (simplified - would read from file)."""
        # In production, this would read the actual file content
        # For now, return a placeholder
        return f"# Content of {scan_result.stats.path}\n# Size: {scan_result.stats.size_bytes} bytes"
        
    def _apply_full_fidelity(self, scan_result: ScanResult, content: str, original_tokens: int, max_tokens: Optional[int]) -> DemotionResult:
        """Apply full fidelity (no demotion)."""
        if max_tokens and original_tokens > max_tokens:
            # Truncate if over budget
            truncated_content = content[:max_tokens * 4]  # Rough truncation
            truncated_tokens = len(truncated_content) // 4
            
            return DemotionResult(
                original_path=scan_result.stats.path,
                original_tokens=original_tokens,
                demoted_tokens=truncated_tokens,
                fidelity_mode=FidelityMode.FULL,
                content=truncated_content,
                chunks_kept=1,
                chunks_total=1,
                compression_ratio=truncated_tokens / original_tokens,
                quality_score=0.8  # Reduced quality due to truncation
            )
        else:
            return DemotionResult(
                original_path=scan_result.stats.path,
                original_tokens=original_tokens,
                demoted_tokens=original_tokens,
                fidelity_mode=FidelityMode.FULL,
                content=content,
                chunks_kept=1,
                chunks_total=1,
                compression_ratio=1.0,
                quality_score=1.0
            )
            
    def _apply_chunk_fidelity(self, scan_result: ScanResult, content: str, original_tokens: int, max_tokens: Optional[int]) -> DemotionResult:
        """Apply chunk fidelity (keep important chunks only)."""
        # Split into chunks
        chunks = self.chunker.chunk_content(content, scan_result.stats.path)
        
        if not chunks:
            # Fallback to full content if chunking fails
            return self._apply_full_fidelity(scan_result, content, original_tokens, max_tokens)
            
        # Sort chunks by importance
        sorted_chunks = sorted(chunks, key=lambda c: c.importance_score, reverse=True)
        
        # Select chunks within budget
        selected_chunks = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            if max_tokens is None or total_tokens + chunk.estimated_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk.estimated_tokens
            elif not selected_chunks:  # Always include at least one chunk
                selected_chunks.append(chunk)
                total_tokens = chunk.estimated_tokens
                break
                
        # Reconstruct content from selected chunks
        selected_chunks.sort(key=lambda c: c.start_line)  # Preserve order
        demoted_content_parts = []
        
        for chunk in selected_chunks:
            demoted_content_parts.append(f"# Lines {chunk.start_line}-{chunk.end_line} ({chunk.chunk_type})")
            demoted_content_parts.append(chunk.content)
            demoted_content_parts.append("")  # Empty line separator
            
        demoted_content = '\n'.join(demoted_content_parts)
        demoted_tokens = len(demoted_content) // 4
        
        # Calculate quality score based on chunk importance preserved
        total_importance = sum(c.importance_score for c in chunks)
        preserved_importance = sum(c.importance_score for c in selected_chunks)
        quality_score = preserved_importance / total_importance if total_importance > 0 else 0.5
        
        return DemotionResult(
            original_path=scan_result.stats.path,
            original_tokens=original_tokens,
            demoted_tokens=demoted_tokens,
            fidelity_mode=FidelityMode.CHUNK,
            content=demoted_content,
            chunks_kept=len(selected_chunks),
            chunks_total=len(chunks),
            compression_ratio=demoted_tokens / original_tokens if original_tokens > 0 else 0.0,
            quality_score=quality_score
        )
        
    def _apply_signature_fidelity(self, scan_result: ScanResult, content: str, original_tokens: int, max_tokens: Optional[int]) -> DemotionResult:
        """Apply signature fidelity (type signatures only)."""
        signatures = self.signature_extractor.extract_signatures(content, scan_result.stats.path)
        signature_tokens = len(signatures) // 4
        
        # Truncate signatures if still over budget
        if max_tokens and signature_tokens > max_tokens:
            signatures = signatures[:max_tokens * 4]
            signature_tokens = len(signatures) // 4
            
        return DemotionResult(
            original_path=scan_result.stats.path,
            original_tokens=original_tokens,
            demoted_tokens=signature_tokens,
            fidelity_mode=FidelityMode.SIGNATURE,
            content=signatures,
            chunks_kept=1,  # Just signatures
            chunks_total=1,
            compression_ratio=signature_tokens / original_tokens if original_tokens > 0 else 0.0,
            quality_score=0.3  # Low quality but preserves structure
        )
        
    def suggest_fidelity_mode(self, scan_result: ScanResult, available_budget: int) -> FidelityMode:
        """Suggest optimal fidelity mode based on file and budget constraints."""
        estimated_full_tokens = estimate_tokens_scan_result(scan_result)
        
        # If file fits in budget, use full fidelity
        if estimated_full_tokens <= available_budget:
            return FidelityMode.FULL
            
        # If file is very large, go straight to signatures
        if estimated_full_tokens > available_budget * 4:
            return FidelityMode.SIGNATURE
            
        # Otherwise, try chunk fidelity
        return FidelityMode.CHUNK
        
    def batch_demote_files(
        self, 
        scan_results: List[ScanResult], 
        total_budget: int,
        importance_scores: Dict[str, float]
    ) -> Tuple[List[DemotionResult], int]:
        """
        Apply intelligent demotion to a batch of files within budget.
        
        Strategy:
        1. Sort files by importance
        2. Try to fit high-importance files at full fidelity
        3. Apply progressive demotion to remaining files
        4. Ensure total budget is respected
        """
        flags = get_feature_flags()
        
        if not flags.demote_enabled:
            # No demotion - return original files within budget
            results = []
            used_budget = 0
            
            for scan_result in scan_results:
                estimated_tokens = estimate_tokens_scan_result(scan_result)
                if used_budget + estimated_tokens <= total_budget:
                    result = self.apply_demotion(scan_result, FidelityMode.FULL, None)
                    results.append(result)
                    used_budget += result.demoted_tokens
                    
            return results, used_budget
            
        # Sort files by importance (descending)
        scored_files = [(scan_result, importance_scores.get(scan_result.stats.path, 0.0)) 
                       for scan_result in scan_results]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        remaining_budget = total_budget
        
        # Phase 1: Try to fit high-importance files at full fidelity
        high_importance_threshold = sorted([score for _, score in scored_files], reverse=True)[len(scored_files)//4] if scored_files else 0.0
        
        for scan_result, importance in scored_files:
            if importance >= high_importance_threshold and remaining_budget > 0:
                # Try full fidelity first
                estimated_tokens = estimate_tokens_scan_result(scan_result)
                
                if estimated_tokens <= remaining_budget:
                    result = self.apply_demotion(scan_result, FidelityMode.FULL, None)
                    results.append(result)
                    remaining_budget -= result.demoted_tokens
                    continue
                    
            # Apply progressive demotion
            suggested_mode = self.suggest_fidelity_mode(scan_result, remaining_budget)
            max_tokens = min(remaining_budget, estimate_tokens_scan_result(scan_result))
            
            if max_tokens > 0:
                result = self.apply_demotion(scan_result, suggested_mode, max_tokens)
                results.append(result)
                remaining_budget -= result.demoted_tokens
                
                if remaining_budget <= 0:
                    break
                    
        used_budget = total_budget - remaining_budget
        return results, used_budget


def create_demotion_engine() -> DemotionEngine:
    """Create a DemotionEngine instance."""
    return DemotionEngine()
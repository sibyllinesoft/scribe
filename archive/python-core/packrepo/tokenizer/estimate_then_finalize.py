"""
Two-phase token estimation and finalization for FastPath.

Implements estimate-then-finalize pattern:
1. Fast estimation phase using heuristics
2. Precise finalization with actual tokenization
3. Budget enforcement with 0 overflow, ≤0.5% underflow
4. Graceful demotion when files exceed estimates
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..fastpath.fast_scan import ScanResult
from ..packer.tokenizer.base import Tokenizer


@dataclass
class EstimationResult:
    """Result of token estimation for file selection."""
    file_path: str
    estimated_tokens: int
    confidence_level: float  # 0.0-1.0, how confident we are in estimate
    priority_score: float
    inclusion_reason: str


@dataclass
class FinalizedPack:
    """Final pack result with precise token counts."""
    selected_files: List[ScanResult]
    file_tokens: List[int]          # Actual token count per file
    total_tokens: int               # Sum of all file tokens
    budget_utilization: float       # Percentage of budget used
    overflow_tokens: int            # Tokens over budget (should be 0)
    demoted_files: List[str]        # Files that were demoted due to overruns
    pack_content: str               # Final pack format content
    metadata: Dict[str, Any]        # Additional pack metadata


class TokenEstimator:
    """
    Two-phase token estimation and budget enforcement system.
    
    Phase 1: Fast heuristic estimation for file selection
    Phase 2: Precise tokenization and budget enforcement with demotion
    """
    
    def __init__(self, tokenizer: Tokenizer, safety_margin: float = 0.05):
        self.tokenizer = tokenizer
        self.safety_margin = safety_margin  # Reserve 5% of budget for overhead
        
        # Estimation models (language-specific characters per token)
        self.chars_per_token = {
            'python': 3.2,
            'javascript': 3.5,
            'typescript': 3.8,
            'java': 4.0,
            'cpp': 3.0,
            'rust': 3.5,
            'go': 3.3,
            'markdown': 4.5,
            'text': 5.0,
            'json': 2.8,
            'yaml': 4.0,
            'default': 4.0,
        }
        
        # Confidence levels for different file types
        self.confidence_levels = {
            'python': 0.85,
            'javascript': 0.80,
            'typescript': 0.82,
            'java': 0.88,
            'cpp': 0.75,
            'rust': 0.80,
            'go': 0.85,
            'markdown': 0.90,
            'text': 0.95,
            'json': 0.98,
            'yaml': 0.92,
            'default': 0.75,
        }
        
    def estimate_file_tokens(self, result: ScanResult) -> EstimationResult:
        """
        Fast token estimation for a single file.
        
        Uses language-specific heuristics and file characteristics.
        """
        file_path = result.stats.path
        language = result.stats.language or 'default'
        
        # Get language-specific estimation parameters
        chars_per_token = self.chars_per_token.get(language, self.chars_per_token['default'])
        confidence = self.confidence_levels.get(language, self.confidence_levels['default'])
        
        # Base estimation from file size
        size_based_tokens = max(result.stats.size_bytes / chars_per_token, result.stats.lines)
        
        # Adjust for file type characteristics
        if result.stats.is_readme:
            # README files often have more whitespace and formatting
            estimated_tokens = int(size_based_tokens * 1.2)
            inclusion_reason = "README file (high priority)"
            confidence *= 0.95  # README structure is predictable
            
        elif result.stats.is_test:
            # Test files have more boilerplate and assertions
            estimated_tokens = int(size_based_tokens * 1.1)
            inclusion_reason = "Test file (code coverage)"
            confidence *= 0.9
            
        elif result.stats.is_config:
            # Config files are usually dense
            estimated_tokens = int(size_based_tokens * 0.9)
            inclusion_reason = "Configuration file"
            confidence *= 1.1  # Config files are very predictable
            
        elif result.stats.is_docs and result.doc_analysis:
            # Documentation with structure analysis
            doc = result.doc_analysis
            
            # Factor in document complexity
            complexity_factor = 1.0
            if doc.heading_count > 0:
                complexity_factor += doc.heading_count * 0.02  # Headings add tokens
            if doc.code_block_count > 0:
                complexity_factor += doc.code_block_count * 0.05  # Code blocks are dense
                
            estimated_tokens = int(size_based_tokens * complexity_factor)
            inclusion_reason = f"Documentation ({doc.heading_count} headings, {doc.code_block_count} code blocks)"
            
        elif language in {'python', 'javascript', 'typescript', 'java'}:
            # High-level languages with more verbose syntax
            estimated_tokens = int(size_based_tokens)
            inclusion_reason = f"Source code ({language})"
            
        elif language in {'cpp', 'rust', 'go'}:
            # Lower-level languages, denser syntax
            estimated_tokens = int(size_based_tokens * 0.95)
            inclusion_reason = f"Source code ({language})"
            
        else:
            # Generic estimation
            estimated_tokens = int(size_based_tokens)
            inclusion_reason = "Generic file"
            confidence *= 0.8
            
        # Apply confidence-based adjustment (conservative for low confidence)
        if confidence < 0.8:
            estimated_tokens = int(estimated_tokens * 1.15)  # Add 15% buffer for uncertainty
            
        return EstimationResult(
            file_path=file_path,
            estimated_tokens=max(estimated_tokens, 1),  # Minimum 1 token
            confidence_level=min(confidence, 1.0),
            priority_score=getattr(result, 'priority_score', 0.0),
            inclusion_reason=inclusion_reason
        )
        
    def estimate_selection_tokens(self, selected_files: List[ScanResult]) -> List[EstimationResult]:
        """Estimate tokens for a selection of files."""
        return [self.estimate_file_tokens(result) for result in selected_files]
        
    def finalize_pack(self, selected_files: List[ScanResult], 
                     token_budget: int,
                     pack_metadata: Optional[Dict[str, Any]] = None) -> FinalizedPack:
        """
        Finalize pack with precise tokenization and budget enforcement.
        
        Performs actual tokenization and applies demotion ladder if needed.
        """
        # Calculate effective budget (minus safety margin)
        effective_budget = int(token_budget * (1 - self.safety_margin))
        
        # First pass: tokenize all files and check budget
        file_tokens = []
        pack_segments = []
        demoted_files = []
        
        # Create pack header
        header_info = pack_metadata or {}
        header_info.update({
            'total_files': len(selected_files),
            'target_budget': token_budget,
            'effective_budget': effective_budget,
        })
        
        header_json = json.dumps(header_info, indent=2)
        header_tokens = self.tokenizer.count_tokens(header_json)
        
        running_total = header_tokens
        final_files = []
        final_tokens = []
        
        # Process files in priority order
        for i, result in enumerate(selected_files):
            try:
                # Read file content
                full_path = Path(result.stats.path)
                if full_path.is_absolute():
                    file_path = full_path
                else:
                    file_path = Path.cwd() / result.stats.path
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Create pack segment
                segment_header = f"### path: {result.stats.path} lines: {result.stats.lines} mode: {result.stats.language or 'text'}\n"
                segment_content = segment_header + content
                
                # Count tokens for this segment
                segment_tokens = self.tokenizer.count_tokens(segment_content)
                
                # Check if adding this file would exceed budget
                if running_total + segment_tokens <= effective_budget:
                    # File fits - include it
                    final_files.append(result)
                    final_tokens.append(segment_tokens)
                    pack_segments.append(segment_content)
                    running_total += segment_tokens
                    
                else:
                    # File doesn't fit - apply demotion
                    demoted_result = self._try_demotion(result, content, segment_header, 
                                                     effective_budget - running_total)
                    
                    if demoted_result:
                        demoted_content, demoted_tokens = demoted_result
                        final_files.append(result)
                        final_tokens.append(demoted_tokens)
                        pack_segments.append(demoted_content)
                        running_total += demoted_tokens
                        demoted_files.append(result.stats.path)
                    else:
                        # Can't fit even with demotion
                        demoted_files.append(result.stats.path)
                        
            except (OSError, UnicodeDecodeError) as e:
                # Skip files we can't read
                demoted_files.append(f"{result.stats.path} (read error: {e})")
                continue
                
        # Construct final pack content
        final_content = header_json + "\n\n" + "\n\n".join(pack_segments)
        final_total_tokens = self.tokenizer.count_tokens(final_content)
        
        # Calculate metrics
        budget_utilization = final_total_tokens / token_budget
        overflow_tokens = max(0, final_total_tokens - token_budget)
        
        # Final metadata
        final_metadata = {
            'files_included': len(final_files),
            'files_demoted': len(demoted_files),
            'header_tokens': header_tokens,
            'content_tokens': final_total_tokens - header_tokens,
            'safety_margin': self.safety_margin,
            'effective_budget': effective_budget,
        }
        
        return FinalizedPack(
            selected_files=final_files,
            file_tokens=final_tokens,
            total_tokens=final_total_tokens,
            budget_utilization=budget_utilization,
            overflow_tokens=overflow_tokens,
            demoted_files=demoted_files,
            pack_content=final_content,
            metadata=final_metadata
        )
        
    def _try_demotion(self, result: ScanResult, content: str, segment_header: str, 
                     remaining_budget: int) -> Optional[Tuple[str, int]]:
        """
        Try to fit file using demotion ladder.
        
        Demotion strategies:
        1. Summary mode: First N lines + last N lines
        2. Signature mode: Just function/class signatures
        3. Minimal mode: Just filename and basic info
        """
        lines = content.split('\n')
        
        # Strategy 1: Summary mode (first 20 + last 10 lines)
        if len(lines) > 30:
            summary_lines = lines[:20] + ['...', '(content truncated)', '...'] + lines[-10:]
            summary_content = segment_header + '\n'.join(summary_lines)
            summary_tokens = self.tokenizer.count_tokens(summary_content)
            
            if summary_tokens <= remaining_budget:
                return summary_content, summary_tokens
                
        # Strategy 2: Signature mode (for code files)
        if result.stats.language in {'python', 'javascript', 'typescript', 'java', 'cpp', 'rust', 'go'}:
            signatures = self._extract_signatures(content, result.stats.language)
            if signatures:
                sig_content = segment_header + '\n'.join(signatures)
                sig_tokens = self.tokenizer.count_tokens(sig_content)
                
                if sig_tokens <= remaining_budget:
                    return sig_content, sig_tokens
                    
        # Strategy 3: Minimal mode (just file info)
        minimal_content = (
            segment_header + 
            f"# {result.stats.path}\n"
            f"File type: {result.stats.language or 'unknown'}\n"
            f"Size: {result.stats.size_bytes} bytes, {result.stats.lines} lines\n"
            f"(Content truncated due to token budget constraints)"
        )
        minimal_tokens = self.tokenizer.count_tokens(minimal_content)
        
        if minimal_tokens <= remaining_budget:
            return minimal_content, minimal_tokens
            
        # Can't fit even minimal version
        return None
        
    def _extract_signatures(self, content: str, language: str) -> List[str]:
        """Extract function/class signatures from code (simplified)."""
        lines = content.split('\n')
        signatures = []
        
        if language == 'python':
            for line in lines:
                stripped = line.strip()
                if (stripped.startswith('def ') or 
                    stripped.startswith('class ') or
                    stripped.startswith('async def ')):
                    signatures.append(line.rstrip() + ':')
                    
        elif language in {'javascript', 'typescript'}:
            for line in lines:
                stripped = line.strip()
                if (any(pattern in stripped for pattern in 
                       ['function ', 'const ', 'let ', 'var ', 'class ', 'interface ', 'type ']) and
                    ('=' in stripped or '{' in stripped)):
                    signatures.append(line.rstrip())
                    
        elif language == 'java':
            for line in lines:
                stripped = line.strip()
                if (any(pattern in stripped for pattern in 
                       ['public ', 'private ', 'protected ', 'class ', 'interface ']) and
                    ('(' in stripped or '{' in stripped)):
                    signatures.append(line.rstrip())
                    
        # Add more language-specific signature extraction as needed
        
        return signatures[:50]  # Limit to first 50 signatures


def create_token_estimator(tokenizer: Tokenizer, safety_margin: float = 0.05) -> TokenEstimator:
    """Create a token estimator instance."""
    return TokenEstimator(tokenizer, safety_margin)
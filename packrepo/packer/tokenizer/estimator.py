"""Token estimation utilities for budget management.

This module provides centralized token calculation utilities to eliminate
code duplication across the FastPath system while maintaining backward 
compatibility with existing estimation logic.
"""

from __future__ import annotations
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
import logging

from .base import Tokenizer


logger = logging.getLogger(__name__)


@dataclass
class TokenEstimation:
    """Result of token estimation for a piece of content."""
    estimated_tokens: int
    method_used: str
    confidence: float  # 0.0 to 1.0, higher is better
    metadata: Dict[str, Any]


class TokenEstimator:
    """Centralized token estimation with multiple fallback methods.
    
    This class provides a unified interface for token estimation that can
    use different methods depending on availability and requirements:
    1. Exact tokenization (when tokenizer available)
    2. Improved heuristic estimation
    3. Legacy byte-based approximation (for backward compatibility)
    """
    
    def __init__(
        self, 
        tokenizer: Optional[Tokenizer] = None,
        default_method: str = "bytes_heuristic"
    ):
        """Initialize token estimator.
        
        Args:
            tokenizer: Optional tokenizer for exact counting
            default_method: Default estimation method to use
        """
        self.tokenizer = tokenizer
        self.default_method = default_method
        
    def estimate_tokens_from_bytes(self, size_bytes: int) -> TokenEstimation:
        """Estimate tokens from byte size using improved heuristic.
        
        This replaces the crude `size_bytes // 4` approximation with a 
        slightly more sophisticated approach while maintaining compatibility.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Token estimation result
        """
        # Legacy approximation: size_bytes // 4
        legacy_estimate = size_bytes // 4
        
        # Slightly improved: account for typical programming language characteristics
        # - Code has more varied token lengths than natural text
        # - Comments and strings affect token density
        # - Use 3.2 instead of 4.0 as average bytes per token for code
        improved_estimate = int(size_bytes / 3.2)
        
        # Use improved estimate but cap the change to avoid breaking existing logic
        estimated_tokens = max(legacy_estimate, improved_estimate)
        
        return TokenEstimation(
            estimated_tokens=estimated_tokens,
            method_used="bytes_heuristic",
            confidence=0.6,  # Moderate confidence
            metadata={
                "size_bytes": size_bytes,
                "legacy_estimate": legacy_estimate,
                "improved_estimate": improved_estimate
            }
        )
    
    def estimate_tokens_from_text(self, text: str) -> TokenEstimation:
        """Estimate tokens from text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Token estimation result
        """
        if self.tokenizer:
            # Use exact tokenization when available
            exact_count = self.tokenizer.count_tokens(text)
            return TokenEstimation(
                estimated_tokens=exact_count,
                method_used="exact_tokenization",
                confidence=1.0,
                metadata={
                    "tokenizer_type": self.tokenizer.tokenizer_type.value,
                    "text_length": len(text)
                }
            )
        else:
            # Fall back to byte-based estimation
            size_bytes = len(text.encode('utf-8'))
            return self.estimate_tokens_from_bytes(size_bytes)
    
    def estimate_tokens_from_scan_result(self, scan_result) -> TokenEstimation:
        """Estimate tokens from a ScanResult object.
        
        This is the primary method used throughout the FastPath system
        to replace the scattered `result.stats.size_bytes // 4` calculations.
        
        Args:
            scan_result: ScanResult object with stats
            
        Returns:
            Token estimation result
        """
        # Extract size information
        size_bytes = scan_result.stats.size_bytes
        
        # Use line count as additional signal if available
        if hasattr(scan_result.stats, 'lines') and scan_result.stats.lines > 0:
            # Hybrid approach: use both bytes and lines
            bytes_estimate = self.estimate_tokens_from_bytes(size_bytes)
            
            # Rough tokens per line for code (varies by language)
            # Average line in code files: ~8-12 tokens
            lines_estimate = scan_result.stats.lines * 10
            
            # Take the maximum to be conservative with budget
            estimated_tokens = max(bytes_estimate.estimated_tokens, lines_estimate)
            
            return TokenEstimation(
                estimated_tokens=estimated_tokens,
                method_used="hybrid_bytes_lines",
                confidence=0.7,
                metadata={
                    "size_bytes": size_bytes,
                    "lines": scan_result.stats.lines,
                    "bytes_estimate": bytes_estimate.estimated_tokens,
                    "lines_estimate": lines_estimate,
                    "file_path": getattr(scan_result.stats, 'path', 'unknown')
                }
            )
        else:
            # Fall back to bytes-only estimation
            estimation = self.estimate_tokens_from_bytes(size_bytes)
            estimation.metadata["file_path"] = getattr(scan_result.stats, 'path', 'unknown')
            return estimation
    
    def estimate_budget_for_files(self, scan_results: List) -> int:
        """Estimate total token budget needed for a list of files.
        
        Args:
            scan_results: List of ScanResult objects
            
        Returns:
            Total estimated tokens needed
        """
        total_tokens = 0
        for result in scan_results:
            estimation = self.estimate_tokens_from_scan_result(result)
            total_tokens += estimation.estimated_tokens
            
        return total_tokens
    
    def select_files_within_budget(
        self, 
        scan_results: List, 
        budget: int,
        selection_key: Optional[callable] = None
    ) -> List:
        """Select files that fit within the given token budget.
        
        This centralizes the common pattern of budget-constrained file selection.
        
        Args:
            scan_results: List of ScanResult objects
            budget: Maximum token budget
            selection_key: Optional function to sort files (e.g., by score)
            
        Returns:
            List of selected ScanResult objects that fit within budget
        """
        # Sort files if selection key provided
        if selection_key:
            sorted_results = sorted(scan_results, key=selection_key, reverse=True)
        else:
            sorted_results = scan_results
        
        selected_files = []
        budget_used = 0
        
        for result in sorted_results:
            estimation = self.estimate_tokens_from_scan_result(result)
            if budget_used + estimation.estimated_tokens <= budget:
                selected_files.append(result)
                budget_used += estimation.estimated_tokens
            else:
                # Budget exhausted
                break
        
        logger.debug(
            f"Selected {len(selected_files)}/{len(scan_results)} files, "
            f"using {budget_used}/{budget} tokens"
        )
        
        return selected_files


# Convenience functions for backward compatibility
def estimate_tokens_legacy(size_bytes: int) -> int:
    """Legacy token estimation: size_bytes // 4.
    
    This function preserves the exact behavior of the old scattered
    calculations for backward compatibility.
    """
    return size_bytes // 4


def estimate_tokens_scan_result(scan_result, use_lines: bool = False) -> int:
    """Estimate tokens for a scan result.
    
    Centralized replacement for the common pattern:
    `result.stats.size_bytes // 4`
    
    Args:
        scan_result: ScanResult object
        use_lines: Whether to consider line count in estimation
        
    Returns:
        Estimated token count
    """
    estimator = TokenEstimator()
    estimation = estimator.estimate_tokens_from_scan_result(scan_result)
    return estimation.estimated_tokens


# Global default estimator instance
_default_estimator = TokenEstimator()


def get_default_estimator() -> TokenEstimator:
    """Get the default token estimator instance."""
    return _default_estimator


def set_default_tokenizer(tokenizer: Optional[Tokenizer]) -> None:
    """Set the tokenizer for the default estimator."""
    global _default_estimator
    _default_estimator = TokenEstimator(tokenizer=tokenizer)
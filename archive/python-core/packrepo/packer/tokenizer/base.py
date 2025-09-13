"""Base tokenizer interface and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any


class TokenizerType(Enum):
    """Supported tokenizer types."""
    
    CL100K_BASE = "cl100k_base"  # GPT-4 tokenizer
    O200K_BASE = "o200k_base"    # GPT-4o tokenizer
    

class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    
    Provides a unified interface for different tokenization schemes
    to support accurate token counting for budget management.
    """
    
    def __init__(self, tokenizer_type: TokenizerType):
        self.tokenizer_type = tokenizer_type
        self._encoder = None
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text into tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod 
    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encode(text))
    
    def estimate_tokens(self, text: str) -> int:
        """
        Fast estimation of token count without full encoding.
        
        Default implementation uses full encoding, but subclasses
        can override with faster approximations if available.
        
        Args:
            text: Input text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        return self.count_tokens(text)
    
    def validate_budget(self, texts: List[str], budget: int) -> tuple[bool, int]:
        """
        Validate that a list of texts fits within token budget.
        
        Args:
            texts: List of text strings to validate
            budget: Maximum allowed tokens
            
        Returns:
            Tuple of (fits_in_budget, total_tokens)
        """
        total_tokens = sum(self.count_tokens(text) for text in texts)
        return total_tokens <= budget, total_tokens
    
    def fit_to_budget(self, texts: List[str], budget: int) -> List[str]:
        """
        Select texts that fit within the given budget.
        
        Selects texts in order until budget would be exceeded.
        
        Args:
            texts: List of text strings to select from
            budget: Maximum allowed tokens
            
        Returns:
            List of texts that fit within budget
        """
        selected = []
        total_tokens = 0
        
        for text in texts:
            text_tokens = self.count_tokens(text)
            if total_tokens + text_tokens <= budget:
                selected.append(text)
                total_tokens += text_tokens
            else:
                break
                
        return selected
    
    def get_info(self) -> Dict[str, Any]:
        """Get tokenizer information for debugging/logging."""
        return {
            "type": self.tokenizer_type.value,
            "name": self.__class__.__name__,
        }
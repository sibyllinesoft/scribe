"""Concrete tokenizer implementations."""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from .base import Tokenizer, TokenizerType


class TikTokenTokenizer(Tokenizer):
    """Tokenizer using tiktoken library for OpenAI tokenizers."""
    
    def __init__(self, tokenizer_type: TokenizerType):
        super().__init__(tokenizer_type)
        self._encoder = None
        self._load_encoder()
    
    def _load_encoder(self):
        """Load the tiktoken encoder."""
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(self.tokenizer_type.value)
        except ImportError:
            raise ImportError(
                "tiktoken is required for OpenAI tokenizers. "
                "Install with: pip install tiktoken"
            )
    
    def encode(self, text: str) -> List[int]:
        """Encode text using tiktoken."""
        if self._encoder is None:
            self._load_encoder()
        return self._encoder.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens using tiktoken."""
        if self._encoder is None:
            self._load_encoder()
        return self._encoder.decode(tokens)
    
    def get_info(self) -> Dict[str, Any]:
        """Get tokenizer information."""
        info = super().get_info()
        info.update({
            "library": "tiktoken",
            "vocab_size": getattr(self._encoder, "n_vocab", None) if self._encoder else None,
        })
        return info


class ApproximateTokenizer(Tokenizer):
    """
    Fast approximate tokenizer using heuristics.
    
    Provides rough token estimates without external dependencies.
    Useful for fallback when tiktoken is not available.
    """
    
    # Approximate tokens per character for different tokenizers
    CHARS_PER_TOKEN = {
        TokenizerType.CL100K_BASE: 4.0,  # ~4 chars per token average
        TokenizerType.O200K_BASE: 3.8,   # Slightly more efficient
    }
    
    def __init__(self, tokenizer_type: TokenizerType):
        super().__init__(tokenizer_type)
        self.chars_per_token = self.CHARS_PER_TOKEN[tokenizer_type]
    
    def encode(self, text: str) -> List[int]:
        """
        Approximate encoding by splitting on whitespace and punctuation.
        
        Returns pseudo-token IDs for compatibility.
        """
        # Simple tokenization heuristic
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return list(range(len(tokens)))  # Return pseudo-IDs
    
    def decode(self, tokens: List[int]) -> str:
        """Decode not supported for approximate tokenizer."""
        raise NotImplementedError("Decode not supported for ApproximateTokenizer")
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count based on character length."""
        return max(1, int(len(text) / self.chars_per_token))
    
    def estimate_tokens(self, text: str) -> int:
        """Fast token estimation using character heuristic."""
        return self.count_tokens(text)
    
    def get_info(self) -> Dict[str, Any]:
        """Get tokenizer information."""
        info = super().get_info()
        info.update({
            "library": "approximate",
            "chars_per_token": self.chars_per_token,
            "note": "Approximate tokenizer using heuristics",
        })
        return info


def get_tokenizer(
    tokenizer_type: TokenizerType, 
    prefer_exact: bool = True
) -> Tokenizer:
    """
    Get a tokenizer instance.
    
    Args:
        tokenizer_type: Type of tokenizer to create
        prefer_exact: Whether to prefer exact tokenizers over approximate ones
        
    Returns:
        Tokenizer instance
        
    Raises:
        ValueError: If tokenizer type is not supported
    """
    if prefer_exact:
        try:
            return TikTokenTokenizer(tokenizer_type)
        except ImportError:
            # Fall back to approximate tokenizer
            return ApproximateTokenizer(tokenizer_type)
    else:
        return ApproximateTokenizer(tokenizer_type)


def get_available_tokenizers() -> List[TokenizerType]:
    """Get list of available tokenizer types."""
    return list(TokenizerType)


def create_tokenizer(tokenizer_name: str, model_name: str = 'gpt-4') -> Tokenizer:
    """
    Create a tokenizer instance by name.
    
    Args:
        tokenizer_name: Name of tokenizer ('tiktoken', 'approximate', etc.)
        model_name: Model name for tokenizer selection
        
    Returns:
        Tokenizer instance
    """
    # Map model names to tokenizer types
    model_to_tokenizer = {
        'gpt-4': TokenizerType.CL100K_BASE,
        'gpt-3.5-turbo': TokenizerType.CL100K_BASE,
        'text-davinci-003': TokenizerType.CL100K_BASE,
        'o1': TokenizerType.O200K_BASE,
        'o1-preview': TokenizerType.O200K_BASE,
    }
    
    tokenizer_type = model_to_tokenizer.get(model_name, TokenizerType.CL100K_BASE)
    
    if tokenizer_name.lower() in ['tiktoken', 'exact']:
        try:
            return TikTokenTokenizer(tokenizer_type)
        except ImportError:
            # Fall back to approximate if tiktoken not available
            return ApproximateTokenizer(tokenizer_type)
    elif tokenizer_name.lower() in ['approximate', 'heuristic']:
        return ApproximateTokenizer(tokenizer_type)
    else:
        # Default to get_tokenizer behavior
        return get_tokenizer(tokenizer_type, prefer_exact=True)
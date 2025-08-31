"""FastPath utility modules.

This package contains common utilities to reduce code duplication across
the FastPath system. These utilities provide standardized implementations
for common patterns like entry point processing, file pattern matching,
and error handling.
"""

from .entry_points import EntryPointConverter
from .file_patterns import FilePatternMatcher
from .error_handling import ErrorHandler

__all__ = [
    'EntryPointConverter',
    'FilePatternMatcher', 
    'ErrorHandler'
]
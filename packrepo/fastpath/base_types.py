"""Base types and data structures for FastPath.

This module contains fundamental type definitions that are imported by other modules,
preventing circular imports while centralizing core type definitions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class EntryPointSpec:
    """Specification for an entry point for personalized centrality."""
    file_path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    weight: float = 1.0
    description: Optional[str] = None

    @property
    def identifier(self) -> str:
        """Get a unique identifier for this entry point."""
        parts = [self.file_path]
        if self.class_name:
            parts.append(f"class:{self.class_name}")
        if self.function_name:
            parts.append(f"func:{self.function_name}")
        return "::".join(parts)
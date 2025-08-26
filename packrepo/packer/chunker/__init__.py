"""Tree-sitter based semantic code chunking with dependency analysis."""

from __future__ import annotations

from .base import Chunk, ChunkKind, ChunkDependency
from .chunker import CodeChunker
from .languages import LanguageSupport

__all__ = [
    "Chunk",
    "ChunkKind", 
    "ChunkDependency",
    "CodeChunker",
    "LanguageSupport",
]
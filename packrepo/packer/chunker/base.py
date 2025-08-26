"""Core data structures for code chunking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Set, Optional, Dict, Any


class ChunkKind(Enum):
    """Types of code chunks identified by semantic analysis."""
    
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    MODULE_HEADER = "module_header"
    TEST = "test"
    MAIN = "main"
    UNKNOWN = "unknown"


@dataclass
class ChunkDependency:
    """Represents a dependency relationship between chunks."""
    
    target_chunk_id: str
    dependency_type: str  # "import", "call", "inheritance", etc.
    strength: float = 1.0  # Dependency strength (0.0-1.0)


@dataclass
class Chunk:
    """
    A semantic code chunk extracted from source files.
    
    Represents a logical unit of code (function, class, etc.) with metadata
    about its location, dependencies, and characteristics for selection algorithms.
    """
    
    # Identity and location
    id: str  # Unique chunk identifier (deterministic)
    path: Path  # Absolute path to source file
    rel_path: str  # Path relative to repository root
    start_line: int  # 1-based line number (inclusive)
    end_line: int  # 1-based line number (inclusive)
    
    # Semantic information
    kind: ChunkKind
    name: str  # Function/class/variable name, or descriptive name
    language: str  # Language identifier (python, typescript, go, etc.)
    
    # Content
    content: str  # Raw source code content
    signature: Optional[str] = None  # Type signature or interface
    docstring: Optional[str] = None  # Documentation string
    
    # Dependencies and relationships
    dependencies: List[ChunkDependency] = None  # Outgoing dependencies
    reverse_dependencies: List[ChunkDependency] = None  # Incoming dependencies
    
    # Selection features
    doc_density: float = 0.0  # Documentation ratio (0.0-1.0)
    test_links: int = 0  # Number of related test files/functions
    complexity_score: float = 0.0  # Code complexity metric
    centrality_score: float = 0.0  # Graph centrality measure
    
    # Token costs for budget management
    full_tokens: int = 0  # Token count for full content
    signature_tokens: int = 0  # Token count for signature only
    summary_tokens: Optional[int] = None  # Token count for summary (if generated)
    
    def __post_init__(self):
        """Initialize empty collections if not provided."""
        if self.dependencies is None:
            self.dependencies = []
        if self.reverse_dependencies is None:
            self.reverse_dependencies = []
    
    @property
    def line_range(self) -> tuple[int, int]:
        """Get the line range as a tuple."""
        return (self.start_line, self.end_line)
    
    @property
    def line_count(self) -> int:
        """Get the number of lines in this chunk."""
        return self.end_line - self.start_line + 1
    
    @property
    def anchor(self) -> str:
        """Generate a line anchor for this chunk."""
        return f"#{self.rel_path}:{self.start_line}"
    
    def has_dependency(self, chunk_id: str) -> bool:
        """Check if this chunk depends on another chunk."""
        return any(dep.target_chunk_id == chunk_id for dep in self.dependencies)
    
    def get_dependency_strength(self, chunk_id: str) -> float:
        """Get the dependency strength to another chunk (0.0 if no dependency)."""
        for dep in self.dependencies:
            if dep.target_chunk_id == chunk_id:
                return dep.strength
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "id": self.id,
            "path": str(self.path),
            "rel_path": self.rel_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind.value,
            "name": self.name,
            "language": self.language,
            "doc_density": self.doc_density,
            "test_links": self.test_links,
            "complexity_score": self.complexity_score,
            "centrality_score": self.centrality_score,
            "full_tokens": self.full_tokens,
            "signature_tokens": self.signature_tokens,
            "summary_tokens": self.summary_tokens,
            "dependencies": [
                {
                    "target": dep.target_chunk_id,
                    "type": dep.dependency_type,
                    "strength": dep.strength
                }
                for dep in self.dependencies
            ]
        }
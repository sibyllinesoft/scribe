"""Pack format for structured repository output."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from ..chunker.base import Chunk


@dataclass
class PackSection:
    """
    A section in the pack body containing code content.
    
    Corresponds to a selected chunk with specific fidelity mode.
    """
    
    chunk_id: str
    path: str  # Relative path from repository root
    start_line: int
    end_line: int
    mode: str  # "full", "summary", "signature"
    content: str
    token_count: int
    
    def format_header(self) -> str:
        """Format the section header."""
        return f"### path: {self.path} lines: {self.start_line}-{self.end_line} mode: {self.mode}"
    
    def format_section(self) -> str:
        """Format the complete section with header and content."""
        header = self.format_header()
        return f"{header}\n\n{self.content}\n"


@dataclass
class PackIndex:
    """
    JSON index containing metadata about the packed repository.
    
    Provides structured information about the packing process,
    selected chunks, and statistics for analysis.
    """
    
    # Version and metadata
    format_version: str = "0.3"
    created_at: str = ""
    packrepo_version: str = "0.1.0"
    
    # Repository information
    repository_url: Optional[str] = None
    commit_hash: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0
    
    # Chunk statistics
    total_chunks: int = 0
    selected_chunks: int = 0
    chunk_breakdown: Dict[str, int] = None  # By chunk kind
    mode_breakdown: Dict[str, int] = None   # By fidelity mode
    
    # Token budget information
    tokenizer: str = ""
    target_budget: int = 0
    actual_tokens: int = 0
    budget_utilization: float = 0.0
    
    # Selection algorithm information
    selector_variant: str = ""
    selector_params: Dict[str, Any] = None
    
    # Quality metrics
    coverage_score: float = 0.0
    diversity_score: float = 0.0
    
    # V2 Clustering metadata (for coverage enhancement)
    clustering_method: Optional[str] = None
    clustering_stats: Optional[Dict[str, Any]] = None
    cluster_centroids: Optional[Dict[str, List[float]]] = None  # cluster_id -> centroid vector
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    
    # Chunk manifest
    chunks: List[Dict[str, Any]] = None
    
    # Manifest integrity
    manifest_digest: Optional[str] = None
    body_spans: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Initialize empty collections if not provided."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.chunk_breakdown is None:
            self.chunk_breakdown = {}
        if self.mode_breakdown is None:
            self.mode_breakdown = {}
        if self.selector_params is None:
            self.selector_params = {}
        if self.chunks is None:
            self.chunks = []
        if self.body_spans is None:
            self.body_spans = []
        # V2 clustering initialization
        if self.clustering_stats is None:
            self.clustering_stats = {}
        if self.cluster_centroids is None:
            self.cluster_centroids = {}
    
    def add_chunk(self, chunk: Chunk, mode: str, tokens: int):
        """Add a chunk to the index."""
        chunk_info = chunk.to_dict()
        chunk_info.update({
            "selected_mode": mode,
            "selected_tokens": tokens,
        })
        self.chunks.append(chunk_info)
        
        # Update statistics
        self.selected_chunks += 1
        self.actual_tokens += tokens
        
        # Update breakdowns
        kind_key = chunk.kind.value
        self.chunk_breakdown[kind_key] = self.chunk_breakdown.get(kind_key, 0) + 1
        self.mode_breakdown[mode] = self.mode_breakdown.get(mode, 0) + 1
    
    def finalize(self):
        """Finalize the index after all chunks are added."""
        if self.target_budget > 0:
            self.budget_utilization = self.actual_tokens / self.target_budget
    
    def set_clustering_metadata(self, clustering_result, embedding_provider=None):
        """
        Store V2 clustering metadata in the index.
        
        Args:
            clustering_result: ClusteringResult from mixed centroid clustering
            embedding_provider: Optional embedding provider for model info
        """
        if not clustering_result:
            return
        
        # Store clustering method and stats
        self.clustering_method = clustering_result.clustering_method
        self.clustering_stats = clustering_result.to_dict()
        
        # Store cluster centroids (convert to lists for JSON serialization)
        centroids = {}
        for cluster in clustering_result.clusters:
            centroids[cluster.cluster_id] = cluster.centroid.tolist()
        self.cluster_centroids = centroids
        
        # Store embedding model information if available
        if embedding_provider and hasattr(embedding_provider, 'get_provider_info'):
            provider_info = embedding_provider.get_provider_info()
            self.embedding_model = provider_info.get('model_name', 'unknown')
            if 'embedding_dimension' in provider_info:
                self.embedding_dimension = int(provider_info['embedding_dimension'])
        
        # Use clustering quality metrics if available
        if hasattr(clustering_result, 'coverage_uniformity'):
            # Store clustering-specific coverage score if different
            # Keep diversity_score as is since it's calculated differently
            pass
    
    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert to canonical dictionary with deterministic key ordering."""
        data = asdict(self)
        return self._order_dict_keys(data)
    
    def _order_dict_keys(self, obj: Any) -> Any:
        """Recursively order dictionary keys for deterministic output."""
        if isinstance(obj, dict):
            # Use specific key ordering for deterministic output
            ordered = OrderedDict()
            
            # Priority order for top-level keys
            key_priority = [
                'format_version', 'created_at', 'packrepo_version',
                'repository_url', 'commit_hash', 'total_files', 'processed_files',
                'total_chunks', 'selected_chunks', 'chunk_breakdown', 'mode_breakdown',
                'tokenizer', 'target_budget', 'actual_tokens', 'budget_utilization',
                'selector_variant', 'selector_params',
                'coverage_score', 'diversity_score',
                'clustering_method', 'clustering_stats', 'cluster_centroids', 
                'embedding_model', 'embedding_dimension',
                'chunks', 'manifest_digest', 'body_spans'
            ]
            
            # Add keys in priority order
            for key in key_priority:
                if key in obj:
                    ordered[key] = self._order_dict_keys(obj[key])
            
            # Add any remaining keys alphabetically
            for key in sorted(obj.keys()):
                if key not in ordered:
                    ordered[key] = self._order_dict_keys(obj[key])
                    
            return ordered
            
        elif isinstance(obj, list):
            return [self._order_dict_keys(item) for item in obj]
        else:
            return obj
    
    def generate_manifest_digest(self, body_content: str) -> str:
        """Generate deterministic manifest digest over body spans."""
        # Create canonical representation of body structure
        body_spans_data = []
        
        for chunk_info in self.chunks:
            span = {
                'chunk_id': chunk_info.get('id', ''),
                'file_path': chunk_info.get('rel_path', ''),
                'start_line': chunk_info.get('start_line', 0),
                'end_line': chunk_info.get('end_line', 0),
                'selected_mode': chunk_info.get('selected_mode', ''),
                'selected_tokens': chunk_info.get('selected_tokens', 0),
                'content_hash': chunk_info.get('content_hash', '')
            }
            body_spans_data.append(span)
        
        # Sort spans deterministically
        body_spans_data.sort(key=lambda x: (x['file_path'], x['start_line'], x['chunk_id']))
        
        # Generate digest over body spans + actual content length
        manifest_data = {
            'body_spans': body_spans_data,
            'body_length': len(body_content.encode('utf-8')),
            'total_selected_tokens': sum(span['selected_tokens'] for span in body_spans_data)
        }
        
        canonical_manifest = json.dumps(manifest_data, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(canonical_manifest.encode('utf-8')).hexdigest()
        
        # Store body spans for validation
        self.body_spans = body_spans_data
        self.manifest_digest = digest
        
        return digest
    
    def to_json(self, indent: int = 2, canonical: bool = False) -> str:
        """Convert index to JSON string."""
        if canonical:
            data = self.to_canonical_dict()
            return json.dumps(data, indent=indent, ensure_ascii=False, separators=(',', ': ') if indent else (',', ':'))
        else:
            return json.dumps(asdict(self), indent=indent, ensure_ascii=False)
    
    def validate_runtime_constraints(self) -> List[str]:
        """Validate runtime constraints and invariants."""
        errors = []
        
        # Budget constraint: 0 overflow
        if self.actual_tokens > self.target_budget:
            errors.append(f"Budget overflow: {self.actual_tokens} > {self.target_budget} tokens")
        
        # Underflow constraint: ≤0.5% allowed
        if self.target_budget > 0:
            utilization = self.actual_tokens / self.target_budget
            if utilization < 0.995:  # Less than 99.5%
                underflow = (self.target_budget - self.actual_tokens) / self.target_budget
                if underflow > 0.005:  # More than 0.5%
                    errors.append(f"Excessive underflow: {underflow:.1%} > 0.5%")
        
        # Chunk consistency
        if self.chunks:
            token_sum = sum(chunk.get('selected_tokens', 0) for chunk in self.chunks)
            if abs(token_sum - self.actual_tokens) > 1:
                errors.append(f"Token sum mismatch: chunks={token_sum}, actual={self.actual_tokens}")
            
            # Check for duplicate chunk IDs
            chunk_ids = [chunk.get('id', '') for chunk in self.chunks]
            if len(chunk_ids) != len(set(chunk_ids)):
                errors.append("Duplicate chunk IDs detected")
            
            # Validate chunk ordering (should be deterministic)
            prev_file = ""
            prev_line = 0
            for i, chunk in enumerate(self.chunks):
                file_path = chunk.get('rel_path', '')
                start_line = chunk.get('start_line', 0)
                
                if file_path < prev_file:
                    errors.append(f"Chunks not sorted by file path at index {i}")
                    break
                elif file_path == prev_file and start_line < prev_line:
                    errors.append(f"Chunks not sorted by line number in {file_path} at index {i}")
                    break
                
                prev_file = file_path
                prev_line = start_line if file_path != prev_file else start_line
        
        return errors


@dataclass
class PackBody:
    """
    Body containing the actual code sections.
    
    Contains all selected chunks formatted with headers and content.
    """
    
    sections: List[PackSection] = None
    
    def __post_init__(self):
        """Initialize empty sections if not provided."""
        if self.sections is None:
            self.sections = []
    
    def add_section(self, section: PackSection):
        """Add a section to the body."""
        self.sections.append(section)
    
    def format_body(self) -> str:
        """Format the complete body content."""
        if not self.sections:
            return ""
        
        formatted_sections = []
        for section in self.sections:
            formatted_sections.append(section.format_section())
        
        return "\n".join(formatted_sections)


class PackFormat:
    """
    Complete pack format combining index and body.
    
    Represents the final output format: JSON index + body sections.
    """
    
    def __init__(self):
        self.index = PackIndex()
        self.body = PackBody()
    
    def add_chunk_selection(
        self, 
        chunk: Chunk, 
        mode: str, 
        content: str,
        token_count: int
    ):
        """
        Add a selected chunk to the pack.
        
        Args:
            chunk: The selected chunk
            mode: Fidelity mode ("full", "summary", "signature")
            content: The actual content to include
            token_count: Token count for this content
        """
        # Add to index
        self.index.add_chunk(chunk, mode, token_count)
        
        # Add to body
        section = PackSection(
            chunk_id=chunk.id,
            path=chunk.rel_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            mode=mode,
            content=content,
            token_count=token_count
        )
        self.body.add_section(section)
    
    def finalize(self, deterministic: bool = False):
        """Finalize the pack format."""
        self.index.finalize()
        
        # Generate manifest digest if requested
        if deterministic:
            body_content = self.body.format_body()
            self.index.generate_manifest_digest(body_content)
    
    def to_string(self, deterministic: bool = False) -> str:
        """
        Convert to the complete pack format string.
        
        Args:
            deterministic: Use canonical JSON ordering and generate manifest digest
        
        Returns:
            Single UTF-8 string with JSON index + body sections.
        """
        self.finalize(deterministic=deterministic)
        
        # Format: JSON index + double newline + body
        if deterministic:
            index_json = self.index.to_json(canonical=True)
        else:
            index_json = self.index.to_json()
            
        body_content = self.body.format_body()
        
        if body_content:
            return f"{index_json}\n\n{body_content}"
        else:
            return index_json
    
    def validate_pack(self) -> List[str]:
        """Validate pack format and constraints."""
        return self.index.validate_runtime_constraints()
    
    def generate_deterministic_hash(self) -> str:
        """Generate deterministic hash of the complete pack."""
        content = self.to_string(deterministic=True)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def write_to_file(self, path: Path):
        """
        Write pack to file.
        
        Args:
            path: Output file path
        """
        content = self.to_string()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pack statistics for analysis."""
        return {
            "total_chunks": self.index.total_chunks,
            "selected_chunks": self.index.selected_chunks,
            "actual_tokens": self.index.actual_tokens,
            "budget_utilization": self.index.budget_utilization,
            "chunk_breakdown": self.index.chunk_breakdown,
            "mode_breakdown": self.index.mode_breakdown,
            "coverage_score": self.index.coverage_score,
            "diversity_score": self.index.diversity_score,
        }
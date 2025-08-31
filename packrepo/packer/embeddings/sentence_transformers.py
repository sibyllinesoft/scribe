"""
Sentence Transformers provider for V2 embeddings system.

Implements semantic embeddings generation using HuggingFace sentence-transformers
library with support for code-specific models and efficient batch processing.
"""

import hashlib
import time
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base import EmbeddingProvider, CodeEmbedding
from ..chunker.base import Chunk


class SentenceTransformersProvider(EmbeddingProvider):
    """
    Embedding provider using sentence-transformers library.
    
    Supports various pre-trained models including code-specific models
    like CodeBERT, GraphCodeBERT, and general semantic models.
    """
    
    # Model presets with their characteristics
    MODEL_PRESETS = {
        'all-MiniLM-L6-v2': {
            'dimension': 384,
            'description': 'Fast, general-purpose semantic model',
            'good_for': 'balanced performance and speed',
        },
        'all-mpnet-base-v2': {
            'dimension': 768, 
            'description': 'High-quality general semantic model',
            'good_for': 'best quality general embeddings',
        },
        'microsoft/codebert-base': {
            'dimension': 768,
            'description': 'Code-specific BERT model',
            'good_for': 'code understanding and similarity',
        },
        'microsoft/graphcodebert-base': {
            'dimension': 768,
            'description': 'Graph-aware code model',
            'good_for': 'structural code understanding',
        },
    }
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize sentence transformers provider.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cpu', 'cuda', None for auto)
            cache_folder: Folder to cache downloaded models
            normalize_embeddings: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding multiple chunks
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersProvider. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        # Load model
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder,
            )
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {e}")
        
        # Get actual embedding dimension
        test_embedding = self.model.encode(["test"], convert_to_numpy=True)
        self._embedding_dimension = test_embedding.shape[1]
        
        # Get model version info
        self._model_version = self._compute_model_version()
    
    def get_model_name(self) -> str:
        """Get the model name identifier."""
        return self.model_name
    
    def get_model_version(self) -> str:
        """Get the model version identifier."""
        return self._model_version
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of generated embeddings."""
        return self._embedding_dimension
    
    def encode_chunk(self, chunk: Chunk) -> CodeEmbedding:
        """
        Generate embedding for a single code chunk.
        
        Args:
            chunk: Code chunk to embed
            
        Returns:
            CodeEmbedding with generated embedding vector and metadata
        """
        text = self.prepare_chunk_text(chunk)
        
        # Generate embedding
        embedding_array = self.model.encode(
            [text], 
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=1,
        )[0]
        
        # Create CodeEmbedding
        content_sha = self._compute_content_sha(chunk)
        
        return CodeEmbedding(
            chunk_id=chunk.id,
            file_path=chunk.rel_path,
            content_sha=content_sha,
            embedding=embedding_array,
            model_name=self.model_name,
            model_version=self._model_version,
            created_at=time.time(),
        )
    
    def encode_chunks(self, chunks: List[Chunk]) -> List[CodeEmbedding]:
        """
        Generate embeddings for multiple code chunks efficiently.
        
        Args:
            chunks: List of code chunks to embed
            
        Returns:
            List of CodeEmbeddings in same order as input chunks
        """
        if not chunks:
            return []
        
        # Prepare texts
        texts = [self.prepare_chunk_text(chunk) for chunk in chunks]
        
        # Generate embeddings in batches
        embedding_arrays = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
            show_progress_bar=len(chunks) > 100,  # Show progress for large batches
        )
        
        # Create CodeEmbeddings
        results = []
        timestamp = time.time()
        
        for chunk, embedding_array in zip(chunks, embedding_arrays):
            content_sha = self._compute_content_sha(chunk)
            
            results.append(CodeEmbedding(
                chunk_id=chunk.id,
                file_path=chunk.rel_path,
                content_sha=content_sha,
                embedding=embedding_array,
                model_name=self.model_name,
                model_version=self._model_version,
                created_at=timestamp,
            ))
        
        return results
    
    def prepare_chunk_text(self, chunk: Chunk) -> str:
        """
        Prepare chunk text for code-specific embedding.
        
        Optimizes text preparation for code understanding by emphasizing
        structure, semantics, and relationships.
        
        Args:
            chunk: Code chunk to prepare
            
        Returns:
            Processed text ready for embedding
        """
        parts = []
        
        # Start with language and file context
        parts.append(f"[{chunk.language.upper()}]")
        if chunk.rel_path:
            file_name = chunk.rel_path.split('/')[-1]
            parts.append(f"File: {file_name}")
        
        # Add kind and name with structure context
        if chunk.kind and chunk.name:
            parts.append(f"{chunk.kind.value.title()}: {chunk.name}")
        
        # Add signature with emphasis
        if chunk.signature and chunk.signature.strip():
            cleaned_sig = chunk.signature.strip()
            if len(cleaned_sig) > 200:  # Limit signature length
                cleaned_sig = cleaned_sig[:200] + "..."
            parts.append(f"Signature: {cleaned_sig}")
        
        # Add docstring with emphasis on key information
        if chunk.docstring and chunk.docstring.strip():
            doc = chunk.docstring.strip()
            if len(doc) > 500:  # Limit docstring length
                # Try to keep first paragraph
                first_para = doc.split('\n\n')[0]
                if len(first_para) <= 500:
                    doc = first_para
                else:
                    doc = doc[:500] + "..."
            parts.append(f"Description: {doc}")
        
        # Add code content with length management
        content = chunk.content or ""
        if content:
            # For code, prioritize keeping complete logical units
            if len(content) > 1500:
                # Try to truncate at logical boundaries (lines)
                lines = content.split('\n')
                truncated_lines = []
                current_length = 0
                
                for line in lines:
                    line_length = len(line) + 1  # +1 for newline
                    if current_length + line_length > 1500:
                        break
                    truncated_lines.append(line)
                    current_length += line_length
                
                if truncated_lines:
                    content = '\n'.join(truncated_lines) + "\n..."
                else:
                    content = content[:1500] + "..."
            
            parts.append(f"Code:\n{content}")
        
        return " ".join(parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = {
            'model_name': self.model_name,
            'model_version': self._model_version,
            'embedding_dimension': self._embedding_dimension,
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size,
            'device': str(self.model.device),
        }
        
        # Add preset information if available
        if self.model_name in self.MODEL_PRESETS:
            preset = self.MODEL_PRESETS[self.model_name]
            info.update({
                'description': preset['description'],
                'good_for': preset['good_for'],
                'expected_dimension': preset['dimension'],
            })
        
        return info
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List available model presets with their characteristics."""
        return cls.MODEL_PRESETS.copy()
    
    @classmethod
    def recommend_model(cls, use_case: str = 'general') -> str:
        """
        Recommend a model based on use case.
        
        Args:
            use_case: 'fast', 'quality', 'code', 'balanced'
            
        Returns:
            Recommended model name
        """
        recommendations = {
            'fast': 'all-MiniLM-L6-v2',
            'quality': 'all-mpnet-base-v2', 
            'code': 'microsoft/codebert-base',
            'balanced': 'all-MiniLM-L6-v2',
            'general': 'all-MiniLM-L6-v2',
        }
        
        return recommendations.get(use_case, 'all-MiniLM-L6-v2')
    
    def _compute_model_version(self) -> str:
        """Compute a version identifier for the loaded model."""
        # Create version based on model name and key model properties
        version_parts = [
            self.model_name,
            str(self._embedding_dimension),
            str(self.normalize_embeddings),
        ]
        
        # Add model-specific info if available
        if hasattr(self.model, '_model_config'):
            config_str = str(self.model._model_config)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            version_parts.append(config_hash)
        
        version_str = '|'.join(version_parts)
        version_hash = hashlib.sha256(version_str.encode()).hexdigest()[:12]
        
        return f"v{version_hash}"
    
    def _compute_content_sha(self, chunk: Chunk) -> str:
        """Compute SHA-256 hash of chunk content for cache key."""
        # Include relevant chunk properties that affect embedding
        content_parts = [
            chunk.rel_path,
            str(chunk.kind.value if chunk.kind else ''),
            chunk.name or '',
            chunk.content or '',
            chunk.signature or '',
            chunk.docstring or '',
            self.model_name,
            self._model_version,
        ]
        
        content_str = '|'.join(content_parts)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()


# Convenience factory functions
def create_fast_provider(**kwargs) -> SentenceTransformersProvider:
    """Create a fast embedding provider optimized for speed."""
    defaults = {
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 64,
        'normalize_embeddings': True,
    }
    defaults.update(kwargs)
    return SentenceTransformersProvider(**defaults)


def create_quality_provider(**kwargs) -> SentenceTransformersProvider:
    """Create a high-quality embedding provider optimized for accuracy."""
    defaults = {
        'model_name': 'all-mpnet-base-v2',
        'batch_size': 32,
        'normalize_embeddings': True,
    }
    defaults.update(kwargs)
    return SentenceTransformersProvider(**defaults)


def create_code_provider(**kwargs) -> SentenceTransformersProvider:
    """Create a code-specific embedding provider."""
    defaults = {
        'model_name': 'microsoft/codebert-base',
        'batch_size': 16,  # Larger model, smaller batches
        'normalize_embeddings': True,
    }
    defaults.update(kwargs)
    return SentenceTransformersProvider(**defaults)